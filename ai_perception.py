#!/usr/bin/env python3
"""
Ru'ya Smart Assistive Glasses — YOLO11n + MiDaS Perception Pipeline

Features:
- Two local YOLO models (yolo11n.pt + final.pt) for object detection + tracking
- MiDaS_small for monocular relative depth estimation
- Proper MiDaS disparity → meters conversion with per-frame normalization
- Exponential smoothing to reduce depth jitter
- Object tracking with persistent IDs and motion classification
- Hazard classification: CRITICAL / HIGH / SAFE
- Terminal logging per frame
- Video file input (webcam code present but commented out)
- Output saved as MP4 (no preview window)

BUG FIX (v2):
  Previously `_detect_objects` skipped any box where `box.id is None`.
  On the first frame after a video loop reset the YOLO tracker has no
  history, so ALL boxes have id=None and the returned list is empty.
  Fix: instead of skipping, we fall back to a frame-local counter ID
  so every detected box is always included regardless of tracker state.

Can be used standalone:
    python ai_perception.py

Or imported by orchestrator:
    from ai_perception import AIPerception, FrameResult
"""

from collections import defaultdict, deque
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import os
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ── Environment ────────────────────────────────────────────────────────────────
os.environ["TORCH_HUB_TRUST_REPOS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"]  = ""
torch.cuda.is_available = lambda: False  # Force CPU


# ── File paths ─────────────────────────────────────────────────────────────────
YOLO_MODEL_SPECS = [
    ("yolo11n", Path("yolo11n.pt")),
    ("final",   Path("final.pt")),
]
CUSTOM_MODEL_CLASS_NAMES = {
    "final": {
        0: "ceiling",
        1: "door",
        2: "wall",
        3: "window",
        4: "object",
    },
}
VIDEO_SOURCE  = Path("streetsun.mp4")
OUTPUT_PATH   = Path("streetsun_output_finalll.mp4")
TEMP_AVI_PATH = Path("_temp_out.avi")


# ── Detection settings ─────────────────────────────────────────────────────────
DISPLAY_WINDOW               = False
YOLO_CONF_THRESHOLD          = 0.25
YOLO_IMG_SIZE                = 640
MIN_FPS_BEFORE_DOWNSCALE     = 2.0
DOWNSCALE_IMG_SIZE           = 480
MERGE_OVERLAPPING_DETECTIONS = True
MODEL_MERGE_IOU_THRESHOLD    = 0.60


# ── Depth / calibration ────────────────────────────────────────────────────────
DEPTH_NEAR_M = 0.5
DEPTH_FAR_M  = 10.0
SMOOTH_ALPHA = 0.35


# ── Hazard thresholds ──────────────────────────────────────────────────────────
DISTANCE_CRITICAL_M = 1.5
DISTANCE_HIGH_M     = 3.5


# ── Approach detection ─────────────────────────────────────────────────────────
HISTORY_LEN          = 10
APPROACH_WINDOW      = 5
APPROACH_DEPTH_DELTA = 0.08
LEAVE_DEPTH_DELTA    = 0.08


# ── Path corridor ──────────────────────────────────────────────────────────────
PATH_CENTER_X_RATIO = 0.50
PATH_WIDTH_RATIO    = 0.45
PATH_MIN_Y_RATIO    = 0.40


# ── Hazard class whitelist ─────────────────────────────────────────────────────
HAZARD_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "fire hydrant", "stop sign", "bench",
    "dog", "cat", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "bottle", "cup",
    "backpack", "suitcase",
}


# ── Colours (BGR) ──────────────────────────────────────────────────────────────
COLORS = {
    "CRITICAL": (0,   0,   255),
    "HIGH":     (0,   200, 255),
    "SAFE":     (0,   200, 0  ),
}


# ══════════════════════════════════════════════════════════════════════════════
# Result dataclass — used by orchestrator
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectedObject:
    """Single detected object returned to the orchestrator."""
    class_name: str
    distance_m: float
    priority:   str       # CRITICAL / HIGH / SAFE
    motion:     str       # APPROACHING / LEAVING / STATIC
    in_path:    bool
    direction:  str       # left / right / ahead
    track_key:  str
    model_name: str


@dataclass
class FrameResult:
    """
    Result of processing one camera frame.
    Returned by AIPerception.process_frame_for_orchestrator().
    """
    objects:          List[DetectedObject] = field(default_factory=list)
    critical_count:   int = 0
    high_count:       int = 0
    safe_count:       int = 0
    annotated_frame:  Optional[np.ndarray] = None

    def has_hazard(self) -> bool:
        return self.critical_count > 0 or self.high_count > 0

    def closest_hazard(self) -> Optional[DetectedObject]:
        """Return the closest CRITICAL or HIGH object, or None."""
        hazards = [o for o in self.objects if o.priority in ("CRITICAL", "HIGH")]
        if not hazards:
            return None
        return min(hazards, key=lambda o: o.distance_m)

    def to_speech(self) -> str:
        """
        Convert frame result to a natural spoken description
        for the TTS module.
        """
        if not self.objects:
            return "I cannot detect any objects around you."

        parts = []

        # Critical hazards first
        critical = [o for o in self.objects if o.priority == "CRITICAL"]
        for obj in critical:
            parts.append(
                f"Warning! {obj.class_name} detected {obj.distance_m:.1f} meters {obj.direction}, approaching."
            )

        # High priority
        high = [o for o in self.objects if o.priority == "HIGH"]
        for obj in high:
            parts.append(
                f"Caution. {obj.class_name} at {obj.distance_m:.1f} meters {obj.direction}."
            )

        # Safe objects summary
        safe = [o for o in self.objects if o.priority == "SAFE"]
        if safe and not critical and not high:
            names = list(dict.fromkeys(o.class_name for o in safe))[:3]
            parts.append(f"I can see {', '.join(names)} nearby. Path appears clear.")

        return " ".join(parts) if parts else "Surroundings appear clear."


# ══════════════════════════════════════════════════════════════════════════════
# Depth model
# ══════════════════════════════════════════════════════════════════════════════

def load_midas():
    try:
        print("Loading MiDaS_small …", flush=True)
        model     = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        device    = torch.device("cpu")
        model.to(device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform  = transforms.small_transform
        print("MiDaS loaded.", flush=True)
        return model, transform, device, True
    except Exception as exc:
        print(f"MiDaS failed to load ({exc}). Using bounding-box fallback.", flush=True)
        return None, None, None, False


def run_midas(model, transform, device, frame_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    batch   = transform(img_rgb).to(device)
    with torch.no_grad():
        pred = model(batch)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
    d_min, d_max = pred.min(), pred.max()
    if d_max - d_min < 1e-6:
        return np.zeros_like(pred, dtype=np.float32)
    return ((pred - d_min) / (d_max - d_min)).astype(np.float32)


def disparity_to_meters(disp_norm: float) -> float:
    meters = DEPTH_FAR_M - disp_norm * (DEPTH_FAR_M - DEPTH_NEAR_M)
    return float(np.clip(meters, DEPTH_NEAR_M, DEPTH_FAR_M))


def bbox_area_to_meters(x1, y1, x2, y2, fw, fh) -> float:
    area_ratio = max(1e-6, (x2 - x1) * (y2 - y1) / (fw * fh))
    meters = 0.25 / area_ratio
    return float(np.clip(meters, DEPTH_NEAR_M, DEPTH_FAR_M))


def sample_depth_from_map(disp_map: np.ndarray, x1, y1, x2, y2) -> float:
    h, w = disp_map.shape[:2]
    x1c  = max(0, min(x1, w - 1))
    y1c  = max(0, min(y1, h - 1))
    x2c  = max(0, min(x2, w))
    y2c  = max(0, min(y2, h))
    if x2c <= x1c or y2c <= y1c:
        return 0.5
    sy     = y1c + int((y2c - y1c) * 0.40)
    sx     = x1c + int((x2c - x1c) * 0.20)
    ex     = x2c - int((x2c - x1c) * 0.20)
    region = disp_map[sy:y2c, sx:ex]
    if region.size == 0:
        region = disp_map[y1c:y2c, x1c:x2c]
    return float(np.median(region)) if region.size > 0 else 0.5


def box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_model_detections(detections, iou_threshold=MODEL_MERGE_IOU_THRESHOLD):
    if not MERGE_OVERLAPPING_DETECTIONS or len(detections) < 2:
        return detections
    kept = []
    for det in sorted(detections, key=lambda d: d["conf"], reverse=True):
        if not any(box_iou(det["bbox"], e["bbox"]) >= iou_threshold for e in kept):
            kept.append(det)
    return kept


def resolve_class_name(model_name: str, model, cls_id: int) -> str:
    custom = CUSTOM_MODEL_CLASS_NAMES.get(model_name)
    if custom and cls_id in custom:
        return custom[cls_id]
    names = model.names
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


# ══════════════════════════════════════════════════════════════════════════════
# Path / hazard helpers
# ══════════════════════════════════════════════════════════════════════════════

def in_user_path(x1, y1, x2, y2, fw, fh) -> bool:
    cx    = (x1 + x2) / 2.0
    by    = y2
    left  = fw * (PATH_CENTER_X_RATIO - PATH_WIDTH_RATIO / 2.0)
    right = fw * (PATH_CENTER_X_RATIO + PATH_WIDTH_RATIO / 2.0)
    min_y = fh * PATH_MIN_Y_RATIO
    return (left <= cx <= right) and (by >= min_y)


# ══════════════════════════════════════════════════════════════════════════════
# Object tracker
# ══════════════════════════════════════════════════════════════════════════════

class ObjectState:
    def __init__(self):
        self.depth_history:  deque = deque(maxlen=HISTORY_LEN)
        self.smoothed_depth: float = DEPTH_FAR_M
        self.motion:         str   = "STATIC"

    def update(self, raw_depth: float):
        if not self.depth_history:
            self.smoothed_depth = raw_depth
        else:
            self.smoothed_depth = (
                SMOOTH_ALPHA * raw_depth + (1.0 - SMOOTH_ALPHA) * self.smoothed_depth
            )
        self.depth_history.append(self.smoothed_depth)
        self._update_motion()

    def _update_motion(self):
        if len(self.depth_history) < APPROACH_WINDOW:
            self.motion = "STATIC"
            return
        recent  = list(self.depth_history)[-APPROACH_WINDOW:]
        changes = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        avg     = float(np.mean(changes))
        if avg < -APPROACH_DEPTH_DELTA:
            self.motion = "APPROACHING"
        elif avg > LEAVE_DEPTH_DELTA:
            self.motion = "LEAVING"
        else:
            self.motion = "STATIC"


class HazardTracker:
    def __init__(self):
        self._states:    dict = defaultdict(ObjectState)
        self._last_seen: dict = {}

    def update(self, track_id, raw_depth: float, frame_no: int) -> ObjectState:
        state = self._states[track_id]
        state.update(raw_depth)
        self._last_seen[track_id] = frame_no
        return state

    def gc(self, frame_no: int, ttl: int = 90):
        stale = [tid for tid, last in self._last_seen.items() if frame_no - last > ttl]
        for tid in stale:
            del self._states[tid]
            del self._last_seen[tid]

    @staticmethod
    def classify(state: ObjectState, class_name: str, in_path: bool) -> str:
        if class_name.lower() not in HAZARD_CLASSES:
            return "SAFE"
        if state.smoothed_depth <= DISTANCE_CRITICAL_M:
            return "CRITICAL"
        if state.smoothed_depth <= DISTANCE_HIGH_M:
            return "HIGH"
        return "SAFE"


# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def draw_box(frame, x1, y1, x2, y2, color, label: str, priority: str):
    overlay = frame.copy()
    alpha   = {"CRITICAL": 0.20, "HIGH": 0.14, "SAFE": 0.06}.get(priority, 0.06)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    thickness = {"CRITICAL": 5, "HIGH": 4, "SAFE": 2}.get(priority, 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    ty = max(th + 10, y1 - 6)
    cv2.rectangle(frame, (x1, ty - th - 8), (x1 + tw + 8, ty + 4), color, -1)
    cv2.putText(frame, label, (x1 + 4, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA)


def draw_hud(frame, width, fps, counts, frame_no):
    cv2.putText(frame, f"Ru'ya  FPS:{fps:.1f}  F:{frame_no}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    cv2.putText(frame, f"Critical:{counts['CRITICAL']}  High:{counts['HIGH']}  Safe:{counts['SAFE']}",
                (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def draw_path_corridor(frame, width, height):
    left  = int(width  * (PATH_CENTER_X_RATIO - PATH_WIDTH_RATIO / 2.0))
    right = int(width  * (PATH_CENTER_X_RATIO + PATH_WIDTH_RATIO / 2.0))
    top   = int(height * PATH_MIN_Y_RATIO)
    cv2.rectangle(frame, (left, top), (right, height - 1), (255, 255, 0), 2)
    cv2.putText(frame, "PATH", (left + 4, top + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


# ══════════════════════════════════════════════════════════════════════════════
# Video I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_avi_writer(path: Path, fps, w, h):
    for codec in ("MJPG", "XVID"):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if writer.isOpened():
            return writer
    raise RuntimeError("Cannot create AVI writer.")


def avi_to_mp4(src: Path, dst: Path, fps, w, h):
    cap  = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open temp file: {src}")
    mp4w = None
    for codec in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(
            str(dst), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if writer.isOpened():
            mp4w = writer
            break
    if mp4w is None:
        cap.release()
        raise RuntimeError("Cannot create MP4 writer.")
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            mp4w.write(f)
    finally:
        cap.release()
        mp4w.release()


def validate_video(path: Path) -> bool:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False
    ok, _ = cap.read()
    cap.release()
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# Main perception class
# ══════════════════════════════════════════════════════════════════════════════

class AIPerception:
    """
    Main perception class.

    Standalone usage:
        perception = AIPerception()
        perception.run(VIDEO_SOURCE)

    Orchestrator usage:
        perception = AIPerception()
        frame      = capture_frame_from_camera()
        result     = perception.process_frame_for_orchestrator(frame)
        speech     = result.to_speech()
    """

    def __init__(self):
        print("Initialising AI Perception …", flush=True)

        missing = [
            f"{name} ({path})"
            for name, path in YOLO_MODEL_SPECS
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError("YOLO model(s) not found: " + ", ".join(missing))

        self.yolo_models = []
        for model_name, model_path in YOLO_MODEL_SPECS:
            print(f"Loading YOLO model: {model_name} from {model_path} ...", flush=True)
            self.yolo_models.append((model_name, YOLO(str(model_path))))

        self.midas, self.midas_tf, self.midas_dev, self.use_midas = load_midas()
        self.tracker  = HazardTracker()
        self.frame_no = 0
        self.img_size = YOLO_IMG_SIZE
        self.fps_hist = deque(maxlen=30)

        # Counter used to generate fallback track keys when box.id is None.
        # This happens on the first frame after a video loop reset — the YOLO
        # tracker has no history so every box.id is None.  Using a synthetic
        # key (frame_no + box_index) means we still process every detection
        # instead of silently dropping them all.
        self._fallback_id_counter = 0

        print("AI Perception ready.", flush=True)

    # ── Internal frame processing ─────────────────────────────────────────────

    def _detect_objects(self, frame: np.ndarray):
        """
        Run both YOLO models on frame, merge, return raw detections.

        KEY FIX: box.id is None on the very first frame of a new tracking
        sequence (e.g. after a video loop reset).  The old code did:
            if box.id is None: continue
        which silently dropped ALL detections and returned an empty list.

        The fix: when box.id is None we generate a synthetic local ID using
        a per-instance counter.  The resulting track_key is unique within
        this call and the detection is always included.  The tracker will
        assign a proper persistent ID on the next frame.
        """
        h, w = frame.shape[:2]

        if self.use_midas:
            disp_norm = run_midas(self.midas, self.midas_tf, self.midas_dev, frame)
        else:
            disp_norm = None

        raw_detections = []
        for model_name, model in self.yolo_models:
            results = model.track(
                frame,
                verbose=False,
                persist=True,
                imgsz=self.img_size,
                device="cpu",
                conf=YOLO_CONF_THRESHOLD,
            )
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < YOLO_CONF_THRESHOLD:
                        continue

                    cls_id   = int(box.cls[0])
                    cls_name = resolve_class_name(model_name, model, cls_id)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # ── FIX: handle None tracker ID ───────────────────────────
                    # box.id is None on the first frame of a new tracking
                    # sequence.  Use a synthetic counter-based ID so we never
                    # drop a detection.
                    if box.id is not None:
                        local_tid = int(box.id[0])
                    else:
                        local_tid = -(self._fallback_id_counter + 1)
                        self._fallback_id_counter += 1
                    # ─────────────────────────────────────────────────────────

                    raw_detections.append({
                        "model_name":     model_name,
                        "conf":           conf,
                        "cls_name":       cls_name,
                        "bbox":           (x1, y1, x2, y2),
                        "local_track_id": local_tid,
                        "track_key":      f"{model_name}:{local_tid}",
                    })

        return merge_model_detections(raw_detections), disp_norm, w, h

    def _process_detections(self, frame, detections, disp_norm, w, h, annotate=True):
        """Process detections, update tracker, optionally annotate frame."""
        counts  = {"CRITICAL": 0, "HIGH": 0, "SAFE": 0}
        objects = []

        for det in detections:
            model_name = det["model_name"]
            cls_name   = det["cls_name"]
            x1, y1, x2, y2 = det["bbox"]
            track_key  = det["track_key"]

            if disp_norm is not None:
                disp_val  = sample_depth_from_map(disp_norm, x1, y1, x2, y2)
                raw_depth = disparity_to_meters(disp_val)
            else:
                raw_depth = bbox_area_to_meters(x1, y1, x2, y2, w, h)

            state    = self.tracker.update(track_key, raw_depth, self.frame_no)
            path_hit = in_user_path(x1, y1, x2, y2, w, h)
            priority = HazardTracker.classify(state, cls_name, path_hit)
            color    = COLORS[priority]

            counts[priority] = counts.get(priority, 0) + 1
            cx = (x1 + x2) / 2.0
            if cx < (w * 0.4):
                direction = "left"
            elif cx > (w * 0.6):
                direction = "right"
            else:
                direction = "ahead"

            objects.append(DetectedObject(
                class_name = cls_name,
                distance_m = state.smoothed_depth,
                priority   = priority,
                motion     = state.motion,
                in_path    = path_hit,
                direction  = direction,
                track_key  = track_key,
                model_name = model_name,
            ))

            status_str  = {"CRITICAL": "HAZARD", "HIGH": "POTENTIAL HAZARD", "SAFE": "SAFE"}[priority]
            local_tid   = det["local_track_id"]
            track_label = f"{model_name}:{local_tid}"

            print(
                f"[{track_label:>12}] {cls_name:<15} | "
                f"Distance: {state.smoothed_depth:.2f}m | "
                f"Status: {status_str:<17} | "
                f"Motion: {state.motion}",
                flush=True,
            )

            if annotate:
                label = (
                    f"{track_label} {cls_name} "
                    f"{state.smoothed_depth:.1f}m "
                    f"{status_str} {state.motion}"
                )
                draw_box(frame, x1, y1, x2, y2, color, label, priority)

        if annotate:
            avg_fps = float(np.mean(self.fps_hist)) if self.fps_hist else 0.0
            draw_hud(frame, w, avg_fps, counts, self.frame_no)
            draw_path_corridor(frame, w, h)

        self.tracker.gc(self.frame_no)
        self.frame_no += 1

        return objects, counts

    # ── Public API for orchestrator ───────────────────────────────────────────

    def process_frame_for_orchestrator(self, frame: np.ndarray) -> FrameResult:
        """
        Process a single camera frame and return a FrameResult.
        Called by the orchestrator for live camera integration.

        Args:
            frame: BGR numpy array from camera

        Returns:
            FrameResult with detected objects, counts, and annotated frame
        """
        t0 = time.time()

        detections, disp_norm, w, h = self._detect_objects(frame)
        annotated = frame.copy()
        objects, counts = self._process_detections(
            annotated, detections, disp_norm, w, h, annotate=True
        )

        elapsed = time.time() - t0
        self.fps_hist.append(1.0 / elapsed if elapsed > 0 else 0.0)

        return FrameResult(
            objects         = objects,
            critical_count  = counts["CRITICAL"],
            high_count      = counts["HIGH"],
            safe_count      = counts["SAFE"],
            annotated_frame = annotated,
        )

    # ── Standalone video processing ───────────────────────────────────────────

    def process_frame(self, frame: np.ndarray):
        """Process frame for standalone video mode (annotates and returns frame)."""
        h, w = frame.shape[:2]
        detections, disp_norm, w, h = self._detect_objects(frame)
        self._process_detections(frame, detections, disp_norm, w, h, annotate=True)
        return frame

    def run(self, source=VIDEO_SOURCE):
        """Standalone video processing loop."""
        # ── WEBCAM (disabled) ──────────────────────────────────────────────────
        # To use webcam instead of a video file:
        # 1. Comment out the video-file block below
        # 2. Uncomment the two lines below
        #
        # cap = cv2.VideoCapture(0)
        # total_frames = -1

        # ── VIDEO FILE (active) ────────────────────────────────────────────────
        cap          = cv2.VideoCapture(str(source))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = make_avi_writer(TEMP_AVI_PATH, fps, width, height)

        print(f"Processing: {source}  {width}x{height} @ {fps:.1f} fps", flush=True)
        t_start   = time.time()
        t_fps_chk = t_start

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                t_f0      = time.time()
                processed = self.process_frame(frame)
                elapsed   = time.time() - t_f0
                self.fps_hist.append(1.0 / elapsed if elapsed > 0 else 0.0)

                now = time.time()
                if now - t_fps_chk >= 1.0:
                    avg = float(np.mean(self.fps_hist)) if self.fps_hist else 0.0
                    if avg < MIN_FPS_BEFORE_DOWNSCALE and self.img_size > DOWNSCALE_IMG_SIZE:
                        self.img_size = DOWNSCALE_IMG_SIZE
                        print(f"Low FPS ({avg:.1f}) — downscaling to {self.img_size}px", flush=True)
                    t_fps_chk = now

                writer.write(processed)

                if DISPLAY_WINDOW:
                    cv2.imshow("Ru'ya Perception", processed)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if self.frame_no % 30 == 0:
                    pct = f"{100*self.frame_no/total_frames:.1f}%" if total_frames > 0 else "?"
                    print(f"  … frame {self.frame_no}/{total_frames} ({pct})", flush=True)

        finally:
            cap.release()
            writer.release()
            if DISPLAY_WINDOW:
                cv2.destroyAllWindows()

        if OUTPUT_PATH.exists():
            OUTPUT_PATH.unlink()
        avi_to_mp4(TEMP_AVI_PATH, OUTPUT_PATH, fps, width, height)

        if not validate_video(OUTPUT_PATH):
            raise RuntimeError(f"Output MP4 not playable: {OUTPUT_PATH}")

        if TEMP_AVI_PATH.exists():
            TEMP_AVI_PATH.unlink()

        total = time.time() - t_start
        print(
            f"\nDone. {self.frame_no} frames in {total:.1f}s "
            f"({self.frame_no/total:.2f} fps avg)\n"
            f"Saved → {OUTPUT_PATH.resolve()}",
            flush=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Entry point (standalone)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Ru'ya Smart Assistive Glasses — Perception Pipeline")
    print("=" * 65)
    perception = AIPerception()
    perception.run(VIDEO_SOURCE)


if __name__ == "__main__":
    main()
