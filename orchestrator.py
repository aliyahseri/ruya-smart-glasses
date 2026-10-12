#!/usr/bin/env python3
"""
Ru'ya Smart Glasses — Orchestrator
====================================
Central controller that connects all available modules.

Modules:
  - STT        : Whisper (speech → text)
  - CMD        : CommandExtractor (text → intent)
  - OCR        : PaddleOCR (image → text)
  - PERCEPTION : AIPerception — YOLO11n + final.pt + MiDaS depth (environment queries)
  - TTS        : pyttsx3 (text → speech)

Pipeline:
  Microphone → STT → CommandExtractor → Route → Module → TTS → Speaker

  ENVIRONMENT_QUERY → AIPerception (both YOLO models + MiDaS depth)
  OCR              → PaddleOCR
  BATTERY          → BatteryManager (reads system battery; replace with GPIO if needed)
  EMERGENCY        → Alert + log

Run:
  python orchestrator.py interactive   ← press Enter before each command
  python orchestrator.py continuous    ← loops automatically
  python orchestrator.py single        ← one command then exit

VIDEO SOURCE:
  Default behaviour uses VIDEO_SOURCE below.
  You can override at runtime using environment variables:
    RUYA_VIDEO_SOURCE     = video | camera
    RUYA_TEST_VIDEO_PATH  = path to .mp4/.avi (when video)
    RUYA_CAMERA_INDEX     = integer (when camera)
    RUYA_CAMERA_URL       = URL (optional, when camera)

ENVIRONMENT QUERY behaviour:
  When the user asks "what's around me?", the orchestrator feeds
  TRACKER_WARMUP_FRAMES consecutive frames into AIPerception before
  reading the final result. This is necessary because YOLO's tracker
  needs several frames to assign stable IDs — on frame 0 every box.id
  is None and all detections are skipped.  Feeding multiple frames
  replicates what happens on a real live camera that has been running
  for a while.  On the actual device the tracker will already be warm,
  so this warm-up is essentially free.

BUG FIX (v2):
  Previously the code discarded all warmup results, then captured one
  brand-new frame after the loop. If that fresh frame happened to land
  on the video's first frame (loop reset), the tracker had no history
  and returned zero detections.

  Fix: the warmup loop now saves the LAST frame result and uses it
  directly — no extra capture after the loop. The tracker is guaranteed
  to be warm because we just ran TRACKER_WARMUP_FRAMES frames through it.
"""

import sys
import os
import re
import time
import threading
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import threading, time


try:
    import psutil
except ImportError:
    psutil = None

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent))

# ── Module imports ────────────────────────────────────────────────────────────
from speech_command_integration import record_audio_to_memory, speech_to_text_from_memory
from command_extractor import CommandExtractor, Intent
from ocr_processor import OCRProcessor
from text_to_speech import text_to_speech as speak, speak_urgent
from ai_perception import AIPerception, FrameResult
from serial_camera import SerialCameraStream

# ══════════════════════════════════════════════════════════════════════════════
# VIDEO SOURCE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
#
#   "video"   → uses TEST_VIDEO_PATH (.mp4) — USE THIS FOR TESTING
#   "camera"  → uses live camera device    — USE THIS ON ACTUAL DEVICE
#
VIDEO_SOURCE    = "camera"                                   # ← switch here
TEST_VIDEO_PATH = str(Path(__file__).parent / "testvideo.mp4")


def _video_config():
    source = (os.getenv("RUYA_VIDEO_SOURCE") or VIDEO_SOURCE or "").strip().lower()
    if not source:
        source = "video"
    test_path = (os.getenv("RUYA_TEST_VIDEO_PATH") or TEST_VIDEO_PATH or "").strip()
    camera_url = (os.getenv("RUYA_CAMERA_URL") or "").strip() or None
    camera_index_raw = os.getenv("RUYA_CAMERA_INDEX")
    camera_index = None
    if camera_index_raw is not None and camera_index_raw.strip() != "":
        try:
            camera_index = int(camera_index_raw.strip())
        except ValueError:
            camera_index = None
    return source, test_path, camera_index, camera_url


def _serial_camera_config():
    port = (os.getenv("RUYA_SERIAL_PORT") or "").strip()
    baud_raw = (os.getenv("RUYA_SERIAL_BAUD") or "").strip()
    baud = 921600
    if baud_raw:
        try:
            baud = int(baud_raw)
        except ValueError:
            baud = 921600
    return port, baud

# How many frames to feed through before reading the final result.
# YOLO's tracker needs ~10 frames to assign stable IDs.
# Increase if detections are still empty; decrease for faster response.
TRACKER_WARMUP_FRAMES = 15
OCR_TARGET_SCAN_FRAMES = TRACKER_WARMUP_FRAMES
OCR_GENERAL_SCAN_FRAMES = 5

OCR_ROUTE_KEYWORDS = (
    "read",
    "scan",
    "ocr",
    "what does",
    "what's written",
    "what is written",
    "what's the text",
    "what is the text",
    "text on",
    "text in",
    "written on",
    "written in",
    "writing on",
    "writing in",
    " say",
    "says",
)

OCR_ROUTE_FILLER = {
    "this", "that", "it", "me", "text", "words", "writing",
    "front", "behind", "left", "right", "ahead", "above", "below",
}

OCR_ROUTE_DIRECTIONS = {
    "front", "behind", "left", "right", "ahead", "above", "below", "back",
}

OCR_ROUTE_TARGET_PATTERNS = (
    re.compile(r"\bread\s+(?:the\s+)?(?:text|words|writing)\s+(?:on|in|from)\s+(?:the\s+)?(\w+)", re.IGNORECASE),
    re.compile(r"\bread\s+(?:the\s+)?(\w+)", re.IGNORECASE),
    re.compile(r"\b(?:text|words|writing|written)\s+(?:on|in|from)\s+(?:the\s+)?(\w+)", re.IGNORECASE),
    re.compile(r"\b(?:what(?:'s| is)\s+(?:the\s+)?)?(?:text|words|writing|written)\s+(?:on|in|from)\s+(?:the\s+)?(\w+)", re.IGNORECASE),
    re.compile(r"\bwhat\s+does\s+(?:the\s+|that\s+)?(\w+)\s+say\b", re.IGNORECASE),
)

# ══════════════════════════════════════════════════════════════════════════════
# VideoStream — continuous background frame reader
# ══════════════════════════════════════════════════════════════════════════════

class VideoStream:
    """
    Runs a background thread that continuously reads frames so the latest
    frame is always available instantly when a command is triggered.

    Supports two sources:
      - A video file (.mp4 / .avi / etc.)  ← for testing
      - A live camera device index         ← for actual device

    When used with a video file the stream loops back to the beginning
    when the video ends, so frames are always available.
    """

    def __init__(self, source):
        """
        Parameters
        ----------
        source : str or int
            Path to a video file  (str)  ← test mode
            Camera device index   (int)  ← device mode
        """
        self._source  = source
        self._cap     = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            raise RuntimeError(f"VideoStream: cannot open source '{source}'")

        self._ret     = False
        self._frame   = None
        self._lock    = threading.Lock()
        self._running = True

        # Read the very first frame before the thread starts so _frame is
        # never None on the first call to read()
        self._ret, self._frame = self._cap.read()

        # Daemon thread — exits automatically when the main program exits
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

        source_label = (
            f"video file: {source}" if isinstance(source, str)
            else f"camera index: {source}"
        )
        print(f"  VideoStream         : running ({source_label})")
        
        # existing line:


    # ── Background reader ─────────────────────────────────────────────────────

    def _update(self):
        """Continuously read frames in a background thread."""
        while self._running:
            ret, frame = self._cap.read()

            # ── Video file: loop back to start when the video ends ────────────
            if isinstance(self._source, str) and not ret:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
            # ─────────────────────────────────────────────────────────────────

            with self._lock:
                self._ret   = ret
                self._frame = frame

    # ── Public API ────────────────────────────────────────────────────────────

    def read(self):
        """Return (success, latest_frame). Thread-safe."""
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def stop(self):
        """Stop the background thread and release the capture device."""
        self._running = False
        self._thread.join(timeout=2)
        self._cap.release()
        print("  VideoStream         : stopped")


class BatteryManager:
    """Read the current device battery when the operating system exposes it."""

    @staticmethod
    def read_status() -> Optional[dict]:
        """
        Return battery status as a dict with:
          - percent: int
          - power_plugged: bool
          - secs_left: Optional[int]
        """
        if psutil is not None:
            try:
                battery = psutil.sensors_battery()
            except Exception as exc:
                print(f"Battery read via psutil failed: {exc}")
            else:
                if battery is not None and battery.percent is not None:
                    return {
                        "percent": int(round(float(battery.percent))),
                        "power_plugged": bool(battery.power_plugged),
                        "secs_left": (
                            int(battery.secsleft)
                            if isinstance(battery.secsleft, (int, float))
                            and battery.secsleft >= 0
                            else None
                        ),
                    }

        power_supply_root = Path("/sys/class/power_supply")
        if not power_supply_root.exists():
            return None

        for candidate in power_supply_root.iterdir():
            type_file = candidate / "type"
            if type_file.exists():
                try:
                    power_type = type_file.read_text(encoding="utf-8").strip().lower()
                except OSError:
                    continue
                if power_type != "battery":
                    continue
            elif not candidate.name.lower().startswith("bat"):
                continue

            capacity_file = candidate / "capacity"
            if not capacity_file.exists():
                continue

            try:
                percent = int(round(float(capacity_file.read_text(encoding="utf-8").strip())))
            except (OSError, ValueError):
                continue

            status_file = candidate / "status"
            try:
                raw_status = (
                    status_file.read_text(encoding="utf-8").strip().lower()
                    if status_file.exists()
                    else ""
                )
            except OSError:
                raw_status = ""

            return {
                "percent": percent,
                "power_plugged": raw_status in {"charging", "full"},
                "secs_left": None,
            }

        return None

    @staticmethod
    def format_duration(seconds: Optional[int]) -> str:
        """Convert battery seconds into a short spoken duration."""
        if seconds is None or seconds < 0:
            return ""

        total_minutes = max(0, int(round(seconds / 60.0)))
        hours, minutes = divmod(total_minutes, 60)

        if hours and minutes:
            return (
                f"{hours} hour{'s' if hours != 1 else ''} "
                f"and {minutes} minute{'s' if minutes != 1 else ''}"
            )
        if hours:
            return f"{hours} hour{'s' if hours != 1 else ''}"
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    


HAZARD_COOLDOWN = {
    "CRITICAL": 4.0,   # seconds between repeating same class
    "HIGH":     8.0,
}
HAZARD_LOOP_INTERVAL = 0.15   # seconds between frame checks (~6 fps)

class HazardWatcher:
    """
    Runs in a background thread. Reads frames from VideoStream,
    runs AIPerception, and proactively speaks warnings for
    CRITICAL/HIGH objects — without waiting for a user command.
    """

    def __init__(self, stream: VideoStream, get_perception_fn, speak_urgent_fn):
        self._stream        = stream
        self._get_perception = get_perception_fn
        self._speak_urgent  = speak_urgent_fn
        self._cooldowns     = {}   # class_name -> last_spoken_time
        self._running       = False
        self._thread        = None

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("  HazardWatcher       : running")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self):
        perception = None
        while self._running:
            time.sleep(HAZARD_LOOP_INTERVAL)

            # Lazy-load perception (may not be ready at startup)
            if perception is None:
                perception = self._get_perception()
            if perception is None:
                continue

            ret, frame = self._stream.read()
            if not ret or frame is None:
                continue

            try:
                result = perception.process_frame_for_orchestrator(frame)
            except Exception:
                continue

            if not result or not result.objects:
                continue

            now = time.time()
            for obj in result.objects:
                if obj.priority not in ("CRITICAL", "HIGH"):
                    continue

                cooldown = HAZARD_COOLDOWN[obj.priority]
                last     = self._cooldowns.get(obj.class_name, 0)
                if now - last < cooldown:
                    continue   # still in cooldown for this class

                self._cooldowns[obj.class_name] = now
                msg = self._build_warning(obj, result)
                self._speak_urgent(msg)
                break   # one alert per frame to avoid stacking

    def _build_warning(self, obj, result) -> str:
        dist = obj.distance_m

        # Direction hint from bounding box centre (if available)
        direction = ""
        if hasattr(obj, "bbox"):
            # bbox assumed as (x1, y1, x2, y2) in pixel coords
            # frame width not directly available — use relative centre
            cx = (obj.bbox[0] + obj.bbox[2]) / 2
            fw = getattr(result, "frame_width", None) or 640
            rel = cx / fw
            if rel < 0.38:
                direction = " to your left"
            elif rel > 0.62:
                direction = " to your right"
            else:
                direction = " directly ahead"

        if obj.priority == "CRITICAL":
            guidance = self._guidance(obj.class_name, direction, dist)
            return (
                f"Warning! {obj.class_name} at {dist:.1f} metres{direction}. "
                f"{guidance}"
            )
        else:
            return f"Caution. {obj.class_name} at {dist:.1f} metres{direction}."

    @staticmethod
    def _guidance(class_name: str, direction: str, dist: float) -> str:
        """Return a short navigation instruction based on hazard type."""
        c = class_name.lower()
        if any(w in c for w in ("car", "truck", "bus", "motorcycle", "vehicle")):
            return "Stop moving. Wait for it to pass."
        if any(w in c for w in ("person", "bicycle", "scooter")):
            return "Slow down and step to the side."
        if any(w in c for w in ("stairs", "step", "curb")):
            return "Watch your step."
        if any(w in c for w in ("hole", "pit", "gap")):
            return "Stop. There is a drop ahead."
        return "Stop and proceed with caution."


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class Orchestrator:
    """
    Central controller for Ru'ya Smart Glasses.
    Connects STT → CommandExtractor → Module → TTS.
    """

    def __init__(self, start_hazard_watcher: bool = True):
        print("\nInitializing Ru'ya Smart Glasses...")

        # Command extractor — always loaded at startup
        self.extractor = CommandExtractor()
        print("  Command extractor   : ready")

        # OCR — loaded on first use
        self._ocr = None
        print("  OCR processor       : ready (loads on first use)")

        # AI Perception — loaded on first use (heavy: 2 YOLO models + MiDaS)
        self._perception: AIPerception = None
        print("  AI Perception       : ready (loads on first use)")

        # Battery manager — reads the device battery on demand
        self._battery = BatteryManager()
        print("  Battery manager     : ready")

        # Video stream — always running in the background
        self._stream = self._init_stream()
        self._hazard_watcher = None
        if start_hazard_watcher:
            self._hazard_watcher = HazardWatcher(
                stream            = self._stream,
                get_perception_fn = self._get_perception,
                speak_urgent_fn   = speak_urgent,
            )
            self._hazard_watcher.start()

        print("System ready.\n")

    # ── Stream initializer ────────────────────────────────────────────────────

    def _init_stream(self) -> VideoStream:
        """
        Initialize the video stream based on VIDEO_SOURCE at the top.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  TESTING WITH A VIDEO FILE                                          │
        │  Set VIDEO_SOURCE = "video" and place your .mp4 at TEST_VIDEO_PATH  │
        │  The video will loop automatically so frames never run out.         │
        ├─────────────────────────────────────────────────────────────────────┤
        │  DEPLOYING ON ACTUAL DEVICE                                         │
        │  Set VIDEO_SOURCE = "camera"                                        │
        │  Uncomment CAMERA_INDEX above and set the correct index (usually 0) │
        └─────────────────────────────────────────────────────────────────────┘
        """
        source, test_path, camera_index, camera_url = _video_config()

        if source == "video":
            # ── TEST MODE: read from a .mp4 / .avi video file ─────────────────
            if test_path and Path(test_path).exists():
                return VideoStream(source=test_path)

            source = "camera"
            # ─────────────────────────────────────────────────────────────────

        if source == "camera":
            # ── DEVICE MODE: live camera ──────────────────────────────────────
            serial_port, serial_baud = _serial_camera_config()
            if os.name == "posix" and serial_port:
                return SerialCameraStream(port=serial_port, baudrate=serial_baud)

            candidates = []

            if os.name == "posix":
                video0 = Path("/dev/video0")
                if video0.exists():
                    candidates.append(0)
                else:
                    for dev in sorted(Path("/dev").glob("video*")):
                        name = dev.name
                        if not name.startswith("video"):
                            continue
                        idx_raw = name[5:]
                        if not idx_raw.isdigit():
                            continue
                        idx = int(idx_raw)
                        if idx >= 10:
                            continue
                        candidates.append(idx)

            if not candidates:
                candidates.append(camera_index if camera_index is not None else 0)

            tried = set()
            for idx in candidates:
                if idx in tried:
                    continue
                tried.add(idx)
                try:
                    return VideoStream(source=idx)
                except Exception:
                    continue

            serial_candidates = []

            if os.name == "posix":
                if serial_port:
                    serial_candidates.append(serial_port)
                else:
                    for dev in sorted(Path("/dev").glob("ttyUSB*")):
                        serial_candidates.append(str(dev))
                    for dev in sorted(Path("/dev").glob("ttyACM*")):
                        serial_candidates.append(str(dev))

            if serial_candidates:
                for port in serial_candidates:
                    try:
                        return SerialCameraStream(port=port, baudrate=serial_baud)
                    except Exception:
                        continue

            if camera_url is not None:
                return VideoStream(source=camera_url)

            raise RuntimeError("No usable /dev/video* camera found, and no usable serial camera, and RUYA_CAMERA_URL is not set.")

        else:
            raise ValueError(
                f"Unknown VIDEO_SOURCE: '{source}'. Use 'video' or 'camera'."
            )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def cleanup(self):
        """Release all resources."""
        if self._hazard_watcher is not None:
            self._hazard_watcher.stop()
        # existing line:
        if self._stream is not None:
            self._stream.stop()

    # ── Lazy loaders ──────────────────────────────────────────────────────────

    def _get_ocr(self):
        """Load OCR model on first use."""
        if self._ocr is None:
            try:
                print("Loading OCR model...")
                self._ocr = OCRProcessor()
                print("OCR model loaded.")
            except Exception as e:
                print(f"OCR load failed: {e}")
                return None
        return self._ocr

    def _get_perception(self) -> AIPerception:
        """Load AIPerception on first use (both YOLO models + MiDaS)."""
        if self._perception is None:
            try:
                print("Loading AI Perception (YOLO11n + final.pt + MiDaS)...")
                self._perception = AIPerception()
                print("AI Perception loaded.")
            except Exception as e:
                print(f"AI Perception load failed: {e}")
                return None
        return self._perception

    # ── Shared frame capture ──────────────────────────────────────────────────

    def _capture_frame(self) -> np.ndarray:
        """
        Grab the latest frame from the video stream.
        Used by both the perception pipeline and OCR.
        Returns instantly — no camera warm-up delay.
        """
        ret, frame = self._stream.read()
        if not ret or frame is None:
            return None
        return frame

    def _filter_ocr_results(self, text_results, min_confidence: float = 0.6) -> list:
        """Keep only confident OCR results with non-empty text."""
        return [
            result for result in text_results
            if result.get("text") and result.get("confidence", 0.0) >= min_confidence
        ]

    def _combine_ocr_text(self, text_results) -> str:
        """Join OCR fragments into a single spoken string."""
        return " ".join(result["text"] for result in text_results).strip()

    def _score_ocr_results(self, text_results) -> float:
        """Prefer OCR outputs that look more like readable text than mirrored noise."""
        if not text_results:
            return float("-inf")

        combined = self._combine_ocr_text(text_results)
        tokens = [token for token in combined.split() if token]
        confidence_score = sum(result.get("confidence", 0.0) for result in text_results)
        alpha_chars = sum(char.isalpha() for char in combined)
        digit_chars = sum(char.isdigit() for char in combined)
        punctuation_chars = sum(
            (not char.isalnum()) and (not char.isspace()) for char in combined
        )
        clean_word_tokens = sum(token.isalpha() for token in tokens)
        short_clean_words = sum(
            token.isalpha() and 2 <= len(token) <= 4 for token in tokens
        )
        edge_symbol_tokens = sum(
            1
            for token in tokens
            if not token[0].isalnum() or not token[-1].isalnum()
        )

        return (
            confidence_score
            + (0.25 * alpha_chars)
            + (0.10 * digit_chars)
            + (0.80 * clean_word_tokens)
            + (0.60 * short_clean_words)
            - (0.45 * punctuation_chars)
            - (0.80 * edge_symbol_tokens)
        )

    def _build_ocr_candidate(self, text_results):
        """Convert OCR fragments into a scored candidate for cross-frame voting."""
        if not text_results:
            return None

        combined = self._combine_ocr_text(text_results)
        if not combined:
            return None

        normalized = re.sub(r"\s+", " ", combined).strip().lower()
        return {
            "text": combined,
            "normalized": normalized,
            "score": self._score_ocr_results(text_results),
        }

    def _choose_best_ocr_candidate(self, candidates) -> str:
        """Choose the most reliable OCR text across several frame samples."""
        if not candidates:
            return ""

        grouped = {}
        for candidate in candidates:
            key = candidate["normalized"]
            entry = grouped.setdefault(
                key,
                {"count": 0, "best_score": float("-inf"), "text": candidate["text"]},
            )
            entry["count"] += 1
            if candidate["score"] > entry["best_score"]:
                entry["best_score"] = candidate["score"]
                entry["text"] = candidate["text"]

        best = max(
            grouped.values(),
            key=lambda item: (item["count"], item["best_score"], len(item["text"])),
        )
        return best["text"]

    def _extract_best_ocr_from_frame(self, ocr, frame: np.ndarray) -> list:
        """Run OCR on the frame and its mirror, then keep the more text-like result."""
        if frame is None:
            return []

        candidates = []
        for label, candidate_frame in (
            ("original", frame),
            ("flipped", cv2.flip(frame, 1)),
        ):
            text_results = ocr.extract_text_from_array(candidate_frame)
            good_results = self._filter_ocr_results(text_results)
            if not good_results:
                continue

            score = self._score_ocr_results(good_results)
            candidates.append((score, label, good_results))

        if not candidates:
            return []

        candidates.sort(key=lambda item: item[0], reverse=True)
        _, label, best_results = candidates[0]
        if label == "flipped":
            print("Using mirrored-frame correction for OCR...")
        return best_results

    def _scan_frames_for_ocr(self, ocr, frame_count: int) -> str:
        """Sample several frames and keep the strongest OCR reading."""
        candidates = []

        print(f"Scanning {frame_count} frame(s) for readable text...")
        for _ in range(frame_count):
            frame = self._capture_frame()
            if frame is None:
                break

            good_results = self._extract_best_ocr_from_frame(ocr, frame)
            candidate = self._build_ocr_candidate(good_results)
            if candidate is not None:
                candidates.append(candidate)

            time.sleep(0.05)

        return self._choose_best_ocr_candidate(candidates)

    def _looks_like_text_query(self, raw_text: str) -> bool:
        """Heuristic OCR fallback when command extraction misses a text request."""
        lower = (raw_text or "").lower()
        return any(keyword in lower for keyword in OCR_ROUTE_KEYWORDS)

    def _extract_ocr_target(self, result) -> str:
        """Extract the requested OCR target from entities or raw speech text."""
        target = (result.entities.get("target") or "").strip().lower()
        if target:
            return target

        raw_text = result.raw_input or ""
        for pattern in OCR_ROUTE_TARGET_PATTERNS:
            match = pattern.search(raw_text)
            if not match:
                continue

            candidate = match.group(1).strip().lower()
            if candidate in OCR_ROUTE_FILLER or candidate in OCR_ROUTE_DIRECTIONS:
                continue
            return candidate

        return ""

    def _frame_has_target_detection(self, detections, target: str) -> bool:
        """Return True if any detection in the frame matches the requested target."""
        target_lower = (target or "").strip().lower()
        if not target_lower:
            return False

        for det in detections:
            class_name = det["cls_name"].lower()
            if target_lower in class_name or class_name in target_lower:
                return True
        return False

    def _scan_target_frames_for_ocr(self, ocr, target: str) -> tuple[int, str]:
        """
        Run detection across several frames and OCR only the frames containing the target.

        Returns:
            (matched_frame_count, best_text)
        """
        perception = self._get_perception()
        if perception is None:
            return 0, ""

        matched_frame_count = 0
        candidates = []

        print(f"Scanning {OCR_TARGET_SCAN_FRAMES} frame(s) for '{target}' before OCR...")

        for _ in range(OCR_TARGET_SCAN_FRAMES):
            frame = self._capture_frame()
            if frame is None:
                break

            detections, _, _, _ = perception._detect_objects(frame)
            if not self._frame_has_target_detection(detections, target):
                time.sleep(0.05)
                continue

            matched_frame_count += 1
            print(f"Detected '{target}' in frame {matched_frame_count}. Running OCR...")

            good_results = self._extract_best_ocr_from_frame(ocr, frame)
            candidate = self._build_ocr_candidate(good_results)
            if candidate is not None:
                candidates.append(candidate)

            time.sleep(0.05)

        return matched_frame_count, self._choose_best_ocr_candidate(candidates)

    # ══════════════════════════════════════════════════════════════════════════
    # Intent handlers
    # ══════════════════════════════════════════════════════════════════════════

    def _handle_environment(self, result) -> str:
        """
        Handle ENVIRONMENT_QUERY using the full AIPerception pipeline.

        WHY WE FEED MULTIPLE FRAMES:
        ─────────────────────────────
        YOLO's tracker (model.track) needs several consecutive frames before
        it assigns stable IDs to detections.  On the very first frame every
        box.id is None, so ai_perception.py skips them all — this is why the
        system says "no objects" even when objects are clearly visible.

        Fix: feed TRACKER_WARMUP_FRAMES frames through the pipeline, saving
        the last result. This avoids the bug where capturing a *new* frame
        after the warmup loop could land on a video loop reset (frame 0),
        causing the tracker to lose all state and report zero detections.

        Using the last warmup frame guarantees the tracker is fully warm.
        """
        perception = self._get_perception()
        if perception is None:
            return "Object detection is not available right now."

        # ── Feed consecutive frames to warm up the tracker ────────────────────
        # IMPORTANT: We save `last_result` from the final warmup iteration
        # and use it directly — no extra _capture_frame() call after the loop.
        # This prevents the tracker-reset bug when the video loops back to
        # frame 0 between the warmup and the final capture.
        print(f"Warming up tracker ({TRACKER_WARMUP_FRAMES} frames)...")
        last_result: FrameResult = None

        for i in range(TRACKER_WARMUP_FRAMES):
            frame = self._capture_frame()
            if frame is None:
                return "I could not read a frame from the video stream."

            last_result = perception.process_frame_for_orchestrator(frame)

            # Let the background thread advance to the next video frame
            time.sleep(0.05)
        # ─────────────────────────────────────────────────────────────────────

        # ── Use the last warmup result (tracker is fully warm here) ───────────
        if last_result is None:
            return "I could not process the video stream."

        frame_result: FrameResult = last_result

        print(
            f"Detected {len(frame_result.objects)} object(s): "
            f"{[o.class_name for o in frame_result.objects]}"
        )
        # ─────────────────────────────────────────────────────────────────────

        # ── Build spoken response ─────────────────────────────────────────────
        location_hint = result.entities.get("location_hint", "")
        target        = result.entities.get("target")

        response = self._build_speech(frame_result)

        if location_hint:
            response = f"Scanning {location_hint}. " + response

        if target:
            found = [
                o for o in frame_result.objects
                if target.lower() in o.class_name.lower()
            ]
            if found:
                obj = found[0]
                response = (
                    f"I found a {obj.class_name} at {obj.distance_m:.1f} meters. "
                    + response
                )
            else:
                response = (
                    f"I could not find a {target} in the current view. " + response
                )
        # ─────────────────────────────────────────────────────────────────────

        return response

    def _build_speech(self, frame_result: FrameResult) -> str:
        """
        Build a spoken description from ALL detected objects regardless of
        their hazard priority.

        ai_perception.py's built-in to_speech() only reports CRITICAL/HIGH
        objects, which produces "no objects detected" even when many SAFE
        objects are present.  This method always reports everything.

        Priority order:
          1. CRITICAL  → "Warning! X approaching at Y metres."
          2. HIGH      → "Caution. X at Y metres."
          3. SAFE      → de-duplicated list with distances
        """
        if not frame_result.objects:
            return "No objects detected."

        motion_words = {
            "APPROACHING": "approaching",
            "LEAVING": "leaving",
            "STATIC": "static",
        }

        lines = []
        for obj in sorted(frame_result.objects, key=lambda o: o.distance_m):
            direction = getattr(obj, "direction", "ahead")
            motion = motion_words.get(getattr(obj, "motion", "STATIC"), "static")
            lines.append(f"{obj.class_name} at {obj.distance_m:.1f} meters {direction}, {motion}")

        return "\n".join(lines)

    # ── Other handlers ────────────────────────────────────────────────────────

    def _handle_ocr(self, result) -> str:
        """Run OCR on every scanned frame where object detection sees the target."""
        ocr = self._get_ocr()
        if ocr is None:
            return "OCR system is not available right now."

        target = self._extract_ocr_target(result)

        if target:
            matched_frame_count, best_text = self._scan_target_frames_for_ocr(ocr, target)

            if matched_frame_count == 0:
                return f"I did not detect any {target} objects in the scanned frames."

            if not best_text:
                return (
                    f"I detected {target} in {matched_frame_count} frame"
                    f"{'s' if matched_frame_count != 1 else ''}, "
                    "but I could not read any confident text from them."
                )

            return (
                f"I detected {target} in {matched_frame_count} frame"
                f"{'s' if matched_frame_count != 1 else ''}. "
                f"The text says: {best_text}"
            )

        best_text = self._scan_frames_for_ocr(ocr, OCR_GENERAL_SCAN_FRAMES)
        if not best_text:
            return "I could not find any text in the image."

        return f"The text says: {best_text}"

    def _handle_battery(self) -> str:
        """Return battery status."""
        status = self._battery.read_status()
        if status is None:
            return "I could not read the device battery right now."

        percent = status["percent"]
        power_plugged = status["power_plugged"]
        secs_left = status["secs_left"]

        if power_plugged:
            if percent >= 100:
                return "Device battery is at 100 percent and fully charged."
            return f"Device battery is at {percent} percent and charging."

        duration = self._battery.format_duration(secs_left)
        if duration:
            return (
                f"Device battery is at {percent} percent. "
                f"Approximately {duration} remaining."
            )

        return f"Device battery is at {percent} percent."

    def _handle_emergency(self) -> str:
        """Trigger emergency alert."""
        # TODO: Add haptic GPIO trigger and BLE notification to companion app
        print("EMERGENCY TRIGGERED")
        return (
            "Emergency alert activated. "
            "Notifying your contacts. Stay calm, help is on the way."
        )

    def _handle_unknown(self) -> str:
        """Handle unrecognized commands."""
        return (
            "I did not understand that. "
            "You can ask me to describe your surroundings, read text, "
            "or check the battery."
        )

    # ── Router ────────────────────────────────────────────────────────────────

    def _route(self, result) -> str:
        """Route a CommandResult to the correct handler."""
        intent = result.intent
        if intent == Intent.OCR or self._looks_like_text_query(result.raw_input):
            return self._handle_ocr(result)
        elif intent == Intent.ENVIRONMENT_QUERY:
            return self._handle_environment(result)
        elif intent == Intent.BATTERY:
            return self._handle_battery()
        elif intent == Intent.EMERGENCY:
            return self._handle_emergency()
        else:
            return self._handle_unknown()

    # ── Single command cycle ──────────────────────────────────────────────────

    def run_once(self):
        """Run one full pipeline cycle: Record → STT → Extract → Route → TTS"""
        print("\n" + "=" * 50)
        print("Listening...")

        # Step 1: Record
        try:
            audio, sample_rate = record_audio_to_memory(duration=3)
        except Exception as e:
            print(f"Recording error: {e}")
            speak("Sorry, I could not record your voice.")
            return

        # Step 2: Speech to text
        try:
            text = speech_to_text_from_memory(audio, sample_rate)
            print(f'You said: "{text}"')
            if not text.strip():
                print("No speech detected.")
                speak("I did not hear anything. Please try again.")
                return
        except Exception as e:
            print(f"STT error: {e}")
            speak("Sorry, I could not understand your speech.")
            return

        # Step 3: Extract intent
        try:
            result = self.extractor.extract(text)
            print(f"Intent     : {result.intent.value}")
            if result.entities:
                print(f"Entities   : {result.entities}")
        except Exception as e:
            print(f"Extraction error: {e}")
            speak("Sorry, I could not process your command.")
            return

        # Step 4: Route to module
        try:
            response = self._route(result)
            print(f"Response   : {response}")
        except Exception as e:
            print(f"Routing error: {e}")
            speak("Sorry, something went wrong.")
            return

        # Step 5: Speak response
        speak(response)

    # ── Run modes ─────────────────────────────────────────────────────────────

    def run_interactive(self):
        """Interactive mode — press Enter before each command."""
        print("Ru'ya — Interactive Mode  (type 'quit' to exit)")
        try:
            while True:
                print("\n" + "=" * 50)
                user_input = input(
                    "Press Enter to speak a command (or type 'quit'): "
                ).strip().lower()
                if user_input in ("quit", "exit", "q"):
                    break
                self.run_once()
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
            print("Goodbye!")

    def run_continuous(self):
        """Continuous mode — loops automatically."""
        print("Ru'ya — Continuous Mode  (Ctrl+C to stop)")
        try:
            while True:
                self.run_once()
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
            print("Goodbye!")

    def run_single(self):
        """Single command then exit."""
        try:
            self.run_once()
        finally:
            self.cleanup()
    
    def run_vision(self):
        interval_raw = (os.getenv("RUYA_VISION_INTERVAL") or "1.0").strip()
        try:
            interval = float(interval_raw)
        except ValueError:
            interval = 1.0
        interval = max(0.1, min(interval, 10.0))

        print("Ru'ya — Vision Mode (Ctrl+C to stop)")
        try:
            while True:
                frame = self._capture_frame()
                if frame is None:
                    time.sleep(0.2)
                    continue
                perception = self._get_perception()
                if perception is None:
                    time.sleep(1.0)
                    continue
                frame_result = perception.process_frame_for_orchestrator(frame)
                print(self._build_speech(frame_result), flush=True)
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
            print("Goodbye!")
            

HAZARD_COOLDOWN = {
    "CRITICAL": 4.0,   # seconds between repeating same class
    "HIGH":     8.0,
}
HAZARD_LOOP_INTERVAL = 0.15   # seconds between frame checks (~6 fps)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python orchestrator.py interactive  # Press Enter before each command")
        print("  python orchestrator.py continuous   # Loops automatically")
        print("  python orchestrator.py single       # One command then exit")
        print("  python orchestrator.py vision       # Continuous scene descriptions to stdout")
        return

    mode = sys.argv[1].lower()
    orc  = Orchestrator(start_hazard_watcher=(mode != "vision"))

    if mode == "interactive":
        orc.run_interactive()
    elif mode == "continuous":
        orc.run_continuous()
    elif mode == "single":
        orc.run_single()
    elif mode == "vision":
        orc.run_vision()
    else:
        print(f"Unknown mode: '{mode}'")
        print("Use: interactive, continuous, single, or vision")


if __name__ == "__main__":
    main()
