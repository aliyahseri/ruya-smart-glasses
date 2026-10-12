import threading
import time
import os

import cv2
import numpy as np


class SerialCameraStream:
    def __init__(self, port: str, baudrate: int = 921600):
        import serial

        self._port = self._open_first_available_port(serial, port)
        self._baudrate = int(baudrate)
        self._ser = serial.Serial(port=self._port, baudrate=self._baudrate, timeout=0.1)
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._running = True
        self._ret = False
        self._frame = None
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    @staticmethod
    def _open_first_available_port(serial, port: str) -> str:
        try:
            test = serial.Serial(port=port, baudrate=115200, timeout=0.1)
            test.close()
            return port
        except Exception:
            pass

        candidates = [f"/dev/ttyUSB{i}" for i in range(4)]
        for p in candidates:
            if not os.path.exists(p):
                continue
            try:
                test = serial.Serial(port=p, baudrate=115200, timeout=0.1)
                test.close()
                return p
            except Exception:
                continue

        raise RuntimeError(f"Serial port not found. Tried: {port} and {', '.join(candidates)}")

    @staticmethod
    def _decode_raw_frame(payload: bytes):
        sizes = {
            160 * 120: (120, 160),
            240 * 176: (176, 240),
            320 * 240: (240, 320),
            640 * 480: (480, 640),
        }
        shape = sizes.get(len(payload))
        if shape is None:
            return None
        gray = np.frombuffer(payload, dtype=np.uint8).reshape(shape)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _update(self):
        while self._running:
            try:
                n = self._ser.in_waiting or 0
                chunk = self._ser.read(n if n > 0 else 1024)
            except Exception:
                time.sleep(0.05)
                continue

            if chunk:
                self._buf.extend(chunk)

            while True:
                framed = self._buf.find(b"\xaa\xbb")
                if framed != -1:
                    if framed > 0:
                        del self._buf[:framed]
                    if len(self._buf) < 2 + 1 + 4:
                        break
                    pkt_type = self._buf[2]
                    payload_len = int.from_bytes(self._buf[3:7], "little", signed=False)
                    total_len = 2 + 1 + 4 + payload_len + 2
                    if payload_len < 0 or payload_len > (6 * 1024 * 1024):
                        del self._buf[:1]
                        continue
                    if len(self._buf) < total_len:
                        break
                    if self._buf[total_len - 2 : total_len] != b"\xbb\xaa":
                        del self._buf[:1]
                        continue
                    payload = bytes(self._buf[7 : 7 + payload_len])
                    del self._buf[:total_len]
                    if pkt_type == 0x01:
                        arr = np.frombuffer(payload, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self._lock:
                                self._frame = frame
                                self._ret = True
                        continue
                    if pkt_type == 0x00:
                        frame = self._decode_raw_frame(payload)
                        if frame is not None:
                            with self._lock:
                                self._frame = frame
                                self._ret = True
                        continue

                start_jpeg = self._buf.find(b"\xff\xd8")
                start_raw = self._buf.find(b"\xaa\xbb")

                start = -1
                mode = None
                if start_jpeg != -1 and (start_raw == -1 or start_jpeg < start_raw):
                    start = start_jpeg
                    mode = "jpeg"
                elif start_raw != -1:
                    start = start_raw
                    mode = "raw"

                if start == -1:
                    if len(self._buf) > (1024 * 1024):
                        self._buf = self._buf[-2:]
                    break

                if start > 0:
                    del self._buf[:start]

                if mode == "raw":
                    end = self._buf.find(b"\xbb\xaa", 2)
                    if end == -1:
                        if len(self._buf) > (2 * 1024 * 1024):
                            self._buf = self._buf[:2]
                        break

                    payload = bytes(self._buf[2:end])
                    del self._buf[: end + 2]

                    frame = self._decode_raw_frame(payload)
                    if frame is not None:
                        with self._lock:
                            self._frame = frame
                            self._ret = True
                    continue

                end = self._buf.find(b"\xff\xd9", 2)
                if end == -1:
                    next_start = self._buf.find(b"\xff\xd8", 2)
                    if next_start != -1:
                        del self._buf[:next_start]
                        continue
                    if len(self._buf) > (2 * 1024 * 1024):
                        self._buf = self._buf[:2]
                    break

                jpg = bytes(self._buf[: end + 2])
                del self._buf[: end + 2]

                arr = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                with self._lock:
                    self._frame = frame
                    self._ret = True
                continue

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def stop(self):
        self._running = False
        try:
            self._thread.join(timeout=2)
        except Exception:
            pass
        try:
            self._ser.close()
        except Exception:
            pass
