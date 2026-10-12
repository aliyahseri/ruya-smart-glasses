import sys
import os

# 1. Get the current hook module object
current_module = sys.modules.get('cv2')

# 2. Temporarily remove 'cv2' from sys.modules and modify sys.path to load the real binary cv2
current_dir = os.path.dirname(os.path.abspath(__file__))
original_path = sys.path.copy()
sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(current_dir)]

original_cv2_module = sys.modules.pop('cv2', None)

try:
    import cv2 as _real_cv2
finally:
    # Restore sys.path and sys.modules
    sys.path = original_path
    if current_module is not None:
        sys.modules['cv2'] = current_module

# 3. Expose all original cv2 attributes and functions in our module's namespace
globals().update({k: v for k, v in _real_cv2.__dict__.items() if k != '__name__'})

# 4. Define our direct ESP32 VideoCapture wrapper
class HookedVideoCapture(_real_cv2.VideoCapture):
    def __init__(self, source, *args, **kwargs):
        if isinstance(source, str) and (source.endswith("testvideo.mp4") or "testvideo.mp4" in source):
            # Direct routing to the verified ESP32 stream URL
            source = "http://192.168.1.54:81/stream"
            print(f"\n[Hook] Redirecting cv2.VideoCapture to live ESP32 Camera: {source}", flush=True)
            print("[Hook] IMPORTANT: Make sure your ESP32 web browser tab/stream is CLOSED, as ESP32 CAM only supports one client at a time!\n", flush=True)
            
        super().__init__(source, *args, **kwargs)

# Bind the hooked class to both namespaces
VideoCapture = HookedVideoCapture
_real_cv2.VideoCapture = HookedVideoCapture
