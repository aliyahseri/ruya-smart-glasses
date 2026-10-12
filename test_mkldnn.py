import easyocr
import numpy as np

print("Initializing EasyOCR reader (CPU)...")
reader = easyocr.Reader(["en"], gpu=False, verbose=False)

# Make a dummy 100x100 BGR image (white image)
img = np.ones((100, 100, 3), dtype=np.uint8) * 255

print("Running EasyOCR on dummy image...")
try:
    results = reader.readtext(img)
    print("Success! Results:", results)
except Exception as e:
    print("EasyOCR test failed:", e)
