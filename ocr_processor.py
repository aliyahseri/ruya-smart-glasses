#!/usr/bin/env python3
"""
Ru'ya Smart Glasses — OCR Processor
=====================================
Multi-backend OCR: EasyOCR (primary) → Tesseract (fallback).
Compatible with ARM64 (Raspberry Pi 4) and x86_64.
"""

import cv2
import numpy as np

# ── Backend imports ───────────────────────────────────────────────────────────

EASYOCR_AVAILABLE = False
TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    pass

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    pass

# (PaddleOCR support removed) Use EasyOCR as primary backend.


class OCRProcessor:
    """
    OCR processor with automatic backend selection.

    Priority: EasyOCR → Tesseract
    All backends expose the same API so the rest of the pipeline is unchanged.
    """

    def __init__(self, preferred_backend=None):
        """
        Args:
            "preferred_backend: Force a specific backend ('easyocr', 'tesseract').
                               If None, picks the best available automatically.
        """
        self.backend = None
        self._reader = None  # EasyOCR reader instance

        # Determine which backend to use
        if preferred_backend:
            self._init_backend(preferred_backend)
        else:
            # Auto-select in priority order
            for name in ("easyocr", "tesseract"):
                try:
                    self._init_backend(name)
                    break
                except Exception as e:
                    print(f"OCR: {name} init failed: {e}")
                    continue

        if self.backend is None:
            raise ImportError(
                "No OCR backend available. Install one of:\n"
                "  pip install easyocr          (recommended)\n"
                "  pip install pytesseract       (+ sudo apt install tesseract-ocr)"
            )

        print(f"OCR ready — backend: {self.backend}")

    # ── Backend initialisation ────────────────────────────────────────────────

    def _init_backend(self, name: str):
        name = name.lower().strip()

        if name == "easyocr" and EASYOCR_AVAILABLE:
            print("Initializing EasyOCR...")
            self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            self.backend = "easyocr"

        elif name == "tesseract" and TESSERACT_AVAILABLE:
            print("Initializing Tesseract OCR...")
            # Quick smoke test
            pytesseract.get_tesseract_version()
            self.backend = "tesseract"

        else:
            raise RuntimeError(f"Backend '{name}' is not installed.")

    # ── Unified extraction from numpy array ───────────────────────────────────

    def extract_text_from_array(self, image_array: np.ndarray) -> list:
        """
        Extract text from a BGR numpy array.

        Returns:
            list of { 'text': str, 'confidence': float, 'bbox': list }
        """
        if image_array is None or image_array.size == 0:
            print("OCR: Empty image array.")
            return []

        try:
            if self.backend == "easyocr":
                return self._easyocr_extract(image_array)
            elif self.backend == "tesseract":
                return self._tesseract_extract(image_array)
            # PaddleOCR removed; only easyocr and tesseract handled
        except Exception as e:
            print(f"OCR error ({self.backend}): {e}")
            # Try fallback
            return self._fallback_extract(image_array)

        return []

    def extract_text(self, image_path: str) -> list:
        """Extract text from an image file path."""
        img = cv2.imread(image_path)
        if img is None:
            print(f"OCR: Cannot read image at {image_path}")
            return []
        return self.extract_text_from_array(img)

    # ── Convenience helpers ───────────────────────────────────────────────────

    def get_text_string(self, image_path: str) -> str:
        """Return all detected text from an image file as a single string."""
        results = self.extract_text(image_path)
        return " ".join(r["text"] for r in results)

    def get_text_string_from_array(self, image_array: np.ndarray) -> str:
        """Return all detected text from a numpy array as a single string."""
        results = self.extract_text_from_array(image_array)
        return " ".join(r["text"] for r in results)

    # ── Backend-specific implementations ──────────────────────────────────────

    def _easyocr_extract(self, img: np.ndarray) -> list:
        """EasyOCR returns list of (bbox, text, confidence)."""
        raw = self._reader.readtext(img)
        extracted = []
        for bbox, text, conf in raw:
            text = text.strip()
            if text:
                extracted.append({
                    "text": text,
                    "confidence": float(conf),
                    "bbox": bbox,
                })
        return extracted

    def _tesseract_extract(self, img: np.ndarray) -> list:
        """Tesseract via pytesseract — returns word-level results."""
        # Convert BGR → RGB for Tesseract
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)

        extracted = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            if text and conf > 0:
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                extracted.append({
                    "text": text,
                    "confidence": conf / 100.0,
                    "bbox": bbox,
                })
        return extracted

    # PaddleOCR removed — only EasyOCR and Tesseract are supported

    # ── Fallback logic ────────────────────────────────────────────────────────

    def _fallback_extract(self, img: np.ndarray) -> list:
        """If the primary backend fails, try the others."""
        fallbacks = []
        if self.backend != "easyocr" and EASYOCR_AVAILABLE:
            fallbacks.append(("easyocr", self._easyocr_extract))
        if self.backend != "tesseract" and TESSERACT_AVAILABLE:
            fallbacks.append(("tesseract", self._tesseract_extract))

        for name, fn in fallbacks:
            try:
                # Lazy-init if needed
                if name == "easyocr" and self._reader is None:
                    self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                print(f"OCR: Falling back to {name}...")
                return fn(img)
            except Exception as e:
                print(f"OCR fallback {name} also failed: {e}")
                continue

        return []


# ── Quick test ────────────────────────────────────────────────────────────────

def test_ocr():
    available = []
    if EASYOCR_AVAILABLE:
        available.append("easyocr")
    if TESSERACT_AVAILABLE:
        available.append("tesseract")
    print(f"Available OCR backends: {available or 'NONE'}")

    if not available:
        print("No OCR backend installed.")
        return

    try:
        processor = OCRProcessor()
        print(f"OCR processor initialized with backend: {processor.backend}")
    except Exception as e:
        print(f"OCR test failed: {e}")


if __name__ == "__main__":
    test_ocr()