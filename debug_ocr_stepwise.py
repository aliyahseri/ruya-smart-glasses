#!/usr/bin/env python3
"""
Step-by-step OCR debugging
"""

import cv2
import numpy as np

def debug_step_by_step():
    print("=== Step-by-step OCR Debug ===")
    
    # Step 1: Check if image exists and can be read
    print("Step 1: Reading image...")
    img = cv2.imread("test_image.jpg")
    if img is None:
        print("ERROR: Cannot read test_image.jpg")
        print("Make sure the image exists and is a valid format")
        return
    print(f"SUCCESS: Image loaded - Shape: {img.shape}, Type: {img.dtype}")
    
    # Step 2: Check EasyOCR import
    print("\nStep 2: Checking EasyOCR...")
    try:
        import easyocr
        print("SUCCESS: EasyOCR imported")
    except ImportError as e:
        print(f"ERROR: EasyOCR not installed - {e}")
        print("Run: pip install easyocr")
        return

    # Step 3: Initialize EasyOCR Reader with minimal config
    print("\nStep 3: Initializing EasyOCR Reader...")
    try:
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        print("SUCCESS: EasyOCR Reader initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize EasyOCR - {e}")
        return

    # Step 4: Run OCR on the image
    print("\nStep 4: Running OCR...")
    try:
        results = reader.readtext(img)
        print(f"Raw OCR results: {results}")
        
        if not results:
            print("ERROR: OCR returned empty results")
            return

        print(f"SUCCESS: OCR found {len(results)} text regions")

        print("\nStep 5: Parsing results...")

        try:
            for i, (bbox, text, score) in enumerate(results):
                print(f"  {i+1}: Text='{text}', Confidence={score:.3f}")

        except Exception as e:
            print(f"ERROR while parsing: {e}")
                
    except Exception as e:
        print(f"ERROR: OCR processing failed - {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_step_by_step()
