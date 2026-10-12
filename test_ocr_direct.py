#!/usr/bin/env python3
"""
Direct OCR test - Test the OCR model without orchestrator
"""

from ocr_processor import OCRProcessor
import cv2

def test_ocr_direct():
    """Test OCR directly on test_image.jpg"""
    
    print("=== Direct OCR Test ===")
    
    try:
        # Initialize OCR
        print("Initializing OCR...")
        processor = OCRProcessor()
        print("OCR initialized successfully!")
        
        # Test on the image file
        print("\nTesting on test_image.jpg...")
        results = processor.extract_text("test_image.jpg")
        
        if results:
            print(f"Found {len(results)} text regions:")
            for i, r in enumerate(results):
                print(f"  {i+1}: '{r['text']}' (confidence: {r['confidence']:.2f})")
        else:
            print("No text detected in the image.")
            
        # Also test as string
        text_string = processor.get_text_string("test_image.jpg")
        print(f"\nCombined text: '{text_string}'")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ocr_direct()
