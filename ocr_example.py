#!/usr/bin/env python3
"""
Simple OCR example - how to use the OCR models
"""

from ocr_processor import OCRProcessor
import cv2

def simple_cr_example():
    """Example of how to call OCR models."""
    
    # Initialize OCR processor
    processor = OCRProcessor()
    
    # Method 1: Extract text from image file
    text_results = processor.extract_text("path/to/your/image.jpg")
    print("Detected text:", text_results)
    
    # Method 2: Get text as single string
    text_string = processor.get_text_string("path/to/your/image.jpg")
    print("Text string:", text_string)
    
    # Method 3: From numpy array (e.g., from camera)
    # image_array = cv2.imread("path/to/image.jpg")
    # text_from_array = processor.get_text_string_from_array(image_array)
    # print("Text from array:", text_from_array)

if __name__ == "__main__":
    simple_ocr_example()
