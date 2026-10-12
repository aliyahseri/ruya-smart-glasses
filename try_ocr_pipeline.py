import sys
import os
from pathlib import Path

# Add workspace to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import cv2
print("Imported cv2, type:", type(cv2))
print("cv2.VideoCapture:", cv2.VideoCapture)

from orchestrator import Orchestrator
from command_extractor import CommandResult, Intent

def test_query(orc, text, intent):
    print(f"\n--- Testing command: '{text}' ({intent.value}) ---")
    result = CommandResult(intent=intent, raw_input=text)
    # Route command to get response
    response = orc._route(result)
    print(f"Response: {response}")

def main():
    print("\n=== Initializing Orchestrator ===")
    orc = Orchestrator()
    try:
        # 1. Battery query
        test_query(orc, "check battery", Intent.BATTERY)
        
        # 2. OCR query (will capture frames and run OCR Processor)
        test_query(orc, "what does this say?", Intent.OCR)
        
        # 3. Environment query (will load YOLO + Depth, warm up tracker)
        test_query(orc, "what's around me?", Intent.ENVIRONMENT_QUERY)

    except Exception as e:
        print(f"Pipeline execution encountered error: {e}")
    finally:
        orc.cleanup()

if __name__ == "__main__":
    main()
