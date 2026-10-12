#!/usr/bin/env python3
"""
Ru'ya Smart Glasses — Orchestrator
====================================
Central controller that connects all available modules.

Current modules:
  - STT  : Whisper (speech → text)
  - CMD  : CommandExtractor (text → intent)
  - OCR  : PaddleOCR (image → text)
  - TTS  : pyttsx3 (text → speech)

Pipeline:
  Microphone → STT → CommandExtractor → Route → [OCR / Battery / Environment / Emergency] → TTS → Speaker

Run:
  python orchestrator.py interactive   ← press Enter before each command
  python orchestrator.py continuous    ← loops automatically
  python orchestrator.py single        ← one command then exit

NOTE:
  Camera is commented out for laptop testing.
  Place an image named "test_image.jpg" in the same folder to test OCR.
  When testing on the actual device, uncomment the camera lines.
"""

import sys
import os
import time
from pathlib import Path
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None

# ── CAMERA IMPORTS (uncomment when testing on actual device) ──────────────────
# import cv2
# import numpy as np
# ─────────────────────────────────────────────────────────────────────────────

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent))

# ── Module imports ────────────────────────────────────────────────────────────
from speech_command_integration import record_audio_to_memory, speech_to_text_from_memory
from command_extractor import CommandExtractor, Intent
from ocr_processor import OCRProcessor
from text_to_speech import text_to_speech as speak

# ── Test image path ───────────────────────────────────────────────────────────
# Place any image with text in the same folder and name it test_image.jpg
TEST_IMAGE_PATH = str(Path(__file__).parent / "test_image.jpg")


class BatteryManager:
    """Read the current device battery when the operating system exposes it."""

    @staticmethod
    def read_status() -> Optional[dict]:
        if psutil is None:
            return None

        try:
            battery = psutil.sensors_battery()
        except Exception as exc:
            print(f"Battery read via psutil failed: {exc}")
            return None

        if battery is None or battery.percent is None:
            return None

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

    @staticmethod
    def format_duration(seconds: Optional[int]) -> str:
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


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class Orchestrator:
    """
    Central controller for Ru'ya Smart Glasses.
    Connects STT → CommandExtractor → Module → TTS.
    """

    def __init__(self):
        print("\nInitializing Ru'ya Smart Glasses...")

        # Command extractor — always loaded at startup
        self.extractor = CommandExtractor()
        print("  Command extractor   : ready")

        # OCR — loaded on first use to save memory
        self._ocr = None
        print("  OCR processor       : ready (loads on first use)")
        self._battery = BatteryManager()
        print("  Battery manager     : ready")

        # ── CAMERA (uncomment when testing on actual device) ──────────────────
        # self._camera = None
        # print("  Camera              : ready (opens on first use)")
        # ─────────────────────────────────────────────────────────────────────
        print("  Camera              : DISABLED (reading from test_image.jpg)")

        print("System ready.\n")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def cleanup(self):
        """Release resources."""
        # ── CAMERA CLEANUP (uncomment when testing on actual device) ──────────
        # if self._camera is not None:
        #     self._camera.release()
        #     self._camera = None
        #     print("Camera released.")
        # ─────────────────────────────────────────────────────────────────────
        pass

    # ── OCR loader ────────────────────────────────────────────────────────────

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

    # ── CAMERA METHODS (uncomment when testing on actual device) ──────────────
    # def _get_camera(self):
    #     """Open camera on first use."""
    #     if self._camera is None:
    #         try:
    #             self._camera = cv2.VideoCapture(0)
    #             if not self._camera.isOpened():
    #                 raise RuntimeError("Cannot open camera.")
    #             print("Camera opened.")
    #         except Exception as e:
    #             print(f"Camera open failed: {e}")
    #             self._camera = False
    #     return self._camera if self._camera else None
    #
    # def _capture_frame(self):
    #     """Capture a single frame from the camera as a numpy array."""
    #     cam = self._get_camera()
    #     if cam is None:
    #         return None
    #     try:
    #         for _ in range(3):
    #             cam.read()
    #         ret, frame = cam.read()
    #         if not ret or frame is None:
    #             print("Failed to capture frame.")
    #             return None
    #         return frame
    #     except Exception as e:
    #         print(f"Capture error: {e}")
    #         return None
    # ─────────────────────────────────────────────────────────────────────────

    # ── Intent handlers ───────────────────────────────────────────────────────

    def _handle_ocr(self, result) -> str:
        """Read text from test_image.jpg (camera disabled for laptop testing)."""
        ocr = self._get_ocr()
        if ocr is None:
            return "OCR system is not available right now."

        # CAMERA VERSION (uncomment when testing on actual device)
        # print("Capturing image for OCR...")
        # frame = self._capture_frame()
        # if frame is None:
        #     return "I could not access the camera."
        # text_results = ocr.extract_text_from_array(frame)

        # IMAGE FILE VERSION (laptop testing)
        print(f"Reading from: {TEST_IMAGE_PATH}")
        if not Path(TEST_IMAGE_PATH).exists():
            return (
                "No test image found. "
                "Please place an image named test_image.jpg "
                "in the same folder as this file."
            )
        text_results = ocr.extract_text(TEST_IMAGE_PATH)

        if not text_results:
            return "I could not find any text in the image."

        # Filter low confidence results
        good_results = [r for r in text_results if r["confidence"] >= 0.6]
        if not good_results:
            return "I detected some text but I am not confident enough to read it."

        combined = " ".join([r["text"] for r in good_results])
        return f"The text says: {combined}"

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

    def _handle_environment(self, result) -> str:
        """Describe the environment around the user."""
        # TODO: Replace with real object detection when Abduallah's module is ready
        location = result.entities.get("location_hint", "around you")
        target   = result.entities.get("target")
        if target:
            return f"I am scanning for a {target} {location}. Object detection is not connected yet."
        else:
            return f"I am scanning {location}. Object detection is not connected yet."

    def _handle_emergency(self) -> str:
        """Trigger emergency alert."""
        # TODO: Add haptic GPIO trigger and BLE notification to companion app
        print("EMERGENCY TRIGGERED")
        return "Emergency alert activated. Notifying your contacts. Stay calm, help is on the way."

    def _handle_unknown(self) -> str:
        """Handle unrecognized commands."""
        return (
            "I did not understand that. "
            "You can ask me to read text, describe your surroundings, or check the battery."
        )

    # ── Router ────────────────────────────────────────────────────────────────

    def _route(self, result) -> str:
        """Route a CommandResult to the correct handler."""
        intent = result.intent
        if intent == Intent.OCR:
            return self._handle_ocr(result)
        elif intent == Intent.BATTERY:
            return self._handle_battery()
        elif intent == Intent.ENVIRONMENT_QUERY:
            return self._handle_environment(result)
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

        # Step 5: Speak the response  ← THIS WAS THE MISSING LINE
        speak(response)

    # ── Run modes ─────────────────────────────────────────────────────────────

    def run_interactive(self):
        """Interactive mode — press Enter before each command."""
        print("Ru'ya — Interactive Mode  (type 'quit' to exit)")
        try:
            while True:
                print("\n" + "=" * 50)
                user_input = input("Press Enter to speak a command (or type 'quit'): ").strip().lower()
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


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python orchestrator.py interactive  # Press Enter before each command")
        print("  python orchestrator.py continuous   # Loops automatically")
        print("  python orchestrator.py single       # One command then exit")
        return

    mode = sys.argv[1].lower()
    orc  = Orchestrator()

    if mode == "interactive":
        orc.run_interactive()
    elif mode == "continuous":
        orc.run_continuous()
    elif mode == "single":
        orc.run_single()
    else:
        print(f"Unknown mode: '{mode}'")
        print("Use: interactive, continuous, or single")


if __name__ == "__main__":
    main()
