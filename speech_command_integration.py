#!/usr/bin/env python3
"""
Ru'ya Smart Glasses — Speech to Command Integration
====================================================
Integrates speech recognition with LLM command extraction.

Flow:
1. Record audio from microphone
2. Convert speech to text using Whisper
3. Extract structured command using LLM
4. Output JSON command result
"""

import json
import sys
import os
from pathlib import Path
import gc
import numpy as np
import psutil  # For memory monitoring

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from command_extractor import CommandExtractor


def record_audio_to_memory(duration=3, sample_rate=16000):
    """
    Records audio directly to memory with optimized settings.

    Returns:
        tuple: (audio_data, sample_rate)
    """
    import sounddevice as sd
    import time
    import threading

    def show_recording_indicator(duration):
        """Show visual recording indicator with countdown"""
        for remaining in range(duration, 0, -1):
            indicator = " Recording" if remaining % 2 == 0 else " Recording"
            print(f"\r{indicator} - {remaining}s", end="", flush=True)
            time.sleep(1)
        print("\r Done!                    ")

    print(" Speak now...")
    time.sleep(0.5)

    # Start visual indicator in separate thread
    indicator_thread = threading.Thread(
        target=show_recording_indicator, args=(duration,)
    )
    indicator_thread.daemon = True
    indicator_thread.start()

    # Record at 16kHz directly to reduce processing
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='int16'
    )

    sd.wait()

    return recording, sample_rate


def speech_to_text_from_memory(audio_data, sample_rate):
    """
    Converts speech audio from memory to text with memory optimization.

    Parameters:
        audio_data: numpy array of audio samples
        sample_rate: sampling rate in Hz

    Returns:
        str: Recognized text
    """
    import whisper

    # Convert to float32 and normalize
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0

    # FIX: flatten to 1D array — prevents 153GB allocation error
    # sounddevice returns shape (N, 1), whisper needs shape (N,)
    audio_data = audio_data.flatten()

    # Load model once and reuse (tiny = ~150MB, works on Pi)
    if not hasattr(speech_to_text_from_memory, 'model'):
        print(" Loading model...")
        speech_to_text_from_memory.model = whisper.load_model("tiny", device="cpu")
        print(" Ready")

    # Transcribe with minimal memory usage
    result = speech_to_text_from_memory.model.transcribe(
        audio_data,
        fp16=False,
        verbose=False,
        task="transcribe",
        language="en"  # Specify language to reduce processing
    )

    # Clear audio data immediately after use
    del audio_data
    gc.collect()

    return result["text"].strip()


def main():
    """Main integration loop for speech-to-command processing."""

    print(" Ru'ya Smart Glasses — Speech Command System")
    print("=" * 50)
    print(" Speak your command and I'll convert it to structured JSON")
    print(" Examples: 'battery', 'read the menu', 'what's in front of me', 'help'")
    print(" No audio files or text history will be saved")
    print()

    # Initialize command extractor
    extractor = CommandExtractor()

    try:
        while True:
            print("\n" + "=" * 50)

            # Step 1: Record audio to memory (3 seconds)
            audio_data, sample_rate = record_audio_to_memory(duration=3)

            # Step 2: Convert speech to text
            print("\n Converting speech to text...")
            try:
                transcribed_text = speech_to_text_from_memory(audio_data, sample_rate)
                print(f" Transcribed: \"{transcribed_text}\"")

                if not transcribed_text.strip():
                    print("  No speech detected. Please try again.")
                    continue

            except Exception as e:
                print(f" Speech-to-text error: {e}")
                continue

            # Step 3: Extract command using LLM
            print("\n Extracting command...")
            try:
                command_result = extractor.extract(transcribed_text)

                # Step 4: Output JSON result
                print("\n Command Result:")
                print("-" * 30)
                json_output = json.dumps(command_result.to_dict(), indent=2)
                print(json_output)
                print("-" * 30)

                # Check for emergency
                if command_result.is_emergency():
                    print(" EMERGENCY DETECTED!")
                    print(" Calling for help...")
                    # TODO: trigger emergency protocols via orchestrator

            except Exception as e:
                print(f" Command extraction error: {e}")
                continue

            # Cleanup
            transcribed_text = None
            command_result = None
            audio_data = None
            aggressive_cleanup()

            # Show memory usage
            memory_mb = get_memory_usage()
            print(f"Memory: {memory_mb:.1f}MB")

            # Ask if user wants to continue
            print("\n Try another command? (y/n): ", end="")
            response = input().strip().lower()
            if response not in ['y', 'yes', '']:
                break

    except KeyboardInterrupt:
        print("\n\n Goodbye!")
    except Exception as e:
        print(f"\n Unexpected error: {e}")


def single_command_test():
    """Test a single command without interactive loop."""

    print(" Single Command Test Mode")
    print("=" * 40)

    # Record audio to memory (3 seconds)
    audio_data, sample_rate = record_audio_to_memory(duration=3)

    try:
        # Speech to text
        text = speech_to_text_from_memory(audio_data, sample_rate)
        print(f" Transcribed: \"{text}\"")

        # Command extraction
        extractor = CommandExtractor()
        result = extractor.extract(text)

        # Output
        json_output = json.dumps(result.to_dict(), indent=2)
        print(f"\n JSON Output:\n{json_output}")

    except Exception as e:
        print(f" Error: {e}")


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def aggressive_cleanup():
    """Perform aggressive memory cleanup."""
    for _ in range(3):
        gc.collect()


def clear_cache():
    """Clear Whisper model cache and force garbage collection."""
    print("Cleaning up cache...")

    if hasattr(speech_to_text_from_memory, 'model'):
        del speech_to_text_from_memory.model
        print("Whisper model cleared")

    aggressive_cleanup()

    memory_mb = get_memory_usage()
    print(f"Memory usage: {memory_mb:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            single_command_test()
        elif sys.argv[1] == "--clear-cache":
            clear_cache()
        else:
            print("Usage:")
            print("  python speech_command_integration.py           # Interactive mode")
            print("  python speech_command_integration.py --test    # Single command test")
            print("  python speech_command_integration.py --clear-cache  # Clear cache")
    else:
        main()