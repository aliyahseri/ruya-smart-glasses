import sounddevice as sd
from scipy.io.wavfile import write
import sys
import time
import threading


def show_recording_indicator(duration):
    """Show visual recording indicator with countdown"""
    for remaining in range(duration, 0, -1):
        # Create recording animation
        indicator = "🔴 RECORDING" if remaining % 2 == 0 else "⏺️  RECORDING"
        print(f"\r{indicator} - {remaining} seconds remaining", end="", flush=True)
        time.sleep(1)
    print("\r✅ Recording complete!                     ")


def record_audio(filename="audio.wav", duration=5, sample_rate=44100):
    """
    Records audio from the system microphone.

    Parameters:
    filename (str): Name of the output audio file
    duration (int): Recording duration in seconds
    sample_rate (int): Sampling rate in Hz
    """

    print("🎤 Get ready to speak...")
    time.sleep(1)
    print("🔴 NOW RECORDING - Speak clearly!")
    
    # Start visual indicator in separate thread
    indicator_thread = threading.Thread(target=show_recording_indicator, args=(duration,))
    indicator_thread.daemon = True
    indicator_thread.start()

    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='int16'
    )

    sd.wait()

    write(filename, sample_rate, recording)

    print(f"💾 Audio saved as: {filename}")
    print(f"📊 Duration: {duration} seconds, Sample rate: {sample_rate}Hz")
