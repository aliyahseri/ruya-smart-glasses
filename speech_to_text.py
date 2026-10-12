import whisper
import numpy as np
from scipy.io import wavfile

# Load Whisper model once at startup (using base model for faster download)
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Whisper model loaded successfully!")


def speech_to_text(audio_path):
    """
    Converts speech audio into text.

    Parameters:
    audio_path (str): Path to the audio file

    Returns:
    str: Recognized text
    """
    
    # Load audio using scipy instead of ffmpeg
    sample_rate, audio_data = wavfile.read(audio_path)
    
    # Convert to float32 and normalize
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    
    # Resample to 16kHz if needed (Whisper expects 16kHz)
    if sample_rate != 16000:
        # Simple resampling using scipy
        from scipy import signal
        num_samples = int(len(audio_data) * 16000 / sample_rate)
        audio_data = signal.resample(audio_data, num_samples)
        sample_rate = 16000
    
    # Transcribe using the loaded audio data
    result = model.transcribe(audio_data)
    
    text = result["text"].strip()
    
    return text
