#!/usr/bin/env python3
"""
Ru'ya Smart Glasses - Text to Speech (Windows, Offline)
=======================================================
Supports fully local/offline engines:
1. Python Piper package + ONNX voice model for the most natural voice.
2. Piper CLI + ONNX voice model if available.
3. Windows SAPI5 via pyttsx3 as a fallback.

Environment variables:
  RUYA_TTS_ENGINE   = auto | piper | sapi
  RUYA_TTS_SPEED    = speaking speed multiplier (default 1.3, 1.0 = normal)
  RUYA_PIPER_PATH   = optional path to piper.exe
  RUYA_PIPER_MODEL  = path to a Piper .onnx voice model
  RUYA_PIPER_CONFIG = optional path to the matching .json config
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
import winsound
from pathlib import Path

import pyttsx3


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TTS_DIR = BASE_DIR / "tts_models"
PIPER_MODEL_GLOB = "*.onnx"
PREFERRED_PIPER_MODEL_NAMES = (
    "en_US-amy-medium.onnx",
    "en_US-amy-low.onnx",
)
_PIPER_VOICE = None
_PIPER_VOICE_KEY = None
_sapi_engine = None


def _get_sapi_engine():
    """Return a shared SAPI5 engine instance, creating it if needed."""
    global _sapi_engine
    if _sapi_engine is None:
        _sapi_engine = pyttsx3.init(driverName="sapi5")
        _pick_sapi_voice(_sapi_engine)
        _sapi_engine.setProperty("rate", max(80, min(int(145 * _tts_speed()), 260)))
        _sapi_engine.setProperty("volume", 1.0)
    return _sapi_engine


def _normalize_text(text: str) -> str:
    """Clean text so both engines read it more naturally."""
    text = re.sub(r"\s+", " ", text.strip())
    text = text.replace("%", " percent")
    text = text.replace("&", " and ")
    return text


def _tts_speed() -> float:
    """Return a safe speaking-speed multiplier."""
    raw = os.getenv("RUYA_TTS_SPEED", "1.3").strip()
    try:
        speed = float(raw)
    except ValueError:
        speed = 1.0

    return max(0.5, min(speed, 2.0))


def _pick_sapi_voice(engine) -> str | None:
    """Prefer the less robotic Windows voice when available."""
    voices = engine.getProperty("voices")
    preferred = ("mark", "zira", "david")

    for wanted in preferred:
        for voice in voices:
            if wanted in voice.name.lower():
                engine.setProperty("voice", voice.id)
                return voice.name

    if voices:
        engine.setProperty("voice", voices[0].id)
        return voices[0].name

    return None


def _speak_with_sapi(text: str) -> None:
    """Fallback local TTS using Windows SAPI."""
    engine = pyttsx3.init(driverName="sapi5")
    try:
        chosen = _pick_sapi_voice(engine)
        if chosen:
            print(f"[TTS] Engine: SAPI5 ({chosen})")
        else:
            print("[TTS] Engine: SAPI5")

        engine.setProperty("rate", max(80, min(int(145 * _tts_speed()), 260)))
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
    finally:
        try:
            engine.stop()
        except Exception:
            pass


def _candidate_piper_paths() -> list[Path]:
    """Return likely locations for piper.exe."""
    candidates = []

    env_path = os.getenv("RUYA_PIPER_PATH")
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            BASE_DIR / "piper.exe",
            BASE_DIR / "piper" / "piper.exe",
            BASE_DIR / "piper" / "bin" / "piper.exe",
            BASE_DIR / "tools" / "piper.exe",
            BASE_DIR / "tools" / "piper" / "piper.exe",
        ]
    )

    which_path = shutil.which("piper")
    if which_path:
        candidates.append(Path(which_path))

    unique = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _find_piper_binary() -> Path | None:
    for path in _candidate_piper_paths():
        if path.is_file():
            return path
    return None


def _find_piper_model() -> tuple[Path | None, Path | None]:
    """Return (model, config) if a local Piper voice is available."""
    env_model = os.getenv("RUYA_PIPER_MODEL")
    env_config = os.getenv("RUYA_PIPER_CONFIG")
    if env_model:
        model = Path(env_model)
        config = Path(env_config) if env_config else model.with_suffix(model.suffix + ".json")
        return (model if model.is_file() else None, config if config.is_file() else None)

    if not DEFAULT_TTS_DIR.exists():
        return None, None

    models = sorted(DEFAULT_TTS_DIR.rglob(PIPER_MODEL_GLOB))

    for preferred_name in PREFERRED_PIPER_MODEL_NAMES:
        for model in models:
            if model.name.lower() == preferred_name.lower():
                config = model.with_suffix(model.suffix + ".json")
                return model, (config if config.is_file() else None)

    for model in models:
        config = model.with_suffix(model.suffix + ".json")
        return model, (config if config.is_file() else None)

    return None, None


def _piper_python_available() -> bool:
    return importlib.util.find_spec("piper") is not None


def _load_python_piper_voice(model: Path, config: Path | None):
    """Load and cache a Piper voice from the Python package."""
    global _PIPER_VOICE, _PIPER_VOICE_KEY

    if config is None:
        raise RuntimeError(f"Missing Piper config for model: {model.name}")

    key = (str(model.resolve()), str(config.resolve()))
    if _PIPER_VOICE is not None and _PIPER_VOICE_KEY == key:
        return _PIPER_VOICE

    from piper import PiperVoice

    voice = PiperVoice.load(model_path=model, config_path=config, use_cuda=False)
    _PIPER_VOICE = voice
    _PIPER_VOICE_KEY = key
    return voice


def _play_wav_file(path: Path) -> None:
    winsound.PlaySound(str(path), winsound.SND_FILENAME)


def _speak_with_python_piper(text: str) -> None:
    """Speak text using the installed Python Piper package."""
    from piper.config import SynthesisConfig

    model, config = _find_piper_model()
    if model is None:
        raise RuntimeError("Piper voice model not found")

    voice = _load_python_piper_voice(model, config)
    print(f"[TTS] Engine: Piper Python ({model.name})")
    syn_config = SynthesisConfig(length_scale=1.0 / _tts_speed())

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        wav_path = Path(handle.name)

    try:
        with wave.open(str(wav_path), "wb") as wav_file:
            voice.synthesize_wav(text, wav_file, syn_config=syn_config)
        _play_wav_file(wav_path)
    finally:
        try:
            wav_path.unlink(missing_ok=True)
        except Exception:
            pass


def _speak_with_piper_cli(text: str) -> None:
    """Speak text using a local Piper CLI binary and model."""
    binary = _find_piper_binary()
    model, config = _find_piper_model()
    if binary is None or model is None:
        raise RuntimeError("Piper binary or model not found")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        wav_path = Path(handle.name)

    try:
        command = [str(binary), "--model", str(model), "--output_file", str(wav_path)]
        if config is not None:
            command.extend(["--config", str(config)])

        completed = subprocess.run(
            command,
            input=text,
            text=True,
            capture_output=True,
            check=False,
        )

        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(stderr or f"Piper CLI failed with exit code {completed.returncode}")

        print(f"[TTS] Engine: Piper CLI ({model.name})")
        _play_wav_file(wav_path)
    finally:
        try:
            wav_path.unlink(missing_ok=True)
        except Exception:
            pass


def _preferred_engine() -> str:
    return os.getenv("RUYA_TTS_ENGINE", "auto").strip().lower()


def text_to_speech(text: str) -> None:
    """Speak text aloud using the best available local offline engine."""
    if not text or not text.strip():
        return

    text = _normalize_text(text)
    preview = text[:80] + ("..." if len(text) > 80 else "")
    print(f"[TTS] Speaking: {preview}")

    engine = _preferred_engine()
    if engine not in {"auto", "piper", "sapi"}:
        engine = "auto"

    if engine in {"auto", "piper"}:
        if _piper_python_available():
            try:
                _speak_with_python_piper(text)
                return
            except Exception as exc:
                print(f"[TTS] Piper Python unavailable: {exc}")

        try:
            _speak_with_piper_cli(text)
            return
        except Exception as exc:
            if engine == "piper":
                print(f"[TTS] Piper error: {exc}")
            else:
                print(f"[TTS] Piper unavailable, falling back to SAPI5: {exc}")

    try:
        _speak_with_sapi(text)
    except Exception as exc:
        print(f"[TTS] SAPI5 error: {exc}")


def speak_urgent(text: str) -> None:
    if not text or not text.strip():
        return
    text = _normalize_text(text)
    print(f"[TTS] Urgent: {text[:80]}")
    text_to_speech(text)


if __name__ == "__main__":
    test = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Ru'ya Smart Glasses are working."
    )
    text_to_speech(test)