"""Tests for the wake word detector.

These run the real openWakeWord model on synthetic audio (silence and
noise) — no microphone involved. They verify the detector loads, accepts
frames, and does not fire on non-speech. A true-positive test needs a real
person saying "hey jarvis"; that lives in tests/manual_audio_check.py.
"""

from pathlib import Path

import numpy as np
import pytest

import openwakeword
from jarvis import config

MODEL_FILE = Path(openwakeword.__file__).parent / "resources" / "models" / "hey_jarvis_v0.1.onnx"

pytestmark = pytest.mark.skipif(
    not MODEL_FILE.exists(),
    reason="wake word model not downloaded — run ./build.sh first",
)


@pytest.fixture(scope="module")
def detector():
    from jarvis.wake_word import WakeWordDetector
    return WakeWordDetector()


def test_silence_does_not_trigger(detector):
    silence = np.zeros(config.FRAME_SAMPLES, dtype=np.int16)
    for _ in range(50):  # 4 seconds of silence
        assert detector.process_frame(silence) is False


def test_random_noise_does_not_trigger(detector):
    rng = np.random.default_rng(seed=42)
    for _ in range(50):  # 4 seconds of white noise
        noise = rng.integers(-3000, 3000, config.FRAME_SAMPLES, dtype=np.int16)
        assert detector.process_frame(noise) is False
