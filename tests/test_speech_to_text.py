"""Round-trip test: Piper speaks a sentence, Whisper transcribes it back.

This proves the whole audio understanding path works without any microphone
or speaker: TTS output (22050 Hz) is resampled to the pipeline's 16 kHz and
fed straight into STT. Marked slow because it loads the Whisper model —
run with:  .venv/bin/pytest -m slow
"""

import numpy as np
import pytest

from jarvis import config

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not config.PIPER_VOICE_PATH.exists(),
        reason="Piper voice not downloaded — run ./build.sh first",
    ),
]


def resample_to_16k(samples: np.ndarray, source_rate: int) -> np.ndarray:
    """Polyphase resampling — the correct method for speech. FFT-based
    resampling (scipy.signal.resample) adds artifacts at chunk edges."""
    from scipy.signal import resample_poly
    from math import gcd
    divisor = gcd(config.SAMPLE_RATE, source_rate)
    up, down = config.SAMPLE_RATE // divisor, source_rate // divisor
    return resample_poly(samples.astype(np.float32), up, down).astype(np.int16)


# Both models are loaded once per module and released when the module's tests
# finish, instead of being rebuilt (and left to the garbage collector) inside
# every test. Whisper alone is ~150 MB of weights.
@pytest.fixture(scope="module")
def stt():
    from jarvis.speech_to_text import SpeechToText
    model = SpeechToText()
    yield model
    del model


@pytest.fixture(scope="module")
def tts():
    from jarvis.text_to_speech import TextToSpeech
    voice = TextToSpeech()
    yield voice
    del voice


def test_tts_output_is_transcribed_back(stt, tts):
    spoken_16k = resample_to_16k(*tts.synthesize(
        "Hello Jarvis, what is the weather like today?"
    ))
    text = stt.transcribe(spoken_16k)

    assert "weather" in text.lower()
    assert "jarvis" in text.lower()


def test_silence_transcribes_to_nothing_or_noise(stt):
    silence = np.zeros(config.SAMPLE_RATE * 2, dtype=np.int16)  # 2 s of silence
    text = stt.transcribe(silence)

    # Whisper sometimes hallucinates a word or two on pure silence; the
    # important thing is it returns a string and does not crash.
    assert isinstance(text, str)
