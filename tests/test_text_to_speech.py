"""Tests for Piper text-to-speech.

Runs real synthesis but NEVER plays anything — the output stays in a numpy
array. Listening to the result is covered by tests/manual_audio_check.py.
"""

import numpy as np
import pytest

from jarvis import config

pytestmark = pytest.mark.skipif(
    not config.PIPER_VOICE_PATH.exists(),
    reason="Piper voice not downloaded — run ./build.sh first",
)


@pytest.fixture(scope="module")
def tts():
    from jarvis.text_to_speech import TextToSpeech
    return TextToSpeech()


def test_synthesize_returns_audio(tts):
    samples, sample_rate = tts.synthesize("Hello, this is a test.")
    assert samples.dtype == np.int16
    assert sample_rate > 0
    # "Hello, this is a test" should be at least half a second of audio
    # and contain actual signal, not silence.
    assert len(samples) > sample_rate / 2
    assert np.abs(samples).max() > 1000


def test_longer_text_gives_longer_audio(tts):
    short, _ = tts.synthesize("Hi.")
    long, _ = tts.synthesize("This is a much longer sentence that should take more time to say.")
    assert len(long) > len(short)
