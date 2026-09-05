"""Text-to-speech using Piper.

Synthesis only — playing the audio through a speaker is the caller's job,
which keeps this module free of any audio-hardware dependency (and easy to
test: synthesize and inspect the samples without making a sound).
"""

import numpy as np
from piper import PiperVoice

from jarvis import config


class TextToSpeech:
    def __init__(self):
        self._voice = PiperVoice.load(config.PIPER_VOICE_PATH)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return (samples, sample_rate) for the spoken version of `text`.

        Samples are int16 mono. Piper's sample rate (22050 Hz for this voice)
        differs from our 16 kHz capture rate — the playback call must use the
        rate returned here, not config.SAMPLE_RATE.
        """
        chunks = list(self._voice.synthesize(text))
        audio = np.concatenate([chunk.audio_int16_array for chunk in chunks])
        return audio, chunks[0].sample_rate
