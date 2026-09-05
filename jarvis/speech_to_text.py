"""Speech-to-text using faster-whisper.

Runs on the CPU on purpose: faster-whisper's GPU backend is CUDA-only and
this machine's RX 9060 XT is AMD. The base.en model with int8 quantization
transcribes a short utterance in well under a second on a desktop CPU, so
the GPU stays free for the LLM (which is the stage that actually needs it).
"""

import numpy as np
from faster_whisper import WhisperModel

from jarvis import config


class SpeechToText:
    def __init__(self):
        # int8 halves memory and speeds up CPU inference with negligible
        # accuracy loss for short commands.
        self._model = WhisperModel(config.STT_MODEL, device="cpu", compute_type="int8")

    def transcribe(self, audio_int16: np.ndarray) -> str:
        """Turn a recorded utterance (int16 @ 16 kHz) into text.

        Returns an empty string if Whisper heard nothing intelligible.
        """
        # Whisper expects float32 samples in the range [-1, 1].
        audio = audio_int16.astype(np.float32) / 32768.0
        segments, _info = self._model.transcribe(audio, language="en")
        # `segments` is a lazy generator; joining it runs the transcription.
        return " ".join(segment.text.strip() for segment in segments).strip()
