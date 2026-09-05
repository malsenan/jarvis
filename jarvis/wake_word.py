"""Wake word detection using openWakeWord's bundled "hey jarvis" model.

Only the one wake word model is loaded. Constructing openWakeWord's Model
without an explicit list loads all six bundled wake words, which wastes CPU
on every frame and lets "alexa" or "hey mycroft" trigger Jarvis. We use the
ONNX backend because tflite-runtime has no Python 3.14 wheels.
"""

import numpy as np
from openwakeword.model import Model

from jarvis import config


class WakeWordDetector:
    def __init__(self):
        self._model = Model(
            wakeword_models=[config.WAKE_WORD_MODEL],
            inference_framework="onnx",
        )

    def process_frame(self, frame: np.ndarray) -> bool:
        """Feed one 80 ms frame (int16, 16 kHz); return True on detection.

        openWakeWord keeps a rolling window internally, so it must see every
        frame in order — call this continuously while waiting.
        """
        scores = self._model.predict(frame)
        # bool() because the score is a numpy float and comparing it yields a
        # numpy bool, which breaks `is True` / `is False` checks for callers.
        detected = bool(scores[config.WAKE_WORD_MODEL] > config.WAKE_WORD_THRESHOLD)
        if detected:
            # Clear the rolling window so the same utterance of "hey jarvis"
            # can't fire the detector a second time.
            self._model.reset()
        return detected
