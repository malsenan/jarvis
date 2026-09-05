"""Records one spoken utterance, deciding when the speaker has finished.

This is a small state machine fed one audio frame at a time:

    WAITING_FOR_SPEECH --(VAD hears speech)--> RECORDING --(trailing silence,
    or the hard time cap)--> done

It is deliberately independent of the microphone: the caller reads frames
from wherever it likes and pushes them in with `add_frame()`. That makes the
whole end-of-speech logic testable with synthetic frames and a fake VAD.

The VAD object just needs two methods (openwakeword.vad.VAD has both):
    predict(frame, frame_size) -> float   # 0..1 speech probability
    reset_states()                        # clear internal LSTM state
"""

from dataclasses import dataclass

import numpy as np

from jarvis import config

# How many 80 ms frames each time window corresponds to.
_FRAMES_PER_SECOND = config.SAMPLE_RATE / config.FRAME_SAMPLES
_PRE_SPEECH_FRAMES = int(config.PRE_SPEECH_BUFFER_SECONDS * _FRAMES_PER_SECOND)
_SILENCE_END_FRAMES = int(config.SILENCE_TO_END_SECONDS * _FRAMES_PER_SECOND)
_MIN_SPEECH_FRAMES = int(config.MIN_UTTERANCE_SECONDS * _FRAMES_PER_SECOND)
_MAX_TOTAL_FRAMES = int(config.MAX_UTTERANCE_SECONDS * _FRAMES_PER_SECOND)
_WAIT_FOR_SPEECH_FRAMES = int(config.WAIT_FOR_SPEECH_SECONDS * _FRAMES_PER_SECOND)


@dataclass
class RecordingResult:
    """What came out of a recording attempt.

    audio is None unless reason is "ok":
      "ok"        — a usable utterance; audio holds int16 samples at 16 kHz.
      "no_speech" — nobody said anything after the wake word.
      "too_short" — a blip (cough, chair squeak) too short to be a sentence.
    """
    reason: str
    audio: np.ndarray | None = None


class SpeechRecorder:
    def __init__(self, vad):
        self._vad = vad
        self.start()

    def start(self) -> None:
        """Reset everything, ready to record a fresh utterance."""
        self._vad.reset_states()
        self._recording = False
        self._frames: list[np.ndarray] = []   # frames of the utterance so far
        self._frames_waited = 0               # frames seen with no speech yet
        self._silence_run = 0                 # consecutive quiet frames while recording
        self._speech_frames = 0               # frames that actually contained speech

    def add_frame(self, frame: np.ndarray) -> RecordingResult | None:
        """Feed one 80 ms int16 frame. Returns None while still listening,
        or a RecordingResult once the utterance is over."""
        # Silero VAD wants <=32 ms chunks internally; frame_size=640 splits
        # our 80 ms frame into two 40 ms chunks, which it handles well.
        is_speech = self._vad.predict(frame, frame_size=640) > config.VAD_SPEECH_THRESHOLD

        if not self._recording:
            # Keep a short rolling buffer so the first syllable — spoken just
            # before the VAD flips to "speech" — is not chopped off.
            self._frames.append(frame)
            if len(self._frames) > _PRE_SPEECH_FRAMES:
                self._frames.pop(0)

            if is_speech:
                self._recording = True
                self._speech_frames = 1
                return None

            self._frames_waited += 1
            if self._frames_waited >= _WAIT_FOR_SPEECH_FRAMES:
                return RecordingResult(reason="no_speech")
            return None

        # --- recording ---
        self._frames.append(frame)
        if is_speech:
            self._speech_frames += 1
            self._silence_run = 0
        else:
            self._silence_run += 1

        utterance_over = (
            self._silence_run >= _SILENCE_END_FRAMES
            or len(self._frames) >= _MAX_TOTAL_FRAMES
        )
        if not utterance_over:
            return None

        if self._speech_frames < _MIN_SPEECH_FRAMES:
            return RecordingResult(reason="too_short")
        return RecordingResult(reason="ok", audio=np.concatenate(self._frames))
