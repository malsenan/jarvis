"""Tests for the end-of-speech state machine in speech_capture.py.

These use a fake VAD whose answers we script, so no microphone, no models
and no audio hardware are involved — we are testing the *logic* of "when is
the user done talking".
"""

import numpy as np
import pytest

from jarvis import config
from jarvis.speech_capture import SpeechRecorder

FRAME = np.zeros(config.FRAME_SAMPLES, dtype=np.int16)
FRAMES_PER_SECOND = config.SAMPLE_RATE / config.FRAME_SAMPLES


class FakeVAD:
    """Returns a scripted sequence of speech probabilities, then silence."""

    def __init__(self, scores):
        self._scores = list(scores)
        self.reset_calls = 0

    def predict(self, frame, frame_size):
        return self._scores.pop(0) if self._scores else 0.0

    def reset_states(self):
        self.reset_calls += 1


def frames(count):
    return [FRAME] * int(count)


def run_recorder(vad, frame_list):
    """Push frames through a fresh recorder until it produces a result."""
    recorder = SpeechRecorder(vad=vad)
    for frame in frame_list:
        result = recorder.add_frame(frame)
        if result is not None:
            return result
    return None


def test_normal_utterance_is_captured():
    # 2 seconds of speech, then enough silence to end the utterance.
    speech_frames = int(2 * FRAMES_PER_SECOND)
    vad = FakeVAD([1.0] * speech_frames)  # silence afterwards
    silence_frames = int(config.SILENCE_TO_END_SECONDS * FRAMES_PER_SECOND)

    result = run_recorder(vad, frames(speech_frames + silence_frames + 5))

    assert result is not None and result.reason == "ok"
    # The captured audio must contain at least the speech itself.
    assert len(result.audio) >= speech_frames * config.FRAME_SAMPLES
    assert result.audio.dtype == np.int16


def test_no_speech_times_out():
    # The VAD never hears anything → recorder should give up.
    vad = FakeVAD([])
    total = int(config.WAIT_FOR_SPEECH_SECONDS * FRAMES_PER_SECOND) + 5

    result = run_recorder(vad, frames(total))

    assert result is not None and result.reason == "no_speech"
    assert result.audio is None


def test_short_blip_is_rejected():
    # One single frame of "speech" (a cough) is below MIN_UTTERANCE_SECONDS.
    vad = FakeVAD([1.0])
    silence_frames = int(config.SILENCE_TO_END_SECONDS * FRAMES_PER_SECOND)

    result = run_recorder(vad, frames(1 + silence_frames + 5))

    assert result is not None and result.reason == "too_short"
    assert result.audio is None


def test_endless_speech_hits_the_hard_cap():
    # The VAD says "speech" forever → the hard cap must end the recording.
    max_frames = int(config.MAX_UTTERANCE_SECONDS * FRAMES_PER_SECOND)
    vad = FakeVAD([1.0] * (max_frames * 2))

    result = run_recorder(vad, frames(max_frames * 2))

    assert result is not None and result.reason == "ok"
    # Capped: not longer than the maximum (plus the pre-speech buffer).
    max_samples = (max_frames + 10) * config.FRAME_SAMPLES
    assert len(result.audio) <= max_samples


def test_start_resets_vad_state():
    vad = FakeVAD([])
    recorder = SpeechRecorder(vad=vad)
    recorder.start()
    # Once in the constructor, once in the explicit start().
    assert vad.reset_calls == 2


def test_pause_mid_sentence_does_not_end_recording():
    # Speech, a pause shorter than SILENCE_TO_END_SECONDS, more speech,
    # then final silence: everything should land in ONE utterance.
    pause = int(config.SILENCE_TO_END_SECONDS * FRAMES_PER_SECOND) - 1
    speech = int(1 * FRAMES_PER_SECOND)
    end_silence = int(config.SILENCE_TO_END_SECONDS * FRAMES_PER_SECOND)
    vad = FakeVAD([1.0] * speech + [0.0] * pause + [1.0] * speech)

    total = speech + pause + speech + end_silence + 5
    result = run_recorder(vad, frames(total))

    assert result is not None and result.reason == "ok"
    assert len(result.audio) >= (speech + pause + speech) * config.FRAME_SAMPLES
