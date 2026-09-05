"""Jarvis — the continuously running voice assistant loop.

Pipeline, per interaction:

    mic --> wake word --> record until silence --> speech-to-text
        --> Ollama (GPU-verified) --> text-to-speech --> speaker

Run it with:

    .venv/bin/python -m jarvis.main

Stop it with Ctrl+C. This is the only module that touches the microphone
and speaker; every stage of the pipeline lives in its own module.
"""

import sys

import numpy as np
import sounddevice as sd
from openwakeword.vad import VAD

from jarvis import config
from jarvis.audio_devices import find_device
from jarvis.ollama_llm import OllamaLLM, GpuNotUsedError
from jarvis.speech_capture import SpeechRecorder
from jarvis.speech_to_text import SpeechToText
from jarvis.text_to_speech import TextToSpeech
from jarvis.wake_word import WakeWordDetector


def read_frame(stream: sd.InputStream) -> np.ndarray:
    """Read one 80 ms frame from the microphone as a 1-D int16 array."""
    frame, overflowed = stream.read(config.FRAME_SAMPLES)
    if overflowed:
        # We fell behind and the audio driver dropped samples. Harmless
        # occasionally; if it prints constantly, something is too slow.
        print("Warning: audio input overflow (dropped samples).")
    return frame[:, 0]  # sounddevice returns shape (N, channels); we want 1-D


def main() -> None:
    # --- LLM first: if the model can't run on the GPU there is no point
    # opening the microphone at all. assert_model_on_gpu() raises a loud
    # GpuNotUsedError if any part of the model is on the CPU.
    #
    # `with` guarantees that an `ollama serve` process we spawned is stopped
    # again on the way out — including when the GPU check below raises. ---
    with OllamaLLM() as llm:
        llm.ensure_server_running()
        llm.load_model()
        llm.assert_model_on_gpu()

        # --- The rest of the pipeline. Each constructor loads its model into
        # memory, so the first interaction has no loading pauses. ---
        print("Loading speech-to-text model...")
        stt = SpeechToText()
        print("Loading text-to-speech voice...")
        tts = TextToSpeech()
        print("Loading wake word model...")
        wake_word = WakeWordDetector()
        recorder = SpeechRecorder(vad=VAD())

        # Devices are chosen by name in config.py; None means system default.
        input_device = find_device(config.INPUT_DEVICE_NAME, "input")
        output_device = find_device(config.OUTPUT_DEVICE_NAME, "output")

        # Give every stream we open a generous buffer. See the comment on
        # AUDIO_LATENCY_SECONDS — the default is small enough that a busy CPU
        # makes playback stutter.
        sd.default.latency = config.AUDIO_LATENCY_SECONDS

        microphone = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=config.FRAME_SAMPLES,
            device=input_device,
        )

        try:
            # `with` closes the microphone stream on the way out; the outer
            # try/finally closes the speaker stream that sd.play() opens.
            with microphone:
                run_conversation_loop(
                    microphone, wake_word, recorder, stt, llm, tts, output_device
                )
        finally:
            sd.stop()  # stops and closes the playback stream sd.play() uses


def run_conversation_loop(microphone, wake_word, recorder, stt, llm, tts, output_device) -> None:
    """Listen, answer, repeat — until Ctrl+C."""
    print("\nJarvis is ready — say 'hey jarvis'. (Ctrl+C to quit)\n")
    while True:
        # ---- 1. Wait for the wake word, one frame at a time. ----
        if not wake_word.process_frame(read_frame(microphone)):
            continue
        print("Wake word detected — listening...")

        # ---- 2. Record the question until the speaker goes quiet. ----
        recorder.start()
        result = None
        while result is None:
            result = recorder.add_frame(read_frame(microphone))
        if result.reason != "ok":
            print(f"Nothing usable recorded ({result.reason}); going back to sleep.")
            continue

        # Pause the mic while thinking and speaking, so Jarvis does not
        # transcribe its own voice as the next question.
        microphone.stop()
        try:
            # ---- 3. Speech to text. ----
            question = stt.transcribe(result.audio)
            print(f"You said: {question!r}")
            if not question:
                continue

            # ---- 4./5. Ask the model and wait for its reply. ----
            reply = llm.ask(question)
            print(f"Jarvis: {reply}")
            if not reply:
                continue

            # ---- 6./7. Speak the reply. ----
            samples, sample_rate = tts.synthesize(reply)
            sd.play(samples, sample_rate, device=output_device)
            sd.wait()  # block until playback finishes
        finally:
            microphone.start()


if __name__ == "__main__":
    try:
        main()
    except GpuNotUsedError as error:
        print(error)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nGoodbye.")
