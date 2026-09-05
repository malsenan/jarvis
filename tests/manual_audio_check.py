"""MANUAL audio hardware checks — run these yourself; they use the mic/speaker.

This file is deliberately NOT named test_*.py, so pytest never collects it
and no automated run will ever open your microphone or make sound.

Each check is a subcommand. Run them from the repo root, in this order:

    1. See every audio device and its index/name (silent, safe):
           .venv/bin/python tests/manual_audio_check.py devices

    2. Speaker check — plays a 1-second tone on the configured output:
           .venv/bin/python tests/manual_audio_check.py tone
       PASS: you hear a clean beep from the speaker you expect.

    3. Microphone check — records 5 seconds, saves it, plays it back:
           .venv/bin/python tests/manual_audio_check.py loopback
       Speak during the countdown. PASS: you hear your own voice played
       back clearly.

    4. TTS check — synthesizes a sentence with Piper and plays it:
           .venv/bin/python tests/manual_audio_check.py tts
       PASS: you hear "Hello, I am Jarvis..." in the Amy voice.

    5. Wake word check — listens live and prints the detection score:
           .venv/bin/python tests/manual_audio_check.py wakeword
       Say "hey jarvis". PASS: the score jumps above the 0.5 threshold
       and "DETECTED" prints. Ctrl+C to stop.

If a check uses the wrong device, set INPUT_DEVICE_NAME / OUTPUT_DEVICE_NAME
in jarvis/config.py (name substrings, e.g. "ATR4697" / "ALC897") and re-run.
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, ".")  # allow running from the repo root without installing
from jarvis import config
from jarvis.audio_devices import find_device

# The loopback check writes the recording here so you can listen to it again
# or inspect it. It is overwritten by each run and is the only file any check
# leaves behind; delete it when you are done.
RECORDING_PATH = Path(tempfile.gettempdir()) / "jarvis_mic_check.wav"

# Match the buffer size Jarvis itself uses, so these checks behave the same way
# the real assistant does. Without it, playback stutters whenever something
# else on the machine is using the CPU (see config.AUDIO_LATENCY_SECONDS).
sd.default.latency = config.AUDIO_LATENCY_SECONDS


def check_devices():
    print(sd.query_devices())
    print(f"\nconfig INPUT_DEVICE_NAME  = {config.INPUT_DEVICE_NAME!r}")
    print(f"config OUTPUT_DEVICE_NAME = {config.OUTPUT_DEVICE_NAME!r}")


def check_tone():
    output = find_device(config.OUTPUT_DEVICE_NAME, "output")
    t = np.linspace(0, 1.0, config.SAMPLE_RATE, endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # 440 Hz, quiet
    print("Playing a 1-second 440 Hz tone...")
    try:
        sd.play(tone, config.SAMPLE_RATE, device=output)
        sd.wait()
    finally:
        sd.stop()  # closes the speaker stream, even on Ctrl+C
    print("Done. Did you hear it on the right speaker?")


def check_loopback():
    input_dev = find_device(config.INPUT_DEVICE_NAME, "input")
    output_dev = find_device(config.OUTPUT_DEVICE_NAME, "output")
    seconds = 5
    print(f"Recording {seconds} seconds — speak now!")
    try:
        recording = sd.rec(
            seconds * config.SAMPLE_RATE,
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="int16",
            device=input_dev,
        )
        for remaining in range(seconds, 0, -1):
            print(f"  {remaining}...")
            time.sleep(1)
        sd.wait()
    finally:
        # Ctrl+C during the countdown must not leave the mic recording.
        sd.stop()

    peak = np.abs(recording).max()
    print(f"Recorded. Peak level: {peak} / 32767 "
          f"({'OK' if peak > 1000 else 'VERY QUIET — is the right mic selected?'})")

    import wave
    with wave.open(str(RECORDING_PATH), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(config.SAMPLE_RATE)
        f.writeframes(recording.tobytes())
    print(f"Saved to {RECORDING_PATH} — playing it back...")

    try:
        sd.play(recording, config.SAMPLE_RATE, device=output_dev)
        sd.wait()
    finally:
        sd.stop()
    print("Done. Did you hear yourself clearly?")


def check_tts():
    from jarvis.text_to_speech import TextToSpeech
    output = find_device(config.OUTPUT_DEVICE_NAME, "output")
    samples, rate = TextToSpeech().synthesize(
        "Hello, I am Jarvis. If you can hear this, text to speech is working."
    )
    print("Playing synthesized speech...")
    try:
        sd.play(samples, rate, device=output)
        sd.wait()
    finally:
        sd.stop()
    print("Done.")


def check_wakeword():
    from jarvis.wake_word import WakeWordDetector
    from jarvis.main import read_frame

    detector = WakeWordDetector()
    input_dev = find_device(config.INPUT_DEVICE_NAME, "input")
    stream = sd.InputStream(
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=config.FRAME_SAMPLES,
        device=input_dev,
    )
    print("Listening — say 'hey jarvis' (Ctrl+C to stop)...")
    with stream:
        while True:
            frame = read_frame(stream)
            # Peek at the raw score so tuning the threshold is easy.
            scores = detector._model.predict(frame)
            score = scores[config.WAKE_WORD_MODEL]
            if score > 0.1:
                print(f"  score: {score:.2f}"
                      + ("   <<< DETECTED" if score > config.WAKE_WORD_THRESHOLD else ""))
            if score > config.WAKE_WORD_THRESHOLD:
                detector._model.reset()


CHECKS = {
    "devices": check_devices,
    "tone": check_tone,
    "loopback": check_loopback,
    "tts": check_tts,
    "wakeword": check_wakeword,
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in CHECKS:
        print(__doc__)
        sys.exit(1)
    try:
        CHECKS[sys.argv[1]]()
    except KeyboardInterrupt:
        print("\nStopped.")
