"""All tunable settings for Jarvis in one place.

Everything the pipeline needs to know lives here so that changing behaviour
never requires hunting through the code. Values marked TODO are the ones we
have deliberately left unset (input/output device choice).
"""

from pathlib import Path

# Root of the repository (this file lives in <repo>/jarvis/config.py).
REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Audio devices
#
# Device selection is BY NAME, never by index — indices shift when USB
# devices are plugged in or PipeWire restarts. Give a case-insensitive
# substring of the device name, or None to use whatever the desktop has
# selected as its default input/output. Jarvis prints the device it ended up
# with at startup, so "default" is never a mystery.
#
# Note: a name that matches a raw ALSA "hw:" entry or a JACK entry will fail
# with "Invalid sample rate" — neither can convert to SAMPLE_RATE below.
#
# List the devices on this machine with:
#     .venv/bin/python -m jarvis.audio_devices
# --------------------------------------------------------------------------
INPUT_DEVICE_NAME: str | None = None   # None = default microphone
OUTPUT_DEVICE_NAME: str | None = None  # None = default speaker

# --------------------------------------------------------------------------
# Audio format
#
# 16 kHz mono 16-bit is what openWakeWord, Silero VAD and Whisper all expect,
# so we capture in that format directly (PipeWire resamples for us).
# One frame is 80 ms = 1280 samples — the frame size openWakeWord recommends.
# --------------------------------------------------------------------------
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 1_280

# How much audio the sound card buffers, in seconds.
#
# PortAudio's own "high latency" setting on this machine is only ~35 ms. That
# is not enough headroom: sounddevice refills the buffer from a Python
# callback, so whenever something else loads the CPU (a video playing, a model
# loading) the refill misses its deadline and the speaker plays the empty
# buffer — you hear chopped, stuttering speech. A fifth of a second is
# imperceptible for an assistant and survives a late refill comfortably.
AUDIO_LATENCY_SECONDS = 0.2

# --------------------------------------------------------------------------
# Wake word (openWakeWord)
# --------------------------------------------------------------------------
WAKE_WORD_MODEL = "hey_jarvis"  # bundled openWakeWord model name
WAKE_WORD_THRESHOLD = 0.5       # detection score above this fires the wake word

# --------------------------------------------------------------------------
# Speech capture (Silero VAD)
#
# After the wake word fires we record until the speaker goes quiet.
# --------------------------------------------------------------------------
VAD_SPEECH_THRESHOLD = 0.5        # VAD score above this counts as "speech"
PRE_SPEECH_BUFFER_SECONDS = 0.5   # audio kept from just before speech starts
SILENCE_TO_END_SECONDS = 0.7      # this much trailing silence ends the utterance
MIN_UTTERANCE_SECONDS = 0.3       # shorter than this = a cough, discard it
MAX_UTTERANCE_SECONDS = 15.0      # hard cap so a noisy room can't record forever
WAIT_FOR_SPEECH_SECONDS = 6.0     # give up if nothing is said after the wake word

# --------------------------------------------------------------------------
# Speech-to-text (faster-whisper)
#
# Runs on CPU: faster-whisper's GPU path is CUDA-only and this machine has an
# AMD card. base.en int8 on a desktop CPU transcribes a short utterance well
# under a second, which is fine for this pipeline.
# --------------------------------------------------------------------------
STT_MODEL = "base.en"

# --------------------------------------------------------------------------
# LLM (Ollama)
# --------------------------------------------------------------------------
OLLAMA_MODEL = "qwen3:14b"
OLLAMA_KEEP_ALIVE = "30m"   # keep the model in VRAM between questions
OLLAMA_THINK = False        # disable qwen3's <think> reasoning for snappy replies
OLLAMA_STARTUP_TIMEOUT_SECONDS = 30  # max wait for a server we spawned ourselves

# The model must be FULLY in VRAM. Anything less means CPU offload and
# unusable latency — the script refuses to run in that state.
GPU_MIN_VRAM_FRACTION = 1.0

# Options passed to Ollama on every generate/chat call.
#
# num_ctx: Ollama's default context is 4096 tokens and it truncates the
# oldest input SILENTLY past that — no error, just answers based on whatever
# fraction survived. qwen3:14b natively supports 32,768.
#
# The tradeoff is VRAM: the KV cache costs ~160 KiB per token of context.
# Measured on this 15.9 GiB card (2026-09-05): at 16384 Ollama reports the
# load as 13.5 GiB all-in (weights + KV cache + compute buffers), 100% in
# VRAM with ~2.4 GiB headroom. 32768 would add ~2.5 GiB more and not fit.
# assert_model_on_gpu() catches it if a change here pushes the model off the
# GPU. The preload in load_model() must use these same options, so the GPU
# check validates the context size we actually run with.
OLLAMA_OPTIONS = {"num_ctx": 16384}

# How many question/answer exchanges to keep as conversation memory. Older
# exchanges are dropped (the system prompt never is). Without a cap the
# history grows until it crosses num_ctx, where Ollama silently truncates
# the OLDEST tokens — i.e. the system prompt — and every reply slows down,
# since the whole history is re-processed on each question. 20 exchanges of
# spoken Q&A is roughly 2-4k tokens: plenty of memory, well under num_ctx.
HISTORY_MAX_TURNS = 20

SYSTEM_PROMPT = (
    "You are Jarvis, a voice assistant. Your replies are spoken aloud by a "
    "text-to-speech engine, so answer in plain conversational sentences: "
    "no markdown, no bullet points, no code blocks. Keep answers short — "
    "one to three sentences unless the user asks for detail."
)

# --------------------------------------------------------------------------
# Text-to-speech (Piper)
# --------------------------------------------------------------------------
PIPER_VOICE_PATH = REPO_ROOT / "models" / "en_US-amy-medium.onnx"
