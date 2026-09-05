#!/usr/bin/env bash
# Builds the Jarvis virtual environment from scratch. Safe to re-run.
#
#   ./build.sh
#
# Afterwards:
#   run Jarvis:        .venv/bin/python -m jarvis.main
#   run the tests:     .venv/bin/pytest          (fast tests)
#                      .venv/bin/pytest -m slow  (slow model-based tests)
#   list audio devices .venv/bin/python -m jarvis.audio_devices
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Creating virtual environment (.venv)"
python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip

echo "==> Installing Python dependencies"
.venv/bin/pip install --quiet -r requirements.txt

# openwakeword is installed separately with --no-deps: its Linux metadata
# demands tflite-runtime, which has no Python 3.14 wheels. We use its ONNX
# backend instead, and requirements.txt carries everything it really needs.
.venv/bin/pip install --quiet --no-deps openwakeword==0.6.0

echo "==> Downloading wake word + VAD models (into the openwakeword package)"
.venv/bin/python -c "import openwakeword.utils; openwakeword.utils.download_models(['hey_jarvis'])"

echo "==> Downloading Piper TTS voice (into models/)"
mkdir -p models
.venv/bin/python -m piper.download_voices --data-dir models en_US-amy-medium

echo "==> Pre-downloading the Whisper STT model (into ~/.cache/huggingface)"
.venv/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8')"

echo
echo "Build complete. Start Jarvis with:  .venv/bin/python -m jarvis.main"
