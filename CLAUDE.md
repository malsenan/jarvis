# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Jarvis is a local voice assistant that runs entirely on one Linux PC. It is a
single continuously running Python process:

```
mic → wake word → record until silence → speech-to-text → Ollama (GPU) → text-to-speech → speaker
     openWakeWord  Silero VAD            faster-whisper   qwen3:14b      Piper
```

Nothing leaves the machine. There is no server, no client, no network protocol.

## Commands

```bash
./build.sh                                   # create .venv + download all models (idempotent)
.venv/bin/python -m jarvis.main              # run Jarvis (Ctrl+C to stop)
.venv/bin/python -m jarvis.audio_devices     # list audio devices (silent)

.venv/bin/pytest                             # fast tests (default: -m "not slow")
.venv/bin/pytest -m slow                     # Piper→Whisper round-trip (~5 s)
.venv/bin/pytest tests/test_ollama_llm.py::test_gpu_check_passes_when_fully_in_vram # a single test
```

There is no linter or formatter configured.

## Environment

- Fedora 43, **Python 3.14 only** (no other interpreter available)
- AMD Radeon RX 9060 XT (ROCm). **No CUDA** — anything CUDA-only must run on CPU
- Ollama is the Fedora RPM; it reports version `0.0.0`, which is a packaging
  artifact, not a broken install
- Audio goes through PipeWire (exposed to PortAudio as ALSA + JACK devices)

Three constraints that look like bugs but are deliberate — do not "fix" them:

1. **openwakeword is installed with `--no-deps` in `build.sh`, not via
   `requirements.txt`.** Its Linux metadata requires `tflite-runtime`, which has
   no Python 3.14 wheels; pip resolves it down to 0.4.0 to satisfy that. We use
   the ONNX backend, and `requirements.txt` carries its real runtime deps.
2. **Ollama must never run as a systemd *system* service.** The SELinux `init_t`
   domain cannot read models from the user's home or open `/dev/kfd`, giving an
   empty model list and silent CPU inference. Run `ollama serve` in a user
   shell, let `jarvis.main` spawn it, or use a systemd **user** unit.
3. **Whisper runs on CPU on purpose** (faster-whisper's GPU path is CUDA-only).
   `base.en` int8 is well under a second per utterance; the GPU stays for the LLM.

## Architecture

Each pipeline stage is one small module under `jarvis/`, and every setting lives
in `jarvis/config.py`. Two invariants hold the design together:

- **`jarvis/main.py` is the only module that touches the microphone or speaker.**
  `text_to_speech.py` synthesizes samples but never plays them;
  `speech_capture.py` is a pure state machine fed frames from outside. This is
  what makes everything else testable without hardware.
- **`ollama_llm.assert_model_on_gpu()` runs before any audio is opened.** It
  compares `size_vram / size` from Ollama's `/api/ps` and raises
  `GpuNotUsedError` on anything below 1.0. Ollama falls back to CPU *silently*,
  and a 14B model on CPU takes 20–30 s per reply, so partial offload is treated
  as a hard failure rather than a slow success.

Audio devices are matched **by name substring, never by index** — indices shift
when USB devices are plugged in or PipeWire restarts.

**Anything initialized must be released, always.** Audio streams, the spawned
`ollama serve` subprocess, files, and models in tests all get a defined
lifetime: a `with` block, a `try/finally`, or a pytest fixture that yields.
`OllamaLLM` is a context manager for exactly this reason — a leaked
`ollama serve` holds the GPU after the script exits.

## Testing

**Never run code that opens the microphone or plays sound.** The user may be
doing something else; unannounced audio is not acceptable.

Automated tests are hardware-free: fakes for the VAD, the Ollama client, and the
device table, plus real (small) ONNX models for wake word and TTS synthesis.
Hardware verification lives in `tests/manual_audio_check.py`, deliberately named
so pytest cannot collect it — the user runs it, never you.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a **junior** engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

Code should read cleanly for a junior engineer: clear names, intuitive flow, a
short explanation per method and per meaningful chunk. Abstract only when needed.
