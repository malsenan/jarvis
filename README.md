# Jarvis — a local voice assistant

A continuously running voice assistant for a single Linux PC. Everything runs
locally; nothing leaves the machine.

```
mic ─→ wake word ─→ record until silence ─→ speech-to-text ─→ Ollama (GPU) ─→ text-to-speech ─→ speaker
      openWakeWord   Silero VAD              faster-whisper    qwen3:14b       Piper
      "hey jarvis"   (state machine)         base.en, CPU      100% VRAM       en_US-amy-medium
```

## Build

```bash
./build.sh
```

Creates `.venv/`, installs `requirements.txt`, and downloads all models
(wake word + VAD into the openwakeword package, the Piper voice into
`models/`, Whisper into `~/.cache/huggingface`). Safe to re-run; a wiped
`.venv` is fully restored by re-running it.

## Run

```bash
.venv/bin/python -m jarvis.main       # Ctrl+C to stop
```

On startup it connects to Ollama (spawning `ollama serve` itself if none is
running), loads the model, and **refuses to run unless the model is 100% in
VRAM**. Ollama falls back to CPU silently when it cannot reach the GPU, and a
14B model on CPU takes 20–30 seconds per reply — slow enough to feel broken
with no visible error, so we fail loudly instead.

## Configuration

Everything tunable is in `jarvis/config.py`. The two device settings:

- `INPUT_DEVICE_NAME` — the microphone
- `OUTPUT_DEVICE_NAME` — the speaker

Both default to `None`, meaning whatever the desktop has selected. Jarvis
prints the device it opened at startup, resolving `default` through PipeWire
so you see the actual microphone and speaker names:

```
Microphone: default -> alsa_input.usb-Conference_USB_microphone_ATR4697-USB-00.mono-fallback
Speaker:    default -> bluez_output.XX_XX_XX_XX_XX_XX.1
```

Devices are matched **by name substring, never by index**, because indices
shift when USB devices are plugged in or PipeWire restarts.

### Choosing the device names

```bash
.venv/bin/python -m jarvis.audio_devices
```

Each line looks like:

```
   0 ATR4697-USB: USB Audio (hw:0,0), ALSA (1 in, 0 out)
   5 HD-Audio Generic: ALC897 Analog (hw:2,0), ALSA (2 in, 2 out)
```

- **`(N in, M out)`** tells you the direction. `N > 0` means it can record
  (usable as `INPUT_DEVICE_NAME`); `M > 0` means it can play
  (usable as `OUTPUT_DEVICE_NAME`). A device with both can do either.
- **The name** is everything between the index and the final `, <host API>`
  — for line 0 that is `ATR4697-USB: USB Audio (hw:0,0)`. You only need a
  unique substring of it, e.g. `"ATR4697"`.
- The trailing `ALSA` / `JACK Audio Connection Kit` is the host API, not part
  of the name. The same physical device usually appears under several host
  APIs; if your substring matches more than one, `find_device` prints a note
  and uses the first match.

If opening a raw `hw:X,Y` device fails with a sample-rate or "device busy"
error, use `"pipewire"` as the device name instead — PipeWire mixes and
resamples, and routes to whatever sink/source is selected in the desktop
sound settings.

**INPUT/OUTPUT CONFIGURED TO NONE WILL CHOOSE CURRENTLY SELECTED DESKTOP OUTPUT.**


### If the audio stutters

`config.AUDIO_LATENCY_SECONDS` (default `0.2`) sets how much audio the sound
card buffers. PortAudio's own default here is ~35 ms, which is too thin —
playback is refilled from a Python callback, so anything else loading the CPU
(a video playing, a model loading) makes it miss the deadline and you hear
chopped speech. Raise it to `0.3`–`0.4` if stuttering persists; the only cost
is a slightly later start to each reply.

## Tests

```bash
.venv/bin/pytest            # fast tests (logic, mocks, small models) — no audio hardware
.venv/bin/pytest -m slow    # + Piper→Whisper round-trip (loads Whisper, ~5 s)
```

No automated test ever opens the microphone or plays sound. Hardware checks are
manual, with pass criteria documented in the file header:

```bash
.venv/bin/python tests/manual_audio_check.py devices    # list devices (silent)
.venv/bin/python tests/manual_audio_check.py tone       # speaker beep
.venv/bin/python tests/manual_audio_check.py loopback   # record 5 s, play back
.venv/bin/python tests/manual_audio_check.py tts        # hear Piper speak
.venv/bin/python tests/manual_audio_check.py wakeword   # live "hey jarvis" scores
```

## Debugging in VS Code

`.vscode/launch.json` defines the run/debug configurations (F5, or the Run and
Debug panel). Each one points at `.venv/bin/python` explicitly, so it works
regardless of which interpreter VS Code has selected — just run `./build.sh`
first.

| Configuration | What it does |
|---|---|
| **Jarvis: run assistant** | runs `jarvis.main` — uses the mic and speaker |
| **Tests: fast** | the default test run (excludes `slow`), silent |
| **Tests: slow** | only the Piper→Whisper round-trip |
| **Tests: current file** | runs the test file open in the editor |
| **Tests: by name (-k)** | prompts for a `-k` expression, e.g. `gpu` |

All test configurations set `justMyCode: false`, so you can step into
pytest and library code as well as `jarvis/`.

## Files

| Path | What it is |
|---|---|
| `jarvis/config.py` | every tunable setting, documented |
| `jarvis/main.py` | the loop; the **only** module touching mic/speaker |
| `jarvis/audio_devices.py` | name→index device lookup |
| `jarvis/ollama_llm.py` | server startup, loud GPU check, chat with history |
| `jarvis/wake_word.py` | openWakeWord wrapper (hey_jarvis only, ONNX) |
| `jarvis/speech_capture.py` | VAD end-of-speech state machine (pure logic) |
| `jarvis/speech_to_text.py` | faster-whisper, CPU int8 |
| `jarvis/text_to_speech.py` | Piper synthesis (no playback) |
| `tests/` | 27 automated tests + `manual_audio_check.py` |
| `build.sh`, `requirements.txt` | reproducible environment |
| `.vscode/launch.json` | VS Code run/debug configurations |
| `models/` | Piper voice (downloaded, not committed) |

## Environment constraints (read before "fixing" anything)

- **Python 3.14.** `tflite-runtime` has no 3.14 wheels, which is why
  `build.sh` installs openwakeword with `--no-deps` and we use its ONNX
  backend. Don't move openwakeword into `requirements.txt` — pip would
  resolve it down to 0.4.0 to satisfy the tflite dependency.
- **Never run Ollama as a systemd *system* service.** The SELinux `init_t`
  domain can't read models out of the user's home directory or open
  `/dev/kfd` → empty model list and silent CPU inference. Run `ollama serve`
  in a user shell, let `jarvis.main` spawn it, or use a systemd **user**
  service.
- **Whisper runs on CPU on purpose** — faster-whisper's GPU path is CUDA-only
  and this machine has an AMD card. `base.en` int8 transcribes a short
  utterance in well under a second, and the GPU stays free for the LLM.

## Not yet decided / next steps

- Input/output device names in `config.py`
- Conversation history is unbounded within a run; fine for now, trim later
- Barge-in (interrupting Jarvis mid-reply) — playback is blocking today
- MCP server / agent tools — out of scope for this skeleton
- Autostart (systemd **user** unit) — deferred deliberately
