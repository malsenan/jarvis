"""Talks to the Ollama server: startup, GPU verification, and chat.

The GPU check exists because Ollama falls back to CPU *silently* when it
cannot reach the GPU — no error, just slow answers. A 14B model on CPU takes
20-30 seconds per short reply, which makes the assistant feel broken with no
obvious cause. So we check explicitly and refuse to run.
"""

import subprocess
import time

import ollama

from jarvis import config


class GpuNotUsedError(RuntimeError):
    """Raised when the Ollama model is not fully loaded into GPU memory."""


class OllamaLLM:
    def __init__(self, client: ollama.Client | None = None):
        # `client` is injectable so tests can pass a fake; normally we build
        # the real one, which talks to http://127.0.0.1:11434.
        self._client = client if client is not None else ollama.Client()
        self._server_process: subprocess.Popen | None = None
        # The running conversation. Every question and answer is appended so
        # the model has context for follow-up questions.
        self._messages: list[dict] = [
            {"role": "system", "content": config.SYSTEM_PROMPT}
        ]

    def ensure_server_running(self) -> None:
        """Connect to the Ollama server, spawning `ollama serve` if needed.

        Important: Ollama must run as a normal user process here, NOT as a
        systemd *system* service. A system unit puts Ollama in the SELinux
        init_t domain, where it can neither read models out of the user's
        home directory nor open /dev/kfd (the GPU) — the result is an empty
        model list and silent CPU inference. Spawning it from this script
        keeps it in the user's unconfined domain, which works.
        """
        if self._server_is_up():
            print("Ollama server already running.")
            return

        print("Ollama server not running — starting `ollama serve`...")
        self._server_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        deadline = time.monotonic() + config.OLLAMA_STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._server_is_up():
                print("Ollama server is up.")
                return
            time.sleep(0.5)

        raise RuntimeError(
            f"`ollama serve` did not become reachable within "
            f"{config.OLLAMA_STARTUP_TIMEOUT_SECONDS}s. "
            f"Try running `ollama serve` in a terminal to see its error output."
        )

    def _server_is_up(self) -> bool:
        try:
            self._client.list()
            return True
        except Exception:
            return False

    def load_model(self) -> None:
        """Load the model into memory now, so the first real question is not
        stuck behind a ~10 second model load. An empty prompt tells Ollama
        "just load it".

        Passing the same options as ask() matters: num_ctx sets the KV cache
        size at load time, so the GPU check that follows must see the model
        loaded with the context size real questions will use."""
        print(f"Loading {config.OLLAMA_MODEL} (this can take a few seconds)...")
        self._client.generate(
            model=config.OLLAMA_MODEL,
            prompt="",
            options=config.OLLAMA_OPTIONS,
            keep_alive=config.OLLAMA_KEEP_ALIVE,
        )

    def assert_model_on_gpu(self) -> None:
        """Fail loudly unless the model is fully resident in GPU memory.

        Ollama's /api/ps reports, per loaded model, its total size and how
        much of it sits in VRAM. size_vram == size means 100% GPU; anything
        less means part of the model is being run on the CPU.
        """
        loaded = self._client.ps().models
        entry = next(
            (m for m in loaded if config.OLLAMA_MODEL in (m.name, m.model)),
            None,
        )
        if entry is None:
            raise GpuNotUsedError(
                self._banner(f"Model '{config.OLLAMA_MODEL}' is not loaded at all. "
                             f"Loaded models: {[m.name for m in loaded]}")
            )

        vram_fraction = entry.size_vram / entry.size if entry.size else 0.0
        if vram_fraction < config.GPU_MIN_VRAM_FRACTION:
            raise GpuNotUsedError(
                self._banner(
                    f"Model '{config.OLLAMA_MODEL}' is NOT fully on the GPU: "
                    f"{entry.size_vram:,} of {entry.size:,} bytes in VRAM "
                    f"({vram_fraction:.0%}). CPU inference is unusably slow.\n"
                    f"Check `journalctl` / the `ollama serve` terminal for GPU "
                    f"errors (look for 'library=cpu' or '/dev/kfd'), and make "
                    f"sure Ollama was NOT started as a systemd *system* "
                    f"service — SELinux blocks GPU access there."
                )
            )
        print(f"GPU check passed: {config.OLLAMA_MODEL} is {vram_fraction:.0%} in VRAM.")

    @staticmethod
    def _banner(message: str) -> str:
        line = "!" * 72
        return f"\n{line}\n!!! GPU CHECK FAILED\n{line}\n{message}\n{line}"

    def ask(self, user_text: str) -> str:
        """Send the user's words to the model and return its reply text."""
        self._messages.append({"role": "user", "content": user_text})
        response = self._client.chat(
            model=config.OLLAMA_MODEL,
            messages=self._messages,
            think=config.OLLAMA_THINK,
            options=config.OLLAMA_OPTIONS,
            keep_alive=config.OLLAMA_KEEP_ALIVE,
        )
        reply = response.message.content.strip()
        self._messages.append({"role": "assistant", "content": reply})
        return reply

    def shutdown(self) -> None:
        """Stop the `ollama serve` process if we were the ones who started it.

        A server that was already running before us is left alone — it is not
        ours to stop. Safe to call more than once.
        """
        if self._server_process is None:
            return
        print("Stopping the Ollama server we started...")
        self._server_process.terminate()
        try:
            self._server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # It ignored SIGTERM. Don't leave it running behind us.
            self._server_process.kill()
            self._server_process.wait()
        self._server_process = None

    # Context manager support, so callers can write `with OllamaLLM() as llm:`
    # and be certain the server is stopped even if something raises.
    def __enter__(self) -> "OllamaLLM":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.shutdown()
