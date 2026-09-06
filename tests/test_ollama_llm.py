"""Tests for the Ollama wrapper, especially the loud GPU check.

A fake client stands in for the real Ollama server, so these run without
Ollama installed or any model loaded.
"""

import subprocess
from types import SimpleNamespace

import pytest

from jarvis import config
from jarvis.ollama_llm import GpuNotUsedError, OllamaLLM


def make_ps_entry(size, size_vram, name=None):
    name = name or config.OLLAMA_MODEL
    return SimpleNamespace(name=name, model=name, size=size, size_vram=size_vram)


class FakeClient:
    """Mimics the bits of ollama.Client that OllamaLLM uses."""

    def __init__(self, ps_models=(), chat_reply="hello there"):
        self._ps_models = list(ps_models)
        self._chat_reply = chat_reply
        self.chat_calls = []
        self.chat_kwargs = []
        self.generate_kwargs = []

    def list(self):
        return SimpleNamespace(models=[])

    def generate(self, **kwargs):
        self.generate_kwargs.append(kwargs)
        return SimpleNamespace(done=True)

    def ps(self):
        return SimpleNamespace(models=self._ps_models)

    def chat(self, model, messages, **kwargs):
        self.chat_calls.append(list(messages))
        self.chat_kwargs.append(kwargs)
        return SimpleNamespace(
            message=SimpleNamespace(content=self._chat_reply)
        )


# ---------------------------------------------------------------- GPU check

def test_gpu_check_passes_when_fully_in_vram():
    llm = OllamaLLM(client=FakeClient(ps_models=[make_ps_entry(100, 100)]))
    llm.assert_model_on_gpu()  # must not raise


def test_gpu_check_fails_when_model_on_cpu():
    llm = OllamaLLM(client=FakeClient(ps_models=[make_ps_entry(100, 0)]))
    with pytest.raises(GpuNotUsedError) as error:
        llm.assert_model_on_gpu()
    assert "GPU CHECK FAILED" in str(error.value)


def test_gpu_check_fails_on_partial_offload():
    # 90% in VRAM still means some layers run on CPU → refuse.
    llm = OllamaLLM(client=FakeClient(ps_models=[make_ps_entry(100, 90)]))
    with pytest.raises(GpuNotUsedError):
        llm.assert_model_on_gpu()


def test_gpu_check_fails_when_model_not_loaded():
    llm = OllamaLLM(client=FakeClient(ps_models=[]))
    with pytest.raises(GpuNotUsedError) as error:
        llm.assert_model_on_gpu()
    assert "not loaded" in str(error.value)


def test_gpu_check_ignores_other_models():
    other = make_ps_entry(100, 100, name="some-other-model:7b")
    llm = OllamaLLM(client=FakeClient(ps_models=[other]))
    with pytest.raises(GpuNotUsedError):
        llm.assert_model_on_gpu()


# ---------------------------------------------------------------- chat

def test_ask_returns_reply_and_keeps_history():
    client = FakeClient(chat_reply="It is sunny.")
    llm = OllamaLLM(client=client)

    reply = llm.ask("what's the weather?")

    assert reply == "It is sunny."
    sent = client.chat_calls[0]
    # System prompt first, then the user's question.
    assert sent[0]["role"] == "system"
    assert sent[-1] == {"role": "user", "content": "what's the weather?"}

    # A follow-up question must include the previous exchange.
    llm.ask("and tomorrow?")
    sent = client.chat_calls[1]
    roles = [m["role"] for m in sent]
    assert roles == ["system", "user", "assistant", "user"]


def test_history_is_bounded_and_system_prompt_survives():
    # Run three times as many exchanges as the cap. Without trimming this
    # would grow forever and eventually push the system prompt out of the
    # context window (Ollama truncates the oldest tokens silently).
    client = FakeClient()
    llm = OllamaLLM(client=client)
    total = config.HISTORY_MAX_TURNS * 3

    for i in range(total):
        llm.ask(f"question {i}")

    # Bounded: system prompt + at most HISTORY_MAX_TURNS exchanges.
    assert len(llm._messages) == 1 + 2 * config.HISTORY_MAX_TURNS
    # The system prompt is never evicted, and stays first.
    assert llm._messages[0] == {"role": "system", "content": config.SYSTEM_PROMPT}
    # Newest exchange kept, oldest dropped.
    contents = [m["content"] for m in llm._messages]
    assert f"question {total - 1}" in contents
    assert "question 0" not in contents
    # Trimming must cut at an exchange boundary: user right after system.
    assert llm._messages[1]["role"] == "user"


def test_request_sent_to_ollama_is_bounded_too():
    # The list SENT to the client is what counts against num_ctx: at most the
    # system prompt, the capped history, and the one new question.
    client = FakeClient()
    llm = OllamaLLM(client=client)

    for i in range(config.HISTORY_MAX_TURNS * 3):
        llm.ask(f"question {i}")

    longest_request = max(len(sent) for sent in client.chat_calls)
    assert longest_request <= 2 + 2 * config.HISTORY_MAX_TURNS


def test_num_ctx_is_sent_to_the_client_on_load_and_chat():
    # Ollama silently truncates at its 4096 default unless num_ctx is passed
    # on EVERY call — including the preload, or the GPU check would validate
    # a smaller KV cache than real questions use.
    client = FakeClient()
    llm = OllamaLLM(client=client)

    llm.load_model()
    llm.ask("hello")

    assert client.generate_kwargs[0]["options"]["num_ctx"] == config.OLLAMA_OPTIONS["num_ctx"]
    assert client.chat_kwargs[0]["options"]["num_ctx"] == config.OLLAMA_OPTIONS["num_ctx"]


# ------------------------------------------------------------- shutdown

class FakeProcess:
    """Stands in for the `ollama serve` subprocess.

    `terminate_ignored=True` makes it play dead: wait() raises TimeoutExpired
    the first time, so we can check that shutdown escalates to kill().
    """

    def __init__(self, terminate_ignored=False):
        self.terminate_ignored = terminate_ignored
        self.terminated = False
        self.killed = False

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        if self.terminate_ignored and not self.killed:
            raise subprocess.TimeoutExpired(cmd="ollama serve", timeout=timeout)
        return 0


def test_shutdown_stops_a_server_we_started():
    llm = OllamaLLM(client=FakeClient())
    process = FakeProcess()
    llm._server_process = process

    llm.shutdown()

    assert process.terminated
    assert not process.killed


def test_shutdown_kills_a_server_that_ignores_terminate():
    llm = OllamaLLM(client=FakeClient())
    process = FakeProcess(terminate_ignored=True)
    llm._server_process = process

    llm.shutdown()

    assert process.terminated
    assert process.killed  # escalated rather than leaving it running


def test_shutdown_leaves_a_pre_existing_server_alone():
    """We only stop the server if we were the ones who started it."""
    llm = OllamaLLM(client=FakeClient())
    assert llm._server_process is None

    llm.shutdown()   # must not raise
    llm.shutdown()   # and must be safe to call twice


def test_context_manager_shuts_down_even_when_the_body_raises():
    llm = OllamaLLM(client=FakeClient())
    process = FakeProcess()

    with pytest.raises(GpuNotUsedError):
        with llm:
            llm._server_process = process
            llm.assert_model_on_gpu()  # nothing loaded -> raises

    assert process.terminated
