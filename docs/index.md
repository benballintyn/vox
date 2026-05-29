# vox

**Model-agnostic LLM execution library for Python.** One interface, every provider.

Write your code once and run it against OpenAI, Anthropic, Google Gemini, OpenRouter, or local models via LM Studio — with streaming, tool use, structured output, reasoning, vision, video, and audio support out of the box.

[![PyPI](https://img.shields.io/pypi/v/vox-llm.svg)](https://pypi.org/p/vox-llm)
[![Python](https://img.shields.io/pypi/pyversions/vox-llm.svg)](https://pypi.org/p/vox-llm)
[![License](https://img.shields.io/pypi/l/vox-llm.svg)](https://github.com/benballintyn/vox/blob/main/LICENSE)
[![Tests](https://github.com/benballintyn/vox/actions/workflows/run_tests.yml/badge.svg)](https://github.com/benballintyn/vox/actions/workflows/run_tests.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/types-mypy-blue.svg)](https://mypy-lang.org/)

## Install

```bash
pip install "vox-llm[all]"
```

For a leaner install, pick just the providers you need: `vox-llm[openai]`, `vox-llm[anthropic]`, `vox-llm[gemini]`. The PyPI name is `vox-llm` (the bare `vox` was taken); the Python import is still `from vox import ...`.

## Five lines

```python
from vox import VoxClient, Message

client = VoxClient(openai_api_key="sk-...")
response = client.complete(
    [Message(role="user", content="Hello")],
    model="gpt-5",
)
print(response.message.text)
```

Swap providers by changing the model name — no other code changes:

```python
client.complete(messages, model="claude-sonnet-4-5-20250929")
client.complete(messages, model="gemini-3.1-flash-lite")
```

## What's in the box

<div class="grid cards" markdown>

-   :material-message-text:{ .lg .middle } **Chat completions**

    ---

    Sync + async, streaming + non-streaming. Normalized `Message`, `Tool`, `ToolCallData` types across every provider.

    [:octicons-arrow-right-24: Getting Started](getting-started.md)

-   :material-tools:{ .lg .middle } **Tool use**

    ---

    Define once, run against any provider. Vox translates tool definitions and results to each provider's native format.

    [:octicons-arrow-right-24: Tools guide](guides/tools.md)

-   :material-shape:{ .lg .middle } **Structured output**

    ---

    Pass a Pydantic model, get a validated typed instance back. Works on OpenAI, Anthropic, Gemini, OpenRouter.

    [:octicons-arrow-right-24: Structured output guide](guides/structured-output.md)

-   :material-brain:{ .lg .middle } **Reasoning**

    ---

    Normalized `ReasoningConfig(level="low"|"medium"|"high")` across OpenAI o-series, Anthropic extended thinking, Gemini 2.5+/3+.

    [:octicons-arrow-right-24: Reasoning reference](reference/reasoning.md)

-   :material-image:{ .lg .middle } **Multimodal**

    ---

    Vision (`ImageContent`) on every provider. Video (`VideoContent`) native on Gemini, client-side frame-extraction fallback elsewhere.

    [:octicons-arrow-right-24: Multimodal guide](guides/multimodal.md)

-   :material-microphone:{ .lg .middle } **Audio I/O**

    ---

    Dedicated `transcribe()` + `synthesize()` methods. OpenAI Whisper + tts-1, Gemini generate_content + TTS.

    [:octicons-arrow-right-24: Audio guide](guides/audio.md)

-   :material-restart:{ .lg .middle } **Retries**

    ---

    Configurable per-call. Exponential backoff with jitter, honours `RateLimitError.retry_after`, streaming-aware.

    [:octicons-arrow-right-24: Reliability guide](guides/reliability.md)

-   :material-chart-line:{ .lg .middle } **Observability**

    ---

    `CallbackHandler` protocol with OpenTelemetry GenAI semantic conventions out of the box. Drop in your own telemetry backend.

    [:octicons-arrow-right-24: Reliability guide](guides/reliability.md)

</div>

## Why vox

- **Typed all the way down.** Pydantic models for every input and output. `mypy --strict` clean. Ships `py.typed`.
- **Lean install.** Core is `pydantic` + `loguru` + `httpx`. Provider SDKs are optional extras.
- **No proxy ambitions.** vox is a library; layer your own infrastructure on top if you need it.
- **Live-tested.** 111 integration tests across the four CI-runnable providers, dispatched before every release.
