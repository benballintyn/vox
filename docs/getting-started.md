# Getting Started

## Install

```bash
# With every provider SDK
pip install "vox-llm[all]"

# Or pick what you need
pip install "vox-llm[openai]"
pip install "vox-llm[anthropic]"
pip install "vox-llm[gemini]"

# Optional: client-side video frame extraction for non-Gemini providers
pip install "vox-llm[video]"
```

The PyPI distribution name is `vox-llm` (the bare `vox` was taken). The Python import name is `vox` — `from vox import VoxClient` works unchanged.

Requires Python 3.11+.

## Configure

Pass API keys directly, or rely on environment variables:

```python
from vox import VoxClient

client = VoxClient(
    openai_api_key="sk-...",            # or OPENAI_API_KEY env var
    anthropic_api_key="sk-ant-...",     # or ANTHROPIC_API_KEY env var
    gemini_api_key="...",               # or GEMINI_API_KEY env var
    openrouter_api_key="sk-or-...",     # or OPENROUTER_API_KEY env var
    lmstudio_base_url="http://localhost:1234/v1",  # default
)
```

## First call

```python
from vox import Message, VoxClient

client = VoxClient()  # picks up keys from environment

response = client.complete(
    [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Explain quantum entanglement in one sentence."),
    ],
    model="gpt-5",
    max_tokens=200,
)

print(response.message.text)
print(f"Tokens: {response.usage.total_tokens}")
print(f"Cost: ${response.usage.estimated_cost:.6f}")
```

## Switching providers

Model name resolution is automatic for the prefixes vox knows:

| Model prefix          | Provider     |
| --------------------- | ------------ |
| `gpt-`, `o1`, `o3`, `o4` | OpenAI       |
| `claude-`             | Anthropic    |
| `gemini-`             | Gemini       |
| `whisper-`, `tts-`    | OpenAI (audio) |

For OpenRouter and LM Studio, pass `provider=` explicitly:

```python
client.complete(
    messages,
    model="meta-llama/llama-3-70b",
    provider="openrouter",
)
```

## Per-provider configuration

Override timeouts, retries, and provider-specific options with `ProviderConfig`:

```python
from vox import ProviderConfig, VoxClient

client = VoxClient(
    provider_configs={
        "openai": ProviderConfig(
            api_key="sk-...",
            timeout=60.0,
            max_retries=3,
        ),
        "openrouter": ProviderConfig(
            api_key="sk-or-...",
            app_name="MyApp",              # X-Title header
            app_url="https://myapp.com",   # HTTP-Referer header
        ),
    }
)
```

## Async

Every entry point has an async variant — `acomplete`, `astream`, `atranscribe`, `asynthesize`:

```python
response = await client.acomplete(
    [Message(role="user", content="Hello")],
    model="claude-sonnet-4-5-20250929",
)
```

## Next steps

- **[Streaming](guides/streaming.md)** — incremental responses, chunk types, tool-call deltas
- **[Tool use](guides/tools.md)** — define tools once, run across providers
- **[Structured output](guides/structured-output.md)** — Pydantic models in, validated instances out
- **[Multimodal](guides/multimodal.md)** — vision and video
- **[Audio I/O](guides/audio.md)** — transcribe and synthesize
- **[Reliability](guides/reliability.md)** — retries and observability callbacks
- **[API Reference](reference/client.md)** — every public type
