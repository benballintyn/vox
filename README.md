# vox

Model-agnostic LLM execution library for Python. One interface, every provider.

Write your code once and run it against OpenAI, Anthropic, Google Gemini, OpenRouter, or local models via LM Studio — with streaming, tool use, structured output, and reasoning support out of the box.

## Installation

```bash
# Core library (no provider SDKs)
pip install vox-llm

# With a specific provider
pip install "vox-llm[openai]"
pip install "vox-llm[anthropic]"
pip install "vox-llm[gemini]"

# All providers
pip install "vox-llm[all]"
```

> **Note**: the PyPI package is `vox-llm` (the name `vox` was already taken).
> The Python import name is still `vox` — `from vox import VoxClient` works
> unchanged.

**From GitHub** (pinned to a tag):

```bash
pip install "vox-llm[all] @ git+https://github.com/benballintyn/vox.git@v0.1.0"
```

Requires Python 3.11+.

## Quick Start

```python
from vox import VoxClient, Message

client = VoxClient(openai_api_key="sk-...")

response = client.complete(
    messages=[Message(role="user", content="What is the speed of light?")],
    model="gpt-4o",
)
print(response.message.text)
```

Switch providers by changing the model name — no other code changes needed:

```python
# OpenAI
response = client.complete(messages, model="gpt-4o")

# Anthropic
response = client.complete(messages, model="claude-sonnet-4-20250514")

# Gemini
response = client.complete(messages, model="gemini-2.5-pro")
```

## Provider Setup

Pass API keys directly or via environment variables:

```python
client = VoxClient(
    openai_api_key="sk-...",           # or OPENAI_API_KEY env var
    anthropic_api_key="sk-ant-...",    # or ANTHROPIC_API_KEY env var
    gemini_api_key="...",              # or GEMINI_API_KEY env var
    openrouter_api_key="sk-or-...",    # or OPENROUTER_API_KEY env var
    lmstudio_base_url="http://localhost:1234/v1",  # default
)
```

### Provider Auto-Detection

Vox resolves the provider from the model name automatically:

| Model prefix | Provider |
|---|---|
| `gpt-`, `o1`, `o3`, `o4` | OpenAI |
| `claude-` | Anthropic |
| `gemini-` | Gemini |

For OpenRouter and LM Studio, pass `provider=` explicitly:

```python
response = client.complete(
    messages=messages,
    model="meta-llama/llama-3-70b",
    provider="openrouter",
)
```

### Per-Provider Configuration

Override defaults with `ProviderConfig`:

```python
from vox import VoxClient, ProviderConfig

client = VoxClient(
    provider_configs={
        "openai": ProviderConfig(
            api_key="sk-...",
            timeout=60.0,
            max_retries=3,
        ),
        "openrouter": ProviderConfig(
            api_key="sk-or-...",
            app_name="MyApp",           # sent as X-Title header
            app_url="https://myapp.com", # sent as HTTP-Referer header
        ),
    }
)
```

## Completions

### Basic

```python
from vox import VoxClient, Message

client = VoxClient(openai_api_key="sk-...")

response = client.complete(
    messages=[
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Explain quantum entanglement."),
    ],
    model="gpt-4o",
    max_tokens=500,
    temperature=0.7,
)

print(response.message.text)
print(f"Tokens: {response.usage.total_tokens}")
```

### Async

```python
response = await client.acomplete(
    messages=[Message(role="user", content="Hello")],
    model="claude-sonnet-4-20250514",
)
```

## Streaming

```python
for chunk in client.stream(
    messages=[Message(role="user", content="Write a haiku about Python.")],
    model="gpt-4o",
):
    if chunk.type == "text":
        print(chunk.text, end="", flush=True)
    elif chunk.type == "usage":
        print(f"\nTokens: {chunk.usage.total_tokens}")
    elif chunk.type == "done":
        print(f"\nFinish reason: {chunk.finish_reason}")
```

### Async Streaming

```python
async for chunk in client.astream(messages=messages, model="gemini-2.5-pro"):
    if chunk.type == "text":
        print(chunk.text, end="")
```

### Stream Chunk Types

| `chunk.type` | Fields | Description |
|---|---|---|
| `"text"` | `text` | Content delta |
| `"tool_call_start"` | `tool_call` | New tool call (id, name, arguments) |
| `"tool_call_delta"` | `tool_call_id`, `arguments_delta` | Partial JSON for tool arguments |
| `"thinking"` | `thinking_text` | Reasoning/thinking delta |
| `"usage"` | `usage` | Final token counts |
| `"done"` | `finish_reason` | Generation complete |

## Tool Use (Function Calling)

Define tools, let the model call them, feed results back:

```python
from vox import VoxClient, Message, Tool, ToolResult

client = VoxClient(openai_api_key="sk-...")

# 1. Define tools
tools = [
    Tool(
        name="get_weather",
        description="Get current weather for a city.",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    ),
]

# 2. Send messages with tools
messages = [Message(role="user", content="What's the weather in Tokyo?")]
response = client.complete(messages=messages, model="gpt-4o", tools=tools)

# 3. Handle tool calls
if response.message.tool_calls:
    messages.append(response.message)  # add assistant's tool call message

    for tc in response.message.tool_calls:
        # Execute the function (your code)
        result = get_weather(tc.arguments["city"])

        # Return result to the model
        tool_result = ToolResult(
            tool_call_id=tc.id,
            name=tc.name,
            content=result,
        )
        messages.append(tool_result.to_message())

    # 4. Get final response
    final = client.complete(messages=messages, model="gpt-4o", tools=tools)
    print(final.message.text)
```

This works identically across OpenAI, Anthropic, Gemini, and OpenRouter — vox translates the tool definitions and results to each provider's native format.

### Provider-native (server-side) tools

Some providers offer server-side tools that run on their infrastructure — Anthropic's `web_search_20250305`, OpenAI's `web_search_preview`, Gemini's Google Search grounding, and others. These have provider-specific shapes and no cross-provider abstraction, so vox does **not** model them as a `Tool`. Instead, the `tools` list accepts raw dicts alongside vox `Tool` objects — raw dicts are passed through to the provider verbatim:

```python
response = client.complete(
    messages=[Message(role="user", content="What's the current 10Y JGB yield?")],
    model="claude-sonnet-4-5-20250929",
    tools=[
        my_function_tool,  # vox Tool — translated to the provider's format
        {                  # raw dict — passed through verbatim
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 5,
        },
    ],
)
```

The caller is responsible for matching the resolved provider's expected schema — a raw dict shaped for one provider won't work on another. An entry that is neither a `Tool` nor a `dict` raises a `TypeError`.

## Structured Output

Pass a Pydantic model to get validated, typed responses:

```python
from pydantic import BaseModel
from vox import VoxClient, Message

class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str
    pros: list[str]
    cons: list[str]

client = VoxClient(openai_api_key="sk-...")

response = client.complete(
    messages=[Message(role="user", content="Review the movie Inception.")],
    model="gpt-4o",
    response_schema=MovieReview,
)

review: MovieReview = response.parsed
print(f"{review.title}: {review.rating}/10")
print(f"Pros: {', '.join(review.pros)}")
```

The schema is automatically converted to each provider's native format:
- **OpenAI**: JSON schema in response_format
- **Anthropic**: Synthetic tool with forced invocation
- **Gemini**: response_schema parameter
- **OpenRouter/LM Studio**: JSON schema in response_format

## Reasoning / Thinking

Enable extended reasoning for models that support it:

```python
from vox import VoxClient, Message, ReasoningConfig

client = VoxClient(anthropic_api_key="sk-ant-...")

response = client.complete(
    messages=[Message(role="user", content="Prove that sqrt(2) is irrational.")],
    model="claude-sonnet-4-20250514",
    reasoning=ReasoningConfig(enabled=True, budget_tokens=10000),
)

# Access thinking blocks
if response.thinking:
    for block in response.thinking:
        print(f"[Thinking] {block.text[:200]}...")

print(response.message.text)
```

### Configuration by Provider

| Provider | Config | Description |
|---|---|---|
| Anthropic | `budget_tokens` | Token budget for extended thinking |
| OpenAI (o-series) | `level` ("low"/"medium"/"high") | Reasoning effort level |
| Gemini 2.5 | `budget_tokens` | Thinking token budget |
| Gemini 3+ | `level` ("low"/"medium"/"high") | Thinking level |

## Multimodal (Vision)

Send images alongside text:

```python
from vox import Message, TextContent, ImageContent

message = Message(
    role="user",
    content=[
        TextContent(text="What's in this image?"),
        ImageContent(
            source_type="url",
            media_type="image/jpeg",
            data="https://example.com/photo.jpg",
        ),
    ],
)

response = client.complete(messages=[message], model="gpt-4o")
```

For base64 images:

```python
import base64

with open("photo.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

message = Message(
    role="user",
    content=[
        TextContent(text="Describe this image."),
        ImageContent(source_type="base64", media_type="image/png", data=b64),
    ],
)
```

### Video input

vox accepts video via a `VideoContent` part that mirrors `ImageContent`'s
shape. Provider routing:

* **Gemini** consumes video natively (inline base64 or hosted URI,
  including YouTube links — `video/mp4`, `video/webm`, etc.).
* **OpenAI, Anthropic, OpenRouter, LM Studio** have no native video
  input today. vox falls back to client-side frame extraction:
  it decodes the video, samples a handful of frames at ~1 fps
  (capped at 8), and substitutes them as `ImageContent` parts before
  dispatch. A loud warning is emitted via `loguru` so the cost
  implication is visible. Install the extra to enable this:
  `pip install 'vox-llm[video]'`. Consumers that want explicit
  control over sampling should pass `ImageContent` parts directly.

```python
from pathlib import Path
from vox import Message, TextContent, VideoContent

video = VideoContent(
    source_type="base64",
    media_type="video/mp4",
    data=Path("clip.mp4").read_bytes(),  # raw bytes auto-base64 encoded
)

response = client.complete(
    messages=[
        Message(
            role="user",
            content=[
                TextContent(text="Summarize what happens in this clip."),
                video,
            ],
        )
    ],
    model="gemini-2.5-pro",  # native; or gpt-5-mini for frame-fallback
)
```

Hosted-URI form (Gemini only — YouTube link or Files-API URI):

```python
VideoContent(
    source_type="url",
    media_type="video/mp4",
    data="https://www.youtube.com/watch?v=...",
)
```

## Audio I/O (transcribe + synthesize)

Audio doesn't fit naturally into the general `complete()` flow — the
flagship reasoning models (Claude Opus / Sonnet, GPT-5, Gemini 3) don't
accept audio natively. vox exposes audio through dedicated methods
that hit each provider's actual STT / TTS surface:

| Provider | `transcribe()` (STT) | `synthesize()` (TTS) |
|---|---|---|
| OpenAI | `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe` | `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts` |
| Gemini | `gemini-3.5-flash`+ (via `generate_content` with audio Part) | `gemini-3.1-flash-tts-preview` (PCM wrapped as WAV) |
| Anthropic / OpenRouter / LM Studio | raises `InvalidRequestError` | raises `InvalidRequestError` |

### Transcribe

```python
from pathlib import Path
from vox import AudioContent, VoxClient

client = VoxClient()

result = client.transcribe(
    AudioContent(
        source_type="base64",
        media_type="audio/wav",
        data=Path("meeting.wav").read_bytes(),  # bytes auto-base64 encoded
    ),
    model="whisper-1",
    language="en",            # ISO-639-1; OpenAI only, Gemini ignores
    prompt="meeting notes",   # optional bias prompt (Whisper)
)

print(result.text)
print(result.language, result.duration)  # populated when provider reports
```

### Synthesize

```python
audio = client.synthesize(
    text="The quick brown fox jumps over the lazy dog.",
    voice="alloy",            # provider-specific values
    model="tts-1",            # or gpt-4o-mini-tts, gemini-3.1-flash-tts-preview
    format="mp3",             # OpenAI: mp3/opus/aac/flac/wav/pcm; Gemini: always wav
    speed=1.0,                # OpenAI only
)

Path("out.mp3").write_bytes(base64.standard_b64decode(audio.data))
```

Available voices:

- **OpenAI** (`vox.providers.openai.OPENAI_TTS_VOICES`):
  `alloy`, `ash`, `ballad`, `coral`, `echo`, `sage`, `shimmer`,
  `verse`, `marin`, `cedar` (`marin` / `cedar` are highest quality).
- **Gemini** (`vox.providers.gemini.GEMINI_TTS_VOICES`):
  `Aoede`, `Charon`, `Fenrir`, `Kore`, `Leda`, `Orus`, `Puck`, `Zephyr`.

Async variants (`atranscribe`, `asynthesize`) mirror the sync API.

## Callbacks (Observability Hooks)

Wire telemetry — OpenTelemetry, Langfuse, Helicone, custom logging,
whatever — without monkey-patching, via the `CallbackHandler`
protocol. Pass any number of handlers to `VoxClient(callbacks=[...])`
and vox fires them around every call.

```python
from vox import CallbackHandler, LoggingHandler, VoxClient

client = VoxClient(
    callbacks=[LoggingHandler()],   # built-in: logs every call via loguru
    capture_content=False,          # default: no PII in event payloads
)
```

Three events per call lifecycle:

| Event | When | Payload |
|---|---|---|
| `on_request(RequestEvent)` | Before the provider call | `model`, `provider`, `method`, `request_kwargs` |
| `on_response(ResponseEvent)` | After a successful response | `model`, `provider`, `method`, `duration_ms`, `usage`, `response` |
| `on_error(ErrorEvent)` | After a failed call (post-retry) | `model`, `provider`, `method`, `duration_ms`, `error` |

Custom handlers implement any subset of the methods:

```python
class CostBudgetTracker:
    def __init__(self) -> None:
        self.spend_usd = 0.0

    def on_response(self, event):
        if event.usage and event.usage.estimated_cost:
            self.spend_usd += event.usage.estimated_cost

tracker = CostBudgetTracker()
client = VoxClient(callbacks=[tracker])
```

### OpenTelemetry without depending on `opentelemetry-api`

Each event ships a `to_otel_attributes()` helper that returns a dict
keyed by the [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
(`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`,
etc.). Consumers wiring vox into OTel get clean spans with the
standard attribute names with one line — vox itself stays
dependency-free.

```python
from opentelemetry import trace

class OTelHandler:
    def on_request(self, event):
        span = trace.get_current_span()
        span.set_attributes(event.to_otel_attributes())

    def on_response(self, event):
        span = trace.get_current_span()
        span.set_attributes(event.to_otel_attributes())
```

### Behaviour

- **No PII by default.** `request_kwargs` strips `messages` / `audio` /
  `text` / `prompt`; `response` is set to `None`. Pass
  `VoxClient(capture_content=True)` to include the full payloads when
  every handler in the list is trusted with sensitive data.
- **Handler exceptions are swallowed** at WARNING level via `loguru`.
  A buggy telemetry handler never breaks the real LLM call.
- **Async paths use a thread executor.** From `acomplete` / `astream`
  / `atranscribe` / `asynthesize`, vox dispatches each handler call
  via `loop.run_in_executor` and returns immediately — a slow handler
  doing blocking I/O won't stall the response.

## Error Handling

All provider errors are normalized to a consistent hierarchy:

```python
from vox.errors import (
    VoxError,              # base class
    AuthenticationError,   # invalid/missing API key
    RateLimitError,        # rate limited (has .retry_after)
    QuotaExceededError,    # billing/quota limit
    InvalidRequestError,   # malformed request
    ProviderError,         # server error (5xx)
    ContentFilterError,    # safety system blocked content
    ModelNotFoundError,    # model doesn't exist
)

try:
    response = client.complete(messages=messages, model="gpt-4o")
except RateLimitError as e:
    print(f"Rate limited by {e.provider}, retry after {e.retry_after}s")
except AuthenticationError as e:
    print(f"Auth failed for {e.provider}: {e}")
except VoxError as e:
    print(f"LLM error: {e}")
```

## API Reference

### VoxClient

```python
VoxClient(
    openai_api_key: str | None = None,
    anthropic_api_key: str | None = None,
    gemini_api_key: str | None = None,
    openrouter_api_key: str | None = None,
    lmstudio_base_url: str = "http://localhost:1234/v1",
    openrouter_app_name: str | None = None,
    openrouter_app_url: str | None = None,
    provider_configs: dict[str, ProviderConfig] | None = None,
)
```

#### Methods

| Method | Signature | Returns |
|---|---|---|
| `complete()` | `(messages, model, *, provider, max_tokens, temperature, tools, response_schema, reasoning, stop, **kwargs)` | `CompletionResponse` |
| `acomplete()` | Same as above | `CompletionResponse` (async) |
| `stream()` | Same as above | `Iterator[StreamChunk]` |
| `astream()` | Same as above | `AsyncIterator[StreamChunk]` |

### CompletionResponse

| Field | Type | Description |
|---|---|---|
| `message` | `Message` | Assistant's response message |
| `usage` | `Usage` | Token counts |
| `provider` | `str` | Provider name |
| `model` | `str` | Model used |
| `finish_reason` | `str \| None` | Why generation stopped |
| `thinking` | `list[ThinkingBlock] \| None` | Reasoning blocks |
| `parsed` | `Any` | Validated Pydantic instance (when `response_schema` used) |

### Message

| Field | Type | Description |
|---|---|---|
| `role` | `"system" \| "user" \| "assistant" \| "tool"` | Message role |
| `content` | `str \| list[ContentPart]` | Text or multimodal content |
| `tool_calls` | `list[ToolCallData] \| None` | Tool calls (assistant messages) |
| `tool_call_id` | `str \| None` | Tool result reference |
| `name` | `str \| None` | Tool name (for tool messages) |

**Property**: `.text` — extracts plain text from any content format.

### Tool

```python
Tool(
    name: str,              # Function name
    description: str,       # What the function does
    parameters: dict,       # JSON Schema for arguments
)
```

### ToolResult

```python
ToolResult(
    tool_call_id: str,      # ID from ToolCallData
    name: str,              # Tool name
    content: str,           # Result content
    is_error: bool = False, # Whether execution failed
)
```

**Method**: `.to_message()` — converts to a `Message` with `role="tool"`.

### Usage

| Field | Type | Description |
|---|---|---|
| `prompt_tokens` | `int` | Input tokens |
| `completion_tokens` | `int` | Output tokens |
| `total_tokens` | `int` | Total tokens |
| `reasoning_tokens` | `int` | Reasoning/thinking tokens |
| `cache_read_tokens` | `int` | Prompt cache hits |
| `cache_creation_tokens` | `int` | Prompt cache writes |

### ProviderConfig

```python
ProviderConfig(
    api_key: str | None = None,
    base_url: str | None = None,
    default_model: str | None = None,
    app_name: str | None = None,     # OpenRouter: X-Title header
    app_url: str | None = None,      # OpenRouter: HTTP-Referer header
    timeout: float = 120.0,
    max_retries: int = 2,
)
```

### ReasoningConfig

```python
ReasoningConfig(
    enabled: bool = True,
    budget_tokens: int | None = None,   # Anthropic, Gemini 2.5
    level: str | None = None,           # "low" | "medium" | "high" — OpenAI o-series, Gemini 3+
)
```

## LM Studio (Local Models)

Run models locally with LM Studio:

```python
client = VoxClient(lmstudio_base_url="http://localhost:1234/v1")

response = client.complete(
    messages=[Message(role="user", content="Hello!")],
    model="local-model",
    provider="lmstudio",
)
```

Make sure LM Studio is running with a model loaded. The default base URL is `http://localhost:1234/v1`.

## License

MIT
