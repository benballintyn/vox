# Structured Output

Pass a Pydantic model as `response_schema=` and get a validated typed instance back on `response.parsed`. Works on OpenAI, Anthropic, Gemini, and OpenRouter.

## Basic

```python
from pydantic import BaseModel
from vox import Message, VoxClient

class WeatherReport(BaseModel):
    city: str
    temperature_f: float
    conditions: str

client = VoxClient()

response = client.complete(
    [Message(role="user", content="Weather in Tokyo right now?")],
    model="gpt-5",
    response_schema=WeatherReport,
)

assert isinstance(response.parsed, WeatherReport)
print(response.parsed.temperature_f)
```

## Nested + enum + list fields

vox's structured-output path handles arbitrarily nested Pydantic models, including `Literal` enums and `list[...]` fields:

```python
from typing import Literal
from pydantic import BaseModel, Field

class Citation(BaseModel):
    url: str
    excerpt: str

class Answer(BaseModel):
    response: str
    confidence: Literal["low", "medium", "high"]
    citations: list[Citation] = Field(default_factory=list)

response = client.complete(messages, model="gpt-5", response_schema=Answer)
print(response.parsed.confidence)
for c in response.parsed.citations:
    print(c.url)
```

## What happens under the hood

vox converts the Pydantic model to each provider's native schema format:

| Provider | Mechanism |
| -------- | --------- |
| OpenAI   | `response_format={"type": "json_schema", "json_schema": {...}, "strict": True}` |
| Anthropic | Tool-use trick — synthesizes a single tool the model is forced to call |
| Gemini   | `response_schema=` on the SDK config (native support) |
| OpenRouter | Routes to the underlying model's schema (OpenAI for most) |

The parsed instance is validated against your Pydantic model. If the model returns malformed JSON or misses required fields, vox raises a `ValidationError` from Pydantic.

## Streaming + structured output

These are mutually exclusive today — `stream()` rejects `response_schema=`. If a consumer needs stream-for-UX-then-validate, file an issue.

## See also

- [`VoxClient.complete()`](../reference/client.md) — `response_schema` parameter
- [`CompletionResponse.parsed`](../reference/responses.md)
