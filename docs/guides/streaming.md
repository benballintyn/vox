# Streaming

Stream tokens as the model produces them. Same provider-agnostic shape across OpenAI, Anthropic, Gemini, and OpenRouter.

## Basic

```python
from vox import Message, VoxClient

client = VoxClient()

for chunk in client.stream(
    [Message(role="user", content="Write a haiku about Python.")],
    model="gpt-5",
):
    if chunk.type == "text":
        print(chunk.text, end="", flush=True)
    elif chunk.type == "usage":
        print(f"\n\nTokens: {chunk.usage.total_tokens}")
    elif chunk.type == "done":
        print(f"Finish reason: {chunk.finish_reason}")
```

## Async

```python
async for chunk in client.astream(messages, model="gemini-3.1-flash-lite"):
    if chunk.type == "text":
        print(chunk.text, end="")
```

## Chunk types

Stream chunks are discriminated by `chunk.type`:

| Type                 | Fields                                  | When it fires                            |
| -------------------- | --------------------------------------- | ---------------------------------------- |
| `"text"`             | `text`                                  | Content delta                            |
| `"tool_call_start"`  | `tool_call`                             | New tool call begins (id, name, args)    |
| `"tool_call_delta"`  | `tool_call_id`, `arguments_delta`       | Partial JSON for that tool call's args   |
| `"thinking"`         | `thinking_text`                         | Reasoning/thinking delta (Anthropic, OpenAI o-series with `summary="auto"`) |
| `"usage"`            | `usage`                                 | Final token counts (one per stream)      |
| `"done"`             | `finish_reason`                         | Generation complete                      |

The `usage` chunk has cost annotated by vox — `chunk.usage.estimated_cost` matches what non-streaming responses get.

## Streaming tool calls

Tool calls in streams arrive as a `tool_call_start` followed by zero or more `tool_call_delta`s. Accumulate the JSON deltas to assemble the full arguments:

```python
import json

calls = {}  # tool_call_id → partial arg string

for chunk in client.stream(messages, model="gpt-5", tools=tools):
    if chunk.type == "tool_call_start":
        calls[chunk.tool_call.id] = ""
    elif chunk.type == "tool_call_delta":
        calls[chunk.tool_call_id] += chunk.arguments_delta
    elif chunk.type == "done":
        # All deltas in; arguments are now complete
        for call_id, args_json in calls.items():
            args = json.loads(args_json)
            ...
```

## Retries + streaming

Vox retries before the first chunk yields, then propagates. See [Reliability](reliability.md#retries) for the rules.

## See also

- [`VoxClient.stream()`](../reference/client.md) / [`VoxClient.astream()`](../reference/client.md)
- [`StreamChunk`](../reference/responses.md)
