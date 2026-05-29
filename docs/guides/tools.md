# Tool Use

Define tools once, run them across every provider. vox translates the tool definitions and results to each provider's native format.

## Basic loop

```python
from vox import Message, Tool, ToolResult, VoxClient

client = VoxClient()

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
response = client.complete(messages, model="gpt-5", tools=tools)

# 3. Handle tool calls
if response.message.tool_calls:
    messages.append(response.message)  # add assistant's tool-call turn

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
    final = client.complete(messages, model="gpt-5", tools=tools)
    print(final.message.text)
```

The same code works against `claude-sonnet-4-5-20250929`, `gemini-3.1-flash-lite`, or any OpenRouter model — vox handles the translation.

## Multiple tool calls per turn

A single response may include multiple tool calls. Always iterate `response.message.tool_calls`:

```python
for tc in response.message.tool_calls:
    result = dispatch[tc.name](**tc.arguments)
    messages.append(ToolResult(
        tool_call_id=tc.id, name=tc.name, content=result
    ).to_message())
```

## Provider-state preservation

`ToolCallData.provider_state` carries opaque per-provider IDs (e.g. Gemini's `thought_signature`, OpenAI's reasoning-item ID for round-tripping). Vox populates it on response and uses it when you pass the message back. Don't touch it.

## Server-side / native tools (raw-dict escape hatch)

Some providers expose **server-side tools** that run on their infrastructure — Anthropic's `web_search_20250305`, OpenAI's `web_search_preview`, Gemini's Google Search grounding. These have provider-specific shapes and no cross-provider abstraction, so vox does NOT model them as `Tool`. Pass raw dicts alongside vox `Tool` objects:

```python
response = client.complete(
    [Message(role="user", content="What's the current 10Y JGB yield?")],
    model="claude-sonnet-4-5-20250929",
    tools=[
        my_function_tool,  # vox Tool — translated for the provider
        {                  # raw dict — passed through verbatim
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 5,
        },
    ],
)
```

You're responsible for matching the resolved provider's expected schema — a raw dict shaped for one provider won't work on another. An entry that's neither a `Tool` nor a `dict` raises `TypeError`.

## Streaming tool calls

See [Streaming → Streaming tool calls](streaming.md#streaming-tool-calls).

## See also

- [`Tool`](../reference/tools.md)
- [`ToolCall`](../reference/tools.md) / [`ToolResult`](../reference/tools.md)
- [`ToolCallData`](../reference/messages.md)
