"""Completion response and streaming types."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from .messages import Message, ToolCallData
from .reasoning import ThinkingBlock


class Usage(BaseModel):
    """Token usage information.

    Args:
        prompt_tokens: Tokens used in the prompt/input.
        completion_tokens: Tokens used in the completion/output.
        total_tokens: Total tokens used.
        reasoning_tokens: Tokens used for reasoning/thinking.
        cache_read_tokens: Tokens read from cache.
        cache_creation_tokens: Tokens used to create cache entries.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


class CompletionResponse(BaseModel):
    """Full completion response from an LLM provider.

    Args:
        message: The assistant's response message.
        usage: Token usage statistics.
        provider: Name of the provider that generated this response.
        model: Model identifier used for this completion.
        finish_reason: Why the model stopped generating.
        thinking: Thinking/reasoning blocks, if reasoning was enabled.
        parsed: Validated Pydantic instance when ``response_schema`` was used.
    """

    message: Message
    usage: Usage
    provider: str
    model: str
    finish_reason: str | None = None
    thinking: list[ThinkingBlock] | None = None
    parsed: Any = None


class StreamChunk(BaseModel):
    """A single chunk from a streaming response.

    Discriminated by the ``type`` field. Consumers iterate and switch on chunk type.

    Args:
        type: The kind of chunk.
        text: Text delta (for type="text").
        tool_call: New tool call starting (for type="tool_call_start").
        tool_call_id: ID of tool call being streamed (for type="tool_call_delta").
        arguments_delta: Partial JSON arguments string (for type="tool_call_delta").
        thinking_text: Thinking text delta (for type="thinking").
        usage: Final usage statistics (for type="usage").
        finish_reason: Why generation stopped (for type="done").
    """

    type: Literal["text", "tool_call_start", "tool_call_delta", "thinking", "usage", "done"]
    text: str = ""
    tool_call: ToolCallData | None = None
    tool_call_id: str | None = None
    arguments_delta: str = ""
    thinking_text: str = ""
    usage: Usage | None = None
    finish_reason: str | None = None
