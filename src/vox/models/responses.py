"""Completion response and streaming types."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from .messages import Message, ToolCallData
from .reasoning import ThinkingBlock

# Normalized finish reasons (a small, stable set the user can switch on).
FinishReason = Literal[
    "stop",  # Natural end of turn / model decided to stop
    "length",  # Hit max_tokens / output token limit
    "tool_calls",  # Stopped to make tool calls
    "content_filter",  # Blocked by safety/content filtering
    "stop_sequence",  # Hit a user-specified stop sequence
    "other",  # Anything else (rare; kept for forward compat)
]

# Mapping from provider-native finish reasons to normalized values.
_FINISH_REASON_MAP: dict[str, FinishReason] = {
    # OpenAI Chat Completions + Responses API
    "stop": "stop",
    "length": "length",
    "max_output_tokens": "length",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "content_filter": "content_filter",
    # Anthropic
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
    "pause_turn": "other",
    # Gemini (uppercase native, but provider lowercases before mapping)
    "max_tokens".lower(): "length",  # also matches Anthropic
    "safety": "content_filter",
    "recitation": "content_filter",
    "blocklist": "content_filter",
    "prohibited_content": "content_filter",
    "spii": "content_filter",
    "malformed_function_call": "other",
    # Stop sequences (multiple provider names)
    "stop_sequence": "stop_sequence",
}


def normalize_finish_reason(raw: str | None) -> FinishReason | None:
    """Normalize a provider-specific finish reason to a common vocabulary.

    Args:
        raw: The provider's native finish reason string, or None.

    Returns:
        A normalized FinishReason, or None if input was None. Unknown values
        map to "other".
    """
    if raw is None:
        return None
    return _FINISH_REASON_MAP.get(raw.lower().strip(), "other")


class Usage(BaseModel):
    """Token usage information.

    Args:
        prompt_tokens: Tokens used in the prompt/input.
        completion_tokens: Tokens used in the completion/output.
        total_tokens: Total tokens used.
        reasoning_tokens: Tokens used for reasoning/thinking.
        cache_read_tokens: Tokens read from cache.
        cache_creation_tokens: Tokens used to create cache entries.
        model: Model identifier that produced this usage. Populated by
            ``VoxClient`` from the request's ``model`` argument so
            ``Usage`` is priceable standalone (without needing a
            separate handle to the response / request).
        estimated_cost: Estimated USD cost computed by
            :func:`vox.estimate_cost` against vox's built-in price
            snapshot (or a ``custom_pricing`` override passed to
            ``VoxClient``). ``None`` when the model is unknown to the
            pricing table or when the ``Usage`` was constructed
            manually outside ``VoxClient``. Estimate only — not a
            substitute for the provider's authoritative billing.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    model: str | None = None
    estimated_cost: float | None = None


class CompletionResponse(BaseModel):
    """Full completion response from an LLM provider.

    Args:
        message: The assistant's response message.
        usage: Token usage statistics.
        provider: Name of the provider that generated this response.
        model: Model identifier used for this completion.
        finish_reason: Normalized stop reason (``stop``, ``length``,
            ``tool_calls``, ``content_filter``, ``stop_sequence``, ``other``).
        raw_finish_reason: The provider's native finish reason string,
            preserved verbatim for debugging.
        thinking: Thinking/reasoning blocks, if reasoning was enabled.
        parsed: Validated Pydantic instance when ``response_schema`` was used.
        response_id: Provider-assigned response identifier. For OpenAI's
            Responses API, this is the ID needed for ``previous_response_id``
            on a follow-up turn (stateful chaining).
    """

    message: Message
    usage: Usage
    provider: str
    model: str
    finish_reason: FinishReason | None = None
    raw_finish_reason: str | None = None
    thinking: list[ThinkingBlock] | None = None
    parsed: Any = None
    response_id: str | None = None


class TranscriptionResponse(BaseModel):
    """Result of a :meth:`VoxClient.transcribe` call.

    Args:
        text: The transcribed text. Always populated.
        language: ISO-639-1 language code, if the provider reports it.
            OpenAI Whisper returns this when
            ``response_format="verbose_json"`` (vox always requests
            verbose). Gemini's transcribe-via-prompt path does not
            report this — populated only when the model emits a
            recognizable language hint.
        duration: Audio duration in seconds, if the provider reports it.
            OpenAI Whisper does; Gemini does not.
        provider: Provider name that produced this transcription.
        model: Model identifier used (e.g. ``"whisper-1"``,
            ``"gemini-3.5-flash"``).
        usage: Token usage, when the provider reports it. OpenAI
            Whisper is priced per audio second (no token surface), so
            ``usage`` is ``None``. Gemini reports input audio tokens +
            output text tokens via the standard ``Usage`` shape.
    """

    text: str
    language: str | None = None
    duration: float | None = None
    provider: str
    model: str
    usage: Usage | None = None


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
