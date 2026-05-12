"""Anthropic provider using the Messages API."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

from pydantic import BaseModel

from .._structured import (
    pydantic_to_anthropic_tool,
    validate_structured_response,
)
from ..errors import (
    AuthenticationError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
)
from ..models.config import ProviderConfig
from ..models.messages import ImageContent, Message, TextContent, ToolCallData
from ..models.reasoning import LEVEL_TO_BUDGET_TOKENS, ReasoningConfig, ThinkingBlock
from ..models.responses import CompletionResponse, StreamChunk, Usage
from ..models.tools import Tool
from .base import Provider


def _import_anthropic() -> Any:
    """Lazily import the anthropic package."""
    try:
        import anthropic

        return anthropic
    except ImportError:
        raise ImportError(
            "The anthropic package is required for the Anthropic provider. "
            "Install it with: pip install vox[anthropic]"
        ) from None


class AnthropicProvider(Provider):
    """Anthropic provider using the Messages API.

    Args:
        config: Provider configuration.
    """

    def __init__(self, config: ProviderConfig | None = None) -> None:
        super().__init__(config)
        self._sync_client: Any = None
        self._async_client: Any = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _get_api_key(self) -> str | None:
        """Resolve API key from config or environment."""
        return self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")

    def _get_sync_client(self) -> Any:
        """Get or create the synchronous Anthropic client."""
        if self._sync_client is None:
            anthropic = _import_anthropic()
            kwargs: dict[str, Any] = {
                "api_key": self._get_api_key(),
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
            }
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._sync_client = anthropic.Anthropic(**kwargs)
        return self._sync_client

    def _get_async_client(self) -> Any:
        """Get or create the asynchronous Anthropic client."""
        if self._async_client is None:
            anthropic = _import_anthropic()
            kwargs: dict[str, Any] = {
                "api_key": self._get_api_key(),
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
            }
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._async_client = anthropic.AsyncAnthropic(**kwargs)
        return self._async_client

    # ── Message translation ──────────────────────────────────────────────

    def _translate_messages(
        self, messages: list[Message]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Translate vox Messages to Anthropic Messages API format.

        System messages are extracted and returned separately, as Anthropic
        uses a dedicated ``system`` parameter.

        Args:
            messages: List of vox Message objects.

        Returns:
            Tuple of (messages list, system prompt string or None).
        """
        system_prompt = None
        result: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.text
                continue

            if msg.role == "tool":
                # Tool results must be content blocks in a user message
                result.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id or "",
                                "content": msg.text,
                                **(
                                    {"is_error": True}
                                    if msg.name and msg.name.startswith("error")
                                    else {}
                                ),
                            }
                        ],
                    }
                )
                continue

            if msg.role == "assistant":
                content_blocks: list[dict[str, Any]] = []
                if msg.text:
                    content_blocks.append({"type": "text", "text": msg.text})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.arguments,
                            }
                        )
                result.append({"role": "assistant", "content": content_blocks or msg.text})
                continue

            # User messages
            content = self._translate_user_content(msg)
            result.append({"role": "user", "content": content})

        return result, system_prompt

    def _translate_user_content(self, msg: Message) -> str | list[dict[str, Any]]:
        """Translate user message content to Anthropic format.

        Args:
            msg: A user message.

        Returns:
            String or list of content blocks.
        """
        if isinstance(msg.content, str):
            return msg.content

        parts: list[dict[str, Any]] = []
        for part in msg.content:
            if isinstance(part, TextContent):
                parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImageContent):
                if part.source_type == "url":
                    parts.append(
                        {
                            "type": "image",
                            "source": {"type": "url", "url": part.data},
                        }
                    )
                else:
                    parts.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": part.media_type,
                                "data": part.data,
                            },
                        }
                    )
        return parts

    def _translate_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        """Translate vox Tools to Anthropic tool format.

        Args:
            tools: List of vox Tool objects.

        Returns:
            List of tool dicts for the ``tools`` parameter.
        """
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    # ── Build request ────────────────────────────────────────────────────

    def _resolve_budget(self, reasoning: ReasoningConfig) -> int:
        """Resolve the thinking budget from a ReasoningConfig.

        Provider-specific override takes priority, then semantic level mapping,
        then a sensible default.

        Args:
            reasoning: The reasoning configuration.

        Returns:
            Token budget for Anthropic's ``thinking.budget_tokens``.
        """
        if reasoning.anthropic and reasoning.anthropic.budget_tokens:
            return reasoning.anthropic.budget_tokens
        if reasoning.level:
            return LEVEL_TO_BUDGET_TOKENS[reasoning.level]
        return LEVEL_TO_BUDGET_TOKENS["medium"]

    def _build_request_kwargs(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        tools: list[Tool] | None,
        response_schema: type[BaseModel] | None,
        reasoning: ReasoningConfig | None,
        stop: list[str] | None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build kwargs for an Anthropic Messages API call.

        Args:
            messages: Conversation messages.
            model: Model identifier.
            max_tokens: Max tokens.
            temperature: Sampling temperature.
            tools: Available tools.
            response_schema: Pydantic model for structured output.
            reasoning: Reasoning config.
            stop: Stop sequences.
            stream: Whether to stream.
            **kwargs: Passthrough.

        Returns:
            Dict of keyword arguments.
        """
        translated_messages, system_prompt = self._translate_messages(messages)

        request: dict[str, Any] = {
            "model": model,
            "messages": translated_messages,
            "max_tokens": max_tokens,
        }

        if system_prompt:
            request["system"] = system_prompt

        # Reasoning and temperature are mutually exclusive in Anthropic
        if reasoning and reasoning.enabled:
            budget = self._resolve_budget(reasoning)
            request["thinking"] = {"type": "enabled", "budget_tokens": budget}
            # Anthropic requires temperature=1 when thinking is enabled
        else:
            request["temperature"] = temperature

        if stream:
            request["stream"] = True

        # Handle tools + structured output
        all_tools = []
        tool_choice = None

        if tools:
            all_tools.extend(self._translate_tools(tools))

        if response_schema:
            all_tools.append(pydantic_to_anthropic_tool(response_schema))
            tool_choice = {"type": "tool", "name": "structured_output"}

        if all_tools:
            request["tools"] = all_tools
        if tool_choice:
            request["tool_choice"] = tool_choice

        if stop:
            request["stop_sequences"] = stop

        request.update(kwargs)
        return request

    # ── Response translation ─────────────────────────────────────────────

    def _translate_response(
        self,
        response: Any,
        model: str,
        response_schema: type[BaseModel] | None = None,
    ) -> CompletionResponse:
        """Translate an Anthropic Messages response to vox CompletionResponse.

        Args:
            response: The raw SDK response.
            model: Model name used.
            response_schema: Pydantic model for structured output validation.

        Returns:
            A vox CompletionResponse.
        """
        text_parts: list[str] = []
        tool_calls: list[ToolCallData] = []
        thinking_blocks: list[ThinkingBlock] = []
        structured_input: dict[str, Any] | None = None

        for block in response.content:
            block_type = getattr(block, "type", None)

            if block_type == "text":
                text_parts.append(block.text)
            elif block_type == "tool_use":
                if response_schema and block.name == "structured_output":
                    structured_input = block.input
                else:
                    tool_calls.append(
                        ToolCallData(
                            id=block.id,
                            name=block.name,
                            arguments=block.input if isinstance(block.input, dict) else {},
                        )
                    )
            elif block_type == "thinking":
                thinking_blocks.append(
                    ThinkingBlock(
                        text=getattr(block, "thinking", ""),
                        token_count=None,
                    )
                )

        message = Message(
            role="assistant",
            content="".join(text_parts),
            tool_calls=tool_calls or None,
        )

        usage = Usage()
        if response.usage:
            usage = Usage(
                prompt_tokens=getattr(response.usage, "input_tokens", 0) or 0,
                completion_tokens=getattr(response.usage, "output_tokens", 0) or 0,
                total_tokens=(
                    (getattr(response.usage, "input_tokens", 0) or 0)
                    + (getattr(response.usage, "output_tokens", 0) or 0)
                ),
                cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                cache_creation_tokens=(
                    getattr(response.usage, "cache_creation_input_tokens", 0) or 0
                ),
            )

        parsed = None
        if response_schema and structured_input is not None:
            parsed = validate_structured_response(response_schema, structured_input)

        return CompletionResponse(
            message=message,
            usage=usage,
            provider="anthropic",
            model=model,
            finish_reason=getattr(response, "stop_reason", None),
            thinking=thinking_blocks or None,
            parsed=parsed,
        )

    # ── Error handling ───────────────────────────────────────────────────

    def _handle_error(self, e: Exception) -> None:
        """Translate anthropic SDK exceptions to vox errors.

        Args:
            e: The caught exception.

        Raises:
            VoxError: The appropriate vox error subclass.
        """
        anthropic = _import_anthropic()

        if isinstance(e, anthropic.AuthenticationError):
            raise AuthenticationError(str(e), provider="anthropic") from e
        if isinstance(e, anthropic.RateLimitError):
            raise RateLimitError(str(e), provider="anthropic") from e
        if isinstance(e, anthropic.BadRequestError):
            msg = str(e)
            if "model" in msg.lower() and "not found" in msg.lower():
                raise ModelNotFoundError(msg, provider="anthropic") from e
            raise InvalidRequestError(msg, provider="anthropic") from e
        if isinstance(e, anthropic.NotFoundError):
            raise ModelNotFoundError(str(e), provider="anthropic") from e
        if isinstance(e, anthropic.PermissionDeniedError):
            msg = str(e).lower()
            if "quota" in msg or "billing" in msg or "credit" in msg:
                raise QuotaExceededError(str(e), provider="anthropic") from e
            raise AuthenticationError(str(e), provider="anthropic") from e
        if isinstance(e, anthropic.APIStatusError):
            status = getattr(e, "status_code", None)
            if status == 402:
                raise QuotaExceededError(str(e), provider="anthropic") from e
            if status == 429:
                raise RateLimitError(str(e), provider="anthropic") from e
            raise ProviderError(str(e), provider="anthropic") from e
        if isinstance(e, anthropic.APIConnectionError):
            raise ProviderError(str(e), provider="anthropic") from e

        raise ProviderError(str(e), provider="anthropic") from e

    # ── Streaming ────────────────────────────────────────────────────────

    def _stream_sync(self, request: dict[str, Any]) -> Iterator[StreamChunk]:
        """Execute a synchronous streaming request.

        Args:
            request: Request kwargs (with stream=True already set).

        Yields:
            StreamChunk instances.
        """
        request.pop("stream", None)
        client = self._get_sync_client()

        with client.messages.stream(**request) as stream:
            for event in stream:
                chunk = self._process_stream_event(event)
                if chunk:
                    yield chunk

            # Emit final usage
            final_message = stream.get_final_message()
            if final_message and final_message.usage:
                yield StreamChunk(
                    type="usage",
                    usage=Usage(
                        prompt_tokens=getattr(final_message.usage, "input_tokens", 0) or 0,
                        completion_tokens=(getattr(final_message.usage, "output_tokens", 0) or 0),
                        total_tokens=(
                            (getattr(final_message.usage, "input_tokens", 0) or 0)
                            + (getattr(final_message.usage, "output_tokens", 0) or 0)
                        ),
                    ),
                )

    async def _stream_async(self, request: dict[str, Any]) -> AsyncIterator[StreamChunk]:
        """Execute an asynchronous streaming request.

        Args:
            request: Request kwargs (with stream=True already set).

        Yields:
            StreamChunk instances.
        """
        request.pop("stream", None)
        client = self._get_async_client()

        async with client.messages.stream(**request) as stream:
            async for event in stream:
                chunk = self._process_stream_event(event)
                if chunk:
                    yield chunk

            final_message = await stream.get_final_message()
            if final_message and final_message.usage:
                yield StreamChunk(
                    type="usage",
                    usage=Usage(
                        prompt_tokens=getattr(final_message.usage, "input_tokens", 0) or 0,
                        completion_tokens=(getattr(final_message.usage, "output_tokens", 0) or 0),
                        total_tokens=(
                            (getattr(final_message.usage, "input_tokens", 0) or 0)
                            + (getattr(final_message.usage, "output_tokens", 0) or 0)
                        ),
                    ),
                )

    def _process_stream_event(self, event: Any) -> StreamChunk | None:
        """Translate an Anthropic stream event to a StreamChunk.

        Args:
            event: A raw SDK stream event.

        Returns:
            A StreamChunk, or None if the event should be skipped.
        """
        event_type = getattr(event, "type", None)

        if event_type == "content_block_start":
            block = event.content_block
            block_type = getattr(block, "type", None)
            if block_type == "tool_use":
                return StreamChunk(
                    type="tool_call_start",
                    tool_call=ToolCallData(
                        id=block.id,
                        name=block.name,
                        arguments={},
                    ),
                )
            return None

        if event_type == "content_block_delta":
            delta = event.delta
            delta_type = getattr(delta, "type", None)

            if delta_type == "text_delta":
                return StreamChunk(type="text", text=delta.text)
            if delta_type == "thinking_delta":
                return StreamChunk(type="thinking", thinking_text=delta.thinking)
            if delta_type == "input_json_delta":
                return StreamChunk(
                    type="tool_call_delta",
                    arguments_delta=delta.partial_json,
                )
            return None

        if event_type == "message_stop":
            return StreamChunk(type="done")

        if event_type == "message_delta":
            delta = event.delta
            stop_reason = getattr(delta, "stop_reason", None)
            if stop_reason:
                return StreamChunk(type="done", finish_reason=stop_reason)

        return None

    # ── Public API ───────────────────────────────────────────────────────

    def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        tools: list[Tool] | None = None,
        response_schema: type[BaseModel] | None = None,
        reasoning: ReasoningConfig | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Synchronous Messages API call.

        Args:
            messages: Conversation messages.
            model: Model identifier.
            max_tokens: Max output tokens.
            temperature: Sampling temperature.
            tools: Available tools.
            response_schema: Pydantic model for structured output.
            reasoning: Reasoning configuration.
            stop: Stop sequences.
            **kwargs: Provider-specific passthrough.

        Returns:
            CompletionResponse with the model's reply.
        """
        resolved_model = self._resolve_model(model)
        request = self._build_request_kwargs(
            messages,
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            response_schema=response_schema,
            reasoning=reasoning,
            stop=stop,
            **kwargs,
        )
        try:
            response = self._get_sync_client().messages.create(**request)
            return self._translate_response(response, resolved_model, response_schema)
        except Exception as e:
            from ..errors import VoxError

            if isinstance(e, VoxError):
                raise
            self._handle_error(e)
            raise

    async def acomplete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        tools: list[Tool] | None = None,
        response_schema: type[BaseModel] | None = None,
        reasoning: ReasoningConfig | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Asynchronous Messages API call.

        Args:
            messages: Conversation messages.
            model: Model identifier.
            max_tokens: Max output tokens.
            temperature: Sampling temperature.
            tools: Available tools.
            response_schema: Pydantic model for structured output.
            reasoning: Reasoning configuration.
            stop: Stop sequences.
            **kwargs: Provider-specific passthrough.

        Returns:
            CompletionResponse with the model's reply.
        """
        resolved_model = self._resolve_model(model)
        request = self._build_request_kwargs(
            messages,
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            response_schema=response_schema,
            reasoning=reasoning,
            stop=stop,
            **kwargs,
        )
        try:
            response = await self._get_async_client().messages.create(**request)
            return self._translate_response(response, resolved_model, response_schema)
        except Exception as e:
            from ..errors import VoxError

            if isinstance(e, VoxError):
                raise
            self._handle_error(e)
            raise

    def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        tools: list[Tool] | None = None,
        reasoning: ReasoningConfig | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Synchronous streaming Messages API call.

        Args:
            messages: Conversation messages.
            model: Model identifier.
            max_tokens: Max output tokens.
            temperature: Sampling temperature.
            tools: Available tools.
            reasoning: Reasoning configuration.
            stop: Stop sequences.
            **kwargs: Provider-specific passthrough.

        Yields:
            StreamChunk instances as events arrive.
        """
        resolved_model = self._resolve_model(model)
        request = self._build_request_kwargs(
            messages,
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            response_schema=None,
            reasoning=reasoning,
            stop=stop,
            stream=True,
            **kwargs,
        )
        try:
            yield from self._stream_sync(request)
        except Exception as e:
            from ..errors import VoxError

            if isinstance(e, VoxError):
                raise
            self._handle_error(e)

    async def astream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        tools: list[Tool] | None = None,
        reasoning: ReasoningConfig | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Asynchronous streaming Messages API call.

        Args:
            messages: Conversation messages.
            model: Model identifier.
            max_tokens: Max output tokens.
            temperature: Sampling temperature.
            tools: Available tools.
            reasoning: Reasoning configuration.
            stop: Stop sequences.
            **kwargs: Provider-specific passthrough.

        Yields:
            StreamChunk instances as events arrive.
        """
        resolved_model = self._resolve_model(model)
        request = self._build_request_kwargs(
            messages,
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            response_schema=None,
            reasoning=reasoning,
            stop=stop,
            stream=True,
            **kwargs,
        )
        try:
            async for chunk in self._stream_async(request):
                yield chunk
        except Exception as e:
            from ..errors import VoxError

            if isinstance(e, VoxError):
                raise
            self._handle_error(e)
