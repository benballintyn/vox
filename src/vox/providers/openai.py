"""OpenAI provider using the Responses API.

This is OpenAI's forward-looking API, separate from Chat Completions.
OpenRouter and LM Studio use Chat Completions instead (see _chat_completions.py).
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

from pydantic import BaseModel

from .._structured import (
    pydantic_to_openai_responses_text_format,
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
from ..models.reasoning import ReasoningConfig, ThinkingBlock
from ..models.responses import CompletionResponse, StreamChunk, Usage
from ..models.tools import Tool
from .base import Provider


def _import_openai() -> Any:
    """Lazily import the openai package."""
    try:
        import openai

        return openai
    except ImportError:
        raise ImportError(
            "The openai package is required for the OpenAI provider. "
            "Install it with: pip install vox[openai]"
        ) from None


class OpenAIProvider(Provider):
    """OpenAI provider using the Responses API.

    Args:
        config: Provider configuration.
    """

    def __init__(self, config: ProviderConfig | None = None) -> None:
        super().__init__(config)
        self._sync_client: Any = None
        self._async_client: Any = None

    @property
    def provider_name(self) -> str:
        return "openai"

    def _get_api_key(self) -> str | None:
        """Resolve API key from config or environment."""
        return self.config.api_key or os.environ.get("OPENAI_API_KEY")

    def _get_sync_client(self) -> Any:
        """Get or create the synchronous OpenAI client."""
        if self._sync_client is None:
            openai = _import_openai()
            kwargs: dict[str, Any] = {
                "api_key": self._get_api_key(),
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
            }
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._sync_client = openai.OpenAI(**kwargs)
        return self._sync_client

    def _get_async_client(self) -> Any:
        """Get or create the asynchronous OpenAI client."""
        if self._async_client is None:
            openai = _import_openai()
            kwargs: dict[str, Any] = {
                "api_key": self._get_api_key(),
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
            }
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._async_client = openai.AsyncOpenAI(**kwargs)
        return self._async_client

    # ── Message translation ──────────────────────────────────────────────

    def _translate_input(self, messages: list[Message]) -> tuple[list[dict[str, Any]], str | None]:
        """Translate vox Messages to Responses API input format.

        The Responses API uses ``input`` (list of items) instead of ``messages``.
        System prompts are passed via ``instructions``.

        Args:
            messages: List of vox Message objects.

        Returns:
            Tuple of (input items list, system instructions or None).
        """
        instructions = None
        items: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                instructions = msg.text
                continue

            if msg.role == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id or "",
                        "output": msg.text,
                    }
                )
                continue

            if msg.role == "assistant":
                # Add text content
                if msg.text:
                    items.append(
                        {
                            "role": "assistant",
                            "type": "message",
                            "content": [{"type": "output_text", "text": msg.text}],
                        }
                    )
                # Add tool calls
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        items.append(
                            {
                                "type": "function_call",
                                "id": tc.id,
                                "call_id": tc.id,
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            }
                        )
                continue

            # User messages
            content = self._translate_user_content(msg)
            items.append(
                {
                    "role": "user",
                    "type": "message",
                    "content": content,
                }
            )

        return items, instructions

    def _translate_user_content(self, msg: Message) -> list[dict[str, Any]]:
        """Translate user message content to Responses API format.

        Args:
            msg: A user message.

        Returns:
            List of content items.
        """
        if isinstance(msg.content, str):
            return [{"type": "input_text", "text": msg.content}]

        parts: list[dict[str, Any]] = []
        for part in msg.content:
            if isinstance(part, TextContent):
                parts.append({"type": "input_text", "text": part.text})
            elif isinstance(part, ImageContent):
                if part.source_type == "url":
                    parts.append(
                        {
                            "type": "input_image",
                            "image_url": part.data,
                        }
                    )
                else:
                    parts.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:{part.media_type};base64,{part.data}",
                        }
                    )
        return parts

    def _translate_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        """Translate vox Tools to Responses API format.

        Args:
            tools: List of vox Tool objects.

        Returns:
            List of tool dicts for the ``tools`` parameter.
        """
        return [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in tools
        ]

    # ── Build request ────────────────────────────────────────────────────

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
        """Build kwargs for a Responses API call.

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
        input_items, instructions = self._translate_input(messages)

        request: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

        if stream:
            request["stream"] = True

        if instructions:
            request["instructions"] = instructions

        if tools:
            request["tools"] = self._translate_tools(tools)

        if response_schema:
            request["text"] = pydantic_to_openai_responses_text_format(response_schema)

        if stop:
            request["stop"] = stop

        if reasoning and reasoning.enabled:
            reasoning_config: dict[str, Any] = {}
            # Provider-specific overrides take priority over semantic level
            if reasoning.openai:
                if reasoning.openai.effort:
                    reasoning_config["effort"] = reasoning.openai.effort
                if reasoning.openai.summary:
                    reasoning_config["summary"] = reasoning.openai.summary
            elif reasoning.level:
                reasoning_config["effort"] = reasoning.level
            if reasoning_config:
                request["reasoning"] = reasoning_config

        request.update(kwargs)
        return request

    # ── Response translation ─────────────────────────────────────────────

    def _translate_response(
        self,
        response: Any,
        model: str,
        response_schema: type[BaseModel] | None = None,
    ) -> CompletionResponse:
        """Translate a Responses API response to vox CompletionResponse.

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

        for item in response.output:
            item_type = getattr(item, "type", None)

            if item_type == "message":
                for content in item.content:
                    content_type = getattr(content, "type", None)
                    if content_type == "output_text":
                        text_parts.append(content.text)
                    elif content_type == "refusal":
                        text_parts.append(content.refusal)

            elif item_type == "function_call":
                args_str = getattr(item, "arguments", "{}")
                tool_calls.append(
                    ToolCallData(
                        id=getattr(item, "call_id", item.id),
                        name=item.name,
                        arguments=json.loads(args_str) if args_str else {},
                    )
                )

            elif item_type == "reasoning":
                for summary in getattr(item, "summary", []):
                    if hasattr(summary, "text"):
                        thinking_blocks.append(ThinkingBlock(text=summary.text))

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
                reasoning_tokens=getattr(response.usage, "reasoning_tokens", 0) or 0,
            )

        parsed = None
        if response_schema and text_parts:
            parsed = validate_structured_response(response_schema, "".join(text_parts))

        finish_reason = None
        if hasattr(response, "status"):
            finish_reason = response.status

        return CompletionResponse(
            message=message,
            usage=usage,
            provider="openai",
            model=model,
            finish_reason=finish_reason,
            thinking=thinking_blocks or None,
            parsed=parsed,
        )

    # ── Error handling ───────────────────────────────────────────────────

    def _handle_error(self, e: Exception) -> None:
        """Translate openai SDK exceptions to vox errors.

        Args:
            e: The caught exception.

        Raises:
            VoxError: The appropriate vox error subclass.
        """
        openai = _import_openai()

        if isinstance(e, openai.AuthenticationError):
            raise AuthenticationError(str(e), provider="openai") from e
        if isinstance(e, openai.RateLimitError):
            raise RateLimitError(str(e), provider="openai") from e
        if isinstance(e, openai.BadRequestError):
            msg = str(e)
            if "model" in msg.lower() and "not found" in msg.lower():
                raise ModelNotFoundError(msg, provider="openai") from e
            raise InvalidRequestError(msg, provider="openai") from e
        if isinstance(e, openai.NotFoundError):
            raise ModelNotFoundError(str(e), provider="openai") from e
        if isinstance(e, openai.PermissionDeniedError):
            msg = str(e).lower()
            if "quota" in msg or "billing" in msg or "credit" in msg:
                raise QuotaExceededError(str(e), provider="openai") from e
            raise AuthenticationError(str(e), provider="openai") from e
        if isinstance(e, openai.UnprocessableEntityError):
            raise InvalidRequestError(str(e), provider="openai") from e
        if isinstance(e, openai.APIStatusError):
            status = getattr(e, "status_code", None)
            if status == 402:
                raise QuotaExceededError(str(e), provider="openai") from e
            if status == 429:
                raise RateLimitError(str(e), provider="openai") from e
            raise ProviderError(str(e), provider="openai") from e
        if isinstance(e, openai.APIConnectionError):
            raise ProviderError(str(e), provider="openai") from e

        raise ProviderError(str(e), provider="openai") from e

    # ── Streaming helpers ────────────────────────────────────────────────

    def _process_stream_event(self, event: Any) -> StreamChunk | None:
        """Translate a Responses API stream event to a StreamChunk.

        Args:
            event: A raw SDK stream event.

        Returns:
            A StreamChunk, or None if the event should be skipped.
        """
        event_type = getattr(event, "type", None)

        if event_type == "response.output_text.delta":
            return StreamChunk(type="text", text=event.delta)

        if event_type == "response.function_call_arguments.delta":
            return StreamChunk(
                type="tool_call_delta",
                tool_call_id=getattr(event, "call_id", ""),
                arguments_delta=event.delta,
            )

        if event_type == "response.output_item.added":
            item = event.item
            if getattr(item, "type", None) == "function_call":
                return StreamChunk(
                    type="tool_call_start",
                    tool_call=ToolCallData(
                        id=getattr(item, "call_id", getattr(item, "id", "")),
                        name=getattr(item, "name", ""),
                        arguments={},
                    ),
                )

        if event_type == "response.reasoning_summary_text.delta":
            return StreamChunk(type="thinking", thinking_text=event.delta)

        if event_type == "response.completed":
            resp = event.response
            usage = None
            if resp.usage:
                usage = Usage(
                    prompt_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
                    completion_tokens=getattr(resp.usage, "output_tokens", 0) or 0,
                    total_tokens=(
                        (getattr(resp.usage, "input_tokens", 0) or 0)
                        + (getattr(resp.usage, "output_tokens", 0) or 0)
                    ),
                    reasoning_tokens=getattr(resp.usage, "reasoning_tokens", 0) or 0,
                )
            return StreamChunk(
                type="done",
                finish_reason=getattr(resp, "status", None),
                usage=usage,
            )

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
        """Synchronous Responses API call.

        Args:
            messages: Conversation messages.
            model: Model identifier (default: gpt-4o).
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
            response = self._get_sync_client().responses.create(**request)
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
        """Asynchronous Responses API call.

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
            response = await self._get_async_client().responses.create(**request)
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
        """Synchronous streaming Responses API call.

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
            response_stream = self._get_sync_client().responses.create(**request)
            for event in response_stream:
                chunk = self._process_stream_event(event)
                if chunk:
                    yield chunk
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
        """Asynchronous streaming Responses API call.

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
            response_stream = await self._get_async_client().responses.create(**request)
            async for event in response_stream:
                chunk = self._process_stream_event(event)
                if chunk:
                    yield chunk
        except Exception as e:
            from ..errors import VoxError

            if isinstance(e, VoxError):
                raise
            self._handle_error(e)
