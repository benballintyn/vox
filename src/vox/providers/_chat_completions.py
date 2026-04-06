"""Shared base for providers using the OpenAI Chat Completions API protocol.

Used by OpenRouter and LM Studio, which expose OpenAI-compatible endpoints.
The actual OpenAI provider uses the newer Responses API instead.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

from loguru import logger
from pydantic import BaseModel

from .._structured import (
    pydantic_to_openai_response_format,
    validate_structured_response,
)
from ..errors import (
    AuthenticationError,
    ContentFilterError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
)
from ..models.config import ProviderConfig
from ..models.messages import ImageContent, Message, TextContent, ToolCallData
from ..models.reasoning import ReasoningConfig
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
            "The openai package is required for Chat Completions-based providers. "
            "Install it with: pip install vox[openai]"
        ) from None


class ChatCompletionsProvider(Provider):
    """Base provider for OpenAI Chat Completions API-compatible endpoints.

    Subclasses override ``provider_name``, ``_default_base_url``, and
    ``_default_api_key_env`` to customize behavior.

    Args:
        config: Provider configuration.
    """

    _default_base_url: str = "https://api.openai.com/v1"
    _default_api_key_env: str = "OPENAI_API_KEY"
    _default_model: str = "gpt-4o"

    def __init__(self, config: ProviderConfig | None = None) -> None:
        super().__init__(config)
        self._sync_client: Any = None
        self._async_client: Any = None

    def _get_api_key(self) -> str | None:
        """Resolve the API key from config or environment."""
        return self.config.api_key or os.environ.get(self._default_api_key_env)

    def _get_base_url(self) -> str:
        """Resolve the base URL from config or default."""
        return self.config.base_url or self._default_base_url

    def _get_default_headers(self) -> dict[str, str]:
        """Return extra headers for the client. Override in subclasses."""
        return {}

    def _get_sync_client(self) -> Any:
        """Get or create the synchronous OpenAI client."""
        if self._sync_client is None:
            openai = _import_openai()
            self._sync_client = openai.OpenAI(
                api_key=self._get_api_key(),
                base_url=self._get_base_url(),
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                default_headers=self._get_default_headers() or None,
            )
        return self._sync_client

    def _get_async_client(self) -> Any:
        """Get or create the asynchronous OpenAI client."""
        if self._async_client is None:
            openai = _import_openai()
            self._async_client = openai.AsyncOpenAI(
                api_key=self._get_api_key(),
                base_url=self._get_base_url(),
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                default_headers=self._get_default_headers() or None,
            )
        return self._async_client

    @property
    def provider_name(self) -> str:
        return "chat_completions"

    # ── Message translation ──────────────────────────────────────────────

    def _translate_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Translate vox Messages to Chat Completions format.

        Args:
            messages: List of vox Message objects.

        Returns:
            List of dicts for the ``messages`` parameter.
        """
        result = []
        for msg in messages:
            translated = self._translate_single_message(msg)
            result.append(translated)
        return result

    def _translate_single_message(self, msg: Message) -> dict[str, Any]:
        """Translate a single Message to Chat Completions format."""
        d: dict[str, Any] = {"role": msg.role}

        if msg.role == "tool":
            d["content"] = msg.text if isinstance(msg.content, str) else msg.text
            if msg.tool_call_id:
                d["tool_call_id"] = msg.tool_call_id
            return d

        # Handle content
        if isinstance(msg.content, str):
            d["content"] = msg.content
        else:
            parts = []
            for part in msg.content:
                if isinstance(part, TextContent):
                    parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ImageContent):
                    if part.source_type == "url":
                        url = part.data
                    else:
                        url = f"data:{part.media_type};base64,{part.data}"
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": url},
                        }
                    )
            d["content"] = parts

        # Handle tool calls in assistant messages
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]

        return d

    def _translate_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        """Translate vox Tools to Chat Completions format.

        Args:
            tools: List of vox Tool objects.

        Returns:
            List of tool dicts for the ``tools`` parameter.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    # ── Response translation ─────────────────────────────────────────────

    def _translate_response(
        self,
        response: Any,
        model: str,
        response_schema: type[BaseModel] | None = None,
    ) -> CompletionResponse:
        """Translate a Chat Completions response to vox CompletionResponse.

        Args:
            response: The raw SDK response.
            model: Model name used.
            response_schema: Pydantic model for structured output validation.

        Returns:
            A vox CompletionResponse.
        """
        choice = response.choices[0]
        msg = choice.message

        # Build tool calls
        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                ToolCallData(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments) if tc.function.arguments else {},
                )
                for tc in msg.tool_calls
            ]

        message = Message(
            role="assistant",
            content=msg.content or "",
            tool_calls=tool_calls,
        )

        usage = Usage()
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )

        parsed = None
        if response_schema and msg.content:
            parsed = validate_structured_response(response_schema, msg.content)

        return CompletionResponse(
            message=message,
            usage=usage,
            provider=self.provider_name,
            model=model,
            finish_reason=choice.finish_reason,
            parsed=parsed,
        )

    # ── Build request kwargs ─────────────────────────────────────────────

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
        """Build the kwargs dict for a Chat Completions API call.

        Args:
            messages: Conversation messages.
            model: Model identifier.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            tools: Available tools.
            response_schema: Pydantic model for structured output.
            reasoning: Reasoning configuration.
            stop: Stop sequences.
            stream: Whether to stream.
            **kwargs: Extra passthrough kwargs.

        Returns:
            Dict of keyword arguments for the SDK call.
        """
        request: dict[str, Any] = {
            "model": model,
            "messages": self._translate_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if stream:
            request["stream"] = True
            request["stream_options"] = {"include_usage": True}

        if tools:
            request["tools"] = self._translate_tools(tools)

        if response_schema:
            request["response_format"] = pydantic_to_openai_response_format(response_schema)

        if stop:
            request["stop"] = stop

        if reasoning and reasoning.enabled and reasoning.level:
            logger.debug(
                "Chat Completions reasoning_effort not widely supported; passing as kwarg"
            )
            request["reasoning_effort"] = reasoning.level

        request.update(kwargs)
        return request

    # ── Error handling ───────────────────────────────────────────────────

    def _handle_error(self, e: Exception) -> None:
        """Translate openai SDK exceptions to vox errors.

        Args:
            e: The caught exception.

        Raises:
            VoxError: The appropriate vox error subclass.
        """
        openai = _import_openai()
        provider = self.provider_name

        if isinstance(e, openai.AuthenticationError):
            raise AuthenticationError(str(e), provider=provider) from e
        if isinstance(e, openai.RateLimitError):
            raise RateLimitError(str(e), provider=provider) from e
        if isinstance(e, openai.BadRequestError):
            msg = str(e)
            if "model" in msg.lower() and "not found" in msg.lower():
                raise ModelNotFoundError(msg, provider=provider) from e
            raise InvalidRequestError(msg, provider=provider) from e
        if isinstance(e, openai.NotFoundError):
            raise ModelNotFoundError(str(e), provider=provider) from e
        if isinstance(e, openai.PermissionDeniedError):
            msg = str(e).lower()
            if "quota" in msg or "billing" in msg or "credit" in msg:
                raise QuotaExceededError(str(e), provider=provider) from e
            raise AuthenticationError(str(e), provider=provider) from e
        if isinstance(e, openai.UnprocessableEntityError):
            raise InvalidRequestError(str(e), provider=provider) from e
        if isinstance(e, openai.APIStatusError):
            status = getattr(e, "status_code", None)
            if status == 402:
                raise QuotaExceededError(str(e), provider=provider) from e
            if status == 429:
                raise RateLimitError(str(e), provider=provider) from e
            raise ProviderError(str(e), provider=provider) from e
        if isinstance(e, openai.APIConnectionError):
            raise ProviderError(str(e), provider=provider) from e

        raise ProviderError(str(e), provider=provider) from e

    # ── Streaming helpers ────────────────────────────────────────────────

    def _translate_stream_chunk(self, chunk: Any) -> StreamChunk | None:
        """Translate a single Chat Completions stream chunk to a StreamChunk.

        Args:
            chunk: A raw SDK stream chunk.

        Returns:
            A StreamChunk, or None if the chunk should be skipped.
        """
        # Usage chunk (final)
        if hasattr(chunk, "usage") and chunk.usage and not chunk.choices:
            return StreamChunk(
                type="usage",
                usage=Usage(
                    prompt_tokens=chunk.usage.prompt_tokens or 0,
                    completion_tokens=chunk.usage.completion_tokens or 0,
                    total_tokens=chunk.usage.total_tokens or 0,
                ),
            )

        if not chunk.choices:
            return None

        choice = chunk.choices[0]
        delta = choice.delta

        # Finish reason
        if choice.finish_reason:
            return StreamChunk(type="done", finish_reason=choice.finish_reason)

        # Tool call start
        if delta.tool_calls:
            tc = delta.tool_calls[0]
            if tc.function and tc.function.name:
                return StreamChunk(
                    type="tool_call_start",
                    tool_call=ToolCallData(
                        id=tc.id or "",
                        name=tc.function.name,
                        arguments={},
                    ),
                )
            if tc.function and tc.function.arguments:
                return StreamChunk(
                    type="tool_call_delta",
                    tool_call_id=tc.id or "",
                    arguments_delta=tc.function.arguments,
                )

        # Text delta
        if delta.content:
            return StreamChunk(type="text", text=delta.content)

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
        """Synchronous Chat Completions call.

        Args:
            messages: Conversation messages.
            model: Model identifier.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            tools: Available tools.
            response_schema: Pydantic model for structured output.
            reasoning: Reasoning configuration.
            stop: Stop sequences.
            **kwargs: Provider-specific passthrough.

        Returns:
            CompletionResponse with the model's reply.
        """
        resolved_model = self._resolve_model(model) or self._default_model
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
            response = self._get_sync_client().chat.completions.create(**request)
            return self._translate_response(response, resolved_model, response_schema)
        except Exception as e:
            if isinstance(
                e,
                (
                    AuthenticationError,
                    RateLimitError,
                    QuotaExceededError,
                    InvalidRequestError,
                    ProviderError,
                    ContentFilterError,
                    ModelNotFoundError,
                ),
            ):
                raise
            self._handle_error(e)
            raise  # unreachable, but satisfies type checker

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
        """Asynchronous Chat Completions call.

        Args:
            messages: Conversation messages.
            model: Model identifier.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            tools: Available tools.
            response_schema: Pydantic model for structured output.
            reasoning: Reasoning configuration.
            stop: Stop sequences.
            **kwargs: Provider-specific passthrough.

        Returns:
            CompletionResponse with the model's reply.
        """
        resolved_model = self._resolve_model(model) or self._default_model
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
            response = await self._get_async_client().chat.completions.create(**request)
            return self._translate_response(response, resolved_model, response_schema)
        except Exception as e:
            if isinstance(
                e,
                (
                    AuthenticationError,
                    RateLimitError,
                    QuotaExceededError,
                    InvalidRequestError,
                    ProviderError,
                    ContentFilterError,
                    ModelNotFoundError,
                ),
            ):
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
        """Synchronous streaming Chat Completions call.

        Args:
            messages: Conversation messages.
            model: Model identifier.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            tools: Available tools.
            reasoning: Reasoning configuration.
            stop: Stop sequences.
            **kwargs: Provider-specific passthrough.

        Yields:
            StreamChunk instances as they arrive.
        """
        resolved_model = self._resolve_model(model) or self._default_model
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
            response_stream = self._get_sync_client().chat.completions.create(**request)
            for chunk in response_stream:
                translated = self._translate_stream_chunk(chunk)
                if translated:
                    yield translated
        except Exception as e:
            if isinstance(
                e,
                (
                    AuthenticationError,
                    RateLimitError,
                    QuotaExceededError,
                    InvalidRequestError,
                    ProviderError,
                    ContentFilterError,
                    ModelNotFoundError,
                ),
            ):
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
        """Asynchronous streaming Chat Completions call.

        Args:
            messages: Conversation messages.
            model: Model identifier.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            tools: Available tools.
            reasoning: Reasoning configuration.
            stop: Stop sequences.
            **kwargs: Provider-specific passthrough.

        Yields:
            StreamChunk instances as they arrive.
        """
        resolved_model = self._resolve_model(model) or self._default_model
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
            response_stream = await self._get_async_client().chat.completions.create(**request)
            async for chunk in response_stream:
                translated = self._translate_stream_chunk(chunk)
                if translated:
                    yield translated
        except Exception as e:
            if isinstance(
                e,
                (
                    AuthenticationError,
                    RateLimitError,
                    QuotaExceededError,
                    InvalidRequestError,
                    ProviderError,
                    ContentFilterError,
                    ModelNotFoundError,
                ),
            ):
                raise
            self._handle_error(e)
