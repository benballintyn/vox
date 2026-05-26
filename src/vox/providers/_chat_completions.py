"""Shared base for providers using the OpenAI Chat Completions API protocol.

Used by OpenRouter and LM Studio, which expose OpenAI-compatible endpoints.
The actual OpenAI provider uses the newer Responses API instead.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterator, Sequence
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
from ..models.responses import (
    CompletionResponse,
    StreamChunk,
    Usage,
    normalize_finish_reason,
)
from ..models.tools import TOOL_SPEC_TYPE_ERROR, Tool, ToolSpec
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


def _extract_retry_after(exc: Exception) -> float | None:
    """Pull a Retry-After value out of an SDK exception's response headers."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


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
            d["content"] = msg.text
            if msg.tool_call_id:
                d["tool_call_id"] = msg.tool_call_id
            return d

        # Handle content
        if isinstance(msg.content, str):
            d["content"] = msg.content
        else:
            parts: list[dict[str, Any]] = []
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

    def _translate_tools(self, tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
        """Translate tool specs to Chat Completions format.

        A vox ``Tool`` is translated to the function-tool shape. A raw dict is
        passed through verbatim — the escape hatch for provider-native tools.

        Args:
            tools: List of vox Tool objects and/or raw provider-native dicts.

        Returns:
            List of tool dicts for the ``tools`` parameter.

        Raises:
            TypeError: If an entry is neither a vox Tool nor a dict.
        """
        result: list[dict[str, Any]] = []
        for t in tools:
            if isinstance(t, dict):
                result.append(t)
            elif isinstance(t, Tool):
                result.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        },
                    }
                )
            else:
                raise TypeError(TOOL_SPEC_TYPE_ERROR.format(got=type(t).__name__))
        return result

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

        raw_finish = choice.finish_reason
        return CompletionResponse(
            message=message,
            usage=usage,
            provider=self.provider_name,
            model=model,
            finish_reason=normalize_finish_reason(raw_finish),
            raw_finish_reason=raw_finish,
            parsed=parsed,
            response_id=getattr(response, "id", None),
        )

    # ── Build request kwargs ─────────────────────────────────────────────

    def _build_request_kwargs(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        tools: Sequence[ToolSpec] | None,
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
            raise RateLimitError(
                str(e),
                retry_after=_extract_retry_after(e),
                provider=provider,
            ) from e
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
                raise RateLimitError(
                    str(e),
                    retry_after=_extract_retry_after(e),
                    provider=provider,
                ) from e
            raise ProviderError(str(e), provider=provider) from e
        if isinstance(e, openai.APIConnectionError):
            raise ProviderError(str(e), provider=provider) from e

        raise ProviderError(str(e), provider=provider) from e

    # ── Streaming helpers ────────────────────────────────────────────────

    def _translate_stream_chunk(
        self, chunk: Any, state: dict[str, Any] | None = None
    ) -> list[StreamChunk]:
        """Translate a single Chat Completions stream chunk to ``StreamChunk``s.

        Args:
            chunk: A raw SDK stream chunk.
            state: Mutable per-stream state. Carries:

                * ``current_tool_call_id`` — id from the first delta of
                  a tool call, replayed on subsequent argument-delta
                  chunks that don't repeat it (vox#20). The Chat
                  Completions SDK sets ``tc.id`` only on the *first*
                  delta for a given tool call; later deltas have
                  ``tc.id is None``.
                * ``usage_emitted`` — dedup guard. Direct OpenAI sends
                  a final ``choices=[]`` chunk carrying ``usage``;
                  OpenRouter includes ``usage`` on the *same* chunk as
                  the final content / ``finish_reason`` (vox#27). We
                  accept usage from either shape but emit exactly one
                  ``type="usage"`` StreamChunk.
                * ``done_emitted`` — dedup guard for repeated
                  ``finish_reason`` across chunks (some proxied
                  providers emit it more than once).

        Returns:
            List of StreamChunk instances (may be empty).
        """
        if state is None:
            state = {}
        chunks: list[StreamChunk] = []

        # First: process the chunk's body if it has choices (text,
        # tool deltas). Usage and done emission happen after, in
        # the order ``text → usage → done`` so consumers iterating
        # by type see a coherent sequence.
        if chunk.choices:
            choice = chunk.choices[0]
            delta = choice.delta

            # Tool call deltas. A single chunk may contain BOTH the
            # function name (first occurrence) and the start of the
            # arguments — emit both, don't return early after the name.
            if delta.tool_calls:
                tc = delta.tool_calls[0]
                # Buffer the id on first sight; reuse it for subsequent
                # name-less / id-less argument deltas (vox#20).
                if tc.id:
                    state["current_tool_call_id"] = tc.id
                tool_call_id = state.get("current_tool_call_id", "")
                if tc.function and tc.function.name:
                    chunks.append(
                        StreamChunk(
                            type="tool_call_start",
                            tool_call=ToolCallData(
                                id=tool_call_id,
                                name=tc.function.name,
                                arguments={},
                            ),
                        )
                    )
                if tc.function and tc.function.arguments:
                    chunks.append(
                        StreamChunk(
                            type="tool_call_delta",
                            tool_call_id=tool_call_id,
                            arguments_delta=tc.function.arguments,
                        )
                    )

            # Text delta
            if delta.content:
                chunks.append(StreamChunk(type="text", text=delta.content))

        # Usage — accept from anywhere it appears in the stream.
        # Direct OpenAI: arrives on a final ``choices=[]`` chunk.
        # OpenRouter (vox#27): on the same chunk as content / finish.
        # State dedups so consumers see exactly one ``usage`` chunk
        # even if the same usage info is repeated.
        if hasattr(chunk, "usage") and chunk.usage and not state.get("usage_emitted"):
            state["usage_emitted"] = True
            chunks.append(
                StreamChunk(
                    type="usage",
                    usage=Usage(
                        prompt_tokens=chunk.usage.prompt_tokens or 0,
                        completion_tokens=chunk.usage.completion_tokens or 0,
                        total_tokens=chunk.usage.total_tokens or 0,
                    ),
                )
            )

        # Finish reason — buffered rather than emitted immediately.
        # OpenRouter (and direct OpenAI with the deprecated
        # ``stream_options.include_usage`` flag) delivers ``usage`` on
        # a *trailing* chunk that arrives AFTER the finish_reason chunk.
        # If we emitted ``done`` here, ``usage`` would land *after*
        # ``done`` in the output stream, breaking the cross-provider
        # contract (text → usage → done). Buffer instead; the public
        # ``stream`` / ``astream`` wrappers call ``_finalize_stream``
        # at end-of-stream to flush a single ``done``.
        #
        # Dedup is automatic: only the *first* finish_reason landing
        # in this buffer wins; subsequent ones (some proxied providers
        # repeat finish_reason across chunks) are silently dropped.
        if chunk.choices:
            choice = chunk.choices[0]
            if choice.finish_reason and "buffered_finish_reason" not in state:
                state["buffered_finish_reason"] = choice.finish_reason

        return chunks

    def _finalize_stream(self, state: dict[str, Any]) -> list[StreamChunk]:
        """Flush any deferred chunks at end-of-stream.

        Called by ``stream`` / ``astream`` after the SDK's chunk
        iterator is exhausted. Currently emits the single ``done``
        chunk buffered by ``_translate_stream_chunk`` so usage (which
        may arrive on a trailing chunk) is always emitted first.

        Args:
            state: The per-stream state dict.

        Returns:
            Chunks to yield after the per-chunk loop.
        """
        chunks: list[StreamChunk] = []
        finish_reason = state.get("buffered_finish_reason")
        if finish_reason and not state.get("done_emitted"):
            state["done_emitted"] = True
            chunks.append(
                StreamChunk(
                    type="done",
                    finish_reason=normalize_finish_reason(finish_reason),
                )
            )
        return chunks

    # ── Public API ───────────────────────────────────────────────────────

    def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        tools: Sequence[ToolSpec] | None = None,
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
        tools: Sequence[ToolSpec] | None = None,
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
        tools: Sequence[ToolSpec] | None = None,
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
        # Per-stream state for the chunk translator: buffers the
        # tool_call id so subsequent argument-delta chunks (which the
        # Chat Completions SDK leaves with ``tc.id is None``) inherit
        # the id from the first delta of the same call (vox#20).
        state: dict[str, Any] = {}
        try:
            response_stream = self._get_sync_client().chat.completions.create(**request)
            for chunk in response_stream:
                yield from self._translate_stream_chunk(chunk, state)
            # Flush deferred ``done`` (and any future buffered chunks).
            yield from self._finalize_stream(state)
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
        tools: Sequence[ToolSpec] | None = None,
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
        state: dict[str, Any] = {}
        try:
            response_stream = await self._get_async_client().chat.completions.create(**request)
            async for chunk in response_stream:
                for translated in self._translate_stream_chunk(chunk, state):
                    yield translated
            for translated in self._finalize_stream(state):
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
