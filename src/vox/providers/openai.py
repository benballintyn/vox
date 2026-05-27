"""OpenAI provider using the Responses API.

This is OpenAI's forward-looking API, separate from Chat Completions.
OpenRouter and LM Studio use Chat Completions instead (see _chat_completions.py).
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterator, Sequence
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
            "The openai package is required for the OpenAI provider. "
            "Install it with: pip install vox[openai]"
        ) from None


def _serialize_reasoning_item(item: Any) -> dict[str, Any]:
    """Capture a Responses-API reasoning output item as a round-trippable dict.

    The Responses API requires the reasoning item that preceded a
    ``function_call`` to be replayed on subsequent turns — without it
    the API rejects the assistant message (vox#25). vox preserves the
    item in the next ``ToolCallData.provider_state``; this helper
    converts the SDK's reasoning item to a dict the API accepts back
    as input verbatim.

    Real SDK items are Pydantic BaseModels, so ``model_dump`` produces
    a faithful round-trip including any ``encrypted_content`` payload
    the model uses to carry internal context. A small fallback handles
    test mocks that aren't real Pydantic models.

    Args:
        item: A Responses API output item of type ``reasoning``.

    Returns:
        A dict suitable for inclusion in the ``input`` list on a
        subsequent ``responses.create`` call.
    """
    if hasattr(item, "model_dump"):
        return item.model_dump(exclude_none=True, mode="json")

    # Fallback for tests / unexpected shapes. Callers must supply
    # concrete values (not auto-generated MagicMocks) for any field
    # they want preserved; missing fields are silently omitted.
    out: dict[str, Any] = {"type": "reasoning"}
    item_id = getattr(item, "id", None)
    if isinstance(item_id, str):
        out["id"] = item_id
    encrypted = getattr(item, "encrypted_content", None)
    if isinstance(encrypted, str):
        out["encrypted_content"] = encrypted
    summary = getattr(item, "summary", None)
    if isinstance(summary, list):
        out["summary"] = [
            {
                "type": getattr(s, "type", "summary_text"),
                "text": getattr(s, "text", ""),
            }
            for s in summary
        ]
    return out


def _extract_retry_after(exc: Exception) -> float | None:
    """Pull a Retry-After value out of an SDK exception's response headers.

    Args:
        exc: The SDK exception (typically with a ``.response`` attribute).

    Returns:
        Seconds to wait before retrying, or None if the header is missing
        or unparseable.
    """
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
                # Add tool calls. The Responses API distinguishes two
                # IDs per call: ``id`` (the function_call *output item*
                # ID, prefixed ``fc_*``) and ``call_id`` (the cross-turn
                # tool-call reference, prefixed ``call_*``). The
                # *output item* ID is what the API expects on
                # inbound ``input[*].id``. vox preserves it in
                # ``ToolCallData.provider_state["openai_fc_id"]`` when
                # the original response came from this provider; fall
                # back to ``tc.id`` for ToolCallData built from scratch
                # (e.g. tests) — those flows aren't sending back a
                # previously-issued ID anyway.
                #
                # Reasoning models additionally require the preceding
                # ``reasoning`` item to be replayed alongside each
                # function_call (vox#25). When the inbound translator
                # captured one, it's stashed in
                # ``provider_state["openai_reasoning_item"]``; emit it
                # as a peer input item just before the function_call.
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        state = tc.provider_state or {}
                        reasoning_item = state.get("openai_reasoning_item")
                        if reasoning_item is not None:
                            items.append(reasoning_item)
                        fc_id = state.get("openai_fc_id", tc.id)
                        items.append(
                            {
                                "type": "function_call",
                                "id": fc_id,
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

        from .._video import substitute_video_with_frames

        content_parts = substitute_video_with_frames(
            list(msg.content), provider_name=self.provider_name
        )

        parts: list[dict[str, Any]] = []
        for part in content_parts:
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

    def _translate_tools(self, tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
        """Translate tool specs to Responses API format.

        A vox ``Tool`` is translated to the function-tool shape. A raw dict is
        passed through verbatim — the escape hatch for OpenAI server-side tools
        such as ``web_search_preview``, ``code_interpreter``, or ``file_search``.

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
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                )
            else:
                raise TypeError(TOOL_SPEC_TYPE_ERROR.format(got=type(t).__name__))
        return result

    # ── Build request ────────────────────────────────────────────────────

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
        stop: list[str] | None = None,
        previous_response_id: str | None = None,
        store: bool | None = None,
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
            previous_response_id: ID of a prior response to chain from. When set,
                the API resumes from that response's state — you only need to
                send the new turn's messages, not the full history. Requires
                ``store=True`` on the prior response (default).
            store: Whether OpenAI should persist this response server-side so a
                later request can reference it via ``previous_response_id``.
                Defaults to True (OpenAI's default) when not specified.
            stream: Whether to stream.
            **kwargs: Passthrough. The Responses API does NOT accept ``stop``
                sequences; any caller-provided ``stop`` will be silently
                dropped before the request is sent.

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

        if previous_response_id is not None:
            request["previous_response_id"] = previous_response_id

        if store is not None:
            request["store"] = store

        # Drop stop sequences if a caller threaded them through; the Responses
        # API rejects this parameter (use a logit_bias on stop tokens instead).
        kwargs.pop("stop", None)

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
        # Buffer the most recent reasoning item so the next function_call
        # can attach it to its provider_state (vox#25). gpt-5 emits a
        # ``reasoning`` item right before each tool call; the Responses
        # API requires it to be replayed on subsequent turns.
        last_reasoning_item: dict[str, Any] | None = None

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
                # Capture both IDs: ``call_id`` is the public reference
                # consumers use in tool result messages; ``id`` is the
                # function_call output-item ID the Responses API
                # demands on inbound assistant messages. See the
                # outbound translator for why both are needed.
                call_id = getattr(item, "call_id", None) or item.id
                fc_id = getattr(item, "id", None) or call_id
                provider_state: dict[str, Any] = {"openai_fc_id": fc_id}
                # Attach the preceding reasoning item (if any) so the
                # outbound translator can replay it on the next turn
                # — required by the API for reasoning models (vox#25).
                # Consume the buffer so subsequent function_calls in
                # the same response don't replay the same item.
                if last_reasoning_item is not None:
                    provider_state["openai_reasoning_item"] = last_reasoning_item
                    last_reasoning_item = None
                tool_calls.append(
                    ToolCallData(
                        id=call_id,
                        name=item.name,
                        arguments=json.loads(args_str) if args_str else {},
                        provider_state=provider_state,
                    )
                )

            elif item_type == "reasoning":
                # Expose the human-readable summary on ``response.thinking``
                # (existing behavior) AND capture the full item dict for
                # outbound round-tripping (vox#25).
                for summary in getattr(item, "summary", []):
                    if hasattr(summary, "text"):
                        thinking_blocks.append(ThinkingBlock(text=summary.text))
                last_reasoning_item = _serialize_reasoning_item(item)

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

        raw_finish = self._extract_raw_finish_reason(response, has_tool_calls=bool(tool_calls))

        return CompletionResponse(
            message=message,
            usage=usage,
            provider="openai",
            model=model,
            finish_reason=normalize_finish_reason(raw_finish),
            raw_finish_reason=raw_finish,
            thinking=thinking_blocks or None,
            parsed=parsed,
            response_id=getattr(response, "id", None),
        )

    def _extract_raw_finish_reason(self, response: Any, *, has_tool_calls: bool) -> str | None:
        """Derive a finish-reason-like string from a Responses API response.

        The Responses API does not return a single ``finish_reason`` field. We
        synthesize one from ``status`` and ``incomplete_details.reason``:

        - status="incomplete" → use incomplete_details.reason (e.g. "max_output_tokens")
        - status="completed" + tool calls in output → "tool_calls"
        - status="completed" otherwise → "stop"

        Args:
            response: The raw Responses API response object.
            has_tool_calls: Whether the output contained any function_call items.

        Returns:
            A provider-native finish reason string suitable for normalization.
        """
        status = getattr(response, "status", None)
        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", None) if details else None
            return reason or "incomplete"
        if status == "completed":
            return "tool_calls" if has_tool_calls else "stop"
        return status

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
            raise RateLimitError(
                str(e),
                retry_after=_extract_retry_after(e),
                provider="openai",
            ) from e
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
                raise RateLimitError(
                    str(e),
                    retry_after=_extract_retry_after(e),
                    provider="openai",
                ) from e
            raise ProviderError(str(e), provider="openai") from e
        if isinstance(e, openai.APIConnectionError):
            raise ProviderError(str(e), provider="openai") from e

        raise ProviderError(str(e), provider="openai") from e

    # ── Streaming helpers ────────────────────────────────────────────────

    def _process_stream_event(
        self, event: Any, state: dict[str, Any] | None = None
    ) -> list[StreamChunk]:
        """Translate a Responses API stream event to ``StreamChunk``s.

        Args:
            event: A raw SDK stream event.
            state: Mutable per-stream state. Carries:

                * ``item_id_to_call_id`` — map populated when
                  ``response.output_item.added`` arrives for a
                  function_call (the item has both an ``id`` (``fc_*``)
                  and ``call_id`` (``call_*``)). Used to resolve
                  ``function_call_arguments.delta`` events (which only
                  carry ``item_id``) to the ``call_id`` consumers see
                  on the start chunk (vox#20).

        Returns:
            List of ``StreamChunk`` instances (often empty).
        """
        if state is None:
            state = {}
        event_type = getattr(event, "type", None)

        if event_type == "response.output_text.delta":
            return [StreamChunk(type="text", text=event.delta)]

        if event_type == "response.function_call_arguments.delta":
            # The SDK emits ``item_id`` (the fc_* output-item id) on
            # delta events — NOT ``call_id``. vox previously read
            # ``event.call_id`` and got "" on every delta, breaking
            # the consumer-side correlation by tool_call_id (vox#20).
            # Resolve via the buffered item_id → call_id map.
            item_id = getattr(event, "item_id", None) or getattr(event, "call_id", "")
            mapping = state.get("item_id_to_call_id", {})
            tool_call_id = mapping.get(item_id, item_id)
            return [
                StreamChunk(
                    type="tool_call_delta",
                    tool_call_id=tool_call_id,
                    arguments_delta=event.delta,
                )
            ]

        if event_type == "response.output_item.added":
            item = event.item
            if getattr(item, "type", None) == "function_call":
                # ``str(...)`` coercion: getattrs return ``Any``, so the
                # ``or`` chain widens to a nullable in mypy's eyes.
                call_id = str(getattr(item, "call_id", None) or getattr(item, "id", "") or "")
                fc_id = str(getattr(item, "id", None) or call_id)
                # Buffer the fc_* → call_* mapping for upcoming
                # arguments.delta events (vox#20) — those events carry
                # only ``item_id`` (the fc_*), but consumers correlate
                # via the call_* that landed on this start chunk.
                state.setdefault("item_id_to_call_id", {})[fc_id] = call_id
                return [
                    StreamChunk(
                        type="tool_call_start",
                        tool_call=ToolCallData(
                            id=call_id,
                            name=getattr(item, "name", ""),
                            arguments={},
                            # Preserve fc_* so the round-trip outbound
                            # translator can use it as ``input[*].id`` on
                            # subsequent turns (vox#17).
                            provider_state={"openai_fc_id": fc_id},
                        ),
                    )
                ]
            return []

        if event_type == "response.reasoning_summary_text.delta":
            return [StreamChunk(type="thinking", thinking_text=event.delta)]

        if event_type == "response.completed":
            # Previously vox attached ``usage`` to the ``done`` chunk
            # rather than emitting a separate ``type="usage"`` chunk
            # — consumers iterating chunk types never saw a usage
            # chunk (vox#18). Now emit two chunks in order: usage, then
            # done.
            resp = event.response
            results: list[StreamChunk] = []
            if resp.usage:
                results.append(
                    StreamChunk(
                        type="usage",
                        usage=Usage(
                            prompt_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
                            completion_tokens=getattr(resp.usage, "output_tokens", 0) or 0,
                            total_tokens=(
                                (getattr(resp.usage, "input_tokens", 0) or 0)
                                + (getattr(resp.usage, "output_tokens", 0) or 0)
                            ),
                            reasoning_tokens=getattr(resp.usage, "reasoning_tokens", 0) or 0,
                        ),
                    )
                )
            has_tool_calls = any(
                getattr(item, "type", None) == "function_call"
                for item in getattr(resp, "output", []) or []
            )
            raw_finish = self._extract_raw_finish_reason(resp, has_tool_calls=has_tool_calls)
            results.append(
                StreamChunk(
                    type="done",
                    finish_reason=normalize_finish_reason(raw_finish),
                )
            )
            return results

        return []

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
        previous_response_id: str | None = None,
        store: bool | None = None,
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
            stop: Ignored. The Responses API does not support stop sequences.
            previous_response_id: ID of a prior response to chain from for
                stateful multi-turn conversations.
            store: Whether OpenAI persists this response for later chaining
                (defaults to True on OpenAI's side).
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
            previous_response_id=previous_response_id,
            store=store,
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
        tools: Sequence[ToolSpec] | None = None,
        response_schema: type[BaseModel] | None = None,
        reasoning: ReasoningConfig | None = None,
        stop: list[str] | None = None,
        previous_response_id: str | None = None,
        store: bool | None = None,
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
            stop: Ignored. The Responses API does not support stop sequences.
            previous_response_id: ID of a prior response to chain from.
            store: Whether OpenAI persists this response for later chaining.
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
            previous_response_id=previous_response_id,
            store=store,
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
        tools: Sequence[ToolSpec] | None = None,
        reasoning: ReasoningConfig | None = None,
        stop: list[str] | None = None,
        previous_response_id: str | None = None,
        store: bool | None = None,
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
            stop: Ignored. The Responses API does not support stop sequences.
            previous_response_id: ID of a prior response to chain from.
            store: Whether OpenAI persists this response for later chaining.
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
            previous_response_id=previous_response_id,
            store=store,
            stream=True,
            **kwargs,
        )
        # Per-stream state for the event translator: maps function_call
        # item_id (fc_*) → call_id (call_*) so argument-delta events
        # (which only carry item_id) can be tagged with the call_id
        # consumers see on the start chunk (vox#20).
        state: dict[str, Any] = {}
        try:
            response_stream = self._get_sync_client().responses.create(**request)
            for event in response_stream:
                yield from self._process_stream_event(event, state)
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
        tools: Sequence[ToolSpec] | None = None,
        reasoning: ReasoningConfig | None = None,
        stop: list[str] | None = None,
        previous_response_id: str | None = None,
        store: bool | None = None,
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
            stop: Ignored. The Responses API does not support stop sequences.
            previous_response_id: ID of a prior response to chain from.
            store: Whether OpenAI persists this response for later chaining.
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
            previous_response_id=previous_response_id,
            store=store,
            stream=True,
            **kwargs,
        )
        state: dict[str, Any] = {}
        try:
            response_stream = await self._get_async_client().responses.create(**request)
            async for event in response_stream:
                for chunk in self._process_stream_event(event, state):
                    yield chunk
        except Exception as e:
            from ..errors import VoxError

            if isinstance(e, VoxError):
                raise
            self._handle_error(e)
