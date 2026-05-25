"""Google Gemini provider using the google-genai SDK."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from pydantic import BaseModel

from .._structured import (
    pydantic_to_gemini_schema,
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
from ..models.reasoning import (
    LEVEL_TO_BUDGET_TOKENS,
    LEVEL_TO_GEMINI3_LEVEL,
    ReasoningConfig,
    ThinkingBlock,
)
from ..models.responses import (
    CompletionResponse,
    StreamChunk,
    Usage,
    normalize_finish_reason,
)
from ..models.tools import TOOL_SPEC_TYPE_ERROR, Tool, ToolSpec
from .base import Provider


def _import_genai() -> Any:
    """Lazily import the google-genai package."""
    try:
        from google import genai

        return genai
    except ImportError:
        raise ImportError(
            "The google-genai package is required for the Gemini provider. "
            "Install it with: pip install vox[gemini]"
        ) from None


def _import_genai_types() -> Any:
    """Lazily import google-genai types."""
    try:
        from google.genai import types

        return types
    except ImportError:
        raise ImportError(
            "The google-genai package is required for the Gemini provider. "
            "Install it with: pip install vox[gemini]"
        ) from None


class GeminiProvider(Provider):
    """Google Gemini provider using the google-genai SDK.

    Uses native async support via ``client.aio``.

    Args:
        config: Provider configuration.
    """

    def __init__(self, config: ProviderConfig | None = None) -> None:
        super().__init__(config)
        self._client: Any = None

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _get_api_key(self) -> str | None:
        """Resolve API key from config or environment."""
        return self.config.api_key or os.environ.get("GOOGLE_API_KEY")

    def _get_client(self) -> Any:
        """Get or create the google-genai client."""
        if self._client is None:
            genai = _import_genai()
            self._client = genai.Client(api_key=self._get_api_key())
        return self._client

    # ── Message translation ──────────────────────────────────────────────

    def _translate_contents(self, messages: list[Message]) -> tuple[list[Any], str | None]:
        """Translate vox Messages to Gemini content format.

        System messages are extracted for the ``system_instruction`` parameter.

        Args:
            messages: List of vox Message objects.

        Returns:
            Tuple of (contents list, system instruction or None).
        """
        types = _import_genai_types()
        system_instruction = None
        contents: list[Any] = []

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.text
                continue

            if msg.role == "tool":
                # Tool result as FunctionResponse
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=msg.name or "unknown",
                                    response={"result": msg.text},
                                )
                            )
                        ],
                    )
                )
                continue

            role = "model" if msg.role == "assistant" else "user"
            parts = self._translate_parts(msg)

            # Add tool call parts for assistant messages
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                name=tc.name,
                                args=tc.arguments,
                            )
                        )
                    )

            contents.append(types.Content(role=role, parts=parts))

        return contents, system_instruction

    def _translate_parts(self, msg: Message) -> list[Any]:
        """Translate message content to Gemini Part objects.

        Args:
            msg: A vox Message.

        Returns:
            List of genai Part objects.
        """
        types = _import_genai_types()

        if isinstance(msg.content, str):
            if msg.content:
                return [types.Part(text=msg.content)]
            return []

        parts: list[Any] = []
        for part in msg.content:
            if isinstance(part, TextContent):
                parts.append(types.Part(text=part.text))
            elif isinstance(part, ImageContent):
                if part.source_type == "url":
                    parts.append(
                        types.Part(
                            file_data=types.FileData(
                                file_uri=part.data,
                                mime_type=part.media_type,
                            )
                        )
                    )
                else:
                    import base64

                    parts.append(
                        types.Part(
                            inline_data=types.Blob(
                                mime_type=part.media_type,
                                data=base64.b64decode(part.data),
                            )
                        )
                    )
        return parts

    def _translate_tools(self, tools: Sequence[ToolSpec]) -> list[Any]:
        """Translate tool specs to Gemini tool objects.

        vox ``Tool`` objects are collected into a single
        ``types.Tool(function_declarations=[...])``. Raw dicts are passed
        through verbatim as separate entries in the tools list — the escape
        hatch for Gemini-native tools such as Google Search grounding
        (``{"google_search": {}}``) or code execution.

        Note: Gemini's tool structure differs from OpenAI/Anthropic — it has
        no flat per-tool dict shape. A pass-through dict must already match
        what the google-genai SDK expects for a ``config.tools`` entry; vox
        does not reshape it. Malformed dicts surface as a request-time error
        from the SDK.

        Args:
            tools: List of vox Tool objects and/or raw provider-native dicts.

        Returns:
            List of genai tool entries.

        Raises:
            TypeError: If an entry is neither a vox Tool nor a dict.
        """
        types = _import_genai_types()
        declarations = []
        native: list[Any] = []
        for t in tools:
            if isinstance(t, dict):
                native.append(t)
            elif isinstance(t, Tool):
                declarations.append(
                    types.FunctionDeclaration(
                        name=t.name,
                        description=t.description,
                        parameters=t.parameters,
                    )
                )
            else:
                raise TypeError(TOOL_SPEC_TYPE_ERROR.format(got=type(t).__name__))

        result: list[Any] = []
        if declarations:
            result.append(types.Tool(function_declarations=declarations))
        result.extend(native)
        return result

    # ── Build request config ─────────────────────────────────────────────

    def _build_generate_config(
        self,
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        response_schema: type[BaseModel] | None,
        reasoning: ReasoningConfig | None,
        stop: list[str] | None,
        **kwargs: Any,
    ) -> Any:
        """Build a GenerateContentConfig for the request.

        Args:
            model: Model identifier (used to detect Gemini 2.5 vs 3 for
                thinking config translation).
            max_tokens: Max output tokens.
            temperature: Sampling temperature.
            response_schema: Pydantic model for structured output.
            reasoning: Reasoning config.
            stop: Stop sequences.
            **kwargs: Passthrough.

        Returns:
            A GenerateContentConfig instance.
        """
        types = _import_genai_types()

        config_kwargs: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

        if stop:
            config_kwargs["stop_sequences"] = stop

        if response_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = pydantic_to_gemini_schema(response_schema)

        if reasoning and reasoning.enabled:
            thinking_config = self._build_thinking_config(reasoning, model)
            if thinking_config:
                config_kwargs["thinking_config"] = thinking_config

        config_kwargs.update(kwargs)
        return types.GenerateContentConfig(**config_kwargs)

    def _build_thinking_config(self, reasoning: ReasoningConfig, model: str) -> Any | None:
        """Build Gemini thinking config from ReasoningConfig.

        Translation rules (in priority order):
            1. ``reasoning.gemini`` overrides — use exactly what the user provides.
            2. ``reasoning.level`` semantic mapping:
               - Gemini 3+ models: map level → ``thinking_level``.
               - Gemini 2.5 (and others): map level → ``thinking_budget``.

        Args:
            reasoning: The reasoning configuration.
            model: Model identifier for version detection.

        Returns:
            A ThinkingConfig instance or None.
        """
        types = _import_genai_types()
        is_gemini3 = model.startswith("gemini-3")

        # Provider-specific overrides take priority
        if reasoning.gemini:
            cfg_kwargs: dict[str, Any] = {}
            if reasoning.gemini.budget_tokens is not None:
                cfg_kwargs["thinking_budget"] = reasoning.gemini.budget_tokens
            if reasoning.gemini.level is not None:
                cfg_kwargs["thinking_level"] = reasoning.gemini.level.upper()
            if cfg_kwargs:
                return types.ThinkingConfig(**cfg_kwargs)

        if reasoning.level:
            if is_gemini3:
                mapped = LEVEL_TO_GEMINI3_LEVEL[reasoning.level]
                return types.ThinkingConfig(thinking_level=mapped.upper())
            return types.ThinkingConfig(thinking_budget=LEVEL_TO_BUDGET_TOKENS[reasoning.level])

        return None

    # ── Response translation ─────────────────────────────────────────────

    def _translate_response(
        self,
        response: Any,
        model: str,
        response_schema: type[BaseModel] | None = None,
    ) -> CompletionResponse:
        """Translate a Gemini response to vox CompletionResponse.

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

        if response.candidates:
            candidate = response.candidates[0]
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    if getattr(part, "thought", False):
                        thinking_blocks.append(ThinkingBlock(text=part.text))
                    else:
                        text_parts.append(part.text)
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append(
                        ToolCallData(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                        )
                    )

        message = Message(
            role="assistant",
            content="".join(text_parts),
            tool_calls=tool_calls or None,
        )

        usage = Usage()
        if response.usage_metadata:
            um = response.usage_metadata
            usage = Usage(
                prompt_tokens=getattr(um, "prompt_token_count", 0) or 0,
                completion_tokens=getattr(um, "candidates_token_count", 0) or 0,
                total_tokens=getattr(um, "total_token_count", 0) or 0,
                reasoning_tokens=getattr(um, "thinking_token_count", 0) or 0,
            )

        parsed = None
        if response_schema and text_parts:
            parsed = validate_structured_response(response_schema, "".join(text_parts))

        raw_finish = None
        if response.candidates:
            fr = getattr(response.candidates[0], "finish_reason", None)
            if fr:
                # Gemini's finish_reason is an enum; normalize to its string name.
                raw_finish = getattr(fr, "name", str(fr)).lower()

        # Gemini reports ``STOP`` even when the response contains tool
        # calls — unlike OpenAI (``tool_calls``) and Anthropic
        # (``tool_use``). Infer the normalized ``tool_calls`` reason from
        # the parsed message so consumers can switch on a single field
        # cross-provider.
        normalized_finish = normalize_finish_reason(raw_finish)
        if tool_calls and normalized_finish == "stop":
            normalized_finish = "tool_calls"

        return CompletionResponse(
            message=message,
            usage=usage,
            provider="gemini",
            model=model,
            finish_reason=normalized_finish,
            raw_finish_reason=raw_finish,
            thinking=thinking_blocks or None,
            parsed=parsed,
            response_id=getattr(response, "response_id", None),
        )

    # ── Error handling ───────────────────────────────────────────────────

    def _handle_error(self, e: Exception) -> None:
        """Translate google-genai exceptions to vox errors.

        Args:
            e: The caught exception.

        Raises:
            VoxError: The appropriate vox error subclass.
        """
        msg = str(e).lower()

        if "api key" in msg or "authentication" in msg or "unauthorized" in msg:
            raise AuthenticationError(str(e), provider="gemini") from e
        if "rate limit" in msg or "resource exhausted" in msg or "429" in msg:
            raise RateLimitError(str(e), provider="gemini") from e
        if "quota" in msg or "billing" in msg:
            raise QuotaExceededError(str(e), provider="gemini") from e
        if "not found" in msg or "404" in msg:
            raise ModelNotFoundError(str(e), provider="gemini") from e
        if "invalid" in msg or "bad request" in msg or "400" in msg:
            raise InvalidRequestError(str(e), provider="gemini") from e
        if "safety" in msg or "blocked" in msg:
            raise ContentFilterError(str(e), provider="gemini") from e

        raise ProviderError(str(e), provider="gemini") from e

    # ── Stream translation ───────────────────────────────────────────────

    def _translate_stream_chunk(self, chunk: Any) -> list[StreamChunk]:
        """Translate a Gemini stream chunk to StreamChunks.

        Args:
            chunk: A raw SDK stream chunk.

        Returns:
            List of StreamChunk instances (may be empty).
        """
        results: list[StreamChunk] = []

        if not chunk.candidates:
            return results

        candidate = chunk.candidates[0]

        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    if getattr(part, "thought", False):
                        results.append(StreamChunk(type="thinking", thinking_text=part.text))
                    else:
                        results.append(StreamChunk(type="text", text=part.text))
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    results.append(
                        StreamChunk(
                            type="tool_call_start",
                            tool_call=ToolCallData(
                                id=f"call_{uuid.uuid4().hex[:8]}",
                                name=fc.name,
                                arguments=dict(fc.args) if fc.args else {},
                            ),
                        )
                    )

        # Check for finish reason
        fr = getattr(candidate, "finish_reason", None)
        if fr:
            raw = getattr(fr, "name", str(fr)).lower()
            normalized = normalize_finish_reason(raw)
            # Gemini reports ``STOP`` even when the same chunk includes a
            # function call. Mirror the non-streaming path's inference:
            # if we just emitted a ``tool_call_start`` in this chunk and
            # the normalized reason would otherwise be ``stop``, surface
            # ``tool_calls`` instead — same cross-provider contract.
            if normalized == "stop" and any(r.type == "tool_call_start" for r in results):
                normalized = "tool_calls"
            results.append(StreamChunk(type="done", finish_reason=normalized))

        return results

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
        """Synchronous Gemini generate_content call.

        Args:
            messages: Conversation messages.
            model: Model identifier (default from config).
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
        contents, system_instruction = self._translate_contents(messages)
        config = self._build_generate_config(
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_schema=response_schema,
            reasoning=reasoning,
            stop=stop,
            **kwargs,
        )

        gen_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "contents": contents,
            "config": config,
        }
        if system_instruction:
            gen_kwargs["config"].system_instruction = system_instruction
        if tools:
            gen_kwargs["config"].tools = self._translate_tools(tools)

        try:
            response = self._get_client().models.generate_content(**gen_kwargs)
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
        **kwargs: Any,
    ) -> CompletionResponse:
        """Asynchronous Gemini generate_content call.

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
        contents, system_instruction = self._translate_contents(messages)
        config = self._build_generate_config(
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_schema=response_schema,
            reasoning=reasoning,
            stop=stop,
            **kwargs,
        )

        gen_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "contents": contents,
            "config": config,
        }
        if system_instruction:
            gen_kwargs["config"].system_instruction = system_instruction
        if tools:
            gen_kwargs["config"].tools = self._translate_tools(tools)

        try:
            response = await self._get_client().aio.models.generate_content(**gen_kwargs)
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
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Synchronous streaming Gemini call.

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
            StreamChunk instances as they arrive.
        """
        resolved_model = self._resolve_model(model)
        contents, system_instruction = self._translate_contents(messages)
        # ``model=`` is required by ``_build_generate_config`` so it can
        # detect Gemini 2.5 vs 3 for thinking-config translation. Missing
        # here previously meant ``stream()`` and ``astream()`` raised
        # TypeError before ever reaching the API — caught by the
        # integration suite, since mocked unit tests stub at a different
        # layer.
        config = self._build_generate_config(
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_schema=None,
            reasoning=reasoning,
            stop=stop,
            **kwargs,
        )

        gen_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "contents": contents,
            "config": config,
        }
        if system_instruction:
            gen_kwargs["config"].system_instruction = system_instruction
        if tools:
            gen_kwargs["config"].tools = self._translate_tools(tools)

        try:
            for chunk in self._get_client().models.generate_content_stream(**gen_kwargs):
                yield from self._translate_stream_chunk(chunk)
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
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Asynchronous streaming Gemini call.

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
            StreamChunk instances as they arrive.
        """
        resolved_model = self._resolve_model(model)
        contents, system_instruction = self._translate_contents(messages)
        # ``model=`` is required by ``_build_generate_config`` so it can
        # detect Gemini 2.5 vs 3 for thinking-config translation. Missing
        # here previously meant ``stream()`` and ``astream()`` raised
        # TypeError before ever reaching the API — caught by the
        # integration suite, since mocked unit tests stub at a different
        # layer.
        config = self._build_generate_config(
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_schema=None,
            reasoning=reasoning,
            stop=stop,
            **kwargs,
        )

        gen_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "contents": contents,
            "config": config,
        }
        if system_instruction:
            gen_kwargs["config"].system_instruction = system_instruction
        if tools:
            gen_kwargs["config"].tools = self._translate_tools(tools)

        try:
            async for chunk in await self._get_client().aio.models.generate_content_stream(
                **gen_kwargs
            ):
                for sc in self._translate_stream_chunk(chunk):
                    yield sc
        except Exception as e:
            from ..errors import VoxError

            if isinstance(e, VoxError):
                raise
            self._handle_error(e)
