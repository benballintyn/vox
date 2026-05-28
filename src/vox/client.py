"""VoxClient facade — the primary entry point for using vox."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Annotation-only imports — VoxClient just forwards these between
    # the caller and the provider, so we never instantiate them here.
    # Keeps the formatter from stripping the imports between Edits.
    from .models.messages import AudioContent
    from .models.responses import TranscriptionResponse

from pydantic import BaseModel

from ._pricing import ModelPricing, estimate_cost
from ._registry import resolve_provider
from ._retry import (
    RetryPolicy,
    retry_async,
    retry_stream_async,
    retry_stream_sync,
    retry_sync,
)
from .errors import InvalidRequestError
from .models.config import ProviderConfig
from .models.messages import Message
from .models.reasoning import ReasoningConfig
from .models.responses import CompletionResponse, StreamChunk, Usage
from .models.tools import ToolSpec
from .providers.base import Provider


class VoxClient:
    """Model-agnostic LLM client.

    Resolves the appropriate provider from the model name and delegates
    requests. Providers are instantiated lazily and cached.

    Args:
        openai_api_key: API key for OpenAI.
        anthropic_api_key: API key for Anthropic.
        gemini_api_key: API key for Google Gemini.
        openrouter_api_key: API key for OpenRouter.
        lmstudio_base_url: Base URL for LM Studio (default: http://localhost:1234/v1).
        openrouter_app_name: App name for OpenRouter's X-Title header.
        openrouter_app_url: App URL for OpenRouter's HTTP-Referer header.
        provider_configs: Per-provider ProviderConfig overrides keyed by provider name.
        custom_pricing: Per-model pricing overrides keyed by model id.
            Entries here take precedence over vox's built-in price
            snapshot when computing ``usage.estimated_cost``. Pass a
            ``ModelPricing(...)`` for any model you want priced
            differently (or for models vox doesn't know about at all).
        retry_policy: Default retry behaviour applied to every call.
            Per-call ``retry_policy=`` overrides this. ``None`` means
            vox's default policy (3 retries with exponential backoff,
            honouring ``RateLimitError.retry_after`` — see
            :class:`RetryPolicy`).
    """

    def __init__(
        self,
        *,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        gemini_api_key: str | None = None,
        openrouter_api_key: str | None = None,
        lmstudio_base_url: str = "http://localhost:1234/v1",
        openrouter_app_name: str | None = None,
        openrouter_app_url: str | None = None,
        provider_configs: dict[str, ProviderConfig] | None = None,
        custom_pricing: dict[str, ModelPricing] | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._provider_configs = provider_configs or {}
        self._api_keys = {
            "openai": openai_api_key,
            "anthropic": anthropic_api_key,
            "gemini": gemini_api_key,
            "openrouter": openrouter_api_key,
        }
        self._lmstudio_base_url = lmstudio_base_url
        self._openrouter_app_name = openrouter_app_name
        self._openrouter_app_url = openrouter_app_url
        self._providers: dict[str, Provider] = {}
        self._custom_pricing: dict[str, ModelPricing] = custom_pricing or {}
        self._default_retry_policy = retry_policy or RetryPolicy()

    def _resolve_retry_policy(self, override: RetryPolicy | None) -> RetryPolicy:
        """Pick the per-call policy if provided, else the client default."""
        return override if override is not None else self._default_retry_policy

    def _populate_cost(self, usage: Usage | None, model: str) -> None:
        """Annotate a ``Usage`` in place with ``model`` + ``estimated_cost``.

        Used after the provider returns to add the pricing-derived
        fields without changing the provider's contract — providers
        stay ignorant of pricing; ``VoxClient`` is the integration
        point. ``None``-tolerant for the rare case a provider returns
        no usage (Usage is currently a required field on
        CompletionResponse, so this is defensive).
        """
        if usage is None:
            return
        usage.model = model
        usage.estimated_cost = estimate_cost(usage, model, self._custom_pricing)

    def _get_provider(self, name: str) -> Provider:
        """Get or create a provider by name.

        Args:
            name: The provider name (e.g. 'openai', 'anthropic').

        Returns:
            The Provider instance.

        Raises:
            InvalidRequestError: If the provider name is not recognized.
        """
        if name in self._providers:
            return self._providers[name]

        provider = self._create_provider(name)
        self._providers[name] = provider
        return provider

    def _create_provider(self, name: str) -> Provider:
        """Instantiate a provider by name.

        Args:
            name: The provider name.

        Returns:
            A new Provider instance.

        Raises:
            InvalidRequestError: If the provider name is not recognized.
        """
        config = self._provider_configs.get(name, ProviderConfig())

        # Apply API key from constructor if not already in config
        if not config.api_key and name in self._api_keys:
            config = config.model_copy(update={"api_key": self._api_keys[name]})

        if name == "openai":
            from .providers.openai import OpenAIProvider

            return OpenAIProvider(config)

        if name == "anthropic":
            from .providers.anthropic import AnthropicProvider

            return AnthropicProvider(config)

        if name == "gemini":
            from .providers.gemini import GeminiProvider

            return GeminiProvider(config)

        if name == "openrouter":
            if not config.app_name and self._openrouter_app_name:
                config = config.model_copy(update={"app_name": self._openrouter_app_name})
            if not config.app_url and self._openrouter_app_url:
                config = config.model_copy(update={"app_url": self._openrouter_app_url})
            from .providers.openrouter import OpenRouterProvider

            return OpenRouterProvider(config)

        if name == "lmstudio":
            if not config.base_url:
                config = config.model_copy(update={"base_url": self._lmstudio_base_url})
            from .providers.lmstudio import LMStudioProvider

            return LMStudioProvider(config)

        raise InvalidRequestError(
            f"Unknown provider '{name}'. "
            "Available: openai, anthropic, gemini, openrouter, lmstudio"
        )

    # ── Public API ───────────────────────────────────────────────────────

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        provider: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        tools: Sequence[ToolSpec] | None = None,
        response_schema: type[BaseModel] | None = None,
        reasoning: ReasoningConfig | None = None,
        stop: list[str] | None = None,
        retry_policy: RetryPolicy | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Synchronous completion.

        Resolves the provider from the model name and delegates.

        Args:
            messages: Conversation messages.
            model: Model identifier (e.g. 'gpt-4o', 'claude-sonnet-4-20250514').
            provider: Explicit provider override (e.g. 'openrouter').
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            tools: Available tools.
            response_schema: Pydantic model for structured output.
            reasoning: Reasoning configuration.
            stop: Stop sequences.
            retry_policy: Per-call retry override. Defaults to the
                client-level policy from the constructor.
            **kwargs: Provider-specific passthrough.

        Returns:
            CompletionResponse with the model's reply.
        """
        resolved = resolve_provider(model, provider)
        adapter = self._get_provider(resolved)
        policy = self._resolve_retry_policy(retry_policy)
        response = retry_sync(
            lambda: adapter.complete(
                messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                response_schema=response_schema,
                reasoning=reasoning,
                stop=stop,
                **kwargs,
            ),
            policy=policy,
        )
        self._populate_cost(response.usage, model)
        return response

    async def acomplete(
        self,
        messages: list[Message],
        *,
        model: str,
        provider: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        tools: Sequence[ToolSpec] | None = None,
        response_schema: type[BaseModel] | None = None,
        reasoning: ReasoningConfig | None = None,
        stop: list[str] | None = None,
        retry_policy: RetryPolicy | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Asynchronous completion.

        Args:
            messages: Conversation messages.
            model: Model identifier.
            provider: Explicit provider override.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            tools: Available tools.
            response_schema: Pydantic model for structured output.
            reasoning: Reasoning configuration.
            stop: Stop sequences.
            retry_policy: Per-call retry override.
            **kwargs: Provider-specific passthrough.

        Returns:
            CompletionResponse with the model's reply.
        """
        resolved = resolve_provider(model, provider)
        adapter = self._get_provider(resolved)
        policy = self._resolve_retry_policy(retry_policy)
        response = await retry_async(
            lambda: adapter.acomplete(
                messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                response_schema=response_schema,
                reasoning=reasoning,
                stop=stop,
                **kwargs,
            ),
            policy=policy,
        )
        self._populate_cost(response.usage, model)
        return response

    def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        provider: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        tools: Sequence[ToolSpec] | None = None,
        reasoning: ReasoningConfig | None = None,
        stop: list[str] | None = None,
        retry_policy: RetryPolicy | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Synchronous streaming completion.

        Args:
            messages: Conversation messages.
            model: Model identifier.
            provider: Explicit provider override.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            tools: Available tools.
            reasoning: Reasoning configuration.
            stop: Stop sequences.
            retry_policy: Per-call retry override. Retries only fire
                before the first chunk yields — once data has started
                arriving, errors propagate as-is.
            **kwargs: Provider-specific passthrough.

        Yields:
            StreamChunk instances as they arrive.
        """
        resolved = resolve_provider(model, provider)
        adapter = self._get_provider(resolved)
        policy = self._resolve_retry_policy(retry_policy)

        def _open_stream() -> Iterator[StreamChunk]:
            return iter(
                adapter.stream(
                    messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tools=tools,
                    reasoning=reasoning,
                    stop=stop,
                    **kwargs,
                )
            )

        for chunk in retry_stream_sync(_open_stream, policy=policy):
            # Annotate the usage chunk with model + estimated_cost so
            # streaming consumers get the same telemetry as non-streamers.
            if chunk.type == "usage":
                self._populate_cost(chunk.usage, model)
            yield chunk

    async def astream(
        self,
        messages: list[Message],
        *,
        model: str,
        provider: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        tools: Sequence[ToolSpec] | None = None,
        reasoning: ReasoningConfig | None = None,
        stop: list[str] | None = None,
        retry_policy: RetryPolicy | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Asynchronous streaming completion.

        Args:
            messages: Conversation messages.
            model: Model identifier.
            provider: Explicit provider override.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            tools: Available tools.
            reasoning: Reasoning configuration.
            stop: Stop sequences.
            retry_policy: Per-call retry override. See :meth:`stream`.
            **kwargs: Provider-specific passthrough.

        Yields:
            StreamChunk instances as they arrive.
        """
        resolved = resolve_provider(model, provider)
        adapter = self._get_provider(resolved)
        policy = self._resolve_retry_policy(retry_policy)

        def _open_stream() -> AsyncIterator[StreamChunk]:
            return adapter.astream(
                messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                reasoning=reasoning,
                stop=stop,
                **kwargs,
            )

        async for chunk in retry_stream_async(_open_stream, policy=policy):
            if chunk.type == "usage":
                self._populate_cost(chunk.usage, model)
            yield chunk

    # ── Audio: transcribe + synthesize ────────────────────────────────

    def transcribe(
        self,
        audio: AudioContent,
        *,
        model: str,
        provider: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
        retry_policy: RetryPolicy | None = None,
        **kwargs: Any,
    ) -> TranscriptionResponse:
        """Synchronously transcribe audio to text.

        Resolves the provider from the model name (same dispatch logic
        as :meth:`complete`) and delegates. Providers without native
        STT raise :class:`InvalidRequestError`.

        Args:
            audio: The audio to transcribe.
            model: STT model identifier (e.g. ``"whisper-1"``,
                ``"gpt-4o-transcribe"``, ``"gemini-3.5-flash"``).
            provider: Explicit provider override (``"openai"`` /
                ``"gemini"``). Auto-resolved from the model name otherwise.
            language: Optional ISO-639-1 hint. OpenAI Whisper uses it;
                Gemini ignores.
            prompt: Optional bias string. OpenAI Whisper uses it as a
                vocab-bias prompt; Gemini uses it as the transcription
                instruction itself (overriding the default).
            retry_policy: Per-call retry override.
            **kwargs: Provider-specific passthrough.

        Returns:
            A :class:`TranscriptionResponse`.
        """
        resolved = resolve_provider(model, provider)
        adapter = self._get_provider(resolved)
        policy = self._resolve_retry_policy(retry_policy)
        response = retry_sync(
            lambda: adapter.transcribe(
                audio,
                model=model,
                language=language,
                prompt=prompt,
                **kwargs,
            ),
            policy=policy,
        )
        if response.usage is not None:
            self._populate_cost(response.usage, model)
        return response

    async def atranscribe(
        self,
        audio: AudioContent,
        *,
        model: str,
        provider: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
        retry_policy: RetryPolicy | None = None,
        **kwargs: Any,
    ) -> TranscriptionResponse:
        """Asynchronously transcribe audio to text. See :meth:`transcribe`."""
        resolved = resolve_provider(model, provider)
        adapter = self._get_provider(resolved)
        policy = self._resolve_retry_policy(retry_policy)
        response = await retry_async(
            lambda: adapter.atranscribe(
                audio,
                model=model,
                language=language,
                prompt=prompt,
                **kwargs,
            ),
            policy=policy,
        )
        if response.usage is not None:
            self._populate_cost(response.usage, model)
        return response

    def synthesize(
        self,
        text: str,
        *,
        voice: str,
        model: str,
        provider: str | None = None,
        format: str | None = None,
        speed: float | None = None,
        instructions: str | None = None,
        retry_policy: RetryPolicy | None = None,
        **kwargs: Any,
    ) -> AudioContent:
        """Synchronously synthesize text to speech.

        Args:
            text: Text to speak.
            voice: Voice name. Provider-specific — see
                ``vox.providers.openai.OPENAI_TTS_VOICES`` and
                ``vox.providers.gemini.GEMINI_TTS_VOICES``.
            model: TTS model identifier.
            provider: Explicit provider override.
            format: Output format (``"mp3"`` / ``"wav"`` / ``"opus"`` /
                ``"aac"`` / ``"flac"`` / ``"pcm"``). OpenAI honours
                this; Gemini always emits PCM-wrapped-as-WAV.
            speed: Playback speed (0.25-4.0). OpenAI only.
            instructions: Voice direction prompt (``gpt-4o-mini-tts``
                and newer only).
            retry_policy: Per-call retry override.
            **kwargs: Provider-specific passthrough.

        Returns:
            An :class:`AudioContent` containing the synthesized audio.
        """
        resolved = resolve_provider(model, provider)
        adapter = self._get_provider(resolved)
        policy = self._resolve_retry_policy(retry_policy)
        return retry_sync(
            lambda: adapter.synthesize(
                text,
                voice=voice,
                model=model,
                format=format,
                speed=speed,
                instructions=instructions,
                **kwargs,
            ),
            policy=policy,
        )

    async def asynthesize(
        self,
        text: str,
        *,
        voice: str,
        model: str,
        provider: str | None = None,
        format: str | None = None,
        speed: float | None = None,
        instructions: str | None = None,
        retry_policy: RetryPolicy | None = None,
        **kwargs: Any,
    ) -> AudioContent:
        """Asynchronously synthesize text to speech. See :meth:`synthesize`."""
        resolved = resolve_provider(model, provider)
        adapter = self._get_provider(resolved)
        policy = self._resolve_retry_policy(retry_policy)
        return await retry_async(
            lambda: adapter.asynthesize(
                text,
                voice=voice,
                model=model,
                format=format,
                speed=speed,
                instructions=instructions,
                **kwargs,
            ),
            policy=policy,
        )
