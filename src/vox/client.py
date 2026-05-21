"""VoxClient facade — the primary entry point for using vox."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from pydantic import BaseModel

from ._registry import resolve_provider
from .errors import InvalidRequestError
from .models.config import ProviderConfig
from .models.messages import Message
from .models.reasoning import ReasoningConfig
from .models.responses import CompletionResponse, StreamChunk
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
        tools: list[ToolSpec] | None = None,
        response_schema: type[BaseModel] | None = None,
        reasoning: ReasoningConfig | None = None,
        stop: list[str] | None = None,
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
            **kwargs: Provider-specific passthrough.

        Returns:
            CompletionResponse with the model's reply.
        """
        resolved = resolve_provider(model, provider)
        adapter = self._get_provider(resolved)
        return adapter.complete(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            response_schema=response_schema,
            reasoning=reasoning,
            stop=stop,
            **kwargs,
        )

    async def acomplete(
        self,
        messages: list[Message],
        *,
        model: str,
        provider: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        tools: list[ToolSpec] | None = None,
        response_schema: type[BaseModel] | None = None,
        reasoning: ReasoningConfig | None = None,
        stop: list[str] | None = None,
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
            **kwargs: Provider-specific passthrough.

        Returns:
            CompletionResponse with the model's reply.
        """
        resolved = resolve_provider(model, provider)
        adapter = self._get_provider(resolved)
        return await adapter.acomplete(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            response_schema=response_schema,
            reasoning=reasoning,
            stop=stop,
            **kwargs,
        )

    def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        provider: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        tools: list[ToolSpec] | None = None,
        reasoning: ReasoningConfig | None = None,
        stop: list[str] | None = None,
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
            **kwargs: Provider-specific passthrough.

        Yields:
            StreamChunk instances as they arrive.
        """
        resolved = resolve_provider(model, provider)
        adapter = self._get_provider(resolved)
        yield from adapter.stream(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            reasoning=reasoning,
            stop=stop,
            **kwargs,
        )

    async def astream(
        self,
        messages: list[Message],
        *,
        model: str,
        provider: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        tools: list[ToolSpec] | None = None,
        reasoning: ReasoningConfig | None = None,
        stop: list[str] | None = None,
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
            **kwargs: Provider-specific passthrough.

        Yields:
            StreamChunk instances as they arrive.
        """
        resolved = resolve_provider(model, provider)
        adapter = self._get_provider(resolved)
        async for chunk in adapter.astream(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            reasoning=reasoning,
            stop=stop,
            **kwargs,
        ):
            yield chunk
