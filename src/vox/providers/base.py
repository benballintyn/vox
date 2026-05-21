"""Abstract base class for LLM provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

from pydantic import BaseModel

from ..models.config import ProviderConfig
from ..models.messages import Message
from ..models.reasoning import ReasoningConfig
from ..models.responses import CompletionResponse, StreamChunk
from ..models.tools import ToolSpec


class Provider(ABC):
    """Abstract base for all LLM provider adapters.

    Each provider must implement four methods covering the sync/async and
    streaming/non-streaming matrix. All parameters after ``messages`` are
    keyword-only.

    Args:
        config: Provider-specific configuration.
    """

    def __init__(self, config: ProviderConfig | None = None) -> None:
        self.config = config or ProviderConfig()

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        tools: list[ToolSpec] | None = None,
        response_schema: type[BaseModel] | None = None,
        reasoning: ReasoningConfig | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Synchronous completion.

        Args:
            messages: Conversation messages.
            model: Model identifier. Falls back to config default.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            tools: Available tools/functions.
            response_schema: Pydantic model for structured output validation.
            reasoning: Reasoning/thinking configuration.
            stop: Stop sequences.
            **kwargs: Provider-specific passthrough options.

        Returns:
            The completion response.

        Raises:
            VoxError: On any provider error.
        """
        ...

    @abstractmethod
    async def acomplete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
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
            model: Model identifier. Falls back to config default.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            tools: Available tools/functions.
            response_schema: Pydantic model for structured output validation.
            reasoning: Reasoning/thinking configuration.
            stop: Stop sequences.
            **kwargs: Provider-specific passthrough options.

        Returns:
            The completion response.

        Raises:
            VoxError: On any provider error.
        """
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
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
            model: Model identifier. Falls back to config default.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            tools: Available tools/functions.
            reasoning: Reasoning/thinking configuration.
            stop: Stop sequences.
            **kwargs: Provider-specific passthrough options.

        Yields:
            StreamChunk instances as they arrive from the provider.
        """
        ...

    @abstractmethod
    async def astream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
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
            model: Model identifier. Falls back to config default.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            tools: Available tools/functions.
            reasoning: Reasoning/thinking configuration.
            stop: Stop sequences.
            **kwargs: Provider-specific passthrough options.

        Yields:
            StreamChunk instances as they arrive from the provider.
        """
        ...

    def _resolve_model(self, model: str | None) -> str:
        """Resolve model name from argument or config default.

        Args:
            model: Explicitly provided model name.

        Returns:
            The resolved model name.

        Raises:
            InvalidRequestError: If no model is specified and no default is configured.
        """
        from ..errors import InvalidRequestError

        resolved = model or self.config.default_model
        if not resolved:
            raise InvalidRequestError(
                "No model specified and no default_model configured.",
                provider=self.provider_name,
            )
        return resolved

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the canonical name of this provider (e.g. 'openai', 'anthropic')."""
        ...
