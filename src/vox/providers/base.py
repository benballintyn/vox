"""Abstract base class for LLM provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, NoReturn

from pydantic import BaseModel

from ..models.config import ProviderConfig
from ..models.messages import AudioContent, Message
from ..models.reasoning import ReasoningConfig
from ..models.responses import CompletionResponse, StreamChunk, TranscriptionResponse
from ..models.tools import ToolSpec

# Names of providers that ship native audio support. Referenced in the
# default-unsupported error message so consumers know where to route.
_AUDIO_SUPPORTED_PROVIDERS = ("openai", "gemini")


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
        tools: Sequence[ToolSpec] | None = None,
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
        tools: Sequence[ToolSpec] | None = None,
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
        tools: Sequence[ToolSpec] | None = None,
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

    # NOTE: declared as a plain ``def`` (not ``async def``) returning an
    # AsyncIterator. Subclasses implement it as an async generator (``async
    # def`` with ``yield``), which satisfies this signature. Declaring the
    # abstract method ``async def`` would type it as returning a Coroutine
    # wrapping the iterator, which the async-generator overrides do not match.
    @abstractmethod
    def astream(
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

    # ── Audio methods ──────────────────────────────────────────────────
    # Not abstract. Subclasses with native audio support (OpenAI, Gemini)
    # override these. Everyone else inherits the default which raises a
    # uniform InvalidRequestError pointing at the providers that do
    # support audio.

    def transcribe(
        self,
        audio: AudioContent,
        *,
        model: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> TranscriptionResponse:
        """Synchronously transcribe audio to text.

        Default behavior: raise :class:`InvalidRequestError`. Providers
        with native STT (OpenAI Whisper, Gemini) override.

        Args:
            audio: The audio to transcribe.
            model: STT model identifier (e.g. ``"whisper-1"``,
                ``"gpt-4o-transcribe"``, ``"gemini-3.5-flash"``).
            language: Optional ISO-639-1 language code to bias the
                model. OpenAI Whisper uses this; Gemini ignores it.
            prompt: Optional context string the STT model uses to bias
                its output (proper nouns, domain vocab). OpenAI
                Whisper only.
            **kwargs: Provider-specific passthrough.

        Returns:
            A :class:`TranscriptionResponse`.

        Raises:
            InvalidRequestError: When the provider has no native STT.
        """
        self._raise_audio_unsupported("transcribe")

    async def atranscribe(
        self,
        audio: AudioContent,
        *,
        model: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> TranscriptionResponse:
        """Asynchronously transcribe audio to text. See :meth:`transcribe`."""
        self._raise_audio_unsupported("transcribe")

    def synthesize(
        self,
        text: str,
        *,
        voice: str,
        model: str | None = None,
        format: str | None = None,
        speed: float | None = None,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> AudioContent:
        """Synchronously synthesize text to speech.

        Default behavior: raise :class:`InvalidRequestError`. Providers
        with native TTS (OpenAI, Gemini) override.

        Args:
            text: The text to speak.
            voice: Voice name. Provider-specific values; see
                ``OPENAI_TTS_VOICES`` / ``GEMINI_TTS_VOICES``.
            model: TTS model identifier (e.g. ``"tts-1"``,
                ``"gpt-4o-mini-tts"``, ``"gemini-3.1-flash-tts-preview"``).
            format: Output audio format. OpenAI: ``"mp3"`` (default) /
                ``"opus"`` / ``"aac"`` / ``"flac"`` / ``"wav"`` /
                ``"pcm"``. Gemini always emits 24 kHz mono PCM
                (wrapped as WAV by vox); the parameter is ignored.
            speed: Playback speed multiplier (0.25-4.0). OpenAI only.
            instructions: Voice-direction prompt (e.g. "Speak in a
                cheerful tone"). Only ``gpt-4o-mini-tts`` and newer.
            **kwargs: Provider-specific passthrough.

        Returns:
            An :class:`AudioContent` containing the synthesized audio.

        Raises:
            InvalidRequestError: When the provider has no native TTS.
        """
        self._raise_audio_unsupported("synthesize")

    async def asynthesize(
        self,
        text: str,
        *,
        voice: str,
        model: str | None = None,
        format: str | None = None,
        speed: float | None = None,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> AudioContent:
        """Asynchronously synthesize text to speech. See :meth:`synthesize`."""
        self._raise_audio_unsupported("synthesize")

    def _raise_audio_unsupported(self, method: str) -> NoReturn:
        """Uniform error path for audio methods on providers without native support."""
        from ..errors import InvalidRequestError

        supported = ", ".join(_AUDIO_SUPPORTED_PROVIDERS)
        raise InvalidRequestError(
            f"{method!s}() is not supported on provider {self.provider_name!r}. "
            f"Providers with native audio support: {supported}.",
            provider=self.provider_name,
        )
