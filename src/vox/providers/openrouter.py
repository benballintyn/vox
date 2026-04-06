"""OpenRouter provider using the Chat Completions API protocol."""

from __future__ import annotations

from ._chat_completions import ChatCompletionsProvider


class OpenRouterProvider(ChatCompletionsProvider):
    """OpenRouter provider.

    Uses the OpenAI Chat Completions API protocol with OpenRouter's endpoint.
    Adds custom headers for app identification.

    Args:
        config: Provider configuration. Use ``app_name`` and ``app_url``
            fields for OpenRouter's X-Title and HTTP-Referer headers.
    """

    _default_base_url: str = "https://openrouter.ai/api/v1"
    _default_api_key_env: str = "OPENROUTER_API_KEY"
    _default_model: str = "anthropic/claude-sonnet-4-20250514"

    @property
    def provider_name(self) -> str:
        return "openrouter"

    def _get_default_headers(self) -> dict[str, str]:
        """Return OpenRouter-specific headers.

        Returns:
            Dict with X-Title and HTTP-Referer if configured.
        """
        headers: dict[str, str] = {}
        if self.config.app_name:
            headers["X-Title"] = self.config.app_name
        if self.config.app_url:
            headers["HTTP-Referer"] = self.config.app_url
        return headers
