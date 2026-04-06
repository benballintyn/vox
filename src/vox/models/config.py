"""Provider configuration types."""

from __future__ import annotations

from pydantic import BaseModel


class ProviderConfig(BaseModel):
    """Configuration for a provider instance.

    Args:
        api_key: API key for authentication.
        base_url: Override the default base URL for the provider.
        default_model: Default model to use if none specified per-request.
        app_name: Application name (used as X-Title header for OpenRouter).
        app_url: Application URL (used as HTTP-Referer header for OpenRouter).
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retries on transient errors.
    """

    api_key: str | None = None
    base_url: str | None = None
    default_model: str | None = None
    app_name: str | None = None
    app_url: str | None = None
    timeout: float = 120.0
    max_retries: int = 2
