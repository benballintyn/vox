"""Vox error hierarchy.

All provider-specific SDK exceptions are caught by adapters and re-raised as
the appropriate VoxError subclass.
"""

from __future__ import annotations


class VoxError(Exception):
    """Base exception for all vox errors.

    Args:
        message: Human-readable error description.
        provider: Name of the provider that raised the error.
    """

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        self.provider = provider
        super().__init__(message)


class AuthenticationError(VoxError):
    """Invalid or missing API key."""


class RateLimitError(VoxError):
    """Rate limit exceeded.

    Args:
        message: Human-readable error description.
        retry_after: Seconds to wait before retrying, if provided by the API.
        provider: Name of the provider that raised the error.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        provider: str | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message, provider=provider)


class QuotaExceededError(VoxError):
    """Billing/quota limit reached."""


class InvalidRequestError(VoxError):
    """Malformed request or unsupported feature for this model."""


class ProviderError(VoxError):
    """Provider-side error (500, service unavailable, etc.)."""


class ContentFilterError(VoxError):
    """Content was filtered/blocked by the provider's safety system."""


class ModelNotFoundError(VoxError):
    """Requested model does not exist on this provider."""
