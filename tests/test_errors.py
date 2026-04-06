"""Tests for vox error hierarchy."""

from vox import (
    AuthenticationError,
    ContentFilterError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    VoxError,
)


class TestErrorHierarchy:
    """Tests for error inheritance and attributes."""

    def test_all_errors_inherit_from_vox_error(self) -> None:
        errors = [
            AuthenticationError,
            RateLimitError,
            QuotaExceededError,
            InvalidRequestError,
            ProviderError,
            ContentFilterError,
            ModelNotFoundError,
        ]
        for error_cls in errors:
            assert issubclass(error_cls, VoxError)
            assert issubclass(error_cls, Exception)

    def test_vox_error_provider_attribute(self) -> None:
        err = VoxError("test error", provider="openai")
        assert err.provider == "openai"
        assert str(err) == "test error"

    def test_vox_error_no_provider(self) -> None:
        err = VoxError("test error")
        assert err.provider is None

    def test_rate_limit_retry_after(self) -> None:
        err = RateLimitError("rate limited", retry_after=30.0, provider="anthropic")
        assert err.retry_after == 30.0
        assert err.provider == "anthropic"

    def test_rate_limit_no_retry_after(self) -> None:
        err = RateLimitError("rate limited")
        assert err.retry_after is None

    def test_errors_are_catchable_as_vox_error(self) -> None:
        try:
            raise AuthenticationError("bad key", provider="openai")
        except VoxError as e:
            assert e.provider == "openai"

    def test_errors_are_catchable_as_exception(self) -> None:
        try:
            raise ModelNotFoundError("no such model", provider="gemini")
        except Exception as e:
            assert isinstance(e, ModelNotFoundError)
