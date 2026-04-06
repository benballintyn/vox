"""Tests for the OpenRouter provider."""

import pytest

from vox import ProviderConfig
from vox.providers.openrouter import OpenRouterProvider


@pytest.fixture
def provider() -> OpenRouterProvider:
    """Create an OpenRouter provider."""
    return OpenRouterProvider(
        ProviderConfig(
            api_key="sk-or-test",
            app_name="TestApp",
            app_url="https://test.com",
        )
    )


class TestOpenRouterProvider:
    """Tests for OpenRouter-specific behavior."""

    def test_provider_name(self, provider: OpenRouterProvider) -> None:
        assert provider.provider_name == "openrouter"

    def test_default_base_url(self) -> None:
        assert OpenRouterProvider._default_base_url == "https://openrouter.ai/api/v1"

    def test_default_api_key_env(self) -> None:
        assert OpenRouterProvider._default_api_key_env == "OPENROUTER_API_KEY"

    def test_custom_headers(self, provider: OpenRouterProvider) -> None:
        headers = provider._get_default_headers()
        assert headers["X-Title"] == "TestApp"
        assert headers["HTTP-Referer"] == "https://test.com"

    def test_no_headers_without_config(self) -> None:
        provider = OpenRouterProvider(ProviderConfig(api_key="sk-or-test"))
        headers = provider._get_default_headers()
        assert headers == {}

    def test_inherits_chat_completions(self) -> None:
        from vox.providers._chat_completions import ChatCompletionsProvider

        assert issubclass(OpenRouterProvider, ChatCompletionsProvider)
