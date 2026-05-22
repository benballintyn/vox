"""Tests for VoxClient and provider registry."""

import pytest

from vox import InvalidRequestError, ProviderConfig, VoxClient
from vox._registry import resolve_provider
from vox.providers.openrouter import OpenRouterProvider


class TestProviderRegistry:
    """Tests for model-to-provider resolution."""

    def test_openai_models(self) -> None:
        assert resolve_provider("gpt-4o") == "openai"
        assert resolve_provider("gpt-4o-mini") == "openai"
        assert resolve_provider("gpt-3.5-turbo") == "openai"

    def test_openai_reasoning_models(self) -> None:
        assert resolve_provider("o1") == "openai"
        assert resolve_provider("o1-preview") == "openai"
        assert resolve_provider("o3") == "openai"
        assert resolve_provider("o3-mini") == "openai"
        assert resolve_provider("o4-mini") == "openai"

    def test_anthropic_models(self) -> None:
        assert resolve_provider("claude-sonnet-4-20250514") == "anthropic"
        assert resolve_provider("claude-3-opus-20240229") == "anthropic"
        assert resolve_provider("claude-haiku-4-5-20251001") == "anthropic"

    def test_gemini_models(self) -> None:
        assert resolve_provider("gemini-2.5-pro") == "gemini"
        assert resolve_provider("gemini-2.0-flash") == "gemini"
        assert resolve_provider("gemini-3.0-pro") == "gemini"

    def test_explicit_provider_override(self) -> None:
        assert resolve_provider("anything", "openrouter") == "openrouter"
        assert resolve_provider("gpt-4o", "lmstudio") == "lmstudio"

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(InvalidRequestError, match="Cannot determine provider"):
            resolve_provider("llama-3-70b")

    def test_unknown_model_with_explicit_provider(self) -> None:
        assert resolve_provider("llama-3-70b", "openrouter") == "openrouter"


class TestVoxClient:
    """Tests for VoxClient facade."""

    def test_create_client(self) -> None:
        client = VoxClient()
        assert client is not None

    def test_create_client_with_keys(self) -> None:
        client = VoxClient(
            openai_api_key="sk-test",
            anthropic_api_key="sk-ant-test",
        )
        assert client._api_keys["openai"] == "sk-test"
        assert client._api_keys["anthropic"] == "sk-ant-test"

    def test_provider_lazy_creation(self) -> None:
        client = VoxClient(openai_api_key="sk-test")
        assert "openai" not in client._providers
        provider = client._get_provider("openai")
        assert "openai" in client._providers
        assert provider is client._get_provider("openai")  # cached

    def test_provider_config_override(self) -> None:
        custom_config = ProviderConfig(api_key="custom-key", timeout=30.0)
        client = VoxClient(provider_configs={"openai": custom_config})
        provider = client._get_provider("openai")
        assert provider.config.api_key == "custom-key"
        assert provider.config.timeout == 30.0

    def test_unknown_provider_raises(self) -> None:
        client = VoxClient()
        with pytest.raises(InvalidRequestError, match="Unknown provider"):
            client._get_provider("nonexistent")

    def test_all_providers_can_be_created(self) -> None:
        client = VoxClient(
            openai_api_key="sk-test",
            anthropic_api_key="sk-ant-test",
            gemini_api_key="AI-test",
            openrouter_api_key="sk-or-test",
        )
        for name in ["openai", "anthropic", "gemini", "openrouter", "lmstudio"]:
            provider = client._get_provider(name)
            assert provider.provider_name == name

    def test_lmstudio_default_api_key(self) -> None:
        client = VoxClient()
        provider = client._get_provider("lmstudio")
        assert provider.config.api_key == "lm-studio"

    def test_openrouter_headers(self) -> None:
        client = VoxClient(
            openrouter_api_key="sk-or-test",
            openrouter_app_name="MyApp",
            openrouter_app_url="https://myapp.com",
        )
        provider = client._get_provider("openrouter")
        assert isinstance(provider, OpenRouterProvider)
        headers = provider._get_default_headers()
        assert headers["X-Title"] == "MyApp"
        assert headers["HTTP-Referer"] == "https://myapp.com"
