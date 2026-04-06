"""Tests for the LM Studio provider."""

import pytest

from vox import ProviderConfig
from vox.providers.lmstudio import LMStudioProvider


@pytest.fixture
def provider() -> LMStudioProvider:
    """Create an LM Studio provider."""
    return LMStudioProvider()


class TestLMStudioProvider:
    """Tests for LM Studio-specific behavior."""

    def test_provider_name(self, provider: LMStudioProvider) -> None:
        assert provider.provider_name == "lmstudio"

    def test_default_base_url(self) -> None:
        assert LMStudioProvider._default_base_url == "http://localhost:1234/v1"

    def test_default_api_key(self, provider: LMStudioProvider) -> None:
        assert provider.config.api_key == "lm-studio"

    def test_custom_base_url(self) -> None:
        provider = LMStudioProvider(ProviderConfig(base_url="http://localhost:5000/v1"))
        assert provider._get_base_url() == "http://localhost:5000/v1"

    def test_custom_api_key(self) -> None:
        provider = LMStudioProvider(ProviderConfig(api_key="custom-key"))
        assert provider.config.api_key == "custom-key"

    def test_inherits_chat_completions(self) -> None:
        from vox.providers._chat_completions import ChatCompletionsProvider

        assert issubclass(LMStudioProvider, ChatCompletionsProvider)

    def test_default_model(self) -> None:
        assert LMStudioProvider._default_model == "local-model"
