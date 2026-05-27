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


class TestVideoFallback:
    """VideoContent on OpenRouter / chat-completions providers falls back to frames."""

    def test_video_replaced_with_image_frames(self, provider: OpenRouterProvider) -> None:
        from unittest.mock import patch as _patch

        from vox import ImageContent, Message, TextContent, VideoContent

        messages = [
            Message(
                role="user",
                content=[
                    TextContent(text="describe"),
                    VideoContent(data="ZmFrZQ==", media_type="video/mp4"),
                ],
            )
        ]

        fake_frames = [
            ImageContent(data="ZnJhbWUx", media_type="image/jpeg"),
        ]
        with _patch(
            "vox._video.substitute_video_with_frames",
            return_value=[
                TextContent(text="describe"),
                *fake_frames,
            ],
        ) as sub:
            translated = provider._translate_messages(messages)

        sub.assert_called_once()
        assert sub.call_args.kwargs["provider_name"] == "openrouter"

        content = translated[0]["content"]
        text_parts = [p for p in content if p["type"] == "text"]
        image_parts = [p for p in content if p["type"] == "image_url"]
        assert len(text_parts) == 1
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
