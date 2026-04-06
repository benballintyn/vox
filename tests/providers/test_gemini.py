"""Tests for the Gemini provider."""

from unittest.mock import MagicMock, patch

import pytest

from vox import Message, ProviderConfig
from vox.providers.gemini import GeminiProvider


@pytest.fixture
def provider() -> GeminiProvider:
    """Create a Gemini provider with a test API key."""
    return GeminiProvider(ProviderConfig(api_key="AI-test", default_model="gemini-2.5-pro"))


def _make_gemini_response(
    text: str = "Hello!",
    function_calls: list[dict] | None = None,
    thinking: list[str] | None = None,
    prompt_tokens: int = 10,
    candidates_tokens: int = 5,
) -> MagicMock:
    """Build a mock Gemini response."""
    parts = []

    if thinking:
        for t in thinking:
            part = MagicMock()
            part.text = t
            part.thought = True
            part.function_call = None
            parts.append(part)

    if text:
        part = MagicMock()
        part.text = text
        part.thought = False
        part.function_call = None
        parts.append(part)

    if function_calls:
        for fc in function_calls:
            part = MagicMock()
            part.text = None
            part.thought = False
            mock_fc = MagicMock()
            mock_fc.name = fc["name"]
            mock_fc.args = fc.get("args", {})
            part.function_call = mock_fc
            parts.append(part)

    mock_content = MagicMock()
    mock_content.parts = parts

    mock_candidate = MagicMock()
    mock_candidate.content = mock_content
    mock_candidate.finish_reason = "STOP"

    mock_usage = MagicMock()
    mock_usage.prompt_token_count = prompt_tokens
    mock_usage.candidates_token_count = candidates_tokens
    mock_usage.total_token_count = prompt_tokens + candidates_tokens
    mock_usage.thinking_token_count = 0

    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = mock_usage

    return mock_response


class TestMessageTranslation:
    """Tests for Gemini message translation."""

    @patch("vox.providers.gemini._import_genai_types")
    def test_system_extraction(self, mock_types, provider: GeminiProvider) -> None:
        mock_types.return_value = MagicMock()
        messages = [
            Message(role="system", content="Be helpful."),
            Message(role="user", content="Hello"),
        ]
        _, system = provider._translate_contents(messages)
        assert system == "Be helpful."

    @patch("vox.providers.gemini._import_genai_types")
    def test_role_mapping(self, mock_types, provider: GeminiProvider) -> None:
        types_mock = MagicMock()
        mock_types.return_value = types_mock

        messages = [Message(role="user", content="Hi")]
        contents, _ = provider._translate_contents(messages)
        # Verify Content was created with role="user"
        types_mock.Content.assert_called()


class TestResponseTranslation:
    """Tests for Gemini response translation."""

    def test_text_response(self, provider: GeminiProvider) -> None:
        mock_resp = _make_gemini_response(text="Hello world!")
        result = provider._translate_response(mock_resp, "gemini-2.5-pro")
        assert result.message.text == "Hello world!"
        assert result.provider == "gemini"
        assert result.usage.prompt_tokens == 10

    def test_function_call_response(self, provider: GeminiProvider) -> None:
        mock_resp = _make_gemini_response(
            text="",
            function_calls=[{"name": "weather", "args": {"city": "NYC"}}],
        )
        result = provider._translate_response(mock_resp, "gemini-2.5-pro")
        assert result.message.tool_calls is not None
        assert result.message.tool_calls[0].name == "weather"

    def test_thinking_response(self, provider: GeminiProvider) -> None:
        mock_resp = _make_gemini_response(
            text="42",
            thinking=["Let me reason..."],
        )
        result = provider._translate_response(mock_resp, "gemini-2.5-pro")
        assert result.thinking is not None
        assert len(result.thinking) == 1

    def test_usage_mapping(self, provider: GeminiProvider) -> None:
        mock_resp = _make_gemini_response(prompt_tokens=100, candidates_tokens=50)
        result = provider._translate_response(mock_resp, "gemini-2.5-pro")
        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 50
        assert result.usage.total_tokens == 150


class TestStreamChunkTranslation:
    """Tests for Gemini stream chunk translation."""

    def test_text_chunk(self, provider: GeminiProvider) -> None:
        part = MagicMock()
        part.text = "Hello"
        part.thought = False
        part.function_call = None

        chunk = MagicMock()
        chunk.candidates = [MagicMock()]
        chunk.candidates[0].content = MagicMock()
        chunk.candidates[0].content.parts = [part]
        chunk.candidates[0].finish_reason = None

        result = provider._translate_stream_chunk(chunk)
        assert len(result) == 1
        assert result[0].type == "text"
        assert result[0].text == "Hello"

    def test_empty_chunk(self, provider: GeminiProvider) -> None:
        chunk = MagicMock()
        chunk.candidates = []
        result = provider._translate_stream_chunk(chunk)
        assert result == []
