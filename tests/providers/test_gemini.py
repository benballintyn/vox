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
            # MagicMock auto-creates attributes, so explicitly pin
            # ``thought_signature`` to whatever the test specifies
            # (``None`` by default) — otherwise vox's
            # ``getattr(part, "thought_signature", None)`` would receive
            # an auto-generated MagicMock and treat it as a real value.
            part.thought_signature = fc.get("thought_signature")
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
    mock_response.response_id = "gemini_test_789"

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
        provider._translate_contents(messages)
        # Verify Content was created with role="user"
        types_mock.Content.assert_called()

    @patch("vox.providers.gemini._import_genai_types")
    def test_assistant_tool_call_replays_thought_signature(
        self, mock_types, provider: GeminiProvider
    ) -> None:
        """vox#22 — outbound function_call Parts must carry ``thought_signature``.

        When a previously-issued ``ToolCallData`` is sent back, the
        provider replays the captured signature on the outgoing Part.
        Without it, Gemini rejects the inbound request when thinking
        is enabled.
        """
        from vox.models.messages import ToolCallData

        types_mock = MagicMock()
        mock_types.return_value = types_mock

        messages = [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCallData(
                        id="call_abc",
                        name="weather",
                        arguments={"city": "NYC"},
                        provider_state={"gemini_thought_signature": b"opaque-bytes"},
                    ),
                ],
            ),
        ]
        provider._translate_contents(messages)

        # types.Part(...) was called with the signature as a kwarg.
        part_call_kwargs = [c.kwargs for c in types_mock.Part.call_args_list]
        sig_call = next(
            (kw for kw in part_call_kwargs if "thought_signature" in kw),
            None,
        )
        assert sig_call is not None, "Part was not constructed with thought_signature"
        assert sig_call["thought_signature"] == b"opaque-bytes"

    @patch("vox.providers.gemini._import_genai_types")
    def test_assistant_tool_call_omits_signature_when_absent(
        self, mock_types, provider: GeminiProvider
    ) -> None:
        """Without provider_state, the outbound Part has no ``thought_signature``.

        Lets consumer-built ToolCallData round-trip cleanly on the first
        turn (no signature exists yet); subsequent turns rely on the
        provider having minted the call and stashed the signature.
        """
        from vox.models.messages import ToolCallData

        types_mock = MagicMock()
        mock_types.return_value = types_mock

        messages = [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCallData(id="call_abc", name="weather", arguments={}),
                ],
            ),
        ]
        provider._translate_contents(messages)
        part_call_kwargs = [c.kwargs for c in types_mock.Part.call_args_list]
        assert not any("thought_signature" in kw for kw in part_call_kwargs)

    @patch("vox.providers.gemini._import_genai_types")
    def test_video_inline_translates_to_blob(self, mock_types, provider: GeminiProvider) -> None:
        """VideoContent(source_type=base64) becomes a Part(inline_data=Blob(video/*))."""
        from vox import Message, TextContent, VideoContent

        types_mock = MagicMock()
        mock_types.return_value = types_mock

        messages = [
            Message(
                role="user",
                content=[
                    TextContent(text="What's in this clip?"),
                    VideoContent(data="AAAA", media_type="video/mp4"),
                ],
            )
        ]
        provider._translate_contents(messages)

        # Inspect kwargs for the Blob constructor call.
        blob_calls = types_mock.Blob.call_args_list
        assert blob_calls, "Expected types.Blob(...) to be called for inline video"
        assert blob_calls[0].kwargs["mime_type"] == "video/mp4"

    @patch("vox.providers.gemini._import_genai_types")
    def test_video_url_translates_to_file_data(self, mock_types, provider: GeminiProvider) -> None:
        """VideoContent(source_type=url) → Part(file_data=FileData(file_uri, mime_type))."""
        from vox import Message, VideoContent

        types_mock = MagicMock()
        mock_types.return_value = types_mock

        messages = [
            Message(
                role="user",
                content=[
                    VideoContent(
                        source_type="url",
                        data="https://youtu.be/dQw4w9WgXcQ",
                        media_type="video/mp4",
                    ),
                ],
            )
        ]
        provider._translate_contents(messages)

        fd_calls = types_mock.FileData.call_args_list
        assert fd_calls, "Expected types.FileData(...) to be called for URL video"
        assert fd_calls[0].kwargs["file_uri"] == "https://youtu.be/dQw4w9WgXcQ"
        assert fd_calls[0].kwargs["mime_type"] == "video/mp4"


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
        # No thought_signature on the source part → no provider_state.
        assert result.message.tool_calls[0].provider_state is None

    def test_function_call_captures_thought_signature(self, provider: GeminiProvider) -> None:
        """vox#22 — capture part-level ``thought_signature`` into provider_state.

        Gemini requires the signature on function_call parts when
        thinking is enabled. vox preserves it through ``provider_state``
        so subsequent turns can replay it.
        """
        mock_resp = _make_gemini_response(
            text="",
            function_calls=[
                {
                    "name": "weather",
                    "args": {"city": "NYC"},
                    "thought_signature": b"opaque-encrypted-bytes",
                },
            ],
        )
        result = provider._translate_response(mock_resp, "gemini-2.5-pro")
        assert result.message.tool_calls is not None
        tc = result.message.tool_calls[0]
        assert tc.provider_state == {"gemini_thought_signature": b"opaque-encrypted-bytes"}

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


class TestReasoningTranslation:
    """Tests for Gemini ReasoningConfig translation.

    These tests patch the genai types to avoid requiring the real SDK to
    accept arbitrary fields on ThinkingConfig.
    """

    @patch("vox.providers.gemini._import_genai_types")
    def test_gemini25_uses_thinking_budget_from_level(
        self, mock_types, provider: GeminiProvider
    ) -> None:
        """For Gemini 2.5 models, level maps to thinking_budget."""
        from vox import ReasoningConfig

        types_mock = MagicMock()
        mock_types.return_value = types_mock

        provider._build_thinking_config(ReasoningConfig(level="high"), model="gemini-2.5-pro")
        types_mock.ThinkingConfig.assert_called_with(thinking_budget=32768)

    @patch("vox.providers.gemini._import_genai_types")
    def test_gemini3_uses_thinking_level_from_level(
        self, mock_types, provider: GeminiProvider
    ) -> None:
        """For Gemini 3 models, level maps to thinking_level."""
        from vox import ReasoningConfig

        types_mock = MagicMock()
        mock_types.return_value = types_mock

        provider._build_thinking_config(ReasoningConfig(level="medium"), model="gemini-3.0-pro")
        types_mock.ThinkingConfig.assert_called_with(thinking_level="MEDIUM")

    @patch("vox.providers.gemini._import_genai_types")
    def test_gemini3_minimal_collapses_to_low(self, mock_types, provider: GeminiProvider) -> None:
        """Gemini 3 doesn't support 'minimal' level; collapse to 'low'."""
        from vox import ReasoningConfig

        types_mock = MagicMock()
        mock_types.return_value = types_mock

        provider._build_thinking_config(ReasoningConfig(level="minimal"), model="gemini-3.0-pro")
        types_mock.ThinkingConfig.assert_called_with(thinking_level="LOW")

    @patch("vox.providers.gemini._import_genai_types")
    def test_gemini_override_budget_takes_priority(
        self, mock_types, provider: GeminiProvider
    ) -> None:
        from vox import GeminiReasoning, ReasoningConfig

        types_mock = MagicMock()
        mock_types.return_value = types_mock

        provider._build_thinking_config(
            ReasoningConfig(
                level="high",
                gemini=GeminiReasoning(budget_tokens=12345),
            ),
            model="gemini-2.5-pro",
        )
        types_mock.ThinkingConfig.assert_called_with(thinking_budget=12345)

    @patch("vox.providers.gemini._import_genai_types")
    def test_no_level_no_override_returns_none(self, mock_types, provider: GeminiProvider) -> None:
        from vox import ReasoningConfig

        types_mock = MagicMock()
        mock_types.return_value = types_mock

        result = provider._build_thinking_config(ReasoningConfig(), model="gemini-2.5-pro")
        assert result is None
        types_mock.ThinkingConfig.assert_not_called()
