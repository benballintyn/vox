"""Tests for vox data models."""

from vox import (
    CompletionResponse,
    ImageContent,
    Message,
    ProviderConfig,
    ReasoningConfig,
    StreamChunk,
    TextContent,
    ThinkingBlock,
    Tool,
    ToolCall,
    ToolResult,
    Usage,
)
from vox.models.messages import ToolCallData


class TestMessage:
    """Tests for the Message model."""

    def test_simple_text_message(self) -> None:
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.text == "hello"
        assert msg.tool_calls is None

    def test_multimodal_message(self) -> None:
        msg = Message(
            role="user",
            content=[
                TextContent(text="What is this?"),
                ImageContent(data="base64data", media_type="image/png"),
            ],
        )
        assert msg.text == "What is this?"
        assert len(msg.content) == 2

    def test_text_property_with_string(self) -> None:
        msg = Message(role="system", content="You are helpful.")
        assert msg.text == "You are helpful."

    def test_text_property_with_parts(self) -> None:
        msg = Message(
            role="user",
            content=[
                TextContent(text="Hello "),
                ImageContent(data="abc"),
                TextContent(text="world"),
            ],
        )
        assert msg.text == "Hello world"

    def test_empty_content(self) -> None:
        msg = Message(role="assistant", content="")
        assert msg.text == ""

    def test_assistant_with_tool_calls(self) -> None:
        msg = Message(
            role="assistant",
            content="Calling tool.",
            tool_calls=[
                ToolCallData(id="call_1", name="search", arguments={"q": "test"}),
            ],
        )
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"

    def test_tool_result_message(self) -> None:
        msg = Message(role="tool", content="result data", tool_call_id="call_1", name="search")
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_1"

    def test_serialization_roundtrip(self) -> None:
        msg = Message(role="user", content="hello")
        data = msg.model_dump()
        restored = Message.model_validate(data)
        assert restored == msg

    def test_url_image(self) -> None:
        img = ImageContent(source_type="url", data="https://example.com/img.png")
        assert img.source_type == "url"


class TestImageContent:
    """Tests for ImageContent's bytes-friendly data accessor."""

    def test_data_accepts_raw_bytes_and_base64_encodes(self) -> None:
        """``ImageContent(data=raw_bytes)`` auto-encodes to a base64 ASCII string.

        Saves callers the ``base64.standard_b64encode(b).decode("ascii")``
        line that would otherwise live at every site that constructs an
        image from a file read or a BytesIO buffer.
        """
        import base64

        raw = b"\x89PNG\r\n\x1a\nfake"
        # ``data`` is annotated ``str`` (its post-validation type); the
        # validator accepts ``bytes`` and converts. mypy can't see the
        # validator's wider input type, so silence the arg-type check.
        img = ImageContent(data=raw, media_type="image/png")  # type: ignore[arg-type]
        assert isinstance(img.data, str)
        assert img.data == base64.standard_b64encode(raw).decode("ascii")

    def test_data_accepts_bytearray(self) -> None:
        import base64

        raw = bytearray(b"\x89PNG\r\nbuf")
        img = ImageContent(data=raw, media_type="image/png")  # type: ignore[arg-type]
        assert img.data == base64.standard_b64encode(bytes(raw)).decode("ascii")

    def test_data_string_passes_through(self) -> None:
        """Strings (base64 or otherwise) are not re-encoded."""
        already_b64 = "iVBORw0KGgo="
        img = ImageContent(data=already_b64, media_type="image/png")
        assert img.data == already_b64

    def test_data_url_string_passes_through(self) -> None:
        """URL strings under ``source_type="url"`` are not touched."""
        url = "https://example.com/cat.jpg"
        img = ImageContent(source_type="url", data=url)
        assert img.data == url


class TestVideoContent:
    """Tests for VideoContent's bytes-friendly data accessor and defaults."""

    def test_default_media_type_is_mp4(self) -> None:
        from vox import VideoContent

        video = VideoContent(data="ZmFrZQ==")
        assert video.media_type == "video/mp4"
        assert video.type == "video"
        assert video.source_type == "base64"

    def test_data_accepts_raw_bytes_and_base64_encodes(self) -> None:
        """``VideoContent(data=raw_bytes)`` auto-encodes to a base64 ASCII string.

        Mirrors the ImageContent ergonomics — callers who hold raw
        video bytes from a file read shouldn't need to base64 by hand.
        """
        import base64

        from vox import VideoContent

        raw = b"\x00\x00\x00\x20ftypisom"
        video = VideoContent(data=raw, media_type="video/mp4")  # type: ignore[arg-type]
        assert isinstance(video.data, str)
        assert video.data == base64.standard_b64encode(raw).decode("ascii")

    def test_data_string_passes_through(self) -> None:
        from vox import VideoContent

        already_b64 = "AAAAIGZ0eXBpc29t"
        video = VideoContent(data=already_b64)
        assert video.data == already_b64

    def test_data_url_string_passes_through(self) -> None:
        """URL strings under ``source_type="url"`` are not touched.

        Gemini supports YouTube URLs and Files-API file URIs through
        this path.
        """
        from vox import VideoContent

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video = VideoContent(source_type="url", data=url, media_type="video/mp4")
        assert video.data == url


class TestAudioContent:
    """Tests for AudioContent's defaults + bytes-friendly data accessor."""

    def test_default_media_type_is_mp3(self) -> None:
        from vox import AudioContent

        audio = AudioContent(data="ZmFrZQ==")
        assert audio.type == "audio"
        assert audio.source_type == "base64"
        assert audio.media_type == "audio/mp3"

    def test_data_accepts_raw_bytes_and_base64_encodes(self) -> None:
        """Mirrors ImageContent/VideoContent ergonomics for audio bytes."""
        import base64

        from vox import AudioContent

        raw = b"RIFF\x00\x00\x00\x00WAVEfake"
        audio = AudioContent(data=raw, media_type="audio/wav")  # type: ignore[arg-type]
        assert isinstance(audio.data, str)
        assert audio.data == base64.standard_b64encode(raw).decode("ascii")

    def test_data_string_passes_through(self) -> None:
        from vox import AudioContent

        already_b64 = "UklGRgAAAABXQVZF"
        audio = AudioContent(data=already_b64, media_type="audio/wav")
        assert audio.data == already_b64

    def test_data_url_string_passes_through(self) -> None:
        from vox import AudioContent

        url = "https://storage.googleapis.com/example/clip.wav"
        audio = AudioContent(source_type="url", data=url, media_type="audio/wav")
        assert audio.data == url


class TestTranscriptionResponse:
    """Smoke tests for the TranscriptionResponse model defaults."""

    def test_minimal_construction(self) -> None:
        from vox import TranscriptionResponse

        r = TranscriptionResponse(text="hello", provider="openai", model="whisper-1")
        assert r.text == "hello"
        assert r.language is None
        assert r.duration is None
        assert r.usage is None

    def test_with_optional_fields(self) -> None:
        from vox import TranscriptionResponse, Usage

        r = TranscriptionResponse(
            text="hello",
            language="en",
            duration=1.2,
            provider="openai",
            model="whisper-1",
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
        assert r.language == "en"
        assert r.duration == 1.2
        assert r.usage is not None


class TestTool:
    """Tests for Tool, ToolCall, and ToolResult models."""

    def test_tool_definition(self) -> None:
        tool = Tool(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        assert tool.name == "get_weather"
        data = tool.model_dump()
        assert data["parameters"]["type"] == "object"

    def test_tool_call(self) -> None:
        tc = ToolCall(id="call_1", name="search", arguments={"query": "test"})
        assert tc.id == "call_1"
        assert tc.arguments["query"] == "test"

    def test_tool_result_to_message(self) -> None:
        result = ToolResult(
            tool_call_id="call_1",
            name="get_weather",
            content="72F and sunny",
        )
        msg = result.to_message()
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_1"
        assert msg.name == "get_weather"
        assert msg.content == "72F and sunny"

    def test_tool_result_error(self) -> None:
        result = ToolResult(
            tool_call_id="call_1",
            name="search",
            content="Connection timeout",
            is_error=True,
        )
        assert result.is_error is True


class TestResponses:
    """Tests for response models."""

    def test_usage_defaults(self) -> None:
        usage = Usage()
        assert usage.prompt_tokens == 0
        assert usage.total_tokens == 0
        assert usage.reasoning_tokens == 0

    def test_completion_response(self) -> None:
        resp = CompletionResponse(
            message=Message(role="assistant", content="Hello!"),
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            provider="openai",
            model="gpt-4o",
            finish_reason="stop",
        )
        assert resp.provider == "openai"
        assert resp.message.text == "Hello!"
        assert resp.usage.total_tokens == 15
        assert resp.parsed is None

    def test_stream_chunk_text(self) -> None:
        chunk = StreamChunk(type="text", text="Hello")
        assert chunk.type == "text"
        assert chunk.text == "Hello"

    def test_stream_chunk_done(self) -> None:
        chunk = StreamChunk(type="done", finish_reason="stop")
        assert chunk.type == "done"
        assert chunk.finish_reason == "stop"

    def test_stream_chunk_usage(self) -> None:
        chunk = StreamChunk(
            type="usage",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        assert chunk.usage is not None
        assert chunk.usage.total_tokens == 15

    def test_stream_chunk_thinking(self) -> None:
        chunk = StreamChunk(type="thinking", thinking_text="Let me think...")
        assert chunk.thinking_text == "Let me think..."


class TestReasoning:
    """Tests for reasoning models."""

    def test_reasoning_config_defaults(self) -> None:
        rc = ReasoningConfig()
        assert rc.enabled is True
        assert rc.level is None
        assert rc.openai is None
        assert rc.anthropic is None
        assert rc.gemini is None

    def test_reasoning_config_with_level(self) -> None:
        rc = ReasoningConfig(level="high")
        assert rc.level == "high"

    def test_reasoning_config_with_minimal_level(self) -> None:
        rc = ReasoningConfig(level="minimal")
        assert rc.level == "minimal"

    def test_reasoning_config_with_openai_override(self) -> None:
        from vox import OpenAIReasoning

        rc = ReasoningConfig(openai=OpenAIReasoning(effort="xhigh", summary="detailed"))
        assert rc.openai is not None
        assert rc.openai.effort == "xhigh"
        assert rc.openai.summary == "detailed"

    def test_reasoning_config_with_anthropic_override(self) -> None:
        from vox import AnthropicReasoning

        rc = ReasoningConfig(anthropic=AnthropicReasoning(budget_tokens=10000))
        assert rc.anthropic is not None
        assert rc.anthropic.budget_tokens == 10000

    def test_reasoning_config_with_gemini_override(self) -> None:
        from vox import GeminiReasoning

        rc = ReasoningConfig(gemini=GeminiReasoning(budget_tokens=8192))
        assert rc.gemini is not None
        assert rc.gemini.budget_tokens == 8192

    def test_reasoning_config_combined_level_and_override(self) -> None:
        from vox import OpenAIReasoning

        rc = ReasoningConfig(level="medium", openai=OpenAIReasoning(effort="xhigh"))
        # Level remains for other providers; openai sub-config overrides for OpenAI
        assert rc.level == "medium"
        assert rc.openai is not None
        assert rc.openai.effort == "xhigh"

    def test_thinking_block(self) -> None:
        tb = ThinkingBlock(text="Step 1: analyze the problem", token_count=50)
        assert tb.text.startswith("Step 1")
        assert tb.token_count == 50


class TestProviderConfig:
    """Tests for ProviderConfig."""

    def test_defaults(self) -> None:
        config = ProviderConfig()
        assert config.api_key is None
        assert config.timeout == 120.0
        assert config.max_retries == 2

    def test_custom_config(self) -> None:
        config = ProviderConfig(
            api_key="sk-test",
            base_url="https://custom.api.com",
            default_model="gpt-4o-mini",
            timeout=60.0,
        )
        assert config.api_key == "sk-test"
        assert config.base_url == "https://custom.api.com"
