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
