"""Tests for the Anthropic provider."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from vox import AnthropicReasoning, Message, ProviderConfig, ReasoningConfig, Tool
from vox.models.messages import ToolCallData
from vox.providers.anthropic import AnthropicProvider


@pytest.fixture
def provider() -> AnthropicProvider:
    """Create an Anthropic provider with a test API key."""
    return AnthropicProvider(
        ProviderConfig(api_key="sk-ant-test", default_model="claude-sonnet-4-20250514")
    )


def _make_anthropic_response(
    text: str = "Hello!",
    tool_use: list[dict] | None = None,
    thinking: list[str] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
    stop_reason: str = "end_turn",
) -> MagicMock:
    """Build a mock Anthropic Messages API response."""
    blocks = []

    if thinking:
        for t in thinking:
            block = MagicMock()
            block.type = "thinking"
            block.thinking = t
            blocks.append(block)

    if text:
        block = MagicMock()
        block.type = "text"
        block.text = text
        blocks.append(block)

    if tool_use:
        for tu in tool_use:
            block = MagicMock()
            block.type = "tool_use"
            block.id = tu["id"]
            block.name = tu["name"]
            block.input = tu["input"]
            blocks.append(block)

    mock_usage = MagicMock()
    mock_usage.input_tokens = input_tokens
    mock_usage.output_tokens = output_tokens
    mock_usage.cache_read_input_tokens = 0
    mock_usage.cache_creation_input_tokens = 0

    mock_response = MagicMock()
    mock_response.content = blocks
    mock_response.usage = mock_usage
    mock_response.stop_reason = stop_reason
    mock_response.id = "msg_test_abc"

    return mock_response


class TestMessageTranslation:
    """Tests for Anthropic message translation."""

    def test_system_prompt_extraction(self, provider: AnthropicProvider) -> None:
        messages = [
            Message(role="system", content="Be helpful."),
            Message(role="user", content="Hello"),
        ]
        translated, system = provider._translate_messages(messages)
        assert system == "Be helpful."
        assert len(translated) == 1
        assert translated[0]["role"] == "user"

    def test_tool_result_as_user_message(self, provider: AnthropicProvider) -> None:
        messages = [
            Message(role="tool", content="72F sunny", tool_call_id="tc_1", name="weather"),
        ]
        translated, _ = provider._translate_messages(messages)
        assert translated[0]["role"] == "user"
        content = translated[0]["content"]
        assert content[0]["type"] == "tool_result"
        assert content[0]["tool_use_id"] == "tc_1"

    def test_assistant_with_tool_use(self, provider: AnthropicProvider) -> None:
        messages = [
            Message(
                role="assistant",
                content="Checking.",
                tool_calls=[
                    ToolCallData(id="tc_1", name="search", arguments={"q": "test"}),
                ],
            ),
        ]
        translated, _ = provider._translate_messages(messages)
        content = translated[0]["content"]
        assert any(b["type"] == "text" for b in content)
        assert any(b["type"] == "tool_use" for b in content)


class TestToolTranslation:
    """Tests for Anthropic tool translation."""

    def test_parameters_to_input_schema(self, provider: AnthropicProvider) -> None:
        tools = [
            Tool(
                name="weather",
                description="Get weather",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            ),
        ]
        result = provider._translate_tools(tools)
        assert result[0]["input_schema"]["type"] == "object"
        assert "name" in result[0]
        assert "description" in result[0]


class TestResponseTranslation:
    """Tests for Anthropic response translation."""

    def test_text_response(self, provider: AnthropicProvider) -> None:
        mock_resp = _make_anthropic_response(text="Hello world!")
        result = provider._translate_response(mock_resp, "claude-sonnet-4-20250514")
        assert result.message.text == "Hello world!"
        assert result.provider == "anthropic"
        assert result.usage.prompt_tokens == 10

    def test_tool_use_response(self, provider: AnthropicProvider) -> None:
        mock_resp = _make_anthropic_response(
            text="",
            tool_use=[{"id": "tu_1", "name": "weather", "input": {"city": "NYC"}}],
        )
        result = provider._translate_response(mock_resp, "claude-sonnet-4-20250514")
        assert result.message.tool_calls is not None
        assert result.message.tool_calls[0].name == "weather"

    def test_thinking_blocks(self, provider: AnthropicProvider) -> None:
        mock_resp = _make_anthropic_response(
            text="The answer is 42.",
            thinking=["Let me think about this..."],
        )
        result = provider._translate_response(mock_resp, "claude-sonnet-4-20250514")
        assert result.thinking is not None
        assert len(result.thinking) == 1
        assert "think" in result.thinking[0].text.lower()


class TestReasoningConfig:
    """Tests for reasoning/thinking parameter building."""

    def test_anthropic_override_sets_exact_budget(self, provider: AnthropicProvider) -> None:
        """Provider-specific override controls budget_tokens precisely."""
        messages = [Message(role="user", content="Think hard")]
        request = provider._build_request_kwargs(
            messages,
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            temperature=1.0,
            tools=None,
            response_schema=None,
            reasoning=ReasoningConfig(anthropic=AnthropicReasoning(budget_tokens=5000)),
            stop=None,
        )
        assert request["thinking"]["type"] == "enabled"
        assert request["thinking"]["budget_tokens"] == 5000
        # Temperature should NOT be set when thinking is enabled
        assert "temperature" not in request

    def test_semantic_level_maps_to_budget(self, provider: AnthropicProvider) -> None:
        """Semantic level maps to a default budget for Anthropic."""
        messages = [Message(role="user", content="Hello")]
        request = provider._build_request_kwargs(
            messages,
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            temperature=1.0,
            tools=None,
            response_schema=None,
            reasoning=ReasoningConfig(level="high"),
            stop=None,
        )
        assert request["thinking"]["type"] == "enabled"
        # "high" maps to 32768 per LEVEL_TO_BUDGET_TOKENS
        assert request["thinking"]["budget_tokens"] == 32768

    def test_anthropic_override_takes_priority_over_level(
        self, provider: AnthropicProvider
    ) -> None:
        """When both level and anthropic.budget_tokens are set, override wins."""
        messages = [Message(role="user", content="Hello")]
        request = provider._build_request_kwargs(
            messages,
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            temperature=1.0,
            tools=None,
            response_schema=None,
            reasoning=ReasoningConfig(
                level="high",
                anthropic=AnthropicReasoning(budget_tokens=12345),
            ),
            stop=None,
        )
        assert request["thinking"]["budget_tokens"] == 12345

    def test_no_reasoning_sets_temperature(self, provider: AnthropicProvider) -> None:
        messages = [Message(role="user", content="Hello")]
        request = provider._build_request_kwargs(
            messages,
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            temperature=0.7,
            tools=None,
            response_schema=None,
            reasoning=None,
            stop=None,
        )
        assert "thinking" not in request
        assert request["temperature"] == 0.7


class TestComplete:
    """Tests for complete with mocked SDK."""

    def test_sync_complete(self, provider: AnthropicProvider) -> None:
        mock_resp = _make_anthropic_response(text="Hi!")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp
        provider._sync_client = mock_client

        result = provider.complete(
            [Message(role="user", content="Hello")],
            model="claude-sonnet-4-20250514",
        )
        assert result.message.text == "Hi!"

    async def test_async_complete(self, provider: AnthropicProvider) -> None:
        mock_resp = _make_anthropic_response(text="Hi!")
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)
        provider._async_client = mock_client

        result = await provider.acomplete(
            [Message(role="user", content="Hello")],
            model="claude-sonnet-4-20250514",
        )
        assert result.message.text == "Hi!"
