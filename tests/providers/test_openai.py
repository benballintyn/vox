"""Tests for the OpenAI Responses API provider."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from vox import Message, ProviderConfig, Tool
from vox.models.messages import ToolCallData
from vox.providers.openai import OpenAIProvider

from .conftest import make_openai_responses_api_response


@pytest.fixture
def provider() -> OpenAIProvider:
    """Create an OpenAI provider with a test API key."""
    return OpenAIProvider(ProviderConfig(api_key="sk-test", default_model="gpt-4o"))


class TestMessageTranslation:
    """Tests for Responses API message translation."""

    def test_simple_user_message(self, provider: OpenAIProvider) -> None:
        messages = [Message(role="user", content="Hello")]
        items, instructions = provider._translate_input(messages)
        assert len(items) == 1
        assert items[0]["role"] == "user"
        assert items[0]["type"] == "message"
        assert items[0]["content"][0]["type"] == "input_text"
        assert items[0]["content"][0]["text"] == "Hello"

    def test_system_message_becomes_instructions(self, provider: OpenAIProvider) -> None:
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hi"),
        ]
        items, instructions = provider._translate_input(messages)
        assert instructions == "You are helpful."
        assert len(items) == 1  # System not in items

    def test_tool_result_message(self, provider: OpenAIProvider) -> None:
        messages = [
            Message(role="tool", content="72F", tool_call_id="call_1", name="weather"),
        ]
        items, _ = provider._translate_input(messages)
        assert items[0]["type"] == "function_call_output"
        assert items[0]["call_id"] == "call_1"
        assert items[0]["output"] == "72F"

    def test_assistant_with_tool_calls(self, provider: OpenAIProvider) -> None:
        messages = [
            Message(
                role="assistant",
                content="Checking weather.",
                tool_calls=[
                    ToolCallData(id="call_1", name="weather", arguments={"city": "NYC"}),
                ],
            ),
        ]
        items, _ = provider._translate_input(messages)
        # Text + function_call
        assert any(i.get("type") == "message" for i in items)
        assert any(i.get("type") == "function_call" for i in items)


class TestToolTranslation:
    """Tests for Responses API tool translation."""

    def test_tool_format(self, provider: OpenAIProvider) -> None:
        tools = [
            Tool(
                name="search",
                description="Search the web",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            ),
        ]
        result = provider._translate_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "search"
        assert result[0]["parameters"]["type"] == "object"


class TestResponseTranslation:
    """Tests for Responses API response translation."""

    def test_text_response(self, provider: OpenAIProvider) -> None:
        mock_resp = make_openai_responses_api_response(content="Hello world!")
        result = provider._translate_response(mock_resp, "gpt-4o")
        assert result.message.text == "Hello world!"
        assert result.provider == "openai"
        assert result.model == "gpt-4o"
        assert result.usage.prompt_tokens == 10

    def test_function_call_response(self, provider: OpenAIProvider) -> None:
        mock_resp = make_openai_responses_api_response(
            content="",
            function_calls=[
                {"id": "call_1", "name": "weather", "arguments": {"city": "NYC"}},
            ],
        )
        result = provider._translate_response(mock_resp, "gpt-4o")
        assert result.message.tool_calls is not None
        assert len(result.message.tool_calls) == 1
        assert result.message.tool_calls[0].name == "weather"
        assert result.message.tool_calls[0].arguments == {"city": "NYC"}


class TestComplete:
    """Tests for the complete method with mocked SDK."""

    def test_sync_complete(self, provider: OpenAIProvider, mocker) -> None:
        mock_resp = make_openai_responses_api_response(content="Hi there!")
        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_resp
        provider._sync_client = mock_client

        result = provider.complete(
            [Message(role="user", content="Hello")],
            model="gpt-4o",
        )
        assert result.message.text == "Hi there!"
        mock_client.responses.create.assert_called_once()

    async def test_async_complete(self, provider: OpenAIProvider, mocker) -> None:
        mock_resp = make_openai_responses_api_response(content="Hi there!")
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=mock_resp)
        provider._async_client = mock_client

        result = await provider.acomplete(
            [Message(role="user", content="Hello")],
            model="gpt-4o",
        )
        assert result.message.text == "Hi there!"
