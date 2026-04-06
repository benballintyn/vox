"""Shared test fixtures for vox."""

import pytest

from vox import ImageContent, Message, TextContent, Tool, ToolResult


@pytest.fixture
def sample_messages() -> list[Message]:
    """Basic conversation messages."""
    return [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello"),
    ]


@pytest.fixture
def sample_user_message() -> Message:
    """A simple user message."""
    return Message(role="user", content="What is 2+2?")


@pytest.fixture
def multimodal_message() -> Message:
    """A message with text and image content."""
    return Message(
        role="user",
        content=[
            TextContent(text="What is this image?"),
            ImageContent(data="iVBORw0KGgo=", media_type="image/png"),
        ],
    )


@pytest.fixture
def sample_tools() -> list[Tool]:
    """Sample tool definitions."""
    return [
        Tool(
            name="get_weather",
            description="Get the current weather for a city.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"},
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius",
                    },
                },
                "required": ["city"],
            },
        ),
        Tool(
            name="search",
            description="Search the web.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
    ]


@pytest.fixture
def tool_result() -> ToolResult:
    """A sample tool result."""
    return ToolResult(
        tool_call_id="call_123",
        name="get_weather",
        content='{"temperature": 72, "conditions": "sunny"}',
    )


@pytest.fixture
def assistant_with_tool_calls() -> Message:
    """An assistant message containing tool calls."""
    from vox.models.messages import ToolCallData

    return Message(
        role="assistant",
        content="Let me check the weather.",
        tool_calls=[
            ToolCallData(
                id="call_123",
                name="get_weather",
                arguments={"city": "NYC"},
            ),
        ],
    )
