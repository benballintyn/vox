"""Provider test fixtures."""

import json
from typing import Any
from unittest.mock import MagicMock


def make_openai_chat_response(
    content: str = "Hello!",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    """Build a mock OpenAI Chat Completions response.

    Args:
        content: The text content of the response.
        tool_calls: List of tool call dicts.
        finish_reason: Finish reason string.
        prompt_tokens: Prompt token count.
        completion_tokens: Completion token count.

    Returns:
        A MagicMock configured as an OpenAI response.
    """
    mock_message = MagicMock()
    mock_message.content = content
    mock_message.tool_calls = None

    if tool_calls:
        mock_tcs = []
        for tc in tool_calls:
            mock_tc = MagicMock()
            mock_tc.id = tc["id"]
            mock_tc.function.name = tc["name"]
            mock_tc.function.arguments = json.dumps(tc["arguments"])
            mock_tcs.append(mock_tc)
        mock_message.tool_calls = mock_tcs

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = finish_reason

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = prompt_tokens
    mock_usage.completion_tokens = completion_tokens
    mock_usage.total_tokens = prompt_tokens + completion_tokens

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    return mock_response


def make_openai_responses_api_response(
    content: str = "Hello!",
    function_calls: list[dict[str, Any]] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
    status: str = "completed",
) -> MagicMock:
    """Build a mock OpenAI Responses API response.

    Args:
        content: The text content.
        function_calls: List of function call dicts.
        input_tokens: Input token count.
        output_tokens: Output token count.
        status: Response status.

    Returns:
        A MagicMock configured as a Responses API response.
    """
    output_items = []

    if content:
        text_content = MagicMock()
        text_content.type = "output_text"
        text_content.text = content

        message_item = MagicMock()
        message_item.type = "message"
        message_item.content = [text_content]
        output_items.append(message_item)

    if function_calls:
        for fc in function_calls:
            fc_item = MagicMock()
            fc_item.type = "function_call"
            fc_item.id = fc["id"]
            fc_item.call_id = fc["id"]
            fc_item.name = fc["name"]
            fc_item.arguments = json.dumps(fc["arguments"])
            output_items.append(fc_item)

    mock_usage = MagicMock()
    mock_usage.input_tokens = input_tokens
    mock_usage.output_tokens = output_tokens
    mock_usage.reasoning_tokens = 0

    mock_response = MagicMock()
    mock_response.output = output_items
    mock_response.usage = mock_usage
    mock_response.status = status

    return mock_response
