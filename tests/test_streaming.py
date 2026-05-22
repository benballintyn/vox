"""Tests for streaming code paths across providers.

Streams are simulated by passing pre-built lists of mock SDK chunks/events
to each provider's ``_translate_stream_chunk`` / ``_process_stream_event``
methods. This avoids needing to mock the full network/SDK stack while still
exercising the chunk-translation logic that was previously untested.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vox import ProviderConfig
from vox.providers._chat_completions import ChatCompletionsProvider
from vox.providers.anthropic import AnthropicProvider
from vox.providers.openai import OpenAIProvider


@pytest.fixture
def chat_completions_provider() -> ChatCompletionsProvider:
    """Minimal concrete subclass for testing the Chat Completions base."""

    class _TestProvider(ChatCompletionsProvider):
        @property
        def provider_name(self) -> str:
            return "test_cc"

    return _TestProvider(ProviderConfig(api_key="sk-test", default_model="gpt-4o"))


# ───────────────────────────────────────────────────────────────────────────
#  Helpers to build mock SDK chunks
# ───────────────────────────────────────────────────────────────────────────


def _cc_chunk(
    *,
    content: str | None = None,
    tool_call_name: str | None = None,
    tool_call_args: str | None = None,
    tool_call_id: str = "call_1",
    finish_reason: str | None = None,
    usage: tuple[int, int, int] | None = None,
) -> MagicMock:
    """Build a single mock OpenAI Chat Completions stream chunk."""
    chunk = MagicMock()

    if usage is not None and content is None and tool_call_name is None:
        # Final usage-only chunk
        prompt, completion, total = usage
        chunk.usage.prompt_tokens = prompt
        chunk.usage.completion_tokens = completion
        chunk.usage.total_tokens = total
        chunk.choices = []
        return chunk

    chunk.usage = None
    choice = MagicMock()
    choice.finish_reason = finish_reason
    delta = MagicMock()

    if tool_call_name or tool_call_args:
        tc = MagicMock()
        tc.id = tool_call_id
        tc.function.name = tool_call_name
        tc.function.arguments = tool_call_args
        delta.tool_calls = [tc]
    else:
        delta.tool_calls = None

    delta.content = content
    choice.delta = delta
    chunk.choices = [choice]
    return chunk


# ───────────────────────────────────────────────────────────────────────────
#  Chat Completions streaming
# ───────────────────────────────────────────────────────────────────────────


class TestChatCompletionsStreaming:
    """Tests for the Chat Completions stream translator."""

    def test_text_delta(self, chat_completions_provider: ChatCompletionsProvider) -> None:
        chunks = chat_completions_provider._translate_stream_chunk(_cc_chunk(content="Hello"))
        assert len(chunks) == 1
        assert chunks[0].type == "text"
        assert chunks[0].text == "Hello"

    def test_tool_call_name_only_chunk(
        self, chat_completions_provider: ChatCompletionsProvider
    ) -> None:
        chunks = chat_completions_provider._translate_stream_chunk(
            _cc_chunk(tool_call_name="search", tool_call_args=None)
        )
        assert len(chunks) == 1
        assert chunks[0].type == "tool_call_start"
        tool_call = chunks[0].tool_call
        assert tool_call is not None
        assert tool_call.name == "search"

    def test_tool_call_arguments_only_chunk(
        self, chat_completions_provider: ChatCompletionsProvider
    ) -> None:
        chunks = chat_completions_provider._translate_stream_chunk(
            _cc_chunk(tool_call_name=None, tool_call_args='{"q":')
        )
        assert len(chunks) == 1
        assert chunks[0].type == "tool_call_delta"
        assert chunks[0].arguments_delta == '{"q":'

    def test_tool_call_name_and_args_in_same_chunk(
        self, chat_completions_provider: ChatCompletionsProvider
    ) -> None:
        """Regression test: a single chunk carrying both name and args must
        emit both events. The pre-v0.1.0 code returned after emitting
        tool_call_start and silently dropped the arguments."""
        chunks = chat_completions_provider._translate_stream_chunk(
            _cc_chunk(tool_call_name="search", tool_call_args='{"q":"test"}')
        )
        assert len(chunks) == 2
        assert chunks[0].type == "tool_call_start"
        tool_call = chunks[0].tool_call
        assert tool_call is not None
        assert tool_call.name == "search"
        assert chunks[1].type == "tool_call_delta"
        assert chunks[1].arguments_delta == '{"q":"test"}'

    def test_finish_reason_normalized(
        self, chat_completions_provider: ChatCompletionsProvider
    ) -> None:
        """Native 'tool_calls' should normalize to 'tool_calls'."""
        chunks = chat_completions_provider._translate_stream_chunk(
            _cc_chunk(finish_reason="tool_calls")
        )
        assert any(c.type == "done" and c.finish_reason == "tool_calls" for c in chunks)

    def test_length_finish_reason_normalized(
        self, chat_completions_provider: ChatCompletionsProvider
    ) -> None:
        chunks = chat_completions_provider._translate_stream_chunk(
            _cc_chunk(finish_reason="length")
        )
        assert any(c.type == "done" and c.finish_reason == "length" for c in chunks)

    def test_usage_final_chunk(self, chat_completions_provider: ChatCompletionsProvider) -> None:
        chunks = chat_completions_provider._translate_stream_chunk(_cc_chunk(usage=(100, 50, 150)))
        assert len(chunks) == 1
        assert chunks[0].type == "usage"
        usage = chunks[0].usage
        assert usage is not None
        assert usage.total_tokens == 150

    def test_empty_chunk(self, chat_completions_provider: ChatCompletionsProvider) -> None:
        chunk = MagicMock()
        chunk.usage = None
        chunk.choices = []
        chunks = chat_completions_provider._translate_stream_chunk(chunk)
        assert chunks == []


# ───────────────────────────────────────────────────────────────────────────
#  Anthropic streaming
# ───────────────────────────────────────────────────────────────────────────


@pytest.fixture
def anthropic_provider() -> AnthropicProvider:
    return AnthropicProvider(
        ProviderConfig(api_key="sk-ant-test", default_model="claude-sonnet-4-20250514")
    )


def _anthropic_event(event_type: str, **kwargs) -> MagicMock:
    event = MagicMock()
    event.type = event_type
    for k, v in kwargs.items():
        setattr(event, k, v)
    return event


class TestAnthropicStreaming:
    """Tests for the Anthropic stream event translator."""

    def test_text_delta(self, anthropic_provider: AnthropicProvider) -> None:
        delta = MagicMock()
        delta.type = "text_delta"
        delta.text = "Hello"
        chunk = anthropic_provider._process_stream_event(
            _anthropic_event("content_block_delta", delta=delta)
        )
        assert chunk is not None
        assert chunk.type == "text"
        assert chunk.text == "Hello"

    def test_thinking_delta(self, anthropic_provider: AnthropicProvider) -> None:
        delta = MagicMock()
        delta.type = "thinking_delta"
        delta.thinking = "Let me reason..."
        chunk = anthropic_provider._process_stream_event(
            _anthropic_event("content_block_delta", delta=delta)
        )
        assert chunk is not None
        assert chunk.type == "thinking"
        assert chunk.thinking_text == "Let me reason..."

    def test_tool_use_block_start(self, anthropic_provider: AnthropicProvider) -> None:
        block = MagicMock()
        block.type = "tool_use"
        block.id = "tc_1"
        block.name = "search"
        chunk = anthropic_provider._process_stream_event(
            _anthropic_event("content_block_start", content_block=block)
        )
        assert chunk is not None
        assert chunk.type == "tool_call_start"
        assert chunk.tool_call is not None
        assert chunk.tool_call.name == "search"

    def test_input_json_delta(self, anthropic_provider: AnthropicProvider) -> None:
        delta = MagicMock()
        delta.type = "input_json_delta"
        delta.partial_json = '{"q":'
        chunk = anthropic_provider._process_stream_event(
            _anthropic_event("content_block_delta", delta=delta)
        )
        assert chunk is not None
        assert chunk.type == "tool_call_delta"
        assert chunk.arguments_delta == '{"q":'

    def test_message_delta_with_stop_reason_normalized(
        self, anthropic_provider: AnthropicProvider
    ) -> None:
        """Anthropic's native 'end_turn' should normalize to 'stop'."""
        delta = MagicMock()
        delta.stop_reason = "end_turn"
        chunk = anthropic_provider._process_stream_event(
            _anthropic_event("message_delta", delta=delta)
        )
        assert chunk is not None
        assert chunk.type == "done"
        assert chunk.finish_reason == "stop"

    def test_message_delta_tool_use_normalized(
        self, anthropic_provider: AnthropicProvider
    ) -> None:
        delta = MagicMock()
        delta.stop_reason = "tool_use"
        chunk = anthropic_provider._process_stream_event(
            _anthropic_event("message_delta", delta=delta)
        )
        assert chunk is not None
        assert chunk.finish_reason == "tool_calls"


# ───────────────────────────────────────────────────────────────────────────
#  OpenAI Responses API streaming
# ───────────────────────────────────────────────────────────────────────────


@pytest.fixture
def openai_provider() -> OpenAIProvider:
    return OpenAIProvider(ProviderConfig(api_key="sk-test", default_model="gpt-5"))


class TestOpenAIStreaming:
    """Tests for the OpenAI Responses API stream event translator."""

    def test_output_text_delta(self, openai_provider: OpenAIProvider) -> None:
        event = MagicMock()
        event.type = "response.output_text.delta"
        event.delta = "Hello"
        chunk = openai_provider._process_stream_event(event)
        assert chunk is not None
        assert chunk.type == "text"
        assert chunk.text == "Hello"

    def test_function_call_added(self, openai_provider: OpenAIProvider) -> None:
        item = MagicMock()
        item.type = "function_call"
        item.call_id = "call_1"
        item.name = "search"
        event = MagicMock()
        event.type = "response.output_item.added"
        event.item = item
        chunk = openai_provider._process_stream_event(event)
        assert chunk is not None
        assert chunk.type == "tool_call_start"
        assert chunk.tool_call is not None
        assert chunk.tool_call.name == "search"

    def test_function_arguments_delta(self, openai_provider: OpenAIProvider) -> None:
        event = MagicMock()
        event.type = "response.function_call_arguments.delta"
        event.call_id = "call_1"
        event.delta = '{"q":'
        chunk = openai_provider._process_stream_event(event)
        assert chunk is not None
        assert chunk.type == "tool_call_delta"
        assert chunk.arguments_delta == '{"q":'

    def test_reasoning_summary_delta(self, openai_provider: OpenAIProvider) -> None:
        event = MagicMock()
        event.type = "response.reasoning_summary_text.delta"
        event.delta = "Considering..."
        chunk = openai_provider._process_stream_event(event)
        assert chunk is not None
        assert chunk.type == "thinking"
        assert chunk.thinking_text == "Considering..."

    def test_response_completed_with_tool_calls(self, openai_provider: OpenAIProvider) -> None:
        """A completed response with function_call output items normalizes
        to 'tool_calls' rather than 'stop'."""
        fc_item = MagicMock()
        fc_item.type = "function_call"

        resp = MagicMock()
        resp.status = "completed"
        resp.output = [fc_item]
        resp.usage.input_tokens = 10
        resp.usage.output_tokens = 5
        resp.usage.reasoning_tokens = 0
        resp.incomplete_details = None

        event = MagicMock()
        event.type = "response.completed"
        event.response = resp

        chunk = openai_provider._process_stream_event(event)
        assert chunk is not None
        assert chunk.type == "done"
        assert chunk.finish_reason == "tool_calls"

    def test_response_completed_max_tokens_normalized(
        self, openai_provider: OpenAIProvider
    ) -> None:
        details = MagicMock()
        details.reason = "max_output_tokens"
        resp = MagicMock()
        resp.status = "incomplete"
        resp.output = []
        resp.usage.input_tokens = 10
        resp.usage.output_tokens = 100
        resp.usage.reasoning_tokens = 0
        resp.incomplete_details = details

        event = MagicMock()
        event.type = "response.completed"
        event.response = resp

        chunk = openai_provider._process_stream_event(event)
        assert chunk is not None
        assert chunk.type == "done"
        assert chunk.finish_reason == "length"
