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

    def test_tool_call_args_delta_inherits_buffered_id(
        self, chat_completions_provider: ChatCompletionsProvider
    ) -> None:
        """vox#20 — later args deltas (``tc.id=None``) inherit the buffered id.

        The Chat Completions SDK only sets ``tc.id`` on the *first*
        delta for a tool call; subsequent deltas have ``tc.id is None``.
        State carried across calls supplies the id.
        """
        state: dict = {}
        # First chunk seeds the buffered id.
        first = _cc_chunk(
            tool_call_name="search",
            tool_call_args='{"q":',
            tool_call_id="call_xyz",
        )
        chat_completions_provider._translate_stream_chunk(first, state)
        assert state["current_tool_call_id"] == "call_xyz"

        # Subsequent delta has tc.id=None on the real SDK; simulate that.
        next_chunk = MagicMock()
        next_chunk.usage = None
        choice = MagicMock()
        choice.finish_reason = None
        delta = MagicMock()
        delta.content = None
        tc = MagicMock()
        tc.id = None  # the bug-triggering case
        tc.function.name = None
        tc.function.arguments = '"test"}'
        delta.tool_calls = [tc]
        choice.delta = delta
        next_chunk.choices = [choice]

        chunks = chat_completions_provider._translate_stream_chunk(next_chunk, state)
        delta_chunk = next(c for c in chunks if c.type == "tool_call_delta")
        assert delta_chunk.tool_call_id == "call_xyz"
        assert delta_chunk.arguments_delta == '"test"}'

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
        """Native 'tool_calls' normalizes to 'tool_calls' (emitted at finalize).

        finish_reason is now buffered by ``_translate_stream_chunk``; the
        ``done`` chunk emerges from ``_finalize_stream`` at end-of-stream.
        """
        state: dict = {}
        chat_completions_provider._translate_stream_chunk(
            _cc_chunk(finish_reason="tool_calls"), state
        )
        finalized = chat_completions_provider._finalize_stream(state)
        assert any(c.type == "done" and c.finish_reason == "tool_calls" for c in finalized)

    def test_length_finish_reason_normalized(
        self, chat_completions_provider: ChatCompletionsProvider
    ) -> None:
        state: dict = {}
        chat_completions_provider._translate_stream_chunk(_cc_chunk(finish_reason="length"), state)
        finalized = chat_completions_provider._finalize_stream(state)
        assert any(c.type == "done" and c.finish_reason == "length" for c in finalized)

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

    def test_usage_extracted_from_chunk_with_choices(
        self, chat_completions_provider: ChatCompletionsProvider
    ) -> None:
        """vox#27 — accept ``usage`` from a chunk that also has ``choices``.

        Direct OpenAI sends a final ``choices=[]`` chunk carrying usage,
        but OpenRouter (and others) include usage on the *same* chunk as
        the final content / ``finish_reason``. vox now accepts both
        shapes and emits exactly one ``usage`` chunk in either case,
        ordered before ``done``.
        """
        # Build a chunk that has BOTH content and usage — the OpenRouter shape.
        chunk = MagicMock()
        chunk.usage = MagicMock()
        chunk.usage.prompt_tokens = 100
        chunk.usage.completion_tokens = 50
        chunk.usage.total_tokens = 150

        choice = MagicMock()
        choice.finish_reason = "stop"
        delta = MagicMock()
        delta.content = "world"
        delta.tool_calls = None
        choice.delta = delta
        chunk.choices = [choice]

        state: dict = {}
        emitted = chat_completions_provider._translate_stream_chunk(chunk, state)
        # finish_reason is buffered, so done emerges only at finalize.
        emitted += chat_completions_provider._finalize_stream(state)
        types = [c.type for c in emitted]
        # Order must be text → usage → done.
        assert types == ["text", "usage", "done"], f"unexpected order: {types}"
        usage_chunk = next(c for c in emitted if c.type == "usage")
        assert usage_chunk.usage is not None
        assert usage_chunk.usage.total_tokens == 150

    def test_usage_emitted_only_once_across_chunks(
        self, chat_completions_provider: ChatCompletionsProvider
    ) -> None:
        """Repeated ``usage`` across chunks dedups to a single emission.

        Belt-and-suspenders for providers that report usage on both a
        content chunk and a trailing choices=[] chunk.
        """
        state: dict = {}
        # First chunk: content + usage (OpenRouter shape).
        first = MagicMock()
        first.usage = MagicMock()
        first.usage.prompt_tokens = 10
        first.usage.completion_tokens = 5
        first.usage.total_tokens = 15
        choice = MagicMock()
        choice.finish_reason = None
        delta = MagicMock()
        delta.content = "hi"
        delta.tool_calls = None
        choice.delta = delta
        first.choices = [choice]
        chunks_a = chat_completions_provider._translate_stream_chunk(first, state)
        assert any(c.type == "usage" for c in chunks_a)
        assert state["usage_emitted"] is True

        # Second chunk: trailing usage-only (direct-OpenAI shape).
        second = MagicMock()
        second.usage = MagicMock()
        second.usage.prompt_tokens = 10
        second.usage.completion_tokens = 5
        second.usage.total_tokens = 15
        second.choices = []
        chunks_b = chat_completions_provider._translate_stream_chunk(second, state)
        assert not any(c.type == "usage" for c in chunks_b), (
            "usage emitted twice despite state guard"
        )

    def test_finish_reason_dedupped_to_single_buffered(
        self, chat_completions_provider: ChatCompletionsProvider
    ) -> None:
        """Repeated finish_reason across chunks buffers only the first.

        Some proxied providers (OpenRouter especially) send
        ``finish_reason`` on more than one chunk near end-of-stream.
        The translator buffers only the first sighting; subsequent
        sightings are silently dropped. ``_finalize_stream`` then emits
        exactly one ``done`` chunk per stream regardless.
        """
        state: dict = {}
        first = chat_completions_provider._translate_stream_chunk(
            _cc_chunk(finish_reason="stop"), state
        )
        # Buffering — no done emitted yet on the per-chunk path.
        assert not any(c.type == "done" for c in first)
        assert state["buffered_finish_reason"] == "stop"

        second = chat_completions_provider._translate_stream_chunk(
            _cc_chunk(finish_reason="length"), state
        )
        assert not any(c.type == "done" for c in second)
        # First-write-wins: the second finish_reason doesn't clobber.
        assert state["buffered_finish_reason"] == "stop"

        finalized = chat_completions_provider._finalize_stream(state)
        done_chunks = [c for c in finalized if c.type == "done"]
        assert len(done_chunks) == 1, f"expected exactly one done at finalize; got {finalized}"
        assert done_chunks[0].finish_reason == "stop"

    def test_usage_emitted_before_done_when_usage_arrives_after(
        self, chat_completions_provider: ChatCompletionsProvider
    ) -> None:
        """vox#27 — usage on a trailing chunk still lands before ``done``.

        OpenRouter (and direct OpenAI with the deprecated include_usage)
        delivers ``usage`` on a chunk AFTER the finish_reason chunk.
        By buffering finish_reason and flushing ``done`` at
        ``_finalize_stream``, vox ensures the cross-provider contract
        ``text → usage → done`` holds even in that arrival order.
        """
        state: dict = {}
        emitted: list = []
        # Final content chunk with finish_reason but no usage.
        emitted += chat_completions_provider._translate_stream_chunk(
            _cc_chunk(content="bye", finish_reason="stop"), state
        )
        # Trailing usage-only chunk.
        emitted += chat_completions_provider._translate_stream_chunk(
            _cc_chunk(usage=(100, 50, 150)), state
        )
        # End-of-stream flush.
        emitted += chat_completions_provider._finalize_stream(state)

        types = [c.type for c in emitted]
        assert types == ["text", "usage", "done"], f"expected text → usage → done; got {types}"


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
    """Tests for the Anthropic stream event translator.

    The translator is stateful across events (vox#19 + vox#20), so each
    test threads an explicit ``state`` dict and the result is a list.
    """

    def test_text_delta(self, anthropic_provider: AnthropicProvider) -> None:
        delta = MagicMock()
        delta.type = "text_delta"
        delta.text = "Hello"
        chunks = anthropic_provider._process_stream_event(
            _anthropic_event("content_block_delta", delta=delta), {}
        )
        assert len(chunks) == 1
        assert chunks[0].type == "text"
        assert chunks[0].text == "Hello"

    def test_thinking_delta(self, anthropic_provider: AnthropicProvider) -> None:
        delta = MagicMock()
        delta.type = "thinking_delta"
        delta.thinking = "Let me reason..."
        chunks = anthropic_provider._process_stream_event(
            _anthropic_event("content_block_delta", delta=delta), {}
        )
        assert len(chunks) == 1
        assert chunks[0].type == "thinking"
        assert chunks[0].thinking_text == "Let me reason..."

    def test_tool_use_block_start(self, anthropic_provider: AnthropicProvider) -> None:
        block = MagicMock()
        block.type = "tool_use"
        block.id = "tc_1"
        block.name = "search"
        state: dict = {}
        chunks = anthropic_provider._process_stream_event(
            _anthropic_event("content_block_start", content_block=block), state
        )
        assert len(chunks) == 1
        assert chunks[0].type == "tool_call_start"
        assert chunks[0].tool_call is not None
        assert chunks[0].tool_call.name == "search"
        # Side effect: buffers the current tool_use id for delta correlation.
        assert state["current_tool_use_id"] == "tc_1"

    def test_input_json_delta_correlates_with_current_tool_use(
        self, anthropic_provider: AnthropicProvider
    ) -> None:
        """vox#20 — ``input_json_delta`` chunks get the buffered tool_use id.

        Anthropic's argument-delta events don't carry the tool_use id
        themselves, so consumers can't correlate without the buffered
        state from the preceding ``content_block_start``.
        """
        # First, simulate the tool_use block start to seed state.
        state: dict = {}
        block = MagicMock()
        block.type = "tool_use"
        block.id = "tc_xyz"
        block.name = "search"
        anthropic_provider._process_stream_event(
            _anthropic_event("content_block_start", content_block=block), state
        )

        delta = MagicMock()
        delta.type = "input_json_delta"
        delta.partial_json = '{"q":'
        chunks = anthropic_provider._process_stream_event(
            _anthropic_event("content_block_delta", delta=delta), state
        )
        assert len(chunks) == 1
        assert chunks[0].type == "tool_call_delta"
        assert chunks[0].tool_call_id == "tc_xyz"
        assert chunks[0].arguments_delta == '{"q":'

    def test_message_delta_emits_usage_and_buffers_stop_reason(
        self, anthropic_provider: AnthropicProvider
    ) -> None:
        """vox#19 — ``message_delta`` emits usage; ``message_stop`` emits done.

        Previously vox emitted ``done`` here AND on ``message_stop`` (a
        duplicate), plus a post-stream ``usage`` chunk (after ``done``).
        New shape: ``message_delta`` → ``usage`` chunk; stop_reason
        buffered for the upcoming ``message_stop``.
        """
        # Seed input_tokens from a preceding message_start.
        state: dict = {}
        msg = MagicMock()
        msg.usage.input_tokens = 100
        anthropic_provider._process_stream_event(
            _anthropic_event("message_start", message=msg), state
        )

        delta = MagicMock()
        delta.stop_reason = "end_turn"
        usage = MagicMock()
        usage.output_tokens = 25
        chunks = anthropic_provider._process_stream_event(
            _anthropic_event("message_delta", delta=delta, usage=usage), state
        )
        assert len(chunks) == 1
        assert chunks[0].type == "usage"
        assert chunks[0].usage is not None
        assert chunks[0].usage.prompt_tokens == 100
        assert chunks[0].usage.completion_tokens == 25
        assert chunks[0].usage.total_tokens == 125
        # stop_reason buffered for message_stop.
        assert state["stop_reason"] == "end_turn"

    def test_message_stop_emits_single_done_with_buffered_reason(
        self, anthropic_provider: AnthropicProvider
    ) -> None:
        """vox#19 — ``message_stop`` is the sole emitter of ``done``."""
        state: dict = {"stop_reason": "tool_use"}
        chunks = anthropic_provider._process_stream_event(_anthropic_event("message_stop"), state)
        assert len(chunks) == 1
        assert chunks[0].type == "done"
        assert chunks[0].finish_reason == "tool_calls"

    def test_message_stop_without_buffered_reason(
        self, anthropic_provider: AnthropicProvider
    ) -> None:
        """``message_stop`` arriving without a buffered reason still emits done."""
        chunks = anthropic_provider._process_stream_event(_anthropic_event("message_stop"), {})
        assert len(chunks) == 1
        assert chunks[0].type == "done"
        assert chunks[0].finish_reason is None


# ───────────────────────────────────────────────────────────────────────────
#  OpenAI Responses API streaming
# ───────────────────────────────────────────────────────────────────────────


@pytest.fixture
def openai_provider() -> OpenAIProvider:
    return OpenAIProvider(ProviderConfig(api_key="sk-test", default_model="gpt-5"))


class TestOpenAIStreaming:
    """Tests for the OpenAI Responses API stream event translator.

    The translator is stateful across events (vox#18 + vox#20), so each
    test threads an explicit ``state`` dict and the result is a list.
    """

    def test_output_text_delta(self, openai_provider: OpenAIProvider) -> None:
        event = MagicMock()
        event.type = "response.output_text.delta"
        event.delta = "Hello"
        chunks = openai_provider._process_stream_event(event, {})
        assert len(chunks) == 1
        assert chunks[0].type == "text"
        assert chunks[0].text == "Hello"

    def test_function_call_added_buffers_id_map(self, openai_provider: OpenAIProvider) -> None:
        """Start chunk emits with call_id; state buffers item_id → call_id."""
        item = MagicMock()
        item.type = "function_call"
        item.id = "fc_abc"
        item.call_id = "call_abc"
        item.name = "search"
        event = MagicMock()
        event.type = "response.output_item.added"
        event.item = item
        state: dict = {}
        chunks = openai_provider._process_stream_event(event, state)
        assert len(chunks) == 1
        assert chunks[0].type == "tool_call_start"
        assert chunks[0].tool_call is not None
        assert chunks[0].tool_call.id == "call_abc"
        assert state["item_id_to_call_id"] == {"fc_abc": "call_abc"}

    def test_function_arguments_delta_resolves_via_state(
        self, openai_provider: OpenAIProvider
    ) -> None:
        """vox#20 — argument deltas carry ``item_id`` (fc_*); resolve to call_*.

        Real SDK ``response.function_call_arguments.delta`` events have
        only ``item_id`` — not ``call_id``. vox previously read
        ``event.call_id`` and got "" on every delta, breaking the
        consumer-side correlation. The buffered ``item_id_to_call_id``
        map produced by the preceding ``output_item.added`` resolves it.
        """
        state: dict = {"item_id_to_call_id": {"fc_abc": "call_abc"}}
        event = MagicMock(spec=["type", "item_id", "delta"])
        event.type = "response.function_call_arguments.delta"
        event.item_id = "fc_abc"
        event.delta = '{"q":'
        chunks = openai_provider._process_stream_event(event, state)
        assert len(chunks) == 1
        assert chunks[0].type == "tool_call_delta"
        assert chunks[0].tool_call_id == "call_abc"
        assert chunks[0].arguments_delta == '{"q":'

    def test_reasoning_summary_delta(self, openai_provider: OpenAIProvider) -> None:
        event = MagicMock()
        event.type = "response.reasoning_summary_text.delta"
        event.delta = "Considering..."
        chunks = openai_provider._process_stream_event(event, {})
        assert len(chunks) == 1
        assert chunks[0].type == "thinking"
        assert chunks[0].thinking_text == "Considering..."

    def test_response_completed_emits_usage_then_done(
        self, openai_provider: OpenAIProvider
    ) -> None:
        """vox#18 — ``response.completed`` emits a separate ``usage`` chunk.

        Previously ``usage`` rode on the ``done`` chunk's ``usage``
        field — consumers iterating chunk types never saw a
        ``type="usage"`` chunk. Now the translator emits two chunks
        in order: usage first, then done.
        """
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

        chunks = openai_provider._process_stream_event(event, {})
        assert [c.type for c in chunks] == ["usage", "done"]
        assert chunks[0].usage is not None
        assert chunks[0].usage.prompt_tokens == 10
        assert chunks[0].usage.completion_tokens == 5
        assert chunks[1].finish_reason == "tool_calls"

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

        chunks = openai_provider._process_stream_event(event, {})
        done = next(c for c in chunks if c.type == "done")
        assert done.finish_reason == "length"
