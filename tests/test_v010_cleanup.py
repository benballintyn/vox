"""Tests for the v0.1.0 cleanup changes.

Covers:
    - normalize_finish_reason: cross-provider native → normalized mapping
    - Message.is_error and ToolResult.to_message() propagation
    - CompletionResponse.response_id population
    - OpenAI: stop param is dropped, previous_response_id + store flow through
    - RateLimitError.retry_after extraction from response headers
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vox import (
    Message,
    ProviderConfig,
    RateLimitError,
    ToolResult,
    normalize_finish_reason,
)
from vox.providers._chat_completions import _extract_retry_after as cc_extract_retry_after
from vox.providers.anthropic import (
    AnthropicProvider,
)
from vox.providers.anthropic import (
    _extract_retry_after as anthropic_extract_retry_after,
)
from vox.providers.openai import (
    OpenAIProvider,
)
from vox.providers.openai import (
    _extract_retry_after as openai_extract_retry_after,
)


class TestFinishReasonNormalization:
    """The normalize_finish_reason helper handles each provider's native vocab."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            # OpenAI Chat Completions + Responses API
            ("stop", "stop"),
            ("length", "length"),
            ("max_output_tokens", "length"),
            ("tool_calls", "tool_calls"),
            ("function_call", "tool_calls"),
            ("content_filter", "content_filter"),
            # Anthropic
            ("end_turn", "stop"),
            ("max_tokens", "length"),
            ("tool_use", "tool_calls"),
            ("refusal", "content_filter"),
            # Gemini (after .lower() preprocessing)
            ("safety", "content_filter"),
            ("recitation", "content_filter"),
            ("prohibited_content", "content_filter"),
            # Stop sequence
            ("stop_sequence", "stop_sequence"),
        ],
    )
    def test_known_values_map_correctly(self, raw: str, expected: str) -> None:
        assert normalize_finish_reason(raw) == expected

    def test_case_insensitive(self) -> None:
        """Gemini returns uppercase enums; normalization is case-insensitive."""
        assert normalize_finish_reason("STOP") == "stop"
        assert normalize_finish_reason("MAX_TOKENS") == "length"

    def test_unknown_value_maps_to_other(self) -> None:
        assert normalize_finish_reason("some_weird_reason") == "other"

    def test_none_passes_through(self) -> None:
        assert normalize_finish_reason(None) is None

    def test_whitespace_tolerated(self) -> None:
        assert normalize_finish_reason("  stop  ") == "stop"


class TestMessageIsError:
    """Tool result error signaling via Message.is_error."""

    def test_message_default_is_not_error(self) -> None:
        m = Message(role="tool", content="ok", tool_call_id="t1")
        assert m.is_error is False

    def test_message_can_be_marked_error(self) -> None:
        m = Message(role="tool", content="boom", tool_call_id="t1", is_error=True)
        assert m.is_error is True

    def test_tool_result_to_message_propagates_is_error(self) -> None:
        result = ToolResult(
            tool_call_id="t1",
            name="search",
            content="Connection timeout",
            is_error=True,
        )
        msg = result.to_message()
        assert msg.is_error is True

    def test_tool_result_to_message_default_no_error(self) -> None:
        result = ToolResult(tool_call_id="t1", name="search", content="ok")
        msg = result.to_message()
        assert msg.is_error is False


class TestAnthropicIsErrorTranslation:
    """Anthropic's tool_result block uses Message.is_error, not name heuristics."""

    @pytest.fixture
    def provider(self) -> AnthropicProvider:
        return AnthropicProvider(
            ProviderConfig(api_key="sk-ant-test", default_model="claude-sonnet-4-20250514")
        )

    def test_is_error_true_sets_flag(self, provider: AnthropicProvider) -> None:
        messages = [
            Message(
                role="tool",
                content="failed",
                tool_call_id="tu_1",
                name="search",
                is_error=True,
            )
        ]
        translated, _ = provider._translate_messages(messages)
        block = translated[0]["content"][0]
        assert block["is_error"] is True

    def test_is_error_false_omits_flag(self, provider: AnthropicProvider) -> None:
        messages = [Message(role="tool", content="ok", tool_call_id="tu_1", name="search")]
        translated, _ = provider._translate_messages(messages)
        block = translated[0]["content"][0]
        # Should NOT have an is_error key when False (Anthropic default is False)
        assert "is_error" not in block

    def test_tool_named_error_handler_is_not_falsely_flagged(
        self, provider: AnthropicProvider
    ) -> None:
        """Regression: pre-v0.1.0 code marked any tool starting with 'error'
        as an error. A tool named 'error_handler' that returned successfully
        should NOT have is_error=True."""
        messages = [
            Message(
                role="tool",
                content="handled gracefully",
                tool_call_id="tu_1",
                name="error_handler",
            )
        ]
        translated, _ = provider._translate_messages(messages)
        block = translated[0]["content"][0]
        assert "is_error" not in block


class TestResponseId:
    """CompletionResponse.response_id is populated by each provider."""

    def test_anthropic_response_id(self) -> None:
        from tests.providers.test_anthropic import _make_anthropic_response

        provider = AnthropicProvider(
            ProviderConfig(api_key="sk-ant-test", default_model="claude-sonnet-4-20250514")
        )
        mock_resp = _make_anthropic_response(text="hi")
        result = provider._translate_response(mock_resp, "claude-sonnet-4-20250514")
        assert result.response_id == "msg_test_abc"

    def test_openai_response_id(self) -> None:
        from tests.providers.conftest import make_openai_responses_api_response

        provider = OpenAIProvider(ProviderConfig(api_key="sk-test", default_model="gpt-5"))
        mock_resp = make_openai_responses_api_response(content="hi")
        result = provider._translate_response(mock_resp, "gpt-5")
        assert result.response_id == "resp_test_456"


class TestOpenAIStateful:
    """OpenAI Responses API: previous_response_id, store, and dropped stop."""

    @pytest.fixture
    def provider(self) -> OpenAIProvider:
        return OpenAIProvider(ProviderConfig(api_key="sk-test", default_model="gpt-5"))

    def test_previous_response_id_flows_to_request(self, provider: OpenAIProvider) -> None:
        request = provider._build_request_kwargs(
            [Message(role="user", content="Continue")],
            model="gpt-5",
            max_tokens=4096,
            temperature=1.0,
            tools=None,
            response_schema=None,
            reasoning=None,
            previous_response_id="resp_prior_abc",
        )
        assert request["previous_response_id"] == "resp_prior_abc"

    def test_store_flag_flows_to_request(self, provider: OpenAIProvider) -> None:
        request = provider._build_request_kwargs(
            [Message(role="user", content="Hi")],
            model="gpt-5",
            max_tokens=4096,
            temperature=1.0,
            tools=None,
            response_schema=None,
            reasoning=None,
            store=False,
        )
        assert request["store"] is False

    def test_neither_param_omitted_when_unset(self, provider: OpenAIProvider) -> None:
        request = provider._build_request_kwargs(
            [Message(role="user", content="Hi")],
            model="gpt-5",
            max_tokens=4096,
            temperature=1.0,
            tools=None,
            response_schema=None,
            reasoning=None,
        )
        assert "previous_response_id" not in request
        assert "store" not in request

    def test_stop_is_dropped(self, provider: OpenAIProvider) -> None:
        """The Responses API has no stop parameter; we silently drop it."""
        request = provider._build_request_kwargs(
            [Message(role="user", content="Hi")],
            model="gpt-5",
            max_tokens=4096,
            temperature=1.0,
            tools=None,
            response_schema=None,
            reasoning=None,
            stop=["END"],
        )
        assert "stop" not in request


class TestRetryAfterExtraction:
    """RateLimitError.retry_after is pulled from response headers."""

    def test_header_present_returns_float(self) -> None:
        exc = MagicMock()
        exc.response.headers = {"retry-after": "42.5"}
        assert openai_extract_retry_after(exc) == 42.5
        assert anthropic_extract_retry_after(exc) == 42.5
        assert cc_extract_retry_after(exc) == 42.5

    def test_capitalized_header(self) -> None:
        exc = MagicMock()
        exc.response.headers = {"Retry-After": "30"}
        assert openai_extract_retry_after(exc) == 30.0

    def test_no_response_returns_none(self) -> None:
        exc = MagicMock(spec=Exception)
        # spec=Exception means no .response attribute
        assert openai_extract_retry_after(exc) is None

    def test_no_header_returns_none(self) -> None:
        exc = MagicMock()
        exc.response.headers = {}
        assert openai_extract_retry_after(exc) is None

    def test_unparseable_returns_none(self) -> None:
        exc = MagicMock()
        exc.response.headers = {"retry-after": "not-a-number"}
        assert openai_extract_retry_after(exc) is None

    def test_rate_limit_error_carries_retry_after(self) -> None:
        err = RateLimitError("rate limited", retry_after=30.0, provider="openai")
        assert err.retry_after == 30.0
