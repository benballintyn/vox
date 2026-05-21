"""Tests for provider-native (server-side) tool pass-through.

Issue #8: passing a raw dict tool spec (e.g. Anthropic's web_search_20250305)
crashed the translator with a cryptic AttributeError. vox now accepts
``Tool | dict`` entries in ``tools``: vox Tools are translated, raw dicts are
passed through verbatim, and anything else raises a clear TypeError.
"""

from __future__ import annotations

import pytest

from vox import Message, ProviderConfig, Tool
from vox.providers.anthropic import AnthropicProvider
from vox.providers.gemini import GeminiProvider
from vox.providers.openai import OpenAIProvider
from vox.providers.openrouter import OpenRouterProvider

# A representative function tool and a representative provider-native tool.
FUNCTION_TOOL = Tool(
    name="get_weather",
    description="Get the weather for a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)
ANTHROPIC_WEB_SEARCH = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 5,
}
OPENAI_WEB_SEARCH = {"type": "web_search_preview"}
GEMINI_GOOGLE_SEARCH = {"google_search": {}}


class TestAnthropicNativeTools:
    """Anthropic _translate_tools accepts dicts and Tools."""

    @pytest.fixture
    def provider(self) -> AnthropicProvider:
        return AnthropicProvider(
            ProviderConfig(api_key="sk-ant-test", default_model="claude-sonnet-4-5-20250929")
        )

    def test_function_tool_translated(self, provider: AnthropicProvider) -> None:
        result = provider._translate_tools([FUNCTION_TOOL])
        assert result == [
            {
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "input_schema": FUNCTION_TOOL.parameters,
            }
        ]

    def test_native_tool_dict_passed_through_verbatim(self, provider: AnthropicProvider) -> None:
        result = provider._translate_tools([ANTHROPIC_WEB_SEARCH])
        assert result == [ANTHROPIC_WEB_SEARCH]

    def test_mixed_list(self, provider: AnthropicProvider) -> None:
        result = provider._translate_tools([FUNCTION_TOOL, ANTHROPIC_WEB_SEARCH])
        assert len(result) == 2
        assert result[0]["name"] == "get_weather"
        assert result[1] == ANTHROPIC_WEB_SEARCH

    def test_bad_type_raises_typeerror(self, provider: AnthropicProvider) -> None:
        with pytest.raises(TypeError, match="must be a vox.Tool or a dict"):
            provider._translate_tools(["not a tool"])  # type: ignore[list-item]

    def test_issue_8_repro_builds_request(self, provider: AnthropicProvider) -> None:
        """The exact repro from issue #8 reaches a well-formed request dict
        instead of crashing with AttributeError."""
        request = provider._build_request_kwargs(
            [Message(role="user", content="What's the 10Y JGB yield?")],
            model="claude-sonnet-4-5-20250929",
            max_tokens=2048,
            temperature=1.0,
            tools=[ANTHROPIC_WEB_SEARCH],
            response_schema=None,
            reasoning=None,
            stop=None,
        )
        assert request["tools"] == [ANTHROPIC_WEB_SEARCH]


class TestOpenAINativeTools:
    """OpenAI Responses API _translate_tools accepts dicts and Tools."""

    @pytest.fixture
    def provider(self) -> OpenAIProvider:
        return OpenAIProvider(ProviderConfig(api_key="sk-test", default_model="gpt-5"))

    def test_function_tool_translated(self, provider: OpenAIProvider) -> None:
        result = provider._translate_tools([FUNCTION_TOOL])
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "get_weather"

    def test_native_tool_dict_passed_through_verbatim(self, provider: OpenAIProvider) -> None:
        result = provider._translate_tools([OPENAI_WEB_SEARCH])
        assert result == [OPENAI_WEB_SEARCH]

    def test_mixed_list(self, provider: OpenAIProvider) -> None:
        result = provider._translate_tools([FUNCTION_TOOL, OPENAI_WEB_SEARCH])
        assert len(result) == 2
        assert result[0]["type"] == "function"
        assert result[1] == OPENAI_WEB_SEARCH

    def test_bad_type_raises_typeerror(self, provider: OpenAIProvider) -> None:
        with pytest.raises(TypeError, match="must be a vox.Tool or a dict"):
            provider._translate_tools([42])  # type: ignore[list-item]


class TestChatCompletionsNativeTools:
    """OpenRouter / LM Studio (Chat Completions base) accept dicts and Tools."""

    @pytest.fixture
    def provider(self) -> OpenRouterProvider:
        return OpenRouterProvider(ProviderConfig(api_key="sk-or-test"))

    def test_function_tool_translated(self, provider: OpenRouterProvider) -> None:
        result = provider._translate_tools([FUNCTION_TOOL])
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"

    def test_native_tool_dict_passed_through_verbatim(self, provider: OpenRouterProvider) -> None:
        native = {"type": "web_search", "some_provider_field": True}
        result = provider._translate_tools([native])
        assert result == [native]

    def test_mixed_list(self, provider: OpenRouterProvider) -> None:
        native = {"type": "web_search"}
        result = provider._translate_tools([FUNCTION_TOOL, native])
        assert len(result) == 2
        assert result[0]["type"] == "function"
        assert result[1] == native

    def test_bad_type_raises_typeerror(self, provider: OpenRouterProvider) -> None:
        with pytest.raises(TypeError, match="must be a vox.Tool or a dict"):
            provider._translate_tools([None])  # type: ignore[list-item]


class TestGeminiNativeTools:
    """Gemini _translate_tools collects Tools and passes dicts through."""

    @pytest.fixture
    def provider(self) -> GeminiProvider:
        return GeminiProvider(ProviderConfig(api_key="AI-test", default_model="gemini-2.5-pro"))

    def test_function_tools_collected_into_one_tool(self, provider: GeminiProvider) -> None:
        result = provider._translate_tools([FUNCTION_TOOL])
        # One genai Tool wrapping the function declarations.
        assert len(result) == 1
        assert len(result[0].function_declarations) == 1
        assert result[0].function_declarations[0].name == "get_weather"

    def test_native_tool_dict_passed_through_verbatim(self, provider: GeminiProvider) -> None:
        result = provider._translate_tools([GEMINI_GOOGLE_SEARCH])
        assert result == [GEMINI_GOOGLE_SEARCH]

    def test_mixed_list(self, provider: GeminiProvider) -> None:
        result = provider._translate_tools([FUNCTION_TOOL, GEMINI_GOOGLE_SEARCH])
        # One collected function-declaration Tool, plus the verbatim dict.
        assert len(result) == 2
        assert len(result[0].function_declarations) == 1
        assert result[1] == GEMINI_GOOGLE_SEARCH

    def test_bad_type_raises_typeerror(self, provider: GeminiProvider) -> None:
        with pytest.raises(TypeError, match="must be a vox.Tool or a dict"):
            provider._translate_tools([("tuple", "not", "tool")])  # type: ignore[list-item]
