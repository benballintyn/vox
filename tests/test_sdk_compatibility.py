"""Offline wire contracts for Vox against the locked major provider SDKs."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Generator
from typing import Any, cast

import anthropic
import httpx
import httpx2
import pytest
from anthropic import Anthropic, AsyncAnthropic
from google import genai
from google.genai import types
from pydantic import BaseModel

from vox import Message, Tool
from vox.models.config import ProviderConfig
from vox.models.reasoning import ReasoningConfig
from vox.providers.anthropic import AnthropicProvider
from vox.providers.gemini import GeminiProvider


class Extracted(BaseModel):
    """Synthetic structured response used to verify SDK-to-Vox validation."""

    label: str
    count: int


LOOKUP = Tool(
    name="lookup",
    description="Look up a synthetic item.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)
MESSAGES = [Message(role="system", content="Be exact."), Message(role="user", content="Hello")]


def _anthropic_payload(body: dict[str, Any]) -> dict[str, Any]:
    tools = body.get("tools") or []
    names = {tool.get("name") for tool in tools}
    if "structured_output" in names:
        content = [
            {
                "type": "tool_use",
                "id": "toolu_structured",
                "name": "structured_output",
                "input": {"label": "boiler", "count": 2},
            }
        ]
        stop_reason = "tool_use"
    elif "lookup" in names:
        content = [
            {
                "type": "tool_use",
                "id": "toolu_lookup",
                "name": "lookup",
                "input": {"query": "boiler"},
            }
        ]
        stop_reason = "tool_use"
    else:
        content = [{"type": "text", "text": "Anthropic text"}]
        stop_reason = "end_turn"
    return {
        "id": "msg_contract",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 11,
            "output_tokens": 4,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def _anthropic_sse() -> str:
    """Return a complete Anthropic Messages SSE response with split usage."""
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_stream",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 17, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "streamed text"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 3},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


@pytest.fixture
def anthropic_provider() -> Generator[tuple[AnthropicProvider, list[dict[str, Any]]], None, None]:
    """Inject a real Anthropic SDK client backed only by httpx2 MockTransport."""
    requests: list[dict[str, Any]] = []

    def response_body(request: Any) -> tuple[dict[str, Any], bool]:
        body = json.loads(request.content)
        requests.append(body)
        return body, bool(body.get("stream"))

    def respond_httpx(request: httpx.Request) -> httpx.Response:
        body, streaming = response_body(request)
        if streaming:
            return httpx.Response(
                200,
                text=_anthropic_sse(),
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        return httpx.Response(200, json=_anthropic_payload(body), request=request)

    def respond_httpx2(request: httpx2.Request) -> httpx2.Response:
        body, streaming = response_body(request)
        if streaming:
            return httpx2.Response(
                200,
                text=_anthropic_sse(),
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        return httpx2.Response(200, json=_anthropic_payload(body), request=request)

    http_client = (
        httpx.Client(transport=httpx.MockTransport(respond_httpx))
        if anthropic.__version__.startswith("0.")
        else httpx2.Client(transport=httpx2.MockTransport(respond_httpx2))
    )
    sdk = Anthropic(api_key="synthetic-key", max_retries=0, http_client=cast(Any, http_client))
    provider = AnthropicProvider(
        ProviderConfig(api_key="synthetic-key", default_model="claude-sonnet-4-5", max_retries=0)
    )
    provider._sync_client = sdk
    try:
        yield provider, requests
    finally:
        sdk.close()


def _gemini_payload(body: dict[str, Any]) -> dict[str, Any]:
    part: dict[str, Any]
    config = body.get("generationConfig") or {}
    tools = body.get("tools") or []
    declarations = [
        declaration for tool in tools for declaration in tool.get("functionDeclarations", [])
    ]
    if config.get("responseMimeType") == "application/json":
        part = {"text": '{"label":"boiler","count":2}'}
    elif any(item.get("name") == "lookup" for item in declarations):
        part = {"functionCall": {"name": "lookup", "args": {"query": "boiler"}}}
    else:
        part = {"text": "Gemini text"}
    return {
        "candidates": [
            {
                "content": {"parts": [part], "role": "model"},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 13,
            "candidatesTokenCount": 5,
            "totalTokenCount": 18,
            "thoughtsTokenCount": 0,
        },
        "modelVersion": "gemini-2.5-flash",
        "responseId": "gemini-contract",
    }


@pytest.fixture
def gemini_provider() -> Generator[tuple[GeminiProvider, list[dict[str, Any]]], None, None]:
    """Inject a real google-genai client backed only by httpx MockTransport."""
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json=_gemini_payload(body), request=request)

    http_client = httpx.Client(transport=httpx.MockTransport(respond))
    sdk = genai.Client(
        api_key="synthetic-key",
        http_options=types.HttpOptions(httpx_client=http_client),
    )
    provider = GeminiProvider(
        ProviderConfig(api_key="synthetic-key", default_model="gemini-2.5-flash", max_retries=0)
    )
    provider._client = sdk
    try:
        yield provider, requests
    finally:
        sdk.close()


@pytest.mark.parametrize(
    ("fixture_name", "expected_text", "prompt_tokens", "completion_tokens"),
    [
        ("anthropic_provider", "Anthropic text", 11, 4),
        ("gemini_provider", "Gemini text", 13, 5),
    ],
)
def test_real_sdk_text_completion_and_usage(
    fixture_name: str,
    expected_text: str,
    prompt_tokens: int,
    completion_tokens: int,
    request: pytest.FixtureRequest,
) -> None:
    """A real SDK serializes the Vox request and validates its response model."""
    provider, captured = request.getfixturevalue(fixture_name)

    response = provider.complete(MESSAGES, max_tokens=123)

    assert response.message.text == expected_text
    assert response.usage.prompt_tokens == prompt_tokens
    assert response.usage.completion_tokens == completion_tokens
    assert response.usage.total_tokens == prompt_tokens + completion_tokens
    assert captured[0]


@pytest.mark.parametrize("fixture_name", ["anthropic_provider", "gemini_provider"])
def test_real_sdk_tool_request_and_response(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Tool schema reaches the wire and the real SDK response maps back to Vox."""
    provider, captured = request.getfixturevalue(fixture_name)

    response = provider.complete(MESSAGES, tools=[LOOKUP])

    assert response.finish_reason == "tool_calls"
    assert response.message.tool_calls is not None
    assert response.message.tool_calls[0].name == "lookup"
    assert response.message.tool_calls[0].arguments == {"query": "boiler"}
    wire = captured[0]
    assert wire.get("tools")


@pytest.mark.parametrize("fixture_name", ["anthropic_provider", "gemini_provider"])
def test_real_sdk_structured_output_round_trip(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Vox's provider-native structured request validates into the Pydantic schema."""
    provider, captured = request.getfixturevalue(fixture_name)

    response = provider.complete(MESSAGES, response_schema=Extracted)

    assert response.parsed == Extracted(label="boiler", count=2)
    wire = captured[0]
    if fixture_name == "anthropic_provider":
        assert any(tool["name"] == "structured_output" for tool in wire["tools"])
    else:
        assert wire["generationConfig"]["responseMimeType"] == "application/json"


def test_anthropic_sampling_moves_to_wire_body_with_explicit_precedence(
    anthropic_provider: tuple[AnthropicProvider, list[dict[str, Any]]],
) -> None:
    """Caller extra_body wins without mutation while SDK 1 accepts the call."""
    provider, captured = anthropic_provider
    explicit = {"temperature": 0.25, "top_p": 0.8}

    provider.complete(MESSAGES, temperature=0.9, top_p=0.7, extra_body=explicit)

    assert explicit == {"temperature": 0.25, "top_p": 0.8}
    assert captured[0]["temperature"] == 0.25
    assert captured[0]["top_p"] == 0.8


def test_anthropic_sync_stream_wire_and_terminal_translation(
    anthropic_provider: tuple[AnthropicProvider, list[dict[str, Any]]],
) -> None:
    """The real SDK stream accepts moved sampling and Vox emits one terminal result."""
    provider, captured = anthropic_provider

    chunks = list(
        provider.stream(
            MESSAGES,
            temperature=0.4,
            top_k=7,
            extra_body={"top_p": 0.6, "metadata_extension": "kept"},
        )
    )

    wire = captured[0]
    assert wire["temperature"] == 0.4
    assert wire["top_k"] == 7
    assert wire["top_p"] == 0.6
    assert wire["metadata_extension"] == "kept"
    assert [chunk.text for chunk in chunks if chunk.type == "text"] == ["streamed text"]
    usage = [chunk.usage for chunk in chunks if chunk.type == "usage"]
    assert len(usage) == 1 and usage[0] is not None and usage[0].total_tokens == 20
    done = [chunk for chunk in chunks if chunk.type == "done"]
    assert len(done) == 1 and done[0].finish_reason == "stop"


def test_anthropic_thinking_omits_sampling_and_none_extra_body_is_valid(
    anthropic_provider: tuple[AnthropicProvider, list[dict[str, Any]]],
) -> None:
    """Thinking remains mutually exclusive with sampling; None adds no empty body."""
    provider, captured = anthropic_provider

    provider.complete(MESSAGES, reasoning=ReasoningConfig(enabled=True, level="minimal"))
    provider.complete(MESSAGES, extra_body=None)

    assert "temperature" not in captured[0]
    assert captured[0]["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert captured[1]["temperature"] == 1.0


@pytest.fixture
async def async_anthropic_provider() -> AsyncGenerator[AnthropicProvider, None]:
    """Real asynchronous Anthropic SDK client on an httpx2 mock transport."""

    async def respond_httpx(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("stream"):
            return httpx.Response(
                200,
                text=_anthropic_sse(),
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        return httpx.Response(200, json=_anthropic_payload(body), request=request)

    async def respond_httpx2(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        if body.get("stream"):
            return httpx2.Response(
                200,
                text=_anthropic_sse(),
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        return httpx2.Response(200, json=_anthropic_payload(body), request=request)

    http_client = (
        httpx.AsyncClient(transport=httpx.MockTransport(respond_httpx))
        if anthropic.__version__.startswith("0.")
        else httpx2.AsyncClient(transport=httpx2.MockTransport(respond_httpx2))
    )
    sdk = AsyncAnthropic(
        api_key="synthetic-key", max_retries=0, http_client=cast(Any, http_client)
    )
    provider = AnthropicProvider(
        ProviderConfig(api_key="synthetic-key", default_model="claude-sonnet-4-5", max_retries=0)
    )
    provider._async_client = sdk
    try:
        yield provider
    finally:
        await sdk.close()


@pytest.fixture
async def async_gemini_provider() -> AsyncGenerator[GeminiProvider, None]:
    """Real asynchronous google-genai SDK client on an httpx mock transport."""

    async def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json=_gemini_payload(body), request=request)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    sdk = genai.Client(
        api_key="synthetic-key",
        http_options=types.HttpOptions(httpx_async_client=http_client),
    )
    provider = GeminiProvider(
        ProviderConfig(api_key="synthetic-key", default_model="gemini-2.5-flash", max_retries=0)
    )
    provider._client = sdk
    try:
        yield provider
    finally:
        await sdk.aio.aclose()


async def _assert_async_contracts(provider: AnthropicProvider | GeminiProvider) -> None:
    text = await provider.acomplete(MESSAGES)
    tool = await provider.acomplete(MESSAGES, tools=[LOOKUP])
    structured = await provider.acomplete(MESSAGES, response_schema=Extracted)

    assert text.usage.total_tokens > 0
    assert tool.message.tool_calls is not None and tool.message.tool_calls[0].name == "lookup"
    assert structured.parsed == Extracted(label="boiler", count=2)


async def test_real_async_anthropic_sdk_contracts(
    async_anthropic_provider: AnthropicProvider,
) -> None:
    """The patched request shape traverses Anthropic's real async SDK."""
    await _assert_async_contracts(async_anthropic_provider)

    chunks = [chunk async for chunk in async_anthropic_provider.astream(MESSAGES)]
    assert [chunk.text for chunk in chunks if chunk.type == "text"] == ["streamed text"]
    assert len([chunk for chunk in chunks if chunk.type == "done"]) == 1


async def test_real_async_gemini_sdk_contracts(async_gemini_provider: GeminiProvider) -> None:
    """The request and response shapes traverse google-genai's async SDK."""
    await _assert_async_contracts(async_gemini_provider)
