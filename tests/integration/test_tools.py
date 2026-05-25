"""Tool-use integration tests.

Covers the ``vox.Tool`` path (function tools). The keystone here is the
**round-trip**: take vox's normalized tool-call response, append it
plus a ``ToolResult`` to history, send a follow-up turn, and confirm
the provider accepts it cleanly. That re-serialization is where the
subtlest translation bugs hide — a one-shot test wouldn't catch them.

Provider-native server-side tool dicts (Anthropic ``web_search_*``,
OpenAI ``web_search_preview``, etc.) are deliberately deferred to a
follow-up PR — they're pricier, slower, and a distinct shape from the
``vox.Tool`` path.
"""

from __future__ import annotations

import json

import pytest

from vox import Message, Tool, ToolResult, VoxClient

from .conftest import ProviderProfile

# A single, unambiguous tool for forcing tool-call behavior in the model.
WEATHER_TOOL = Tool(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The city name."},
        },
        "required": ["city"],
    },
)

# A prompt that reliably both (a) forces a tool call on turn 1 and
# (b) elicits a textual summary on turn 2 after a tool result is
# provided. Modern frontier models (gpt-5, claude haiku 4.5, gemini 3.1)
# honor it consistently. We don't pass provider-specific ``tool_choice``
# kwargs because that abstraction differs across providers — flake budget
# is absorbed by pytest-rerunfailures. Critically: the prompt does NOT
# include "do not reply with text", because the model honors that on
# turn 2 as well, producing an empty assistant reply and breaking the
# round-trip test.
FORCE_PROMPT = (
    "Use the get_weather tool to look up the weather for Paris, then "
    "summarize the result in one sentence."
)


def test_forced_tool_call(profile: ProviderProfile, client: VoxClient) -> None:
    """A forced tool call surfaces as a normalized ``ToolCallData`` entry.

    Asserts on the structural contract: ``finish_reason=="tool_calls"``,
    at least one entry on ``message.tool_calls``, correct tool name, and
    ``arguments`` parsed into a dict containing the expected key. Values
    are *not* asserted — the test verifies translation, not model quality.
    """
    response = client.complete(
        [Message(role="user", content=FORCE_PROMPT)],
        model=profile.model,
        provider=profile.name,
        tools=[WEATHER_TOOL],
        max_tokens=1024,
    )

    assert response.finish_reason == "tool_calls"
    assert response.message.tool_calls, "expected at least one tool_call"

    weather_calls = [tc for tc in response.message.tool_calls if tc.name == "get_weather"]
    assert weather_calls, (
        f"no get_weather call; got {[tc.name for tc in response.message.tool_calls]}"
    )

    args = weather_calls[0].arguments
    assert isinstance(args, dict)
    assert "city" in args, f"expected 'city' in arguments; got {args}"


def test_tool_round_trip(profile: ProviderProfile, client: VoxClient) -> None:
    """The keystone test: vox re-serializes its own tool-call output back.

    Flow:
      1. Force a tool call (turn 1).
      2. Append the assistant message (containing the vox-normalized
         ``ToolCallData``) and a ``ToolResult`` for it.
      3. Send the conversation back (turn 2).
      4. Assert turn 2 returns a clean ``"stop"`` with non-empty text.

    If vox's outbound translation of its own normalized types is broken
    — wrong field names, wrong content shape, missing tool_call_id — the
    provider rejects turn 2 and this test fails. That's the value.
    """
    history: list[Message] = [Message(role="user", content=FORCE_PROMPT)]

    turn1 = client.complete(
        history,
        model=profile.model,
        provider=profile.name,
        tools=[WEATHER_TOOL],
        max_tokens=1024,
    )
    assert turn1.finish_reason == "tool_calls"
    assert turn1.message.tool_calls

    # Append the assistant turn verbatim — this is the message vox built,
    # and the provider must accept it back on the next call.
    history.append(turn1.message)

    # Append one tool result per tool_call. The content can be anything
    # parseable; what matters is the id/name plumbing.
    for tc in turn1.message.tool_calls:
        history.append(
            ToolResult(
                tool_call_id=tc.id,
                name=tc.name,
                content='{"temperature_c": 18, "conditions": "cloudy"}',
            ).to_message()
        )

    turn2 = client.complete(
        history,
        model=profile.model,
        provider=profile.name,
        tools=[WEATHER_TOOL],
        max_tokens=1024,
    )
    assert turn2.finish_reason == "stop"
    assert turn2.message.text, "expected non-empty assistant reply after tool result"


def test_streaming_tool_call(profile: ProviderProfile, client: VoxClient) -> None:
    """Streaming a forced tool call yields tool-call chunks with parseable args.

    Every supported provider must emit at least one ``tool_call_start``
    chunk identifying the tool. Some providers also stream the arguments
    in ``tool_call_delta`` chunks (OpenAI, Anthropic); others deliver
    them whole on the start chunk (Gemini). The test accepts either path
    and verifies the final arguments — taken from start + deltas — parse
    as a JSON object containing the expected key.
    """
    if profile.name in ("openai", "anthropic", "openrouter"):
        # On OpenAI (Responses API), OpenRouter (Chat Completions via
        # _chat_completions.py), and Anthropic streaming, vox emits a
        # ``tool_call_start`` with empty arguments, then either no
        # ``tool_call_delta`` chunks or chunks whose ``tool_call_id``
        # doesn't correlate with the start. Accumulated args end up ``{}``.
        # Tracked as vox#20.
        pytest.xfail(
            "vox streaming tool_call_delta accumulation broken on OpenAI / "
            "Anthropic / OpenRouter — args end up empty — vox#20"
        )
    chunks = list(
        client.stream(
            [Message(role="user", content=FORCE_PROMPT)],
            model=profile.model,
            provider=profile.name,
            tools=[WEATHER_TOOL],
            max_tokens=1024,
        )
    )

    starts = [c for c in chunks if c.type == "tool_call_start"]
    assert starts, f"no tool_call_start chunks; got {[c.type for c in chunks]}"

    weather_start = next(
        (c for c in starts if c.tool_call and c.tool_call.name == "get_weather"),
        None,
    )
    assert weather_start is not None, "no tool_call_start for get_weather"
    assert weather_start.tool_call is not None

    # Accumulate any streamed argument deltas for this tool call id.
    call_id = weather_start.tool_call.id
    deltas = "".join(
        c.arguments_delta
        for c in chunks
        if c.type == "tool_call_delta" and c.tool_call_id == call_id
    )

    # Final arguments come either from concatenating streamed deltas
    # (token-by-token providers like OpenAI/Anthropic) or from the start
    # chunk's whole arguments (Gemini-style providers).
    args = json.loads(deltas) if deltas else weather_start.tool_call.arguments

    assert isinstance(args, dict)
    assert "city" in args, f"expected 'city' in streamed arguments; got {args}"
