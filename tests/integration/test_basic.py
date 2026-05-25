"""Basic completion tests — the structural contract on the response shape."""

from __future__ import annotations

from vox import (
    Message,
    OpenAIReasoning,
    ReasoningConfig,
    VoxClient,
)

from .conftest import ProviderProfile


def test_complete_returns_normalized_response(profile: ProviderProfile, client: VoxClient) -> None:
    """Live ``complete`` returns a fully-populated ``CompletionResponse``.

    Asserts on shape only: role, non-empty text, normalized finish_reason,
    provider/model echoes, and that token accounting is internally consistent.
    Never asserts on model content — that's a separate smoke test.
    """
    response = client.complete(
        [Message(role="user", content="Say hi.")],
        model=profile.model,
        max_tokens=1024,
    )

    assert response.message.role == "assistant"
    assert response.message.text, "expected non-empty assistant text"
    assert response.provider == profile.name
    assert response.model  # may be echoed verbatim or normalized — just non-empty
    assert response.finish_reason == "stop"
    assert response.raw_finish_reason is not None
    assert response.usage.prompt_tokens > 0
    # `total_tokens` should account for everything the provider counted.
    # Some providers (e.g. OpenAI Responses API) report it; others compute
    # via prompt+completion. Either way it must be ≥ prompt_tokens.
    assert response.usage.total_tokens >= response.usage.prompt_tokens


async def test_acomplete_parity(profile: ProviderProfile, client: VoxClient) -> None:
    """Async ``acomplete`` produces the same shape as sync ``complete``.

    Verifies the async path's translation layer matches sync — they share
    most code but each provider has its own awaitable client call.
    """
    response = await client.acomplete(
        [Message(role="user", content="Say hi.")],
        model=profile.model,
        max_tokens=1024,
    )

    assert response.message.role == "assistant"
    assert response.message.text
    assert response.provider == profile.name
    assert response.finish_reason == "stop"


def test_pong_smoke(profile: ProviderProfile, client: VoxClient) -> None:
    """Constrained-prompt content smoke.

    The only test that asserts on model output — and only because the
    prompt is constrained enough that getting *anything* else means the
    plumbing is grossly broken (200 OK, garbage response).
    """
    response = client.complete(
        [
            Message(
                role="user",
                content="Reply with only the single word PONG and nothing else.",
            )
        ],
        model=profile.model,
        max_tokens=1024,
    )
    assert "pong" in response.message.text.lower()


def test_finish_reason_length(profile: ProviderProfile, client: VoxClient) -> None:
    """Tiny ``max_tokens`` forces ``finish_reason=="length"``.

    Verifies finish-reason *normalization against the live native value*
    — the canonical drift point. Each provider has a different native
    string (``"length"`` on OpenAI, ``"max_tokens"`` on Anthropic, etc.)
    and they all must collapse to the normalized ``"length"``.

    Models that reason unconditionally (gpt-5 family) need reasoning
    explicitly disabled — otherwise reasoning eats the tiny budget and
    the model returns no visible text, possibly with a non-length stop.
    """
    reasoning: ReasoningConfig | None = None
    if profile.requires_disable_reasoning_for_length:
        reasoning = ReasoningConfig(openai=OpenAIReasoning(effort="minimal"))

    response = client.complete(
        [Message(role="user", content="Tell me a long story about robots.")],
        model=profile.model,
        max_tokens=5,
        reasoning=reasoning,
    )
    assert response.finish_reason == "length"


def test_multi_turn_conversation(profile: ProviderProfile, client: VoxClient) -> None:
    """Multi-turn history round-trips through the provider's message format.

    Sends system + user + assistant + user — the assistant message is
    vox-authored and must serialize back into the provider's expected
    shape on the second turn.
    """
    response = client.complete(
        [
            Message(role="system", content="You are a terse assistant."),
            Message(role="user", content="What's the capital of France?"),
            Message(role="assistant", content="Paris."),
            Message(role="user", content="And of Spain?"),
        ],
        model=profile.model,
        max_tokens=1024,
    )
    assert response.message.text
    assert response.finish_reason == "stop"
