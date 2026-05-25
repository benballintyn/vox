"""Reasoning / thinking integration tests.

Verifies the cross-provider ``ReasoningConfig.level`` abstraction —
``minimal``/``low``/``medium``/``high`` — maps to each provider's native
control (OpenAI ``effort``, Anthropic ``budget_tokens``, Gemini 2.5
``budget_tokens`` / Gemini 3 ``thinking_level``) without rejection by
the live API.

Output text is not asserted; the contract under test is that the
reasoning path *engages* and the response *parses*.
"""

from __future__ import annotations

import pytest

from vox import (
    AnthropicReasoning,
    Message,
    OpenAIReasoning,
    ReasoningConfig,
    VoxClient,
)

from .conftest import ProviderProfile


def test_reasoning_low_engages(reasoning_profile: ProviderProfile, client: VoxClient) -> None:
    """``ReasoningConfig(level='low')`` produces a successful, reasoned response.

    Engagement is measured by total token consumption beyond the prompt
    — the strict signal would be ``usage.reasoning_tokens > 0``, but
    providers vary in whether they break that out separately or fold it
    into completion_tokens. ``total_tokens > prompt_tokens`` is the
    portable lower bound.
    """
    response = client.complete(
        [
            Message(
                role="user",
                content="What is 47 * 83? Think step by step.",
            )
        ],
        model=reasoning_profile.model,
        reasoning=ReasoningConfig(level="low"),
        max_tokens=4096,
    )
    assert response.message.text
    assert response.usage.total_tokens > response.usage.prompt_tokens


@pytest.mark.parametrize("level", ["minimal", "low", "medium", "high"])
def test_reasoning_level_sweep(
    reasoning_profile: ProviderProfile,
    client: VoxClient,
    level: str,
) -> None:
    """All four semantic levels round-trip through the live API.

    The point is to catch translation regressions: any level that the
    provider rejects (because the mapping produced an out-of-range
    budget, an invalid effort string, etc.) fails the test.
    """
    response = client.complete(
        [Message(role="user", content="Briefly: what is 2 + 2?")],
        model=reasoning_profile.model,
        reasoning=ReasoningConfig(level=level),  # type: ignore[arg-type]
        max_tokens=4096,
    )
    # Either visible text or a recognized finish_reason — proves we got a parsed response.
    assert response.message.text or response.finish_reason is not None


def test_thinking_blocks_exposed(reasoning_profile: ProviderProfile, client: VoxClient) -> None:
    """Providers that expose thinking populate ``response.thinking``.

    Anthropic surfaces extended-thinking blocks by default; other
    providers either don't expose them (Gemini) or only on explicit
    opt-in (OpenAI ``summary="auto"``). This test only asserts on
    providers flagged ``exposes_thinking_blocks=True``.
    """
    if not reasoning_profile.exposes_thinking_blocks:
        pytest.skip(f"{reasoning_profile.name} does not expose thinking blocks by default")

    response = client.complete(
        [Message(role="user", content="What is 47 * 83? Think step by step.")],
        model=reasoning_profile.model,
        reasoning=ReasoningConfig(level="low"),
        max_tokens=4096,
    )
    assert response.thinking is not None
    assert len(response.thinking) > 0
    assert response.thinking[0].text


def test_anthropic_escape_hatch(client: VoxClient) -> None:
    """Anthropic-specific override: explicit ``budget_tokens`` is honored.

    Verifies the escape-hatch path — when a user populates
    ``ReasoningConfig.anthropic`` directly, that overrides the
    level-derived default budget. Successful call proves the override
    reached the API.
    """
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    response = client.complete(
        [Message(role="user", content="What is 47 * 83?")],
        model="claude-haiku-4-5",
        reasoning=ReasoningConfig(
            level="low",
            anthropic=AnthropicReasoning(budget_tokens=2048),
        ),
        max_tokens=4096,
    )
    assert response.message.text


def test_openai_escape_hatch(client: VoxClient) -> None:
    """OpenAI-specific override: explicit ``effort`` is honored."""
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    response = client.complete(
        [Message(role="user", content="What is 47 * 83?")],
        model="gpt-5-mini",
        reasoning=ReasoningConfig(
            openai=OpenAIReasoning(effort="minimal"),
        ),
        max_tokens=2048,
    )
    assert response.message.text or response.finish_reason is not None
