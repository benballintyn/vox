"""Reasoning/thinking configuration and response types."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# Default budget tokens for budget-based providers (Anthropic, Gemini 2.5),
# derived from the semantic level. These are sensible defaults; users who
# need precise control should populate the provider-specific sub-config.
LEVEL_TO_BUDGET_TOKENS: dict[str, int] = {
    "minimal": 1024,
    "low": 4096,
    "medium": 16384,
    "high": 32768,
}

# Mapping from semantic level to Gemini 3 thinkingLevel values.
# Gemini 3 only supports low/medium/high, so "minimal" collapses to "low".
LEVEL_TO_GEMINI3_LEVEL: dict[str, Literal["low", "medium", "high"]] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
}


class OpenAIReasoning(BaseModel):
    """OpenAI-specific reasoning options for GPT-5 and o-series models.

    Use this to access OpenAI features that don't translate cross-provider,
    such as the ``xhigh`` effort tier or reasoning summary verbosity.

    Args:
        effort: Native OpenAI reasoning effort. Overrides ``ReasoningConfig.level``
            when set. ``minimal`` was added with GPT-5 (Aug 2025) for fast responses.
        summary: Verbosity of reasoning summaries returned in the response.
            Without this, the model will reason but no summary text is exposed.
    """

    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    summary: Literal["auto", "concise", "detailed"] | None = None


class AnthropicReasoning(BaseModel):
    """Anthropic-specific extended thinking options.

    Args:
        budget_tokens: Maximum tokens to spend on thinking. Minimum 1024.
            Overrides the default budget derived from ``ReasoningConfig.level``.
    """

    budget_tokens: int | None = None


class GeminiReasoning(BaseModel):
    """Gemini-specific thinking options.

    For Gemini 2.5, use ``budget_tokens`` (maps to ``thinking_budget``).
    For Gemini 3, use ``level`` (maps to ``thinking_level``).

    Args:
        budget_tokens: Token budget for Gemini 2.5 series.
        level: Thinking level for Gemini 3 series.
    """

    budget_tokens: int | None = None
    level: Literal["low", "medium", "high"] | None = None


class ReasoningConfig(BaseModel):
    """Provider-agnostic reasoning configuration.

    The semantic ``level`` maps to each provider's native control:

    | level   | OpenAI effort | Anthropic budget | Gemini 2.5 budget | Gemini 3 level |
    |---------|---------------|------------------|-------------------|----------------|
    | minimal | minimal       | 1024             | 1024              | low            |
    | low     | low           | 4096             | 4096              | low            |
    | medium  | medium        | 16384            | 16384             | medium         |
    | high    | high          | 32768            | 32768             | high           |

    For provider-specific control (e.g. OpenAI's ``xhigh``, exact Anthropic
    token budgets, reasoning summary verbosity), populate the matching
    sub-config. Sub-config values override the bucket-based mapping for that
    provider only.

    Args:
        enabled: Whether reasoning is enabled at all.
        level: Cross-provider semantic intensity.
        openai: OpenAI-specific overrides (effort, summary).
        anthropic: Anthropic-specific overrides (budget_tokens).
        gemini: Gemini-specific overrides (budget_tokens for 2.5, level for 3).
    """

    enabled: bool = True
    level: Literal["minimal", "low", "medium", "high"] | None = None
    openai: OpenAIReasoning | None = None
    anthropic: AnthropicReasoning | None = None
    gemini: GeminiReasoning | None = None


class ThinkingBlock(BaseModel):
    """A thinking/reasoning block from the model's response.

    Args:
        text: The thinking text content.
        token_count: Number of tokens used for this thinking block, if available.
    """

    text: str
    token_count: int | None = None
