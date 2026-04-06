"""Reasoning/thinking configuration and response types."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ReasoningConfig(BaseModel):
    """Configuration for reasoning/thinking tokens.

    Provider-specific translation:
        - OpenAI o-series: ``level`` maps to ``reasoning.effort``
        - Anthropic: ``budget_tokens`` maps to ``thinking.budget_tokens``
        - Gemini 2.5: ``budget_tokens`` maps to ``thinkingBudget``
        - Gemini 3+: ``level`` maps to ``thinkingLevel``

    Args:
        enabled: Whether reasoning is enabled.
        budget_tokens: Maximum tokens for thinking (Anthropic, Gemini 2.5).
        level: Thinking level (OpenAI o-series reasoning_effort, Gemini 3 thinkingLevel).
    """

    enabled: bool = True
    budget_tokens: int | None = None
    level: Literal["low", "medium", "high"] | None = None


class ThinkingBlock(BaseModel):
    """A thinking/reasoning block from the model's response.

    Args:
        text: The thinking text content.
        token_count: Number of tokens used for this thinking block, if available.
    """

    text: str
    token_count: int | None = None
