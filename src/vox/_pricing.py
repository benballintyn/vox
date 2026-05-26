"""Per-model pricing data + cost estimation.

This module owns vox's snapshot of model prices and the math for turning
a :class:`vox.Usage` into an estimated dollar cost. It is intentionally a
*snapshot* rather than a live data source — vox is a library, not a
service, so we vendor the data and consumers pin a vox version to get
the prices that came with it.

Override at runtime by passing ``custom_pricing`` to ``VoxClient`` or
``estimate_cost``; entries there override the built-in table.

Prices are USD, expressed as **dollars per million tokens** in the
public ``ModelPricing`` constructor — easier to read and matches how
providers publish rates. The math converts internally.

The pricing snapshot below is current as of **2026-05-26**, transcribed
from LiteLLM's public ``model_prices_and_context_window.json`` (MIT
licensed; data is freely usable). See
``https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json``
for the upstream source. Where LiteLLM only had a date-stamped variant
key for a model family (e.g. ``claude-3-5-sonnet-20240620``), the
canonical undated id (``claude-3-5-sonnet``) uses the same rates —
Anthropic prices a family uniformly across its dated revisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models.responses import Usage


# Snapshot date — bump when the table is refreshed.
PRICING_SNAPSHOT_DATE = "2026-05-26"


@dataclass(frozen=True)
class ModelPricing:
    """Per-million-token pricing for a single model. USD.

    Args:
        input_per_million: Dollars per million input (prompt) tokens at
            the standard, *uncached* rate.
        output_per_million: Dollars per million output (completion)
            tokens. For providers where ``reasoning_tokens`` is a subset
            of ``completion_tokens`` (OpenAI, Gemini), reasoning is
            already billed via the completion total at this rate — vox
            does not bill it separately.
        cache_read_per_million: Dollars per million tokens read from
            prompt cache. ``None`` if the provider doesn't expose
            caching (or doesn't charge separately for it). Anthropic
            and OpenAI both surface this.
        cache_creation_per_million: Dollars per million tokens written
            to prompt cache. Anthropic-specific — cache *creation* is a
            distinct, more expensive operation than cache reads. Other
            providers leave this ``None`` (their cache writes are part
            of the standard input charge, or they don't expose it).
    """

    input_per_million: float
    output_per_million: float
    cache_read_per_million: float | None = None
    cache_creation_per_million: float | None = None


# ── Built-in price snapshot ────────────────────────────────────────────
#
# Keyed by canonical model id. Family-name keys (e.g. ``gpt-5-mini``)
# match both the canonical id and any longer date-stamped/aliased id
# via the prefix-fallback in ``resolve_pricing``.

MODEL_PRICING: dict[str, ModelPricing] = {
    # ── OpenAI ────────────────────────────────────────────────────────
    "gpt-5": ModelPricing(
        input_per_million=1.25,
        output_per_million=10.0,
        cache_read_per_million=0.125,
    ),
    "gpt-5-mini": ModelPricing(
        input_per_million=0.25,
        output_per_million=2.0,
        cache_read_per_million=0.025,
    ),
    "gpt-5-nano": ModelPricing(
        input_per_million=0.05,
        output_per_million=0.40,
        cache_read_per_million=0.005,
    ),
    "gpt-5-pro": ModelPricing(
        input_per_million=15.0,
        output_per_million=120.0,
    ),
    "gpt-4o": ModelPricing(
        input_per_million=2.50,
        output_per_million=10.0,
        cache_read_per_million=1.25,
    ),
    "gpt-4o-mini": ModelPricing(
        input_per_million=0.15,
        output_per_million=0.60,
        cache_read_per_million=0.075,
    ),
    "gpt-4-turbo": ModelPricing(
        input_per_million=10.0,
        output_per_million=30.0,
    ),
    "gpt-4": ModelPricing(
        input_per_million=30.0,
        output_per_million=60.0,
    ),
    "gpt-3.5-turbo": ModelPricing(
        input_per_million=0.50,
        output_per_million=1.50,
    ),
    # OpenAI reasoning models.
    "o1": ModelPricing(
        input_per_million=15.0,
        output_per_million=60.0,
        cache_read_per_million=7.50,
    ),
    "o1-mini": ModelPricing(
        # LiteLLM only has azure / replicate variants for o1-mini;
        # the rates are consistent across those listings.
        input_per_million=1.10,
        output_per_million=4.40,
        cache_read_per_million=0.55,
    ),
    "o1-pro": ModelPricing(
        input_per_million=150.0,
        output_per_million=600.0,
    ),
    "o3": ModelPricing(
        input_per_million=2.0,
        output_per_million=8.0,
        cache_read_per_million=0.50,
    ),
    "o3-mini": ModelPricing(
        input_per_million=1.10,
        output_per_million=4.40,
        cache_read_per_million=0.55,
    ),
    "o3-pro": ModelPricing(
        input_per_million=20.0,
        output_per_million=80.0,
    ),
    "o4-mini": ModelPricing(
        input_per_million=1.10,
        output_per_million=4.40,
        cache_read_per_million=0.275,
    ),
    # ── Anthropic ─────────────────────────────────────────────────────
    "claude-opus-4-5": ModelPricing(
        input_per_million=5.0,
        output_per_million=25.0,
        cache_read_per_million=0.50,
        cache_creation_per_million=6.25,
    ),
    "claude-sonnet-4-5": ModelPricing(
        input_per_million=3.0,
        output_per_million=15.0,
        cache_read_per_million=0.30,
        cache_creation_per_million=3.75,
    ),
    "claude-haiku-4-5": ModelPricing(
        input_per_million=1.0,
        output_per_million=5.0,
        cache_read_per_million=0.10,
        cache_creation_per_million=1.25,
    ),
    "claude-opus-4": ModelPricing(
        input_per_million=15.0,
        output_per_million=75.0,
        cache_read_per_million=1.50,
        cache_creation_per_million=18.75,
    ),
    "claude-sonnet-4": ModelPricing(
        input_per_million=3.0,
        output_per_million=15.0,
        cache_read_per_million=0.30,
        cache_creation_per_million=3.75,
    ),
    "claude-3-5-sonnet": ModelPricing(
        input_per_million=3.0,
        output_per_million=15.0,
        cache_read_per_million=0.30,
        cache_creation_per_million=3.75,
    ),
    "claude-3-5-haiku": ModelPricing(
        input_per_million=0.80,
        output_per_million=4.0,
        cache_read_per_million=0.08,
        cache_creation_per_million=1.0,
    ),
    "claude-3-opus": ModelPricing(
        input_per_million=15.0,
        output_per_million=75.0,
        cache_read_per_million=1.50,
        cache_creation_per_million=18.75,
    ),
    "claude-3-sonnet": ModelPricing(
        input_per_million=3.0,
        output_per_million=15.0,
        cache_read_per_million=0.30,
        cache_creation_per_million=3.75,
    ),
    "claude-3-haiku": ModelPricing(
        input_per_million=0.25,
        output_per_million=1.25,
        cache_read_per_million=0.03,
        cache_creation_per_million=0.30,
    ),
    # ── Google Gemini ─────────────────────────────────────────────────
    "gemini-3-pro": ModelPricing(
        input_per_million=2.0,
        output_per_million=12.0,
        cache_read_per_million=0.20,
    ),
    "gemini-3-flash": ModelPricing(
        input_per_million=0.50,
        output_per_million=3.0,
        cache_read_per_million=0.05,
    ),
    "gemini-3.1-pro": ModelPricing(
        input_per_million=2.0,
        output_per_million=12.0,
        cache_read_per_million=0.20,
    ),
    "gemini-3.1-flash-lite": ModelPricing(
        input_per_million=0.25,
        output_per_million=1.50,
        cache_read_per_million=0.025,
    ),
    "gemini-2.5-pro": ModelPricing(
        input_per_million=1.25,
        output_per_million=10.0,
        cache_read_per_million=0.125,
    ),
    "gemini-2.5-flash": ModelPricing(
        input_per_million=0.30,
        output_per_million=2.50,
        cache_read_per_million=0.03,
    ),
    "gemini-2.5-flash-lite": ModelPricing(
        input_per_million=0.10,
        output_per_million=0.40,
        cache_read_per_million=0.01,
    ),
    "gemini-2.0-flash": ModelPricing(
        input_per_million=0.10,
        output_per_million=0.40,
        cache_read_per_million=0.025,
    ),
    "gemini-2.0-flash-lite": ModelPricing(
        input_per_million=0.075,
        output_per_million=0.30,
        cache_read_per_million=0.01875,
    ),
}


# ── Resolution + cost math ─────────────────────────────────────────────


def _strip_vendor_prefix(model: str) -> str:
    """Strip OpenRouter-style ``vendor/`` prefix from a model id.

    ``openai/gpt-5-mini`` → ``gpt-5-mini``. Idempotent on un-prefixed ids.
    """
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def resolve_pricing(
    model: str,
    custom_pricing: dict[str, ModelPricing] | None = None,
) -> ModelPricing | None:
    """Look up pricing for a model id, with two layers of fallback.

    Lookup order:

    1. Exact match in ``custom_pricing`` (if provided) — full override.
    2. Exact match in ``custom_pricing`` after stripping any
       ``vendor/`` prefix.
    3. Exact match in the built-in :data:`MODEL_PRICING` table.
    4. Exact match in :data:`MODEL_PRICING` after stripping
       ``vendor/`` prefix (handles OpenRouter-style ``openai/gpt-5-mini``).
    5. **Longest-prefix** match against :data:`MODEL_PRICING` keys
       (handles Anthropic's date-stamped variants like
       ``claude-sonnet-4-5-20250929`` → ``claude-sonnet-4-5``).

    Args:
        model: Model identifier as passed to vox.
        custom_pricing: Per-call / per-client override dict, keyed the
            same way as :data:`MODEL_PRICING`.

    Returns:
        A :class:`ModelPricing` instance, or ``None`` if no match in
        either table.
    """
    custom = custom_pricing or {}

    # Layers 1 + 2: custom override.
    if model in custom:
        return custom[model]
    stripped = _strip_vendor_prefix(model)
    if stripped != model and stripped in custom:
        return custom[stripped]

    # Layers 3 + 4: built-in exact match.
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    if stripped != model and stripped in MODEL_PRICING:
        return MODEL_PRICING[stripped]

    # Layer 5: longest-prefix match in MODEL_PRICING (after vendor strip).
    # Pick the longest key that ``stripped`` starts with, so
    # ``claude-sonnet-4-5-20250929`` chooses ``claude-sonnet-4-5`` over
    # ``claude-sonnet-4``.
    candidates = [k for k in MODEL_PRICING if stripped.startswith(k + "-") or stripped == k]
    if candidates:
        best = max(candidates, key=len)
        return MODEL_PRICING[best]

    return None


def estimate_cost(
    usage: Usage,
    model: str,
    custom_pricing: dict[str, ModelPricing] | None = None,
) -> float | None:
    """Estimate the USD cost of a completion from its ``Usage``.

    The math:

    * ``prompt_tokens * input_rate``
    * ``completion_tokens * output_rate``
    * ``cache_read_tokens * cache_read_rate`` (if both are present)
    * ``cache_creation_tokens * cache_creation_rate`` (if both are present)

    ``reasoning_tokens`` is intentionally **not** billed separately —
    on providers that report it (OpenAI, Gemini), it is already counted
    inside ``completion_tokens``. Billing it again would double-count.
    On providers that don't report reasoning_tokens at all (Anthropic,
    Chat Completions / OpenRouter), the field stays 0 and is harmless.

    Returns ``None`` when ``resolve_pricing`` can't find a match for
    ``model`` — pricing is best-effort, not guaranteed.

    Args:
        usage: The :class:`vox.Usage` to price.
        model: Model identifier that produced the usage.
        custom_pricing: Optional override dict (same shape as
            :data:`MODEL_PRICING`); entries here take precedence.

    Returns:
        Estimated USD cost as a ``float``, or ``None`` if the model is
        unknown to both tables.
    """
    pricing = resolve_pricing(model, custom_pricing)
    if pricing is None:
        return None

    cost = 0.0
    cost += (usage.prompt_tokens / 1_000_000.0) * pricing.input_per_million
    cost += (usage.completion_tokens / 1_000_000.0) * pricing.output_per_million
    if usage.cache_read_tokens and pricing.cache_read_per_million is not None:
        cost += (usage.cache_read_tokens / 1_000_000.0) * pricing.cache_read_per_million
    if usage.cache_creation_tokens and pricing.cache_creation_per_million is not None:
        cost += (usage.cache_creation_tokens / 1_000_000.0) * pricing.cache_creation_per_million
    return cost
