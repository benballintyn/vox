"""Tests for ``vox._pricing`` — the per-model price table + cost math."""

from __future__ import annotations

import pytest

from vox import (
    MODEL_PRICING,
    ModelPricing,
    Usage,
    estimate_cost,
    resolve_pricing,
)

# A pricing fixture used across several tests. Round numbers chosen so
# the arithmetic is obvious in the assertions.
_RATES = ModelPricing(
    input_per_million=1.0,
    output_per_million=2.0,
    cache_read_per_million=0.1,
    cache_creation_per_million=1.5,
)


class TestResolvePricing:
    """Tests for the five-layer lookup (exact / custom / vendor / prefix)."""

    def test_exact_match_in_builtin(self) -> None:
        assert resolve_pricing("gpt-5-mini") is MODEL_PRICING["gpt-5-mini"]

    def test_unknown_model_returns_none(self) -> None:
        assert resolve_pricing("some-random-model-name") is None

    def test_vendor_prefix_stripped(self) -> None:
        """OpenRouter-style ``openai/gpt-5-mini`` matches ``gpt-5-mini``."""
        assert resolve_pricing("openai/gpt-5-mini") is MODEL_PRICING["gpt-5-mini"]
        assert resolve_pricing("anthropic/claude-haiku-4-5") is MODEL_PRICING["claude-haiku-4-5"]

    def test_longest_prefix_match_for_date_stamped_variants(self) -> None:
        """Anthropic's dated variants fall back to the canonical family.

        ``claude-sonnet-4-5-20250929`` should match ``claude-sonnet-4-5``,
        not the shorter ``claude-sonnet-4``.
        """
        sonnet_4_5 = MODEL_PRICING["claude-sonnet-4-5"]
        assert resolve_pricing("claude-sonnet-4-5-20250929") is sonnet_4_5

    def test_longest_prefix_match_picks_most_specific(self) -> None:
        """``claude-3-5-sonnet-20240620`` matches ``claude-3-5-sonnet``."""
        c35s = MODEL_PRICING["claude-3-5-sonnet"]
        assert resolve_pricing("claude-3-5-sonnet-20240620") is c35s

    def test_custom_pricing_overrides_builtin(self) -> None:
        override = ModelPricing(input_per_million=999.0, output_per_million=999.0)
        custom = {"gpt-5-mini": override}
        assert resolve_pricing("gpt-5-mini", custom) is override
        # Built-in unchanged outside of this call.
        assert resolve_pricing("gpt-5-mini") is MODEL_PRICING["gpt-5-mini"]

    def test_custom_pricing_adds_unknown_model(self) -> None:
        custom = {"my-org/in-house-model": _RATES}
        assert resolve_pricing("my-org/in-house-model", custom) is _RATES

    def test_custom_pricing_layer_respects_vendor_strip(self) -> None:
        """Custom entry under bare key is found via ``openai/<key>`` too."""
        custom = {"my-shiny-model": _RATES}
        assert resolve_pricing("openai/my-shiny-model", custom) is _RATES


class TestEstimateCost:
    """Tests for the cost math itself."""

    def test_basic_input_output_cost(self) -> None:
        """1M input + 0.5M output at $1 / $2 per million = $1 + $1 = $2."""
        usage = Usage(prompt_tokens=1_000_000, completion_tokens=500_000)
        custom = {"flat": _RATES}
        cost = estimate_cost(usage, "flat", custom)
        assert cost == pytest.approx(2.0)

    def test_cache_read_billed_at_cache_rate(self) -> None:
        """``cache_read_tokens`` is charged at ``cache_read_per_million``."""
        usage = Usage(
            prompt_tokens=1_000_000,
            completion_tokens=0,
            cache_read_tokens=1_000_000,
        )
        custom = {"flat": _RATES}
        # 1M input @ $1 + 1M cache_read @ $0.1 = $1.10
        assert estimate_cost(usage, "flat", custom) == pytest.approx(1.1)

    def test_cache_creation_billed_at_creation_rate(self) -> None:
        """``cache_creation_tokens`` (Anthropic-specific) at its own rate."""
        usage = Usage(
            prompt_tokens=0,
            completion_tokens=0,
            cache_creation_tokens=1_000_000,
        )
        custom = {"flat": _RATES}
        assert estimate_cost(usage, "flat", custom) == pytest.approx(1.5)

    def test_reasoning_tokens_not_billed_separately(self) -> None:
        """``reasoning_tokens`` is a subset of ``completion_tokens``; do NOT
        bill it separately or we double-count (vox docstring contract).
        """
        usage = Usage(
            prompt_tokens=0,
            completion_tokens=500_000,
            reasoning_tokens=200_000,  # subset of completion_tokens
        )
        custom = {"flat": _RATES}
        # Only the completion_tokens count contributes; reasoning_tokens
        # adds nothing.
        # 0.5M @ $2 = $1.0
        assert estimate_cost(usage, "flat", custom) == pytest.approx(1.0)

    def test_unknown_model_returns_none(self) -> None:
        usage = Usage(prompt_tokens=100, completion_tokens=50)
        assert estimate_cost(usage, "no-such-model") is None

    def test_cache_rates_optional(self) -> None:
        """Models without cache rates: cache_* token columns are ignored."""
        no_cache = ModelPricing(input_per_million=1.0, output_per_million=2.0)
        usage = Usage(
            prompt_tokens=1_000_000,
            completion_tokens=0,
            cache_read_tokens=999_999,  # would be huge if billed
            cache_creation_tokens=999_999,
        )
        cost = estimate_cost(usage, "no-cache", {"no-cache": no_cache})
        # Only the prompt_tokens line contributes.
        assert cost == pytest.approx(1.0)

    def test_zero_usage_zero_cost(self) -> None:
        usage = Usage()  # all zeros
        custom = {"flat": _RATES}
        assert estimate_cost(usage, "flat", custom) == pytest.approx(0.0)

    def test_real_model_lookup(self) -> None:
        """Sanity check the built-in table with a real entry.

        gpt-5-mini: $0.25 input / $2.00 output per million.
        1M in + 0.5M out = $0.25 + $1.00 = $1.25.
        """
        usage = Usage(prompt_tokens=1_000_000, completion_tokens=500_000)
        assert estimate_cost(usage, "gpt-5-mini") == pytest.approx(1.25)


class TestUsageFields:
    """Tests for the new ``Usage.model`` and ``Usage.estimated_cost`` fields."""

    def test_defaults_are_none(self) -> None:
        """Both new fields default to None for backwards compatibility."""
        u = Usage(prompt_tokens=10, completion_tokens=5)
        assert u.model is None
        assert u.estimated_cost is None

    def test_can_set_both(self) -> None:
        u = Usage(
            prompt_tokens=10,
            completion_tokens=5,
            model="gpt-5-mini",
            estimated_cost=0.00001,
        )
        assert u.model == "gpt-5-mini"
        assert u.estimated_cost == pytest.approx(0.00001)


class TestVoxClientPopulates:
    """Integration of the pricing layer with VoxClient (mocked provider)."""

    def test_complete_populates_cost_via_client(self, mocker) -> None:
        """``VoxClient.complete`` annotates the returned response's usage."""
        from vox import CompletionResponse, Message, VoxClient

        client = VoxClient(openai_api_key="sk-test")

        # Build a fake provider that returns a deterministic response.
        fake_response = CompletionResponse(
            message=Message(role="assistant", content="hi"),
            usage=Usage(prompt_tokens=1_000_000, completion_tokens=500_000),
            provider="openai",
            model="gpt-5-mini",
            finish_reason="stop",
            raw_finish_reason="stop",
        )
        fake_adapter = mocker.MagicMock()
        fake_adapter.complete.return_value = fake_response
        mocker.patch.object(client, "_get_provider", return_value=fake_adapter)

        response = client.complete([Message(role="user", content="x")], model="gpt-5-mini")
        assert response.usage.model == "gpt-5-mini"
        # $0.25 + $1.00 = $1.25
        assert response.usage.estimated_cost == pytest.approx(1.25)

    def test_custom_pricing_on_client_overrides_builtin(self, mocker) -> None:
        from vox import CompletionResponse, Message, VoxClient

        override = ModelPricing(input_per_million=1000.0, output_per_million=2000.0)
        client = VoxClient(
            openai_api_key="sk-test",
            custom_pricing={"gpt-5-mini": override},
        )

        fake_response = CompletionResponse(
            message=Message(role="assistant", content="hi"),
            usage=Usage(prompt_tokens=1_000_000, completion_tokens=0),
            provider="openai",
            model="gpt-5-mini",
            finish_reason="stop",
        )
        fake_adapter = mocker.MagicMock()
        fake_adapter.complete.return_value = fake_response
        mocker.patch.object(client, "_get_provider", return_value=fake_adapter)

        response = client.complete([Message(role="user", content="x")], model="gpt-5-mini")
        # Override rate: 1M @ $1000 = $1000
        assert response.usage.estimated_cost == pytest.approx(1000.0)
