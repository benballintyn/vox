# Cost Estimation

vox populates `usage.estimated_cost` on every response (and streaming `usage` chunk), computed from the token counts and a built-in price table.

## Out of the box

```python
response = client.complete(messages, model="gpt-5")

print(response.usage.prompt_tokens)         # 142
print(response.usage.completion_tokens)     # 87
print(response.usage.estimated_cost)        # 0.000714  (USD)
```

The cost is set by `VoxClient` after the provider returns — providers stay ignorant of pricing.

## Custom pricing

Override vox's built-in table for any model:

```python
from vox import ModelPricing, VoxClient

client = VoxClient(
    custom_pricing={
        "gpt-5": ModelPricing(
            input_per_million=2.50,
            output_per_million=10.00,
            cache_read_per_million=0.25,
            cache_creation_per_million=3.125,
        ),
        # Add a model vox doesn't know about
        "my-custom-finetune": ModelPricing(
            input_per_million=1.00,
            output_per_million=4.00,
        ),
    }
)
```

Entries in `custom_pricing` take precedence over `MODEL_PRICING`.

## Known limitations

- **Estimate, not authoritative.** The provider's billing is the truth. vox's table is a snapshot — providers occasionally change prices.
- **No reasoning-token surcharge.** OpenAI o-series and others charge reasoning tokens at the output rate; vox does too, but if a provider adds tiered reasoning pricing later, vox's estimate may drift.
- **Audio models aren't in `MODEL_PRICING` yet.** `whisper-1` / `tts-1` / `gpt-4o-mini-tts` produce `estimated_cost = None` until that gap is closed.

## See also

- [`Usage`](../reference/responses.md) — every field
- [`ModelPricing`](../reference/pricing.md), [`MODEL_PRICING`](../reference/pricing.md), [`estimate_cost`](../reference/pricing.md)
