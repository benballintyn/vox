# Error Handling

vox normalizes provider-specific SDK exceptions to a consistent hierarchy. Catch the base class for catch-all, or the specific subclasses for targeted handling.

## The hierarchy

```python
from vox.errors import (
    VoxError,              # base class — catches everything below
    AuthenticationError,   # invalid/missing API key
    RateLimitError,        # rate limited (has .retry_after)
    QuotaExceededError,    # billing/quota limit
    InvalidRequestError,   # malformed request, unsupported feature
    ProviderError,         # server error (5xx, transient)
    ContentFilterError,    # safety system blocked content
    ModelNotFoundError,    # model doesn't exist on this provider
)
```

## Usage

```python
from vox.errors import (
    AuthenticationError,
    QuotaExceededError,
    RateLimitError,
    VoxError,
)

try:
    response = client.complete(messages, model="gpt-5")
except RateLimitError as e:
    print(f"Rate limited by {e.provider}, retry after {e.retry_after}s")
except QuotaExceededError as e:
    print(f"Out of budget on {e.provider}: {e}")
except AuthenticationError as e:
    print(f"Auth failed for {e.provider}: {e}")
except VoxError as e:
    print(f"LLM error: {e}")
```

Every error carries `.provider` (the provider name that raised it). `RateLimitError` additionally carries `.retry_after` (seconds, or `None` if the provider didn't supply one).

## Errors vox propagates

vox catches and normalizes SDK exceptions, but does NOT catch:

- **Validation errors from your code.** If your `Pydantic` model fails to deserialize a structured-output response, `pydantic.ValidationError` propagates verbatim.
- **`KeyboardInterrupt` / `SystemExit`.** Always.
- **Programming errors** — `TypeError` / `ValueError` from passing bad args (e.g. an entry in `tools=` that's neither `Tool` nor `dict`).

## Errors + retries

`RateLimitError` and `ProviderError` are the only error classes the default `RetryPolicy` retries. Everything else propagates immediately. See [Reliability → What gets retried](reliability.md#what-gets-retried).

## See also

- [`VoxError`](../reference/errors.md) and subclasses
- [Reliability guide](reliability.md) — retry behaviour
