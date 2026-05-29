# Reliability (Retries + Callbacks)

vox bundles two reliability primitives consumers typically want from an LLM client: a unified retry policy that honours `RateLimitError.retry_after`, and a callback protocol for wiring telemetry without monkey-patching.

## Retries

The default policy retries up to **3 times with exponential backoff and jitter**, honouring any `retry_after` value the provider supplies on a `RateLimitError`.

### Configure at the client level

```python
from vox import RetryPolicy, VoxClient

client = VoxClient(
    retry_policy=RetryPolicy(
        max_retries=5,
        base_delay=1.0,           # first retry ~1s, then ~2s, ~4s, ...
        max_delay=30.0,           # cap any single sleep
        exponential_factor=2.0,
        jitter=0.25,              # ±25% randomization
    )
)
```

### Override per call

```python
client.complete(
    messages,
    model="gpt-5",
    retry_policy=RetryPolicy(max_retries=0),  # disable for this call
)
```

### What gets retried

Only `RateLimitError` and `ProviderError` by default — the transient-by-nature ones. `InvalidRequestError`, `AuthenticationError`, `ContentFilterError`, `ModelNotFoundError`, and non-vox exceptions propagate immediately. Customize via `RetryPolicy(retry_on=(...))`.

### `retry_after` precedence

When `RateLimitError.retry_after` is set, vox sleeps for that value (capped by `max_delay`) instead of the computed backoff. Server knows best.

### Streaming

Retries fire only **before the first chunk is yielded**. Once data has started arriving, errors propagate — replaying a partial stream would surprise the consumer.

## Callbacks (Observability Hooks)

Wire telemetry — OpenTelemetry, Langfuse, Helicone, custom logging — without monkey-patching the client.

### Quick start

```python
from vox import LoggingHandler, VoxClient

client = VoxClient(
    callbacks=[LoggingHandler()],   # built-in: logs every call via loguru
    capture_content=False,          # default: no PII in event payloads
)
```

### Three events per lifecycle

| Event       | When                                | Payload                                       |
| ----------- | ----------------------------------- | --------------------------------------------- |
| `on_request(RequestEvent)`   | Before the provider call          | `model`, `provider`, `method`, `request_kwargs` |
| `on_response(ResponseEvent)` | After a successful response       | `model`, `provider`, `method`, `duration_ms`, `usage`, `response` |
| `on_error(ErrorEvent)`       | After a failed call (post-retry)  | `model`, `provider`, `method`, `duration_ms`, `error` |

### Custom handler

```python
class CostBudgetTracker:
    def __init__(self) -> None:
        self.spend_usd = 0.0

    def on_response(self, event):
        if event.usage and event.usage.estimated_cost:
            self.spend_usd += event.usage.estimated_cost

tracker = CostBudgetTracker()
client = VoxClient(callbacks=[tracker])
```

Implement any subset of the three methods; missing methods are no-ops.

### OpenTelemetry without depending on `opentelemetry-api`

Each event ships a `to_otel_attributes()` helper returning a dict keyed by [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, etc.). Consumers wiring OTel get one-line span attribution:

```python
from opentelemetry import trace

class OTelHandler:
    def on_request(self, event):
        span = trace.get_current_span()
        span.set_attributes(event.to_otel_attributes())

    def on_response(self, event):
        span = trace.get_current_span()
        span.set_attributes(event.to_otel_attributes())
```

vox itself stays dependency-free; consumers who don't use OTel never notice the helper.

### Behaviour

- **No PII by default.** `request_kwargs` strips `messages` / `audio` / `text` / `prompt`; `response` is set to `None`. Pass `capture_content=True` to include the full payloads when every handler is trusted with sensitive data.
- **Handler exceptions are swallowed** at WARNING via `loguru` — a buggy telemetry handler never breaks the real LLM call.
- **Async paths use `loop.run_in_executor`** (fire-and-forget). Slow handlers don't block the response on `acomplete` / `astream` / `atranscribe` / `asynthesize`.

## See also

- [`RetryPolicy`](../reference/retries.md)
- [`CallbackHandler`](../reference/callbacks.md), [`RequestEvent`](../reference/callbacks.md), [`ResponseEvent`](../reference/callbacks.md), [`ErrorEvent`](../reference/callbacks.md)
- [`LoggingHandler`](../reference/callbacks.md)
