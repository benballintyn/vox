# Callbacks

Observability hooks fired around every entry point. See the [Reliability guide](../guides/reliability.md#callbacks-observability-hooks) for usage.

## CallbackHandler

::: vox.CallbackHandler

## Events

Each event ships a `to_otel_attributes()` helper keyed by [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

::: vox.RequestEvent

::: vox.ResponseEvent

::: vox.ErrorEvent

## Built-in handlers

::: vox.LoggingHandler
