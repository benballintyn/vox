"""Callback hooks for request / response / error events.

Lets consumers wire telemetry (OpenTelemetry, Langfuse, Helicone,
DataDog, custom logging) into vox without monkey-patching. The
:class:`CallbackHandler` protocol defines three methods; pass a list
of implementations to :class:`vox.VoxClient` and vox fires them at the
right moments.

Design notes:

* **Sync protocol only** for v1. From async paths (``acomplete`` /
  ``astream`` / etc.) vox schedules handler calls via
  ``loop.run_in_executor`` so a slow telemetry handler never blocks
  the LLM response. If async-native handlers turn out to be needed,
  we can add an ``AsyncCallbackHandler`` later.
* **Handler exceptions are swallowed** at WARNING level — a buggy
  telemetry handler must never break the real call.
* **No PII by default.** ``RequestEvent`` and ``ResponseEvent`` carry
  the request/response *summaries*; the actual prompt text and reply
  content are only included when the consumer opts in via
  ``VoxClient(capture_content=True)``. Matches every responsible
  telemetry library.
* **OTel-friendly without depending on opentelemetry-api.** Each
  event ships a :meth:`to_otel_attributes` helper returning a dict
  keyed by the OpenTelemetry GenAI semantic-convention attribute
  names (``gen_ai.system`` / ``gen_ai.request.model`` /
  ``gen_ai.usage.input_tokens`` / etc.). Consumers who use OTel can
  ``span.set_attributes(event.to_otel_attributes())`` and they're
  done. Consumers who don't never notice the helper.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel, ConfigDict

from .errors import VoxError

# Method name → OTel `gen_ai.operation.name` value. Per the OTel
# GenAI semantic conventions
# (https://opentelemetry.io/docs/specs/semconv/gen-ai/).
_OTEL_OPERATION_FOR: dict[str, str] = {
    "complete": "chat",
    "acomplete": "chat",
    "stream": "chat",
    "astream": "chat",
    "transcribe": "audio.transcription",
    "atranscribe": "audio.transcription",
    "synthesize": "audio.speech",
    "asynthesize": "audio.speech",
}


Method = Literal[
    "complete",
    "acomplete",
    "stream",
    "astream",
    "transcribe",
    "atranscribe",
    "synthesize",
    "asynthesize",
]


class _EventBase(BaseModel):
    """Shared base for callback event payloads."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RequestEvent(_EventBase):
    """Fires before a request goes out.

    Args:
        model: Model identifier the consumer requested.
        provider: Resolved provider name.
        method: Which entry-point method is firing the event.
        request_kwargs: Summary of the call kwargs. By default omits
            the actual prompt text / audio bytes / images — only
            included when the client was constructed with
            ``capture_content=True``. Always includes structural
            fields like ``max_tokens``, ``temperature``, tool names.
    """

    model: str
    provider: str
    method: Method
    request_kwargs: dict[str, Any]

    def to_otel_attributes(self) -> dict[str, Any]:
        """Render as a dict keyed by OTel `gen_ai.*` attribute names."""
        attrs: dict[str, Any] = {
            "gen_ai.system": self.provider,
            "gen_ai.operation.name": _OTEL_OPERATION_FOR[self.method],
            "gen_ai.request.model": self.model,
        }
        for key in ("max_tokens", "temperature", "top_p"):
            if key in self.request_kwargs and self.request_kwargs[key] is not None:
                attrs[f"gen_ai.request.{key}"] = self.request_kwargs[key]
        return attrs


class ResponseEvent(_EventBase):
    """Fires after a successful response.

    Args:
        model: Model identifier used.
        provider: Resolved provider name.
        method: Which entry-point method is firing.
        duration_ms: Wall-clock duration from request-start to
            response-received, in milliseconds. Measured via
            ``time.perf_counter`` (monotonic).
        usage: Token usage if the response carried it. ``None`` for
            audio responses where usage isn't reported (e.g. OpenAI
            Whisper, which is priced per audio second).
        response: The vox response object (``CompletionResponse`` /
            ``TranscriptionResponse`` / ``AudioContent`` /
            ``StreamChunk`` for the terminal usage chunk).
    """

    model: str
    provider: str
    method: Method
    duration_ms: float
    usage: Any = None  # Usage | None — kept loose to avoid circular imports
    response: Any = None

    def to_otel_attributes(self) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            "gen_ai.system": self.provider,
            "gen_ai.operation.name": _OTEL_OPERATION_FOR[self.method],
            "gen_ai.response.model": self.model,
        }
        # Duration in ms — OTel doesn't have a canonical attribute for
        # this on spans (it's the span duration itself), but consumers
        # logging events outside a span often want it explicitly.
        attrs["vox.duration_ms"] = self.duration_ms

        # Usage tokens per OTel semantic conventions.
        if self.usage is not None:
            for src_attr, otel_attr in (
                ("prompt_tokens", "gen_ai.usage.input_tokens"),
                ("completion_tokens", "gen_ai.usage.output_tokens"),
            ):
                val = getattr(self.usage, src_attr, None)
                if val:
                    attrs[otel_attr] = val
            # vox-specific extras that don't have OTel attrs yet.
            for extra in (
                "reasoning_tokens",
                "cache_read_tokens",
                "cache_creation_tokens",
                "estimated_cost",
            ):
                val = getattr(self.usage, extra, None)
                if val:
                    attrs[f"vox.usage.{extra}"] = val

        # Response-ID and finish reasons when present.
        response_id = getattr(self.response, "response_id", None)
        if response_id:
            attrs["gen_ai.response.id"] = response_id
        finish = getattr(self.response, "finish_reason", None)
        if finish:
            attrs["gen_ai.response.finish_reasons"] = [finish]
        return attrs


class ErrorEvent(_EventBase):
    """Fires when a call raises (including after exhausted retries).

    Args:
        model: Model identifier the consumer requested.
        provider: Resolved provider name.
        method: Which entry-point method was running.
        duration_ms: Wall-clock from request-start to error-raised.
        error: The :class:`VoxError` that propagated. Provider-side
            exceptions are normalised to a VoxError subclass before
            this fires.
    """

    model: str
    provider: str
    method: Method
    duration_ms: float
    error: VoxError

    def to_otel_attributes(self) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            "gen_ai.system": self.provider,
            "gen_ai.operation.name": _OTEL_OPERATION_FOR[self.method],
            "gen_ai.request.model": self.model,
            "vox.duration_ms": self.duration_ms,
            # OTel uses error.type / error.message on the span event.
            "error.type": type(self.error).__name__,
            "error.message": str(self.error),
        }
        return attrs


@runtime_checkable
class CallbackHandler(Protocol):
    """Telemetry hook protocol.

    Implement any subset of the three methods. vox calls only the
    methods that are defined on the handler instance; missing methods
    are no-ops.

    Handlers should be **fast and non-blocking**. From async paths,
    vox dispatches each call via ``loop.run_in_executor`` so a
    handler doing blocking I/O won't stall the LLM response — but a
    very slow sync handler called from a sync path will still delay
    the caller. Telemetry backends that do network I/O should buffer
    and flush asynchronously, or accept the cost of being on the hot
    path.

    Exceptions raised from a handler are caught and logged at
    WARNING level; they never propagate.
    """

    def on_request(self, event: RequestEvent) -> None:
        """Called immediately before the provider call is dispatched."""
        ...

    def on_response(self, event: ResponseEvent) -> None:
        """Called after a successful response is received."""
        ...

    def on_error(self, event: ErrorEvent) -> None:
        """Called when a call ultimately raises (after any retries)."""
        ...


class LoggingHandler:
    """Built-in handler that logs every event via ``loguru``.

    Useful default for adding "log every LLM call" with one line:

    .. code-block:: python

        client = VoxClient(callbacks=[LoggingHandler()])

    Args:
        request_level: ``loguru`` level for :meth:`on_request`.
            Defaults to ``"DEBUG"``.
        response_level: Level for :meth:`on_response`. Defaults to
            ``"INFO"``.
        error_level: Level for :meth:`on_error`. Defaults to
            ``"WARNING"``.
    """

    def __init__(
        self,
        *,
        request_level: str = "DEBUG",
        response_level: str = "INFO",
        error_level: str = "WARNING",
    ) -> None:
        self._request_level = request_level
        self._response_level = response_level
        self._error_level = error_level

    def on_request(self, event: RequestEvent) -> None:
        logger.log(
            self._request_level,
            "vox request | provider={p} method={m} model={model}",
            p=event.provider,
            m=event.method,
            model=event.model,
        )

    def on_response(self, event: ResponseEvent) -> None:
        usage_str = ""
        if event.usage is not None:
            usage_str = (
                f" tokens={getattr(event.usage, 'total_tokens', 0)}"
                f" cost={getattr(event.usage, 'estimated_cost', None)}"
            )
        logger.log(
            self._response_level,
            "vox response | provider={p} method={m} model={model} duration={d:.0f}ms{u}",
            p=event.provider,
            m=event.method,
            model=event.model,
            d=event.duration_ms,
            u=usage_str,
        )

    def on_error(self, event: ErrorEvent) -> None:
        logger.log(
            self._error_level,
            "vox error | provider={p} method={m} model={model} duration={d:.0f}ms"
            " error_type={et} message={msg}",
            p=event.provider,
            m=event.method,
            model=event.model,
            d=event.duration_ms,
            et=type(event.error).__name__,
            msg=str(event.error),
        )


# ── Dispatch helpers (internal) ────────────────────────────────────────


def _safe_dispatch(handler: CallbackHandler, method: str, event: Any) -> None:
    """Call a handler method and swallow any exception to WARNING.

    A handler that doesn't define the method is a no-op (handlers can
    implement any subset of on_request / on_response / on_error).
    """
    fn = getattr(handler, method, None)
    if fn is None:
        return
    try:
        fn(event)
    except Exception as e:
        logger.warning(
            "vox callback handler {h}.{m} raised {et}: {msg}",
            h=type(handler).__name__,
            m=method,
            et=type(e).__name__,
            msg=e,
        )


def fire_sync(handlers: list[CallbackHandler], method: str, event: Any) -> None:
    """Fire ``method`` on each handler synchronously. For sync paths."""
    for h in handlers:
        _safe_dispatch(h, method, event)


def fire_async(handlers: list[CallbackHandler], method: str, event: Any) -> None:
    """Fire ``method`` on each handler from an async caller.

    Uses ``loop.run_in_executor`` to push each (sync) handler call
    onto the default thread pool, then returns immediately without
    awaiting completion. Slow telemetry handlers therefore never
    block the LLM response on async paths.
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — fall back to sync dispatch. Shouldn't
        # happen from a real async caller, but keeps the helper safe
        # to call defensively.
        fire_sync(handlers, method, event)
        return

    for h in handlers:
        loop.run_in_executor(None, _safe_dispatch, h, method, event)
