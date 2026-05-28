"""Tests for vox._callbacks — event models, dispatch helpers, LoggingHandler.

End-to-end VoxClient integration is also covered here: every entry
point should fire ``on_request`` + ``on_response`` (or ``on_error``)
with the right method label.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from vox import (
    AudioContent,
    CallbackHandler,
    ErrorEvent,
    LoggingHandler,
    Message,
    ProviderConfig,
    RequestEvent,
    ResponseEvent,
    Usage,
    VoxClient,
)
from vox._callbacks import _safe_dispatch, fire_async, fire_sync
from vox.errors import InvalidRequestError, RateLimitError

# ── Helpers ────────────────────────────────────────────────────────────


class _Recorder:
    """Test handler that records every event it receives."""

    def __init__(self) -> None:
        self.requests: list[RequestEvent] = []
        self.responses: list[ResponseEvent] = []
        self.errors: list[ErrorEvent] = []

    def on_request(self, event: RequestEvent) -> None:
        self.requests.append(event)

    def on_response(self, event: ResponseEvent) -> None:
        self.responses.append(event)

    def on_error(self, event: ErrorEvent) -> None:
        self.errors.append(event)


# ── Event models ──────────────────────────────────────────────────────


class TestRequestEvent:
    def test_to_otel_attributes_basic(self) -> None:
        event = RequestEvent(
            model="gpt-5",
            provider="openai",
            method="complete",
            request_kwargs={"max_tokens": 1024, "temperature": 0.5},
        )
        attrs = event.to_otel_attributes()
        assert attrs["gen_ai.system"] == "openai"
        assert attrs["gen_ai.operation.name"] == "chat"
        assert attrs["gen_ai.request.model"] == "gpt-5"
        assert attrs["gen_ai.request.max_tokens"] == 1024
        assert attrs["gen_ai.request.temperature"] == 0.5

    def test_to_otel_attributes_omits_missing(self) -> None:
        event = RequestEvent(
            model="claude-haiku-4-5",
            provider="anthropic",
            method="acomplete",
            request_kwargs={},
        )
        attrs = event.to_otel_attributes()
        assert "gen_ai.request.max_tokens" not in attrs
        assert "gen_ai.request.temperature" not in attrs


class TestResponseEvent:
    def test_to_otel_attributes_with_usage(self) -> None:
        event = ResponseEvent(
            model="gpt-5",
            provider="openai",
            method="complete",
            duration_ms=120.5,
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        attrs = event.to_otel_attributes()
        assert attrs["gen_ai.system"] == "openai"
        assert attrs["gen_ai.response.model"] == "gpt-5"
        assert attrs["vox.duration_ms"] == 120.5
        assert attrs["gen_ai.usage.input_tokens"] == 10
        assert attrs["gen_ai.usage.output_tokens"] == 20

    def test_to_otel_attributes_carries_vox_extras(self) -> None:
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            reasoning_tokens=5,
            cache_read_tokens=8,
            estimated_cost=0.001,
        )
        event = ResponseEvent(
            model="gpt-5",
            provider="openai",
            method="complete",
            duration_ms=10.0,
            usage=usage,
        )
        attrs = event.to_otel_attributes()
        assert attrs["vox.usage.reasoning_tokens"] == 5
        assert attrs["vox.usage.cache_read_tokens"] == 8
        assert attrs["vox.usage.estimated_cost"] == 0.001

    def test_to_otel_includes_response_id_and_finish(self) -> None:
        resp = MagicMock()
        resp.response_id = "resp_abc"
        resp.finish_reason = "stop"
        event = ResponseEvent(
            model="m",
            provider="openai",
            method="complete",
            duration_ms=1.0,
            usage=None,
            response=resp,
        )
        attrs = event.to_otel_attributes()
        assert attrs["gen_ai.response.id"] == "resp_abc"
        assert attrs["gen_ai.response.finish_reasons"] == ["stop"]


class TestErrorEvent:
    def test_to_otel_attributes(self) -> None:
        err = RateLimitError("rl", retry_after=5.0, provider="openai")
        event = ErrorEvent(
            model="gpt-5",
            provider="openai",
            method="complete",
            duration_ms=200.0,
            error=err,
        )
        attrs = event.to_otel_attributes()
        assert attrs["error.type"] == "RateLimitError"
        assert "rl" in attrs["error.message"]
        assert attrs["vox.duration_ms"] == 200.0


# ── Dispatch helpers ───────────────────────────────────────────────────


class TestSafeDispatch:
    def test_calls_existing_method(self) -> None:
        rec = _Recorder()
        event = RequestEvent(model="m", provider="openai", method="complete", request_kwargs={})
        _safe_dispatch(rec, "on_request", event)
        assert rec.requests == [event]

    def test_swallows_handler_exception(self, caplog: pytest.LogCaptureFixture) -> None:
        class _Boom:
            def on_request(self, event: RequestEvent) -> None:
                raise RuntimeError("boom")

            def on_response(self, event: ResponseEvent) -> None: ...
            def on_error(self, event: ErrorEvent) -> None: ...

        event = RequestEvent(model="m", provider="openai", method="complete", request_kwargs={})
        # Should not raise.
        _safe_dispatch(_Boom(), "on_request", event)

    def test_missing_method_is_noop(self) -> None:
        """``_safe_dispatch`` no-ops when the handler lacks the method.

        Even though the :class:`CallbackHandler` protocol declares all
        three methods, the dispatcher uses ``getattr(..., None)`` so
        consumers can ship partial implementations as a runtime
        convenience.
        """

        class _PartialHandler:
            def on_request(self, event: RequestEvent) -> None:
                pass

        # No on_response — should be a no-op, not an AttributeError.
        event = ResponseEvent(
            model="m",
            provider="openai",
            method="complete",
            duration_ms=1.0,
        )
        _safe_dispatch(_PartialHandler(), "on_response", event)  # type: ignore[arg-type]


class TestFireSync:
    def test_fires_all_handlers_in_order(self) -> None:
        a, b = _Recorder(), _Recorder()
        event = RequestEvent(model="m", provider="openai", method="complete", request_kwargs={})
        fire_sync([a, b], "on_request", event)
        assert a.requests == [event]
        assert b.requests == [event]

    def test_one_handler_failure_does_not_skip_others(self) -> None:
        class _Boom:
            def on_request(self, event: RequestEvent) -> None:
                raise RuntimeError("boom")

            def on_response(self, event: ResponseEvent) -> None: ...
            def on_error(self, event: ErrorEvent) -> None: ...

        good = _Recorder()
        event = RequestEvent(model="m", provider="openai", method="complete", request_kwargs={})
        fire_sync([_Boom(), good], "on_request", event)
        # Good handler still ran.
        assert good.requests == [event]


class TestFireAsync:
    async def test_dispatches_via_executor(self) -> None:
        import asyncio

        rec = _Recorder()
        event = RequestEvent(model="m", provider="openai", method="complete", request_kwargs={})
        fire_async([rec], "on_request", event)
        # Let the executor run.
        await asyncio.sleep(0.05)
        assert rec.requests == [event]


# ── LoggingHandler ─────────────────────────────────────────────────────


class TestLoggingHandler:
    def test_logs_each_event_at_configured_level(self, mocker: MockerFixture) -> None:
        from vox._callbacks import logger

        log = mocker.spy(logger, "log")
        handler = LoggingHandler(request_level="INFO", response_level="INFO", error_level="ERROR")

        handler.on_request(
            RequestEvent(model="m", provider="openai", method="complete", request_kwargs={})
        )
        handler.on_response(
            ResponseEvent(model="m", provider="openai", method="complete", duration_ms=1.0)
        )
        handler.on_error(
            ErrorEvent(
                model="m",
                provider="openai",
                method="complete",
                duration_ms=1.0,
                error=RateLimitError("rl", provider="openai"),
            )
        )
        # Three calls, levels respected.
        levels = [c.args[0] for c in log.call_args_list]
        assert levels == ["INFO", "INFO", "ERROR"]


# ── VoxClient integration ─────────────────────────────────────────────


class TestVoxClientIntegration:
    """Each entry point fires on_request + on_response, or on_error on failure."""

    def _make_client_with_recorder(self) -> tuple[VoxClient, _Recorder]:
        rec = _Recorder()
        client = VoxClient(openai_api_key="sk-test", callbacks=[rec])
        return client, rec

    def test_complete_fires_request_and_response(self) -> None:
        from vox.models.responses import CompletionResponse

        client, rec = self._make_client_with_recorder()
        from vox.providers.openai import OpenAIProvider

        provider = OpenAIProvider(ProviderConfig(api_key="sk-test", default_model="gpt-5"))
        # Pre-populate the cache so the client uses our hand-built provider.
        client._providers["openai"] = provider

        # Stub the actual SDK call.
        fake_response = CompletionResponse(
            message=Message(role="assistant", content="hi"),
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            provider="openai",
            model="gpt-5",
            finish_reason="stop",
        )
        # Patch the real provider's .complete method.
        provider.complete = MagicMock(return_value=fake_response)  # type: ignore[method-assign]

        result = client.complete([Message(role="user", content="hi")], model="gpt-5")
        assert result is fake_response

        # Recorder saw the lifecycle.
        assert len(rec.requests) == 1
        req = rec.requests[0]
        assert req.method == "complete"
        assert req.model == "gpt-5"
        assert req.provider == "openai"
        # Without capture_content=True, messages is stripped from kwargs.
        assert "messages" not in req.request_kwargs
        # Structural fields preserved.
        assert req.request_kwargs.get("max_tokens") == 4096

        assert len(rec.responses) == 1
        resp = rec.responses[0]
        assert resp.method == "complete"
        assert resp.duration_ms >= 0
        assert resp.usage is fake_response.usage
        # Without capture_content, response payload is None.
        assert resp.response is None

    def test_capture_content_flag_includes_payload(self) -> None:
        from vox.models.responses import CompletionResponse

        rec = _Recorder()
        client = VoxClient(openai_api_key="sk-test", callbacks=[rec], capture_content=True)
        from vox.providers.openai import OpenAIProvider

        provider = OpenAIProvider(ProviderConfig(api_key="sk-test", default_model="gpt-5"))
        client._providers["openai"] = provider
        fake_response = CompletionResponse(
            message=Message(role="assistant", content="hi"),
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            provider="openai",
            model="gpt-5",
        )
        provider.complete = MagicMock(return_value=fake_response)  # type: ignore[method-assign]

        client.complete([Message(role="user", content="hi")], model="gpt-5")
        req = rec.requests[0]
        assert "messages" in req.request_kwargs
        resp = rec.responses[0]
        assert resp.response is fake_response

    def test_complete_fires_on_error_when_provider_raises(self) -> None:
        client, rec = self._make_client_with_recorder()
        from vox.providers.openai import OpenAIProvider

        provider = OpenAIProvider(ProviderConfig(api_key="sk-test", default_model="gpt-5"))
        client._providers["openai"] = provider
        provider.complete = MagicMock(  # type: ignore[method-assign]
            side_effect=InvalidRequestError("bad", provider="openai")
        )

        with pytest.raises(InvalidRequestError):
            client.complete([Message(role="user", content="hi")], model="gpt-5")

        assert len(rec.requests) == 1
        assert len(rec.responses) == 0
        assert len(rec.errors) == 1
        err = rec.errors[0]
        assert err.method == "complete"
        assert isinstance(err.error, InvalidRequestError)
        assert err.duration_ms >= 0

    def test_handler_exception_does_not_break_call(self) -> None:
        """A buggy handler is logged and swallowed; the call still returns."""
        from vox.models.responses import CompletionResponse

        class _Boom:
            def on_request(self, event: RequestEvent) -> None:
                raise RuntimeError("telemetry handler bug")

            def on_response(self, event: ResponseEvent) -> None:
                raise RuntimeError("more bugs")

            def on_error(self, event: ErrorEvent) -> None: ...

        client = VoxClient(openai_api_key="sk-test", callbacks=[_Boom()])
        from vox.providers.openai import OpenAIProvider

        provider = OpenAIProvider(ProviderConfig(api_key="sk-test", default_model="gpt-5"))
        client._providers["openai"] = provider
        fake_response = CompletionResponse(
            message=Message(role="assistant", content="hi"),
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            provider="openai",
            model="gpt-5",
        )
        provider.complete = MagicMock(return_value=fake_response)  # type: ignore[method-assign]

        result = client.complete([Message(role="user", content="hi")], model="gpt-5")
        assert result is fake_response  # Call succeeded despite handler raising.

    async def test_acomplete_fires_async(self) -> None:
        import asyncio

        from vox.models.responses import CompletionResponse

        client, rec = self._make_client_with_recorder()
        from vox.providers.openai import OpenAIProvider

        provider = OpenAIProvider(ProviderConfig(api_key="sk-test", default_model="gpt-5"))
        client._providers["openai"] = provider
        fake_response = CompletionResponse(
            message=Message(role="assistant", content="hi"),
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            provider="openai",
            model="gpt-5",
        )
        provider.acomplete = AsyncMock(return_value=fake_response)  # type: ignore[method-assign]

        await client.acomplete([Message(role="user", content="hi")], model="gpt-5")
        # Let executor drain.
        await asyncio.sleep(0.05)
        assert len(rec.requests) == 1
        assert rec.requests[0].method == "acomplete"
        assert len(rec.responses) == 1

    def test_transcribe_fires_lifecycle(self) -> None:
        from vox.models.responses import TranscriptionResponse

        client, rec = self._make_client_with_recorder()
        from vox.providers.openai import OpenAIProvider

        provider = OpenAIProvider(ProviderConfig(api_key="sk-test", default_model="whisper-1"))
        client._providers["openai"] = provider
        provider.transcribe = MagicMock(  # type: ignore[method-assign]
            return_value=TranscriptionResponse(text="hello", provider="openai", model="whisper-1")
        )

        result = client.transcribe(
            AudioContent(data="ZmFrZQ=="), model="whisper-1", provider="openai"
        )
        assert result.text == "hello"
        assert rec.requests[0].method == "transcribe"
        assert rec.responses[0].method == "transcribe"

    def test_synthesize_fires_lifecycle(self) -> None:
        client, rec = self._make_client_with_recorder()
        from vox.providers.openai import OpenAIProvider

        provider = OpenAIProvider(ProviderConfig(api_key="sk-test", default_model="tts-1"))
        client._providers["openai"] = provider
        fake_audio = AudioContent(data="ZmFrZQ==", media_type="audio/mp3")
        provider.synthesize = MagicMock(return_value=fake_audio)  # type: ignore[method-assign]

        result = client.synthesize("Hello world", voice="alloy", model="tts-1", provider="openai")
        assert result is fake_audio
        assert rec.requests[0].method == "synthesize"
        assert rec.responses[0].method == "synthesize"


class TestCallbackHandlerProtocol:
    def test_recorder_satisfies_protocol(self) -> None:
        """isinstance check against the runtime-checkable Protocol."""
        rec = _Recorder()
        assert isinstance(rec, CallbackHandler)

    def test_logging_handler_satisfies_protocol(self) -> None:
        assert isinstance(LoggingHandler(), CallbackHandler)
