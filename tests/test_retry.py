"""Tests for vox._retry — RetryPolicy + sync/async helpers + stream wrappers."""

from __future__ import annotations

import asyncio

import pytest
from pytest_mock import MockerFixture

from vox._retry import (
    RetryPolicy,
    _compute_delay,
    _should_retry,
    retry_async,
    retry_stream_async,
    retry_stream_sync,
    retry_sync,
)
from vox.errors import (
    AuthenticationError,
    ContentFilterError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
)


class TestRetryPolicy:
    def test_defaults(self) -> None:
        p = RetryPolicy()
        assert p.max_retries == 3
        assert p.base_delay == 1.0
        assert p.max_delay == 30.0
        assert p.exponential_factor == 2.0
        assert p.jitter == 0.25
        assert RateLimitError in p.retry_on
        assert ProviderError in p.retry_on

    def test_zero_retries_disables(self) -> None:
        p = RetryPolicy(max_retries=0)
        assert p.max_retries == 0


class TestShouldRetry:
    def test_whitelisted_errors_retry(self) -> None:
        p = RetryPolicy()
        assert _should_retry(RateLimitError("rl"), p)
        assert _should_retry(ProviderError("pe"), p)

    def test_non_whitelisted_errors_do_not_retry(self) -> None:
        p = RetryPolicy()
        assert not _should_retry(InvalidRequestError("ir"), p)
        assert not _should_retry(AuthenticationError("auth"), p)
        assert not _should_retry(ContentFilterError("cf"), p)

    def test_non_vox_errors_do_not_retry(self) -> None:
        p = RetryPolicy()
        assert not _should_retry(ValueError("not vox"), p)
        assert not _should_retry(RuntimeError("nope"), p)

    def test_custom_retry_on_overrides(self) -> None:
        # User can opt InvalidRequestError into retries if they want.
        p = RetryPolicy(retry_on=(InvalidRequestError,))
        assert _should_retry(InvalidRequestError("ir"), p)
        assert not _should_retry(RateLimitError("rl"), p)


class TestComputeDelay:
    def test_rate_limit_retry_after_takes_precedence(self) -> None:
        p = RetryPolicy(base_delay=1.0, max_delay=30.0, jitter=0.0)
        err = RateLimitError("rl", retry_after=7.0)
        # Server told us 7 seconds; we use that regardless of attempt number.
        assert _compute_delay(err, attempt=0, policy=p) == 7.0
        assert _compute_delay(err, attempt=5, policy=p) == 7.0

    def test_rate_limit_retry_after_capped_at_max_delay(self) -> None:
        p = RetryPolicy(max_delay=10.0, jitter=0.0)
        err = RateLimitError("rl", retry_after=999.0)
        assert _compute_delay(err, attempt=0, policy=p) == 10.0

    def test_exponential_backoff_without_jitter(self) -> None:
        p = RetryPolicy(base_delay=1.0, exponential_factor=2.0, jitter=0.0)
        err = ProviderError("pe")
        assert _compute_delay(err, attempt=0, policy=p) == 1.0
        assert _compute_delay(err, attempt=1, policy=p) == 2.0
        assert _compute_delay(err, attempt=2, policy=p) == 4.0
        assert _compute_delay(err, attempt=3, policy=p) == 8.0

    def test_exponential_capped_at_max_delay(self) -> None:
        p = RetryPolicy(base_delay=1.0, exponential_factor=2.0, jitter=0.0, max_delay=5.0)
        err = ProviderError("pe")
        # 2**3 == 8, but capped at 5
        assert _compute_delay(err, attempt=3, policy=p) == 5.0

    def test_jitter_bounds(self) -> None:
        # With jitter=0.25, delay is in [0.75 * base, 1.25 * base]
        p = RetryPolicy(base_delay=4.0, exponential_factor=1.0, jitter=0.25)
        err = ProviderError("pe")
        for _ in range(50):
            d = _compute_delay(err, attempt=0, policy=p)
            assert 3.0 <= d <= 5.0


class TestRetrySync:
    def test_success_first_attempt(self) -> None:
        calls = [0]

        def call() -> str:
            calls[0] += 1
            return "ok"

        result = retry_sync(call, RetryPolicy())
        assert result == "ok"
        assert calls[0] == 1

    def test_retries_until_success(self, mocker: MockerFixture) -> None:
        mocker.patch("time.sleep")  # don't actually wait
        attempts = [0]

        def call() -> str:
            attempts[0] += 1
            if attempts[0] < 3:
                raise RateLimitError("transient")
            return "ok"

        result = retry_sync(call, RetryPolicy(max_retries=3))
        assert result == "ok"
        assert attempts[0] == 3

    def test_gives_up_after_max_retries(self, mocker: MockerFixture) -> None:
        mocker.patch("time.sleep")
        attempts = [0]

        def call() -> str:
            attempts[0] += 1
            raise RateLimitError("permanent")

        with pytest.raises(RateLimitError):
            retry_sync(call, RetryPolicy(max_retries=2))
        assert attempts[0] == 3  # initial + 2 retries

    def test_non_whitelisted_error_propagates_immediately(self, mocker: MockerFixture) -> None:
        sleep = mocker.patch("time.sleep")
        attempts = [0]

        def call() -> str:
            attempts[0] += 1
            raise InvalidRequestError("bad input")

        with pytest.raises(InvalidRequestError):
            retry_sync(call, RetryPolicy(max_retries=5))
        assert attempts[0] == 1
        sleep.assert_not_called()

    def test_max_retries_zero_runs_once(self, mocker: MockerFixture) -> None:
        sleep = mocker.patch("time.sleep")
        attempts = [0]

        def call() -> str:
            attempts[0] += 1
            raise RateLimitError("nope")

        with pytest.raises(RateLimitError):
            retry_sync(call, RetryPolicy(max_retries=0))
        assert attempts[0] == 1
        sleep.assert_not_called()

    def test_retry_after_is_used(self, mocker: MockerFixture) -> None:
        sleep = mocker.patch("time.sleep")
        attempts = [0]

        def call() -> str:
            attempts[0] += 1
            if attempts[0] < 2:
                raise RateLimitError("hold", retry_after=4.5)
            return "ok"

        retry_sync(call, RetryPolicy(max_retries=1, jitter=0.0))
        sleep.assert_called_once_with(4.5)


class TestRetryAsync:
    async def test_success_first_attempt(self) -> None:
        async def call() -> str:
            return "ok"

        assert await retry_async(call, RetryPolicy()) == "ok"

    async def test_retries_until_success(self, mocker: MockerFixture) -> None:
        async def fake_sleep(_: float) -> None:
            return None

        mocker.patch.object(asyncio, "sleep", fake_sleep)
        attempts = [0]

        async def call() -> str:
            attempts[0] += 1
            if attempts[0] < 2:
                raise ProviderError("transient")
            return "ok"

        result = await retry_async(call, RetryPolicy(max_retries=2))
        assert result == "ok"
        assert attempts[0] == 2

    async def test_gives_up_after_max_retries(self, mocker: MockerFixture) -> None:
        async def fake_sleep(_: float) -> None:
            return None

        mocker.patch.object(asyncio, "sleep", fake_sleep)

        async def call() -> str:
            raise ProviderError("permanent")

        with pytest.raises(ProviderError):
            await retry_async(call, RetryPolicy(max_retries=1))

    async def test_non_whitelisted_error_propagates_immediately(self) -> None:
        async def call() -> str:
            raise AuthenticationError("bad key")

        with pytest.raises(AuthenticationError):
            await retry_async(call, RetryPolicy())


class TestRetryStreamSync:
    def test_streams_chunks_when_no_error(self) -> None:
        def make_iter():
            return iter(["a", "b", "c"])

        result = list(retry_stream_sync(make_iter, RetryPolicy()))
        assert result == ["a", "b", "c"]

    def test_retries_when_first_chunk_fails(self, mocker: MockerFixture) -> None:
        mocker.patch("time.sleep")
        attempts = [0]

        def make_iter():
            attempts[0] += 1
            if attempts[0] < 2:
                # Raise BEFORE first yield
                def gen():
                    raise RateLimitError("transient")
                    yield  # unreachable

                return gen()
            return iter(["a", "b"])

        result = list(retry_stream_sync(make_iter, RetryPolicy(max_retries=1)))
        assert result == ["a", "b"]
        assert attempts[0] == 2

    def test_does_not_retry_mid_stream(self, mocker: MockerFixture) -> None:
        """Once the first chunk has yielded, errors propagate.

        Replaying a partial stream would surprise the consumer.
        """
        sleep = mocker.patch("time.sleep")
        attempts = [0]

        def make_iter():
            attempts[0] += 1

            def gen():
                yield "a"  # this succeeds, putting us past the retry window
                raise ProviderError("mid-stream failure")

            return gen()

        gen = retry_stream_sync(make_iter, RetryPolicy(max_retries=3))
        assert next(gen) == "a"
        with pytest.raises(ProviderError):
            next(gen)
        # Only one attempt — the retry window closed after first chunk.
        assert attempts[0] == 1
        sleep.assert_not_called()


class TestRetryStreamAsync:
    async def test_streams_chunks_when_no_error(self) -> None:
        async def gen():
            for x in ["a", "b"]:
                yield x

        def make_iter():
            return gen()

        chunks = [x async for x in retry_stream_async(make_iter, RetryPolicy())]
        assert chunks == ["a", "b"]

    async def test_retries_when_first_chunk_fails(self, mocker: MockerFixture) -> None:
        async def fake_sleep(_: float) -> None:
            return None

        mocker.patch.object(asyncio, "sleep", fake_sleep)
        attempts = [0]

        def make_iter():
            attempts[0] += 1
            if attempts[0] < 2:

                async def bad():
                    raise RateLimitError("transient")
                    yield  # unreachable

                return bad()

            async def good():
                yield "a"

            return good()

        chunks = [x async for x in retry_stream_async(make_iter, RetryPolicy(max_retries=1))]
        assert chunks == ["a"]
        assert attempts[0] == 2

    async def test_does_not_retry_mid_stream(self, mocker: MockerFixture) -> None:
        async def fake_sleep(_: float) -> None:
            return None

        mocker.patch.object(asyncio, "sleep", fake_sleep)
        attempts = [0]

        def make_iter():
            attempts[0] += 1

            async def gen():
                yield "a"
                raise ProviderError("mid-stream")

            return gen()

        gen = retry_stream_async(make_iter, RetryPolicy(max_retries=3))
        chunks = []
        with pytest.raises(ProviderError):
            async for x in gen:
                chunks.append(x)
        assert chunks == ["a"]
        assert attempts[0] == 1


class TestVoxClientIntegration:
    """End-to-end: per-call retry_policy=RetryPolicy(max_retries=0) disables retries."""

    def test_per_call_override(self, mocker: MockerFixture) -> None:
        from unittest.mock import MagicMock

        from vox import Message, ProviderConfig, RetryPolicy, VoxClient
        from vox.providers.openai import OpenAIProvider

        sleep = mocker.patch("time.sleep")

        client = VoxClient(openai_api_key="sk-test")
        provider = OpenAIProvider(ProviderConfig(api_key="sk-test", default_model="gpt-5"))
        client._providers["openai"] = provider

        mock_sdk = MagicMock()
        # Always raise RateLimitError
        mock_sdk.responses.create = MagicMock(side_effect=RateLimitError("rl", provider="openai"))
        provider._sync_client = mock_sdk

        # max_retries=0 ⇒ exactly one attempt, no sleep.
        with pytest.raises(RateLimitError):
            client.complete(
                [Message(role="user", content="hi")],
                model="gpt-5",
                retry_policy=RetryPolicy(max_retries=0),
            )
        sleep.assert_not_called()
