"""Configurable retry layer for vox provider calls.

vox's per-provider SDKs (openai, anthropic, google-genai) each ship
their own retry logic with their own policies and defaults — that's
useful but uneven across providers and doesn't honour the
:attr:`vox.RateLimitError.retry_after` that vox already extracts from
provider responses.

This module adds a thin vox-level retry layer on top:

* Uniform policy across providers (one :class:`RetryPolicy` instead of
  five SDK-specific knobs).
* Honours ``RateLimitError.retry_after`` — when the provider tells us
  *exactly* how long to wait, we listen.
* Selective: retries only the error classes the consumer whitelists
  (defaults to transient ones: ``RateLimitError`` and ``ProviderError``).
  Never retries ``InvalidRequestError`` / ``AuthenticationError`` /
  ``ContentFilterError`` / ``ModelNotFoundError``.
* Streaming-aware: retries only **before the first chunk is yielded**.
  Once the consumer has started receiving data, errors propagate
  rather than replaying mid-stream.

Configured at the client level via ``VoxClient(retry_policy=...)`` and
overridable per call via ``retry_policy=`` on each public method.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import TypeVar

from pydantic import BaseModel, Field

from .errors import ProviderError, RateLimitError, VoxError

T = TypeVar("T")

_DEFAULT_RETRY_ON: tuple[type[VoxError], ...] = (RateLimitError, ProviderError)


class RetryPolicy(BaseModel):
    """Retry behaviour for vox provider calls.

    Args:
        max_retries: Maximum number of retry attempts AFTER the initial
            request. ``0`` disables retries entirely. Default 3, so up
            to 4 attempts total.
        base_delay: Base delay in seconds for exponential backoff.
            Default 1.0 (first retry waits ~1s, second ~2s, third ~4s).
        max_delay: Hard cap on any single sleep. Prevents runaway
            backoff when the model raises huge ``retry_after`` values.
            Default 30.0 seconds.
        exponential_factor: Multiplier between successive retries.
            Default 2.0.
        jitter: Symmetric randomization fraction (``±jitter * delay``).
            Default 0.25 (±25%) — recommended to prevent thundering-herd
            retry storms across many clients hitting the same provider.
        retry_on: Tuple of VoxError subclasses that should trigger a
            retry. Anything not in this list propagates immediately.
            Default ``(RateLimitError, ProviderError)``.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_factor: float = 2.0
    jitter: float = 0.25
    retry_on: tuple[type[VoxError], ...] = Field(default=_DEFAULT_RETRY_ON)

    model_config = {"arbitrary_types_allowed": True}


def _should_retry(error: BaseException, policy: RetryPolicy) -> bool:
    """Return True if this error class is in the whitelist."""
    if not isinstance(error, VoxError):
        return False
    return isinstance(error, policy.retry_on)


def _compute_delay(error: BaseException, attempt: int, policy: RetryPolicy) -> float:
    """Pick the sleep duration before the next attempt.

    If the error is a :class:`RateLimitError` carrying a server-supplied
    ``retry_after``, use that (capped by ``max_delay``). Otherwise fall
    back to exponential backoff with jitter.

    Args:
        error: The exception caught from the most recent attempt.
        attempt: Zero-indexed attempt number that just failed (so
            ``0`` for the first failure, ``1`` for the second, etc.).
        policy: The active retry policy.
    """
    if isinstance(error, RateLimitError) and error.retry_after is not None:
        return min(float(error.retry_after), policy.max_delay)

    base = policy.base_delay * (policy.exponential_factor**attempt)
    delay = min(base, policy.max_delay)
    if policy.jitter > 0:
        # Symmetric ±jitter*delay
        delay += delay * policy.jitter * (2 * random.random() - 1)
    return max(0.0, delay)


# ── Sync ───────────────────────────────────────────────────────────────


def retry_sync(call: Callable[[], T], policy: RetryPolicy) -> T:
    """Execute ``call()`` with retry per ``policy``.

    The callable is invoked fresh on each attempt — the retry layer is
    transparent to the underlying provider (no state to reset).
    """
    last_error: BaseException | None = None
    for attempt in range(policy.max_retries + 1):
        try:
            return call()
        except BaseException as e:
            last_error = e
            if not _should_retry(e, policy):
                raise
            if attempt >= policy.max_retries:
                raise
            time.sleep(_compute_delay(e, attempt, policy))
    # Unreachable in practice — the loop either returns or raises.
    assert last_error is not None
    raise last_error


def retry_stream_sync(make_iter: Callable[[], Iterator[T]], policy: RetryPolicy) -> Iterator[T]:
    """Stream wrapper: retry only before the first chunk is yielded.

    Once the underlying iterator has produced at least one item, any
    subsequent error propagates — replaying a partial stream would
    surprise the consumer.
    """
    for attempt in range(policy.max_retries + 1):
        try:
            it = make_iter()
            first = next(it)
        except StopIteration:
            return  # empty stream; nothing to yield
        except BaseException as e:
            if not _should_retry(e, policy) or attempt >= policy.max_retries:
                raise
            time.sleep(_compute_delay(e, attempt, policy))
            continue

        # First chunk succeeded — past this point, errors propagate.
        yield first
        yield from it
        return


# ── Async ──────────────────────────────────────────────────────────────


async def retry_async(call: Callable[[], Awaitable[T]], policy: RetryPolicy) -> T:
    """Awaitable counterpart of :func:`retry_sync`."""
    last_error: BaseException | None = None
    for attempt in range(policy.max_retries + 1):
        try:
            return await call()
        except BaseException as e:
            last_error = e
            if not _should_retry(e, policy):
                raise
            if attempt >= policy.max_retries:
                raise
            await asyncio.sleep(_compute_delay(e, attempt, policy))
    assert last_error is not None
    raise last_error


async def retry_stream_async(
    make_iter: Callable[[], AsyncIterator[T]], policy: RetryPolicy
) -> AsyncIterator[T]:
    """Async-stream wrapper: retry only before the first chunk is yielded."""
    for attempt in range(policy.max_retries + 1):
        try:
            it = make_iter()
            first = await it.__anext__()
        except StopAsyncIteration:
            return
        except BaseException as e:
            if not _should_retry(e, policy) or attempt >= policy.max_retries:
                raise
            await asyncio.sleep(_compute_delay(e, attempt, policy))
            continue

        yield first
        async for chunk in it:
            yield chunk
        return
