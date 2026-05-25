"""Error-type integration tests.

Only the *deterministic* error paths are live-tested:

* ``AuthenticationError`` — bad API key; rejected before generation.
* ``ModelNotFoundError`` / ``InvalidRequestError`` — nonexistent model
  string. Asserted as a union because some providers collapse "unknown
  model" to a generic 400; documenting that variance is part of the
  test's value.
* ``InvalidRequestError`` (or any ``VoxError``) — a malformed parameter
  the provider rejects.

The remaining error types — ``RateLimitError``, ``QuotaExceededError``,
``ProviderError``, ``ContentFilterError`` — are NOT live-tested. They
can't be triggered deterministically without hammering, an exhausted
billing account, a real outage, or a moving safety system. They stay
covered by the mocked unit tests in ``tests/providers/``.
"""

from __future__ import annotations

import pytest

from vox import (
    AuthenticationError,
    InvalidRequestError,
    Message,
    ModelNotFoundError,
    VoxClient,
    VoxError,
)

from .conftest import ProviderProfile


def _client_with_bad_key(profile: ProviderProfile) -> VoxClient:
    """Build a client with a bogus key for *only* ``profile``'s provider."""
    kwargs: dict[str, str] = {
        "openai_api_key": "sk-vox-integration-bogus",
        "anthropic_api_key": "sk-ant-vox-integration-bogus",
        "gemini_api_key": "vox-integration-bogus",
        "openrouter_api_key": "sk-or-vox-integration-bogus",
    }
    return VoxClient(**kwargs)  # type: ignore[arg-type]


def test_bad_api_key_raises_authentication_error(profile: ProviderProfile) -> None:
    """A deliberately bogus API key surfaces as ``AuthenticationError``.

    Deterministic and free-ish — providers reject before generation.
    """
    client = _client_with_bad_key(profile)
    with pytest.raises(AuthenticationError):
        client.complete(
            [Message(role="user", content="hi")],
            model=profile.model,
            max_tokens=16,
        )


def test_nonexistent_model_raises(profile: ProviderProfile, client: VoxClient) -> None:
    """Requesting a model that doesn't exist surfaces as a vox error.

    Accepts ``ModelNotFoundError`` *or* ``InvalidRequestError`` because
    providers vary: some return a dedicated 404 with "model" in the
    message (→ ``ModelNotFoundError``); others collapse to a generic
    400 (→ ``InvalidRequestError``). Either way vox must wrap it — a
    raw SDK exception leaking through is the failure mode this test
    guards against.
    """
    with pytest.raises((ModelNotFoundError, InvalidRequestError)):
        client.complete(
            [Message(role="user", content="hi")],
            model=profile.bad_model_id,
            provider=profile.name,
            max_tokens=16,
        )


def test_malformed_request_raises_vox_error(profile: ProviderProfile, client: VoxClient) -> None:
    """A clearly-invalid request surfaces as some ``VoxError`` subclass.

    Uses a negative ``max_tokens`` — universally rejected. The specific
    subclass varies by provider (some validate client-side in the SDK,
    others 400 from the API), so the assertion is the base class. The
    invariant being tested is "vox doesn't leak raw SDK exceptions."
    """
    with pytest.raises(VoxError):
        client.complete(
            [Message(role="user", content="hi")],
            model=profile.model,
            max_tokens=-5,
        )
