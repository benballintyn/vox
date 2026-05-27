"""Shared fixtures for the live integration suite.

Provider profiles + key-aware fixtures: a test parametrized over
``profile`` runs once per provider whose API key is present and skips
the rest. Centralizing model IDs here means model deprecations are a
one-line change.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from vox import VoxClient


@dataclass(frozen=True)
class ProviderProfile:
    """A live-testable provider configuration.

    Attributes:
        name: Provider name as resolved by ``vox._registry``.
        model: Model ID for ordinary completions / streaming / tools / vision.
        env_var: Environment variable that must hold the API key.
        supports_vision: Whether ``model`` accepts image inputs.
        supports_native_video: Whether ``model`` accepts ``video/*``
            content natively (currently: Gemini only). vox falls back
            to client-side frame extraction for the rest, but native
            video is a different code path worth testing end-to-end.
        supports_reasoning: Whether ``model`` supports ``ReasoningConfig``.
        exposes_thinking_blocks: Whether ``response.thinking`` is populated
            by default with reasoning enabled. (Anthropic: yes. OpenAI:
            only with ``summary="auto"`` etc. Gemini: thinking is
            internalized; no blocks exposed via the public API.)
        bad_model_id: A model ID guaranteed not to exist on this provider,
            used by the ``ModelNotFoundError`` test.
        requires_disable_reasoning_for_length: Models that reason
            unconditionally (e.g. gpt-5 family) need reasoning explicitly
            disabled for the ``finish_reason=="length"`` test to behave —
            otherwise reasoning tokens eat the entire output budget and
            the call may return no visible text.
    """

    name: str
    model: str
    env_var: str
    supports_vision: bool = True
    supports_native_video: bool = False
    supports_reasoning: bool = True
    exposes_thinking_blocks: bool = False
    bad_model_id: str = ""
    requires_disable_reasoning_for_length: bool = False
    # Audio: per-provider STT/TTS model + voice. ``None``/empty marks
    # the provider as not supporting that side; the audio_profile
    # fixture filters accordingly.
    stt_model: str | None = None
    tts_model: str | None = None
    tts_voice: str = ""


# Profiles in stable order. Add a new provider by appending an entry.
PROFILES: list[ProviderProfile] = [
    ProviderProfile(
        name="openai",
        model="gpt-5-mini",
        env_var="OPENAI_API_KEY",
        exposes_thinking_blocks=False,
        bad_model_id="gpt-bogus-9999-does-not-exist",
        requires_disable_reasoning_for_length=True,
        stt_model="whisper-1",
        tts_model="tts-1",
        tts_voice="alloy",
    ),
    ProviderProfile(
        name="anthropic",
        model="claude-haiku-4-5",
        env_var="ANTHROPIC_API_KEY",
        exposes_thinking_blocks=True,
        bad_model_id="claude-bogus-9999-does-not-exist",
    ),
    ProviderProfile(
        name="gemini",
        model="gemini-3.1-flash-lite",
        env_var="GEMINI_API_KEY",
        exposes_thinking_blocks=False,
        bad_model_id="gemini-bogus-9999-does-not-exist",
        supports_native_video=True,
        stt_model="gemini-3.5-flash",
        tts_model="gemini-3.1-flash-tts-preview",
        tts_voice="Kore",
    ),
    ProviderProfile(
        name="openrouter",
        model="openai/gpt-5-mini",
        env_var="OPENROUTER_API_KEY",
        exposes_thinking_blocks=False,
        bad_model_id="bogus-vendor/bogus-model-9999",
        requires_disable_reasoning_for_length=True,
    ),
]


# ── pytest collection: mark the whole package `integration` ────────────


_INTEGRATION_DIR = Path(__file__).parent


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-mark items in this package as ``integration`` + ``flaky``.

    pytest's ``pytest_collection_modifyitems`` hook receives the full
    session item list regardless of which conftest defines it — so we
    explicitly filter to items located under this directory, otherwise
    we'd mark every unit test in the repo as integration too.

    Avoids needing ``@pytest.mark.integration`` on every test function.
    ``flaky`` (pytest-rerunfailures) reruns transient network / 5xx /
    rate-limit hiccups so a single blip doesn't fail the suite — an
    actual assertion bug fails twice and surfaces normally, just slower.
    """
    integration_mark = pytest.mark.integration
    flaky_mark = pytest.mark.flaky(reruns=2, reruns_delay=3)
    for item in items:
        try:
            item.path.relative_to(_INTEGRATION_DIR)
        except ValueError:
            continue  # not ours — leave it alone
        item.add_marker(integration_mark)
        item.add_marker(flaky_mark)


# ── Provider-parametrized fixtures ─────────────────────────────────────


def _require_key(profile: ProviderProfile) -> ProviderProfile:
    """Skip the current test if ``profile``'s API key env var is absent."""
    if not os.environ.get(profile.env_var):
        pytest.skip(f"{profile.env_var} not set; skipping {profile.name}")
    return profile


@pytest.fixture(params=PROFILES, ids=lambda p: p.name)
def profile(request: pytest.FixtureRequest) -> ProviderProfile:
    """Parametrize a test across every provider with a key present."""
    return _require_key(request.param)


@pytest.fixture(
    params=[p for p in PROFILES if p.supports_vision],
    ids=lambda p: p.name,
)
def vision_profile(request: pytest.FixtureRequest) -> ProviderProfile:
    """Like ``profile``, but limited to vision-capable providers."""
    return _require_key(request.param)


@pytest.fixture(
    params=[p for p in PROFILES if p.supports_reasoning],
    ids=lambda p: p.name,
)
def reasoning_profile(request: pytest.FixtureRequest) -> ProviderProfile:
    """Like ``profile``, but limited to reasoning-capable providers."""
    return _require_key(request.param)


@pytest.fixture(
    params=[p for p in PROFILES if p.supports_native_video],
    ids=lambda p: p.name,
)
def video_profile(request: pytest.FixtureRequest) -> ProviderProfile:
    """Like ``profile``, but limited to providers with native video input."""
    return _require_key(request.param)


@pytest.fixture(
    params=[p for p in PROFILES if p.stt_model],
    ids=lambda p: p.name,
)
def stt_profile(request: pytest.FixtureRequest) -> ProviderProfile:
    """Profiles that support speech-to-text natively."""
    return _require_key(request.param)


@pytest.fixture(
    params=[p for p in PROFILES if p.tts_model],
    ids=lambda p: p.name,
)
def tts_profile(request: pytest.FixtureRequest) -> ProviderProfile:
    """Profiles that support text-to-speech natively."""
    return _require_key(request.param)


# ── Client + image fixtures ────────────────────────────────────────────


@pytest.fixture
def client() -> VoxClient:
    """A ``VoxClient`` seeded from the environment.

    Absent keys are passed as ``None``; the per-test profile fixtures
    skip cases that need a missing key, so the client only ever gets
    used for providers it can serve.
    """
    return VoxClient(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
    )


@pytest.fixture(scope="session")
def red_square_b64() -> str:
    """Base64-encoded contents of the bundled 32x32 red-square PNG.

    Used by multimodal tests with a constrained ``ImageContent`` payload.
    """
    path = Path(__file__).parent / "fixtures" / "red_square.png"
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")
