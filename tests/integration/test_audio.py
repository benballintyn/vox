"""Audio I/O integration tests — exercises ``transcribe()`` + ``synthesize()``.

Two paths covered per provider with native audio support:

* **synthesize**: ask the provider's TTS to speak a known phrase;
  assert the returned :class:`AudioContent` has a plausible audio
  payload (non-trivial size, correct MIME).
* **transcribe**: feed a session-cached TTS-generated WAV of a known
  phrase back through each provider's STT; assert the transcript
  contains the expected words.

A session-scoped fixture generates the test audio once via OpenAI TTS
(requires ``OPENAI_API_KEY``). That keeps the cost to a single TTS
call per session — every transcribe test reuses the bytes. If the
OpenAI key is absent, the transcribe path is skipped (the
session-fixture itself skips), but the per-provider synthesize tests
still run for whichever providers' keys are present.

Cost: ~$0.0002 per session for the TTS fixture, plus negligible
transcribe + synthesize charges per provider. Manual dispatch only.
"""

from __future__ import annotations

import base64
import os

import pytest

from vox import AudioContent, VoxClient

from .conftest import ProviderProfile

# Short, neutral phrase with consonant variety so transcription has a
# fair shot — kept lower-case for case-insensitive assertion below.
KNOWN_PHRASE = "Hello world. The quick brown fox jumps over the lazy dog."

# At least one of these words must appear (case-insensitive) in the
# transcript for the test to pass. Tolerant on purpose: STT models
# vary on punctuation, capitalization, and "the" preservation, but a
# working pipeline always lands at least one of these content words.
_EXPECTED_TRANSCRIBE_WORDS = {"hello", "world", "fox", "quick", "brown", "lazy", "dog"}


def _assert_transcript_matches(text: str) -> None:
    lower = text.lower()
    matched = [w for w in _EXPECTED_TRANSCRIBE_WORDS if w in lower]
    assert matched, f"transcript missed every expected word; got: {text!r}"


@pytest.fixture(scope="session")
def known_phrase_wav(client: VoxClient) -> AudioContent:
    """A WAV of OpenAI TTS speaking ``KNOWN_PHRASE``. Session-cached.

    Skipped if ``OPENAI_API_KEY`` is missing — every transcribe test
    that depends on this fixture inherits the skip.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY required to generate the transcribe fixture audio")
    return client.synthesize(
        text=KNOWN_PHRASE,
        voice="alloy",
        model="tts-1",
        format="wav",
    )


# ── synthesize ─────────────────────────────────────────────────────────


def test_synthesize_returns_plausible_audio(
    tts_profile: ProviderProfile,
    client: VoxClient,
) -> None:
    """Provider TTS returns ``AudioContent`` with non-trivial audio bytes."""
    audio = client.synthesize(
        text="Hello world.",
        voice=tts_profile.tts_voice,
        model=tts_profile.tts_model or "",
        provider=tts_profile.name,
    )
    assert isinstance(audio, AudioContent)
    raw = base64.standard_b64decode(audio.data)
    # A "hello world" TTS clip is at least a few KB even at low
    # quality — a 100-byte response would mean we got an empty or
    # error-degraded payload.
    assert len(raw) > 1024, f"audio payload suspiciously small: {len(raw)} bytes"
    # MIME should be one of the known audio types.
    assert audio.media_type.startswith("audio/")


# ── transcribe ─────────────────────────────────────────────────────────


def test_transcribe_recovers_known_phrase(
    stt_profile: ProviderProfile,
    client: VoxClient,
    known_phrase_wav: AudioContent,
) -> None:
    """Provider STT recovers at least one expected word from a known clip."""
    result = client.transcribe(
        known_phrase_wav,
        model=stt_profile.stt_model or "",
        provider=stt_profile.name,
    )
    assert result.provider == stt_profile.name
    _assert_transcript_matches(result.text)


async def test_atranscribe_parity(
    stt_profile: ProviderProfile,
    client: VoxClient,
    known_phrase_wav: AudioContent,
) -> None:
    """Async transcribe path matches sync."""
    result = await client.atranscribe(
        known_phrase_wav,
        model=stt_profile.stt_model or "",
        provider=stt_profile.name,
    )
    _assert_transcript_matches(result.text)


async def test_asynthesize_parity(
    tts_profile: ProviderProfile,
    client: VoxClient,
) -> None:
    """Async synthesize path matches sync."""
    audio = await client.asynthesize(
        text="Hello world.",
        voice=tts_profile.tts_voice,
        model=tts_profile.tts_model or "",
        provider=tts_profile.name,
    )
    raw = base64.standard_b64decode(audio.data)
    assert len(raw) > 1024
