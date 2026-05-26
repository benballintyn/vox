"""Streaming completion tests — chunk-type sequence + concatenation."""

from __future__ import annotations

import pytest

from vox import Message, VoxClient

from .conftest import ProviderProfile

# OpenRouter doesn't surface a usage chunk on the proxied Chat
# Completions stream, even with ``stream_options.include_usage=True``
# set on the request. Tracked as vox#27 (needs an OpenRouter-specific
# usage-extraction path, probably reading the response headers). For
# all other providers the rest of the streaming contract holds.
_NO_USAGE_CHUNK_PROVIDERS = {"openrouter"}


def test_stream_chunk_sequence(profile: ProviderProfile, client: VoxClient) -> None:
    """``stream`` yields a well-formed chunk sequence.

    Structural contract for every provider:

    * at least one ``text`` chunk (the response had visible content),
    * exactly one terminal ``done`` chunk carrying a ``finish_reason``,
    * concatenated text deltas are non-empty,
    * a ``usage`` chunk arrives before ``done`` — except on providers
      with a documented usage gap (currently OpenRouter, vox#27).

    This is the only place where the stream-event translation in each
    provider is exercised end-to-end.
    """
    chunks = list(
        client.stream(
            [Message(role="user", content="Say hi.")],
            model=profile.model,
            provider=profile.name,
            max_tokens=1024,
        )
    )

    types = [c.type for c in chunks]
    assert "text" in types, f"no text chunks; got {types}"
    assert types.count("done") == 1, f"expected exactly one done chunk; got {types}"
    assert types[-1] == "done", f"done chunk must be last; got {types}"

    text = "".join(c.text for c in chunks if c.type == "text")
    assert text, "concatenated stream text was empty"

    done = next(c for c in chunks if c.type == "done")
    assert done.finish_reason, "done chunk missing finish_reason"

    if profile.name in _NO_USAGE_CHUNK_PROVIDERS:
        pytest.xfail(f"{profile.name} doesn't surface a usage chunk on streaming — vox#27")
    assert "usage" in types, f"no usage chunk; got {types}"
    usage = next(c for c in chunks if c.type == "usage")
    assert usage.usage is not None
    assert usage.usage.prompt_tokens > 0


async def test_astream_chunk_sequence(profile: ProviderProfile, client: VoxClient) -> None:
    """Async ``astream`` produces the same chunk sequence as sync ``stream``."""
    chunks = []
    async for chunk in client.astream(
        [Message(role="user", content="Say hi.")],
        model=profile.model,
        provider=profile.name,
        max_tokens=1024,
    ):
        chunks.append(chunk)

    types = [c.type for c in chunks]
    assert "text" in types
    assert types.count("done") == 1
    assert types[-1] == "done"
    assert "".join(c.text for c in chunks if c.type == "text")

    if profile.name in _NO_USAGE_CHUNK_PROVIDERS:
        pytest.xfail(f"{profile.name} doesn't surface a usage chunk on streaming — vox#27")
    assert "usage" in types
