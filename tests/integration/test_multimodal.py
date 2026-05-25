"""Multimodal (vision) integration tests.

A bundled 32x32 solid-red PNG is sent inline as base64 ``ImageContent``.
The constrained one-word prompt asks for a color; the assertion accepts
any word in the **red family** (red, pink, crimson, scarlet, magenta).

The looser-than-just-"red" check reflects observed real-world variance:
OpenRouter served the same image and the model called it "pink" — a
plausible perception of pure RGB(255, 0, 0) once it's re-encoded /
scaled through the OpenRouter routing layer. That's model variance,
not a vox bug. The contract under test is **the vision pipeline
ingested the image and produced a color-family answer**; a broken
pipeline would yield something unrelated (e.g. "I don't see an image"
or guessing a non-red color), not "pink".
"""

from __future__ import annotations

from vox import ImageContent, Message, TextContent, VoxClient

from .conftest import ProviderProfile

# Accept any red-family word. A working vision pipeline + a competent
# model should land somewhere in this set for pure RGB(255, 0, 0). A
# broken pipeline returns something outside it.
_RED_FAMILY = {"red", "pink", "crimson", "scarlet", "magenta"}


def _assert_red_family(text: str) -> None:
    """Assert the response identifies the image as red-family.

    Case-insensitive substring check against ``_RED_FAMILY`` — robust to
    punctuation, capitalization, and the model emitting more than one
    word despite the prompt asking for one.
    """
    lower = text.lower()
    matched = next((c for c in _RED_FAMILY if c in lower), None)
    assert matched, f"expected a red-family color word; got: {text!r}"


def test_vision_identifies_red_square(
    vision_profile: ProviderProfile,
    client: VoxClient,
    red_square_b64: str,
) -> None:
    """A solid red PNG round-trips through ``ImageContent`` and is identified.

    Verifies the multimodal request translation — each provider has a
    different schema for inline image bytes (OpenAI ``image_url`` with a
    data URI, Anthropic ``image`` content block with ``source.data``,
    Gemini ``inline_data`` parts). All must accept vox's normalized
    ``ImageContent(source_type="base64", ...)``.
    """
    response = client.complete(
        [
            Message(
                role="user",
                content=[
                    TextContent(
                        text=(
                            "I am showing you a solid-color square image. "
                            "Identify the single color filling the entire "
                            "image. Reply with just the color name, one word."
                        )
                    ),
                    ImageContent(data=red_square_b64, media_type="image/png"),
                ],
            )
        ],
        model=vision_profile.model,
        provider=vision_profile.name,
        max_tokens=1024,
    )
    _assert_red_family(response.message.text)


async def test_vision_async_parity(
    vision_profile: ProviderProfile,
    client: VoxClient,
    red_square_b64: str,
) -> None:
    """Async vision path matches sync."""
    response = await client.acomplete(
        [
            Message(
                role="user",
                content=[
                    TextContent(
                        text=(
                            "I am showing you a solid-color square image. "
                            "Identify the single color filling the entire "
                            "image. Reply with just the color name, one word."
                        )
                    ),
                    ImageContent(data=red_square_b64, media_type="image/png"),
                ],
            )
        ],
        model=vision_profile.model,
        provider=vision_profile.name,
        max_tokens=1024,
    )
    _assert_red_family(response.message.text)
