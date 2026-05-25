"""Multimodal (vision) integration tests.

A bundled 32x32 solid-red PNG is sent inline as base64 ``ImageContent``.
A constrained one-word prompt makes the response assertable — if any
vision-capable model fails to identify a solid red square as red, the
plumbing is broken, not the model.
"""

from __future__ import annotations

from vox import ImageContent, Message, TextContent, VoxClient

from .conftest import ProviderProfile


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
                    TextContent(text="What color is this image? Reply with a single word."),
                    ImageContent(data=red_square_b64, media_type="image/png"),
                ],
            )
        ],
        model=vision_profile.model,
        max_tokens=1024,
    )
    assert "red" in response.message.text.lower(), (
        f"expected 'red' in response; got: {response.message.text!r}"
    )


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
                    TextContent(text="What color is this image? Reply with a single word."),
                    ImageContent(data=red_square_b64, media_type="image/png"),
                ],
            )
        ],
        model=vision_profile.model,
        max_tokens=1024,
    )
    assert "red" in response.message.text.lower()
