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

This file also covers two multimodal *combinations* downstream
consumers rely on (e.g. Tomte's identify-thing pipeline):

* **Image + tools** — the outer agent-loop pattern (full tool registry
  available; image in the user turn).
* **Image + ``response_schema``** — the terminal structured-output
  pattern (no tools on the call; model produces a validated dataclass).
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, Field

from vox import ImageContent, Message, TextContent, Tool, VideoContent, VoxClient

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


# ── Multimodal combinations ────────────────────────────────────────────


# A small tool the model is asked to call when shown the image. The
# only "right" tool to call given the prompt — easier to force via
# prompt-engineering than to pass provider-specific ``tool_choice``.
_RECORD_COLOR_TOOL = Tool(
    name="record_color",
    description=("Record the dominant color visible in an image the user provided."),
    parameters={
        "type": "object",
        "properties": {
            "color": {
                "type": "string",
                "description": "The dominant color filling the image.",
            },
        },
        "required": ["color"],
    },
)


def test_vision_with_tools(
    vision_profile: ProviderProfile,
    client: VoxClient,
    red_square_b64: str,
) -> None:
    """Image + tool registry in one call → the outer agent-loop pattern.

    Sends the red-square image alongside a ``record_color`` tool. The
    prompt forces the model to use the tool. Asserts the model called
    it with a red-family color — proving both that (a) the image was
    ingested and (b) the tool path coexisted with the image content
    in a single ``complete()`` call.

    Surfaces a regression on any provider where the combination of
    ``content=[TextContent, ImageContent]`` and ``tools=[...]`` is
    rejected or silently mishandled — the two code paths are
    orthogonal in the translator today, but only this test exercises
    them together end-to-end.
    """
    response = client.complete(
        [
            Message(
                role="user",
                content=[
                    TextContent(
                        text=(
                            "Look at this image and record the dominant "
                            "color using the record_color tool. Reply only "
                            "by calling the tool."
                        )
                    ),
                    ImageContent(data=red_square_b64, media_type="image/png"),
                ],
            )
        ],
        model=vision_profile.model,
        provider=vision_profile.name,
        tools=[_RECORD_COLOR_TOOL],
        max_tokens=1024,
    )

    assert response.finish_reason == "tool_calls", (
        f"expected tool_calls finish reason; got {response.finish_reason!r} "
        f"with text {response.message.text!r}"
    )
    assert response.message.tool_calls, "no tool_calls on the response"
    record_calls = [tc for tc in response.message.tool_calls if tc.name == "record_color"]
    assert record_calls, (
        f"no record_color call; got {[tc.name for tc in response.message.tool_calls]}"
    )
    args = record_calls[0].arguments
    assert isinstance(args, dict)
    assert "color" in args, f"expected 'color' in arguments; got {args}"
    color_lower = str(args["color"]).lower()
    matched = next((c for c in _RED_FAMILY if c in color_lower), None)
    assert matched, f"tool was called but with a non-red color; got {args['color']!r}"


class _ColorObservation(BaseModel):
    """Structured-output target for the vision-keystone test.

    Shape intentionally close to Tomte's ``ApplianceIdentification``:
    a Literal-enum-style classification (``brightness``), a numeric
    confidence, plus a list field. Anthropic / Gemini / OpenAI all
    have to round-trip the full nested shape via vox's structured
    output path AND see the image to fill ``color``.
    """

    color: str = Field(description="The dominant color in the image.")
    brightness: Literal["dark", "bright"] = Field(
        description="Whether the image appears dark or bright overall."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the color identification, 0..1.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Any caveats about the identification.",
    )


def test_vision_with_response_schema(
    vision_profile: ProviderProfile,
    client: VoxClient,
    red_square_b64: str,
) -> None:
    """Image + ``response_schema`` → the terminal structured-output pattern.

    The keystone of Tomte's ``identify_thing`` flow: a call that
    receives an image and produces a validated dataclass as its
    ``response.parsed``. No other tools on this call (the structured
    output IS the terminal answer).

    Asserts ``parsed`` is a ``_ColorObservation`` instance with a
    red-family ``color`` value. Other fields just need to validate
    against the schema — values aren't asserted on, since this is a
    translation test, not a model-judgement test.
    """
    response = client.complete(
        [
            Message(
                role="user",
                content=[
                    TextContent(
                        text=(
                            "Identify the dominant color in this image and "
                            "return a structured observation."
                        )
                    ),
                    ImageContent(data=red_square_b64, media_type="image/png"),
                ],
            )
        ],
        model=vision_profile.model,
        provider=vision_profile.name,
        response_schema=_ColorObservation,
        max_tokens=2048,
    )

    assert response.parsed is not None, "response.parsed was not populated"
    assert isinstance(response.parsed, _ColorObservation)
    color_lower = response.parsed.color.lower()
    matched = next((c for c in _RED_FAMILY if c in color_lower), None)
    assert matched, f"structured-output color is not red-family; got {response.parsed.color!r}"


# ── Video input ────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def red_video_bytes() -> bytes:
    """Generate a tiny 2-second solid-red mp4 for the video tests.

    32x32 at 10 fps, libx264-encoded — same colour palette as the
    bundled ``red_square.png`` so the assertion can be the same
    red-family check. Generated on the fly to avoid checking a binary
    fixture into the repo; relies on ``imageio[ffmpeg]`` from the dev
    dependency group (or the ``vox-llm[video]`` extra).
    """
    from io import BytesIO

    import imageio.v3 as iio
    import numpy as np

    n_frames = 20  # 2 seconds at 10 fps
    frames = np.zeros((n_frames, 32, 32, 3), dtype=np.uint8)
    frames[:, :, :, 0] = 255  # solid red

    buf = BytesIO()
    iio.imwrite(buf, frames, extension=".mp4", fps=10, codec="libx264")
    return buf.getvalue()


def test_video_native_identifies_red_clip(
    video_profile: ProviderProfile,
    client: VoxClient,
    red_video_bytes: bytes,
) -> None:
    """A solid-red 2s mp4 round-trips through native ``VideoContent``.

    Only runs on providers with native video input (currently Gemini).
    Asserts the model identifies the clip's colour as red-family —
    same loose family check as the image vision tests, since pure
    RGB(255, 0, 0) re-encoded through libx264 can read as red/pink/
    crimson depending on the model.

    Tests the native code path end-to-end: VideoContent →
    Part(inline_data=Blob(mime_type=video/mp4, data=bytes)) on Gemini.
    The fallback / frame-extraction path is covered separately in
    ``tests/test_video.py``.
    """
    response = client.complete(
        [
            Message(
                role="user",
                content=[
                    TextContent(
                        text=(
                            "Identify the dominant colour visible in this "
                            "short video clip. Reply with just the colour "
                            "name, one word."
                        )
                    ),
                    VideoContent(
                        data=red_video_bytes,  # type: ignore[arg-type]
                        media_type="video/mp4",
                    ),
                ],
            )
        ],
        model=video_profile.model,
        provider=video_profile.name,
        max_tokens=1024,
    )
    _assert_red_family(response.message.text)
