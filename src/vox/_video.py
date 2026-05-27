"""Client-side video decoding for providers without native video input.

Used by the OpenAI, Anthropic, and Chat Completions providers to fall
back to image frames when a :class:`vox.VideoContent` is sent. Gemini
consumes video natively and does not use this module.

The :func:`extract_frames` helper is intentionally minimal — uniform
temporal sampling, JPEG output, hard caps. Consumers that need more
control (different sampling, key-frames only, specific resolutions)
should bypass this fallback by passing :class:`vox.ImageContent` parts
directly.

This module imports ``imageio`` lazily. Callers must guard with
:func:`require_imageio` before invoking :func:`extract_frames`.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from .errors import InvalidRequestError
from .models.messages import ImageContent, VideoContent

_IMAGEIO_INSTALL_HINT = (
    "Video fallback requires the optional 'video' extra. "
    "Install with: pip install 'vox-llm[video]'"
)

# imageio reading from a BytesIO needs an extension hint to pick a
# backend (it can't sniff the format from the buffer alone). Map vox's
# video MIME types to file extensions that ffmpeg/imageio recognise.
_MIME_TO_EXTENSION: dict[str, str] = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "video/x-matroska": ".mkv",
    "video/ogg": ".ogv",
    "video/mpeg": ".mpeg",
}


def _extension_for(media_type: str) -> str:
    """Return the file extension imageio should use for a video MIME type."""
    return _MIME_TO_EXTENSION.get(media_type.lower(), ".mp4")


def require_imageio() -> Any:
    """Import and return the imageio.v3 module, raising a clean vox error if missing.

    Returns:
        The ``imageio.v3`` module.

    Raises:
        InvalidRequestError: If ``imageio`` (or its ffmpeg backend) is
            not installed.
    """
    try:
        import imageio.v3 as iio
    except ImportError as e:  # pragma: no cover — exercised in tests via mocker
        raise InvalidRequestError(_IMAGEIO_INSTALL_HINT) from e
    return iio


def extract_frames(
    video: VideoContent,
    *,
    fps: float = 1.0,
    max_frames: int = 8,
    jpeg_quality: int = 85,
) -> list[ImageContent]:
    """Decode a video and return uniformly-sampled JPEG frames.

    Sampling strategy: pick up to ``ceil(duration * fps)`` frames,
    capped at ``max_frames``, evenly spaced across the video's
    duration. Each frame is JPEG-encoded and wrapped as an
    :class:`ImageContent`.

    Args:
        video: The video to decode. Must have ``source_type="base64"``
            — URL-sourced videos are refused (see Raises) because the
            fallback path has no opinion about how to fetch them.
        fps: Target frames per second of source video to sample at.
            Defaults to 1.0 (one frame per second).
        max_frames: Hard cap on the number of frames returned.
            Defaults to 8.
        jpeg_quality: JPEG encode quality (1-100). Defaults to 85.

    Returns:
        A list of :class:`ImageContent` with ``media_type="image/jpeg"``
        and base64-encoded JPEG bytes in ``data``. Always at least 1
        frame for a valid video, never more than ``max_frames``.

    Raises:
        InvalidRequestError: If ``imageio`` is not installed, if
            ``video.source_type == "url"``, or if the video bytes
            cannot be decoded.
    """
    if video.source_type == "url":
        raise InvalidRequestError(
            "URL-sourced VideoContent cannot be frame-extracted for "
            "non-native providers. Either download the video bytes "
            "and pass them as a base64 VideoContent, or send the "
            "video to a provider with native video input (Gemini)."
        )

    iio = require_imageio()
    raw = base64.standard_b64decode(video.data)
    extension = _extension_for(video.media_type)
    buf = BytesIO(raw)

    try:
        # immeta returns dict-like with at least 'fps' and 'duration'
        # for video containers.
        meta = iio.immeta(buf, extension=extension)
    except Exception as e:
        raise InvalidRequestError(f"Could not decode video: {e}") from e

    src_fps = float(meta.get("fps") or 30.0)
    duration = meta.get("duration")
    if duration is None:
        # Fall back to counting frames if duration metadata is absent.
        buf.seek(0)
        n_frames_total = sum(1 for _ in iio.imiter(buf, extension=extension))
        duration = n_frames_total / src_fps
    else:
        duration = float(duration)
        n_frames_total = round(duration * src_fps)

    if n_frames_total <= 0:
        raise InvalidRequestError("Video appears to have zero frames.")

    # How many frames do we want?
    target = max(1, min(max_frames, round(duration * fps)))
    if target == 1:
        # Single-frame case: take the middle frame.
        sample_indices = [n_frames_total // 2]
    else:
        # Uniform sampling across [0, n_frames_total) with `target` slots.
        step = n_frames_total / target
        sample_indices = [int(i * step) for i in range(target)]

    want = set(sample_indices)
    frames_by_index: dict[int, Any] = {}
    buf.seek(0)
    for i, frame in enumerate(iio.imiter(buf, extension=extension)):
        if i in want:
            frames_by_index[i] = frame
            if len(frames_by_index) == len(want):
                break

    out: list[ImageContent] = []
    for idx in sample_indices:
        frame = frames_by_index.get(idx)
        if frame is None:
            continue  # decoder gave up early; skip rather than fail
        out_buf = BytesIO()
        iio.imwrite(out_buf, frame, extension=".jpg", quality=jpeg_quality)
        out.append(
            ImageContent(
                source_type="base64",
                media_type="image/jpeg",
                data=out_buf.getvalue(),  # type: ignore[arg-type]
            )
        )

    if not out:
        raise InvalidRequestError("Frame extraction yielded zero frames.")

    return out


def substitute_video_with_frames(
    parts: list[Any],
    *,
    provider_name: str,
    fps: float = 1.0,
    max_frames: int = 8,
) -> list[Any]:
    """Replace any VideoContent parts in a content-part list with extracted frames.

    Emits a ``loguru`` warning per substituted video so consumers can
    see the cost implication. If no VideoContent is present, returns
    ``parts`` unchanged (with no warning).

    Args:
        parts: A list of vox content parts (TextContent / ImageContent
            / VideoContent).
        provider_name: Name of the calling provider, used in the warning
            message (e.g. ``"openai"``).
        fps: Frame sampling rate, forwarded to :func:`extract_frames`.
        max_frames: Frame cap, forwarded to :func:`extract_frames`.

    Returns:
        A new list with each VideoContent replaced by N ImageContent
        items, preserving the original ordering of other parts.
    """
    from loguru import logger

    if not any(isinstance(p, VideoContent) for p in parts):
        return parts

    new_parts: list[Any] = []
    for part in parts:
        if isinstance(part, VideoContent):
            frames = extract_frames(part, fps=fps, max_frames=max_frames)
            logger.warning(
                "Provider {provider} has no native video input; auto-extracted "
                "{n} frames from {mt} video (fps={fps}, max={max_frames}). "
                "Image-token cost will be ~{n}x a single image. Pass "
                "ImageContent parts directly to suppress this.",
                provider=provider_name,
                n=len(frames),
                mt=part.media_type,
                fps=fps,
                max_frames=max_frames,
            )
            new_parts.extend(frames)
        else:
            new_parts.append(part)
    return new_parts
