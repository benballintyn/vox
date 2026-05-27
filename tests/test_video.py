"""Tests for vox._video — frame extraction and the substitute helper.

Generates a small synthetic video at module load (via imageio) so each
test runs without needing a fixture file checked into the repo.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import numpy as np
import pytest
from pytest_mock import MockerFixture

from vox import ImageContent, TextContent, VideoContent
from vox._video import (
    extract_frames,
    require_imageio,
    substitute_video_with_frames,
)
from vox.errors import InvalidRequestError

# How long the test video is and at what frame rate. Kept tiny — the
# substitution-and-cap tests only care that frames count math works.
TEST_VIDEO_DURATION_S = 4
TEST_VIDEO_FPS = 10
TEST_VIDEO_SIZE = (32, 32)


@pytest.fixture(scope="module")
def video_bytes() -> bytes:
    """Synthesize a tiny mp4 in memory: 4 seconds at 10 fps, 32x32.

    Each frame is a solid color that walks through the spectrum, so
    the decoded frames are visually distinguishable (helpful when
    debugging sampling logic).
    """
    import imageio.v3 as iio

    n_frames = TEST_VIDEO_DURATION_S * TEST_VIDEO_FPS
    frames = np.zeros((n_frames, *TEST_VIDEO_SIZE, 3), dtype=np.uint8)
    for i in range(n_frames):
        # Walk red channel from 0..255 across the clip.
        frames[i, :, :, 0] = int(255 * i / max(1, n_frames - 1))

    buf = BytesIO()
    iio.imwrite(
        buf,
        frames,
        extension=".mp4",
        fps=TEST_VIDEO_FPS,
        codec="libx264",
    )
    return buf.getvalue()


@pytest.fixture
def base64_video(video_bytes: bytes) -> VideoContent:
    return VideoContent(
        source_type="base64",
        media_type="video/mp4",
        data=base64.standard_b64encode(video_bytes).decode("ascii"),
    )


class TestRequireImageio:
    def test_returns_module_when_installed(self) -> None:
        mod = require_imageio()
        assert hasattr(mod, "imwrite")

    def test_raises_invalid_request_when_missing(self, mocker: MockerFixture) -> None:
        """Mock import to raise ImportError and confirm we re-raise as vox error."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("imageio"):
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        mocker.patch.object(builtins, "__import__", side_effect=fake_import)

        with pytest.raises(InvalidRequestError, match="vox-llm\\[video\\]"):
            require_imageio()


class TestExtractFrames:
    def test_happy_path_returns_image_contents(self, base64_video: VideoContent) -> None:
        frames = extract_frames(base64_video, fps=1.0, max_frames=8)
        assert len(frames) >= 1
        assert all(isinstance(f, ImageContent) for f in frames)
        assert all(f.media_type == "image/jpeg" for f in frames)
        assert all(f.source_type == "base64" for f in frames)
        # Each frame should be a non-trivial JPEG.
        for f in frames:
            raw = base64.standard_b64decode(f.data)
            assert raw[:3] == b"\xff\xd8\xff"  # JPEG SOI marker

    def test_max_frames_caps_output(self, base64_video: VideoContent) -> None:
        # 4s clip at fps=10 would yield 40 frames; cap forces 3.
        frames = extract_frames(base64_video, fps=10.0, max_frames=3)
        assert len(frames) == 3

    def test_low_fps_yields_fewer_frames(self, base64_video: VideoContent) -> None:
        # 4-second video at fps=0.5 → 2 frames target.
        frames = extract_frames(base64_video, fps=0.5, max_frames=8)
        assert len(frames) == 2

    def test_single_frame_when_target_is_one(self, base64_video: VideoContent) -> None:
        frames = extract_frames(base64_video, fps=0.1, max_frames=8)
        # 4s * 0.1fps = 0.4 → rounds to 0 → clamped to 1.
        assert len(frames) == 1

    def test_url_source_refused(self) -> None:
        video = VideoContent(
            source_type="url",
            media_type="video/mp4",
            data="https://example.com/clip.mp4",
        )
        with pytest.raises(InvalidRequestError, match="URL-sourced VideoContent"):
            extract_frames(video)

    def test_invalid_video_bytes_raises(self) -> None:
        garbage = VideoContent(data=base64.standard_b64encode(b"not a video").decode())
        with pytest.raises(InvalidRequestError, match="Could not decode video"):
            extract_frames(garbage)


class TestSubstituteVideoWithFrames:
    def test_passthrough_when_no_video(self) -> None:
        parts = [TextContent(text="hi"), ImageContent(data="ZmFrZQ==")]
        result = substitute_video_with_frames(parts, provider_name="openai")
        assert result == parts

    def test_replaces_video_with_frames(self, base64_video: VideoContent) -> None:
        parts = [TextContent(text="describe"), base64_video]
        result = substitute_video_with_frames(parts, provider_name="openai", fps=0.5, max_frames=8)
        # Text part preserved; video replaced by N image parts.
        assert isinstance(result[0], TextContent)
        assert len(result) >= 2
        assert all(isinstance(p, ImageContent) for p in result[1:])

    def test_emits_loud_warning(self, base64_video: VideoContent, mocker: MockerFixture) -> None:
        spy = mocker.patch("loguru.logger.warning")
        substitute_video_with_frames(
            [base64_video], provider_name="anthropic", fps=1.0, max_frames=4
        )
        spy.assert_called_once()
        msg, kwargs = spy.call_args.args[0], spy.call_args.kwargs
        assert "no native video input" in msg
        assert kwargs["provider"] == "anthropic"
        assert kwargs["n"] >= 1
