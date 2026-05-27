"""Message and content types for LLM conversations."""

from __future__ import annotations

import base64
from typing import Any, Literal

from pydantic import BaseModel, field_validator


class TextContent(BaseModel):
    """Text content block within a message."""

    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """Image content for multimodal messages.

    Supports both base64-encoded images and URLs.

    Args:
        type: Content-part discriminator. Always ``"image"``.
        source_type: ``"base64"`` (default — ``data`` is a base64 string
            *or* raw image bytes, see ``data`` below) or ``"url"``
            (``data`` is an http(s) URL).
        media_type: MIME type, e.g. ``"image/png"`` / ``"image/jpeg"``
            / ``"image/webp"``. Ignored when ``source_type="url"``.
        data: For ``source_type="base64"``, either a base64-encoded
            ASCII string OR raw ``bytes`` — raw bytes are
            auto-encoded for you (saves the caller a base64 line
            when reading an image from disk / a buffer). For
            ``source_type="url"``, the URL string. Always stored
            internally as a ``str`` after construction.
    """

    type: Literal["image"] = "image"
    source_type: Literal["base64", "url"] = "base64"
    media_type: str = "image/png"
    data: str

    @field_validator("data", mode="before")
    @classmethod
    def _bytes_to_base64(cls, v: Any) -> Any:
        """Auto-encode raw image bytes to a base64 ASCII string.

        Callers commonly hold raw bytes (from a file read, a Pillow
        ``BytesIO``, a downloaded blob) and would otherwise have to
        write ``base64.standard_b64encode(b).decode("ascii")`` at
        every call site. This validator does it for them. Strings pass
        through untouched — including base64 strings and URLs.

        Only ``bytes`` / ``bytearray`` get the conversion treatment; if
        you somehow have raw image bytes that happen to live behind a
        ``source_type="url"`` (which makes no sense), the conversion
        still runs, but you'd get a base64 string in a URL field —
        garbage in, garbage out.
        """
        if isinstance(v, (bytes, bytearray)):
            return base64.standard_b64encode(bytes(v)).decode("ascii")
        return v


class VideoContent(BaseModel):
    """Video content for multimodal messages.

    Sibling of :class:`ImageContent`. Same source/data shape; the
    per-provider routing differs:

    * **Gemini** consumes video natively as a ``video/*`` ``Part`` —
      either an ``inline_data`` blob (base64) or a ``file_data`` URI
      (hosted file, including YouTube URLs).
    * **OpenAI, Anthropic, OpenRouter, LM Studio** have no native
      video-input content part as of this writing. vox falls back to
      client-side frame extraction (via the optional ``vox-llm[video]``
      extra) and substitutes :class:`ImageContent` parts before
      dispatch. A loud ``loguru`` warning is emitted so the cost
      implication is visible. Consumers who want explicit control over
      sampling can pass ``ImageContent`` parts directly instead.

    Args:
        type: Content-part discriminator. Always ``"video"``.
        source_type: ``"base64"`` (default — ``data`` is a base64 string
            or raw video ``bytes``, auto-encoded) or ``"url"``
            (``data`` is a URL or provider file URI — for Gemini, the
            URL may be a YouTube link).
        media_type: MIME type, e.g. ``"video/mp4"`` / ``"video/webm"``
            / ``"video/quicktime"``. Ignored when ``source_type="url"``
            for some providers (Gemini still uses it).
        data: For ``source_type="base64"``, either a base64 ASCII
            string OR raw ``bytes`` — raw bytes are auto-encoded for
            you. For ``source_type="url"``, the URL string. Always
            stored internally as ``str``.
    """

    type: Literal["video"] = "video"
    source_type: Literal["base64", "url"] = "base64"
    media_type: str = "video/mp4"
    data: str

    @field_validator("data", mode="before")
    @classmethod
    def _bytes_to_base64(cls, v: Any) -> Any:
        """Auto-encode raw video bytes to a base64 ASCII string.

        Mirrors :meth:`ImageContent._bytes_to_base64` — callers commonly
        hold raw bytes from ``Path.read_bytes()`` or a downloaded blob
        and would otherwise need to base64-encode at every call site.
        """
        if isinstance(v, (bytes, bytearray)):
            return base64.standard_b64encode(bytes(v)).decode("ascii")
        return v


ContentPart = TextContent | ImageContent | VideoContent


class ToolCallData(BaseModel):
    """A tool call made by the model, embedded in a message.

    Args:
        id: Public, cross-provider identifier for the call. Consumers
            reference this in the ``tool_call_id`` of a ``ToolResult`` /
            tool-role ``Message`` when replying.
        name: Tool/function name.
        arguments: Parsed arguments as a dict.
        provider_state: Opaque per-provider state attached to the call
            by the provider that produced it. Each provider's adapter
            populates its own keys (e.g. ``openai_fc_id`` carries the
            ``fc_*`` item ID needed when round-tripping through the
            Responses API; ``gemini_thought_signature`` carries the
            encrypted bytes Gemini requires on subsequent turns). The
            *consumer* never needs to read or modify this — pass the
            ``ToolCallData`` back unchanged in the assistant message
            history and the provider that minted it will use what it
            stored. Built-from-scratch ``ToolCallData`` (e.g. in tests)
            can leave this ``None``.
    """

    id: str
    name: str
    arguments: dict[str, Any]
    provider_state: dict[str, Any] | None = None


class Message(BaseModel):
    """A single message in an LLM conversation.

    Args:
        role: The role of the message sender.
        content: Text string or list of content parts (for multimodal).
        tool_calls: Tool calls made by the assistant in this message.
        tool_call_id: ID of the tool call this message is a result for.
        name: Name of the tool for tool result messages.
        is_error: Whether this tool result represents an error. Only meaningful
            for ``role="tool"`` messages. Providers that support tool error
            signaling (e.g. Anthropic) use this flag.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart] = ""
    tool_calls: list[ToolCallData] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    is_error: bool = False

    @property
    def text(self) -> str:
        """Extract plain text content from the message.

        Returns:
            Concatenated text from all TextContent parts, or the string content directly.
        """
        if isinstance(self.content, str):
            return self.content
        return "".join(p.text for p in self.content if isinstance(p, TextContent))
