"""Message and content types for LLM conversations."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class TextContent(BaseModel):
    """Text content block within a message."""

    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """Image content for multimodal messages.

    Supports both base64-encoded images and URLs.
    """

    type: Literal["image"] = "image"
    source_type: Literal["base64", "url"] = "base64"
    media_type: str = "image/png"
    data: str  # base64 string or URL


ContentPart = TextContent | ImageContent


class ToolCallData(BaseModel):
    """A tool call made by the model, embedded in a message."""

    id: str
    name: str
    arguments: dict[str, Any]


class Message(BaseModel):
    """A single message in an LLM conversation.

    Args:
        role: The role of the message sender.
        content: Text string or list of content parts (for multimodal).
        tool_calls: Tool calls made by the assistant in this message.
        tool_call_id: ID of the tool call this message is a result for.
        name: Name of the tool for tool result messages.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart] = ""
    tool_calls: list[ToolCallData] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @property
    def text(self) -> str:
        """Extract plain text content from the message.

        Returns:
            Concatenated text from all TextContent parts, or the string content directly.
        """
        if isinstance(self.content, str):
            return self.content
        return "".join(p.text for p in self.content if isinstance(p, TextContent))
