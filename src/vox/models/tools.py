"""Tool/function calling types."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Tool(BaseModel):
    """Tool/function definition for LLM tool calling.

    Args:
        name: The name of the tool/function.
        description: A description of what the tool does.
        parameters: JSON Schema object describing the tool's parameters.
    """

    name: str
    description: str
    parameters: dict[str, Any]


# A tool entry accepted by provider methods. Either a vox ``Tool`` (translated
# to the provider's function-tool format) or a raw dict passed through to the
# provider verbatim. The dict form is the escape hatch for provider-native
# server-side tools — e.g. Anthropic's ``web_search_20250305``, OpenAI's
# ``web_search_preview`` — which have provider-specific shapes and no
# cross-provider abstraction. The caller is responsible for matching the
# resolved provider's expected schema.
ToolSpec = Tool | dict[str, Any]

# Shared error message for invalid entries in a `tools` list. Used by every
# provider's `_translate_tools` so the failure mode is consistent.
TOOL_SPEC_TYPE_ERROR = (
    "Each entry in `tools` must be a vox.Tool or a dict. Got {got}. "
    "Pass a vox.Tool for ordinary function tools, or a raw dict for "
    "provider-native server-side tools (e.g. Anthropic's web_search_20250305)."
)


class ToolCall(BaseModel):
    """A tool call made by the model.

    Args:
        id: Unique identifier for this tool call.
        name: Name of the tool/function to call.
        arguments: Parsed arguments for the tool call.
    """

    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """Result of a tool call execution.

    Convenience class for building tool result messages.

    Args:
        tool_call_id: ID of the tool call this result is for.
        name: Name of the tool that was called.
        content: String content of the tool result.
        is_error: Whether this result represents an error.
    """

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False

    def to_message(self) -> Message:  # noqa: F821
        """Convert this tool result to a Message with role='tool'.

        Returns:
            A Message instance suitable for inclusion in a conversation,
            with ``is_error`` propagated from this ToolResult.
        """
        from .messages import Message

        return Message(
            role="tool",
            content=self.content,
            tool_call_id=self.tool_call_id,
            name=self.name,
            is_error=self.is_error,
        )
