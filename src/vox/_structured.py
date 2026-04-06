"""Structured output helpers.

Converts Pydantic models to provider-specific schema formats and validates
raw LLM responses back into typed Pydantic instances.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from .errors import InvalidRequestError


def pydantic_to_openai_response_format(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to OpenAI ``response_format`` for Chat Completions.

    Args:
        model_cls: The Pydantic model class.

    Returns:
        A dict suitable for the ``response_format`` parameter.
    """
    schema = model_cls.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model_cls.__name__,
            "strict": True,
            "schema": schema,
        },
    }


def pydantic_to_openai_responses_text_format(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to OpenAI Responses API ``text.format``.

    Args:
        model_cls: The Pydantic model class.

    Returns:
        A dict suitable for the ``text`` parameter's ``format`` field.
    """
    schema = model_cls.model_json_schema()
    return {
        "format": {
            "type": "json_schema",
            "name": model_cls.__name__,
            "strict": True,
            "schema": schema,
        }
    }


def pydantic_to_anthropic_tool(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to a synthetic Anthropic tool definition.

    Anthropic structured outputs work by defining a tool whose input_schema
    matches the desired output schema, then forcing the model to call that tool.

    Args:
        model_cls: The Pydantic model class.

    Returns:
        A tool definition dict for Anthropic's API.
    """
    schema = model_cls.model_json_schema()
    return {
        "name": "structured_output",
        "description": (
            f"Return a structured response matching the {model_cls.__name__} schema. "
            "You MUST call this tool with your response."
        ),
        "input_schema": schema,
    }


def pydantic_to_gemini_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to a Gemini response schema.

    Args:
        model_cls: The Pydantic model class.

    Returns:
        JSON Schema dict for Gemini's ``response_schema`` parameter.
    """
    return model_cls.model_json_schema()


def validate_structured_response(
    model_cls: type[BaseModel],
    raw: str | dict[str, Any],
) -> BaseModel:
    """Validate a raw LLM response against a Pydantic model.

    Args:
        model_cls: The Pydantic model class to validate against.
        raw: Raw response as a JSON string or dict.

    Returns:
        A validated instance of ``model_cls``.

    Raises:
        InvalidRequestError: If validation fails.
    """
    try:
        if isinstance(raw, str):
            return model_cls.model_validate_json(raw)
        return model_cls.model_validate(raw)
    except (ValidationError, json.JSONDecodeError) as e:
        raise InvalidRequestError(
            f"Structured output validation failed for {model_cls.__name__}: {e}"
        ) from e
