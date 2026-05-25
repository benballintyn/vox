"""Structured output helpers.

Converts Pydantic models to provider-specific schema formats and validates
raw LLM responses back into typed Pydantic instances.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from .errors import InvalidRequestError


def _enforce_openai_strict_schema(schema: Any) -> Any:
    """Transform a JSON Schema into OpenAI strict-mode-compatible form.

    OpenAI's structured output (both ``response_format.json_schema`` on
    Chat Completions and ``text.format`` on the Responses API) only
    accepts schemas that satisfy *strict mode*. Two of those constraints
    are violated by the schemas Pydantic's ``model_json_schema()`` emits
    by default:

    1. Every ``type: object`` (or any node with ``properties``) must
       have ``additionalProperties: false``.
    2. Every property must be listed in ``required`` — strict mode does
       not allow optional fields at the JSON Schema level.

    This walks the schema recursively — including children under
    ``$defs`` / ``definitions``, ``items``, ``anyOf`` / ``oneOf`` /
    ``allOf`` — and applies both invariants in-place. Non-object nodes
    pass through untouched.

    It also collapses any node containing ``$ref`` down to just
    ``{"$ref": <pointer>}``. OpenAI strict mode forbids sibling
    keywords on a ``$ref`` (per the historical JSON Schema spec, ``$ref``
    replaced the rest of the node), but Pydantic emits
    ``{"$ref": ..., "description": ...}`` for referenced sub-models.

    Other strict-mode constraints (limited keywords elsewhere, no
    ``default``, ``format`` restrictions, etc.) are the caller's
    responsibility — vox fixes only the requirements that
    ``model_json_schema()`` doesn't satisfy out of the box.

    Args:
        schema: A JSON Schema dict (or sub-schema).

    Returns:
        A new dict with the strict-mode invariants applied. The input
        is not mutated.
    """
    if not isinstance(schema, dict):
        return schema

    # ``$ref`` nodes are not allowed to carry sibling keywords in OpenAI
    # strict mode — per the JSON Schema spec, ``$ref`` historically
    # replaced all other keywords on its node. Pydantic emits
    # ``{"$ref": ..., "description": ...}`` for referenced sub-models,
    # which the API rejects. Strip everything else.
    if "$ref" in schema:
        return {"$ref": schema["$ref"]}

    result = dict(schema)

    # Object nodes get the two strict-mode invariants. Pydantic emits
    # ``"type": "object"`` for models; we also catch the schema-author
    # case of bare ``"properties": {...}`` without an explicit type.
    if result.get("type") == "object" or "properties" in result:
        properties = result.get("properties", {}) or {}
        result["additionalProperties"] = False
        result["required"] = list(properties.keys())
        result["properties"] = {k: _enforce_openai_strict_schema(v) for k, v in properties.items()}

    # ``items`` — array element schema (single or list-of-schemas tuple form).
    items = result.get("items")
    if isinstance(items, list):
        result["items"] = [_enforce_openai_strict_schema(i) for i in items]
    elif isinstance(items, dict):
        result["items"] = _enforce_openai_strict_schema(items)

    # Combinators.
    for key in ("anyOf", "oneOf", "allOf"):
        if key in result and isinstance(result[key], list):
            result[key] = [_enforce_openai_strict_schema(s) for s in result[key]]

    # Definitions (both Draft 2020-12 ``$defs`` and legacy ``definitions``).
    for defs_key in ("$defs", "definitions"):
        if defs_key in result and isinstance(result[defs_key], dict):
            result[defs_key] = {
                k: _enforce_openai_strict_schema(v) for k, v in result[defs_key].items()
            }

    return result


def pydantic_to_openai_response_format(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to OpenAI ``response_format`` for Chat Completions.

    The emitted schema is transformed into strict-mode-compatible form
    (``additionalProperties: false`` + every property in ``required``)
    so the Chat Completions ``response_format.json_schema`` path accepts
    it. See :func:`_enforce_openai_strict_schema`.

    Args:
        model_cls: The Pydantic model class.

    Returns:
        A dict suitable for the ``response_format`` parameter.
    """
    schema = _enforce_openai_strict_schema(model_cls.model_json_schema())
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

    Same strict-mode transformation as
    :func:`pydantic_to_openai_response_format` — the Responses API
    enforces the same requirements as Chat Completions for json_schema
    outputs.

    Args:
        model_cls: The Pydantic model class.

    Returns:
        A dict suitable for the ``text`` parameter's ``format`` field.
    """
    schema = _enforce_openai_strict_schema(model_cls.model_json_schema())
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
