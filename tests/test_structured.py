"""Tests for structured output helpers."""

import json
from enum import StrEnum

import pytest
from pydantic import BaseModel

from vox._structured import (
    _enforce_openai_strict_schema,
    pydantic_to_anthropic_tool,
    pydantic_to_gemini_schema,
    pydantic_to_openai_response_format,
    pydantic_to_openai_responses_text_format,
    validate_structured_response,
)
from vox.errors import InvalidRequestError


class WeatherReport(BaseModel):
    """Test model for structured output."""

    city: str
    temperature_f: float
    conditions: str
    forecast: list[str]


class SimpleResult(BaseModel):
    """Minimal test model."""

    answer: str
    confidence: float


class _Continent(StrEnum):
    EUROPE = "europe"
    ASIA = "asia"


class _City(BaseModel):
    name: str
    population: int


class _CountryReport(BaseModel):
    """Nested + list + enum + $defs schema target for strict-mode tests."""

    country: str
    continent: _Continent
    cities: list[_City]


class TestOpenAIStrictSchemaEnforcement:
    """Tests for ``_enforce_openai_strict_schema`` — vox#21.

    The two invariants tested everywhere:

    * Every object node has ``additionalProperties: false``.
    * Every object node's ``required`` lists every key in ``properties``.

    Non-object nodes (strings, enums, arrays-of-strings) pass through
    untouched. Recursion covers ``properties``, ``items``, ``$defs`` /
    ``definitions``, and ``anyOf`` / ``oneOf`` / ``allOf``.
    """

    def test_flat_object(self) -> None:
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        }
        result = _enforce_openai_strict_schema(schema)
        assert result["additionalProperties"] is False
        assert sorted(result["required"]) == ["a", "b"]

    def test_overrides_existing_required(self) -> None:
        """Pydantic puts only non-default fields in ``required``; strict needs all."""
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a"],
        }
        result = _enforce_openai_strict_schema(schema)
        assert sorted(result["required"]) == ["a", "b"]

    def test_recurses_into_nested_object(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                }
            },
        }
        result = _enforce_openai_strict_schema(schema)
        inner = result["properties"]["inner"]
        assert inner["additionalProperties"] is False
        assert inner["required"] == ["x"]

    def test_recurses_into_array_items(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                    },
                }
            },
        }
        result = _enforce_openai_strict_schema(schema)
        item = result["properties"]["rows"]["items"]
        assert item["additionalProperties"] is False
        assert item["required"] == ["id"]

    def test_recurses_into_defs(self) -> None:
        schema = {
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {"k": {"type": "string"}},
                }
            },
            "type": "object",
            "properties": {"x": {"$ref": "#/$defs/Inner"}},
        }
        result = _enforce_openai_strict_schema(schema)
        inner = result["$defs"]["Inner"]
        assert inner["additionalProperties"] is False
        assert inner["required"] == ["k"]

    def test_recurses_into_anyof(self) -> None:
        schema = {
            "anyOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}},
                {"type": "string"},
            ]
        }
        result = _enforce_openai_strict_schema(schema)
        obj_branch = result["anyOf"][0]
        assert obj_branch["additionalProperties"] is False
        assert obj_branch["required"] == ["a"]
        # Non-object branch untouched.
        assert result["anyOf"][1] == {"type": "string"}

    def test_non_object_passes_through(self) -> None:
        schema = {"type": "string", "enum": ["a", "b"]}
        result = _enforce_openai_strict_schema(schema)
        assert result == schema

    def test_strips_siblings_from_ref_nodes(self) -> None:
        """OpenAI rejects ``description`` (and other keys) on a ``$ref`` node.

        Pydantic emits ``{"$ref": ..., "description": ...}`` for
        referenced sub-models; collapse to just ``{"$ref": ...}``.
        """
        schema = {
            "$ref": "#/$defs/Inner",
            "description": "this should be stripped",
            "title": "ditto",
        }
        result = _enforce_openai_strict_schema(schema)
        assert result == {"$ref": "#/$defs/Inner"}

    def test_strips_ref_siblings_when_nested(self) -> None:
        """``$ref`` sibling-stripping recurses through containers."""
        schema = {
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {"k": {"type": "string"}},
                }
            },
            "type": "object",
            "properties": {
                "x": {"$ref": "#/$defs/Inner", "description": "an Inner"},
            },
        }
        result = _enforce_openai_strict_schema(schema)
        assert result["properties"]["x"] == {"$ref": "#/$defs/Inner"}

    def test_does_not_mutate_input(self) -> None:
        """Transformation must be non-destructive."""
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
        }
        original = json.loads(json.dumps(schema))
        _enforce_openai_strict_schema(schema)
        assert schema == original


class TestOpenAIResponseFormat:
    """Tests for Chat Completions response_format generation."""

    def test_basic_schema(self) -> None:
        result = pydantic_to_openai_response_format(WeatherReport)
        assert result["type"] == "json_schema"
        assert result["json_schema"]["name"] == "WeatherReport"
        assert result["json_schema"]["strict"] is True
        schema = result["json_schema"]["schema"]
        assert "city" in schema["properties"]
        assert "forecast" in schema["properties"]

    def test_simple_model(self) -> None:
        result = pydantic_to_openai_response_format(SimpleResult)
        assert result["json_schema"]["name"] == "SimpleResult"
        props = result["json_schema"]["schema"]["properties"]
        assert props["confidence"]["type"] == "number"

    def test_strict_invariants_applied_at_top_level(self) -> None:
        """vox#21 — Chat Completions strict mode requires ``additionalProperties: false``."""
        result = pydantic_to_openai_response_format(WeatherReport)
        schema = result["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert sorted(schema["required"]) == sorted(WeatherReport.model_fields.keys())

    def test_strict_invariants_applied_to_nested_defs(self) -> None:
        """Nested models surface as ``$defs``; strict mode applies there too."""
        result = pydantic_to_openai_response_format(_CountryReport)
        schema = result["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        inner_city = schema["$defs"]["_City"]
        assert inner_city["additionalProperties"] is False
        assert sorted(inner_city["required"]) == ["name", "population"]


class TestOpenAIResponsesTextFormat:
    """Tests for Responses API text.format generation."""

    def test_basic_schema(self) -> None:
        result = pydantic_to_openai_responses_text_format(WeatherReport)
        assert "format" in result
        fmt = result["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["name"] == "WeatherReport"
        assert fmt["strict"] is True

    def test_strict_invariants_applied_at_top_level(self) -> None:
        """vox#21 — Responses API enforces the same strict-mode requirements."""
        result = pydantic_to_openai_responses_text_format(WeatherReport)
        schema = result["format"]["schema"]
        assert schema["additionalProperties"] is False
        assert sorted(schema["required"]) == sorted(WeatherReport.model_fields.keys())

    def test_strict_invariants_applied_to_nested_defs(self) -> None:
        result = pydantic_to_openai_responses_text_format(_CountryReport)
        schema = result["format"]["schema"]
        inner_city = schema["$defs"]["_City"]
        assert inner_city["additionalProperties"] is False
        assert sorted(inner_city["required"]) == ["name", "population"]


class TestAnthropicTool:
    """Tests for Anthropic synthetic tool generation."""

    def test_basic_tool(self) -> None:
        result = pydantic_to_anthropic_tool(WeatherReport)
        assert result["name"] == "structured_output"
        assert "WeatherReport" in result["description"]
        assert "input_schema" in result
        schema = result["input_schema"]
        assert "city" in schema["properties"]


class TestGeminiSchema:
    """Tests for Gemini response schema generation."""

    def test_basic_schema(self) -> None:
        result = pydantic_to_gemini_schema(WeatherReport)
        assert "properties" in result
        assert "city" in result["properties"]


class TestValidateStructuredResponse:
    """Tests for response validation."""

    def test_valid_json_string(self) -> None:
        raw = json.dumps(
            {
                "city": "NYC",
                "temperature_f": 72.0,
                "conditions": "sunny",
                "forecast": ["clear", "warm"],
            }
        )
        result = validate_structured_response(WeatherReport, raw)
        assert isinstance(result, WeatherReport)
        assert result.city == "NYC"
        assert result.temperature_f == 72.0

    def test_valid_dict(self) -> None:
        raw = {
            "city": "LA",
            "temperature_f": 85.0,
            "conditions": "hot",
            "forecast": ["sunny"],
        }
        result = validate_structured_response(WeatherReport, raw)
        assert isinstance(result, WeatherReport)
        assert result.city == "LA"

    def test_invalid_json_string(self) -> None:
        with pytest.raises(InvalidRequestError, match="validation failed"):
            validate_structured_response(WeatherReport, "not json")

    def test_missing_required_field(self) -> None:
        raw = {"city": "NYC"}  # Missing temperature_f, conditions, forecast
        with pytest.raises(InvalidRequestError, match="validation failed"):
            validate_structured_response(WeatherReport, raw)

    def test_wrong_type_raises(self) -> None:
        raw = {
            "city": "NYC",
            "temperature_f": "not a number",
            "conditions": "sunny",
            "forecast": ["clear"],
        }
        with pytest.raises(InvalidRequestError, match="validation failed"):
            validate_structured_response(WeatherReport, raw)
