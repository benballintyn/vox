"""Tests for structured output helpers."""

import json

import pytest
from pydantic import BaseModel

from vox._structured import (
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


class TestOpenAIResponsesTextFormat:
    """Tests for Responses API text.format generation."""

    def test_basic_schema(self) -> None:
        result = pydantic_to_openai_responses_text_format(WeatherReport)
        assert "format" in result
        fmt = result["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["name"] == "WeatherReport"
        assert fmt["strict"] is True


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
