"""Structured-output integration tests.

Schema-translation bugs typically only surface on non-flat shapes (nested
models, arrays, enums). Both flat and nested cases are exercised.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from vox import Message, VoxClient

from .conftest import ProviderProfile


class City(BaseModel):
    """Flat-shape schema target."""

    name: str = Field(description="The city name.")
    country: str = Field(description="The country it's in.")
    population: int = Field(description="Approximate population.")


class Continent(StrEnum):
    EUROPE = "europe"
    ASIA = "asia"
    AFRICA = "africa"
    NORTH_AMERICA = "north_america"
    SOUTH_AMERICA = "south_america"
    OCEANIA = "oceania"
    ANTARCTICA = "antarctica"


class CountryReport(BaseModel):
    """Nested + list + enum schema target."""

    country: str = Field(description="The country name.")
    continent: Continent = Field(description="Which continent.")
    cities: list[City] = Field(description="Major cities in this country.")


def test_flat_structured_output(profile: ProviderProfile, client: VoxClient) -> None:
    """``response_schema`` produces a validated Pydantic instance.

    Asserts structure (correct type, fields populated) only — never on
    the model's choice of values.
    """
    response = client.complete(
        [Message(role="user", content="Tell me about Paris, France.")],
        model=profile.model,
        response_schema=City,
        max_tokens=2048,
    )
    assert response.parsed is not None, "response.parsed was not populated"
    assert isinstance(response.parsed, City)
    assert response.parsed.name
    assert response.parsed.country
    assert response.parsed.population > 0


def test_nested_list_enum_structured_output(profile: ProviderProfile, client: VoxClient) -> None:
    """Nested + list + enum schema validates end-to-end.

    Each provider has its own schema-translation path
    (OpenAI ``response_format``, Anthropic synthetic tool, Gemini
    ``response_schema``). Flat shapes often work even when the translator
    has subtle bugs; non-flat shapes are the real test.
    """
    response = client.complete(
        [
            Message(
                role="user",
                content=("Produce a country report for France including 2-3 major cities."),
            )
        ],
        model=profile.model,
        response_schema=CountryReport,
        max_tokens=2048,
    )
    assert response.parsed is not None
    assert isinstance(response.parsed, CountryReport)
    assert response.parsed.country
    assert isinstance(response.parsed.continent, Continent)
    assert response.parsed.cities, "expected non-empty cities list"
    assert all(isinstance(c, City) for c in response.parsed.cities)
