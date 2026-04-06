"""Provider registry for model-to-provider resolution."""

from __future__ import annotations

from .errors import InvalidRequestError

# Prefix-based mapping from model name to provider name.
# Checked in order; first match wins.
PROVIDER_PREFIXES: list[tuple[str, str]] = [
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("claude-", "anthropic"),
    ("gemini-", "gemini"),
]


def resolve_provider(model: str, explicit_provider: str | None = None) -> str:
    """Determine which provider to use for a given model.

    Args:
        model: The model identifier string.
        explicit_provider: Explicitly specified provider name, takes priority.

    Returns:
        The provider name string.

    Raises:
        InvalidRequestError: If the provider cannot be determined.
    """
    if explicit_provider:
        return explicit_provider

    for prefix, provider in PROVIDER_PREFIXES:
        if model.startswith(prefix):
            return provider

    raise InvalidRequestError(
        f"Cannot determine provider for model '{model}'. "
        "Use the provider= parameter to specify explicitly."
    )
