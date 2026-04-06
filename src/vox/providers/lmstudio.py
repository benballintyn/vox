"""LM Studio provider using the Chat Completions API protocol.

Targets locally-hosted models via LM Studio's OpenAI-compatible endpoint.
"""

from __future__ import annotations

from ..models.config import ProviderConfig
from ._chat_completions import ChatCompletionsProvider


class LMStudioProvider(ChatCompletionsProvider):
    """LM Studio local provider.

    Uses the OpenAI Chat Completions API protocol with LM Studio's local endpoint.
    API key validation is skipped by LM Studio on localhost.

    Args:
        config: Provider configuration. Defaults to ``http://localhost:1234/v1``
            and ``api_key="lm-studio"``.
    """

    _default_base_url: str = "http://localhost:1234/v1"
    _default_api_key_env: str = "LMSTUDIO_API_KEY"
    _default_model: str = "local-model"

    def __init__(self, config: ProviderConfig | None = None) -> None:
        config = config or ProviderConfig()
        if config.api_key is None:
            config.api_key = "lm-studio"
        super().__init__(config)

    @property
    def provider_name(self) -> str:
        return "lmstudio"
