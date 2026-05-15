# Changelog

All notable changes to vox are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — TBD

First public release. Distributed on PyPI as `vox-llm` (the bare name was taken);
the Python import remains `import vox`.

### Added
- `VoxClient` — unified entry point with sync/async + streaming + tool-use.
- Providers: OpenAI, Anthropic, Google Gemini, OpenRouter, LM Studio.
- Cross-provider tool calling via a single `Tool` schema.
- Structured output via Pydantic `response_schema=` — translated to each
  provider's native format.
- Vision / multimodal messages (`ImageContent`).
- Reasoning / extended thinking config for Anthropic, OpenAI o-series, and
  Gemini 2.5+.
- Normalised error hierarchy: `AuthenticationError`, `RateLimitError`,
  `QuotaExceededError`, `InvalidRequestError`, `ProviderError`,
  `ContentFilterError`, `ModelNotFoundError`.

[Unreleased]: https://github.com/benballintyn/vox/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/benballintyn/vox/releases/tag/v0.1.0
