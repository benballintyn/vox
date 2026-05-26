# Changelog

All notable changes to vox are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0](https://github.com/benballintyn/vox/compare/v0.2.0...v0.3.0) (2026-05-26)


### Features

* **pricing:** post-flight cost estimation on Usage ([#33](https://github.com/benballintyn/vox/issues/33)) ([bccffff](https://github.com/benballintyn/vox/commit/bccffffb65d52aeae535570cd502f6e08a881603))

## [0.2.0](https://github.com/benballintyn/vox/compare/v0.1.2...v0.2.0) (2026-05-26)


### Features

* **messages:** accept raw bytes in ImageContent + multimodal combo coverage ([#30](https://github.com/benballintyn/vox/issues/30)) ([f2d5fa4](https://github.com/benballintyn/vox/commit/f2d5fa412e9c7d153cb03355f1c377787a2dcbf1))

## [0.1.2](https://github.com/benballintyn/vox/compare/v0.1.1...v0.1.2) (2026-05-26)


### Bug Fixes

* **gemini:** missing model kwarg + tool-call finish_reason; add live integration suite ([#16](https://github.com/benballintyn/vox/issues/16)) ([c7ee002](https://github.com/benballintyn/vox/commit/c7ee0021e92bf32c93a9e3d33c6e584796fbbf69))
* **openai:** enforce strict-mode JSON Schema for structured output ([#23](https://github.com/benballintyn/vox/issues/23)) ([295744a](https://github.com/benballintyn/vox/commit/295744a46adfc5bc2f19e324cf0308b15af2f6e6))
* **openai:** preserve and replay reasoning items on tool round-trip ([#28](https://github.com/benballintyn/vox/issues/28)) ([faf2abb](https://github.com/benballintyn/vox/commit/faf2abb550ed1b850101e666d066e8e6bba3c036))
* **openrouter:** extract usage from chunks that also carry choices ([#29](https://github.com/benballintyn/vox/issues/29)) ([f086b70](https://github.com/benballintyn/vox/commit/f086b70b0660d02d6468a47fc9fa9d6d40c2d909))
* **streaming:** usage chunks, ordering, and tool-arg delta correlation ([#26](https://github.com/benballintyn/vox/issues/26)) ([d95a4d5](https://github.com/benballintyn/vox/commit/d95a4d5cdc31f02b20d1c5eca237ba31b1e1d75a))
* **tools:** preserve provider-specific tool-call state across turns ([#24](https://github.com/benballintyn/vox/issues/24)) ([7adfc5c](https://github.com/benballintyn/vox/commit/7adfc5cbcaa68f4088ef7661407c21d2f16d249d))


### Documentation

* add ROADMAP.md with forward-looking next steps ([#14](https://github.com/benballintyn/vox/issues/14)) ([8e805ed](https://github.com/benballintyn/vox/commit/8e805edbca44c786b82fbec411bfafefbb04be08))

## [0.1.1](https://github.com/benballintyn/vox/compare/v0.1.0...v0.1.1) (2026-05-22)


### Bug Fixes

* **tool-use:** accept provider-native tool dicts in tools list ([#8](https://github.com/benballintyn/vox/issues/8)) ([#9](https://github.com/benballintyn/vox/issues/9)) ([c43d9f7](https://github.com/benballintyn/vox/commit/c43d9f7c6d9e2a9312c2dc6c630baa3cf8882330))
* **packaging:** ship py.typed marker so downstream type checkers use vox's type hints (PEP 561) ([#11](https://github.com/benballintyn/vox/issues/11)) ([1b8d767](https://github.com/benballintyn/vox/commit/1b8d76718c20ef1cff75d9da3c6e68ff6a71f80c))

## [Unreleased]

## [0.1.0] — 2026-05-14

First public release. Distributed on PyPI as `vox-llm` (the bare name was
taken); the Python import remains `import vox`.

### Core

- `VoxClient` — one entry point, every provider. Resolves the provider from
  the model name; sync, async, streaming, and async-streaming variants.
- Providers: OpenAI (Chat Completions + Responses API), Anthropic, Google
  Gemini, OpenRouter, LM Studio.
- Vision / multimodal messages via `ImageContent` (base64 or URL).

### Tool use

- Single `Tool` schema translated to each provider's native function-calling
  format.
- `ToolResult` + `Message.is_error` for explicit error propagation — replaces
  the earlier name-prefix heuristic.

### Structured output

- Pass a Pydantic model as `response_schema=`; the SDK translates it to JSON
  Schema (OpenAI / OpenRouter / LM Studio), Anthropic synthetic tool, or
  Gemini's `response_schema` parameter, then validates the response back.

### Reasoning / extended thinking

- `ReasoningConfig.level` — cross-provider semantic intensity
  (`minimal` / `low` / `medium` / `high`) for portable code.
- Per-provider sub-configs (`OpenAIReasoning`, `AnthropicReasoning`,
  `GeminiReasoning`) override the semantic level when you need provider-
  specific control (GPT-5 `minimal`, Anthropic budget tokens, Gemini 2.5
  budgets, Gemini 3 `thinkingLevel`).
- Thinking blocks surface on `CompletionResponse.thinking` and as `thinking`
  stream chunks.

### Stateful conversations (OpenAI Responses API)

- `previous_response_id` + `store=True` on OpenAI calls lets the model
  remember prior turns server-side without resending history.
- `CompletionResponse.response_id` is populated by every provider so callers
  can chain follow-ups uniformly.

### Streaming

- `StreamChunk` discriminated union: `text`, `tool_call_start`,
  `tool_call_delta`, `thinking`, `usage`, `done`.
- Chat-Completions stream translator emits both `tool_call_start` and the
  first `tool_call_delta` from a single SDK chunk, fixing a bug where
  initial argument deltas were dropped (broke JSON for OpenRouter-routed
  non-OpenAI providers).

### Errors

- Normalised hierarchy: `AuthenticationError`, `RateLimitError`,
  `QuotaExceededError`, `InvalidRequestError`, `ProviderError`,
  `ContentFilterError`, `ModelNotFoundError` — all subclasses of `VoxError`.
- `RateLimitError.retry_after` is populated from each provider's
  `Retry-After` response header.
- `finish_reason` normalised across providers — `stop`, `length`,
  `tool_calls`, `content_filter`, `stop_sequence`, `other`. The raw
  provider string remains on `CompletionResponse.raw_finish_reason`.

[Unreleased]: https://github.com/benballintyn/vox/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/benballintyn/vox/releases/tag/v0.1.0
