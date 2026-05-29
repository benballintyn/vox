# Roadmap

vox is **demand-driven** — most features ship when a downstream consumer needs one, not speculatively. The two exceptions are correctness work (always shipped) and high-value enablers the maintainer wants.

The authoritative roadmap lives in [`ROADMAP.md`](https://github.com/benballintyn/vox/blob/main/ROADMAP.md) in the repo. The short version:

## Shipped

The "Priority candidates — build proactively" tier is currently empty — both items shipped:

- **Audio I/O** (v0.5.0) — `transcribe()` + `synthesize()` methods. See the [Audio guide](guides/audio.md).
- **Video input** (v0.4.0) — `VideoContent` with native Gemini support + client-side frame-extraction fallback. See the [Multimodal guide](guides/multimodal.md).
- **Unified retry / backoff** (v0.6.0) — `RetryPolicy` honouring `RateLimitError.retry_after`. See [Reliability → Retries](guides/reliability.md#retries).
- **Observability hooks** (v0.6.0) — `CallbackHandler` protocol with OpenTelemetry alignment. See [Reliability → Callbacks](guides/reliability.md#callbacks-observability-hooks).

## Demand-driven (waiting for a consumer pull)

- **Prompt caching control** — `cache_control` markers for Anthropic, context-caching API for Gemini.
- **Pre-flight token counting** — sibling to `estimate_cost`.
- **PDF / document content blocks** — `DocumentContent` content-part type parallel to `ImageContent` / `VideoContent` / `AudioContent`.
- **Tool-choice normalization** — typed `tool_choice` parameter.
- **Server-side built-in tools** — typed `WebSearchTool()` / `CodeExecutionTool()` / etc. (today via the raw-dict escape hatch).
- **Tool definitions from Pydantic models / function signatures** — `Tool.from_function(weather_lookup)`.
- **Additional providers** — Bedrock, Azure OpenAI, Vertex AI, Mistral, Groq.
- **Streaming + structured output together** — currently mutually exclusive.
- **Anthropic citations**, **Gemini safety settings**, **OpenAI service tier**.

## Bigger lifts; demand-driven

- **Logprobs.**
- **Batch API.**
- **Live / realtime APIs** (OpenAI Realtime, Gemini Live).

## Explicitly out of scope

- **Embeddings** — `google-genai` directly.
- **OAuth / subscription auth** — API-key only by design.
- **Infrastructure coupling** — no proxy / Redis / health registry; vox stays a library.
- **Image generation.**
- **Rate-limiting / cost-limiting enforcement** — cost tracking is sufficient; budget enforcement belongs at the application layer.
