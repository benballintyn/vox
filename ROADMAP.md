# vox roadmap

Status: **v0.1.1 shipped.** The initial design — five providers (OpenAI,
Anthropic, Gemini, OpenRouter, LM Studio), sync + async, streaming, tool
calling, structured output, reasoning/thinking, multimodal — is complete and
released on PyPI as `vox-llm`. Nothing from the original plan is half-built.

This document is forward-looking: candidate next work, not commitments.

## How vox is prioritized

vox is **demand-driven**. The governing rule (see the "Vox: core LLM client
across projects" preference): when a downstream project — Tomte, Ithildin, a
new one — needs an LLM capability vox lacks, the feature ships *in vox first*
rather than reaching for a provider SDK directly. So most items below get
built when a consumer actually needs them, not speculatively.

The one exception is correctness work, which happens regardless of demand.

## Correctness — do regardless of demand

- **Live integration tests.** The entire suite is currently mocked — every
  provider test exercises a fake SDK client. Provider APIs drift (response
  shapes, new fields, renamed events), and a mocked suite cannot catch that.
  The `@pytest.mark.integration` marker already exists as the hook; what's
  missing is an opt-in CI job (keys via repo secrets, run on demand or on a
  schedule) that hits each provider's real API with a tiny prompt and
  asserts the translation still holds. This is the highest-value next step —
  it protects every other feature.

## Priority candidates — build proactively, ahead of demand

These are flagged above the standard demand-driven list because each
is (a) a substantial cross-provider abstraction lift in its own right,
and (b) anticipates patterns (multimodal voice assistants, video
analysis pipelines) likely to land on vox before the canonical
"a consumer is blocked on this" trigger.

- **Audio I/O.** Speech-to-text input, text-to-speech output. Each
  provider's API is differently shaped — OpenAI exposes audio via the
  Responses API on the `gpt-4o-audio` family (input + output
  modalities); Gemini accepts native audio input on 2.x+ via the
  standard `inline_data` / `file_data` parts, with text-to-speech as a
  separate TTS surface; Anthropic doesn't yet have a native audio path.
  Likely shape: an `AudioContent` content-part type parallel to
  `ImageContent` (raw `bytes` / base64 / URL, with `media_type`); a
  streaming-audio-out path on `CompletionResponse` or as a new content
  kind. Surface design is non-trivial; co-design with a real consumer
  pattern if possible.
- **Video input.** Gemini natively supports video as a `Part`
  (`inline_data` with `video/*` MIME, or via uploaded files); OpenAI's
  Responses API accepts video frames through specific models;
  Anthropic doesn't have a native video surface. Likely shape: a
  `VideoContent` content-part type parallel to `ImageContent`, same
  bytes/base64/URL surface, per-provider routing. Less total surface
  than audio (input only — no output side), but high per-provider
  variance.

## Candidate features — pull in when a consumer needs one

- **Prompt caching control.** `Usage` already *reports* `cache_read_tokens`
  and `cache_creation_tokens`, but there is no way to *set* cache
  breakpoints. Anthropic needs `cache_control` markers on content blocks;
  Gemini has an explicit context-caching API; OpenAI caches automatically
  (no surface needed). Real cost savings for consumers with large, stable
  system prompts.
- **Unified retry / backoff.** vox currently relies on each SDK's own
  `max_retries`. A vox-level, configurable retry policy that honours the
  `retry_after` already extracted onto `RateLimitError` would give
  consistent behaviour across providers.
- **Pre-flight token counting.** Each provider has its own tokenizer
  (tiktoken for OpenAI, Anthropic's tokenizer endpoint, etc.). Adding
  one would let consumers estimate token usage before paying for the
  API call. Heavier than cost estimation; held until a consumer pulls.
  (Post-flight **cost estimation** shipped in **v0.3.0** —
  `usage.estimated_cost` + the built-in `MODEL_PRICING` snapshot + a
  `custom_pricing` override on `VoxClient`. See `vox._pricing`.)
- **OpenAI stop sequences.** The Responses API has no `stop` parameter, so
  `stop` is silently dropped for the OpenAI provider. If a consumer needs
  it, it can be emulated with a `logit_bias` on the stop-token IDs.
- **Request metadata passthrough.** A unified way to attach provider
  `metadata` / `user` tracking fields to a request.
- **Streaming + structured output together.** Currently mutually exclusive
  by design (`response_schema` is rejected on stream methods). Could be
  supported as stream-for-UX-then-validate.
- **Additional providers.** AWS Bedrock, Azure OpenAI, Vertex AI, Mistral,
  Groq, etc. Add only when a consumer needs one — OpenRouter already covers
  much of this surface.

## Needs a design decision first

- **Optional tool-loop helper.** vox is deliberately low-level: it does one
  request/response, and the caller runs the tool-call loop. Consumers
  (e.g. Ithildin's `AnalysisAgent`) hand-roll that loop. A thin, optional
  `run_tools`-style convenience could de-duplicate it — but it risks
  scope creep into agent-framework territory. Decide the boundary before
  building.

## Explicitly out of scope

Recording these so the roadmap doesn't drift into them:

- **Embeddings.** The embedding-model market moves faster than chat APIs;
  embeddings stay in `google-genai` directly (the documented carve-out).
- **OAuth / subscription auth.** vox is API-key-only by design.
- **Infrastructure coupling.** No health registries, no Redis-backed
  routing, no proxy service. vox is a library; a consuming application can
  layer those on top.
