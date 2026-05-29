# Handoff — vox (next agent session)

Working doc for the next Claude session picking up vox. Snapshot date:
**2026-05-28**, after v0.6.0 shipped. Keep this current; update on
session end if the picture has shifted.

## Where things stand

| | |
|---|---|
| **Latest release** | `vox-llm` 0.6.0 on PyPI. `import vox`. |
| **Repo** | github.com/benballintyn/vox (public, personal). Local at `~/personal/repos/vox`. Website field on the GitHub "About" sidebar links to `pypi.org/p/vox-llm`. |
| **Open issues** | None. |
| **Integration suite** | 111 tests, manual dispatch only (`gh workflow run integration.yml --ref <branch>`). Lives in `tests/integration/`. |
| **Unit suite** | 342 tests under `pytest -m "not integration"`. Run in CI on every PR across Python 3.11/3.12/3.13. |
| **Release automation** | release-please. `feat:` → minor, `fix:` → patch, `chore:`/`docs:`/`ci:`/`test:`/etc. → no bump. PR titles are load-bearing (squash-merge subject == PR title). |
| **PyPI publish** | Auto via `release_please.yml` → `pypi-publish.yml` (reusable workflow). **`attestations: false`** is set — PyPI doesn't support PEP 740 attestations through reusable-workflow chains. Don't re-enable without checking [pypa#166](https://github.com/pypa/gh-action-pypi-publish/issues/166). |
| **Pre-commit** | Installed locally on this clone. Hooks: pre-commit-hooks v6, ruff v0.15.14 (check + format), detect-secrets with a `.secrets.baseline` allowlisting test-fixture placeholders. New maintainers should run `pre-commit install` after cloning. |

## What ships

vox is a model-agnostic LLM client covering five providers (OpenAI
Responses API, Anthropic, Gemini, OpenRouter, LM Studio). The
abstraction layer:

**Chat / completion:**

- `complete` / `acomplete` / `stream` / `astream` — the four chat
  entry points.
- Normalized `Message` / `Tool` / `ToolCallData` / `ToolResult` /
  `ReasoningConfig` / `ImageContent` / `VideoContent` (Gemini native;
  client-side frame-extraction fallback elsewhere) /
  `CompletionResponse` / `StreamChunk` / `Usage` (with `model` +
  `estimated_cost` populated by VoxClient).
- `response_schema` → validated Pydantic instance on the response.

**Audio (dedicated methods, not bolted into `complete`):**

- `transcribe` / `atranscribe` — STT via OpenAI Whisper, Gemini
  generate_content with audio Part. Returns `TranscriptionResponse`.
- `synthesize` / `asynthesize` — TTS via OpenAI tts-1 / gpt-4o-mini-tts,
  Gemini gemini-3.1-flash-tts-preview (PCM-wrapped-as-WAV). Returns
  `AudioContent`.
- Anthropic / OpenRouter / LM Studio raise `InvalidRequestError`.

**Reliability + observability:**

- `RetryPolicy` — configurable per-call retries with exponential
  backoff + jitter; honours `RateLimitError.retry_after`;
  streaming-aware (retries only before first chunk).
- `CallbackHandler` protocol — `on_request` / `on_response` /
  `on_error` events fired around every entry point. Each event has a
  `to_otel_attributes()` helper keyed to OpenTelemetry GenAI
  semantic conventions. Built-in `LoggingHandler`. `capture_content`
  flag controls whether prompt/response payloads appear in events
  (default: no PII).

**Cost:**

- `ModelPricing` + `estimate_cost` + `MODEL_PRICING` for post-flight
  cost estimation. `VoxClient(custom_pricing={...})` to override.

All of this is end-to-end-tested against live provider APIs via
`tests/integration/` — 111 tests across the four CI-runnable
providers. The suite is manual dispatch (not on every PR); run it
before merging release PRs.

## Architecture cheat sheet

```
src/vox/
  client.py            VoxClient facade (resolves provider, wraps with
                        retry + callbacks, populates Usage cost)
  _registry.py         model-name → provider resolution (prefix + explicit)
  _pricing.py          ModelPricing + MODEL_PRICING + estimate_cost
  _retry.py            RetryPolicy + retry_sync/async + retry_stream_*
  _callbacks.py        CallbackHandler protocol + event models + LoggingHandler
  _video.py            Client-side frame extraction for non-Gemini providers
                        (used via vox-llm[video] extra)
  _structured.py       Pydantic → provider schema translators
  errors.py            normalized VoxError hierarchy
  models/
    messages.py        Message, ContentPart (Text/Image/Video),
                        AudioContent (input/output container, not in
                        ContentPart union), ToolCallData
    config.py          ProviderConfig
    reasoning.py       ReasoningConfig + per-provider escape hatches
    responses.py       CompletionResponse, Usage (+ model +
                        estimated_cost), StreamChunk, TranscriptionResponse
    tools.py           Tool / ToolCall / ToolResult / ToolSpec
  providers/
    base.py            Abstract Provider (+ default transcribe/synthesize
                        that raise InvalidRequestError)
    _chat_completions.py  Shared base for OpenAI Chat Completions protocol
    openai.py          Responses API + Whisper + tts-1/gpt-4o-mini-tts
    anthropic.py
    gemini.py          generate_content + native video/audio +
                        TTS via response_modalities=["AUDIO"]
    openrouter.py      thin wrapper over _chat_completions.py
    lmstudio.py        thin wrapper over _chat_completions.py
```

VoxClient is the integration point — providers stay ignorant of
cost, custom_pricing, callbacks, and the public model name. The
client wraps each adapter call in `retry_sync`/`retry_async` (or the
streaming variants), fires callbacks around the wrapped call, and
populates `usage.model` and `usage.estimated_cost` after the provider
returns.

Streaming providers use a per-stream `state: dict[str, Any]` to carry
context across events (e.g. buffered tool-call IDs, deferred `done`).

## Recent work (this session — 2026-05-26 → 2026-05-28)

Eight PRs landed across three feature releases plus three hygiene PRs:

| # | Theme | Release |
|---|---|---|
| 37 | `VideoContent` content part (Gemini native + frame-extraction fallback) | v0.4.0 |
| 39 | Bump CI actions off Node 20 ahead of deprecation | (chore) |
| 40 | Audio I/O — `transcribe`/`synthesize` methods (OpenAI Whisper + tts-1, Gemini generate_content + TTS) | v0.5.0 |
| 42 | `RetryPolicy` honouring `RateLimitError.retry_after`; streaming-aware | v0.6.0 |
| 43 | `CallbackHandler` protocol + event models with `to_otel_attributes()` | v0.6.0 |
| 45 | Adopt `detect-secrets` baseline + bump pre-commit deps to current | (chore) |
| 46 | README status badges | (docs) |

The earlier session (May 22–26) brought live-integration coverage
and the cluster of bug-fix PRs (#16, 23, 24, 26, 28–30, 32, 33) that
landed v0.3.0. That earlier work is still the foundation; everything
since adds new surfaces or polish.

## Operational gotchas

These will bite the next session if not respected:

1. **The PostToolUse formatter race is gone.** The global Claude
   Code hook that auto-ran `ruff check --fix` on every Edit/Write
   was removed from `~/.claude/settings.json` on 2026-05-28. ruff
   now runs only on `git commit` via pre-commit. If a future
   maintainer reinstalls the hook with `--fix`, the import-stripping
   race comes back — keep ruff to pre-commit only.

2. **Squash-merge commit subject = PR title** *only* if the repo's
   "Default commit message → Pull request title" setting is on. It
   is on `vox` — but the *PR title* still has to be a Conventional
   Commit (`feat:` / `fix:` / `chore:` / etc.). release-please reads
   the squash subject; mistyping `fix:` as `chore:` means no patch
   bump.

3. **PyPI Trusted Publishers × reusable workflows.** v0.1.2 and
   v0.2.0 both failed the automated publish step before we figured
   this out. See the memory page
   `PyPI Trusted Publishers × reusable workflows`. Short version:
   keep `attestations: false`; if you need to re-publish a failed
   release manually, `gh workflow run pypi-publish.yml -f tag=vX.Y.Z`
   is the documented break-glass.

4. **Integration suite is manual dispatch only.** Don't expect every
   PR to run live API tests. Dispatch via `gh workflow run
   integration.yml --ref <branch>` before merging anything
   provider-touching. The `release-vox` skill walks this for
   releases.

5. **vox is demand-driven.** ROADMAP §"How vox is prioritized" —
   most features ship when a downstream consumer (Tomte / Ithildin /
   new) needs one. The exceptions are correctness work (always),
   high-value enablers a maintainer wants, and items the maintainer
   has explicitly scoped (the Litellm-gap closing of retry +
   callbacks was the latter). Resist speculative breadth.

6. **`detect-secrets` will block commits if its baseline drifts.**
   The `.secrets.baseline` file allowlists existing test-fixture
   placeholders by hash. If you add a new fixture string that looks
   "secret-like" (e.g. `"sk-test-xyz"`), the pre-commit hook will
   refuse the commit until you re-run `detect-secrets scan >
   .secrets.baseline` to refresh.

## Skills + memory pages

For onboarding faster, search the claude-memory KG with queries
about:

- `Vox release skill` — the operational skill that walks every
  release. Auto-loads when user says "release vox" / "ship vox" /
  etc. **Note:** the skill was updated 2026-05-28 to NOT mention
  downstream consumers in the post-release output — see the
  feedback memory page below.
- `Vox: core LLM client across projects` — the "use vox for all LLM
  access" preference + the release model.
- `Vox release scope` (feedback) — don't surface downstream consumer
  pin bumps as a "follow-up" to a vox release; the release ends at
  the artifact links.
- `PyPI Trusted Publishers × reusable workflows: PEP 740 attestation
  gotcha` — the publish failure we hit twice. Read before
  re-enabling attestations.
- `GitHub Actions: GITHUB_TOKEN events don't trigger workflows` —
  why the release-please → pypi-publish workflow_call architecture
  exists.
- `Squash-merge + Conventional Commits` — why PR titles matter.
- `PostToolUse formatter strips unused imports between edits` —
  historical; the hook is gone but the memory captures *why* it
  happened in case anyone considers reintroducing the pattern.

The `release-vox` skill at `~/.claude/skills/release-vox/SKILL.md`
covers the release process end-to-end including failure modes.

## Open roadmap items (ROADMAP.md)

The "Priority candidates — build proactively" tier is **empty**
after this session — both Audio I/O and Video input shipped. The
"Unified retry / backoff" and "Observability hooks" candidates from
the demand-driven list also shipped. Remaining demand-driven items
(only do these when a real consumer pull surfaces):

- **Prompt caching control.** `cache_control` markers on content
  blocks for Anthropic; explicit context-caching API for Gemini;
  OpenAI auto-caches and already populates `cache_read_tokens`.
  Significant cost-savings opportunity for consumers with large
  stable system prompts (Tomte's identify_thing pattern would
  benefit).
- **Pre-flight token counting.** Each provider has a tokenizer
  endpoint (OpenAI = tiktoken local, Anthropic =
  `messages/count_tokens`, Gemini = `models.count_tokens`). Sibling
  to `estimate_cost`.
- **OpenAI stop sequences** via `logit_bias` emulation (Responses
  API has no `stop`).
- **Request metadata passthrough** — unified surface for `metadata`
  / `user` tracking fields.
- **Streaming + structured output together** — currently mutually
  exclusive by design; stream-for-UX-then-validate is the obvious
  shape.
- **Additional providers** — Bedrock, Azure OpenAI, Vertex AI,
  Mistral, Groq. OpenRouter already covers most of this surface in
  practice.

## Brainstorm: bigger API gaps we don't cover yet

Logged for the next session to pick from when a consumer pulls.
None are committed.

### Likely to pull soonest

1. **Server-side built-in tools** (web search, code execution, file
   search, computer use). All three majors expose them via wildly
   different shapes. Currently vox accepts them via the raw-dict
   escape hatch (`ToolSpec = Tool | dict[str, Any]`). A typed
   abstraction like `WebSearchTool()` / `CodeExecutionTool()` /
   `FileSearchTool(vector_store_id=...)` would translate
   per-provider. Tomte's research workflows are a strong pull
   candidate.

2. **PDF / document content blocks.** Anthropic has native
   `document` content blocks (PDF bytes inline). OpenAI does it
   through the Files API plus the `file_search` tool. Gemini does
   it through uploaded files. A `DocumentContent` content-part type
   parallel to `ImageContent` / `VideoContent` / `AudioContent`
   would unify. Tomte's `ingest_document` tool would benefit.

3. **Tool-choice normalization.** Currently passed via `**kwargs`.
   A typed `tool_choice: Literal["auto", "required", "none"] | str`
   (the str being a tool name) on `complete()` would normalize
   across providers (OpenAI's `"required"`, Anthropic's `{"type":
   "tool", "name": ...}`, Gemini's `function_calling_config`).

### Quality of life

4. **Tool definitions from Pydantic models / function signatures.**
   `Tool.from_function(weather_lookup)` that introspects parameters
   into JSON Schema. Ergonomic shortcut; Tomte has hand-rolled this
   pattern.

5. **Cost / token budget enforcement.** `VoxClient(max_cost_per_call=0.10)`
   raises before sending if the estimated cost (via vox's price
   table) exceeds the cap. Useful for runaway tool-loop safety. The
   maintainer explicitly declined this 2026-05-27 in favour of
   keeping cost concerns at the application layer; revisit only on
   pull.

### Bigger lifts; demand-driven

6. **Logprobs.** OpenAI + Gemini expose token-level log
   probabilities; Anthropic doesn't. Niche but useful for scoring /
   analysis. Surface as `LogProbs` on the response.

7. **Batch API.** OpenAI and Anthropic both have one (50% discount,
   24-hour latency). Different APIs; vox could abstract job
   creation / polling / result fetching. Useful for bulk inference
   workloads.

8. **Live / realtime APIs.** OpenAI Realtime, Gemini Live.
   Bidirectional streaming with voice. Very different shape from
   chat completion; likely its own module rather than an extension
   of `client.stream`.

### Provider-specific surfaces worth exposing

9. **Anthropic Citations.** When Anthropic returns citations for
   document inputs, vox currently drops them. Useful for RAG;
   propagate as a new field on the response message.

10. **Gemini safety settings.** Configurable `HARM_CATEGORY_*`
    thresholds. Currently not exposed; consumers get the defaults.

11. **OpenAI service tier.** `service_tier` (auto / scale /
    priority / flex) — affects pricing and latency.

### Hygiene worth knowing about

- **Audio model pricing.** `MODEL_PRICING` doesn't include
  `whisper-1` / `tts-1` / `gpt-4o-mini-tts`. Means
  `Usage.estimated_cost` is silently `None` for Gemini transcribe
  (which does report usage). Trivial — add ~5 entries to
  `_pricing.py`.
- **Sync/async duplication in audio providers.** Each provider has
  4 near-identical method bodies (sync + async × transcribe +
  synthesize). Cosmetic refactor, not urgent.
- **Codecov badge.** Would require a Codecov upload step in CI.
  Skipped in the badges PR (#46); can be wired in later.
- **Docs site** (in progress as of this snapshot — mkdocs-material
  + mkdocstrings-python + GitHub Pages). Companion PR to this one.

### Explicitly out of scope (per existing carve-outs)

- **Embeddings** — `google-genai` directly; ROADMAP §"Explicitly
  out of scope".
- **OAuth / subscription auth** — API-key only by design.
- **Infrastructure coupling** — no proxy / Redis / health registry.
  vox stays a library.
- **Image generation** — separate API surface; out of scope.
- **Rate-limiting / cost-limiting enforcement** — maintainer
  decision 2026-05-27; cost tracking is sufficient.

## Consumers to know about

- **Tomte** (`~/personal/repos/tomte`) — uses vox extensively for
  chat + tools + vision + structured output.
- **Ithildin** (`~/personal/repos/ithildin`) — analysis agent that
  uses vox tool-calling.

The maintainer manages consumer-pin bumps on their own cadence (see
the `Vox release scope` feedback memory page) — do **not** surface
"bump Tomte / Ithildin pins" as a follow-up to vox work.

## Don't forget

- **CHANGELOG.md is generated by release-please** — don't hand-edit
  on main. Edits on the release PR are OK if you merge it without
  another push to main in between.
- **Branch hygiene**: `feat/…` / `fix/…` / `test/…` / `ci/…` /
  `docs/…` / `refactor/…` / `chore/…`. Always go through PR; never
  commit to main directly. Branch protection enforces this.
- **`py.typed` marker ships** — vox is a typed library. Keep it
  type-clean (`mypy --strict` against `src` + `tests`).
- **Live tests cost money.** Cents per dispatch, but it's real
  money — don't dispatch the integration suite speculatively. The
  four CI-runnable providers' keys come from repo secrets
  configured by the user.
- **detect-secrets is in pre-commit.** New test-fixture strings
  that look secret-like will fail the commit until you re-scan the
  baseline (see operational gotcha #6).
