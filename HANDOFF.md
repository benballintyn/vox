# Handoff — vox (next agent session)

Working doc for the next Claude session picking up vox. Snapshot date:
**2026-05-26**, after v0.3.0 shipped. Keep this current; update on
session end if the picture has shifted.

## Where things stand

| | |
|---|---|
| **Latest release** | `vox-llm` 0.3.0 on PyPI. `import vox`. |
| **Repo** | github.com/benballintyn/vox (public, personal). Local at `~/personal/repos/vox`. |
| **Open issues** | None. |
| **Integration suite** | 99 passed, 3 skipped, 0 xfailed, 0 failed. Lives in `tests/integration/`; manual dispatch only (`gh workflow run integration.yml --ref <branch>`). |
| **Unit suite** | 244 passed under `pytest -m "not integration"`. Run in CI on every PR. |
| **Release automation** | release-please. `feat:` → minor, `fix:` → patch, `chore:`/`docs:`/`ci:`/`test:`/etc. → no bump. PR titles are load-bearing (squash-merge subject == PR title). |
| **PyPI publish** | Auto via `release_please.yml` → `pypi-publish.yml` (reusable workflow). **`attestations: false`** is set — PyPI doesn't support PEP 740 attestations through reusable-workflow chains. Don't re-enable without checking [pypa#166](https://github.com/pypa/gh-action-pypi-publish/issues/166). |

## What ships

vox is a model-agnostic LLM client covering five providers (OpenAI
Responses API, Anthropic, Gemini, OpenRouter, LM Studio). The
abstraction layer:

- `complete` / `acomplete` / `stream` / `astream` — the four entry
  points.
- Normalized `Message` / `Tool` / `ToolCallData` / `ToolResult` /
  `ReasoningConfig` / `ImageContent` (accepts raw `bytes`) /
  `CompletionResponse` / `StreamChunk` / `Usage` (with `model` +
  `estimated_cost` populated by VoxClient).
- `response_schema` → validated Pydantic instance on the response.
- `ModelPricing` + `estimate_cost` + `MODEL_PRICING` for cost
  estimation. `VoxClient(custom_pricing={...})` to override.

All of this is end-to-end-tested against live provider APIs via
`tests/integration/` — 99 tests across the four CI-runnable providers.
The suite is manual dispatch (not on every PR); run it before
merging release PRs.

## Architecture cheat sheet

```
src/vox/
  client.py            VoxClient facade (resolves provider, populates Usage cost)
  _registry.py         model-name → provider resolution (prefix + explicit)
  _pricing.py          ModelPricing + MODEL_PRICING + estimate_cost + resolve_pricing
  _structured.py       Pydantic → provider schema translators
  errors.py            normalized VoxError hierarchy
  models/
    messages.py        Message, ContentPart (Text/Image), ToolCallData
    config.py          ProviderConfig
    reasoning.py       ReasoningConfig + per-provider escape hatches
    responses.py       CompletionResponse, Usage (+ model + estimated_cost), StreamChunk
    tools.py           Tool / ToolCall / ToolResult / ToolSpec
  providers/
    base.py            Abstract Provider
    _chat_completions.py  Shared base for OpenAI Chat Completions protocol
    openai.py          Responses API
    anthropic.py
    gemini.py
    openrouter.py      thin wrapper over _chat_completions.py
    lmstudio.py        thin wrapper over _chat_completions.py
```

VoxClient is the integration point — providers stay ignorant of cost,
custom_pricing, and the public model name. The client populates
`usage.model` and `usage.estimated_cost` after the provider returns.

Streaming providers use a per-stream `state: dict[str, Any]` to carry
context across events (e.g. buffered tool-call IDs, deferred `done`).

## Recent work (the last big push that brought us here)

Eight PRs landed in close succession (May 22–26):

| # | Theme | Issues closed |
|---|---|---|
| 16 | Live integration test suite + two Gemini fixes | — (laid the foundation) |
| 23 | OpenAI strict-mode JSON Schema for structured output | 21 |
| 24 | `ToolCallData.provider_state` for tool-call round-trips | 17, 22 |
| 26 | Streaming refactor: usage chunks, ordering, tool-arg correlation | 18, 19, 20 |
| 28 | OpenAI reasoning-item replay on tool round-trip | 25 |
| 29 | OpenRouter streaming usage extraction | 27 |
| 30 | ImageContent accepts raw bytes + multimodal combo tests | — |
| 32 | `attestations: false` (PyPI reusable-workflow limit) | — |
| 33 | Cost estimation (Usage.model, Usage.estimated_cost, MODEL_PRICING) | — |

The integration suite was the keystone — it surfaced every bug above.
Worth understanding the path: **mocks proved the translators given a
response shape; only live tests proved the response shape was still
real.** Six provider-translation bugs surfaced this way.

## Operational gotchas

These will bite the next session if not respected:

1. **PostToolUse formatter strips unused imports between edits.** Add
   an import AND a use of it in the same Write/Edit, or the formatter
   between calls will delete it. Burned us multiple times during the
   above PRs. See the memory page
   `PostToolUse formatter strips unused imports between edits`.

2. **Squash-merge commit subject = PR title** *only* if the repo's
   "Default commit message → Pull request title" setting is on. It
   is on `vox` — but the *PR title* still has to be a Conventional
   Commit (`feat:` / `fix:` / `chore:` / etc.). release-please reads
   the squash subject; mistyping `fix:` as `chore:` means no patch
   bump.

3. **PyPI Trusted Publishers × reusable workflows.** v0.1.2 and v0.2.0
   both failed the automated publish step before we figured this out.
   See the memory page `PyPI Trusted Publishers × reusable workflows`.
   Short version: keep `attestations: false`; if you need to re-publish
   a failed release manually, `gh workflow run pypi-publish.yml -f
   tag=vX.Y.Z` is the documented break-glass.

4. **Integration suite is manual dispatch only.** Don't expect every
   PR to run live API tests. Dispatch via `gh workflow run
   integration.yml --ref <branch>` before merging anything
   provider-touching. The `release-vox` skill walks this for releases.

5. **vox is demand-driven.** ROADMAP §"How vox is prioritized" — most
   features ship when a downstream consumer (Tomte / Ithildin / new)
   needs one. The exceptions are correctness work (always) and
   high-value enablers a maintainer wants. Resist speculative breadth.

## Skills + memory pages

For onboarding faster, search the claude-memory KG with queries about:

- `Vox release skill` — the operational skill that walks every release.
  Auto-loads when user says "release vox" / "ship vox" / etc.
- `Vox: core LLM client across projects` — the "use vox for all LLM
  access" preference + the release model.
- `PyPI Trusted Publishers × reusable workflows: PEP 740 attestation
  gotcha` — the publish failure we hit twice. Read before re-enabling
  attestations.
- `GitHub Actions: GITHUB_TOKEN events don't trigger workflows` —
  why the release-please → pypi-publish workflow_call architecture
  exists.
- `Squash-merge + Conventional Commits` — why PR titles matter.
- `PostToolUse formatter strips unused imports between edits` — the
  edit-flow gotcha.

The `release-vox` skill at `~/.claude/skills/release-vox/SKILL.md`
covers the release process end-to-end including failure modes.

## Open roadmap items (ROADMAP.md)

### Priority candidates — build proactively (added 2026-05-26)

The maintainer flagged these as worth building ahead of a specific
consumer pull. Each is a substantial cross-provider abstraction lift.

- **Audio I/O.** STT input, TTS output. OpenAI Responses API has audio
  modalities on the `gpt-4o-audio` family; Gemini has native audio
  input on 2.x+ and a separate TTS surface; Anthropic has no native
  audio yet. Likely shape: `AudioContent` content-part type parallel to
  `ImageContent`. See ROADMAP §"Priority candidates" for the design
  sketch.
- **Video input.** Gemini natively (`video/*` `inline_data` or
  uploaded files); OpenAI Responses API supports it on select models;
  Anthropic has no native video. Likely shape: `VideoContent`
  content-part type parallel to `ImageContent`.

### Demand-driven candidates — pull in when a consumer needs one

- **Prompt caching control.** `cache_control` markers on content blocks
  for Anthropic; explicit context-caching API for Gemini; OpenAI
  auto-caches and already populates `cache_read_tokens`. Significant
  cost-savings opportunity for consumers with large stable system
  prompts (Tomte's identify_thing pattern would benefit).
- **Unified retry / backoff** honoring the `retry_after` already
  surfaced on `RateLimitError`.
- **Pre-flight token counting.** Each provider has a tokenizer
  endpoint (OpenAI = tiktoken local, Anthropic = `messages/count_tokens`,
  Gemini = `models.count_tokens`). Sibling to `estimate_cost`.
- **OpenAI stop sequences** via `logit_bias` emulation (Responses API
  has no `stop`).
- **Request metadata passthrough** — unified surface for `metadata` /
  `user` tracking fields.
- **Streaming + structured output together** — currently mutually
  exclusive by design; stream-for-UX-then-validate is the obvious
  shape.
- **Additional providers** — Bedrock, Azure OpenAI, Vertex AI, Mistral,
  Groq. OpenRouter already covers most of this surface in practice.

## Brainstorm: bigger API gaps we don't cover yet

(These came out of an "audit the full provider API surface" exercise
on 2026-05-26. None are committed; logged here for the next session
to pick from when a consumer pulls.)

### Likely to pull soonest

1. **Server-side built-in tools** (web search, code execution, file
   search, computer use). All three majors expose them via wildly
   different shapes. Currently vox accepts them via the raw-dict
   escape hatch (`ToolSpec = Tool | dict[str, Any]`). A typed
   abstraction like `WebSearchTool()` / `CodeExecutionTool()` /
   `FileSearchTool(vector_store_id=...)` would translate per-provider.
   Tomte's research workflows are a strong pull candidate.

2. **PDF / document content blocks.** Anthropic has native `document`
   content blocks (PDF bytes inline). OpenAI does it through the Files
   API plus the `file_search` tool. Gemini does it through uploaded
   files. A `DocumentContent` content-part type, parallel to
   `ImageContent`, would unify. Tomte's `ingest_document` tool would
   benefit.

3. **Tool-choice normalization.** Currently passed via `**kwargs`. A
   typed `tool_choice: Literal["auto", "required", "none"] | str`
   (the str being a tool name) on `complete()` would normalize across
   providers (OpenAI's `"required"`, Anthropic's `{"type": "tool",
   "name": ...}`, Gemini's `function_calling_config`).

### Quality of life / observability

4. **Observability hooks.** Pluggable callbacks for request-sent /
   response-received / usage-recorded events. Lets consumers wire
   telemetry (DataDog, OpenTelemetry, custom logging) without
   monkey-patching the client.

5. **Tool definitions from Pydantic models / function signatures.**
   `Tool.from_function(weather_lookup)` that introspects parameters
   into JSON Schema. Ergonomic shortcut; Tomte has hand-rolled this
   pattern.

6. **Cost / token budget enforcement.** `VoxClient(max_cost_per_call=0.10)`
   raises before sending if the estimated cost (via vox's price table)
   exceeds the cap. Useful for runaway tool-loop safety.

### Bigger lifts; demand-driven

7. **Logprobs.** OpenAI + Gemini expose token-level log probabilities;
   Anthropic doesn't. Niche but useful for scoring / analysis. Surface
   as `LogProbs` on the response.

8. **Batch API.** OpenAI and Anthropic both have one (50% discount,
   24-hour latency). Different APIs; vox could abstract job creation /
   polling / result fetching. Useful for bulk inference workloads.

9. **Live / realtime APIs.** OpenAI Realtime, Gemini Live. Bidirectional
   streaming with voice. Very different shape from chat completion;
   likely its own module rather than an extension of `client.stream`.

(Audio I/O and video input were originally in this tier; they were
promoted to ROADMAP §"Priority candidates" on 2026-05-26 because the
maintainer wants them built ahead of consumer pull.)

### Provider-specific surfaces worth exposing

12. **Anthropic Citations.** When Anthropic returns citations for
    document inputs, vox currently drops them. Useful for RAG;
    propagate as a new field on the response message.

13. **Gemini safety settings.** Configurable `HARM_CATEGORY_*`
    thresholds. Currently not exposed; consumers get the defaults.

14. **OpenAI service tier.** `service_tier` (auto / scale / priority /
    flex) — affects pricing and latency.

### Explicitly out of scope (per existing carve-outs)

- **Embeddings** — `google-genai` directly; ROADMAP §"Explicitly out
  of scope".
- **OAuth / subscription auth** — API-key only by design.
- **Infrastructure coupling** — no proxy / Redis / health registry.
  vox stays a library.
- **Image generation** — separate API surface; out of scope.

## Consumers to know about

- **Tomte** (`~/personal/repos/tomte`) — uses vox extensively. Vision
  + tools + response_schema (per the multimodal-pull discussion on
  2026-05-25). Pin should be bumped to `vox-llm = ">=0.3.0,<0.4.0"`
  to get the latest fixes + cost estimation + bytes-accepting
  `ImageContent`.
- **Ithildin** (`~/personal/repos/ithildin`) — analysis agent that uses
  vox tool-calling.

When something lands in vox that those consumers can adopt, mentioning
it to the user (so they bump their consumer pin) is part of the loop.
The `release-vox` skill has a "post-release" step for exactly this.

## Don't forget

- **CHANGELOG.md is generated by release-please** — don't hand-edit on
  main. Edits on the release PR are OK if you merge it without
  another push to main in between.
- **Branch hygiene**: `feat/…` / `fix/…` / `test/…` / `ci/…` /
  `docs/…` / `refactor/…` / `chore/…`. Always go through PR; never
  commit to main directly. Branch protection enforces this.
- **`py.typed` marker ships** — vox is a typed library. Keep it
  type-clean (`mypy --strict` against `src` + `tests`).
- **Live tests cost money.** Cents per dispatch, but it's real money
  — don't dispatch the integration suite speculatively. The four
  CI-runnable providers' keys come from repo secrets configured by
  the user.
