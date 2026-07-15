# OpenCode harness → wyltek-gpt gap analysis

**Date:** 2026-07-15
**Author:** research pass (Claude Code)
**Subject under study:** [OpenCode](https://opencode.ai) — the open-source terminal AI coding agent by SST ([github.com/sst/opencode](https://github.com/sst/opencode)).
**Compared to:** wyltek-gpt (`/home/phill/local-chatbot`) — a mobile-first, self-hostable local-inference chat PWA for Ollama models.
**Version note:** OpenCode docs surveyed as of mid-2026 (community docs mirror pinned at v1.2.27; internals cross-checked against the [cefboud OpenCode deep-dive, 2026](https://cefboud.com/posts/coding-agents-internals-opencode-deepdive/) and the official [opencode.ai/docs](https://opencode.ai/docs/)). This is research only — **no code was changed**.

> **Framing.** wyltek-gpt is *not* a coding IDE and should not become one. It is a local-model chat PWA with harness-like machinery (a tool registry, a hybrid tool-calling loop, capability probing, progressive-disclosure context injection, and a bounded write→check→fix agentic loop). OpenCode is a mature terminal *coding* agent. The value of this comparison is to mine OpenCode's harness patterns for the handful that transfer to a small-local-model chat product — and to explicitly rule out the ones that don't.

---

## 1. OpenCode at a glance

OpenCode runs a **client/server architecture**. A backend server owns all AI-model communication, tool execution, and session storage (a local SQLite database); the frontends — terminal TUI, desktop app, VS Code extension, web UI — are thin clients that talk to that server over HTTP + Server-Sent Events (a shared event bus rebroadcasts every message part in real time, so multiple clients can watch one session). It is **bring-your-own-key**: no built-in models, 75+ providers (Anthropic, OpenAI, Google, OpenAI-compatible endpoints, and **local via Ollama**) reached through Vercel's AI SDK, so the loop code is provider-agnostic.

The agent loop is the classic **"LLM in a loop with actions"** built on the AI SDK's `streamText`. It assembles system prompt + history + tool schemas + user input, streams a typed `fullStream` of events (`start-step`, `tool-call`, `tool-result`, `text-delta`, `finish-step`, …), executes tool `execute()` functions in the Bun runtime, feeds results back, and continues until a `stopWhen`/`steps` condition fires. At each step boundary it snapshots the working tree (git) for rollback and tallies token usage/cost. **When token usage crosses ~90% of the model's context limit it auto-summarizes the conversation** via a dedicated compaction/summary agent. It ships built-in **agents** (Build, Plan) and **subagents** (General, Explore, Scout) plus invisible **system agents** (Title, Summary, Compaction), each configurable via markdown+frontmatter or JSON with per-agent model, prompt, temperature, step cap, tool set, and **allow/ask/deny permissions**. It speaks **MCP** (local and remote servers) and supports **custom commands** (markdown prompt templates with argument, shell, and file interpolation).

---

## 2. Feature matrix

| Capability | OpenCode | wyltek-gpt | Transferability note |
|---|---|---|---|
| Agentic loop (LLM + tools, bounded) | Yes — `streamText`, `stopWhen`/steps, typed event stream | **Partial** — recursive `relay()` in `app.py`, per-bridge iteration caps (`MAX_CELLC_ITERS=5`, `MAX_HTMLDEMO_ITERS=4`), write→check→fix | Core pattern already present and well-shaped; gaps are in cross-cutting concerns (cancellation, retries, unified step accounting), not the loop itself. |
| Tool registry / definition | Yes — tools = description + JSON-schema params + `execute()` | Yes — `OperationRegistry` (YAML-declared `local`/`bridge`/`converter` ops) + in-process bridges (`cellc`, `open_palette`, `html_demo`) with `available()/tool_schemas()/dispatch()` | Registry model is comparable and arguably cleaner for its scope (declarative slot validators as the security seam). |
| Hybrid tool-calling for weak models | No (assumes capable models) | **Yes (differentiator)** — native `tool_calls` OR parsed `op:<name> {json}` fence fallback | wyltek-gpt is *ahead* here; local small models need this. Keep it. |
| Model capability handling | Provider metadata via models.dev | Yes — `/api/show` metadata probe → `capabilities.json`, UI glyphs, warn-and-allow op gate | Comparable; wyltek-gpt's per-model glyphs + op capability gate are a nice local-first touch. |
| Per-tool permission model | Yes — `allow`/`ask`/`deny`, bash glob patterns, per-agent | **Partial** — hardcoded: file-producing ops route through a Y/N/E confirm card; `cellc`/`demo` tools auto-run in-loop | Declarative permissions would generalize the current ad-hoc gating; matters as roadmap adds shell + filesystem tools. |
| Provider abstraction | Yes — 75+ providers, one code path | **Missing** — Ollama-only | An OpenAI-compatible shim (llama.cpp server, LM Studio, vLLM, remote tailnet) transfers; cloud BYOK does not (off-mission). |
| Server-side session persistence | Yes — SQLite, resumable | **Missing** — `session_id` is only a workspace *file* folder; chat history lives client-side | Roadmap already lists "Persistent chat history (SQLite)". Enabler for good compaction + cross-device. |
| Context compaction / summarization | **Yes — auto at ~90% context** | **Missing** — history grows unbounded until it overflows `num_ctx` | **The single highest-value gap** (directly the stated pain point). |
| Progressive-disclosure context | Partial (rules/AGENTS.md, on-demand file reads) | **Yes (strength)** — cellc/html-demo reference injected only on detected intent; per-file inject cap (60k chars) | wyltek-gpt already does this well; OpenCode validates the approach. |
| Handle/pointer for large payloads | Implicit (files on disk, read on demand) | **Yes (strength)** — html-demo `DemoStore` keeps full HTML server-side under a `demo_id`, patched via find/replace | Same instinct as OpenCode's edit tool; generalizable to other large artifacts. |
| Token-budget accounting | Yes — usage/cost per `finish-step` | **Partial** — Ollama `prompt_eval_count`/`eval_count` already stream to a perf bar, but not used as a budget | Cheap to turn into a live context meter + compaction trigger. |
| Sub-agents / task delegation | Yes — Task tool spawns isolated-context subagent, returns final text | **Missing** (except `auto_router` captioner, a one-off) | Context-isolation is a big lever for small models; higher effort. |
| Agents/modes (build/plan, named presets) | Yes — primary + system agents, per-agent config | **Partial** — implicit "modes" via intent detection (cellc/demo) + specialized routes (`translate`, auto-caption) | Formalizing modes = selectable system prompt + model + tool subset; reuses existing injection machinery. |
| Custom commands / reusable prompts | Yes — markdown+frontmatter, `$ARGUMENTS`, `!`shell``, `@file` | **Missing** (roadmap: "Conversation presets") | Lightweight prompt-template/preset system is a good mobile UX win. |
| MCP client | Yes — local + remote servers | **Missing** — bespoke bridge protocol instead | Standard tool ecosystem, but bloats context and is real work; medium value for this product. |
| LSP / codebase awareness | Yes | **Missing / N/A** | Does not transfer — no code-editing IDE surface. |
| Git snapshot + revert per step | Yes | **Missing / N/A** | Does not transfer — workspace holds converted files, not a source tree. |
| Multi-client (TUI/desktop/IDE/web) over one server | Yes (SSE bus) | **Partial** — one server, one PWA client; already streams | wyltek-gpt is single-user by design; the extra clients are off-mission. |
| Config format | JSON/JSONC, layered merge, `{env:}`/`{file:}` substitution | `config.yaml` (converters/operations/bridges/prompt) | Comparable for its scope; would need new sections for permissions/modes/commands. |

---

## 3. Ranked gap / opportunity list (by value-to-effort for a local-model chat PWA)

Effort key: **S** = hours, **M** = a day or few, **L** = multi-day / structural.

### 1. Conversation compaction / auto-summarization — **Effort M** ⭐ context-window
- **OpenCode:** when token usage exceeds ~90% of the context limit, a compaction/summary system agent produces a "detailed but concise" summary ("what we did, what files we're on, what's next") and the old turns are folded into it, so long sessions survive a fixed window.
- **wyltek-gpt lacks:** any history trimming. Chat history is replayed verbatim each turn (`app.py` `chat()` rebuilds `messages` from the incoming list). On bigger local models this is *exactly* the reported pain point — the window fills with stale turns and the model truncates mid-stream (the code even bumps `_CHAT_DEFAULT_NUM_CTX` to 24576 to paper over it).
- **Why it matters here:** local models have *smaller* windows than the cloud models OpenCode targets, so compaction is more urgent, not less. This is the top item.
- **Maps onto existing machinery:** add a `summarize()` helper that calls the same Ollama endpoint (ideally a configurable `small_model`, mirroring the auto_router/captioner pattern already in `config.yaml`) to collapse the oldest N turns into a single synthetic `system`/`assistant` summary message when the running token estimate crosses a threshold. The progressive-disclosure code already *removes* content in one direction (only inject reference on intent); this is the symmetric move on history. Best done after item 2 so the summary can be persisted.

### 2. Server-side session persistence (SQLite) — **Effort M** ⭐ context-window enabler
- **OpenCode:** every session's message parts persist to a local store (SQLite/disk); sessions are listable, resumable, and multiple can run concurrently.
- **wyltek-gpt lacks:** server-side history entirely. `session_id` only names a workspace *file* directory (`_resolve_in_workspace`); the transcript lives in the browser. No resume, no cross-device continuity (a real loss for a PWA installed on phone *and* opened on desktop), and — critically — nowhere for a compaction summary (item 1) to live authoritatively.
- **Why it matters here:** it's already on the roadmap ("Persistent chat history (SQLite)"), and it's the enabler that makes compaction, titles, and cross-device use real rather than client-only hacks.
- **Maps onto existing machinery:** a `sessions` table keyed by the `session_id` that already threads through every endpoint; store messages + a `summary` column. Fits the existing single-uvicorn-worker assumption (same as `CapabilityCache`'s "cross-process writes aren't a concern" note).

### 3. Token-budget meter + compaction trigger — **Effort S** ⭐ context-window
- **OpenCode:** computes usage/cost at every `finish-step` and surfaces it.
- **wyltek-gpt lacks:** a budget view. But it *already receives* Ollama's `prompt_eval_count`/`eval_count` in the `stats` frame (`_stream_one`) and only renders them as a perf bar.
- **Why it matters here:** it's the cheap mechanism that *drives* item 1 — you can't auto-compact well without knowing how full the window is. Also a genuinely useful UI signal on a phone ("context 78% full").
- **Maps onto existing machinery:** the `__stats__` sentinel already flows to the frontend; add a `% of num_ctx` computation (prompt_eval_count / num_ctx) and use the same number server-side as the compaction trigger. Smallest-effort, high-leverage item in the list.

### 4. Declarative per-tool permission model — **Effort S–M**
- **OpenCode:** `permission: { edit: "ask", bash: "ask" }` with `allow`/`ask`/`deny` and bash-command glob patterns, overridable per agent.
- **wyltek-gpt lacks:** a declarative layer — the confirm behavior is hardcoded (file-producing ops go through the Y/N/E card via `/api/operations/run`; `cellc_check`/`preview_demo`/etc. auto-run inside `relay()`). That's fine today but ad-hoc.
- **Why it matters here:** the roadmap adds a "Restricted shell tool" and `filesystem.read`/`filesystem.write`. The moment tools can mutate the host beyond the sandboxed workspace, "which tools auto-run vs. confirm vs. are denied" wants to be data, not code.
- **Maps onto existing machinery:** add a `permission: auto | confirm | deny` field to `Operation`/the YAML op spec and to the bridge tool descriptors; `relay()` consults it instead of the current name-based partition deciding who auto-runs. Generalizes the existing warn-and-allow capability gate philosophy.

### 5. Named modes / lightweight system-agents — **Effort M**
- **OpenCode:** Build/Plan primary agents + invisible Title/Summary/Compaction system agents, each a named config bundle (prompt + model + tool subset + temperature + step cap).
- **wyltek-gpt lacks:** first-class modes. It has *de facto* modes today — intent regexes flip on cellc/html-demo guidance, and `translate`/auto-caption are hand-wired specialized routes with their own prompts and model overrides — but they're scattered and implicit.
- **Why it matters here:** (a) a user-selectable mode picker (e.g. "HTML demo", "Translate", "Vision caption", "Plain chat") is clearer on mobile than hoping a regex fires; (b) the *system-agent* half (a cheap `small_model` Title agent to auto-name chats, and the Summary agent for item 1) is directly reusable.
- **Maps onto existing machinery:** a `modes:` section in `config.yaml` (name → system-prompt fragment + optional model override + enabled op/bridge subset), consumed by `_full_system_prompt()` and the tool-schema assembly, which already conditionally include cellc/demo material.

### 6. Custom commands / reusable prompt presets — **Effort S–M**
- **OpenCode:** markdown files with YAML frontmatter become `/commands`; templates interpolate `$ARGUMENTS`/`$1`, `!`shell`` output, and `@file` contents; optional `subtask` isolation.
- **wyltek-gpt lacks:** any saved-prompt mechanism (roadmap: "Conversation presets — saved recipes with pinned flags").
- **Why it matters here:** on a phone, retyping a structured prompt is painful; pinned presets ("summarize this file", "make a bouncing-ball demo", "transcribe + translate") are a real UX win and dovetail with modes (item 5) and the existing file-injection path (`_inject_text_files`).
- **Maps onto existing machinery:** a `commands/` dir of markdown or a `commands:` config block; the `@file` idea is already implemented as `text_files`/`image_files` injection — a preset just pre-fills the prompt and target files.

### 7. Sub-agent / task delegation for context isolation — **Effort L**
- **OpenCode:** the Task tool spawns a subagent in a *fresh isolated context*, runs it to completion, and returns only its final text to the parent — a deliberate context-saver.
- **wyltek-gpt lacks:** this generality (the `auto_router` captioner is a primitive, single-purpose instance of the idea: route an image to a dedicated model outside the main chat).
- **Why it matters here:** for a small local window it's a major lever — "caption these 5 images" or "digest this 40-page PDF" could run in a throwaway context and hand back a paragraph, instead of dumping everything into the main transcript. Conceptually second only to compaction as a context strategy.
- **Maps onto existing machinery:** would need the session/loop refactor from items 1–2 first (a subtask is essentially a nested `relay()` with its own message list and its result summarized back). Ranked lower purely on effort (L), not on value.

### 8. Provider abstraction beyond Ollama (OpenAI-compatible shim) — **Effort M**
- **OpenCode:** one code path over 75+ providers incl. OpenAI-compatible local servers.
- **wyltek-gpt lacks:** any non-Ollama backend; `OLLAMA_URL` and Ollama's `/api/chat`, `/api/tags`, `/api/show` are wired throughout.
- **Why it matters here:** *cloud BYOK is off-mission* (the product's whole point is local inference), but an **OpenAI-compatible chat shim** would unlock llama.cpp-server, LM Studio, vLLM, and remote tailnet endpoints without abandoning the local-first stance.
- **Maps onto existing machinery:** a thin adapter behind the current `_stream_one`; the wrinkle is capability probing — `/api/show` is Ollama-specific, so a non-Ollama backend needs a fallback (config-declared capabilities, or a one-shot inference probe like the old path). Medium value, medium effort; keep it optional.

### 9. MCP client support — **Effort L, medium value**
- **OpenCode:** local + remote MCP servers; their tools auto-register alongside built-ins.
- **wyltek-gpt lacks:** MCP; it uses a bespoke bridge interface (`available()/tool_schemas()/dispatch()`).
- **Why it matters here (with caveats):** MCP would let wyltek-gpt consume a whole ecosystem of tools by standard. But OpenCode itself warns "MCP servers add to your context, so be careful which you enable" — and small local models degrade badly with large tool lists. For a small-window local product this is a double-edged sword.
- **Maps onto existing machinery:** an MCP-client "bridge" that adapts MCP tool descriptors into the existing `tool_schemas()`/`dispatch()` shape — the bridge abstraction is already the right seam. Gate it hard behind item 4 (permissions) and item 5 (per-mode tool subsets) so it can't flood a 24k window. Lowest value-to-effort of the transferable set.

---

## 4. Context-window management — the focused view

This is wyltek-gpt's live pain point, so pulling the thread together:

**What wyltek-gpt already does well (keep these — OpenCode validates them):**
- **Progressive disclosure.** Reference material (cellc language reference, html-demo house-style guidance) is injected into the system prompt *only when an intent regex fires* (`_inject_cellc_context`, `_inject_html_demo_context`), never on every turn. This is a stronger-than-OpenCode stance for small windows.
- **Handle/pointer for big artifacts.** The html-demo `DemoStore` keeps full HTML server-side under a `demo_id` and iterates via unique find/replace patches — the full document crosses context at most once. This is the same instinct as OpenCode's file-edit tool and is worth generalizing to any large model-produced artifact.
- **Per-attachment inject caps.** `_MAX_TEXT_INJECT_CHARS = 60_000` with a visible truncation marker stops one file from eating the budget.
- **Lean tool results.** The plan explicitly keeps tool results to "actionable diagnostics, never the HTML or a DOM dump" — the same discipline OpenCode relies on to keep loops affordable.

**What's missing (in priority order, = items 1/3/2/7 above):**
1. **History compaction/summarization** — the biggest hole. Nothing trims prior turns; the window just fills. (Item 1.)
2. **A budget signal to trigger it** — `prompt_eval_count` is already in hand but unused as a control input. (Item 3.)
3. **A durable place to store the summary** — needs server-side sessions. (Item 2.)
4. **Context isolation via subtasks** — offload bulky one-shot work (captioning, PDF digest) to a throwaway context. (Item 7.)

**Beyond OpenCode parity (worth considering given the environment):** OpenCode leans on *summarization + on-demand file reads*, not retrieval. wyltek-gpt's host already runs a RAG API (per the environment notes). A future option to **retrieve only relevant chunks** of a large attachment (instead of injecting whole files up to 60k chars) would beat both whole-file injection and summarization for large reference documents — but that's a wyltek-gpt-specific extension, not an OpenCode port.

---

## 5. Explicitly does NOT transfer

| OpenCode feature | Why it doesn't transfer to wyltek-gpt |
|---|---|
| **LSP integration / codebase awareness** | No code-editing IDE surface; wyltek-gpt operates on uploaded files, not a source tree. |
| **Git snapshot + per-step revert** | The workspace holds converted/generated files, not a versioned repo; snapshotting a chat's file folder adds little over the existing unique-output-path naming. |
| **VS Code extension / desktop app / TUI clients** | wyltek-gpt is a PWA by deliberate design; extra client surfaces are off-mission. |
| **Build/Plan *coding* agents (edit + unsupervised bash on a repo)** | wyltek-gpt is not a repository agent. The *permission concept* transfers (item 4); the repo-editing agents themselves do not. |
| **75+ cloud providers / bring-your-own-key billing** | The product's raison d'être is local inference. (Only the OpenAI-*compatible-local* slice transfers — item 8.) |
| **OpenCode Zen / Go curated model marketplace** | Cloud model brokerage; irrelevant to a local-Ollama product. |
| **Multi-concurrent-client SSE fan-out** | Single-user app; it already streams to its one PWA client. Session persistence (item 2) is worth having for *resume/cross-device*, not for concurrent viewers. |
| **`@`-mention live subagent switching in a TUI** | Terminal-interaction idiom; a mobile mode-picker (item 5) is the right analog, not `@`-mentions. |

---

## Sources
- [OpenCode — official site](https://opencode.ai/) and [docs](https://opencode.ai/docs/): [Agents](https://opencode.ai/docs/agents/), [Config](https://opencode.ai/docs/config/), [Commands](https://opencode.ai/docs/commands/), [MCP servers](https://opencode.ai/docs/mcp-servers/), [CLI](https://opencode.ai/docs/cli/)
- [How Coding Agents Actually Work: Inside OpenCode — Moncef Abboud (2026)](https://cefboud.com/posts/coding-agents-internals-opencode-deepdive/) — agent loop, step control, git snapshots, ~90% auto-summarization, provider abstraction
- [sst/opencode on GitHub](https://github.com/sst/opencode) and [DeepWiki: sst/opencode](https://deepwiki.com/sst/opencode)
- [opencode-docs mirror (v1.2.27)](https://github.com/mudrii/opencode-docs)
- wyltek-gpt source (read directly): `backend/app.py`, `backend/operations.py`, `backend/capabilities.py`, `backend/probe.py`, `README.md`, and `docs/superpowers/plans/2026-07-15-html-demo-skill.md`
