# HTML-Demo Skill — Design Spec

**Date:** 2026-07-15
**Project:** wyltek-gpt (`~/local-chatbot`)
**Status:** Approved for planning

## Purpose

Give every model in the local Ollama fleet a reliable, house-styled ability to
build **self-contained HTML demos** — single-file pages with good physics,
layout, touch handling, and design — and to *test* them in a real headless
browser and iterate until they work. The finished demo is saved to disk and
previewable full-screen inside the PWA on a phone (via Tailscale).

The skill is delivered as a **full agentic build-test-fix loop**, not just a
prompt tweak: the model writes HTML, renders it headless, reads back
diagnostics (and a screenshot when it can see), edits, and repeats.

**Primary constraint: context efficiency.** Generating full HTML with the bigger
local models routinely hits context limits today. The design treats keeping the
loop context-lean as a first-class goal, not an afterthought — see
[Context strategy](#context-strategy-the-lean-context-aid).

## Goals

- A tool-capable model can write → preview → edit → save a self-contained demo
  without leaving the chat, **without blowing the context window**.
- Non-tool models still benefit via the existing `op:<name> {...}` prompt-block
  path and the injected house-style guidance.
- Vision-capable chat models additionally receive a screenshot for design/visual
  critique; non-vision models get text diagnostics only.
- Finished demos persist to `demos/` and render full-screen in the PWA.
- The skill degrades gracefully when Playwright/Chromium is absent.

## Non-Goals (YAGNI)

- No dedicated second "critic" model. The screenshot goes to the chat model
  **only if that model is vision-capable**. No separate critic orchestration.
- No gallery view. Save + inline full-screen preview only; no thumbnail grid,
  no demo-library management screen.
- No fine-tuning / no weight changes. This is prompt + tool orchestration.
- No multi-file projects, bundlers, or external asset pipelines. Self-contained
  single-file HTML only.

## Architecture

The skill is a **new bridge cut from the proven `cellc` mold** (`backend/bridges/cellc.py`).
It lives at `backend/skills/html_demo/` (populating the currently-empty
`backend/skills/` package) and exposes the same interface contract the chat
loop already knows how to consume:

| Function | Role | Mirrors in `cellc` |
|---|---|---|
| `available() -> bool` | Gate registration on Playwright+Chromium presence | `cellc.available()` |
| `tool_schemas() -> list[dict]` | Expose the tools to Ollama | `cellc.tool_schemas()` |
| `dispatch(name, args) -> dict` | Execute a tool call | `cellc.dispatch()` |
| `guidance() -> str` | House-style knowledge, loaded on demand | `cellc.language_reference()` |
| `examples() -> list[dict]` | Gold-standard few-shot demos | `cellc.list_examples()` |

**Wiring** in `/api/chat` (`backend/app.py`), immediately after the existing
cellc block:

```python
tools = OPERATIONS.tool_schemas() if OPERATIONS.enabled else []
if cellc_bridge.available():
    tools = tools + cellc_bridge.tool_schemas()
if html_demo.available():                       # NEW
    tools = tools + html_demo.tool_schemas()     # NEW
```

Tool-call results are folded back into the loop through the same partition/
summarize path the cellc calls use (`_partition_cellc_calls` /
`_summarize_cellc_step` gain an html_demo-aware sibling, or are generalized).

### Server-side HTML handle store

The current demo HTML lives **server-side**, keyed by a short `demo_id`, in an
in-process store (`store.py`). The full document crosses the model's context
**exactly once** — on the initial draft. Every subsequent iteration references
`demo_id` and sends only small edits. This is the core of the context strategy.

## Context strategy (the lean context aid)

Ranked by impact, largest first:

1. **HTML lives under a handle, not in context.** After the first `preview_demo`,
   the model works against `demo_id`; the full file is never re-sent. This is the
   single biggest saving — a demo rewritten 3× would otherwise cost ~4× its tokens.
2. **Patch-based edits with rewrite fallback.** The model fixes via
   `patch_demo(demo_id, [{find, replace}])` — only the changed spans cross
   context. If a `find` string doesn't match uniquely, the tool returns a clear
   failure telling the model to re-send the full HTML via `preview_demo`
   (`demo_id` reused) — so a botched edit never dead-ends the loop.
3. **Progressive-disclosure guidance.** The base system prompt carries only a
   short stub. The full `guidance()` + `examples()` inject **on first engagement**
   (first `preview_demo` call), mirroring `cellc.language_reference()`. Chats that
   never build a demo pay zero guidance tokens.
4. **Lean tool results.** Diagnostics return only what's actionable (errors,
   `raf_ran`, sanity flags) — never a dump of the DOM or the HTML. Screenshots
   are downscaled and attached **only** to vision-capable models.
5. **Iteration cap (~3–4).** Bounds worst-case context growth and forces the
   model to converge or hand back.
6. **`num_ctx` headroom.** The loop already runs with a raised `num_ctx`
   (the app bumps Ollama's 4096 default); the above keeps usage well under it.

## Tools

### 1. `preview_demo(html, name?, settle_ms?, interactions?)`

Create-or-replace + **test**. Stores `html` under a `demo_id` (new, or replacing
the current one for a reused `demo_id`) and renders it headless (Playwright).

**Parameters:**
- `html` (string, required) — the full self-contained HTML document.
- `name` (string, optional) — human label; used later for the filename.
- `settle_ms` (int, optional, default 700) — wait after load before capture, so
  `requestAnimationFrame`-driven physics/animation has unfolded.
- `interactions` (array, optional) — synthetic pointer interactions to exercise
  touch/gesture handling before capture (e.g. `[{type:"pointerdown",x,y}, ...]`).

**Returns:** `demo_id` + diagnostics (below). Screenshot attached only if the
chat model is vision-capable.

### 2. `patch_demo(demo_id, edits)`

The **lean edit** step. Applies find/replace edits to the stored HTML, then
re-renders — one round-trip, small payload.

**Parameters:**
- `demo_id` (string, required) — handle from `preview_demo`.
- `edits` (array, required) — `[{find, replace}]`; each `find` must match the
  stored HTML **uniquely**.

**Returns:**
- On success: updated diagnostics (+ screenshot if vision), same shape as `preview_demo`.
- On failure (a `find` missing or ambiguous): `{applied:false, reason, hint}`
  instructing the model to re-send the full HTML via `preview_demo` with the same
  `demo_id` — the **rewrite fallback**.

### Diagnostics (returned by `preview_demo` / `patch_demo`)

- `loaded` — did the document parse and load.
- `console_errors` — JS console error/warning messages.
- `exceptions` — uncaught page exceptions.
- `render_sanity` — heuristic "is anything actually drawn" (non-blank canvas /
  non-empty body / element count).
- `raf_ran` — whether `requestAnimationFrame` fired at least once (physics proof).
- `viewport` — captured dimensions.
- `screenshot` — downscaled PNG, **only if the chat model is vision-capable**.

### 3. `save_demo(demo_id, name?)`

The **finish** step. Writes the stored demo and surfaces it to the human.

**Parameters:**
- `demo_id` (string, required) — handle to persist.
- `name` (string, optional) — overrides the label; slugified to a filename.

**Returns:** `path` (`demos/<slug>.html`) + `preview_url`; emits a PWA preview card.

## Knowledge — the house-style guidance

Concise, high-signal. Delivered by `guidance()`. Covers:

- **Self-contained rules:** one file; no CDNs / external deps; works offline;
  inline all CSS and JS; embed assets as data URIs.
- **Physics:** `requestAnimationFrame` loop with delta-time; euler/verlet
  integration; basic collision detection/resolution; easing functions.
- **Layout:** viewport meta; `100dvh` over `100vh`; flexbox/grid; safe-area
  insets (`env(safe-area-inset-*)`); responsive units.
- **Touch:** pointer events (not mouse-only); `touch-action`; `preventDefault`
  on scroll-hijack; basic gesture patterns.
- **Design:** spacing/type scale; dark-mode aware (`prefers-color-scheme`);
  motion tokens. Cross-references the existing `ui-design` skill's principles.

Kept deliberately lean (it shares the loop's context budget). Plus **1–2
gold-standard example demos** supplied by `examples()` as few-shot anchors.

## Data flow / loop

1. User asks for a demo.
2. Model sees the short stub + the tools.
3. Model writes HTML, calls `preview_demo` → backend stores under `demo_id`,
   renders, returns diagnostics (+ screenshot if vision). `guidance()` +
   `examples()` inject on this first engagement.
4. Model reads diagnostics, calls `patch_demo(demo_id, edits)` to fix →
   re-render → diagnostics. On patch failure, falls back to full `preview_demo`
   with the same `demo_id`. **Capped at ~3–4 iterations.**
5. When satisfied, model calls `save_demo(demo_id)`.
6. Frontend shows a preview card.
7. User taps → full-screen `<iframe>` renders the saved file on the phone.

## Frontend

- **Preview card** component in the chat stream (reuses existing card patterns:
  cellc step summaries, the enhance Y/N/E confirm card).
- Tap → **full-screen `<iframe>`** loading the saved demo URL.
- **Static route** serving `demos/` (e.g. `GET /demos/<name>.html`).
- Minimal new surface; no gallery.

## Module layout

```
backend/skills/
  __init__.py
  html_demo/
    __init__.py       # available(), tool_schemas(), dispatch(), guidance(), examples()
    render.py         # Playwright headless render + diagnostics + screenshot
    store.py          # demo_id handle store (in-process) + patch apply + write to demos/
    guidance.md       # the house-style text (loaded by guidance())
    examples/         # 1-2 gold-standard .html anchors
frontend/
  app.js              # preview-card handling + full-screen iframe view
  index.html/style.css# card + iframe styles
backend/app.py        # wire html_demo into /api/chat tool assembly + tool-result loop
demos/                # gitignored output dir (created on first save)
install.sh            # add Playwright/Chromium audit
```

## Risks & guardrails

| Risk | Severity | Mitigation |
|---|---|---|
| **Context budget** — HTML re-sends overflow `num_ctx` (the live pain point) | High | Server-side handle store (HTML crosses context once); patch-based edits; progressive-disclosure guidance; lean diagnostics; iteration cap. See [Context strategy](#context-strategy-the-lean-context-aid) |
| **Small-model patch fumbles** — bad `find` string | Medium | Unique-match requirement + clear failure that triggers the full-rewrite fallback; loop never dead-ends |
| **Time-based capture** — a still misses animation/touch | Medium | Default `settle_ms` ~700; optional synthetic `interactions`; report `raf_ran` |
| **Headless ≠ real phone** — touch/viewport differ | Medium | Diagnostics note the caveat; real proof is tap-to-preview on device |
| **Playwright/Chromium absent** | Medium | `available()` returns false → skill silently doesn't register; `install.sh` audit flags it; no crash |
| **Non-tool models** can't call tools | Low | Existing `op:<name> {...}` prompt-block path already handles this class |
| **Stale/leaking handle store** | Low | Bound store size / TTL per chat; handles are cheap strings, HTML dropped after save or expiry |
| **Untrusted HTML rendered headless** | Low | Ephemeral Playwright context, no host FS/network creds exposed; saved files served static read-only |

## Success criteria

- A native-tool model (e.g. `devstral-small-2:24b`) completes write → preview →
  patch → save for a moving demo, and the saved file renders full-screen in the PWA.
- The full HTML crosses the model's context **once**; subsequent iterations are
  patch-sized — measurably lower context use than a rewrite-each-turn loop.
- A non-vision tool model gets useful text diagnostics and converges without a
  screenshot.
- Skill registers only when Playwright/Chromium is present; app starts cleanly
  when it isn't.
- Base chat `num_ctx` usage is unchanged when no demo is being built (guidance is
  not always-injected).

## Open items for the plan

- Exact generalization of `_partition_cellc_calls`/`_summarize_cellc_step` vs a
  parallel html_demo path.
- Handle-store lifetime: per-chat vs global, TTL/size bound, cleanup trigger.
- Patch matching rules: exact unique substring only, or allow anchored/whitespace-
  tolerant matching; how ambiguity is reported.
- Screenshot downscale dimensions and PNG budget.
- Slug collision handling in `store.py`.
