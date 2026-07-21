# HTML-Artifact Model Fine-Tune — Design Spec

## Purpose

Produce a locally-servable model, fine-tuned on this machine (Radeon RX 7900 XTX,
24 GB, gfx1100, ROCm 7.2.2), that is materially better than stock at generating
one-shot self-contained HTML artifacts **and** at driving the existing
`html_demo` tool loop inside wyltek-gpt.

The model is not an end in itself. It ships into wyltek-gpt as a selectable
Ollama model. Tool-calling competence is therefore a hard requirement, not a
nice-to-have: a model that writes beautiful HTML but cannot emit valid
`tool_calls` is useless in this application.

## Goals

1. Improve one-shot HTML artifact quality along four axes, all four in scope:
   motion/interaction, generative canvas/WebGL visuals, layout/typography, and
   rich data visualisation.
2. Emit **native** OpenAI-format `tool_calls` for `preview_demo` / `patch_demo` /
   `save_demo` reliably enough to stop relying on the `op:<name>` client-side
   fallback (`backend/app.py:477`).
3. Recover from `preview_demo` diagnostics — read console errors / missing rAF /
   blank canvas, and patch rather than restart.
4. Preserve general reasoning. The tuned model must not become a stylist that
   has forgotten how to plan.

## Non-Goals (YAGNI)

- **Not teaching facts.** Factual recall stays with the RAG service on
  wyltek-10700. Fine-tuning teaches form and behaviour, not knowledge.
- **Not baking in tool schemas.** Tool *names and parameters* are supplied at
  runtime by `tool_schemas()`. We train the *protocol* (well-formed calls, when
  to call, how to read results), not the catalogue — otherwise every schema edit
  invalidates the adapter.
- **Not pretraining, not full fine-tuning.** QLoRA adapters only.
- **Not multi-file projects.** Single self-contained document, matching the
  existing `preview_demo` contract.
- **Not a general coding model.** Narrow beats broad at this scale.

## Key findings from discovery

These reshaped the design and are recorded because they are non-obvious.

### 1. The vault is not a corpus

Surveyed `~/Documents/loacal-vault`:

| Source | Raw count | Actually usable |
|---|---|---|
| `.html` files | 4 | ~0 — three are graphify's generated `graph.html` (identical 241-line template); one is scraped phishing evidence |
| `.md` with ` ```html ` fences | 9 | ~5 unique — FiberQuest/Wyltek design docs duplicated across `Archive/` and `Projects/`; snippets, not whole artifacts |
| notes mentioning canvas/webgl/shader | 49 | 3 contain real code |

Usable starting material is **5–8 partial examples** against a need of roughly
500–1000. `backend/skills/html_demo/examples/` holds exactly one file
(`bouncing-ball.html`, 1.1 KB).

**The corpus is the project.** Training is the cheap step at the end.

### 2. VRAM forces a model-size decision

Estimated QLoRA budget, batch size 1, gradient checkpointing:

| | 27B | 14B / 12B |
|---|---|---|
| 4-bit weights | ~15 GB | ~7.5 GB |
| LoRA + Adam states | ~1.2 GB | ~0.7 GB |
| Activations @ 4k | ~4 GB | ~2 GB |
| **Total @ 4k** | ~20 GB (tight) | ~10 GB |
| **Total @ 8k** | ~26 GB → OOM | ~16 GB (fits) |

**These are estimates, not measurements.** Phase 0 replaces them with facts.

The tension: a rich artifact is 3–8k tokens, so 27B risks training on truncated
documents — teaching the model to stop mid-`<div>`. Truncation is a *correctness*
failure, whereas insufficient capacity is a *quality* failure that degrades
gracefully. That asymmetry argues for context length over parameter count, but
the bake-off decides.

Partially mitigated by finding 3: `patch_demo` turns carry diffs, not whole
documents, so multi-turn trajectories cost less than turn count implies.

### 3. The tool loop already exists — and unifies the two goals

`backend/skills/html_demo/__init__.py` already implements:

- `preview_demo(html, name?, demo_id?, settle_ms?, interactions?)` → renders
  headless, returns diagnostics (load ok, JS console errors, exceptions,
  DOM/canvas sanity, whether `requestAnimationFrame` ran) plus a `demo_id`
- `patch_demo(demo_id, edits)` → unique find/replace, re-render
- `save_demo(demo_id, name?)` → persist and surface a preview card

Three consequences:

- **The two objectives collapse into one.** The model never needs to one-shot
  blind; it has ground-truth feedback. Driving the loop well *is* the HTML
  capability. Style and tool-use stop competing for adapter capacity.
- **The corpus quality filter is already written and in production.** Render
  diagnostics are a mechanical accept/reject gate on generated artifacts.
- **The target format is settled** — OpenAI-standard `tool_calls`, natively
  supported by both candidate families. We are improving reliability on a
  standard protocol, not teaching an exotic one.

### 4. Candidate models — ROCm-viable subset

| Repo | Usable here | Why |
|---|---|---|
| `unsloth/gemma-4-12b-it` | ✅ | bf16 HF weights, quantize at load |
| `unsloth/Qwen3.6-27B` | ✅ | bf16 HF weights |
| `unsloth/gemma-4-31B-it-unsloth-bnb-4bit` | ✅ | pre-quantized; "too big" control |
| `*-NVFP4` | ❌ | NVIDIA Blackwell FP4, needs sm_100 |
| `*-MLX-*` | ❌ | Apple Silicon only |

`Qwen3.6-35B-A3B` is excluded: MoE keeps all experts resident (~18 GB in 4-bit),
so it loses to the same squeeze as 27B despite 3B active parameters.

## Architecture — four phases

The fine-tune is deliberately last and smallest.

| Phase | Output | Gate to proceed |
|---|---|---|
| 0. Bake-off | A defensible base-model choice | Measurements recorded |
| 1. Corpus | 500–1000 filtered trajectories | Render pass-rate acceptable |
| 2. Train | LoRA adapter | Training completes, loss sane |
| 3. Eval + serve | Verdict + GGUF in Ollama | Beats untuned baseline |

### Phase 0 — Bake-off

Three measurements per candidate (`gemma-4-12b-it`, `Qwen3.6-27B`, optionally
`gemma-4-31B`):

**A. VRAM ceiling.** Binary-search maximum sequence length to OOM at batch 1
with gradient checkpointing. Replaces the estimate table above.

**B. Reasoning retention.** Score on a held-out general reasoning set *before*
tuning; run a short throwaway LoRA on a corpus slice; score *again*. The metric
is **degradation per unit of style training**, not raw capability. A stronger
base that degrades faster can lose to a weaker one that holds its shape. This
requires a baseline captured before any training exists — the step usually
skipped, and the reason "did it help?" is normally unanswerable afterwards.

**C. Native tool-call reliability.** Given `tool_schemas()` and a set of demo
requests, what fraction produce well-formed native `tool_calls` versus falling
back to `op:<name>` blocks? This is the pass/fail metric that most directly
predicts usefulness in wyltek-gpt.

### Phase 1 — Corpus (the bulk of the work)

**Source: distillation.** Drive a strong model through the *real* tool loop under
the actual one-shot constraints, capturing whole trajectories. Chosen over
scraping because scraped artifacts (CodePen/Awwwards) are multi-file, CDN-
dependent and framework-laden — training on them teaches
`<script src="three.min.js">`, which violates the single-file constraint the
corpus is supposed to enforce. Distillation also avoids licensing ambiguity.

**Unit of training data is a trajectory, not a file:**

```
user request
  → assistant tool_call preview_demo(html)
  → tool result: diagnostics
  → assistant tool_call patch_demo(demo_id, edits)     [0..n times]
  → tool result: diagnostics
  → assistant tool_call save_demo(demo_id)
```

This single format teaches document quality, tool protocol, and error recovery
together. It also functions as **rehearsal data**: because tool calls appear
throughout, style training reinforces rather than erodes tool-calling — the
standard mitigation for catastrophic forgetting.

**Filtering** is mechanical via `preview_demo` diagnostics: reject anything with
console errors, exceptions, blank canvas, or no rAF where motion was requested.
A human keep/kill pass on aesthetics follows, since diagnostics prove *validity*,
not *quality*.

**Coverage** is stratified across the four target axes so one aesthetic does not
dominate.

**Seeds.** The `Knowledge — house-style guidance` section of the existing
`html_demo` skill design, plus the `artifact-design` / `ui-design` /
`frontend-design` skills, already encode the intended aesthetic. Use them as
generator system prompts so the corpus inherits a consistent house style rather
than generic model defaults.

### Phase 2 — Train

Conservative QLoRA. Low rank (start r=16), few epochs, early stopping, held-out
validation. The goal is a stylistic nudge, not a personality transplant —
aggressive training is precisely what destroys the reasoning measured in Phase 0B.

Target modules: attention (`q,k,v,o`) plus MLP (`gate,up,down`).

### Phase 3 — Eval and serve

Two tiers, both against the **untuned base as control**:

- **Objective**, from the existing harness: render pass-rate, console-error rate,
  rAF-fired rate, native tool-call validity rate, mean patch iterations to a
  clean render.
- **Subjective**: blind A/B of tuned vs. untuned output on held-out prompts,
  scored for aesthetics. Model-judge for volume, spot-checked by hand.

Ship condition: objective metrics improve, reasoning retention is within an
agreed tolerance, and blind A/B favours the tuned model. Export GGUF via
`unsloth export`, serve from Ollama, select in wyltek-gpt.

## Risks and guardrails

| Risk | Mitigation |
|---|---|
| Corpus never reaches viable size — the usual failure mode | Phase 1 gated on a counted, filtered corpus before any training spend. Fail fast if generation throughput is too low. |
| Catastrophic forgetting of reasoning / tool use | Trajectory corpus acts as rehearsal; Phase 0B measures degradation explicitly; conservative rank and early stopping. |
| Truncated training documents at 4k | Phase 0A measures the real ceiling; prefer whole documents; `patch_demo` diffs reduce per-turn cost. |
| Tuned model overfits to house style, becomes monotonous | Stratify across four axes; hold out prompts unlike any training prompt. |
| Distillation terms-of-service | Confirm the generating provider's terms permit training use before bulk generation. Output stays local and unpublished by default; revisit if the adapter is ever released. |
| Estimates treated as measurements | Phase 0 exists solely to convert them. No training commitment until it completes. |
| VRAM contention with other services | Training is exclusive-use on driveThree; check GPU is free before runs. |

## Success criteria

1. Native `tool_call` validity rate materially above the untuned base — enough
   that the `op:` fallback is not the normal path for this model.
2. Render pass-rate on first `preview_demo` above the untuned base.
3. Reasoning retention within agreed tolerance of the untuned base.
4. Blind A/B on aesthetics favours the tuned model.
5. Runs as a selectable model in wyltek-gpt via Ollama, on 24 GB, no regressions.

## Open items for the plan

- Exact reasoning benchmark and tolerance threshold for Phase 0B.
- Target corpus size — 500 vs 1000 — decided by observed generation throughput
  and marginal quality.
- Whether Phase 0 tests `gemma-4-31B` at all, or only after 12B vs 27B resolves.
- Whether `interactions?` (scripted interaction in `preview_demo`) should be
  exercised in trajectories, or deferred.
- Corpus storage location and format (JSONL), and whether it lives in this repo
  or beside it.
