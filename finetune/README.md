# Fine-tune tooling

Phase 0 bake-off for the HTML-artifact model fine-tune.
Spec: `docs/superpowers/specs/2026-07-21-html-artifact-model-finetune-design.md`

## Two venvs

- `dump_schemas.py` runs under the **app** venv (`.venv/bin/python`, 3.10)
- Everything else runs under the **unsloth** venv
  (`~/.unsloth/studio/unsloth_studio/bin/python`, 3.13)

They never import each other. `bakeoff/schemas/html_demo_tools.json` is the bridge.

## Running the bake-off

```bash
# 1. Regenerate the schema bridge (only when html_demo tools change)
.venv/bin/python -m finetune.bakeoff.dump_schemas

# 2. Probes (GPU must be otherwise idle)
UV=~/.unsloth/studio/unsloth_studio/bin/python
$UV -m finetune.bakeoff.vram_probe      --model unsloth/gemma-4-12b-it
$UV -m finetune.bakeoff.toolcall_probe  --model unsloth/gemma-4-12b-it
$UV -m finetune.bakeoff.reasoning_probe --model unsloth/gemma-4-12b-it

# 3. Report
$UV -m finetune.bakeoff.report
```

Results land in `bakeoff/results/` (gitignored).

Probes must be run as **modules** (`python -m finetune.bakeoff.<name>`) from the
repo root. Script form (`python finetune/bakeoff/vram_probe.py`) puts
`finetune/bakeoff/` on `sys.path` instead of the repo root, so the absolute
`from finetune.bakeoff...` imports fail with `ModuleNotFoundError`.

## Pre-flight

- **The GPU must be otherwise idle.** ~3 GB of the card's 24 GB is held by the
  desktop compositor, so budget **~21 GB usable**. Stop Ollama/ComfyUI first; a
  probe that OOMs on someone else's memory is a wasted measurement, not a result.
- **Verify the tool schemas actually reach the prompt before trusting 0C.** Some
  tokenizer chat templates SILENTLY IGNORE the `tools` kwarg when their Jinja
  never references it. When that happens `native_rate` reads ~0 for reasons that
  have nothing to do with model ability. For each candidate, render one
  `apply_chat_template(..., tools=...)` output and confirm the tool schemas
  appear in the prompt text.
- **Run 0A first.** 0B and 0C default to 2048 and 4096 and do not know what 0A
  measured. They now warn on stderr when a measured ceiling is lower than the
  length they are about to use, but they will not override an explicit
  `--max-seqlen`.

## Recommended order: cheap probes first

Do NOT run all three probes on all candidates up front. 0B costs roughly 4x what
0A and 0C cost together, because it generates 200 times per model (100 GSM8K rows,
scored before AND after the smoke LoRA). Run the cheap probes across every
candidate first and let them eliminate models before paying for 0B.

**Stage 1 — 0A + 0C on every candidate (~20 min each).**

```bash
UV=~/.unsloth/studio/unsloth_studio/bin/python
for M in unsloth/gemma-4-12b-it unsloth/gpt-oss-20b-unsloth-bnb-4bit; do
  $UV -m finetune.bakeoff.vram_probe     --model "$M"
  $UV -m finetune.bakeoff.toolcall_probe --model "$M"
done
$UV -m finetune.bakeoff.report
```

**Pass bar for 0A (measured 2026-07-22, see spec finding 2):** 16 frontier
one-shot HTML games run 5,331–15,273 tokens, median 7,909, p90 11,865. So:

| measured `max_seqlen` | verdict |
|---|---|
| < 8,192 | **disqualified** — cannot hold a median artifact |
| 8,192–12,287 | marginal — covers ~56%, only if nothing better clears |
| >= 12,288 | viable — covers p90 |

Eliminate here on that bar, or on a poor `native_rate`. Either is disqualifying on
its own and costs ~20 minutes to learn instead of ~105.

**Stage 2 — 0B only on survivors.**

```bash
$UV -m finetune.bakeoff.reasoning_probe --model <surviving-model>
```

### Rough cost per candidate

Estimates, not measurements — generation throughput on this card has not been
benchmarked for these models. Treat as order-of-magnitude.

| | gemma-12b | gpt-oss-20b |
|---|---|---|
| 0A vram | ~10 min | ~12 min |
| 0C toolcall | ~6 min | ~7 min |
| 0B reasoning | ~35 min | ~40 min |
| **compute total** | **~50 min** | **~60 min** |

0A spends most of its time reloading the model at each binary-search step (~6
full loads). That is deliberate: reloading is the only reliable way to guarantee
a clean allocator state between OOM attempts, and a leaked allocation would
silently lower the measured ceiling.

### Disk, before you start

`load_in_4bit=True` against a plain HF repo downloads the **full bf16 weights**
and quantizes at load. Only gpt-oss-20b has a pre-quantized `bnb-4bit` repo.

| Model | Download |
|---|---|
| `unsloth/gemma-4-12b-it` | ~24 GB (bf16) |
| `unsloth/gpt-oss-20b-unsloth-bnb-4bit` | ~12 GB (4-bit) |

~36 GB total against ~50 GB free on that drive as of 2026-07-21 — fits, with room
to spare. Check before starting anyway.

**Qwen3.6-27B was dropped** (2026-07-22) and is not in the run. Its estimated ~4k
ceiling fits 0 of 16 measured artifacts, and its ~54 GB bf16 download exceeded free
space. It was eliminated by measurement rather than by probing — see spec finding 2.

### Shrinking the run

`--n` (default 100 GSM8K rows) dominates 0B. `--n 50` roughly halves it. The cost
is precision: at n=50 each accuracy carries ~+/-7pp binomial noise, and `retention`
is a ratio of two such estimates — so small differences between candidates stop
meaning anything. Prefer eliminating candidates in Stage 1 over lowering `--n`.

## Interpreting the table

- **Rank on `retention`, never on raw accuracy.** The GSM8K "last number wins"
  heuristic penalises verbose models. That bias cancels within a single model's
  own before/after comparison, but it does NOT cancel across models, so
  `accuracy_before` is not comparable between candidates.
- **Floor on that rule: ignore `retention` when `accuracy_before` is below
  ~0.10.** `retention(0, 0)` returns 1.0 by design, so a model that scores 0.0
  both times posts a perfect-looking retention while having demonstrated
  nothing. Below that floor the ratio is noise divided by noise.
- **`training_effective: False` voids that row's retention.** It means the smoke
  LoRA changed nothing, so no degradation was measured — a retention of 1.0
  there means "not measured", not "held its shape".
- **Compare `loss_delta` across candidates before comparing retention.**
  `training_effective` is only a boolean floor (loss went down). Two rows can
  both be True with wildly different deltas, and a 27B absorbs far less change
  than a 12B under identical LoRA settings — so retention is confounded with
  model size unless the candidates took comparable training pressure.
- **A jump from `unparseable_before` to `unparseable_after` is the signature of
  smoke-corpus bleed** poisoning the GSM8K scorer, not a reasoning regression.
  Inspect raw generations before believing `accuracy_after`.
