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
