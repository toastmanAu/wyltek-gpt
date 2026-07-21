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
.venv/bin/python finetune/bakeoff/dump_schemas.py

# 2. Probes (GPU must be otherwise idle)
UV=~/.unsloth/studio/unsloth_studio/bin/python
$UV finetune/bakeoff/vram_probe.py      --model unsloth/gemma-4-12b-it
$UV finetune/bakeoff/toolcall_probe.py  --model unsloth/gemma-4-12b-it
$UV finetune/bakeoff/reasoning_probe.py --model unsloth/gemma-4-12b-it

# 3. Report
$UV finetune/bakeoff/report.py
```

Results land in `bakeoff/results/` (gitignored).
