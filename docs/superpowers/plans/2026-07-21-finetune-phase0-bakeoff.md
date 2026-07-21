# Fine-Tune Phase 0 Bake-Off Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure `unsloth/gemma-4-12b-it` against `unsloth/Qwen3.6-27B` on this machine across VRAM ceiling, reasoning retention under tuning, and native tool-call reliability — producing a defensible base-model choice before any corpus work begins.

**Architecture:** A `finetune/bakeoff/` package inside `local-chatbot`, living beside the spec and the `html_demo` skill it measures against. All GPU/model work sits behind injectable function parameters so the decision logic (binary search, response classification, answer scoring) is unit-testable on CPU with no model download. Three independent probes write JSON result files; a final reporter aggregates them into a comparison table.

**Tech Stack:** Python 3.13 (unsloth venv), unsloth 2026.7.4, transformers 4.57.6, peft 0.18.1, trl 0.23.1, torch 2.10.0+rocm7.2.2, pytest.

## Global Constraints

- **Two venvs, do not mix.** Bake-off code runs under `~/.unsloth/studio/unsloth_studio/bin/python` (3.13). The app runs under `~/local-chatbot/.venv/bin/python` (3.10). Cross-venv imports are forbidden — the schema bridge is a generated JSON file.
- **Usable VRAM is ~21 GB, not 24 GB.** The desktop compositor holds ~3 GB (measured: `rocm-smi` reports 12% allocated at idle). All ceilings are measured against real free memory, not nameplate.
- **GPU is exclusive-use during probes.** Stop `wan-worker`/ComfyUI/Ollama large models before running. A probe that OOMs because something else held VRAM is a wasted measurement.
- **Model repos, exact strings:** `unsloth/gemma-4-12b-it`, `unsloth/Qwen3.6-27B`. Not the `-NVFP4` variants (NVIDIA Blackwell only), not the `-MLX-` variants (Apple Silicon only).
- **All probes write JSON to `finetune/bakeoff/results/`.** That directory is gitignored; results are reported, not committed.
- **No network in unit tests.** Tests must pass with the GPU busy and the network down.
- **Commit message format:** `<type>: <description>` per `~/.claude/rules/git-workflow.md`. No attribution trailer.

---

## File Structure

| Path | Responsibility |
|---|---|
| `finetune/README.md` | What this directory is, how to run the bake-off |
| `finetune/bakeoff/__init__.py` | Package marker |
| `finetune/bakeoff/dump_schemas.py` | Runs under the **app** venv; exports `html_demo` tool schemas to JSON |
| `finetune/bakeoff/schemas/html_demo_tools.json` | Generated bridge artifact (committed — it is an interface contract) |
| `finetune/bakeoff/search.py` | Pure binary-search logic for max sequence length |
| `finetune/bakeoff/vram_probe.py` | Probe 0A — drives `search.py` with a real GPU load attempt |
| `finetune/bakeoff/classify.py` | Pure classifier: native tool_call vs `op:` fallback vs none |
| `finetune/bakeoff/toolcall_probe.py` | Probe 0C — generates responses, classifies them |
| `finetune/bakeoff/score.py` | Pure GSM8K answer extraction and scoring |
| `finetune/bakeoff/reasoning_probe.py` | Probe 0B — scores reasoning before/after a smoke LoRA |
| `finetune/bakeoff/report.py` | Aggregates result JSON into a comparison table |
| `finetune/tests/test_*.py` | Unit tests for the four pure modules |

Pure logic (`search`, `classify`, `score`) is separated from GPU drivers (`*_probe`) specifically so the plan is testable. Every probe takes its expensive operation as a parameter.

---

### Task 1: Scaffold and schema bridge

**Files:**
- Create: `finetune/README.md`
- Create: `finetune/bakeoff/__init__.py`
- Create: `finetune/bakeoff/dump_schemas.py`
- Create: `finetune/.gitignore`
- Test: `finetune/tests/test_schema_bridge.py`

**Interfaces:**
- Consumes: `backend.skills.html_demo.tool_schemas()` (existing, verified to import cleanly under the app venv)
- Produces: `finetune/bakeoff/schemas/html_demo_tools.json` — a JSON array of OpenAI-format tool definitions. Task 3 and Task 5 read this file.

- [ ] **Step 1: Write the failing test**

Create `finetune/tests/test_schema_bridge.py`:

```python
import json
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent.parent / "bakeoff" / "schemas" / "html_demo_tools.json"


def test_schema_file_exists():
    assert SCHEMA_PATH.is_file(), f"missing {SCHEMA_PATH}; run dump_schemas.py"


def test_schema_has_three_html_demo_tools():
    tools = json.loads(SCHEMA_PATH.read_text())
    names = [t["function"]["name"] for t in tools]
    assert names == ["preview_demo", "patch_demo", "save_demo"]


def test_schemas_are_openai_shaped():
    tools = json.loads(SCHEMA_PATH.read_text())
    for t in tools:
        assert t["type"] == "function"
        fn = t["function"]
        assert isinstance(fn["name"], str) and fn["name"]
        assert isinstance(fn["description"], str) and fn["description"]
        assert fn["parameters"]["type"] == "object"
        assert isinstance(fn["parameters"]["required"], list)


def test_preview_demo_requires_html():
    tools = json.loads(SCHEMA_PATH.read_text())
    preview = next(t for t in tools if t["function"]["name"] == "preview_demo")
    assert "html" in preview["function"]["parameters"]["required"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/local-chatbot && ~/.unsloth/studio/unsloth_studio/bin/python -m pytest finetune/tests/test_schema_bridge.py -v
```

Expected: FAIL — `assert False, "missing .../html_demo_tools.json; run dump_schemas.py"`

- [ ] **Step 3: Write the scaffold and dumper**

Create `finetune/bakeoff/__init__.py` (empty file).

Create `finetune/.gitignore`:

```
bakeoff/results/
```

Create `finetune/bakeoff/dump_schemas.py`:

```python
"""Export html_demo tool schemas to JSON.

Runs under the APP venv (python 3.10), not the unsloth venv — it imports
backend code. The generated JSON is the only bridge between the two
environments; nothing in bakeoff/ ever imports backend directly.

Usage (from repo root):
    .venv/bin/python finetune/bakeoff/dump_schemas.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = Path(__file__).parent / "schemas" / "html_demo_tools.json"


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    from backend.skills.html_demo import tool_schemas

    schemas = tool_schemas()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(schemas, indent=2) + "\n")
    names = [t["function"]["name"] for t in schemas]
    print(f"wrote {len(schemas)} tool schemas to {OUT_PATH}: {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `finetune/README.md`:

```markdown
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
```

- [ ] **Step 4: Generate the bridge and run tests**

```bash
cd ~/local-chatbot
.venv/bin/python finetune/bakeoff/dump_schemas.py
~/.unsloth/studio/unsloth_studio/bin/python -m pytest finetune/tests/test_schema_bridge.py -v
```

Expected: dumper prints `wrote 3 tool schemas to ... ['preview_demo', 'patch_demo', 'save_demo']`, then 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add finetune/README.md finetune/.gitignore finetune/bakeoff/__init__.py \
        finetune/bakeoff/dump_schemas.py finetune/bakeoff/schemas/html_demo_tools.json \
        finetune/tests/test_schema_bridge.py
git commit -m "feat: scaffold fine-tune bakeoff with html_demo schema bridge"
```

---

### Task 2: Binary-search logic for max sequence length

**Files:**
- Create: `finetune/bakeoff/search.py`
- Test: `finetune/tests/test_search.py`

**Interfaces:**
- Produces: `find_max_seqlen(fits, lo=512, hi=16384, step=512) -> int` where `fits: Callable[[int], bool]` returns True if that sequence length trains without OOM. Returns the largest multiple of `step` that fits, or `0` if even `lo` fails. Task 3 (`vram_probe.py`) supplies the real `fits`.

- [ ] **Step 1: Write the failing test**

Create `finetune/tests/test_search.py`:

```python
from finetune.bakeoff.search import find_max_seqlen


def test_finds_exact_threshold():
    # everything up to and including 4096 fits
    calls = []

    def fits(n):
        calls.append(n)
        return n <= 4096

    assert find_max_seqlen(fits, lo=512, hi=16384, step=512) == 4096


def test_returns_zero_when_nothing_fits():
    assert find_max_seqlen(lambda n: False, lo=512, hi=8192, step=512) == 0


def test_returns_hi_when_everything_fits():
    assert find_max_seqlen(lambda n: True, lo=512, hi=8192, step=512) == 8192


def test_is_logarithmic_not_linear():
    calls = []

    def fits(n):
        calls.append(n)
        return n <= 4096

    find_max_seqlen(fits, lo=512, hi=16384, step=512)
    # 32 candidate steps -> must probe far fewer than 32 times
    assert len(calls) <= 8, f"too many probes: {calls}"


def test_result_is_multiple_of_step():
    result = find_max_seqlen(lambda n: n <= 5000, lo=512, hi=16384, step=512)
    assert result % 512 == 0
    assert result == 4608
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/local-chatbot && ~/.unsloth/studio/unsloth_studio/bin/python -m pytest finetune/tests/test_search.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'finetune.bakeoff.search'`

- [ ] **Step 3: Write minimal implementation**

Create `finetune/bakeoff/search.py`:

```python
"""Pure binary search for the largest workable sequence length.

Kept free of torch/unsloth imports so it is unit-testable on CPU with no
model present. The caller injects `fits`, which does the expensive thing.
"""
from typing import Callable


def find_max_seqlen(
    fits: Callable[[int], bool],
    lo: int = 512,
    hi: int = 16384,
    step: int = 512,
) -> int:
    """Largest multiple of `step` in [lo, hi] for which `fits` is True.

    Assumes `fits` is monotonic: if n fits, everything below n fits. Returns
    0 when even `lo` fails. Probes O(log n) times, since each probe may cost
    minutes of GPU time.
    """
    if not fits(lo):
        return 0

    best = lo
    low, high = lo, hi
    while low <= high:
        mid = ((low + high) // 2 // step) * step
        if mid < low:
            mid = low
        if fits(mid):
            best = max(best, mid)
            low = mid + step
        else:
            high = mid - step
    return best
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/local-chatbot && ~/.unsloth/studio/unsloth_studio/bin/python -m pytest finetune/tests/test_search.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add finetune/bakeoff/search.py finetune/tests/test_search.py
git commit -m "feat: add binary search for max trainable sequence length"
```

---

### Task 3: VRAM probe (Phase 0A)

**Files:**
- Create: `finetune/bakeoff/vram_probe.py`
- Test: `finetune/tests/test_vram_probe.py`

**Interfaces:**
- Consumes: `find_max_seqlen` from Task 2.
- Produces: `write_result(model: str, max_seqlen: int, free_vram_gb: float, out_dir: Path) -> Path` writing `results/vram_<slug>.json` with keys `model`, `probe`, `max_seqlen`, `free_vram_gb`. Task 6 (`report.py`) reads these. Slug rule: model string lowercased with `/` and `.` replaced by `-`.

- [ ] **Step 1: Write the failing test**

Create `finetune/tests/test_vram_probe.py`:

```python
import json
from finetune.bakeoff.vram_probe import write_result, slugify


def test_slugify_makes_safe_filename():
    assert slugify("unsloth/gemma-4-12b-it") == "unsloth-gemma-4-12b-it"
    assert slugify("unsloth/Qwen3.6-27B") == "unsloth-qwen3-6-27b"


def test_write_result_creates_readable_json(tmp_path):
    path = write_result("unsloth/gemma-4-12b-it", 8192, 21.3, tmp_path)
    assert path.is_file()
    data = json.loads(path.read_text())
    assert data["model"] == "unsloth/gemma-4-12b-it"
    assert data["probe"] == "vram"
    assert data["max_seqlen"] == 8192
    assert data["free_vram_gb"] == 21.3


def test_write_result_filename_matches_slug(tmp_path):
    path = write_result("unsloth/Qwen3.6-27B", 4096, 21.0, tmp_path)
    assert path.name == "vram_unsloth-qwen3-6-27b.json"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/local-chatbot && ~/.unsloth/studio/unsloth_studio/bin/python -m pytest finetune/tests/test_vram_probe.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'finetune.bakeoff.vram_probe'`

- [ ] **Step 3: Write implementation**

Create `finetune/bakeoff/vram_probe.py`:

```python
"""Probe 0A: largest sequence length that trains without OOM.

Loads the model in 4-bit, attaches a LoRA, and attempts one forward+backward
at a candidate sequence length. Binary-searches the ceiling.

Run with the GPU otherwise idle — a probe that OOMs because Ollama held VRAM
is a wasted measurement.
"""
import argparse
import gc
import json
import re
from pathlib import Path

from finetune.bakeoff.search import find_max_seqlen

DEFAULT_OUT = Path(__file__).parent / "results"


def slugify(model: str) -> str:
    return re.sub(r"[/.]+", "-", model).lower()


def write_result(model: str, max_seqlen: int, free_vram_gb: float, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"vram_{slugify(model)}.json"
    path.write_text(json.dumps({
        "model": model,
        "probe": "vram",
        "max_seqlen": max_seqlen,
        "free_vram_gb": free_vram_gb,
    }, indent=2) + "\n")
    return path


def free_vram_gb() -> float:
    import torch
    free, _total = torch.cuda.mem_get_info()
    return round(free / 1024**3, 2)


def make_fits(model: str, rank: int = 16):
    """Return a `fits(seqlen)` that actually attempts a training step."""
    import torch
    from unsloth import FastLanguageModel

    def fits(seqlen: int) -> bool:
        m = tok = out = ids = None
        try:
            m, tok = FastLanguageModel.from_pretrained(
                model_name=model,
                max_seq_length=seqlen,
                load_in_4bit=True,
                dtype=None,
            )
            m = FastLanguageModel.get_peft_model(
                m,
                r=rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_alpha=rank,
                use_gradient_checkpointing="unsloth",
            )
            ids = torch.randint(0, 1000, (1, seqlen), device="cuda")
            out = m(input_ids=ids, labels=ids)
            out.loss.backward()
            print(f"  seqlen={seqlen}: OK")
            return True
        except torch.OutOfMemoryError:
            print(f"  seqlen={seqlen}: OOM")
            return False
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"  seqlen={seqlen}: OOM (RuntimeError)")
            return False
        finally:
            del m, tok, out, ids
            gc.collect()
            torch.cuda.empty_cache()

    return fits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lo", type=int, default=512)
    ap.add_argument("--hi", type=int, default=16384)
    ap.add_argument("--step", type=int, default=512)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    free = free_vram_gb()
    print(f"free VRAM before probe: {free} GB")
    if free < 18:
        print("WARNING: <18 GB free. Stop Ollama/ComfyUI before probing.")

    ceiling = find_max_seqlen(make_fits(args.model), args.lo, args.hi, args.step)
    path = write_result(args.model, ceiling, free, args.out)
    print(f"max_seqlen={ceiling} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/local-chatbot && ~/.unsloth/studio/unsloth_studio/bin/python -m pytest finetune/tests/test_vram_probe.py -v
```

Expected: 3 tests PASS (no GPU or model needed — only pure helpers are tested).

- [ ] **Step 5: Commit**

```bash
git add finetune/bakeoff/vram_probe.py finetune/tests/test_vram_probe.py
git commit -m "feat: add VRAM ceiling probe for bakeoff candidates"
```

---

### Task 4: Tool-call response classifier (Phase 0C logic)

**Files:**
- Create: `finetune/bakeoff/classify.py`
- Test: `finetune/tests/test_classify.py`

**Interfaces:**
- Produces: `classify_response(text: str) -> str` returning exactly one of `"native"`, `"fallback"`, or `"none"`. `"native"` means a well-formed tool call in either `<tool_call>{json}</tool_call>` form or a bare JSON object with `name` and `arguments`. `"fallback"` means an `op:<name> {json}` block, the degraded path in `backend/app.py:477`. Task 5 (`toolcall_probe.py`) consumes this.

- [ ] **Step 1: Write the failing test**

Create `finetune/tests/test_classify.py`:

```python
from finetune.bakeoff.classify import classify_response


def test_qwen_style_tool_call_tags_are_native():
    text = '<tool_call>{"name": "preview_demo", "arguments": {"html": "<html></html>"}}</tool_call>'
    assert classify_response(text) == "native"


def test_bare_json_tool_call_is_native():
    text = '{"name": "preview_demo", "arguments": {"html": "<html></html>"}}'
    assert classify_response(text) == "native"


def test_native_detected_with_surrounding_prose():
    text = 'Sure, rendering that now.\n<tool_call>{"name": "save_demo", "arguments": {"demo_id": "d1"}}</tool_call>'
    assert classify_response(text) == "native"


def test_op_block_is_fallback():
    text = '```op:preview_demo\n{"html": "<html></html>"}\n```'
    assert classify_response(text) == "fallback"


def test_unfenced_op_block_is_fallback():
    # Back-compat: some earlier prompt revisions asked for the bare
    # `op:<name> {...}` form with no code fence. Still counts as fallback.
    text = 'op:preview_demo {"html": "<html></html>"}'
    assert classify_response(text) == "fallback"


def test_prose_only_is_none():
    assert classify_response("Here is some HTML you could use.") == "none"


def test_malformed_json_in_tags_is_not_native():
    text = '<tool_call>{"name": "preview_demo", "arguments":</tool_call>'
    assert classify_response(text) == "none"


def test_json_without_name_is_not_native():
    assert classify_response('{"arguments": {"html": "x"}}') == "none"


def test_native_wins_when_both_present():
    text = 'op:preview_demo {"html":"x"}\n<tool_call>{"name":"preview_demo","arguments":{"html":"x"}}</tool_call>'
    assert classify_response(text) == "native"


def test_empty_string_is_none():
    assert classify_response("") == "none"


def test_bare_json_call_followed_by_prose_with_stray_brace_is_native():
    # A greedy first-`{`-to-last-`}` regex would span from the tool call's
    # opening brace all the way to the brace in "{like colors}", producing a
    # blob that fails to parse as JSON at all.
    text = (
        '{"name": "preview_demo", "arguments": {"html": "<p>hi</p>"}}\n\n'
        "Tell me if you want changes {like colors}."
    )
    assert classify_response(text) == "native"


def test_two_bare_json_tool_calls_in_one_response_is_native():
    # A greedy regex spans from the first `{` to the LAST `}` here, capturing
    # both objects concatenated together, which is not valid JSON.
    text = (
        '{"name":"preview_demo","arguments":{"html":"<p>1</p>"}}\n'
        '{"name":"save_demo","arguments":{"demo_id":"d1"}}'
    )
    assert classify_response(text) == "native"


def test_js_object_literal_with_name_and_arguments_keys_in_code_fence_is_native():
    # The probe prompts ask the model to write HTML/JS demos, so generated
    # source commonly contains object literals. If such a literal happens to
    # be valid JSON *and* happens to use exactly the keys "name" and
    # "arguments", it is structurally identical to a real bare-JSON tool
    # call under this module's own definition of one (Task 4 interface:
    # "a bare JSON object with name and arguments") — there is no reliable
    # surface-syntax signal that distinguishes "demo source that coincidentally
    # matches" from "an actual tool call the model meant to make". We choose
    # NATIVE rather than trying to special-case code fences, since fence-aware
    # exclusion would also suppress genuine tool calls that some models wrap
    # in ```json fences, which is a worse failure mode for this measurement.
    text = (
        "Here's your demo:\n"
        "```html\n"
        "<script>\n"
        'const sceneConfig = {"name": "particle-system", "arguments": {"count": 500}};\n'
        "</script>\n"
        "```"
    )
    assert classify_response(text) == "native"


def test_native_wins_over_fenced_op_block():
    text = (
        '```op:preview_demo\n{"html":"x"}\n```\n'
        '<tool_call>{"name":"preview_demo","arguments":{"html":"x"}}</tool_call>'
    )
    assert classify_response(text) == "native"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/local-chatbot && ~/.unsloth/studio/unsloth_studio/bin/python -m pytest finetune/tests/test_classify.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'finetune.bakeoff.classify'`

- [ ] **Step 3: Write minimal implementation**

Create `finetune/bakeoff/classify.py`:

```python
"""Classify a model response as native tool call, op: fallback, or neither.

Mirrors what backend/app.py + frontend/app.js actually accept. Model families
differ in surface syntax (Qwen emits <tool_call> tags, others emit bare JSON),
so both count as native. The fallback path is the fenced ```op:<name>\\n{...}
form backend/app.py's operations prompt block instructs non-tool-calling
models to emit (see `_operations_prompt_block`), which frontend/app.js parses
with `OP_FENCE_RE`. The older unfenced `op:<name> {...}` form is also accepted
for back-compat with earlier prompt revisions.
"""
import json
import re

_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_OP_FENCED = re.compile(r"```op:[A-Za-z_][A-Za-z0-9_]*\s*\n")
_OP_UNFENCED = re.compile(r"^\s*op:([A-Za-z_][A-Za-z0-9_]*)\s*\{", re.MULTILINE)

NATIVE = "native"
FALLBACK = "fallback"
NONE = "none"


def _is_tool_call_payload(obj: object) -> bool:
    return isinstance(obj, dict) and "name" in obj and "arguments" in obj


def _payload_from_string(raw: str) -> bool:
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return False
    return _is_tool_call_payload(obj)


def _has_bare_json_tool_call(text: str) -> bool:
    """Scan every ``{`` position for a standalone decodable JSON object.

    A single greedy regex spanning the first ``{`` to the last ``}`` in the
    whole response does not isolate one JSON object — any stray brace
    elsewhere (trailing prose, a second tool call) breaks the parse or
    silently merges unrelated objects. Trying at each ``{`` with
    ``json.JSONDecoder.raw_decode`` lets the real JSON grammar find each
    object's true extent instead of guessing with a regex.
    """
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text, idx)
        except ValueError:
            continue
        if _is_tool_call_payload(obj):
            return True
    return False


def classify_response(text: str) -> str:
    if not text:
        return NONE

    for match in _TOOL_CALL_TAG.findall(text):
        if _payload_from_string(match):
            return NATIVE

    if _has_bare_json_tool_call(text):
        return NATIVE

    if _OP_FENCED.search(text) or _OP_UNFENCED.search(text):
        return FALLBACK

    return NONE
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/local-chatbot && ~/.unsloth/studio/unsloth_studio/bin/python -m pytest finetune/tests/test_classify.py -v
```

Expected: 14 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add finetune/bakeoff/classify.py finetune/tests/test_classify.py
git commit -m "feat: add native-vs-fallback tool call classifier"
```

---

### Task 5: Tool-call probe driver (Phase 0C)

**Files:**
- Create: `finetune/bakeoff/toolcall_probe.py`
- Create: `finetune/bakeoff/prompts/toolcall_prompts.json`
- Test: `finetune/tests/test_toolcall_probe.py`

**Interfaces:**
- Consumes: `classify_response` (Task 4); `schemas/html_demo_tools.json` (Task 1); `slugify` from `finetune.bakeoff.vram_probe` (Task 3).
- Produces: `summarize(labels: list[str]) -> dict` returning `{"n", "native", "fallback", "none", "native_rate"}` where `native_rate` is `native / n` rounded to 3 places, and `0.0` when `n == 0`. Task 6 reads the written JSON.

- [ ] **Step 1: Write the failing test**

Create `finetune/tests/test_toolcall_probe.py`:

```python
import json
from pathlib import Path

from finetune.bakeoff.toolcall_probe import summarize, load_prompts

PROMPTS = Path(__file__).parent.parent / "bakeoff" / "prompts" / "toolcall_prompts.json"


def test_summarize_counts_and_rates():
    out = summarize(["native", "native", "fallback", "none"])
    assert out == {"n": 4, "native": 2, "fallback": 1, "none": 1, "native_rate": 0.5}


def test_summarize_handles_empty():
    out = summarize([])
    assert out["n"] == 0
    assert out["native_rate"] == 0.0


def test_summarize_all_native_is_rate_one():
    assert summarize(["native"] * 7)["native_rate"] == 1.0


def test_prompt_file_has_at_least_ten_prompts():
    prompts = load_prompts(PROMPTS)
    assert len(prompts) >= 10
    assert all(isinstance(p, str) and p.strip() for p in prompts)


def test_prompts_cover_all_four_aesthetic_axes():
    blob = " ".join(load_prompts(PROMPTS)).lower()
    for axis in ("animat", "canvas", "layout", "chart"):
        assert axis in blob, f"no prompt covers {axis}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/local-chatbot && ~/.unsloth/studio/unsloth_studio/bin/python -m pytest finetune/tests/test_toolcall_probe.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'finetune.bakeoff.toolcall_probe'`

- [ ] **Step 3: Write prompts and implementation**

Create `finetune/bakeoff/prompts/toolcall_prompts.json`:

```json
[
  "Build me a demo where coloured particles orbit the cursor and trail behind it.",
  "Make a page with a scroll-driven animation where sections fade and slide in as you reach them.",
  "Create a canvas demo with an animated plasma field using sine noise.",
  "Show me a WebGL shader demo with a rotating gradient blob.",
  "Build an editorial-style article layout with an asymmetric grid and strong type hierarchy.",
  "Make a pricing page layout with three tiers and confident spacing rhythm.",
  "Create an animated bar chart that transitions between two datasets on click.",
  "Build an interactive line chart with hover tooltips, no charting library.",
  "Make a bouncing ball demo with realistic squash and stretch.",
  "Build a starfield that accelerates when you hold the mouse down.",
  "Create a card grid with a smooth flip animation on hover.",
  "Make a dark-mode dashboard layout with a sparkline and three stat tiles."
]
```

Create `finetune/bakeoff/toolcall_probe.py`:

```python
"""Probe 0C: how often does the model emit a NATIVE tool call?

Feeds html_demo tool schemas plus a demo request through the model's chat
template and classifies the response. This is the metric that most directly
predicts usefulness in wyltek-gpt: a model stuck on the op: fallback path
(backend/app.py:477) is a model the app has to work around.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from finetune.bakeoff.classify import classify_response
from finetune.bakeoff.vram_probe import slugify

HERE = Path(__file__).parent
DEFAULT_OUT = HERE / "results"
DEFAULT_PROMPTS = HERE / "prompts" / "toolcall_prompts.json"
DEFAULT_SCHEMAS = HERE / "schemas" / "html_demo_tools.json"


def load_prompts(path: Path) -> list[str]:
    return json.loads(Path(path).read_text())


def summarize(labels: list[str]) -> dict:
    counts = Counter(labels)
    n = len(labels)
    return {
        "n": n,
        "native": counts.get("native", 0),
        "fallback": counts.get("fallback", 0),
        "none": counts.get("none", 0),
        "native_rate": round(counts.get("native", 0) / n, 3) if n else 0.0,
    }


def generate(model_name: str, prompts: list[str], tools: list[dict],
             max_seqlen: int = 4096, max_new_tokens: int = 512) -> list[str]:
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=max_seqlen,
        load_in_4bit=True, dtype=None,
    )
    FastLanguageModel.for_inference(model)

    outputs = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        ids = tok.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True,
            return_tensors="pt", tokenize=True,
        ).to("cuda")
        gen = model.generate(input_ids=ids, max_new_tokens=max_new_tokens,
                             do_sample=False)
        text = tok.decode(gen[0][ids.shape[-1]:], skip_special_tokens=True)
        outputs.append(text)
        print(f"  [{classify_response(text):8s}] {p[:56]}")
    return outputs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--schemas", type=Path, default=DEFAULT_SCHEMAS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    tools = json.loads(args.schemas.read_text())
    texts = generate(args.model, prompts, tools)
    labels = [classify_response(t) for t in texts]
    result = {"model": args.model, "probe": "toolcall", **summarize(labels)}

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"toolcall_{slugify(args.model)}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"native_rate={result['native_rate']} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/local-chatbot && ~/.unsloth/studio/unsloth_studio/bin/python -m pytest finetune/tests/test_toolcall_probe.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add finetune/bakeoff/toolcall_probe.py finetune/bakeoff/prompts/toolcall_prompts.json \
        finetune/tests/test_toolcall_probe.py
git commit -m "feat: add native tool call reliability probe"
```

---

### Task 6: GSM8K scoring logic (Phase 0B logic)

**Files:**
- Create: `finetune/bakeoff/score.py`
- Test: `finetune/tests/test_score.py`

**Interfaces:**
- Produces: `extract_answer(text: str) -> str | None` (last number in the text, commas and `$` stripped, sign preserved) and `is_correct(prediction: str, gold: str) -> bool` where `gold` is a raw GSM8K answer field containing `#### <number>`. Task 7 consumes both.

- [ ] **Step 1: Write the failing test**

Create `finetune/tests/test_score.py`:

```python
from finetune.bakeoff.score import extract_answer, is_correct


def test_extracts_last_number():
    assert extract_answer("First 5, then 12. The answer is 18.") == "18"


def test_strips_commas_and_currency():
    assert extract_answer("Total cost is $1,250") == "1250"


def test_handles_negative():
    assert extract_answer("The change is -42") == "-42"


def test_handles_decimal():
    assert extract_answer("It comes to 3.5 hours") == "3.5"


def test_returns_none_when_no_number():
    assert extract_answer("I am not sure about this one.") is None


def test_is_correct_parses_gsm8k_gold_marker():
    gold = "She had 5 apples and ate 2.\n#### 3"
    assert is_correct("So the answer is 3", gold) is True


def test_is_correct_rejects_wrong_answer():
    gold = "#### 3"
    assert is_correct("The answer is 4", gold) is False


def test_is_correct_false_when_no_prediction_number():
    assert is_correct("I don't know", "#### 3") is False


def test_is_correct_ignores_thousands_separator_mismatch():
    assert is_correct("The answer is 1,250", "#### 1250") is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/local-chatbot && ~/.unsloth/studio/unsloth_studio/bin/python -m pytest finetune/tests/test_score.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'finetune.bakeoff.score'`

- [ ] **Step 3: Write minimal implementation**

Create `finetune/bakeoff/score.py`:

```python
"""GSM8K answer extraction and scoring.

Standard convention: the gold field ends with '#### <number>'. For the
prediction we take the LAST number mentioned, which is the usual heuristic
for chain-of-thought output that reasons before concluding.
"""
import re

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
_GOLD_MARKER = re.compile(r"####\s*(-?[\d,]+\.?\d*)")


def _normalize(raw: str) -> str:
    return raw.replace(",", "").replace("$", "").rstrip(".")


def extract_answer(text: str) -> str | None:
    if not text:
        return None
    matches = _NUMBER.findall(text)
    if not matches:
        return None
    return _normalize(matches[-1])


def is_correct(prediction: str, gold: str) -> bool:
    marker = _GOLD_MARKER.search(gold or "")
    gold_value = _normalize(marker.group(1)) if marker else extract_answer(gold)
    pred_value = extract_answer(prediction)
    if gold_value is None or pred_value is None:
        return False
    try:
        return float(pred_value) == float(gold_value)
    except ValueError:
        return pred_value == gold_value
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/local-chatbot && ~/.unsloth/studio/unsloth_studio/bin/python -m pytest finetune/tests/test_score.py -v
```

Expected: 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add finetune/bakeoff/score.py finetune/tests/test_score.py
git commit -m "feat: add GSM8K answer extraction and scoring"
```

---

### Task 7: Reasoning retention probe (Phase 0B)

**Files:**
- Create: `finetune/bakeoff/reasoning_probe.py`
- Test: `finetune/tests/test_reasoning_probe.py`

**Interfaces:**
- Consumes: `extract_answer`, `is_correct` (Task 6); `slugify` (Task 3).
- Produces: `retention(before: float, after: float) -> float` — `after / before` rounded to 3 places, `1.0` when `before == 0`. Written to `results/reasoning_<slug>.json` with keys `model`, `probe`, `n`, `accuracy_before`, `accuracy_after`, `retention`. Task 8 reads it.

This is the measurement that decides the bake-off: **degradation per unit of style training**, not raw capability. A stronger base that degrades faster can lose to a weaker one that holds its shape.

- [ ] **Step 1: Write the failing test**

Create `finetune/tests/test_reasoning_probe.py`:

```python
import json
from finetune.bakeoff.reasoning_probe import retention, write_result


def test_retention_is_ratio():
    assert retention(0.80, 0.72) == 0.9


def test_retention_of_perfect_hold_is_one():
    assert retention(0.5, 0.5) == 1.0


def test_retention_above_one_when_tuning_helped():
    assert retention(0.50, 0.55) == 1.1


def test_retention_guards_divide_by_zero():
    assert retention(0.0, 0.0) == 1.0


def test_write_result_roundtrips(tmp_path):
    path = write_result("unsloth/gemma-4-12b-it", 100, 0.80, 0.72, tmp_path)
    data = json.loads(path.read_text())
    assert data["probe"] == "reasoning"
    assert data["accuracy_before"] == 0.80
    assert data["accuracy_after"] == 0.72
    assert data["retention"] == 0.9
    assert data["n"] == 100
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/local-chatbot && ~/.unsloth/studio/unsloth_studio/bin/python -m pytest finetune/tests/test_reasoning_probe.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'finetune.bakeoff.reasoning_probe'`

- [ ] **Step 3: Write implementation**

Create `finetune/bakeoff/reasoning_probe.py`:

```python
"""Probe 0B: how much reasoning survives a short style fine-tune?

Scores GSM8K before tuning, runs a deliberately small throwaway LoRA on
HTML-ish text, scores again. The output is a RETENTION RATIO, not an
accuracy — we are comparing degradation rates across candidate bases.

The smoke corpus is intentionally tiny and synthetic. It exists to apply
gradient pressure in the same shape the real corpus will, not to teach
anything useful.
"""
import argparse
import json
from pathlib import Path

from finetune.bakeoff.score import is_correct
from finetune.bakeoff.vram_probe import slugify

DEFAULT_OUT = Path(__file__).parent / "results"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def retention(before: float, after: float) -> float:
    if before == 0:
        return 1.0
    return round(after / before, 3)


def write_result(model: str, n: int, before: float, after: float, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"reasoning_{slugify(model)}.json"
    path.write_text(json.dumps({
        "model": model,
        "probe": "reasoning",
        "n": n,
        "accuracy_before": before,
        "accuracy_after": after,
        "retention": retention(before, after),
    }, indent=2) + "\n")
    return path


def load_gsm8k(n: int):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{n}]")
    return [(r["question"], r["answer"]) for r in ds]


def score(model, tok, rows, max_new_tokens: int = 256) -> float:
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    correct = 0
    for question, gold in rows:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True, return_tensors="pt", tokenize=True,
        ).to("cuda")
        gen = model.generate(input_ids=ids, max_new_tokens=max_new_tokens,
                             do_sample=False)
        text = tok.decode(gen[0][ids.shape[-1]:], skip_special_tokens=True)
        correct += int(is_correct(text, gold))
    return round(correct / len(rows), 3)


def smoke_train(model, tok, steps: int, max_seqlen: int):
    """Apply gradient pressure in the shape the real corpus will."""
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    model = FastLanguageModel.get_peft_model(
        model, r=16, target_modules=TARGET_MODULES, lora_alpha=16,
        use_gradient_checkpointing="unsloth",
    )
    samples = [{"text":
        "<|user|>Make a demo with a bouncing ball.<|assistant|>"
        "<!DOCTYPE html><html><head><style>body{margin:0;background:#111}"
        "canvas{display:block}</style></head><body><canvas id=c></canvas>"
        "<script>const x=document.getElementById('c').getContext('2d');"
        "let y=0,v=2;function f(){y+=v;if(y>200||y<0)v*=-1;"
        "x.clearRect(0,0,300,300);x.beginPath();x.arc(150,y,20,0,7);"
        "x.fill();requestAnimationFrame(f)}f();</script></body></html>"
    }] * (steps * 2)

    trainer = SFTTrainer(
        model=model, tokenizer=tok, train_dataset=Dataset.from_list(samples),
        args=SFTConfig(max_steps=steps, per_device_train_batch_size=1,
                       gradient_accumulation_steps=1, learning_rate=2e-4,
                       max_seq_length=max_seqlen, logging_steps=5,
                       output_dir="/tmp/bakeoff_smoke", report_to="none"),
    )
    trainer.train()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100, help="GSM8K test rows")
    ap.add_argument("--steps", type=int, default=60, help="smoke LoRA steps")
    ap.add_argument("--max-seqlen", type=int, default=2048)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    from unsloth import FastLanguageModel

    rows = load_gsm8k(args.n)
    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seqlen,
        load_in_4bit=True, dtype=None,
    )

    before = score(model, tok, rows)
    print(f"accuracy BEFORE tuning: {before}")

    model = smoke_train(model, tok, args.steps, args.max_seqlen)

    after = score(model, tok, rows)
    print(f"accuracy AFTER  tuning: {after}")

    path = write_result(args.model, args.n, before, after, args.out)
    print(f"retention={retention(before, after)} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/local-chatbot && ~/.unsloth/studio/unsloth_studio/bin/python -m pytest finetune/tests/test_reasoning_probe.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add finetune/bakeoff/reasoning_probe.py finetune/tests/test_reasoning_probe.py
git commit -m "feat: add reasoning retention probe with smoke LoRA"
```

---

### Task 8: Report aggregation

**Files:**
- Create: `finetune/bakeoff/report.py`
- Test: `finetune/tests/test_report.py`

**Interfaces:**
- Consumes: all `results/*.json` written by Tasks 3, 5, 7.
- Produces: `collect(results_dir: Path) -> dict[str, dict]` keyed by model name, each value merging that model's probe results; and `render_table(collected: dict) -> str` producing a markdown table. Terminal deliverable — nothing consumes these.

- [ ] **Step 1: Write the failing test**

Create `finetune/tests/test_report.py`:

```python
import json
from finetune.bakeoff.report import collect, render_table


def _write(tmp_path, name, payload):
    (tmp_path / name).write_text(json.dumps(payload))


def test_collect_merges_probes_per_model(tmp_path):
    _write(tmp_path, "vram_m.json",
           {"model": "m", "probe": "vram", "max_seqlen": 8192})
    _write(tmp_path, "toolcall_m.json",
           {"model": "m", "probe": "toolcall", "native_rate": 0.75})
    out = collect(tmp_path)
    assert out["m"]["max_seqlen"] == 8192
    assert out["m"]["native_rate"] == 0.75


def test_collect_separates_models(tmp_path):
    _write(tmp_path, "vram_a.json", {"model": "a", "probe": "vram", "max_seqlen": 8192})
    _write(tmp_path, "vram_b.json", {"model": "b", "probe": "vram", "max_seqlen": 4096})
    out = collect(tmp_path)
    assert set(out) == {"a", "b"}
    assert out["b"]["max_seqlen"] == 4096


def test_collect_empty_dir_is_empty_dict(tmp_path):
    assert collect(tmp_path) == {}


def test_render_table_includes_headers_and_values():
    table = render_table({"m": {"max_seqlen": 8192, "native_rate": 0.75, "retention": 0.9}})
    assert "max_seqlen" in table
    assert "8192" in table
    assert "0.75" in table
    assert table.startswith("|")


def test_render_table_shows_dash_for_missing_probe():
    table = render_table({"m": {"max_seqlen": 8192}})
    assert "—" in table
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/local-chatbot && ~/.unsloth/studio/unsloth_studio/bin/python -m pytest finetune/tests/test_report.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'finetune.bakeoff.report'`

- [ ] **Step 3: Write minimal implementation**

Create `finetune/bakeoff/report.py`:

```python
"""Aggregate probe results into a comparison table."""
import argparse
import json
from pathlib import Path

DEFAULT_RESULTS = Path(__file__).parent / "results"
COLUMNS = ["max_seqlen", "native_rate", "accuracy_before", "accuracy_after", "retention"]


def collect(results_dir: Path) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        data = json.loads(path.read_text())
        model = data.get("model")
        if not model:
            continue
        entry = merged.setdefault(model, {})
        for k, v in data.items():
            if k not in ("model", "probe"):
                entry[k] = v
    return merged


def render_table(collected: dict[str, dict]) -> str:
    header = "| model | " + " | ".join(COLUMNS) + " |"
    divider = "|" + "---|" * (len(COLUMNS) + 1)
    rows = [
        "| " + model + " | " +
        " | ".join(str(vals.get(c, "—")) for c in COLUMNS) + " |"
        for model, vals in sorted(collected.items())
    ]
    return "\n".join([header, divider, *rows])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = ap.parse_args()
    collected = collect(args.results)
    if not collected:
        print(f"no results in {args.results}; run the probes first")
        return 1
    print(render_table(collected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the full test suite**

```bash
cd ~/local-chatbot && ~/.unsloth/studio/unsloth_studio/bin/python -m pytest finetune/tests/ -v
```

Expected: all 45 tests PASS across the eight test files (4 schema_bridge + 5 search + 3 vram_probe + 9 classify + 5 toolcall_probe + 9 score + 5 reasoning_probe + 5 report).

- [ ] **Step 5: Commit**

```bash
git add finetune/bakeoff/report.py finetune/tests/test_report.py
git commit -m "feat: add bakeoff result aggregation and comparison table"
```

---

### Task 9: Execute the bake-off and record findings

This task is **manual and GPU-bound**. No new code — it runs what Tasks 1–8 built and writes down what happened. Expect several hours, dominated by model downloads (~8 GB for 12B, ~17 GB for 27B in 4-bit) and GSM8K inference.

**Files:**
- Create: `docs/superpowers/specs/2026-07-21-bakeoff-results.md`

- [ ] **Step 1: Free the GPU**

```bash
systemctl --user stop wan-worker 2>/dev/null; ollama stop --all 2>/dev/null
rocm-smi --showmemuse | grep -i "VRAM%"
```

Expected: VRAM% at or near idle (~12%, desktop only). If higher, find and stop the holder before proceeding — a probe that OOMs on someone else's memory is a wasted measurement.

- [ ] **Step 2: Run all three probes for gemma-4-12b-it**

```bash
cd ~/local-chatbot
UV=~/.unsloth/studio/unsloth_studio/bin/python
$UV finetune/bakeoff/vram_probe.py      --model unsloth/gemma-4-12b-it
$UV finetune/bakeoff/toolcall_probe.py  --model unsloth/gemma-4-12b-it
$UV finetune/bakeoff/reasoning_probe.py --model unsloth/gemma-4-12b-it
```

Expected: three JSON files in `finetune/bakeoff/results/`.

- [ ] **Step 3: Run all three probes for Qwen3.6-27B**

```bash
cd ~/local-chatbot
UV=~/.unsloth/studio/unsloth_studio/bin/python
$UV finetune/bakeoff/vram_probe.py      --model unsloth/Qwen3.6-27B
$UV finetune/bakeoff/toolcall_probe.py  --model unsloth/Qwen3.6-27B
$UV finetune/bakeoff/reasoning_probe.py --model unsloth/Qwen3.6-27B --max-seqlen 2048
```

Expected: three more JSON files. If the 27B VRAM probe returns `max_seqlen=0`, that is a **finding, not a failure** — record it and stop testing 27B.

- [ ] **Step 4: Generate the comparison table**

```bash
cd ~/local-chatbot && ~/.unsloth/studio/unsloth_studio/bin/python finetune/bakeoff/report.py
```

Expected: a markdown table with one row per model.

- [ ] **Step 5: Write up the decision**

Create `docs/superpowers/specs/2026-07-21-bakeoff-results.md` containing: the generated table verbatim; the chosen base model; the reasoning behind the choice (weighing `max_seqlen` against `retention` against `native_rate`); and — importantly — how the measured VRAM numbers compare to the estimates in the design spec, since that spec explicitly flagged them as unverified.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-07-21-bakeoff-results.md
git commit -m "docs: record Phase 0 bakeoff results and base model choice"
```

---

## Self-Review

**Spec coverage.** Phase 0A → Tasks 2–3. Phase 0B → Tasks 6–7. Phase 0C → Tasks 4–5. Aggregation and decision → Tasks 8–9. The spec's "Open items" on the reasoning benchmark is resolved here: GSM8K, 100 rows, retention ratio. Corpus size, 31B testing, `interactions?`, and JSONL location remain Phase 1 concerns and are deliberately out of scope.

**Placeholder scan.** No TBD/TODO. Every code step contains complete runnable code. Task 9 is manual by nature but its steps are exact commands with expected output.

**Type consistency.** `slugify` is defined once in `vram_probe.py` and imported by `toolcall_probe.py` and `reasoning_probe.py`. `write_result` appears in both `vram_probe` and `reasoning_probe` with different signatures — these are module-local and never cross-imported, so there is no collision. `classify_response` returns the same three string literals used by `summarize`. `COLUMNS` in `report.py` matches keys written by all three probes.

**Known limitation, recorded deliberately.** `classify.py` handles `<tool_call>` tags and bare JSON. If Gemma 4 emits a third surface syntax, Task 9 Step 2 will show an implausibly low `native_rate` — inspect a raw generation before trusting that number, and extend `classify.py` with a new failing test if a new form appears.
