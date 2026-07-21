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
import sys
from pathlib import Path

from finetune.bakeoff.search import find_max_seqlen

DEFAULT_OUT = Path(__file__).parent / "results"
# The spec budgets ~21 GB usable of the card's 24 GB: roughly 3 GB is held by
# the desktop compositor and never available to a probe. Anything below this
# means something else is also holding VRAM.
USABLE_VRAM_GB = 21


def slugify(model: str) -> str:
    return re.sub(r"[/.]+", "-", model).lower()


def measured_ceiling(model: str, results_dir: Path) -> int | None:
    """The max_seqlen probe 0A recorded for `model`, or None if not yet run.

    Deliberately forgiving: a missing, unreadable or malformed vram result is
    simply "no measurement available", never a reason to abort a later probe.
    """
    path = Path(results_dir) / f"vram_{slugify(model)}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, RecursionError):
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("max_seqlen")
    return value if isinstance(value, int) else None


def warn_if_above_measured_ceiling(model: str, seqlen: int, results_dir: Path) -> bool:
    """Warn (loudly, on stderr) when a later probe asks for more than 0A measured.

    Probe 0A measures the largest sequence length that trains without OOM, but
    0B/0C have their own defaults (2048 and 4096) that know nothing about it.
    If a candidate's measured ceiling is LOWER, the later probe OOMs hours into
    a run with no explanation. Returns True when a warning was emitted.

    Warns rather than overrides on purpose: silently rewriting a value the user
    typed on the command line would be a worse surprise than an OOM.
    """
    ceiling = measured_ceiling(model, results_dir)
    if ceiling is None or ceiling >= seqlen:
        return False
    print(
        f"WARNING: probe 0A measured max_seqlen={ceiling} for {model}, but this "
        f"probe is about to use {seqlen}. That is ABOVE the measured ceiling and "
        f"is likely to OOM. Consider re-running with --max-seqlen {ceiling}.",
        file=sys.stderr,
    )
    return True


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
    if free < USABLE_VRAM_GB:
        print(f"WARNING: <{USABLE_VRAM_GB} GB free. Stop Ollama/ComfyUI before probing.")

    ceiling = find_max_seqlen(make_fits(args.model), args.lo, args.hi, args.step)
    path = write_result(args.model, ceiling, free, args.out)
    print(f"max_seqlen={ceiling} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
