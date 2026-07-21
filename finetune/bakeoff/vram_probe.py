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
