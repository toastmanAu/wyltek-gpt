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
