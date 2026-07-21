"""Probe 0B: how much reasoning survives a short style fine-tune?

Scores GSM8K before tuning, runs a deliberately small throwaway LoRA on
HTML-ish text, scores again. The output is a RETENTION RATIO, not an
accuracy — we are comparing degradation rates across candidate bases.

The smoke corpus is intentionally tiny and synthetic. It exists to apply
gradient pressure in the same shape the real corpus will, not to teach
anything useful.

The probe is only as trustworthy as the training pressure it applies: if the
smoke LoRA is a no-op, every candidate reports retention ~= 1.0 whether or
not it actually held its shape. `smoke_train` therefore returns a loss
trajectory alongside the model, and `main` records whether that trajectory
actually decreased (`training_effective`) plus how many post-tuning
generations came back with no extractable number at all
(`unparseable_after` vs `unparseable_before`) — a spike there is the
signature of the smoke corpus bleeding into GSM8K generations rather than a
real reasoning regression.
"""
import argparse
import json
from pathlib import Path

from finetune.bakeoff.score import extract_answer, is_correct
from finetune.bakeoff.vram_probe import slugify, warn_if_above_measured_ceiling

DEFAULT_OUT = Path(__file__).parent / "results"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Distinct, digit-light HTML/CSS demo snippets. Each teaches a genuinely
# different bit of "style" (canvas animation, CSS grid, SVG, transitions,
# flex layout, gradients) so the smoke LoRA applies gradient pressure in the
# SHAPE of a real style corpus rather than memorizing one repeated string.
# Kept AS digit-light as the markup allows (see module docstring / Finding 4):
# a digit-dense corpus risks `extract_answer`'s last-number heuristic grabbing
# a memorized corpus digit instead of the model's actual GSM8K conclusion. Note
# this is a tendency, not a uniform property — sample 0 (the canvas
# bouncing-ball demo) carries notably more digits than the other five because
# the animation maths needs them. That is the sample to suspect first if
# `unparseable_after` jumps or accuracy_after looks oddly low.
SMOKE_SAMPLES = [
    "<!DOCTYPE html><html><head><style>body{margin:0;background:navy}"
    "canvas{display:block}</style></head><body><canvas id=c></canvas>"
    "<script>const ctx=document.getElementById('c').getContext('2d');"
    "let y=0,v=2;function f(){y+=v;if(y>200||y<0)v*=-1;"
    "ctx.clearRect(0,0,300,300);ctx.beginPath();"
    "ctx.arc(150,y,20,0,Math.PI*2);ctx.fill();"
    "requestAnimationFrame(f)}f();</script></body></html>",

    "<!DOCTYPE html><html><head><style>.grid{display:grid;"
    "grid-template-columns:repeat(auto-fit,minmax(1rem,1fr));gap:1rem}"
    ".card{background:whitesmoke;border-radius:0.5rem;padding:1rem}"
    "</style></head><body><div class=grid>"
    "<div class=card>One</div><div class=card>Two</div>"
    "<div class=card>Three</div></div></body></html>",

    "<!DOCTYPE html><html><head><style>svg{background:aliceblue}"
    "</style></head><body>"
    "<svg viewBox='0 0 1 1'>"
    "<polyline points='0,1 1,0' stroke=teal fill=none stroke-width=0.02/>"
    "</svg></body></html>",

    "<!DOCTYPE html><html><head><style>.btn{background:coral;"
    "color:white;padding:0.5rem 1rem;border-radius:0.25rem;"
    "transition:transform 0.2s ease}.btn:hover{transform:scale(1.1)}"
    "</style></head><body><button class=btn>Hover me</button>"
    "</body></html>",

    "<!DOCTYPE html><html><head><style>body{display:flex;"
    "flex-direction:column;align-items:center;justify-content:center;"
    "min-height:100vh;font-family:sans-serif}</style></head>"
    "<body><h1>Centered</h1><p>Flex layout demo</p></body></html>",

    "<!DOCTYPE html><html><head><style>.gradient{background:"
    "linear-gradient(to right,indigo,magenta);height:100vh}"
    "</style></head><body><div class=gradient></div></body></html>",
]


def retention(before: float, after: float) -> float:
    if before == 0:
        return 1.0
    return round(after / before, 3)


def training_effective(loss_start: float | None, loss_end: float | None) -> bool:
    """Did the smoke LoRA actually move loss, or was it a no-op?

    True only when both readings are present and loss strictly decreased.
    Missing values (e.g. `for_training` was never called and the trainer
    logged nothing) or a flat/increasing loss must report False — this is
    the falsifiable check that keeps a retention of ~1.0 from silently
    meaning "nothing was measured".
    """
    if loss_start is None or loss_end is None:
        return False
    return loss_end < loss_start


def write_result(
    model: str,
    n: int,
    before: float,
    after: float,
    loss_start: float | None,
    loss_end: float | None,
    unparseable_before: int,
    unparseable_after: int,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"reasoning_{slugify(model)}.json"
    path.write_text(json.dumps({
        "model": model,
        "probe": "reasoning",
        "n": n,
        "accuracy_before": before,
        "accuracy_after": after,
        "retention": retention(before, after),
        "loss_start": loss_start,
        "loss_end": loss_end,
        "training_effective": training_effective(loss_start, loss_end),
        "unparseable_before": unparseable_before,
        "unparseable_after": unparseable_after,
    }, indent=2) + "\n")
    return path


def load_gsm8k(n: int):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{n}]")
    return [(r["question"], r["answer"]) for r in ds]


def score(model, tok, rows, max_new_tokens: int = 256) -> tuple[float, int]:
    """Return (accuracy, unparseable_count).

    unparseable_count is how many generations produced no extractable
    number at all — tracked so callers can compare it before/after tuning
    and catch corpus bleed (Finding 4) instead of silently mis-scoring it
    as a reasoning regression.
    """
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    correct = 0
    unparseable = 0
    for question, gold in rows:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True, return_tensors="pt", tokenize=True,
        ).to("cuda")
        gen = model.generate(input_ids=ids, max_new_tokens=max_new_tokens,
                             do_sample=False)
        text = tok.decode(gen[0][ids.shape[-1]:], skip_special_tokens=True)
        if extract_answer(text) is None:
            unparseable += 1
        correct += int(is_correct(text, gold))
    return round(correct / len(rows), 3), unparseable


def smoke_train(model, tok, steps: int, max_seqlen: int):
    """Apply gradient pressure in the shape the real corpus will.

    Returns (model, loss_start, loss_end). Both losses come straight from
    the trainer's own logged history (`trainer.state.log_history`) — this
    is the falsifiable check for Finding 1: if `for_training` were skipped
    or the LoRA pass were otherwise a no-op, loss would stay flat and
    `training_effective` (computed by the caller) would report False
    instead of a misleadingly perfect retention.
    """
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    model = FastLanguageModel.get_peft_model(
        model, r=16, target_modules=TARGET_MODULES, lora_alpha=16,
        use_gradient_checkpointing="unsloth",
    )
    # Unsloth pairs for_inference()/for_training() around generation vs
    # training passes; score() already calls for_inference() for the BEFORE
    # measurement, so training must explicitly flip the model back before
    # the trainer runs or the LoRA pass applies no real gradient pressure.
    FastLanguageModel.for_training(model)

    # Format the smoke stimulus with THIS model's own chat template so the
    # training shape matches what score() evaluates with apply_chat_template
    # — a hardcoded "<|user|>...<|assistant|>" literal only resembles a real
    # chat turn for some candidates and is an uncontrolled confound across
    # the rest (Finding 2).
    formatted = [
        tok.apply_chat_template(
            [{"role": "user", "content": "Make a short HTML/CSS demo."},
             {"role": "assistant", "content": sample}],
            tokenize=False,
        )
        for sample in SMOKE_SAMPLES
    ]
    n_rows = max(steps, len(formatted))  # margin so cycling covers `steps`
    samples = [{"text": formatted[i % len(formatted)]} for i in range(n_rows)]

    trainer = SFTTrainer(
        # `processing_class`, NOT `tokenizer`: TRL renamed this parameter and
        # the installed trl (0.23.1) has no `tokenizer` parameter at all. Under
        # bare TRL that is an immediate TypeError; under unsloth's patched
        # SFTTrainer (which does accept **kwargs) it is worse — the tokenizer
        # is silently swallowed. Either way this fires only AFTER the ~100
        # GSM8K generations of the before-pass, so getting it wrong wastes a
        # model download plus a full scoring run.
        model=model, processing_class=tok, train_dataset=Dataset.from_list(samples),
        # `max_length` is what TRL actually truncates on; `max_seq_length` is
        # stored inert (it reads back as the value we set while max_length
        # quietly stays at its 1024 default). Passing only max_seq_length made
        # --max-seqlen a lie: the ceiling probe 0A measures would never be the
        # length 0B trains at. Both are passed — max_seq_length is accepted
        # under the unsloth-patched SFTConfig this probe always runs beneath.
        args=SFTConfig(max_steps=steps, per_device_train_batch_size=1,
                       gradient_accumulation_steps=1, learning_rate=2e-4,
                       max_length=max_seqlen, max_seq_length=max_seqlen,
                       logging_steps=1,
                       output_dir="/tmp/bakeoff_smoke", report_to="none"),
    )
    trainer.train()

    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    loss_start = losses[0] if losses else None
    loss_end = losses[-1] if losses else None
    return model, loss_start, loss_end


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100, help="GSM8K test rows")
    ap.add_argument("--steps", type=int, default=60, help="smoke LoRA steps")
    ap.add_argument("--max-seqlen", type=int, default=2048)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    # Probe 0A's measured ceiling is not otherwise wired into this probe: the
    # --max-seqlen default (2048) knows nothing about what 0A found.
    warn_if_above_measured_ceiling(args.model, args.max_seqlen, args.out)

    from unsloth import FastLanguageModel

    rows = load_gsm8k(args.n)
    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seqlen,
        load_in_4bit=True, dtype=None,
    )

    before, unparseable_before = score(model, tok, rows)
    print(f"accuracy BEFORE tuning: {before}")

    model, loss_start, loss_end = smoke_train(model, tok, args.steps, args.max_seqlen)

    after, unparseable_after = score(model, tok, rows)
    print(f"accuracy AFTER  tuning: {after}")

    effective = training_effective(loss_start, loss_end)
    print(f"loss_start={loss_start} loss_end={loss_end} training_effective={effective}")
    if not effective:
        print("WARNING: smoke training did not measurably reduce loss — "
              "the retention number below is NOT trustworthy. Check that "
              "FastLanguageModel.for_training() ran and that steps/lr are "
              "sufficient to move loss at all.")

    if unparseable_after > unparseable_before:
        print(f"NOTE: unparseable predictions rose {unparseable_before} -> "
              f"{unparseable_after} after tuning — check for smoke-corpus "
              f"digit bleed into GSM8K generations before trusting accuracy_after.")

    path = write_result(
        args.model, args.n, before, after, loss_start, loss_end,
        unparseable_before, unparseable_after, args.out,
    )
    print(f"retention={retention(before, after)} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
