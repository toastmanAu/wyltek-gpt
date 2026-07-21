"""Probe 0C: how often does the model emit a NATIVE tool call?

Feeds html_demo tool schemas plus a demo request through the model's chat
template and classifies the response. This is the metric that most directly
predicts usefulness in wyltek-gpt: a model stuck on the op: fallback path
(backend/app.py:172 and backend/app.py:498) is a model the app has to work
around.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from finetune.bakeoff.classify import classify_response
from finetune.bakeoff.vram_probe import slugify, warn_if_above_measured_ceiling

DEFAULT_MAX_SEQLEN = 4096

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
             max_seqlen: int = DEFAULT_MAX_SEQLEN,
             max_new_tokens: int = 512) -> list[str]:
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
    # Probe 0A's measured ceiling is not otherwise wired into this probe.
    warn_if_above_measured_ceiling(args.model, DEFAULT_MAX_SEQLEN, args.out)
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
