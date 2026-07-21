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
