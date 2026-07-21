"""Aggregate probe results into a comparison table."""
import argparse
import json
from pathlib import Path

DEFAULT_RESULTS = Path(__file__).parent / "results"
COLUMNS = ["max_seqlen", "native_rate", "accuracy_before", "accuracy_after", "retention",
           "training_effective"]


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
