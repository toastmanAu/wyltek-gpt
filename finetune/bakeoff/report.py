"""Aggregate probe results into a comparison table."""
import argparse
import json
import sys
from pathlib import Path

DEFAULT_RESULTS = Path(__file__).parent / "results"
COLUMNS = ["max_seqlen", "native_rate", "accuracy_before", "accuracy_after", "retention",
           "training_effective"]


def collect(results_dir: Path) -> dict[str, dict]:
    """Merge per-model probe JSON files from results_dir into one dict per model.

    Resilient to a run that produced a partially-written or otherwise bad file:
    such files are skipped with a warning on stderr rather than aborting the
    whole aggregation (a single truncated file must not hide every model's
    results after a run that may have taken hours).

    Cross-probe key collisions (two files disagreeing on the meaning of the
    same key, e.g. "n") are preserved rather than silently overwritten: the
    value about to be shadowed is stashed under a probe-namespaced key
    (`<probe>_<key>`) before the new value takes the plain key, and a warning
    is printed. Identical colliding values are harmless and produce no
    namespaced key or warning.
    """
    merged: dict[str, dict] = {}
    # Per-model, per-key: which probe most recently supplied the current value.
    # Needed so a later collision can be attributed to the right probe prefix.
    sources: dict[str, dict[str, str]] = {}

    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            raw = path.read_text()
        except OSError as exc:
            print(f"warning: skipping {path}: could not read file ({exc})", file=sys.stderr)
            continue

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"warning: skipping {path}: malformed JSON ({exc})", file=sys.stderr)
            continue

        if not isinstance(data, dict):
            print(
                f"warning: skipping {path}: expected a JSON object at the top level, "
                f"got {type(data).__name__}",
                file=sys.stderr,
            )
            continue

        model = data.get("model")
        if not model:
            continue

        probe_name = data.get("probe") or path.stem
        entry = merged.setdefault(model, {})
        entry_sources = sources.setdefault(model, {})

        for k, v in data.items():
            if k in ("model", "probe"):
                continue
            if k in entry and entry[k] != v:
                prev_probe = entry_sources.get(k, "unknown")
                namespaced_key = f"{prev_probe}_{k}"
                print(
                    f"warning: {path.name}: key '{k}' means different things across "
                    f"probes for model '{model}' ({prev_probe}={entry[k]!r} vs "
                    f"{probe_name}={v!r}); preserving prior value as '{namespaced_key}'",
                    file=sys.stderr,
                )
                entry[namespaced_key] = entry[k]
            entry[k] = v
            entry_sources[k] = probe_name

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
