"""Aggregate probe results into a comparison table."""
import argparse
import json
import sys
from pathlib import Path

DEFAULT_RESULTS = Path(__file__).parent / "results"
# Decision columns first, then the diagnostics that tell a reader whether the
# decision columns can be trusted at all (see `loss_delta` and the Task 9
# interpretation checklist in finetune/README.md).
COLUMNS = ["max_seqlen", "native_rate", "accuracy_before", "accuracy_after", "retention",
           "training_effective", "loss_delta", "unparseable_before", "unparseable_after"]


def loss_delta(loss_start: float | None, loss_end: float | None) -> float | None:
    """How far loss actually fell during the smoke LoRA (start - end).

    `training_effective` is only a boolean floor: it says loss moved DOWN, not
    how far. Two candidates can both report True having received wildly
    different training pressure — and a 27B absorbs far less change than a 12B
    under identical LoRA settings, which confounds retention with model size.
    A visible delta is what lets a human notice that the candidates were not
    comparably trained before ranking them on retention.

    Returns None when either reading is missing; the caller omits the key in
    that case so the table renders an unambiguous dash.
    """
    if loss_start is None or loss_end is None:
        return None
    return round(loss_start - loss_end, 4)


def collect(results_dir: Path) -> dict[str, dict]:
    """Merge per-model probe JSON files from results_dir into one dict per model.

    Contract: NEVER let one bad file kill the report. Any file that cannot be
    read, decoded, parsed, or that is not a JSON object is skipped with a
    warning on stderr naming the file and the reason. A single truncated file
    must not hide every model's results after a run that may have taken hours,
    so the exception net here is deliberately broad rather than clever —
    unreadable bytes (OSError), invalid UTF-8 (UnicodeDecodeError, which
    subclasses ValueError and so is NOT caught by an `except OSError`),
    malformed JSON (JSONDecodeError), and deeply-nested-but-valid JSON
    (RecursionError) are all merely "this file is unusable".

    Key collisions (two files disagreeing on the value of the same key, e.g.
    "n" from the toolcall and reasoning probes) are preserved rather than
    silently overwritten: the value about to be shadowed is stashed under a
    `<probe>_<key>` key before the new value takes the plain key, and a warning
    is printed. If that namespaced key is itself already occupied — which
    happens when the SAME probe is re-run and leaves two differing files for
    one model — a numeric suffix is appended so no differing value is ever
    dropped. Identical colliding values are harmless and produce no namespaced
    key or warning.
    """
    merged: dict[str, dict] = {}
    # Per-model, per-key: which probe most recently supplied the current value.
    # Needed so a later collision can be attributed to the right probe prefix.
    sources: dict[str, dict[str, str]] = {}

    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"warning: skipping {path}: could not read file "
                  f"({type(exc).__name__}: {exc})", file=sys.stderr)
            continue

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, RecursionError, ValueError) as exc:
            print(f"warning: skipping {path}: malformed JSON "
                  f"({type(exc).__name__}: {exc})", file=sys.stderr)
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
                # Guard the namespaced slot: re-running the SAME probe leaves
                # two differing files under one probe name, and an unguarded
                # assignment would clobber the first collision's stashed value
                # (three files with n=1,2,3 used to yield {n:3, dup_n:2} —
                # the 1 gone, silently). Suffix until we find a free slot.
                namespaced_key = f"{prev_probe}_{k}"
                if namespaced_key in entry:
                    suffix = 2
                    while f"{namespaced_key}_{suffix}" in entry:
                        suffix += 1
                    namespaced_key = f"{namespaced_key}_{suffix}"
                    note = (f"repeat collision under the same probe name "
                            f"'{prev_probe}' — this usually means that probe "
                            f"was re-run and left more than one result file")
                else:
                    note = "different probes disagree on this key"
                print(
                    f"warning: {path.name}: key '{k}' collides for model "
                    f"'{model}' ({prev_probe}={entry[k]!r} vs "
                    f"{probe_name}={v!r}); {note}; preserving prior value as "
                    f"'{namespaced_key}'",
                    file=sys.stderr,
                )
                entry[namespaced_key] = entry[k]
            entry[k] = v
            entry_sources[k] = probe_name

    for entry in merged.values():
        delta = loss_delta(entry.get("loss_start"), entry.get("loss_end"))
        if delta is not None:
            entry["loss_delta"] = delta

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
