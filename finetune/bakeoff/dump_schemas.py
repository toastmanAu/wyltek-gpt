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
