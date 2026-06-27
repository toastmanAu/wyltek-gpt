# cellc integration setup (wyltek-gpt, driveThree)

The /api/cellc/* endpoints call the cellc CellScript compiler via the
cellc-mcp package. To enable them:

1. Build cellc once (needs a sibling ckb-sdk-rust @ v5.1.0 checkout):
       cd ~/CellScript && cargo build --release -p cellscript --bin cellc

2. Install the bridge package into wyltek-gpt's venv:
       ~/local-chatbot/.venv/bin/pip install -e ~/cellc-mcp

3. Start uvicorn with these env vars set:
       CELLC_BIN=/home/phill/CellScript/target/release/cellc
       CELLSCRIPT_REPO=/home/phill/CellScript
       .venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000

Without these, /api/cellc/status returns {"available": false} and the
write/check endpoints return 503 — wyltek-gpt still boots normally.

## Tests
    .venv/bin/python -m pytest -v                 # unit (offline)
    CELLC_BIN=... CELLSCRIPT_REPO=... .venv/bin/python -m pytest -m needs_cellc -v
