import os
import shutil
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from backend import app as app_module

pytestmark = pytest.mark.needs_cellc
_HAS_CELLC = bool(os.environ.get("CELLC_BIN") or shutil.which("cellc"))
client = TestClient(app_module.app)

_BAD = "module probe::broken\n\nresource Token has store { amount: u64\n"


@pytest.mark.skipif(not _HAS_CELLC, reason="cellc not installed")
def test_real_cellc_check_drives_one_loop(monkeypatch):
    # Turn 1: model calls cellc_check on bad source. Turn 2: model emits text.
    seq = iter([
        [("tool_calls", [{"function": {"name": "cellc_check", "arguments": {"source": _BAD}}}]), ("done", None)],
        [("chunk", "I see the error"), ("done", None)],
    ])

    def fake_stream_one(body):
        evts = next(seq)
        async def gen(_b):
            for e in evts:
                yield e
        return gen(body)

    monkeypatch.setattr(app_module, "_stream_one", fake_stream_one, raising=False)
    resp = client.post("/api/chat", json={"model": "m", "messages": [{"role": "user", "content": "check this"}]})
    body = resp.text
    assert "__cellc_step__" in body
    assert "error" in body.lower()        # the real cellc reported the syntax error
    assert "I see the error" in body      # second turn streamed after the real tool result
