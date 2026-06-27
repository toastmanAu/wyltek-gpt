import os
import shutil

import pytest
from fastapi.testclient import TestClient

from backend import app as app_module

pytestmark = pytest.mark.needs_cellc
_HAS_CELLC = bool(os.environ.get("CELLC_BIN") or shutil.which("cellc"))
client = TestClient(app_module.app)

_GOOD = """\
module probe::token
resource Token has store, create, consume, replace { amount: u64, symbol: [u8; 8] }
action transfer_token(token: Token, to: Address) -> next_token: Token {
    verification
        consume token
        create next_token = Token { amount: token.amount, symbol: token.symbol } with_lock(to)
}
"""
_BAD = "module probe::broken\n\nresource Token has store { amount: u64\n"


@pytest.mark.skipif(not _HAS_CELLC, reason="cellc not installed")
def test_status_reports_available():
    assert client.get("/api/cellc/status").json()["available"] is True


@pytest.mark.skipif(not _HAS_CELLC, reason="cellc not installed")
def test_check_good_then_bad():
    good = client.post("/api/cellc/check", json={"source": _GOOD}).json()
    assert good["ok"] is True
    bad = client.post("/api/cellc/check", json={"source": _BAD}).json()
    assert bad["ok"] is False and bad["error_count"] >= 1 and bad["diagnostics"]


@pytest.mark.skipif(not _HAS_CELLC, reason="cellc not installed")
def test_explain_e0014():
    assert client.post("/api/cellc/explain", json={"code": "E0014"}).json()["ecode"] == "E0014"
