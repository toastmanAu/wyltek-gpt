from unittest import mock

from fastapi.testclient import TestClient

from backend import app as app_module

client = TestClient(app_module.app)


def test_status_endpoint():
    with mock.patch.object(app_module.cellc_bridge, "status",
                           return_value={"available": True, "binary": "/x/cellc"}):
        r = client.get("/api/cellc/status")
    assert r.status_code == 200
    assert r.json()["available"] is True


def test_check_happy_path():
    with mock.patch.object(app_module.cellc_bridge, "available", return_value=True), \
         mock.patch.object(app_module.cellc_bridge, "check",
                           return_value={"ok": False, "diagnostics": [], "error_count": 1}) as chk:
        r = client.post("/api/cellc/check", json={"source": "module x", "target_profile": "ckb"})
    assert r.status_code == 200
    assert r.json()["error_count"] == 1
    chk.assert_called_once_with(source="module x", target_profile="ckb", full=False)


def test_check_empty_source_400():
    with mock.patch.object(app_module.cellc_bridge, "available", return_value=True):
        r = client.post("/api/cellc/check", json={"source": ""})
    assert r.status_code == 400


def test_check_bad_profile_400():
    with mock.patch.object(app_module.cellc_bridge, "available", return_value=True):
        r = client.post("/api/cellc/check", json={"source": "module x", "target_profile": "evm"})
    assert r.status_code == 400


def test_check_oversized_source_400():
    big = "x" * 200_001
    with mock.patch.object(app_module.cellc_bridge, "available", return_value=True):
        r = client.post("/api/cellc/check", json={"source": big})
    assert r.status_code in (400, 413)  # brief specifies "400/413" — impl returns 413


def test_check_503_when_unavailable():
    with mock.patch.object(app_module.cellc_bridge, "available", return_value=False):
        r = client.post("/api/cellc/check", json={"source": "module x"})
    assert r.status_code == 503


def test_explain_bad_code_400():
    with mock.patch.object(app_module.cellc_bridge, "available", return_value=True):
        r = client.post("/api/cellc/explain", json={"code": "E14; rm -rf"})
    assert r.status_code == 400


def test_example_bad_name_400():
    with mock.patch.object(app_module.cellc_bridge, "available", return_value=True):
        r = client.get("/api/cellc/example/..%2f..%2fetc%2fpasswd")
    assert r.status_code in (400, 404)
