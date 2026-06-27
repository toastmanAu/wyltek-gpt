from unittest import mock

from backend.bridges import cellc


def test_check_forwards_to_cellc_mcp():
    with mock.patch.object(cellc, "_server") as srv:
        srv.cellc_check.return_value = {"ok": True, "diagnostics": []}
        out = cellc.check("module x", target_profile="ckb")
    srv.cellc_check.assert_called_once_with(source="module x", target_profile="ckb", full=False)
    assert out == {"ok": True, "diagnostics": []}


def test_explain_forwards_code():
    with mock.patch.object(cellc, "_server") as srv:
        srv.cellc_explain.return_value = {"ecode": "E0014"}
        out = cellc.explain("E0014")
    srv.cellc_explain.assert_called_once_with(code="E0014")
    assert out["ecode"] == "E0014"


def test_get_example_forwards_name():
    with mock.patch.object(cellc, "_server") as srv:
        srv.cellc_get_example.return_value = {"name": "token", "source": "module x"}
        out = cellc.get_example("token")
    srv.cellc_get_example.assert_called_once_with(name="token")
    assert out["name"] == "token"


def test_available_false_when_binary_missing(monkeypatch):
    # find_cellc raising → not available
    def _raise():
        raise cellc._CellcNotFound("no binary")
    monkeypatch.setattr(cellc, "_find_cellc", _raise)
    assert cellc.available() is False
    assert cellc.status() == {"available": False, "binary": None}


def test_available_true_when_binary_present(monkeypatch):
    monkeypatch.setattr(cellc, "_find_cellc", lambda: "/path/to/cellc")
    assert cellc.available() is True
    assert cellc.status() == {"available": True, "binary": "/path/to/cellc"}
