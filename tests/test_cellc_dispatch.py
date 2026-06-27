from unittest import mock

from backend.bridges import cellc


def test_tool_schemas_shape():
    schemas = cellc.tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert names == {"cellc_check", "cellc_explain", "cellc_get_example", "cellc_language_reference"}
    for s in schemas:
        assert s["type"] == "function"
        assert "parameters" in s["function"]
    # cellc_check requires source
    check = next(s for s in schemas if s["function"]["name"] == "cellc_check")
    assert "source" in check["function"]["parameters"]["required"]


def test_dispatch_check_validates_and_forwards():
    with mock.patch.object(cellc, "_server") as srv:
        srv.cellc_check.return_value = {"ok": True, "diagnostics": []}
        out = cellc.dispatch("cellc_check", {"source": "module x", "target_profile": "ckb"})
    srv.cellc_check.assert_called_once_with(source="module x", target_profile="ckb", full=False)
    assert out["ok"] is True


def test_dispatch_empty_source_is_tool_error_not_raise():
    out = cellc.dispatch("cellc_check", {"source": ""})
    assert out["tool_error"] is True
    assert "source" in out["stderr"].lower()


def test_dispatch_bad_profile_tool_error():
    out = cellc.dispatch("cellc_check", {"source": "module x", "target_profile": "evm"})
    assert out["tool_error"] is True


def test_dispatch_bad_code_tool_error():
    out = cellc.dispatch("cellc_explain", {"code": "E14; rm -rf"})
    assert out["tool_error"] is True


def test_dispatch_unknown_tool_error():
    out = cellc.dispatch("cellc_nope", {})
    assert out["tool_error"] is True


def test_dispatch_language_reference_no_args():
    with mock.patch.object(cellc, "_server") as srv:
        srv.cellc_language_reference.return_value = "REF"
        out = cellc.dispatch("cellc_language_reference", {})
    assert out == {"reference": "REF"}
