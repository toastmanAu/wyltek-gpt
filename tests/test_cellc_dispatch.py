from unittest import mock

from backend.bridges import cellc


def test_tool_schemas_shape():
    schemas = cellc.tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert {"cellc_check", "cellc_explain", "cellc_get_example", "cellc_language_reference"} <= names
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


def test_tool_schemas_has_six_tools():
    names = {s["function"]["name"] for s in cellc.tool_schemas()}
    assert names == {
        "cellc_check", "cellc_explain", "cellc_get_example",
        "cellc_language_reference", "cellc_metadata", "cellc_list_examples",
    }


def test_cellc_tool_names_has_six():
    assert len(cellc.CELLC_TOOL_NAMES) == 6
    assert "cellc_metadata" in cellc.CELLC_TOOL_NAMES
    assert "cellc_list_examples" in cellc.CELLC_TOOL_NAMES


def test_dispatch_metadata_forwards():
    from unittest import mock
    with mock.patch.object(cellc, "_server") as srv:
        srv.cellc_metadata.return_value = {"resources_count": 2}
        out = cellc.dispatch("cellc_metadata", {"source": "module x"})
    srv.cellc_metadata.assert_called_once_with(source="module x", target_profile="ckb", full=False)
    assert out["resources_count"] == 2


def test_dispatch_list_examples():
    from unittest import mock
    with mock.patch.object(cellc, "_server") as srv:
        srv.cellc_list_examples.return_value = [{"name": "token"}]
        out = cellc.dispatch("cellc_list_examples", {})
    assert out == {"examples": [{"name": "token"}]}
