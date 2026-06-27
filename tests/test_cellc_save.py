from unittest import mock

from backend import app as app_module
from backend.operations import make_cellc_save_operation


def test_make_cellc_save_operation_schema():
    op = make_cellc_save_operation()
    assert op.id == "cellc_save"
    assert op.kind == "cellc_save"
    schema = op.to_tool_schema()
    props = schema["function"]["parameters"]["properties"]
    assert "name" in props and "source" in props
    assert set(schema["function"]["parameters"]["required"]) == {"name", "source"}


def test_cellc_save_not_in_auto_loop_names():
    # cellc_save must route to the confirm card, never the auto-loop
    assert "cellc_save" not in app_module.cellc_bridge.CELLC_TOOL_NAMES


def test_cellc_save_partitions_as_op_call():
    calls = [{"function": {"name": "cellc_save", "arguments": {"name": "t", "source": "x"}}}]
    cellc_calls, op_calls = app_module._partition_cellc_calls(calls)
    assert cellc_calls == []
    assert len(op_calls) == 1 and op_calls[0]["function"]["name"] == "cellc_save"
