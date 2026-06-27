import json
from unittest import mock

from fastapi.testclient import TestClient

from backend import app as app_module

client = TestClient(app_module.app)


def _events(*evts):
    async def gen(body):
        for e in evts:
            yield e
    return gen


def _read(resp):
    return resp.text


def test_partition_separates_cellc_and_ops():
    calls = [
        {"function": {"name": "cellc_check", "arguments": {"source": "x"}}},
        {"function": {"name": "trim_video", "arguments": {}}},
    ]
    cellc_calls, op_calls = app_module._partition_cellc_calls(calls)
    assert len(cellc_calls) == 1 and cellc_calls[0]["function"]["name"] == "cellc_check"
    assert len(op_calls) == 1 and op_calls[0]["function"]["name"] == "trim_video"


def test_summarize_step_check_failed():
    s = app_module._summarize_cellc_step(
        "cellc_check",
        {"ok": False, "error_count": 2, "diagnostics": [{"line": 12}, {"line": 20}]},
    )
    assert "2" in s and "12" in s


def test_loop_executes_cellc_and_reinvokes(monkeypatch):
    # First model turn emits a cellc_check tool call; second turn emits text.
    turn1 = _events(
        ("tool_calls", [{"function": {"name": "cellc_check", "arguments": {"source": "module x"}}}]),
        ("done", None),
    )
    turn2 = _events(("chunk", "fixed it"), ("done", None))
    bodies = []
    calls = iter([turn1, turn2])

    def fake_stream_one(body):
        bodies.append(body)
        return next(calls)(body)

    monkeypatch.setattr(app_module, "_stream_one_factoryless", None, raising=False)
    monkeypatch.setattr(app_module, "cellc_bridge", mock.Mock(
        available=mock.Mock(return_value=True),
        CELLC_TOOL_NAMES=frozenset({"cellc_check"}),
        tool_schemas=mock.Mock(return_value=[]),
        dispatch=mock.Mock(return_value={"ok": False, "error_count": 1, "diagnostics": [{"line": 3}]}),
    ))
    # Patch _stream_one used inside the handler:
    monkeypatch.setattr(app_module, "_stream_one", fake_stream_one, raising=False)

    resp = client.post("/api/chat", json={"model": "m", "messages": [{"role": "user", "content": "write a token"}]})
    body = _read(resp)
    assert "__cellc_step__" in body          # a transparent step was emitted
    assert "fixed it" in body                # the model's second turn streamed
    # the second _stream_one body carried a tool-result message
    assert len(bodies) == 2
    assert any(m.get("role") == "tool" for m in bodies[1]["messages"])


def test_loop_stops_at_cap(monkeypatch):
    # Every turn re-emits a cellc call; ensure we don't exceed MAX_CELLC_ITERS re-invokes.
    def always_cellc(body):
        async def gen(_b):
            yield ("tool_calls", [{"function": {"name": "cellc_check", "arguments": {"source": "x"}}}])
            yield ("done", None)
        return gen(body)
    count = {"n": 0}
    def counting(body):
        count["n"] += 1
        return always_cellc(body)
    monkeypatch.setattr(app_module, "cellc_bridge", mock.Mock(
        available=mock.Mock(return_value=True),
        CELLC_TOOL_NAMES=frozenset({"cellc_check"}),
        tool_schemas=mock.Mock(return_value=[]),
        dispatch=mock.Mock(return_value={"ok": False, "error_count": 1, "diagnostics": []}),
    ))
    monkeypatch.setattr(app_module, "_stream_one", counting, raising=False)
    client.post("/api/chat", json={"model": "m", "messages": [{"role": "user", "content": "x"}]})
    # initial call + at most MAX_CELLC_ITERS re-invokes
    assert count["n"] <= app_module.MAX_CELLC_ITERS + 1
