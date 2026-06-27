from unittest import mock

from backend import app as app_module


def test_is_cellc_intent_true():
    assert app_module._is_cellc_intent("write a CellScript token contract")
    assert app_module._is_cellc_intent("check this .cell file")
    assert app_module._is_cellc_intent("make a CKB contract")


def test_is_cellc_intent_false():
    assert not app_module._is_cellc_intent("what's the weather today")
    assert not app_module._is_cellc_intent("write a python function")


def test_full_system_prompt_has_hint_when_available(monkeypatch):
    monkeypatch.setattr(app_module.cellc_bridge, "available", lambda: True)
    # Assert on the actual hint block, not a bare substring: "cellc_check" can
    # also appear via a registered op's description (e.g. cellc_save), so the
    # canonical signal is the hint text itself.
    assert app_module._CELLC_PROMPT_HINT in app_module._full_system_prompt()


def test_full_system_prompt_no_hint_when_unavailable(monkeypatch):
    monkeypatch.setattr(app_module.cellc_bridge, "available", lambda: False)
    # The hint block must be absent; do NOT assert on bare "cellc_check", which
    # legitimately appears in the cellc_save op description when that op is
    # registered (CELLC_BIN set in the environment).
    assert app_module._CELLC_PROMPT_HINT not in app_module._full_system_prompt()


def test_design_notes_injected_on_cellc_intent(monkeypatch):
    monkeypatch.setattr(app_module.cellc_bridge, "available", lambda: True)
    # build the system message the way /api/chat does for a cellc-intent message
    msgs = [{"role": "system", "content": app_module._full_system_prompt()}]
    app_module._inject_cellc_context(msgs, "write a CellScript token contract")
    sys = msgs[0]["content"]
    assert "CellScript design notes" in sys or "Cell-model design rules" in sys
    assert "lock_args" in sys.lower()


def test_design_notes_absent_on_plain_chat(monkeypatch):
    monkeypatch.setattr(app_module.cellc_bridge, "available", lambda: True)
    msgs = [{"role": "system", "content": app_module._full_system_prompt()}]
    app_module._inject_cellc_context(msgs, "what's the weather")
    assert "Cell-model design rules" not in msgs[0]["content"]
