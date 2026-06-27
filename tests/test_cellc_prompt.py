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
    assert "cellc_check" in app_module._full_system_prompt()


def test_full_system_prompt_no_hint_when_unavailable(monkeypatch):
    monkeypatch.setattr(app_module.cellc_bridge, "available", lambda: False)
    # base prompt should not advertise cellc tooling
    assert "cellc_check" not in app_module._full_system_prompt()
