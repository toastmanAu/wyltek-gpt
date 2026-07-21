import json
from finetune.bakeoff.reasoning_probe import retention, write_result


def test_retention_is_ratio():
    assert retention(0.80, 0.72) == 0.9


def test_retention_of_perfect_hold_is_one():
    assert retention(0.5, 0.5) == 1.0


def test_retention_above_one_when_tuning_helped():
    assert retention(0.50, 0.55) == 1.1


def test_retention_guards_divide_by_zero():
    assert retention(0.0, 0.0) == 1.0


def test_write_result_roundtrips(tmp_path):
    path = write_result("unsloth/gemma-4-12b-it", 100, 0.80, 0.72, tmp_path)
    data = json.loads(path.read_text())
    assert data["probe"] == "reasoning"
    assert data["accuracy_before"] == 0.80
    assert data["accuracy_after"] == 0.72
    assert data["retention"] == 0.9
    assert data["n"] == 100
