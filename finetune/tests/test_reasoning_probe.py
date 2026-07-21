import json
from finetune.bakeoff.reasoning_probe import retention, write_result, training_effective


def test_retention_is_ratio():
    assert retention(0.80, 0.72) == 0.9


def test_retention_of_perfect_hold_is_one():
    assert retention(0.5, 0.5) == 1.0


def test_retention_above_one_when_tuning_helped():
    assert retention(0.50, 0.55) == 1.1


def test_retention_guards_divide_by_zero():
    assert retention(0.0, 0.0) == 1.0


def test_retention_guards_asymmetric_zero():
    # before == 0 but after > 0 must still hit the guard, not divide.
    assert retention(0.0, 0.5) == 1.0


def test_write_result_roundtrips(tmp_path):
    path = write_result(
        model="unsloth/gemma-4-12b-it",
        n=100,
        before=0.80,
        after=0.72,
        loss_start=1.9,
        loss_end=1.2,
        unparseable_before=1,
        unparseable_after=4,
        out_dir=tmp_path,
    )
    data = json.loads(path.read_text())
    assert data["probe"] == "reasoning"
    assert data["model"] == "unsloth/gemma-4-12b-it"
    assert data["n"] == 100
    assert data["accuracy_before"] == 0.80
    assert data["accuracy_after"] == 0.72
    assert data["retention"] == 0.9
    assert data["loss_start"] == 1.9
    assert data["loss_end"] == 1.2
    assert data["training_effective"] is True
    assert data["unparseable_before"] == 1
    assert data["unparseable_after"] == 4


def test_training_effective_true_when_loss_decreased():
    assert training_effective(1.9, 1.2) is True


def test_training_effective_false_when_loss_increased():
    assert training_effective(1.0, 1.5) is False


def test_training_effective_false_when_loss_unchanged():
    assert training_effective(1.2, 1.2) is False


def test_training_effective_false_when_loss_start_missing():
    assert training_effective(None, 1.2) is False


def test_training_effective_false_when_loss_end_missing():
    assert training_effective(1.2, None) is False
