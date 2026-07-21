import json
from finetune.bakeoff.vram_probe import (
    measured_ceiling,
    slugify,
    warn_if_above_measured_ceiling,
    write_result,
)


def test_slugify_makes_safe_filename():
    assert slugify("unsloth/gemma-4-12b-it") == "unsloth-gemma-4-12b-it"
    assert slugify("unsloth/Qwen3.6-27B") == "unsloth-qwen3-6-27b"


def test_write_result_creates_readable_json(tmp_path):
    path = write_result("unsloth/gemma-4-12b-it", 8192, 21.3, tmp_path)
    assert path.is_file()
    data = json.loads(path.read_text())
    assert data["model"] == "unsloth/gemma-4-12b-it"
    assert data["probe"] == "vram"
    assert data["max_seqlen"] == 8192
    assert data["free_vram_gb"] == 21.3


def test_write_result_filename_matches_slug(tmp_path):
    path = write_result("unsloth/Qwen3.6-27B", 4096, 21.0, tmp_path)
    assert path.name == "vram_unsloth-qwen3-6-27b.json"


def test_measured_ceiling_reads_recorded_value(tmp_path):
    write_result("unsloth/gemma-4-12b-it", 6144, 21.0, tmp_path)
    assert measured_ceiling("unsloth/gemma-4-12b-it", tmp_path) == 6144


def test_measured_ceiling_none_when_probe_not_run(tmp_path):
    assert measured_ceiling("unsloth/never-probed", tmp_path) is None


def test_measured_ceiling_none_on_unusable_file(tmp_path):
    # A malformed vram result means "no measurement", never a crash: this is
    # consulted at the START of an expensive probe.
    (tmp_path / "vram_unsloth-broken.json").write_text("{not json")
    assert measured_ceiling("unsloth/broken", tmp_path) is None


def test_warns_when_requested_seqlen_exceeds_measured_ceiling(tmp_path, capsys):
    write_result("unsloth/qwen3-27b", 2048, 21.0, tmp_path)
    assert warn_if_above_measured_ceiling("unsloth/qwen3-27b", 4096, tmp_path) is True
    err = capsys.readouterr().err
    assert "2048" in err and "4096" in err


def test_no_warning_when_within_measured_ceiling(tmp_path, capsys):
    write_result("unsloth/gemma-4-12b-it", 8192, 21.0, tmp_path)
    assert warn_if_above_measured_ceiling("unsloth/gemma-4-12b-it", 2048, tmp_path) is False
    assert capsys.readouterr().err == ""


def test_no_warning_when_ceiling_unknown(tmp_path, capsys):
    # 0A not yet run for this model: the later probe must proceed silently
    # rather than nag about a measurement that does not exist.
    assert warn_if_above_measured_ceiling("unsloth/unprobed", 4096, tmp_path) is False
    assert capsys.readouterr().err == ""
