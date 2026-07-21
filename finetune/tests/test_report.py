import json
from finetune.bakeoff.report import collect, render_table


def _write(tmp_path, name, payload):
    (tmp_path / name).write_text(json.dumps(payload))


def test_collect_merges_probes_per_model(tmp_path):
    _write(tmp_path, "vram_m.json",
           {"model": "m", "probe": "vram", "max_seqlen": 8192})
    _write(tmp_path, "toolcall_m.json",
           {"model": "m", "probe": "toolcall", "native_rate": 0.75})
    out = collect(tmp_path)
    assert out["m"]["max_seqlen"] == 8192
    assert out["m"]["native_rate"] == 0.75


def test_collect_separates_models(tmp_path):
    _write(tmp_path, "vram_a.json", {"model": "a", "probe": "vram", "max_seqlen": 8192})
    _write(tmp_path, "vram_b.json", {"model": "b", "probe": "vram", "max_seqlen": 4096})
    out = collect(tmp_path)
    assert set(out) == {"a", "b"}
    assert out["b"]["max_seqlen"] == 4096


def test_collect_empty_dir_is_empty_dict(tmp_path):
    assert collect(tmp_path) == {}


def test_render_table_includes_headers_and_values():
    table = render_table({"m": {"max_seqlen": 8192, "native_rate": 0.75, "retention": 0.9}})
    assert "max_seqlen" in table
    assert "8192" in table
    assert "0.75" in table
    assert table.startswith("|")


def test_render_table_shows_dash_for_missing_probe():
    table = render_table({"m": {"max_seqlen": 8192}})
    assert "—" in table


def test_render_table_shows_training_effective_false():
    table = render_table({"m": {"max_seqlen": 8192, "training_effective": False}})
    assert "False" in table
