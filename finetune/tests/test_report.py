import json
from finetune.bakeoff.report import collect, render_table


def _write(tmp_path, name, payload):
    (tmp_path / name).write_text(json.dumps(payload))


def _write_raw(tmp_path, name, text):
    (tmp_path / name).write_text(text)


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


def test_collect_skips_malformed_json_but_keeps_good_ones(tmp_path, capsys):
    _write(tmp_path, "vram_a.json", {"model": "a", "probe": "vram", "max_seqlen": 8192})
    _write_raw(tmp_path, "vram_x.json", '{"model": "x", "max_seqlen": 100')  # truncated
    _write(tmp_path, "vram_b.json", {"model": "b", "probe": "vram", "max_seqlen": 4096})

    out = collect(tmp_path)

    assert set(out) == {"a", "b"}
    assert out["a"]["max_seqlen"] == 8192
    assert out["b"]["max_seqlen"] == 4096
    err = capsys.readouterr().err
    assert "vram_x.json" in err


def test_collect_skips_non_dict_json(tmp_path, capsys):
    _write(tmp_path, "vram_a.json", {"model": "a", "probe": "vram", "max_seqlen": 8192})
    _write_raw(tmp_path, "vram_bad.json", "[1, 2, 3]")

    out = collect(tmp_path)

    assert set(out) == {"a"}
    err = capsys.readouterr().err
    assert "vram_bad.json" in err


def test_collect_skips_unreadable_file(tmp_path, capsys):
    # A directory matching the *.json glob pattern: Path.read_text() raises
    # IsADirectoryError (an OSError subclass) — a portable way to exercise
    # the OSError branch without relying on permission bits (which root
    # ignores, making chmod-based tests unreliable in CI/sandboxes).
    (tmp_path / "vram_dir.json").mkdir()
    _write(tmp_path, "vram_a.json", {"model": "a", "probe": "vram", "max_seqlen": 8192})

    out = collect(tmp_path)

    assert set(out) == {"a"}
    err = capsys.readouterr().err
    assert "vram_dir.json" in err


def test_collect_keeps_both_values_on_cross_probe_key_collision(tmp_path, capsys):
    _write(tmp_path, "reasoning_m.json",
           {"model": "m", "probe": "reasoning", "n": 100, "retention": 0.9})
    _write(tmp_path, "toolcall_m.json",
           {"model": "m", "probe": "toolcall", "n": 40, "native_rate": 0.75})

    out = collect(tmp_path)

    values = set(out["m"].values())
    # Both original "n" values must survive somewhere in the merged entry —
    # one under the plain "n" key, the other under a probe-namespaced key.
    assert 100 in values
    assert 40 in values
    namespaced = [k for k in out["m"] if k != "n" and k.endswith("_n")]
    assert namespaced, f"expected a probe-namespaced 'n' key, got {out['m']!r}"
    assert out["m"]["native_rate"] == 0.75
    assert out["m"]["retention"] == 0.9
    err = capsys.readouterr().err
    assert "'n'" in err or '"n"' in err or " n " in err


def test_collect_identical_colliding_values_do_not_namespace(tmp_path, capsys):
    _write(tmp_path, "reasoning_m.json",
           {"model": "m", "probe": "reasoning", "n": 100})
    _write(tmp_path, "toolcall_m.json",
           {"model": "m", "probe": "toolcall", "n": 100})

    out = collect(tmp_path)

    assert out["m"]["n"] == 100
    assert not [k for k in out["m"] if k != "n" and k.endswith("_n")]
    err = capsys.readouterr().err
    assert err == ""


def test_render_table_regression_zero_and_false_are_not_dashes():
    # Locks in the fix a prior review verified: `vals.get(c, "—")` must
    # distinguish ABSENT (dash) from FALSY-BUT-PRESENT (literal 0.0/False).
    table = render_table({
        "m": {
            "max_seqlen": 8192,
            "native_rate": 0.0,
            "training_effective": False,
        }
    })
    assert "0.0" in table
    assert "False" in table
    row = [line for line in table.splitlines() if line.startswith("| m ")][0]
    assert "—" in row  # the still-missing columns should show as dashes
    assert row.count("—") == 3  # accuracy_before, accuracy_after, retention
