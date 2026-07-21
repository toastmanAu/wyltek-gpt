import json
from finetune.bakeoff.report import COLUMNS, collect, loss_delta, render_table


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
    # Derived from COLUMNS rather than hardcoded, so adding a column extends
    # this guarantee instead of breaking it: EVERY absent column renders a
    # dash, and the two falsy-but-present ones above render literally.
    present = {"max_seqlen", "native_rate", "training_effective"}
    assert row.count("—") == len([c for c in COLUMNS if c not in present])


def test_collect_skips_invalid_utf8_but_keeps_good_ones(tmp_path, capsys):
    # UnicodeDecodeError subclasses ValueError, NOT OSError, so an `except
    # OSError` around read_text() misses it entirely. A write truncated
    # mid-multi-byte-UTF-8 sequence is exactly the OOM/Ctrl-C scenario in the
    # threat model, and it used to kill the whole report.
    _write(tmp_path, "vram_a.json", {"model": "a", "probe": "vram", "max_seqlen": 8192})
    (tmp_path / "vram_bytes.json").write_bytes(b"\x00\x01\x02\xff\xfe")

    out = collect(tmp_path)

    assert set(out) == {"a"}
    assert out["a"]["max_seqlen"] == 8192
    err = capsys.readouterr().err
    assert "vram_bytes.json" in err


def test_collect_skips_deeply_nested_json(tmp_path, capsys):
    # Deeply-nested but otherwise VALID JSON blows the recursion limit inside
    # json.loads, raising RecursionError (not JSONDecodeError).
    _write(tmp_path, "vram_a.json", {"model": "a", "probe": "vram", "max_seqlen": 8192})
    _write_raw(tmp_path, "vram_deep.json", "[" * 200000 + "]" * 200000)

    out = collect(tmp_path)

    assert set(out) == {"a"}
    err = capsys.readouterr().err
    assert "vram_deep.json" in err


def test_collect_same_probe_name_reuse_keeps_every_differing_value(tmp_path, capsys):
    # A re-run probe can leave two differing files for one model. The old
    # namespacing wrote `<probe>_<key>` without checking occupancy, so a second
    # collision clobbered the first and a value vanished silently.
    _write(tmp_path, "dup_1.json", {"model": "m", "probe": "dup", "n": 1})
    _write(tmp_path, "dup_2.json", {"model": "m", "probe": "dup", "n": 2})
    _write(tmp_path, "dup_3.json", {"model": "m", "probe": "dup", "n": 3})

    out = collect(tmp_path)

    values = set(out["m"].values())
    assert {1, 2, 3} <= values, f"a differing value was lost: {out['m']!r}"
    err = capsys.readouterr().err
    assert "dup" in err


def test_loss_delta_normal():
    assert loss_delta(2.5, 1.5) == 1.0


def test_loss_delta_negative_when_loss_rose():
    assert loss_delta(1.0, 1.5) == -0.5


def test_loss_delta_none_when_either_missing():
    assert loss_delta(None, 1.5) is None
    assert loss_delta(2.5, None) is None
    assert loss_delta(None, None) is None


def test_collect_derives_loss_delta(tmp_path):
    _write(tmp_path, "reasoning_m.json",
           {"model": "m", "probe": "reasoning", "loss_start": 2.5, "loss_end": 1.5})
    out = collect(tmp_path)
    assert out["m"]["loss_delta"] == 1.0


def test_collect_omits_loss_delta_when_losses_missing(tmp_path):
    _write(tmp_path, "vram_m.json", {"model": "m", "probe": "vram", "max_seqlen": 8192})
    out = collect(tmp_path)
    assert "loss_delta" not in out["m"]


def test_render_table_includes_diagnostic_columns():
    # loss_delta / unparseable_* are written to JSON and printed at run time,
    # but the TABLE is what drives the decision weeks later.
    table = render_table({
        "m": {"loss_delta": 0.8, "unparseable_before": 2, "unparseable_after": 31}
    })
    assert "loss_delta" in table
    assert "unparseable_before" in table
    assert "unparseable_after" in table
    assert "0.8" in table
    assert "31" in table
