import json
from finetune.bakeoff.vram_probe import write_result, slugify


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
