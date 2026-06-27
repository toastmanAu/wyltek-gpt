import os
import shutil
from pathlib import Path

import pytest

from backend import app as app_module

pytestmark = pytest.mark.needs_cellc
_HAS_CELLC = bool(os.environ.get("CELLC_BIN") or shutil.which("cellc"))

_GOOD = """\
module probe::token
resource Token has store, create, consume, replace { amount: u64, symbol: [u8; 8] }
"""
_BAD = "module probe::broken\n\nresource Token has store { amount: u64\n"


@pytest.mark.skipif(not _HAS_CELLC, reason="cellc not installed")
def test_save_writes_good_refuses_bad(tmp_path):
    out = app_module._run_cellc_save_op_sync(tmp_path, {"name": "good", "source": _GOOD})
    assert out.read_text() == _GOOD
    with pytest.raises(app_module.HTTPException):
        app_module._run_cellc_save_op_sync(tmp_path, {"name": "bad", "source": _BAD})
    assert not (tmp_path / "bad.cell").exists()
