import os
import shutil

import pytest


@pytest.fixture
def cellc_available():
    return bool(os.environ.get("CELLC_BIN") or shutil.which("cellc"))
