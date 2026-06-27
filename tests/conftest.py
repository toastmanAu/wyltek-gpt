import os
import shutil

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_cellc: requires a built cellc binary")


@pytest.fixture
def cellc_available():
    return bool(os.environ.get("CELLC_BIN") or shutil.which("cellc"))
