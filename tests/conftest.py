"""Keep generated test artifacts inside the authorized application directory."""

from pathlib import Path
import tempfile
import pytest


@pytest.fixture
def tmp_path():
    root = Path(__file__).parents[1] / ".pytest_cache" / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as directory:
        yield Path(directory)
