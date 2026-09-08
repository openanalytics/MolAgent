"""Test that a corrupt token store raises rather than silently resetting."""
import json, os, pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_corrupt_store_raises_not_resets(tmp_path):
    """_load_store must raise on JSONDecodeError, not return an empty store."""
    from _auth import _load_store

    corrupt = tmp_path / "auth_tokens.json"
    corrupt.write_text("{ this is not valid json }")

    with pytest.raises(Exception):   # JSONDecodeError or subclass
        _load_store(corrupt)


def test_missing_store_returns_empty(tmp_path):
    """_load_store must return empty store when the file does not exist (first run)."""
    from _auth import _load_store

    missing = tmp_path / "no_such_file.json"
    result = _load_store(missing)
    assert result == {"admin_token_hash": None, "users": {}}
