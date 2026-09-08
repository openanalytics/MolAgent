"""Tests that paths outside the output root are rejected."""
import pytest
from pathlib import Path
import sys, os

# Add mcp/ to path so we can import directly
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_under_output_root_accepts_valid(tmp_path, monkeypatch):
    monkeypatch.setenv("MOLAGENT_OUTPUT_ROOT", str(tmp_path))
    from _pipeline import _under_output_root, _output_root

    valid = tmp_path / "run1"
    result = _under_output_root(valid)
    assert result == valid.resolve()


def test_under_output_root_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("MOLAGENT_OUTPUT_ROOT", str(tmp_path))
    from _pipeline import _under_output_root

    outside = tmp_path / ".." / "etc" / "passwd"
    with pytest.raises(ValueError):
        _under_output_root(outside)


def test_under_output_root_rejects_absolute_outside(tmp_path, monkeypatch):
    monkeypatch.setenv("MOLAGENT_OUTPUT_ROOT", str(tmp_path))
    from _pipeline import _under_output_root

    with pytest.raises(ValueError):
        _under_output_root(Path("/etc/shadow"))
