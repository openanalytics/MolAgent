"""Tests for merge_models.py — property name inference and CLI structure."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Add scripts to path for import
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "train-pipeline" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


class TestInferPropertyName:
    """Test the infer_property_name helper function."""

    def test_regression_model(self):
        from merge_models import infer_property_name
        assert infer_property_name("MolagentFiles/gamma1_stackingregmodel.pt") == "gamma1"

    def test_classification_model(self):
        from merge_models import infer_property_name
        assert infer_property_name("MolagentFiles/gamma1_stackingclfmodel.pt") == "gamma1"

    def test_refitted_regression(self):
        from merge_models import infer_property_name
        assert infer_property_name("MolagentFiles/gamma1_refitted_stackingregmodel.pt") == "gamma1"

    def test_refitted_classification(self):
        from merge_models import infer_property_name
        assert infer_property_name("MolagentFiles/gamma1_refitted_stackingclfmodel.pt") == "gamma1"

    def test_property_with_underscore(self):
        from merge_models import infer_property_name
        assert infer_property_name("MolagentFiles/log_p_stackingregmodel.pt") == "log_p"

    def test_deep_path(self):
        from merge_models import infer_property_name
        assert infer_property_name("/some/deep/path/Y_stackingregmodel.pt") == "Y"

    def test_class_prefix_stripped(self):
        """Class_ prefix is not relevant for merge — only in predict."""
        from merge_models import infer_property_name
        # merge_models doesn't strip Class_ — only predict does
        result = infer_property_name("MolagentFiles/Class_toxic_stackingclfmodel.pt")
        assert result == "Class_toxic"


class TestMergeModelsCLI:
    """Test CLI interface structure (without actually loading models)."""

    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "merge_models.py"), "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "--model-files" in result.stdout
        assert "--output-file" in result.stdout
        assert "--verify-encoder" in result.stdout
        assert "--cleanup" in result.stdout

    def test_missing_required_args(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "merge_models.py")],
            capture_output=True, text=True
        )
        assert result.returncode != 0

    def test_model_file_not_found(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "merge_models.py"),
             "--model-files", str(tmp_path / "nonexistent.pt"),
             "--output-file", str(tmp_path / "out.pt")],
            capture_output=True, text=True
        )
        assert result.returncode != 0
        assert "not found" in result.stderr

    def test_single_file_shortcircuit(self, tmp_path):
        """Single model file should be copied, not merged."""
        src = tmp_path / "gamma1_stackingregmodel.pt"
        src.write_bytes(b"fake model content")
        out = tmp_path / "merged.pt"

        # Single file path uses shutil.copy2 — no load_model/save_model
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "merge_models.py"),
             "--model-files", str(src),
             "--output-file", str(out)],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert out.exists()
        # Parse JSON from last line of stdout (load_model prints to stdout too)
        output = json.loads(result.stdout.strip().split("\n")[-1])
        assert output["merged"] is False
        assert output["properties"] == ["gamma1"]
