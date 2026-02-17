"""Tests for predict.py — property inference and CLI structure."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "predict" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


class TestInferPropertyFromFilename:
    """Test the infer_property_from_filename helper."""

    def test_regression(self):
        from predict import infer_property_from_filename
        assert infer_property_from_filename("MolagentFiles/gamma1_stackingregmodel.pt") == "gamma1"

    def test_classification(self):
        from predict import infer_property_from_filename
        assert infer_property_from_filename("MolagentFiles/gamma1_stackingclfmodel.pt") == "gamma1"

    def test_refitted(self):
        from predict import infer_property_from_filename
        assert infer_property_from_filename("MolagentFiles/gamma1_refitted_stackingregmodel.pt") == "gamma1"

    def test_class_prefix_stripped(self):
        from predict import infer_property_from_filename
        assert infer_property_from_filename("MolagentFiles/Class_toxic_stackingclfmodel.pt") == "toxic"

    def test_refitted_classification_with_class(self):
        from predict import infer_property_from_filename
        assert infer_property_from_filename("MolagentFiles/Class_toxic_refitted_stackingclfmodel.pt") == "toxic"

    def test_underscore_property(self):
        from predict import infer_property_from_filename
        assert infer_property_from_filename("MolagentFiles/log_p_stackingregmodel.pt") == "log_p"

    def test_merged_model_name(self):
        """Merged model names don't follow the per-property pattern."""
        from predict import infer_property_from_filename
        result = infer_property_from_filename("MolagentFiles/merged_stackingregmodel.pt")
        assert result == "merged"

    def test_merged_refitted(self):
        from predict import infer_property_from_filename
        result = infer_property_from_filename("MolagentFiles/merged_refitted_stackingregmodel.pt")
        assert result == "merged"


class TestPredictCLI:
    """Test CLI interface structure."""

    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "predict.py"), "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "--model-file" in result.stdout
        assert "--properties" in result.stdout
        assert "--smiles-file" in result.stdout
        assert "--smiles-list" in result.stdout

    def test_properties_option_present(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "predict.py"), "--help"],
            capture_output=True, text=True
        )
        assert "--properties" in result.stdout
        # Click wraps long help text across lines, so check individual words
        assert "Properties" in result.stdout
