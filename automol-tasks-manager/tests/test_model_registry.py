"""Tests for model_registry.py — registry operations and merged model detection."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "train-pipeline" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from model_registry import load_registry, save_registry, generate_model_id, get_registry_path


class TestRegistryIO:
    """Test registry load/save operations."""

    def test_load_empty(self, tmp_path):
        assert load_registry(str(tmp_path)) == []

    def test_load_invalid_json(self, tmp_path):
        (tmp_path / "model_registry.json").write_text("not json")
        assert load_registry(str(tmp_path)) == []

    def test_save_and_load(self, tmp_path):
        entries = [{"id": "test-1", "model_file": "test.pt"}]
        save_registry(str(tmp_path), entries)
        loaded = load_registry(str(tmp_path))
        assert loaded == entries

    def test_save_creates_directory(self, tmp_path):
        nested = tmp_path / "sub" / "dir"
        save_registry(str(nested), [{"id": "test"}])
        assert (nested / "model_registry.json").exists()


class TestGenerateModelId:
    """Test model ID generation."""

    def test_basic_id(self):
        from datetime import datetime
        ts = datetime(2026, 2, 7, 3, 53)
        result = generate_model_id("raw_data", "gamma1_gamma2", ts, set())
        assert result == "raw_data-gamma1_gamma2-20260207-0353"

    def test_dedup_id(self):
        from datetime import datetime
        ts = datetime(2026, 2, 7, 3, 53)
        existing = {"raw_data-gamma1-20260207-0353"}
        result = generate_model_id("raw_data", "gamma1", ts, existing)
        assert result == "raw_data-gamma1-20260207-0353-2"

    def test_dedup_id_multiple(self):
        from datetime import datetime
        ts = datetime(2026, 2, 7, 3, 53)
        existing = {
            "raw_data-gamma1-20260207-0353",
            "raw_data-gamma1-20260207-0353-2",
        }
        result = generate_model_id("raw_data", "gamma1", ts, existing)
        assert result == "raw_data-gamma1-20260207-0353-3"


class TestRegisterMergedModel:
    """Test register command with merged model pipeline state."""

    def _make_pipeline_state(self, tmp_path, merged=False):
        """Create a pipeline state file."""
        outputs = {
            "model_files": {
                "gamma1": str(tmp_path / "gamma1_stackingregmodel.pt"),
                "gamma2": str(tmp_path / "gamma2_stackingregmodel.pt"),
            },
            "refitted_model_files": {
                "gamma1": str(tmp_path / "gamma1_refitted_stackingregmodel.pt"),
                "gamma2": str(tmp_path / "gamma2_refitted_stackingregmodel.pt"),
            },
            "train_info": {
                "gamma1": str(tmp_path / "gamma1_train_info.json"),
                "gamma2": str(tmp_path / "gamma2_train_info.json"),
            },
        }
        if merged:
            outputs["merged_model_file"] = str(tmp_path / "merged_stackingregmodel.pt")
            outputs["merged_refitted_model_file"] = str(tmp_path / "merged_refitted_stackingregmodel.pt")

        state = {
            "pipeline_version": "1.0",
            "current_step": 8,
            "steps_completed": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "pipeline_complete": True,
            "config": {
                "data_file": "raw_data.csv",
                "target_properties": ["gamma1", "gamma2"],
                "task_type": "regression",
                "feature_keys": ["Bottleneck"],
                "computational_load": "cheap",
                "output_folder": str(tmp_path),
            },
            "outputs": outputs,
            "metrics": {"gamma1": {"r2": 0.85}, "gamma2": {"r2": 0.90}},
        }
        state_path = tmp_path / "pipeline_state.json"
        state_path.write_text(json.dumps(state, indent=2))
        return state_path

    def test_register_individual_model(self, tmp_path):
        state_path = self._make_pipeline_state(tmp_path, merged=False)
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "model_registry.py"),
             "register", "--pipeline-state", str(state_path)],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        output = json.loads(result.stdout.strip())
        assert output["status"] == "registered"

        # Verify model_format
        registry = load_registry(str(tmp_path))
        assert len(registry) == 1
        assert registry[0]["model_format"] == "individual"
        assert isinstance(registry[0]["model_file"], list)

    def test_register_merged_model(self, tmp_path):
        state_path = self._make_pipeline_state(tmp_path, merged=True)
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "model_registry.py"),
             "register", "--pipeline-state", str(state_path)],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        output = json.loads(result.stdout.strip())
        assert output["status"] == "registered"

        # Verify model_format
        registry = load_registry(str(tmp_path))
        assert len(registry) == 1
        assert registry[0]["model_format"] == "merged"
        assert isinstance(registry[0]["model_file"], str)
        assert "merged_refitted" in registry[0]["model_file"]

    def test_register_idempotency(self, tmp_path):
        state_path = self._make_pipeline_state(tmp_path, merged=True)
        # Register twice
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / "model_registry.py"),
                 "register", "--pipeline-state", str(state_path)],
                capture_output=True, text=True
            )
            assert result.returncode == 0

        output = json.loads(result.stdout.strip())
        assert output["status"] == "skipped"
        registry = load_registry(str(tmp_path))
        assert len(registry) == 1  # Not duplicated

    def test_merged_prefers_refitted(self, tmp_path):
        """When both merged_model_file and merged_refitted_model_file exist,
        merged_refitted_model_file should be preferred."""
        state_path = self._make_pipeline_state(tmp_path, merged=True)
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "model_registry.py"),
             "register", "--pipeline-state", str(state_path)],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        registry = load_registry(str(tmp_path))
        assert "refitted" in registry[0]["model_file"]
        assert registry[0]["is_refitted"] is True


class TestListModels:
    """Test list command."""

    def test_list_empty(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "model_registry.py"),
             "list", "--directory", str(tmp_path)],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "No models registered" in result.stdout

    def test_list_json_empty(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "model_registry.py"),
             "list", "--format", "json", "--directory", str(tmp_path)],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert json.loads(result.stdout.strip()) == []
