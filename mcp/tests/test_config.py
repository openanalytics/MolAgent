"""Tests for mcp/_config.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from _config import TrainingConfig


def test_training_config_has_optional_sdf_and_protein_folder_fields():
    config = TrainingConfig(
        csv_file="data.csv",
        smiles_column="smiles",
        properties=["pChEMBL"],
        task="Regression",
    )
    assert config.sdf_file is None
    assert config.protein_folder is None


def test_training_config_accepts_sdf_and_protein_folder_values():
    config = TrainingConfig(
        csv_file="data.csv",
        smiles_column="smiles",
        properties=["pChEMBL"],
        task="Regression",
        sdf_file="/abs/path/Selected_dockings.sdf",
        protein_folder="/abs/path/pdbs",
    )
    assert config.sdf_file == "/abs/path/Selected_dockings.sdf"
    assert config.protein_folder == "/abs/path/pdbs"
