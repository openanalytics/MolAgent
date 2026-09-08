"""Tests for the TrainingConfig mirror in app/backend/routes/train.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.routes.train import TrainingConfig


def test_training_config_accepts_and_forwards_sdf_and_protein_folder():
    cfg = TrainingConfig(
        csv_file="ds_abc123",
        smiles_column="original_smiles",
        properties=["pChEMBL"],
        task="Regression",
        sdf_file="/abs/path/Selected_dockings.sdf",
        protein_folder="/abs/path/pdbs",
    )
    dumped = cfg.model_dump(exclude_none=True)
    assert dumped["sdf_file"] == "/abs/path/Selected_dockings.sdf"
    assert dumped["protein_folder"] == "/abs/path/pdbs"


def test_training_config_omits_sdf_and_protein_folder_when_unset():
    cfg = TrainingConfig(
        csv_file="ds_abc123",
        smiles_column="smiles",
        properties=["logP"],
        task="Regression",
    )
    dumped = cfg.model_dump(exclude_none=True)
    assert "sdf_file" not in dumped
    assert "protein_folder" not in dumped
