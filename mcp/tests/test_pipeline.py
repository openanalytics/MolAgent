"""Unit tests for _pipeline.py.

Run with:
    uv run pytest mcp/tests/test_pipeline.py -v

Uses a synthetic 60-row SMILES+property CSV so no real data files are needed.
Tests use computational_load="free" to keep runtimes short.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Allow imports from mcp/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from _config import TrainingConfig, TrainingResult
from _pipeline import run_full_pipeline, _scripts_dir


@pytest.fixture(autouse=True)
def _output_root_to_tmp(tmp_path, monkeypatch):
    """Point MOLAGENT_OUTPUT_ROOT at tmp_path.

    These tests pass output_folder under tmp_path; the path-containment guard
    (_under_output_root) rejects anything outside the configured output root.
    """
    monkeypatch.setenv("MOLAGENT_OUTPUT_ROOT", str(tmp_path))


# ── Synthetic data ─────────────────────────────────────────────────────────────

SYNTHETIC_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # testosterone
    "CN1CCC[C@H]1c2cccnc2",  # nicotine
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",  # pyrene
    "O=C(O)c1ccc(N)cc1",  # 4-aminobenzoic acid
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "OC(=O)c1ccccc1O",  # salicylic acid
    "c1ccc(cc1)c2ccncc2",  # 4-phenylpyridine
    "CC(=O)c1ccccc1",  # acetophenone
    "Cc1ccc(cc1)S(=O)(=O)N",  # toluenesulfonamide
    "CCOc1ccccc1",  # ethoxybenzene
]

# Repeat to get 60 rows and add numeric property
import random as _random
_random.seed(42)


def make_synthetic_csv(tmp_path: Path, n: int = 60) -> Path:
    """Write a synthetic CSV with SMILES + two numeric properties."""
    rows = ["smiles,logP,activity"]
    for i in range(n):
        smi = SYNTHETIC_SMILES[i % len(SYNTHETIC_SMILES)]
        logp = round(1.5 + _random.gauss(0, 0.8), 3)
        activity = round(abs(_random.gauss(5, 1.5)), 3)
        rows.append(f"{smi},{logp},{activity}")
    csv_path = tmp_path / "synthetic_data.csv"
    csv_path.write_text("\n".join(rows))
    return csv_path


def make_binary_clf_csv(tmp_path: Path, n: int = 60) -> Path:
    """Write a synthetic CSV for binary classification."""
    rows = ["smiles,active"]
    for i in range(n):
        smi = SYNTHETIC_SMILES[i % len(SYNTHETIC_SMILES)]
        active = i % 2  # balanced 0/1
        rows.append(f"{smi},{active}")
    csv_path = tmp_path / "clf_data.csv"
    csv_path.write_text("\n".join(rows))
    return csv_path


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_detect_script_runs(tmp_path):
    """detect_dataset.py should run and return valid JSON with smiles_column."""
    import subprocess
    csv_path = make_synthetic_csv(tmp_path)
    detect_script = _scripts_dir() / "detect_dataset.py"
    if not detect_script.exists():
        pytest.skip(f"detect_dataset.py not found at {detect_script}")

    result = subprocess.run(
        ["uv", "run", "--active", str(detect_script), "--csv-file", str(csv_path)],
        capture_output=True,
        text=True,
        cwd=str(_scripts_dir()),
        env={**os.environ, "PYTHONPATH": str(_scripts_dir())},
    )
    assert result.returncode == 0, f"detect_dataset failed: {result.stderr}"
    detection = json.loads(result.stdout)
    assert "smiles_column" in detection
    assert detection["smiles_column"] == "smiles"
    assert len(detection.get("targets", [])) >= 1


def test_full_pipeline_regression(tmp_path):
    """Full regression pipeline returns valid TrainingResult with model + dashboard."""
    csv_path = make_synthetic_csv(tmp_path)
    config = TrainingConfig(
        csv_file=str(csv_path),
        smiles_column="smiles",
        properties=["logP"],
        task="Regression",
        computational_load="free",
        feature_keys=["rdkit"],
        split_strategy="stratified",
        output_folder=str(tmp_path / "run"),
        refit=False,  # skip refit for speed
    )

    result = asyncio.run(run_full_pipeline(config))

    assert isinstance(result, TrainingResult)
    assert result.run_id
    # model_id should be set (registry registered it)
    assert result.model_id, "No model_id returned — registry registration may have failed"
    # model_path should point to a real file
    assert Path(result.model_path).exists(), "model file not found on disk"
    assert Path(result.model_path).stat().st_size > 1000, "model file suspiciously small"
    # dashboard should be HTML
    assert "<html" in result.dashboard_html.lower()
    assert "plotly" in result.dashboard_html.lower() or "Plotly" in result.dashboard_html
    # metrics should have R2/RMSE/MAE for logP
    assert "logP" in result.metrics
    prop_metrics = result.metrics["logP"]
    assert "R2" in prop_metrics or "RMSE" in prop_metrics


def test_pipeline_deterministic(tmp_path):
    """Two runs with MOLAGENT_DETERMINISTIC=true produce identical model bytes."""
    csv_path = make_synthetic_csv(tmp_path)
    config = TrainingConfig(
        csv_file=str(csv_path),
        smiles_column="smiles",
        properties=["logP"],
        task="Regression",
        computational_load="free",
        feature_keys=["rdkit"],
        split_strategy="stratified",
        output_folder=str(tmp_path / "run1"),
        refit=False,
    )
    config2 = config.model_copy(update={"output_folder": str(tmp_path / "run2")})

    env_backup = os.environ.get("MOLAGENT_DETERMINISTIC")
    os.environ["MOLAGENT_DETERMINISTIC"] = "true"
    try:
        r1 = asyncio.run(run_full_pipeline(config))
        r2 = asyncio.run(run_full_pipeline(config2))
    finally:
        if env_backup is None:
            os.environ.pop("MOLAGENT_DETERMINISTIC", None)
        else:
            os.environ["MOLAGENT_DETERMINISTIC"] = env_backup

    # Both runs should produce the same model bytes on disk
    assert Path(r1.model_path).read_bytes() == Path(r2.model_path).read_bytes(), \
        "Deterministic runs produced different models"


def test_pipeline_classification(tmp_path):
    """Classification pipeline completes and returns accuracy metric."""
    csv_path = make_binary_clf_csv(tmp_path)
    config = TrainingConfig(
        csv_file=str(csv_path),
        smiles_column="smiles",
        properties=["active"],
        task="Classification",
        categorical=True,
        computational_load="free",
        feature_keys=["rdkit"],
        split_strategy="stratified",
        output_folder=str(tmp_path / "clf_run"),
        refit=False,
    )

    result = asyncio.run(run_full_pipeline(config))

    assert isinstance(result, TrainingResult)
    assert "active" in result.metrics
    assert "Accuracy" in result.metrics["active"]


def test_pipeline_multi_property_merges(tmp_path):
    """Multi-property pipeline produces a merged model file."""
    csv_path = make_synthetic_csv(tmp_path)
    config = TrainingConfig(
        csv_file=str(csv_path),
        smiles_column="smiles",
        properties=["logP", "activity"],
        task="Regression",
        computational_load="free",
        feature_keys=["rdkit"],
        split_strategy="stratified",
        output_folder=str(tmp_path / "multi_run"),
        refit=False,
    )

    result = asyncio.run(run_full_pipeline(config))

    # merged model should be referenced in the result
    assert "merged" in result.model_filename.lower()
    assert Path(result.model_path).exists()
    assert len(result.metrics) == 2


# ── _build_train_args ────────────────────────────────────────────────────────


def test_build_train_args_includes_sdf_and_protein_folder(tmp_path):
    """Finding 1: sdf_file/protein_folder must reach the train script's argv,
    or prolif/AffGraph silently run with no receptor structures available."""
    from _pipeline import _build_train_args

    config = TrainingConfig(
        csv_file="dummy.csv",
        smiles_column="smiles",
        properties=["pChEMBL"],
        task="Regression",
        feature_keys=["prolif", "AffGraph"],
        sdf_file="/abs/path/Selected_dockings.sdf",
        protein_folder="/abs/path/pdbs",
    )
    split_csv = tmp_path / "split.csv"
    run_folder = tmp_path / "run"

    _, args = _build_train_args(config, split_csv, run_folder)

    assert "--sdf-file" in args
    assert args[args.index("--sdf-file") + 1] == "/abs/path/Selected_dockings.sdf"
    assert "--protein-folder" in args
    assert args[args.index("--protein-folder") + 1] == "/abs/path/pdbs"


def test_build_train_args_omits_sdf_and_protein_folder_when_unset(tmp_path):
    """A plain CSV run (no 3D features) must not pass --sdf-file/--protein-folder at all."""
    from _pipeline import _build_train_args

    config = TrainingConfig(
        csv_file="dummy.csv",
        smiles_column="smiles",
        properties=["logP"],
        task="Regression",
        feature_keys=["rdkit"],
    )
    split_csv = tmp_path / "split.csv"
    run_folder = tmp_path / "run"

    _, args = _build_train_args(config, split_csv, run_folder)

    assert "--sdf-file" not in args
    assert "--protein-folder" not in args
