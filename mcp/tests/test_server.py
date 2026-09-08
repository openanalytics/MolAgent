"""Integration tests for mcp/server.py using FastMCP in-process client.

Run with:
    uv run pytest mcp/tests/test_server.py -v

The FastMCP Client can connect directly to an in-process FastMCP instance
(no network, no subprocess) — ideal for fast CI testing.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import pytest

# Allow imports from mcp/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from _config import TrainingConfig

# ── Helpers ────────────────────────────────────────────────────────────────────

SYNTHETIC_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "CN1CCC[C@H]1c2cccnc2",
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",
    "O=C(O)c1ccc(N)cc1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CC(=O)Nc1ccc(O)cc1",
    "OC(=O)c1ccccc1O",
    "c1ccc(cc1)c2ccncc2",
    "CC(=O)c1ccccc1",
    "Cc1ccc(cc1)S(=O)(=O)N",
    "CCOc1ccccc1",
]

random.seed(42)


def make_synthetic_csv(tmp_path: Path, n: int = 60) -> Path:
    rows = ["smiles,logP"]
    for i in range(n):
        smi = SYNTHETIC_SMILES[i % len(SYNTHETIC_SMILES)]
        logp = round(1.5 + random.gauss(0, 0.8), 3)
        rows.append(f"{smi},{logp}")
    csv_path = tmp_path / "synthetic_data.csv"
    csv_path.write_text("\n".join(rows))
    return csv_path


def minimal_config(tmp_path: Path) -> dict:
    csv_path = make_synthetic_csv(tmp_path)
    return TrainingConfig(
        csv_file=str(csv_path.resolve()),
        smiles_column="smiles",
        properties=["logP"],
        task="Regression",
        computational_load="free",
        feature_keys=["rdkit"],
        split_strategy="stratified",
        output_folder=str(tmp_path / "run"),
        refit=False,
    ).model_dump()


# ── Test fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _output_root_to_tmp(tmp_path, monkeypatch):
    """Point MOLAGENT_OUTPUT_ROOT at tmp_path.

    These tests pass output_folder under tmp_path; the path-containment guard
    (_under_output_root) rejects anything outside the configured output root.
    """
    monkeypatch.setenv("MOLAGENT_OUTPUT_ROOT", str(tmp_path))


@pytest.fixture(scope="module")
def server_mcp():
    """Import the FastMCP app — fails fast if imports are broken."""
    from server import mcp  # noqa: PLC0415 (deferred import OK in fixture)
    return mcp


# ── Tool listing ──────────────────────────────────────────────────────────────

def test_tools_listed(server_mcp):
    """Server should expose exactly three tools."""
    from fastmcp import Client

    async def _run():
        async with Client(server_mcp) as client:
            tools = await client.list_tools()
            return [t.name for t in tools]

    tool_names = asyncio.run(_run())
    assert "start_training_session" in tool_names
    assert "answer_training_question" in tool_names
    assert "train_and_visualize" in tool_names
    assert "gather_training_config" not in tool_names


# ── Session lifecycle ─────────────────────────────────────────────────────────

def test_start_session_returns_detected_config(server_mcp, tmp_path):
    """start_training_session should return session_id, detected defaults, and options."""
    from fastmcp import Client

    csv_path = make_synthetic_csv(tmp_path)

    async def _run():
        async with Client(server_mcp) as client:
            result = await client.call_tool("start_training_session", {"csv_file": str(csv_path)})
            return result.data

    data = asyncio.run(_run())
    assert "session_id" in data
    assert "detected" in data
    assert "options" in data
    assert "question" in data
    detected = data["detected"]
    assert "smiles_column" in detected
    assert "properties" in detected
    assert "task" in detected
    assert "computational_load" in detected
    options = data["options"]
    assert "computational_load" in options
    assert "task" in options
    assert isinstance(data["session_id"], str)
    assert len(data["session_id"]) > 0


def test_start_session_fresh_each_call(server_mcp, tmp_path):
    """Two calls with the same CSV should return different session_ids."""
    from fastmcp import Client

    csv_path = make_synthetic_csv(tmp_path)

    async def _run():
        async with Client(server_mcp) as client:
            r1 = await client.call_tool("start_training_session", {"csv_file": str(csv_path)})
            r2 = await client.call_tool("start_training_session", {"csv_file": str(csv_path)})
            return r1.data["session_id"], r2.data["session_id"]

    sid1, sid2 = asyncio.run(_run())
    assert sid1 != sid2


def test_confirm_returns_config(server_mcp, tmp_path):
    """confirm=True should finalize the session and return a valid TrainingConfig."""
    from fastmcp import Client

    csv_path = make_synthetic_csv(tmp_path)

    async def _run():
        async with Client(server_mcp) as client:
            start = await client.call_tool("start_training_session", {"csv_file": str(csv_path)})
            sid = start.data["session_id"]
            answer = await client.call_tool(
                "answer_training_question",
                {"session_id": sid, "confirm": True},
            )
            return answer.data

    data = asyncio.run(_run())
    assert data["config"] is not None
    assert data["question"] is None
    # config should be valid
    cfg = TrainingConfig.model_validate(data["config"])
    assert cfg.smiles_column == "smiles"
    assert "logP" in cfg.properties


def test_override_then_confirm(server_mcp, tmp_path):
    """Overriding a field then confirming should apply the override to the returned config."""
    from fastmcp import Client

    csv_path = make_synthetic_csv(tmp_path)

    async def _run():
        async with Client(server_mcp) as client:
            start = await client.call_tool("start_training_session", {"csv_file": str(csv_path)})
            sid = start.data["session_id"]
            # apply override
            await client.call_tool(
                "answer_training_question",
                {"session_id": sid, "computational_load": "moderate"},
            )
            # confirm
            answer = await client.call_tool(
                "answer_training_question",
                {"session_id": sid, "confirm": True},
            )
            return answer.data

    data = asyncio.run(_run())
    assert data["config"]["computational_load"] == "moderate"


def test_override_and_confirm_in_one_call(server_mcp, tmp_path):
    """Passing both an override and confirm=True in one call should work."""
    from fastmcp import Client

    csv_path = make_synthetic_csv(tmp_path)

    async def _run():
        async with Client(server_mcp) as client:
            start = await client.call_tool("start_training_session", {"csv_file": str(csv_path)})
            sid = start.data["session_id"]
            answer = await client.call_tool(
                "answer_training_question",
                {"session_id": sid, "computational_load": "free", "confirm": True},
            )
            return answer.data

    data = asyncio.run(_run())
    assert data["config"]["computational_load"] == "free"
    assert data["question"] is None


def test_unknown_session_id_returns_error(server_mcp):
    """answer_training_question with an unknown session_id should raise McpError."""
    from fastmcp import Client

    async def _run():
        async with Client(server_mcp) as client:
            result = await client.call_tool(
                "answer_training_question",
                {"session_id": "00000000-0000-0000-0000-000000000000", "confirm": True},
            )
            return result

    with pytest.raises(Exception) as exc_info:
        asyncio.run(_run())
    assert "not found or expired" in str(exc_info.value).lower() or "session" in str(exc_info.value).lower()


def test_invalid_smiles_column_returns_validation_message(server_mcp, tmp_path):
    """Supplying a non-existent smiles_column should return validation_error=True, not crash."""
    from fastmcp import Client

    csv_path = make_synthetic_csv(tmp_path)

    async def _run():
        async with Client(server_mcp) as client:
            start = await client.call_tool("start_training_session", {"csv_file": str(csv_path)})
            sid = start.data["session_id"]
            answer = await client.call_tool(
                "answer_training_question",
                {"session_id": sid, "smiles_column": "nonexistent_column"},
            )
            return answer.data

    data = asyncio.run(_run())
    assert data["config"] is None
    assert data["validation_error"] is True
    assert "not found" in data["question"].lower() or "available" in data["question"].lower()


def test_invalid_property_column_returns_validation_message(server_mcp, tmp_path):
    """Supplying a non-existent property column should return validation_error=True."""
    from fastmcp import Client

    csv_path = make_synthetic_csv(tmp_path)

    async def _run():
        async with Client(server_mcp) as client:
            start = await client.call_tool("start_training_session", {"csv_file": str(csv_path)})
            sid = start.data["session_id"]
            answer = await client.call_tool(
                "answer_training_question",
                {"session_id": sid, "properties": ["does_not_exist"]},
            )
            return answer.data

    data = asyncio.run(_run())
    assert data["config"] is None
    assert data["validation_error"] is True
    assert "not found" in data["question"].lower() or "available" in data["question"].lower()


def test_invalid_column_does_not_change_config(server_mcp, tmp_path):
    """A column validation error should leave the session config unchanged and session open."""
    from fastmcp import Client

    csv_path = make_synthetic_csv(tmp_path)

    async def _run():
        async with Client(server_mcp) as client:
            start = await client.call_tool("start_training_session", {"csv_file": str(csv_path)})
            sid = start.data["session_id"]
            original_smiles = start.data["detected"]["smiles_column"]
            # bad override — session must remain open
            await client.call_tool(
                "answer_training_question",
                {"session_id": sid, "smiles_column": "bad_col"},
            )
            # confirm — should use the original smiles column, session still valid
            answer = await client.call_tool(
                "answer_training_question",
                {"session_id": sid, "confirm": True},
            )
            return answer.data, original_smiles

    data, original = asyncio.run(_run())
    assert data["config"]["smiles_column"] == original


def test_valid_override_response_has_no_validation_error(server_mcp, tmp_path):
    """A successful override should return validation_error=False."""
    from fastmcp import Client

    csv_path = make_synthetic_csv(tmp_path)

    async def _run():
        async with Client(server_mcp) as client:
            start = await client.call_tool("start_training_session", {"csv_file": str(csv_path)})
            sid = start.data["session_id"]
            answer = await client.call_tool(
                "answer_training_question",
                {"session_id": sid, "computational_load": "free"},
            )
            return answer.data

    data = asyncio.run(_run())
    assert data["validation_error"] is False
    assert data["config"] is None  # not finalized yet


# ── train_and_visualize (unchanged) ───────────────────────────────────────────

def test_train_returns_model_id(server_mcp, tmp_path):
    """train_and_visualize should return a model_id and the model file should exist."""
    from fastmcp import Client

    config = minimal_config(tmp_path)

    async def _run():
        async with Client(server_mcp) as client:
            result = await client.call_tool("train_and_visualize", {"config": config})
            return result.data

    data = asyncio.run(_run())
    assert "model_id" in data
    assert data["model_id"], "No model_id returned"
    assert Path(data["model_path"]).exists(), "model file not on disk"
    assert Path(data["model_path"]).stat().st_size > 1000, "model file suspiciously small"


def test_train_returns_dashboard_html(server_mcp, tmp_path):
    """train_and_visualize should return a non-empty HTML dashboard string."""
    from fastmcp import Client

    config = minimal_config(tmp_path)

    async def _run():
        async with Client(server_mcp) as client:
            result = await client.call_tool("train_and_visualize", {"config": config})
            return result.data

    data = asyncio.run(_run())
    assert "dashboard_html" in data
    html = data["dashboard_html"]
    assert len(html) > 500
    assert "<html" in html.lower()
    assert "plotly" in html.lower() or "Plotly" in html


def test_train_returns_metrics(server_mcp, tmp_path):
    """train_and_visualize should return a metrics dict with numeric values."""
    from fastmcp import Client

    config = minimal_config(tmp_path)

    async def _run():
        async with Client(server_mcp) as client:
            result = await client.call_tool("train_and_visualize", {"config": config})
            return result.data

    data = asyncio.run(_run())
    assert "metrics" in data
    metrics = data["metrics"]
    assert len(metrics) >= 1
    for prop_metrics in metrics.values():
        assert len(prop_metrics) >= 1
        for v in prop_metrics.values():
            assert isinstance(v, (int, float))


def test_train_invalid_config_raises(server_mcp):
    """train_and_visualize should raise McpError for an invalid config."""
    from fastmcp import Client

    async def _run():
        async with Client(server_mcp) as client:
            result = await client.call_tool("train_and_visualize", {"config": {"csv_file": "/nonexistent.csv"}})
            return result

    with pytest.raises(Exception):
        asyncio.run(_run())


def test_train_result_has_paths(server_mcp, tmp_path):
    """train_and_visualize should return absolute filesystem paths."""
    from fastmcp import Client

    config = minimal_config(tmp_path)

    async def _run():
        async with Client(server_mcp) as client:
            result = await client.call_tool("train_and_visualize", {"config": config})
            return result.data

    data = asyncio.run(_run())
    assert Path(data["model_path"]).exists(), f"model_path does not exist: {data['model_path']}"
    assert Path(data["dashboard_path"]).exists(), f"dashboard_path does not exist: {data['dashboard_path']}"
    assert Path(data["pipeline_state_path"]).exists()
