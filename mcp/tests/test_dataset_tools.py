"""Integration tests for dataset MCP tools + purge_stale.

Run with:
    uv run pytest mcp/tests/test_dataset_tools.py -v

Uses FastMCP in-process client (no network). Tests run in local/stdio mode
so caller is None → treated as admin with owner_id="__local__".
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def isolate_env(tmp_path, monkeypatch):
    """Point output root at tmp_path for each test."""
    monkeypatch.setenv("MOLAGENT_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.delenv("MOLAGENT_AUTH_REQUIRED", raising=False)
    monkeypatch.delenv("PHARMAOS_MOLAGENT_ROOT", raising=False)
    monkeypatch.delenv("MOLAGENT_REGISTRY_PATH", raising=False)


@pytest.fixture
def server_mcp():
    from server import mcp
    return mcp


def make_csv_b64(rows=10) -> tuple[str, str]:
    """Generate a simple CSV and return (filename, base64 content)."""
    lines = ["smiles,logP"]
    for i in range(rows):
        lines.append(f"CCO{i},{1.5 + i * 0.1:.2f}")
    content = "\n".join(lines)
    return "test_data.csv", base64.b64encode(content.encode()).decode()


# ── upload_dataset ─────────────────────────────────────────────────────────


def test_upload_dataset(server_mcp, tmp_path):
    from fastmcp import Client

    filename, b64 = make_csv_b64()

    async def _run():
        async with Client(server_mcp) as client:
            result = await client.call_tool("upload_dataset", {
                "filename": filename,
                "file_content_b64": b64,
            })
            return result.data

    data = asyncio.run(_run())
    assert "dataset_id" in data
    assert data["dataset_id"].startswith("ds_")
    assert data["filename"] == "test_data.csv"
    assert data["row_count"] == 10
    assert data["columns"] == ["smiles", "logP"]
    assert data["size_bytes"] > 0

    # File should exist on disk
    uploaded_file = tmp_path / "uploads" / "__local__" / "test_data.csv"
    assert uploaded_file.exists()


def test_upload_identical_content_returns_same_id(server_mcp, tmp_path):
    """Scenario 2: same bytes → same dataset_id, no duplicate entry."""
    from fastmcp import Client

    filename, b64 = make_csv_b64(5)

    async def _run():
        async with Client(server_mcp) as client:
            r1 = await client.call_tool("upload_dataset", {"filename": filename, "file_content_b64": b64})
            r2 = await client.call_tool("upload_dataset", {"filename": filename, "file_content_b64": b64})
            return r1.data, r2.data

    d1, d2 = asyncio.run(_run())
    assert d1["dataset_id"] == d2["dataset_id"]
    assert d1["filename"] == d2["filename"]


def test_upload_same_name_different_content_upserts(server_mcp, tmp_path):
    """Scenario 1: same filename, different content → same id, updated metadata."""
    from fastmcp import Client

    filename, b64_v1 = make_csv_b64(5)
    _, b64_v2 = make_csv_b64(10)  # different row count → different bytes

    async def _run():
        async with Client(server_mcp) as client:
            r1 = await client.call_tool("upload_dataset", {"filename": filename, "file_content_b64": b64_v1})
            r2 = await client.call_tool("upload_dataset", {"filename": filename, "file_content_b64": b64_v2})
            listed = await client.call_tool("list_datasets", {})
            return r1.data, r2.data, listed.data

    d1, d2, listed = asyncio.run(_run())
    assert d1["dataset_id"] == d2["dataset_id"]
    assert d2["row_count"] == 10
    # Only one entry in the registry for this filename
    matching = [ds for ds in listed["datasets"] if ds["filename"] == filename]
    assert len(matching) == 1


def test_upload_invalid_base64(server_mcp):
    from fastmcp import Client

    async def _run():
        async with Client(server_mcp) as client:
            result = await client.call_tool("upload_dataset", {
                "filename": "bad.csv",
                "file_content_b64": "not-valid-base64!!!",
            })
            return result

    with pytest.raises(Exception):
        asyncio.run(_run())


# ── list_datasets ──────────────────────────────────────────────────────────


def test_list_datasets_empty(server_mcp):
    from fastmcp import Client

    async def _run():
        async with Client(server_mcp) as client:
            result = await client.call_tool("list_datasets", {})
            return result.data

    data = asyncio.run(_run())
    assert data["datasets"] == []


def test_list_datasets_after_upload(server_mcp):
    from fastmcp import Client

    filename, b64 = make_csv_b64()

    async def _run():
        async with Client(server_mcp) as client:
            await client.call_tool("upload_dataset", {"filename": filename, "file_content_b64": b64})
            result = await client.call_tool("list_datasets", {})
            return result.data

    data = asyncio.run(_run())
    datasets = data["datasets"]
    assert len(datasets) == 1
    assert datasets[0]["filename"] == "test_data.csv"
    assert "uploaded_at" in datasets[0]
    assert "last_used" in datasets[0]


def test_list_datasets_reports_type_csv_for_plain_upload(server_mcp):
    from fastmcp import Client

    filename, b64 = make_csv_b64()

    async def _run():
        async with Client(server_mcp) as client:
            await client.call_tool("upload_dataset", {"filename": filename, "file_content_b64": b64})
            result = await client.call_tool("list_datasets", {})
            return result.data

    data = asyncio.run(_run())
    assert data["datasets"][0]["type"] == "csv"


# ── delete_dataset ─────────────────────────────────────────────────────────


def test_delete_dataset(server_mcp, tmp_path):
    from fastmcp import Client

    filename, b64 = make_csv_b64()

    async def _run():
        async with Client(server_mcp) as client:
            upload = await client.call_tool("upload_dataset", {"filename": filename, "file_content_b64": b64})
            ds_id = upload.data["dataset_id"]
            delete = await client.call_tool("delete_dataset", {"dataset_id": ds_id})
            remaining = await client.call_tool("list_datasets", {})
            return delete.data, remaining.data

    del_result, list_result = asyncio.run(_run())
    assert del_result["status"] == "deleted"
    assert list_result["datasets"] == []

    # File should be gone
    uploaded_file = tmp_path / "uploads" / "__local__" / "test_data.csv"
    assert not uploaded_file.exists()


def test_delete_nonexistent(server_mcp):
    from fastmcp import Client

    async def _run():
        async with Client(server_mcp) as client:
            await client.call_tool("delete_dataset", {"dataset_id": "ds_nonexistent"})

    with pytest.raises(Exception):
        asyncio.run(_run())


# ── start_training_session with dataset_id ─────────────────────────────────


def test_start_session_with_dataset_id(server_mcp):
    from fastmcp import Client

    filename, b64 = make_csv_b64(30)

    async def _run():
        async with Client(server_mcp) as client:
            upload = await client.call_tool("upload_dataset", {"filename": filename, "file_content_b64": b64})
            ds_id = upload.data["dataset_id"]
            session = await client.call_tool("start_training_session", {"dataset_id": ds_id})
            return session.data

    data = asyncio.run(_run())
    assert "session_id" in data
    assert "detected" in data


def test_start_session_no_args(server_mcp):
    from fastmcp import Client

    async def _run():
        async with Client(server_mcp) as client:
            await client.call_tool("start_training_session", {})

    with pytest.raises(Exception):
        asyncio.run(_run())


# ── purge_stale ────────────────────────────────────────────────────────────


def test_purge_stale_dry_run(server_mcp, tmp_path):
    from fastmcp import Client
    from _data_registry import load_json_list, atomic_write_json, data_registry_path

    # Upload then backdate the last_used
    filename, b64 = make_csv_b64()

    async def _upload():
        async with Client(server_mcp) as client:
            return (await client.call_tool("upload_dataset", {"filename": filename, "file_content_b64": b64})).data

    upload_data = asyncio.run(_upload())

    # Backdate last_used to 60 days ago
    reg_path = data_registry_path()
    entries = load_json_list(reg_path)
    entries[0]["last_used"] = "2026-03-01T00:00:00"
    atomic_write_json(reg_path, entries)

    async def _purge():
        async with Client(server_mcp) as client:
            return (await client.call_tool("admin_manage", {
                "action": "purge_stale",
                "max_age_days": 30,
            })).data

    result = asyncio.run(_purge())
    assert result["dry_run"] is True
    assert result["total_datasets"] == 1
    assert result["datasets_to_purge"][0]["id"] == upload_data["dataset_id"]


def test_purge_stale_force(server_mcp, tmp_path):
    from fastmcp import Client
    from _data_registry import load_json_list, atomic_write_json, data_registry_path

    filename, b64 = make_csv_b64()

    async def _upload():
        async with Client(server_mcp) as client:
            return (await client.call_tool("upload_dataset", {"filename": filename, "file_content_b64": b64})).data

    upload_data = asyncio.run(_upload())

    # Backdate
    reg_path = data_registry_path()
    entries = load_json_list(reg_path)
    entries[0]["last_used"] = "2026-03-01T00:00:00"
    atomic_write_json(reg_path, entries)

    async def _purge():
        async with Client(server_mcp) as client:
            return (await client.call_tool("admin_manage", {
                "action": "purge_stale",
                "max_age_days": 30,
                "force": True,
            })).data

    result = asyncio.run(_purge())
    assert result["dry_run"] is False
    assert upload_data["dataset_id"] in result["purged_datasets"]

    # Registry should be empty
    assert load_json_list(reg_path) == []

    # File should be deleted
    uploaded = tmp_path / "uploads" / "__local__" / "test_data.csv"
    assert not uploaded.exists()


def test_purge_stale_keeps_recent(server_mcp, tmp_path):
    from fastmcp import Client
    from _data_registry import load_json_list, data_registry_path

    filename, b64 = make_csv_b64()

    async def _run():
        async with Client(server_mcp) as client:
            await client.call_tool("upload_dataset", {"filename": filename, "file_content_b64": b64})
            # Don't backdate — should not be purged
            result = await client.call_tool("admin_manage", {
                "action": "purge_stale",
                "max_age_days": 30,
                "force": True,
            })
            return result.data

    result = asyncio.run(_run())
    assert result["purged_datasets"] == []
    assert len(load_json_list(data_registry_path())) == 1


def test_purge_stale_deletes_orphan_model_file(server_mcp, tmp_path):
    """A stale model entry with no run_folder must have its model_file deleted."""
    from fastmcp import Client
    from _data_registry import atomic_write_json, load_json_list

    # Model with an individual model_file path and NO run_folder
    model_pt = tmp_path / "orphan_stackingregmodel.pt"
    model_pt.write_bytes(b"fake model bytes")
    model_reg = tmp_path / "model_registry.json"
    atomic_write_json(model_reg, [{
        "id": "orphan-model",
        "model_file": str(model_pt),
        "last_used": "2026-03-01T00:00:00",  # stale
        "created_at": "2026-03-01T00:00:00",
    }])

    async def _purge():
        async with Client(server_mcp) as client:
            return (await client.call_tool("admin_manage", {
                "action": "purge_stale",
                "max_age_days": 30,
                "force": True,
            })).data

    result = asyncio.run(_purge())
    assert "orphan-model" in result["purged_models"]
    assert not model_pt.exists(), "orphan model file should have been deleted"
    assert load_json_list(model_reg) == []


# ── purge_orphans ─────────────────────────────────────────────────────────────


def test_purge_orphans_dry_run(server_mcp, tmp_path):
    """Orphaned folders (not in registry) are reported in dry-run."""
    from fastmcp import Client

    # Create a folder that looks like a failed run (not in any registry)
    orphan_folder = tmp_path / "dataset-prop1-20260101_0000"
    orphan_folder.mkdir()
    (orphan_folder / "pipeline_state.json").write_text("{}")
    # Backdate mtime to 10 days ago

    old_mtime = time.time() - 10 * 86400
    os.utime(orphan_folder, (old_mtime, old_mtime))

    async def _purge():
        async with Client(server_mcp) as client:
            return (await client.call_tool("admin_manage", {
                "action": "purge_orphans",
                "max_age_days": 7,
            })).data

    result = asyncio.run(_purge())
    assert result["dry_run"] is True
    assert result["total"] == 1
    assert str(orphan_folder) in result["orphaned_folders"]
    # Folder still exists (dry-run)
    assert orphan_folder.exists()


def test_purge_orphans_force(server_mcp, tmp_path):
    """Orphaned folders are deleted in force mode."""
    from fastmcp import Client


    orphan_folder = tmp_path / "dataset-prop1-20260101_0000"
    orphan_folder.mkdir()
    (orphan_folder / "model.pt").write_bytes(b"fake")
    old_mtime = time.time() - 10 * 86400
    os.utime(orphan_folder, (old_mtime, old_mtime))

    async def _purge():
        async with Client(server_mcp) as client:
            return (await client.call_tool("admin_manage", {
                "action": "purge_orphans",
                "max_age_days": 7,
                "force": True,
            })).data

    result = asyncio.run(_purge())
    assert result["dry_run"] is False
    assert str(orphan_folder) in result["purged_folders"]
    assert not orphan_folder.exists()


def test_purge_orphans_keeps_registered(server_mcp, tmp_path):
    """Folders referenced by the model registry are not purged."""
    from fastmcp import Client
    from _data_registry import atomic_write_json


    # Create a run folder that IS registered
    run_folder = tmp_path / "dataset-prop1-20260201_0000"
    run_folder.mkdir()
    (run_folder / "model.pt").write_bytes(b"fake")
    old_mtime = time.time() - 30 * 86400
    os.utime(run_folder, (old_mtime, old_mtime))

    # Register it
    model_reg = tmp_path / "model_registry.json"
    atomic_write_json(model_reg, [{
        "id": "registered-model",
        "run_folder": str(run_folder),
        "last_used": "2026-06-01T00:00:00",
    }])

    async def _purge():
        async with Client(server_mcp) as client:
            return (await client.call_tool("admin_manage", {
                "action": "purge_orphans",
                "max_age_days": 7,
                "force": True,
            })).data

    result = asyncio.run(_purge())
    assert result["purged_folders"] == []
    assert run_folder.exists()


def test_purge_orphans_keeps_recent(server_mcp, tmp_path):
    """Recent orphaned folders (newer than max_age_days) are kept."""
    from fastmcp import Client

    # Create a folder with current mtime (not old enough to purge)
    recent_folder = tmp_path / "dataset-prop1-20260601_0000"
    recent_folder.mkdir()
    (recent_folder / "pipeline_state.json").write_text("{}")

    async def _purge():
        async with Client(server_mcp) as client:
            return (await client.call_tool("admin_manage", {
                "action": "purge_orphans",
                "max_age_days": 7,
                "force": True,
            })).data

    result = asyncio.run(_purge())
    assert result["purged_folders"] == []
    assert recent_folder.exists()
