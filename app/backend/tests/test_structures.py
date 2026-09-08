"""Tests for POST /api/structures/upload."""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import pytest

# Tests run with cwd anywhere; resolve the `app/` directory so `backend` resolves
# as a top-level package, matching how the app is run (`uvicorn backend.main:app`
# with cwd=app/).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient

from backend.main import app
import backend.routes.structures as structures_module


@pytest.fixture
def client(monkeypatch):
    calls: dict = {}

    async def fake_call_tool(name, arguments):
        calls["name"] = name
        calls["arguments"] = arguments
        return {
            "structure_id": "ds_fake123",
            "target_name": arguments.get("target_name"),
            "sdf_path": "structures/__local__/ABL/Selected_dockings.sdf",
            "pdb_folder": "structures/__local__/ABL/pdbs",
            "sdf_record_count": 1,
            "pdb_count": 2,
        }

    monkeypatch.setattr(structures_module, "call_tool", fake_call_tool)
    with TestClient(app) as test_client:
        yield test_client, calls


def test_upload_structure_assembles_pdb_files_json(client):
    test_client, calls = client

    files = [
        ("sdf_file", ("ligands.sdf", io.BytesIO(b"fake sdf content"), "chemical/x-mdl-sdfile")),
        ("pdb_files", ("2E2B_receptor.pdb", io.BytesIO(b"ATOM fake pdb 1"), "chemical/x-pdb")),
        ("pdb_files", ("3ABC_receptor.pdb", io.BytesIO(b"ATOM fake pdb 2"), "chemical/x-pdb")),
    ]
    data = {"target_name": "ABL"}

    response = test_client.post("/api/structures/upload", data=data, files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["structure_id"] == "ds_fake123"

    assert calls["name"] == "upload_3d_structure"
    args = calls["arguments"]
    assert args["target_name"] == "ABL"
    assert "sdf_content_b64" in args
    assert base64.b64decode(args["sdf_content_b64"]) == b"fake sdf content"

    pdb_entries = json.loads(args["pdb_files"])
    assert len(pdb_entries) == 2
    filenames = {e["filename"] for e in pdb_entries}
    assert filenames == {"2E2B_receptor.pdb", "3ABC_receptor.pdb"}
    for e in pdb_entries:
        assert "content_b64" in e
        decoded = base64.b64decode(e["content_b64"])
        assert decoded.startswith(b"ATOM fake pdb")

    assert args.get("ligands_per_pdb_b64") is None


def test_upload_structure_includes_optional_ligands_csv(client):
    test_client, calls = client

    files = [
        ("sdf_file", ("ligands.sdf", io.BytesIO(b"fake sdf"), "chemical/x-mdl-sdfile")),
        ("pdb_files", ("2E2B_receptor.pdb", io.BytesIO(b"ATOM fake pdb"), "chemical/x-pdb")),
        ("ligands_csv", ("ligands.csv", io.BytesIO(b"pdb,ligand\n2E2B,lig1\n"), "text/csv")),
    ]
    data = {"target_name": "ABL"}

    response = test_client.post("/api/structures/upload", data=data, files=files)

    assert response.status_code == 200
    args = calls["arguments"]
    assert args["ligands_per_pdb_b64"] is not None
    assert base64.b64decode(args["ligands_per_pdb_b64"]) == b"pdb,ligand\n2E2B,lig1\n"
