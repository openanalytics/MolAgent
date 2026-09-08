"""Tests for the 3D structure upload tool."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import rdkit  # noqa: F401
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False


def _make_minimal_sdf() -> bytes:
    """Create a minimal valid SDF with one molecule, pdb prop, and a property."""
    return b"""\n     RDKit          3D\n\n  6  6  0  0  0  0  0  0  0  0999 V2000\n    1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    0.7500    1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -0.7500    1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -0.7500   -1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    0.7500   -1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n  1  2  2  0\n  2  3  1  0\n  3  4  2  0\n  4  5  1  0\n  5  6  2  0\n  6  1  1  0\nM  END\n> <pdb>\n2E2B\n\n> <pChEMBL>\n6.5\n\n$$$$\n"""


def _make_minimal_pdb() -> bytes:
    """Create a minimal PDB file."""
    return b"ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00  0.00           N\nEND\n"


def _mol_block(pdb_prop: str = "2E2B", extra_props: str = "") -> str:
    """One SDF molecule record (without the trailing $$$$), with an optional
    'pdb' property and arbitrary extra SDF property blocks appended."""
    body = (
        "\n     RDKit          3D\n\n"
        "  6  6  0  0  0  0  0  0  0  0999 V2000\n"
        "    1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    0.7500    1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "   -0.7500    1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "   -1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "   -0.7500   -1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    0.7500   -1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "  1  2  2  0\n  2  3  1  0\n  3  4  2  0\n  4  5  1  0\n  5  6  2  0\n  6  1  1  0\n"
        "M  END\n"
    )
    if pdb_prop:
        body += f"> <pdb>\n{pdb_prop}\n\n"
    body += extra_props
    return body


def _make_sdf_with_missing_property() -> bytes:
    """Two molecules: first has a valid pChEMBL value, second is missing it entirely."""
    mol1 = _mol_block("2E2B", "> <pChEMBL>\n6.5\n\n")
    mol2 = _mol_block("2E2B")  # no pChEMBL property at all
    return (mol1 + "$$$$\n" + mol2 + "$$$$\n").encode()


def _make_sdf_with_non_numeric_property() -> bytes:
    """Two molecules: first valid, second has a non-numeric pChEMBL value."""
    mol1 = _mol_block("2E2B", "> <pChEMBL>\n6.5\n\n")
    mol2 = _mol_block("2E2B", "> <pChEMBL>\nN/A\n\n")
    return (mol1 + "$$$$\n" + mol2 + "$$$$\n").encode()


class TestUpload3dStructure:
    """Unit tests for _3d_tools module (no MCP server needed)."""

    @pytest.mark.skipif(not _HAS_RDKIT, reason="rdkit not installed")
    def test_extract_3d_data_produces_csv(self, tmp_path):
        from _3d_tools import extract_3d_data

        sdf_path = tmp_path / "ligands.sdf"
        sdf_path.write_bytes(_make_minimal_sdf())
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        (pdb_dir / "2E2B_receptor.pdb").write_bytes(_make_minimal_pdb())

        result = extract_3d_data(
            sdf_file=str(sdf_path),
            property_key="pChEMBL",
            data_dir=str(tmp_path / "output"),
            file_nm="data.csv",
        )

        assert "data_file" in result
        csv_path = Path(result["data_file"])
        assert csv_path.exists()
        import pandas as pd
        df = pd.read_csv(csv_path)
        assert "original_smiles" in df.columns
        assert "pChEMBL" in df.columns
        assert "pdb" in df.columns
        assert len(df) == 1

    @pytest.mark.skipif(not _HAS_RDKIT, reason="rdkit not installed")
    def test_extract_3d_data_skips_missing_property(self, tmp_path):
        """A molecule missing property_key must be skipped, not crash with float('')."""
        from _3d_tools import extract_3d_data

        sdf_path = tmp_path / "ligands.sdf"
        sdf_path.write_bytes(_make_sdf_with_missing_property())

        result = extract_3d_data(
            sdf_file=str(sdf_path),
            property_key="pChEMBL",
            data_dir=str(tmp_path / "output"),
            file_nm="data.csv",
        )

        assert result["mol_count"] == 1
        import pandas as pd
        df = pd.read_csv(result["data_file"])
        assert len(df) == 1
        assert df["pChEMBL"].iloc[0] == 6.5

    @pytest.mark.skipif(not _HAS_RDKIT, reason="rdkit not installed")
    def test_extract_3d_data_skips_non_numeric_property(self, tmp_path):
        """A molecule with a non-numeric property value must be skipped, not crash."""
        from _3d_tools import extract_3d_data

        sdf_path = tmp_path / "ligands.sdf"
        sdf_path.write_bytes(_make_sdf_with_non_numeric_property())

        result = extract_3d_data(
            sdf_file=str(sdf_path),
            property_key="pChEMBL",
            data_dir=str(tmp_path / "output"),
            file_nm="data.csv",
        )

        assert result["mol_count"] == 1
        import pandas as pd
        df = pd.read_csv(result["data_file"])
        assert len(df) == 1
        assert df["pChEMBL"].iloc[0] == 6.5

    def test_validate_structure_upload_inputs(self, tmp_path):
        from _3d_tools import _validate_target_name, _validate_pdb_files_json

        assert _validate_target_name("ABL") == "ABL"
        assert _validate_target_name("my_target_1") == "my_target_1"

        with pytest.raises(ValueError):
            _validate_target_name("../escape")
        with pytest.raises(ValueError):
            _validate_target_name("")

        pdb_json = json.dumps([
            {"filename": "2E2B_receptor.pdb", "content_b64": base64.b64encode(b"ATOM").decode()}
        ])
        result = _validate_pdb_files_json(pdb_json)
        assert len(result) == 1
        assert result[0]["filename"] == "2E2B_receptor.pdb"

        with pytest.raises(ValueError):
            _validate_pdb_files_json("not json")

    def test_store_3d_structure(self, tmp_path):
        from _3d_tools import _store_3d_structure

        sdf_bytes = _make_minimal_sdf()
        pdb_entries = [
            {"filename": "2E2B_receptor.pdb", "content_b64": base64.b64encode(_make_minimal_pdb()).decode()}
        ]

        result = _store_3d_structure(
            output_root=tmp_path,
            owner_id="local",
            target_name="ABL",
            sdf_bytes=sdf_bytes,
            pdb_entries=pdb_entries,
            ligands_csv_bytes=None,
        )

        assert (tmp_path / "structures" / "local" / "ABL" / "Selected_dockings.sdf").exists()
        assert (tmp_path / "structures" / "local" / "ABL" / "pdbs" / "2E2B_receptor.pdb").exists()
        assert result["sdf_path"].endswith("Selected_dockings.sdf")
        assert result["pdb_folder"].endswith("pdbs")

    def test_store_3d_structure_reports_sdf_record_count_not_mol_count(self, tmp_path):
        """Finding 3: the raw '$$$$' count must be reported as sdf_record_count,
        not the ambiguous/misleading 'mol_count' key (validated count comes from
        extract_3d_data instead)."""
        from _3d_tools import _store_3d_structure

        sdf_bytes = _make_sdf_with_missing_property()  # 2 records, only 1 has a valid property
        pdb_entries = [
            {"filename": "2E2B_receptor.pdb", "content_b64": base64.b64encode(_make_minimal_pdb()).decode()}
        ]

        result = _store_3d_structure(
            output_root=tmp_path,
            owner_id="local",
            target_name="ABL",
            sdf_bytes=sdf_bytes,
            pdb_entries=pdb_entries,
            ligands_csv_bytes=None,
        )

        assert "mol_count" not in result
        assert result["sdf_record_count"] == 2  # raw $$$$ count, unrelated to validity

    def test_store_3d_structure_reports_total_pdb_bytes(self, tmp_path):
        """Finding 4: decoded PDB byte sizes should be tracked during the single
        decode in _store_3d_structure, so callers don't need to re-decode base64
        just to compute size_bytes."""
        from _3d_tools import _store_3d_structure

        pdb_bytes = _make_minimal_pdb()
        pdb_entries = [
            {"filename": "2E2B_receptor.pdb", "content_b64": base64.b64encode(pdb_bytes).decode()},
            {"filename": "3ABC_receptor.pdb", "content_b64": base64.b64encode(pdb_bytes).decode()},
        ]

        result = _store_3d_structure(
            output_root=tmp_path,
            owner_id="local",
            target_name="ABL",
            sdf_bytes=_make_minimal_sdf(),
            pdb_entries=pdb_entries,
            ligands_csv_bytes=None,
        )

        assert result["total_pdb_bytes"] == len(pdb_bytes) * 2


@pytest.fixture(autouse=True)
def isolate_env(tmp_path, monkeypatch):
    """Point output root + data registry at tmp_path for each integration test."""
    monkeypatch.setenv("MOLAGENT_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.delenv("MOLAGENT_AUTH_REQUIRED", raising=False)
    monkeypatch.delenv("PHARMAOS_MOLAGENT_ROOT", raising=False)
    monkeypatch.delenv("MOLAGENT_REGISTRY_PATH", raising=False)


@pytest.fixture
def server_mcp():
    from server import mcp
    return mcp


@pytest.mark.skipif(not _HAS_RDKIT, reason="rdkit not installed")
class TestUpload3dStructureIntegration:
    """Integration tests through the MCP server (upload_3d_structure + start_training_session)."""

    def test_upload_3d_structure_registers_entry_with_3d_metadata(self, server_mcp, tmp_path):
        """Finding 5: the registry entry for a freshly uploaded structure must already
        carry its 3D metadata (type/sdf_path/pdb_folder) — there is no window where it
        appears in the registry without them, since it's written in a single locked op."""
        from fastmcp import Client
        from _data_registry import load_json_list, data_registry_path

        pdb_json = json.dumps([
            {"filename": "2E2B_receptor.pdb", "content_b64": base64.b64encode(_make_minimal_pdb()).decode()}
        ])

        async def _run():
            async with Client(server_mcp) as client:
                result = await client.call_tool("upload_3d_structure", {
                    "target_name": "ABL",
                    "sdf_content_b64": base64.b64encode(_make_minimal_sdf()).decode(),
                    "pdb_files": pdb_json,
                })
                return result.data

        data = asyncio.run(_run())
        assert data["structure_id"].startswith("ds_")
        assert "sdf_record_count" in data
        assert "mol_count" not in data

        entries = load_json_list(data_registry_path())
        assert len(entries) == 1
        entry = entries[0]
        assert entry["id"] == data["structure_id"]
        assert entry["type"] == "3d_structure"
        assert entry["sdf_path"] == data["sdf_path"]
        assert entry["pdb_folder"] == data["pdb_folder"]

    def test_start_training_session_default_property_key(self, server_mcp, tmp_path):
        """Without property_key, start_training_session should fall back to pChEMBL.
        Uses dataset_id — structure_id has been removed and unified into
        dataset_id + registry-entry type detection."""
        from fastmcp import Client

        pdb_json = json.dumps([
            {"filename": "2E2B_receptor.pdb", "content_b64": base64.b64encode(_make_minimal_pdb()).decode()}
        ])

        async def _run():
            async with Client(server_mcp) as client:
                upload = await client.call_tool("upload_3d_structure", {
                    "target_name": "ABL",
                    "sdf_content_b64": base64.b64encode(_make_minimal_sdf()).decode(),
                    "pdb_files": pdb_json,
                })
                structure_id = upload.data["structure_id"]
                session = await client.call_tool("start_training_session", {
                    "dataset_id": structure_id,
                })
                return session.data

        data = asyncio.run(_run())
        assert data["structure_extraction"]["property_key"] == "pChEMBL"
        assert data["structure_extraction"]["mol_count"] == 1

    def test_start_training_session_custom_property_key(self, server_mcp, tmp_path):
        """Finding 2: property_key should let a caller extract a different SDF
        property (e.g. pIC50) instead of the hardcoded 'pChEMBL' default."""
        from fastmcp import Client

        sdf_bytes = _mol_block("2E2B", "> <pIC50>\n7.2\n\n").encode() + b"$$$$\n"
        pdb_json = json.dumps([
            {"filename": "2E2B_receptor.pdb", "content_b64": base64.b64encode(_make_minimal_pdb()).decode()}
        ])

        async def _run():
            async with Client(server_mcp) as client:
                upload = await client.call_tool("upload_3d_structure", {
                    "target_name": "ABL2",
                    "sdf_content_b64": base64.b64encode(sdf_bytes).decode(),
                    "pdb_files": pdb_json,
                })
                structure_id = upload.data["structure_id"]
                session = await client.call_tool("start_training_session", {
                    "dataset_id": structure_id,
                    "property_key": "pIC50",
                })
                return session.data

        data = asyncio.run(_run())
        assert data["structure_extraction"]["property_key"] == "pIC50"
        assert data["structure_extraction"]["mol_count"] == 1

    def test_start_training_session_exposes_3d_feature_keys_only_for_structure_dataset(self, server_mcp, tmp_path):
        """prolif/AffGraph must appear in options.feature_keys for a 3D structure
        dataset_id, and must NOT appear for a plain CSV dataset_id."""
        from fastmcp import Client

        pdb_json = json.dumps([
            {"filename": "2E2B_receptor.pdb", "content_b64": base64.b64encode(_make_minimal_pdb()).decode()}
        ])

        async def _run():
            async with Client(server_mcp) as client:
                upload = await client.call_tool("upload_3d_structure", {
                    "target_name": "ABL",
                    "sdf_content_b64": base64.b64encode(_make_minimal_sdf()).decode(),
                    "pdb_files": pdb_json,
                })
                structure_id = upload.data["structure_id"]
                structure_session = await client.call_tool("start_training_session", {
                    "dataset_id": structure_id,
                })

                csv_upload = await client.call_tool("upload_dataset", {
                    "filename": "plain.csv",
                    "file_content_b64": base64.b64encode(b"smiles,logP\nCCO,1.5\n").decode(),
                })
                csv_session = await client.call_tool("start_training_session", {
                    "dataset_id": csv_upload.data["dataset_id"],
                })
                return structure_session.data, csv_session.data

        structure_data, csv_data = asyncio.run(_run())
        assert "prolif" in structure_data["options"]["feature_keys"]
        assert "AffGraph" in structure_data["options"]["feature_keys"]
        assert "prolif" not in csv_data["options"]["feature_keys"]
        assert "AffGraph" not in csv_data["options"]["feature_keys"]
        assert structure_data["detected"]["sdf_file"], "detected.sdf_file must be set for a 3D structure session"
        assert structure_data["detected"]["protein_folder"], "detected.protein_folder must be set for a 3D structure session"
        assert csv_data["detected"]["sdf_file"] is None
        assert csv_data["detected"]["protein_folder"] is None

    def test_confirmed_config_includes_sdf_file_and_protein_folder(self, server_mcp, tmp_path):
        """answer_training_question's finalized config must carry sdf_file/protein_folder
        for a 3D structure dataset, so _pipeline.py can forward them to the train script."""
        from fastmcp import Client

        pdb_json = json.dumps([
            {"filename": "2E2B_receptor.pdb", "content_b64": base64.b64encode(_make_minimal_pdb()).decode()}
        ])

        async def _run():
            async with Client(server_mcp) as client:
                upload = await client.call_tool("upload_3d_structure", {
                    "target_name": "ABL",
                    "sdf_content_b64": base64.b64encode(_make_minimal_sdf()).decode(),
                    "pdb_files": pdb_json,
                })
                structure_id = upload.data["structure_id"]
                session = await client.call_tool("start_training_session", {
                    "dataset_id": structure_id,
                })
                session_id = session.data["session_id"]
                confirmed = await client.call_tool("answer_training_question", {
                    "session_id": session_id,
                    "confirm": True,
                })
                return confirmed.data

        data = asyncio.run(_run())
        config = data["config"]
        assert config["sdf_file"], "sdf_file missing from finalized config"
        assert config["protein_folder"], "protein_folder missing from finalized config"
        assert config["sdf_file"].endswith("Selected_dockings.sdf")
        assert config["protein_folder"].endswith("pdbs")
        assert Path(config["sdf_file"]).is_absolute()
        assert Path(config["protein_folder"]).is_absolute()

    def test_answer_training_question_rejects_3d_feature_keys_for_plain_csv_session(self, server_mcp, tmp_path):
        """Finding 1: a plain-CSV session's config has no sdf_file/protein_folder, so
        answer_training_question must reject feature_keys requiring 3D structure data
        (e.g. 'prolif') instead of silently accepting them into the config — even though
        'prolif' is present in the global list_feature_generators() registry."""
        from fastmcp import Client

        async def _run():
            async with Client(server_mcp) as client:
                csv_upload = await client.call_tool("upload_dataset", {
                    "filename": "plain.csv",
                    "file_content_b64": base64.b64encode(b"smiles,logP\nCCO,1.5\n").decode(),
                })
                csv_session = await client.call_tool("start_training_session", {
                    "dataset_id": csv_upload.data["dataset_id"],
                })
                session_id = csv_session.data["session_id"]
                answer = await client.call_tool("answer_training_question", {
                    "session_id": session_id,
                    "feature_keys": ["prolif"],
                })
                return answer.data

        data = asyncio.run(_run())
        assert data["validation_error"] is True
        assert data["config"] is None
        assert "prolif" in data["question"]

    def test_list_datasets_reports_type_3d_structure(self, server_mcp, tmp_path):
        from fastmcp import Client

        pdb_json = json.dumps([
            {"filename": "2E2B_receptor.pdb", "content_b64": base64.b64encode(_make_minimal_pdb()).decode()}
        ])

        async def _run():
            async with Client(server_mcp) as client:
                await client.call_tool("upload_3d_structure", {
                    "target_name": "ABL",
                    "sdf_content_b64": base64.b64encode(_make_minimal_sdf()).decode(),
                    "pdb_files": pdb_json,
                })
                result = await client.call_tool("list_datasets", {})
                return result.data

        data = asyncio.run(_run())
        assert data["datasets"][0]["type"] == "3d_structure"

    def test_delete_dataset_removes_3d_structure_directory(self, server_mcp, tmp_path):
        """Finding 2: delete_dataset must shutil.rmtree the whole
        structures/<owner>/<target>/ directory for a 3d_structure entry — not
        call unlink() on what is actually a directory."""
        from fastmcp import Client

        pdb_json = json.dumps([
            {"filename": "2E2B_receptor.pdb", "content_b64": base64.b64encode(_make_minimal_pdb()).decode()}
        ])

        async def _upload():
            async with Client(server_mcp) as client:
                return (await client.call_tool("upload_3d_structure", {
                    "target_name": "ABL",
                    "sdf_content_b64": base64.b64encode(_make_minimal_sdf()).decode(),
                    "pdb_files": pdb_json,
                })).data

        upload_data = asyncio.run(_upload())
        struct_dir = tmp_path / "structures" / "__local__" / "ABL"
        assert struct_dir.exists()

        # Simulate a prior extraction, so the "prepared/" output is also present.
        prepared_dir = struct_dir / "prepared"
        prepared_dir.mkdir()
        (prepared_dir / "data.csv").write_text("original_smiles,pChEMBL,pdb\n")

        async def _delete():
            async with Client(server_mcp) as client:
                return (await client.call_tool("delete_dataset", {
                    "dataset_id": upload_data["structure_id"],
                })).data

        delete_result = asyncio.run(_delete())
        assert delete_result["status"] == "deleted"
        assert not struct_dir.exists(), "3d_structure directory (SDF + PDBs + prepared/) should be fully removed"
