"""3D structure upload and extraction utilities.

Provides:
  - upload_3d_structure: MCP tool for remote upload of SDF + PDB bundles
  - extract_3d_data: internal utility to parse SDF into training CSV
  - Validation helpers for input sanitization
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastmcp import Context

logger = logging.getLogger(__name__)

_VALID_TARGET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _validate_target_name(name: str) -> str:
    if not _VALID_TARGET_RE.match(name):
        raise ValueError(
            f"Invalid target_name '{name}': must be 1-64 alphanumeric/underscore/dash chars, "
            "starting with a letter or digit."
        )
    if ".." in name or "/" in name or "\\" in name:
        raise ValueError(f"Invalid target_name '{name}': path traversal not allowed.")
    return name


def _validate_pdb_files_json(pdb_files_json: str) -> list[dict]:
    try:
        entries = json.loads(pdb_files_json)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"pdb_files must be valid JSON array: {exc}")

    if not isinstance(entries, list) or len(entries) == 0:
        raise ValueError("pdb_files must be a non-empty JSON array.")

    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"pdb_files[{i}] must be an object with 'filename' and 'content_b64'.")
        if "filename" not in entry or "content_b64" not in entry:
            raise ValueError(f"pdb_files[{i}] missing 'filename' or 'content_b64'.")
        fn = Path(entry["filename"]).name
        if not fn.endswith(".pdb"):
            raise ValueError(f"pdb_files[{i}] filename must end with .pdb, got: {fn}")
        entry["filename"] = fn

    return entries


def _store_3d_structure(
    output_root: Path,
    owner_id: str,
    target_name: str,
    sdf_bytes: bytes,
    pdb_entries: list[dict],
    ligands_csv_bytes: Optional[bytes],
) -> dict:
    struct_dir = output_root / "structures" / owner_id / target_name
    pdb_dir = struct_dir / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)

    sdf_path = struct_dir / "Selected_dockings.sdf"
    sdf_path.write_bytes(sdf_bytes)

    total_pdb_bytes = 0
    for entry in pdb_entries:
        pdb_path = pdb_dir / entry["filename"]
        decoded = base64.b64decode(entry["content_b64"])
        pdb_path.write_bytes(decoded)
        total_pdb_bytes += len(decoded)

    if ligands_csv_bytes:
        csv_path = struct_dir / "Ligands_per_pdb.csv"
        csv_path.write_bytes(ligands_csv_bytes)

    rel_base = f"structures/{owner_id}/{target_name}"
    return {
        "sdf_path": f"{rel_base}/Selected_dockings.sdf",
        "pdb_folder": f"{rel_base}/pdbs",
        "target_dir": rel_base,
        # Raw count of "$$$$" record delimiters in the SDF bytes — NOT the
        # validated molecule count (that comes from extract_3d_data, which
        # skips unparseable records). Kept separate so the upload response
        # and the training CSV never disagree on row counts.
        "sdf_record_count": sdf_bytes.count(b"$$$$"),
        "pdb_count": len(pdb_entries),
        "total_pdb_bytes": total_pdb_bytes,
    }


def extract_3d_data(
    sdf_file: str,
    property_key: str = "pChEMBL",
    data_dir: str = "prepared_data",
    file_nm: str = "data.csv",
) -> dict:
    """Extract SMILES, property values, and PDB refs from an SDF file.

    Molecules missing ``property_key`` or with a non-numeric value are skipped
    (like the existing ``mol is None`` handling), so a single incomplete SDF
    record cannot crash the whole extraction.

    Returns dict with 'data_file' (path to generated CSV), 'pdb_folder',
    'sdf_file', and 'mol_count'.
    """
    import pandas as pd
    from rdkit import Chem

    pdb_list = []
    original_smiles = []
    aff_val = []

    for mol in Chem.SDMolSupplier(sdf_file, removeHs=False):
        if mol is None:
            continue
        prop_dict = mol.GetPropsAsDict()

        val_str = str(prop_dict.get(property_key, ""))
        if val_str and val_str[0] in "<>=":
            val_str = val_str.lstrip("<>=")
        try:
            val = float(val_str)
        except ValueError:
            continue

        pdb_list.append(prop_dict.get("pdb", ""))
        aff_val.append(val)
        original_smiles.append(Chem.MolToSmiles(mol))

    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, file_nm)

    df = pd.DataFrame({
        "original_smiles": original_smiles,
        property_key: aff_val,
        "pdb": pdb_list,
    })
    df.to_csv(out_path, index=False)

    return {
        "data_file": os.path.abspath(out_path),
        "sdf_file": os.path.abspath(sdf_file),
        "pdb_folder": os.path.abspath(os.path.join(os.path.dirname(sdf_file), "pdbs")),
        "mol_count": len(df),
    }


def register_3d_tools(mcp_instance):
    """Register the upload_3d_structure tool on a FastMCP instance.

    Called from server.py to add the 3D tool to the server.
    """
    from fastmcp.exceptions import McpError
    from mcp.types import INTERNAL_ERROR

    @mcp_instance.tool
    async def upload_3d_structure(
        target_name: str,
        ctx: Context,
        sdf_content_b64: Optional[str] = None,
        pdb_files: Optional[str] = None,
        ligands_per_pdb_b64: Optional[str] = None,
        local_sdf_path: Optional[str] = None,
        local_pdb_folder: Optional[str] = None,
        local_ligands_per_pdb: Optional[str] = None,
    ) -> dict:
        """Upload a 3D structure bundle (SDF + PDB files) for structure-based ML.

        Use this to upload docked ligands (SDF) and protein structures (PDBs)
        for training with 3D feature generators (ProLIF, AffGraph).

        The server stores files in the expected directory layout:
          structures/<owner>/<target_name>/Selected_dockings.sdf
          structures/<owner>/<target_name>/pdbs/*.pdb

        Returns a structure_id — pass it as `dataset_id` to start_training_session.

        Two modes — provide one or the other:
          Remote (base64): sdf_content_b64 + pdb_files (JSON array with content_b64 per file)
          Local path:      local_sdf_path + local_pdb_folder (absolute paths on the server filesystem)

        Args:
            target_name: Identifier for the target (e.g. "ABL"). Alphanumeric + underscore.
            sdf_content_b64: Base64-encoded SDF file with docked ligands. Each entry must have a 'pdb' property.
            pdb_files: JSON array of objects: [{"filename": "2E2B_receptor.pdb", "content_b64": "..."}]
            ligands_per_pdb_b64: Optional base64-encoded CSV summarizing ligands per PDB.
            local_sdf_path: Absolute path to SDF file on the server filesystem (local mode).
            local_pdb_folder: Absolute path to folder of PDB receptor files (local mode).
            local_ligands_per_pdb: Absolute path to ligands-per-PDB CSV (local mode, optional).
        """
        from _auth import LOCAL_USER_ID
        from _pipeline import _output_root

        # Import server helpers (circular-safe: this runs after server.py defines them)
        import server as _srv
        caller = _srv._get_caller(ctx)
        caller = _srv._require_auth(caller)
        owner_id = caller.get("owner_id", LOCAL_USER_ID)

        # Validate target name
        try:
            target_name = _validate_target_name(target_name)
        except ValueError as exc:
            raise McpError(code=INTERNAL_ERROR, message=str(exc))

        # ── Local path mode ──────────────────────────────────────────────────────
        if local_sdf_path is not None or local_pdb_folder is not None:
            if not local_sdf_path or not local_pdb_folder:
                raise McpError(code=INTERNAL_ERROR,
                    message="Both local_sdf_path and local_pdb_folder must be provided together.")
            sdf_p = Path(local_sdf_path)
            pdb_p = Path(local_pdb_folder)
            if not sdf_p.is_file():
                raise McpError(code=INTERNAL_ERROR, message=f"local_sdf_path not found: {sdf_p}")
            if not pdb_p.is_dir():
                raise McpError(code=INTERNAL_ERROR, message=f"local_pdb_folder not found: {pdb_p}")

            sdf_bytes = sdf_p.read_bytes()
            pdb_entries = []
            for pdb_file in sorted(pdb_p.iterdir()):
                if pdb_file.suffix.lower() == ".pdb":
                    pdb_entries.append({
                        "filename": pdb_file.name,
                        "content_b64": base64.b64encode(pdb_file.read_bytes()).decode(),
                    })
            if not pdb_entries:
                raise McpError(code=INTERNAL_ERROR, message=f"No .pdb files found in {pdb_p}")

            ligands_bytes = None
            if local_ligands_per_pdb:
                lp = Path(local_ligands_per_pdb)
                if lp.is_file():
                    ligands_bytes = lp.read_bytes()

        # ── Base64 mode ──────────────────────────────────────────────────────────
        else:
            if not sdf_content_b64 or not pdb_files:
                raise McpError(code=INTERNAL_ERROR,
                    message="Provide either (sdf_content_b64 + pdb_files) or (local_sdf_path + local_pdb_folder).")

            max_bytes = _srv._max_upload_bytes()
            if len(sdf_content_b64) > max_bytes * 4 // 3 + 4:
                raise McpError(code=INTERNAL_ERROR, message=f"SDF exceeds max upload size ({max_bytes // (1024*1024)} MB).")

            try:
                sdf_bytes = base64.b64decode(sdf_content_b64)
            except Exception as exc:
                raise McpError(code=INTERNAL_ERROR, message=f"Invalid base64 in sdf_content_b64: {exc}")

            if len(sdf_bytes) > max_bytes:
                raise McpError(code=INTERNAL_ERROR, message=f"SDF exceeds max upload size ({max_bytes // (1024*1024)} MB).")

            try:
                pdb_entries = _validate_pdb_files_json(pdb_files)
            except ValueError as exc:
                raise McpError(code=INTERNAL_ERROR, message=str(exc))

            total_pdb_b64_len = sum(len(e["content_b64"]) for e in pdb_entries)
            if total_pdb_b64_len > max_bytes * 4 // 3 + 4:
                raise McpError(code=INTERNAL_ERROR, message=f"PDB files exceed max upload size ({max_bytes // (1024*1024)} MB).")

            ligands_bytes = None
            if ligands_per_pdb_b64:
                try:
                    ligands_bytes = base64.b64decode(ligands_per_pdb_b64)
                except Exception as exc:
                    raise McpError(code=INTERNAL_ERROR, message=f"Invalid base64 in ligands_per_pdb_b64: {exc}")

        # Store files
        result = _store_3d_structure(
            output_root=_output_root(),
            owner_id=owner_id,
            target_name=target_name,
            sdf_bytes=sdf_bytes,
            pdb_entries=pdb_entries,
            ligands_csv_bytes=ligands_bytes,
        )

        # Register in data registry. Single locked read-modify-write (rather
        # than register_dataset() followed by a re-acquired patch) so the
        # entry never becomes visible to another reader without its 3D
        # metadata (sdf_path/pdb_folder/type) already attached.
        from _data_registry import load_json_list, atomic_write_json, data_registry_path, _lock_path
        from _auth import _acquire_lock, _release_lock

        reg_path = data_registry_path()
        lock_fd = _acquire_lock(_lock_path(reg_path))
        if lock_fd is None:
            raise McpError(code=INTERNAL_ERROR, message="Could not acquire data registry lock.")
        try:
            data = load_json_list(reg_path)
            now = datetime.now().isoformat(timespec="seconds")
            canonical_filename = f"{target_name}_3d_structure"
            # Upsert: if same target_name already exists for this owner, update in-place
            existing = next(
                (e for e in data
                 if e.get("owner") == owner_id and e.get("filename") == canonical_filename),
                None,
            )
            if existing:
                existing.update({
                    "file_path": result["target_dir"],
                    "size_bytes": len(sdf_bytes) + result["total_pdb_bytes"],
                    "row_count": result["sdf_record_count"],
                    "last_used": now,
                    "sdf_path": result["sdf_path"],
                    "pdb_folder": result["pdb_folder"],
                })
                entry = existing
            else:
                entry = {
                    "id": f"ds_{secrets.token_urlsafe(12)}",
                    "filename": canonical_filename,
                    "owner": owner_id,
                    "file_path": result["target_dir"],
                    "size_bytes": len(sdf_bytes) + result["total_pdb_bytes"],
                    "columns": ["original_smiles", "pdb", "property"],
                    "row_count": result["sdf_record_count"],
                    "uploaded_at": now,
                    "last_used": now,
                    "type": "3d_structure",
                    "sdf_path": result["sdf_path"],
                    "pdb_folder": result["pdb_folder"],
                }
                data.append(entry)
            atomic_write_json(reg_path, data)
        finally:
            _release_lock(lock_fd, _lock_path(reg_path))

        return {
            "structure_id": entry["id"],
            "target_name": target_name,
            "sdf_path": result["sdf_path"],
            "pdb_folder": result["pdb_folder"],
            "sdf_record_count": result["sdf_record_count"],
            "pdb_count": result["pdb_count"],
        }
