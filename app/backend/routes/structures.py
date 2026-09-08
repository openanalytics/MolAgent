"""3D structure upload routes — forwards SDF + PDB bundles to MCP upload_3d_structure."""

import base64
import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..mcp_client import call_tool, MCPAuthError

router = APIRouter(prefix="/api/structures", tags=["structures"])


@router.post("/upload")
async def upload_structure(
    target_name: str = Form(...),
    sdf_file: UploadFile = File(...),
    pdb_files: list[UploadFile] = File(...),
    ligands_csv: UploadFile | None = File(None),
):
    """Upload a 3D structure bundle (SDF + PDB files) — forwards base64 to MCP upload_3d_structure."""
    sdf_bytes = await sdf_file.read()
    sdf_b64 = base64.b64encode(sdf_bytes).decode()

    pdb_entries = []
    for f in pdb_files:
        content = await f.read()
        pdb_entries.append({
            "filename": f.filename or "structure.pdb",
            "content_b64": base64.b64encode(content).decode(),
        })

    ligands_b64 = None
    if ligands_csv is not None:
        ligands_bytes = await ligands_csv.read()
        ligands_b64 = base64.b64encode(ligands_bytes).decode()

    try:
        result = await call_tool("upload_3d_structure", {
            "target_name": target_name,
            "sdf_content_b64": sdf_b64,
            "pdb_files": json.dumps(pdb_entries),
            "ligands_per_pdb_b64": ligands_b64,
        })
    except MCPAuthError as exc:
        raise HTTPException(403, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, f"Upload failed: {exc}")
    return result
