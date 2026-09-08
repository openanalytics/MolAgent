"""Registry routes — model listing, merging, deletion, and download via MCP."""

import base64
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from ..job_store import create_job, launch_task_job
from ..mcp_client import call_tool, call_tool_as_task, get_mcp_settings, MCPAuthError

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/registry", tags=["registry"])


@router.get("")
async def list_registry():
    """List models visible to the configured MCP token."""
    try:
        result = await call_tool("list_models", {})
        return result.get("models", [])
    except MCPAuthError as exc:
        raise HTTPException(403, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, f"Failed to list models: {exc}")


@router.get("/{model_id}")
async def get_registry_entry(model_id: str):
    """Get a specific model entry."""
    try:
        result = await call_tool("list_models", {})
    except RuntimeError as exc:
        raise HTTPException(500, f"Failed to query registry: {exc}")

    for entry in result.get("models", []):
        if entry.get("id") == model_id:
            return entry
    raise HTTPException(404, f"Model {model_id} not found in registry")


@router.delete("/{model_id}")
async def delete_registry_entry(model_id: str):
    """Delete a model (registry entry + files on disk)."""
    try:
        result = await call_tool("delete_model", {"model_id": model_id})
    except MCPAuthError as exc:
        raise HTTPException(403, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, f"Failed to delete model: {exc}")
    return result


class MergeRequest(BaseModel):
    model_ids: list[str]
    output_name: Optional[str] = None
    verify_encoder: bool = True


@router.post("/merge")
async def merge_models(req: MergeRequest):
    """Merge multiple models into a single multi-property model."""
    args = {
        "model_ids": req.model_ids,
        "verify_encoder": req.verify_encoder,
    }
    if req.output_name:
        args["output_name"] = req.output_name
    try:
        task = await call_tool_as_task("merge_models", args)
    except MCPAuthError as exc:
        raise HTTPException(403, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, f"Failed to start merge: {exc}")
    job = create_job("Merge models")
    launch_task_job(job, task)
    return {"job_id": job.id}


@router.post("/{model_id}/download")
async def download_model_endpoint(model_id: str):
    """Download a model file via MCP and save to output folder (or return bytes)."""
    try:
        result = await call_tool("download_model", {"model_id": model_id})
    except MCPAuthError as exc:
        raise HTTPException(403, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, f"Failed to download model: {exc}")

    model_b64 = result.get("model_b64")
    model_filename = result.get("model_filename", "model.pt")
    if not model_b64:
        raise HTTPException(500, "No model data returned")

    model_bytes = base64.b64decode(model_b64)
    output_folder = get_mcp_settings().output_folder
    if output_folder:
        dest_dir = Path(output_folder)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / model_filename
        dest.write_bytes(model_bytes)
        _logger.info("Model saved to %s", dest)
        return {
            "model_id": model_id,
            "model_filename": model_filename,
            "saved_path": str(dest),
            "size_bytes": result.get("size_bytes"),
        }

    return Response(
        content=model_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{model_filename}"'},
    )
