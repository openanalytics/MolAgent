"""Dataset routes — upload, list, and delete datasets via MCP."""

import base64

from fastapi import APIRouter, HTTPException, UploadFile

from ..mcp_client import call_tool, MCPAuthError

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post("/upload")
async def upload_dataset(file: UploadFile):
    """Upload a CSV dataset (multipart) — forwards base64 to MCP upload_dataset."""
    contents = await file.read()
    b64 = base64.b64encode(contents).decode()
    try:
        result = await call_tool("upload_dataset", {
            "filename": file.filename or "upload.csv",
            "file_content_b64": b64,
        })
    except MCPAuthError as exc:
        raise HTTPException(403, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, f"Upload failed: {exc}")
    return result


@router.get("")
async def list_datasets():
    """List datasets visible to the configured MCP token."""
    try:
        result = await call_tool("list_datasets", {})
        return result.get("datasets", [])
    except MCPAuthError as exc:
        raise HTTPException(403, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, f"Failed to list datasets: {exc}")


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset (registry entry + file on disk)."""
    try:
        result = await call_tool("delete_dataset", {"dataset_id": dataset_id})
    except MCPAuthError as exc:
        raise HTTPException(403, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, f"Failed to delete dataset: {exc}")
    return result
