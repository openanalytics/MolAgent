"""Prediction routes — calls MCP predict tool."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..job_store import JobStatus, create_job, get_job, launch_task_job
from ..mcp_client import call_tool_as_task, MCPAuthError
from ..schemas.job import JobCreatedResponse

router = APIRouter(prefix="/api/predict", tags=["predict"])


class PredictRequest(BaseModel):
    model_id: Optional[str] = None
    model_file: Optional[str] = None
    smiles_file: Optional[str] = None
    smiles_list: Optional[list[str]] = None
    smiles_column: str = "smiles"
    properties: Optional[list[str]] = None
    compute_sd: bool = True
    blender_properties: Optional[list[str]] = None
    blender_values: Optional[dict[str, float]] = None
    convert_log10: Optional[bool] = None

    model_config = {"protected_namespaces": ()}


@router.post("/run", response_model=JobCreatedResponse)
async def run_predict(req: PredictRequest):
    if not req.model_id and not req.model_file:
        raise HTTPException(400, "Provide model_id or model_file")
    if not req.smiles_file and not req.smiles_list:
        raise HTTPException(400, "Provide smiles_file or smiles_list")

    args = req.model_dump(exclude_none=True)
    try:
        task = await call_tool_as_task("predict", args)
    except MCPAuthError as exc:
        raise HTTPException(403, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, f"Failed to start prediction: {exc}")
    job = create_job(f"Predict ({req.model_id or req.model_file})")
    launch_task_job(job, task)
    return JobCreatedResponse(job_id=job.id)


@router.get("/{job_id}/result")
async def predict_result(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.SUCCESS:
        raise HTTPException(400, "Job not complete")
    if not job.result:
        raise HTTPException(404, "No result available")

    result = job.result
    output_csv = result.get("output_csv")
    if output_csv:
        return {"csv_path": output_csv, "filename": Path(output_csv).name}
    csv_filename = result.get("csv_filename")
    if csv_filename:
        return {"csv_path": None, "filename": csv_filename}
    return result


@router.get("/{job_id}/download")
async def predict_download(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.SUCCESS:
        raise HTTPException(400, "Job not complete")

    result = job.result or {}

    # Local mode: serve file from disk
    output_csv = result.get("output_csv")
    if output_csv and Path(output_csv).exists():
        return FileResponse(output_csv, filename=Path(output_csv).name, media_type="text/csv")

    # Remote mode: serve inline CSV content from MCP response
    csv_content = result.get("csv_content")
    if csv_content:
        from fastapi.responses import Response
        filename = result.get("csv_filename", "predictions.csv")
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    raise HTTPException(404, "Prediction CSV not found")
