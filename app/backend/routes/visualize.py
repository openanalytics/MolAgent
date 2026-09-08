"""Visualization routes — serve dashboards from training results or disk."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from ..config import settings
from ..job_store import get_job
from ..mcp_client import get_mcp_settings

router = APIRouter(prefix="/api/visualize", tags=["visualize"])


@router.get("/{run_id}/html")
async def serve_dashboard(run_id: str):
    """Serve dashboard HTML from disk — checks output_folder first (where auto-save
    writes in remote mode), then the MCP output_root (local mode)."""
    output_folder = get_mcp_settings().output_folder
    if output_folder:
        path = (Path(output_folder) / run_id / "dashboard.html").resolve()
        if not path.is_relative_to(Path(output_folder).resolve()):
            raise HTTPException(400, "Invalid run_id")
        if path.exists():
            return HTMLResponse(path.read_text(encoding="utf-8"))
    html_path = (settings.output_root / run_id / "dashboard.html").resolve()
    if not html_path.is_relative_to(settings.output_root.resolve()):
        raise HTTPException(400, "Invalid run_id")
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    raise HTTPException(404, "Dashboard not available for this run")


@router.get("/job/{job_id}/html")
async def serve_dashboard_from_job(job_id: str):
    """Serve dashboard HTML from a completed training job result (remote mode).
    The train_and_visualize MCP tool returns dashboard_html inline."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if not job.result:
        raise HTTPException(400, "Job has no result")

    result = job.result
    dashboard_html = result.get("dashboard_html")
    if dashboard_html:
        return HTMLResponse(dashboard_html)

    # Fallback: check if result has dashboard_path (local mode)
    dashboard_path = result.get("dashboard_path")
    if dashboard_path:
        p = Path(dashboard_path)
        if p.exists():
            return HTMLResponse(p.read_text(encoding="utf-8"))

    raise HTTPException(404, "Dashboard not available in job result")
