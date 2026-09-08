from fastapi import APIRouter, HTTPException

from ..job_store import cancel_job, get_job, refresh_job_status
from ..schemas.job import JobLogsResponse, JobStatusResponse

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

_LARGE_RESULT_KEYS = {"dashboard_html"}


def _slim_result(result):
    """Strip large payload fields from result to keep status responses lightweight."""
    if not isinstance(result, dict):
        return result
    return {k: v for k, v in result.items() if k not in _LARGE_RESULT_KEYS}


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    # Refresh from MCP task if status notifications may have been missed
    await refresh_job_status(job)
    return JobStatusResponse(
        id=job.id,
        description=job.description,
        status=job.status.value,
        exit_code=job.exit_code,
        created_at=job.created_at,
        finished_at=job.finished_at,
        result=_slim_result(job.result),
        progress=job.progress,
        progress_total=job.progress_total,
        progress_label=job.progress_label,
    )


@router.get("/{job_id}/logs", response_model=JobLogsResponse)
async def job_logs(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JobLogsResponse(id=job.id, lines=job.log_lines)


@router.post("/{job_id}/cancel")
async def job_cancel(job_id: str):
    """Cancel a running job via the MCP task protocol."""
    success = await cancel_job(job_id)
    if not success:
        raise HTTPException(400, "Job not found or not cancellable")
    return {"status": "cancelled", "job_id": job_id}
