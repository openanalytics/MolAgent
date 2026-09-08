"""Job store backed by FastMCP ToolTask handles.

Wraps FastMCP's background task protocol (SEP-1686) while exposing the same
JobStatusResponse API the frontend already polls. Jobs are stored in-memory
as thin wrappers around ToolTask — if the server uses Redis-backed Docket,
task state survives backend restarts (the task_id can be re-queried).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# prevent GC from collecting fire-and-forget tasks
_background_tasks: set[asyncio.Task] = set()


def _fire_and_forget(coro) -> None:
    t = asyncio.get_running_loop().create_task(coro)
    _background_tasks.add(t)
    t.add_done_callback(_background_tasks.discard)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class Job:
    """Thin wrapper around a FastMCP ToolTask or a direct async call."""
    id: str
    description: str
    status: JobStatus = JobStatus.PENDING
    log_lines: list[str] = field(default_factory=list)
    exit_code: int | None = None
    result: Any = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    progress: int = 0
    progress_total: int = 0
    progress_label: str = ""
    # FastMCP ToolTask handle (fastmcp_tasks.ToolTask)
    _task: Any = field(default=None, repr=False)
    # MCP task ID for cancellation/status queries
    task_id: str | None = None
    # Optional callback invoked after job completes successfully
    _on_complete: Callable[["Job"], Awaitable[None]] | None = field(default=None, repr=False)


_store: dict[str, Job] = {}


def create_job(description: str, progress_total: int = 0) -> Job:
    job_id = uuid.uuid4().hex[:12]
    job = Job(id=job_id, description=description, progress_total=progress_total)
    _store[job_id] = job
    return job


def get_job(job_id: str) -> Job | None:
    return _store.get(job_id)


def list_jobs() -> list[Job]:
    return list(_store.values())


def launch_task_job(job: Job, task) -> None:
    """Attach a ToolTask to a job and start the polling loop in the background."""
    job._task = task
    job.task_id = task.task_id
    job.status = JobStatus.RUNNING
    _fire_and_forget(_wait_for_result(job, task))


def _err_from_obj(obj) -> str | None:
    """Try to extract an error string from an MCP-style exception or its cause."""
    err = getattr(obj, "error", None)
    if isinstance(err, str):
        return err
    if err is not None and hasattr(err, "message"):
        msg = err.message
        return msg if isinstance(msg, str) else str(msg)
    return None


def _is_internal_noise(msg: str) -> bool:
    """True for messages that are Python internals rather than user-visible errors."""
    return not msg or "object has no attribute" in msg or msg == "None"


def _rewrite_task_not_found(msg: str, task_id: str | None) -> str:
    """Replace a raw task-not-found error with a human-readable OOM/crash hint."""
    if task_id and task_id in msg:
        logger.debug("Task-not-found detail: %s", msg)
        return (f"MCP server lost track of task {task_id}. "
                "The server likely restarted (OOM or crash) during training. "
                "Try reducing computational load or increasing server memory.")
    return msg


def _extract_error_message(exc: Exception) -> str:
    """Extract a readable error message from MCP/docket exceptions.

    Walks the full cause chain so that internal AttributeErrors from the
    fastmcp/docket layer don't hide the original tool error message.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    last_resort = str(exc) or "Unknown error"
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        msg = _err_from_obj(cur) or str(cur)
        if msg and not _is_internal_noise(msg):
            return msg
        cur = cur.__cause__ or cur.__context__
    return last_resort


async def _finalize_job(job: Job) -> None:
    """Set finished_at and fire on_complete callback if the job succeeded."""
    job.finished_at = datetime.now(timezone.utc)
    if job.status == JobStatus.SUCCESS and job._on_complete:
        try:
            await job._on_complete(job)
        except Exception:
            logger.debug("on_complete callback failed", exc_info=True)


# 48 hours — training can run for a very long time
_TASK_TIMEOUT = 48 * 3600.0
_POLL_INTERVAL = 5.0  # seconds between progress polls during task wait
# NOTE: each poll issues tasks/get which refreshes the server-side task TTL.
# Keep _POLL_INTERVAL well below the server's execution_ttl (Docket default: 15 min)
# or long-running tasks may be lost if polling is interrupted.
_MAX_CONSECUTIVE_ERRORS = 5  # tolerate this many consecutive network errors before giving up


async def _wait_for_result(job: Job, task) -> None:
    """Wait for background task completion, polling for progress every few seconds."""
    elapsed = 0.0
    _consecutive_errors = 0
    try:
        # Short-interval poll loop: each iteration either reaches a terminal
        # state (break) or updates progress and continues.
        while elapsed < _TASK_TIMEOUT:
            remaining = _TASK_TIMEOUT - elapsed
            poll_time = min(_POLL_INTERVAL, remaining)
            try:
                await task.wait(timeout=poll_time)
                break  # task reached a terminal state
            except TimeoutError:
                elapsed += poll_time
                _consecutive_errors = 0  # reset on a clean "still running" signal
                # Still running — update job progress from current status
                try:
                    status_update = await task.status()
                    _on_task_status(job, status_update)
                    _consecutive_errors = 0
                except Exception:
                    pass
            except Exception:
                # Transient network/server error — tolerate up to _MAX_CONSECUTIVE_ERRORS
                elapsed += poll_time
                _consecutive_errors += 1
                if _consecutive_errors > _MAX_CONSECUTIVE_ERRORS:
                    raise
        else:
            raise TimeoutError(f"Task timed out after {_TASK_TIMEOUT}s")

        # task.wait() returned cleanly — check whether it failed
        _GENERIC_FAIL_PREFIXES = ("task failed", "failed", "error")
        try:
            status = await task.status()
            task_state = getattr(status, "status", None)
            if task_state is not None and str(task_state) == "failed":
                raw_msg = (getattr(status, "status_message", None) or "").strip()
                low = raw_msg.lower()
                is_generic = (
                    not raw_msg
                    or low in _GENERIC_FAIL_PREFIXES
                    or any(low.startswith(p) and len(low) < len(p) + 30
                           for p in _GENERIC_FAIL_PREFIXES)
                    or "object has no attribute" in low
                )
                if raw_msg and not is_generic:
                    msg = _rewrite_task_not_found(raw_msg, job.task_id)
                    job.log_lines.append(f"ERROR: {msg}")
                    job.status = JobStatus.FAILED
                    job.exit_code = 1
                    return
        except Exception:
            pass  # status() failed — fall through to task.result() path

        result = await task.result()
        job.result = _parse_task_result(result)
        job.status = JobStatus.SUCCESS
        job.exit_code = 0
        if job.progress_total:
            job.progress = job.progress_total
            job.progress_label = "Complete"
    except TimeoutError:
        job.log_lines.append(f"ERROR: Task timed out after {_TASK_TIMEOUT}s")
        job.status = JobStatus.FAILED
        job.exit_code = 1
    except Exception as exc:
        msg = _rewrite_task_not_found(_extract_error_message(exc), job.task_id)
        job.log_lines.append(f"ERROR: {msg}")
        job.status = JobStatus.FAILED
        job.exit_code = 1
    finally:
        await _finalize_job(job)


# Labels the MCP server sends via progress.set_message() — order matches pipeline steps
_STEP_LABELS_ORDER = [
    "Preparing data",
    "Splitting data",
    "Training model",
    "Merging models",
    "Evaluating model",
    "Refitting model",
    "Merging refitted models",
    "Generating dashboard",
    "Complete",
]


def _step_number_from_label(label: str) -> int | None:
    """Derive a numeric step from a progress message label."""
    try:
        return _STEP_LABELS_ORDER.index(label) + 1
    except ValueError:
        return None


def _on_task_status(job: Job, status) -> None:
    """Callback fired by ToolTask on status change notifications.

    Only updates progress/label here — terminal state (SUCCESS/FAILED) is set
    exclusively by _wait_for_result to avoid races.
    """
    try:
        mcp_status = status.status if hasattr(status, "status") else str(status)
        if mcp_status == "working":
            job.status = JobStatus.RUNNING

        # Extract progress from notification — MCP sends status_message (snake_case in FastMCP 4 / SDK v2),
        # not numeric progress fields, so we derive the step number from the label
        msg = getattr(status, "status_message", None) or getattr(status, "message", None)
        if msg:
            job.progress_label = msg
            step = _step_number_from_label(msg)
            if step is not None:
                job.progress = step
    except Exception:
        logger.debug("Error processing task status notification", exc_info=True)


async def cancel_job(job_id: str) -> bool:
    """Cancel a running job via the MCP task protocol."""
    job = get_job(job_id)
    if job is None or job._task is None:
        return False
    if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
        return False
    try:
        await job._task.cancel()
        job.status = JobStatus.FAILED
        job.log_lines.append("Cancelled by user")
        job.finished_at = datetime.now(timezone.utc)
        return True
    except Exception as exc:
        logger.warning("Failed to cancel task %s: %s", job.task_id, exc)
        return False


async def refresh_job_status(job: Job) -> None:
    """Query the MCP server for latest task status (useful if notifications missed).

    Only queries if the job is still running — avoids unnecessary server calls.
    """
    if job._task is None:
        return
    if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
        return
    try:
        status = await job._task.status()
        _on_task_status(job, status)
    except Exception:
        logger.debug("Failed to refresh task status for job %s", job.id, exc_info=True)


def _parse_task_result(result) -> Any:
    """Convert a ToolTask result (CallToolResult) to a plain dict."""
    if result is None:
        return None
    if isinstance(result, (dict, list)):
        return result
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (json.JSONDecodeError, ValueError):
            return result
    # FastMCP 4: CallToolResult.data is the parsed Python value (preferred)
    if hasattr(result, "data") and result.data is not None:
        return result.data
    # Fallback: join text content blocks and JSON-parse
    if hasattr(result, "content"):
        texts = []
        for content in result.content:
            if hasattr(content, "text"):
                texts.append(content.text)
        combined = "\n".join(texts)
        try:
            return json.loads(combined)
        except (json.JSONDecodeError, ValueError):
            return combined
    return result
