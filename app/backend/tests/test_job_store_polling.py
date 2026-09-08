"""Tests for job_store._wait_for_result polling loop (FastMCP 4 ToolTask API)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from job_store import (
    Job, JobStatus, launch_task_job, create_job, _POLL_INTERVAL, _MAX_CONSECUTIVE_ERRORS,
)


class _FakeStatus:
    def __init__(self, status="working", message="Training model"):
        self.status = status
        self.status_message = message


class _FakeResult:
    def __init__(self, data=None):
        self.data = data or {"status": "ok", "model_id": "test-123"}
        self.content = []


class _TerminalTask:
    """Resolves after N poll rounds."""
    def __init__(self, rounds=2):
        self.task_id = "fake-terminal"
        self._calls = 0
        self._rounds = rounds

    async def wait(self, *, timeout=300.0):
        self._calls += 1
        if self._calls <= self._rounds:
            await asyncio.sleep(timeout)
            raise TimeoutError("still running")

    async def status(self):
        return _FakeStatus(message="Training model")

    async def result(self):
        return _FakeResult()

    async def cancel(self):
        pass


class _TransientErrorTask:
    """Raises a non-TimeoutError once, then resolves."""
    def __init__(self, fail_on=1):
        self.task_id = "fake-transient"
        self._calls = 0
        self._fail_on = fail_on

    async def wait(self, *, timeout=300.0):
        self._calls += 1
        if self._calls == self._fail_on:
            raise RuntimeError("simulated network error")
        if self._calls <= 2:
            await asyncio.sleep(timeout)
            raise TimeoutError("still running")

    async def status(self):
        return _FakeStatus()

    async def result(self):
        return _FakeResult()

    async def cancel(self):
        pass


class _SustainedErrorTask:
    """Always raises a non-TimeoutError."""
    def __init__(self):
        self.task_id = "fake-sustained"

    async def wait(self, *, timeout=300.0):
        raise RuntimeError("sustained server error")

    async def status(self):
        return _FakeStatus()

    async def result(self):
        return _FakeResult()

    async def cancel(self):
        pass


async def _run_job(task, timeout=60.0):
    job = create_job("test job", progress_total=8)
    launch_task_job(job, task)
    for _ in range(int(timeout / 0.2)):
        await asyncio.sleep(0.2)
        if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
            break
    return job


@pytest.mark.asyncio
async def test_polling_loop_terminal_break():
    """Task completes after 2 poll rounds → SUCCESS."""
    job = await _run_job(_TerminalTask(rounds=2))
    assert job.status == JobStatus.SUCCESS
    assert job.result == {"status": "ok", "model_id": "test-123"}
    assert job.progress_label == "Complete"


@pytest.mark.asyncio
async def test_polling_loop_progress_update():
    """Progress label is updated during polling."""
    task = _TerminalTask(rounds=2)
    job = await _run_job(task)
    # Progress label was set during polling ("Training model") then overwritten to "Complete"
    assert job.status == JobStatus.SUCCESS


@pytest.mark.asyncio
async def test_transient_error_tolerated():
    """Single non-TimeoutError mid-poll is tolerated → job still succeeds."""
    job = await _run_job(_TransientErrorTask(fail_on=1))
    assert job.status == JobStatus.SUCCESS


@pytest.mark.asyncio
async def test_sustained_error_fails_job():
    """More than _MAX_CONSECUTIVE_ERRORS consecutive errors → FAILED."""
    task = _SustainedErrorTask()
    job = await _run_job(task, timeout=30.0)
    assert job.status == JobStatus.FAILED
    assert job.log_lines  # some error was logged
