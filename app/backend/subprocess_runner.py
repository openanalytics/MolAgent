"""Async subprocess runner — launches CLI scripts and streams output to jobs."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .config import settings
from .job_store import Job, JobStatus


async def run_script(
    job: Job,
    script_path: Path,
    args: list[str],
    *,
    use_uv: bool = False,
    capture_stdout: bool = False,
    cwd: Path | None = None,
    env_overrides: dict[str, str] | None = None,
) -> None:
    env = {**os.environ}
    env["MOLAGENT_PLUGIN_ROOT"] = str(settings.plugin_root)
    env["MOLAGENT_OUTPUT_ROOT"] = str(settings.output_root)
    if env_overrides:
        env.update(env_overrides)

    if use_uv:
        cmd = [settings.uv_path, "run", str(script_path)] + args
    else:
        env["PYTHONPATH"] = str(script_path.parent)
        cmd = [str(settings.venv_python), str(script_path)] + args

    work_dir = cwd or settings.plugin_root
    job.status = JobStatus.RUNNING

    try:
        if capture_stdout:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(work_dir),
            )
            stdout_data, stderr_data = await proc.communicate()
            if stderr_data:
                for line in stderr_data.decode(errors="replace").splitlines():
                    job.log_lines.append(line)
            raw = stdout_data.decode(errors="replace").strip()
            try:
                job.result = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                job.result = raw
        else:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                cwd=str(work_dir),
            )
            async for raw_line in proc.stdout:
                job.log_lines.append(raw_line.decode(errors="replace").rstrip())
            await proc.wait()

        job.exit_code = proc.returncode
        job.status = JobStatus.SUCCESS if proc.returncode == 0 else JobStatus.FAILED
    except Exception as exc:
        job.log_lines.append(f"ERROR: {exc}")
        job.status = JobStatus.FAILED
        job.exit_code = -1
    finally:
        job.finished_at = datetime.now(timezone.utc)


def launch(job: Job, script_path: Path, args: list[str], **kwargs) -> None:
    asyncio.get_running_loop().create_task(run_script(job, script_path, args, **kwargs))
