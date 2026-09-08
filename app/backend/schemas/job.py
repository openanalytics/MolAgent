from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class JobStatusResponse(BaseModel):
    id: str
    description: str
    status: str
    exit_code: int | None = None
    created_at: datetime
    finished_at: datetime | None = None
    result: Any = None
    progress: int = 0
    progress_total: int = 0
    progress_label: str = ""


class JobLogsResponse(BaseModel):
    id: str
    lines: list[str]


class JobCreatedResponse(BaseModel):
    job_id: str
