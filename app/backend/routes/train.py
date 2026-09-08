"""Training pipeline routes — upload, detect, configure, and run via MCP."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ..config import settings
from ..job_store import Job, create_job, launch_task_job
from ..mcp_client import call_tool, call_tool_as_task, get_mcp_settings, MCPAuthError
from ..schemas.job import JobCreatedResponse

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/train", tags=["train"])


# ── Schemas ──────────────────────────────────────────────────────────────────


class TrainingConfig(BaseModel):
    """Mirrors MCP TrainingConfig — the parameters exposed to the frontend."""

    csv_file: str
    smiles_column: str
    properties: list[str]
    task: Literal["Regression", "Classification", "RegressionClassification"]
    sep: str = ","
    computational_load: Literal["free", "cheap", "moderate", "expensive"] = "cheap"
    feature_keys: list[str] = ["Bottleneck"]
    use_log10: bool = False
    use_logit: bool = False
    categorical: bool = False
    nb_classes: Optional[list[int]] = None
    class_values: Optional[list] = None
    class_quantiles: Optional[list] = None
    split_strategy: Literal["mixed", "stratified", "leave_group_out"] = "mixed"
    test_size: float = 0.25
    outer_folds: int = 4
    random_state: int = 42
    n_jobs_inner: int = 2
    n_jobs_outer: Optional[int] = None
    refit: bool = True
    include_test_in_refit: bool = True
    # Tier 1 extensions
    blender_properties: Optional[list[str]] = None
    scorer: Optional[str] = None
    use_advanced: bool = False
    use_sample_weight: bool = False
    clustering_method: str = "Bottleneck"
    n_clusters: int = 20
    butina_cutoff: float = 0.6
    # Tier 2 extensions — advanced model configuration
    base_list: Optional[list[str]] = None
    blender_list: Optional[list[str]] = None
    red_dim_list: Optional[list[str]] = None
    ensemble_config: Optional[str] = None
    search_type: Optional[str] = None
    randomized_iterations: Optional[int] = None
    # 3D structure features — never set directly by the frontend, just present
    # so model_dump() forwards whatever the MCP session produced.
    sdf_file: Optional[str] = None
    protein_folder: Optional[str] = None

    model_config = {"protected_namespaces": ()}


class RunPipelineRequest(BaseModel):
    config: TrainingConfig

    model_config = {"protected_namespaces": ()}


class DetectRequest(BaseModel):
    csv_path: Optional[str] = None
    dataset_id: Optional[str] = None

    model_config = {"protected_namespaces": ()}


# ── Upload ───────────────────────────────────────────────────────────────────


@router.post("/upload")
async def upload_file(file: UploadFile):
    """Upload a CSV — routes through MCP upload_dataset for registry tracking."""
    import base64

    contents = await file.read()
    b64 = base64.b64encode(contents).decode()
    try:
        mcp_result = await call_tool("upload_dataset", {
            "filename": file.filename or "upload.csv",
            "file_content_b64": b64,
        })
    except MCPAuthError as exc:
        raise HTTPException(403, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, f"Upload failed: {exc}")

    # Return format compatible with the frontend UploadResult interface
    return {
        "path": mcp_result.get("dataset_id"),
        "filename": mcp_result.get("filename"),
        "size": mcp_result.get("size_bytes", len(contents)),
        "columns": mcp_result.get("columns"),
        "dataset_id": mcp_result.get("dataset_id"),
    }


# ── Detect (calls MCP start_training_session) ───────────────────────────────


@router.post("/detect")
async def detect_dataset(req: DetectRequest):
    """Call MCP start_training_session. Accepts csv_path OR dataset_id."""
    args: dict = {}
    if req.dataset_id:
        args["dataset_id"] = req.dataset_id
    elif req.csv_path:
        if not Path(req.csv_path).exists():
            raise HTTPException(400, f"File not found: {req.csv_path}")
        args["csv_file"] = req.csv_path
    else:
        raise HTTPException(400, "Provide either csv_path or dataset_id")

    try:
        result = await call_tool("start_training_session", args)
    except MCPAuthError as exc:
        raise HTTPException(403, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, f"Detection failed: {exc}")

    return result


# ── Configure (calls MCP answer_training_question) ──────────────────────────


class ConfigureRequest(BaseModel):
    session_id: str
    confirm: Optional[bool] = None
    smiles_column: Optional[str] = None
    properties: Optional[list[str]] = None
    task: Optional[Literal["Regression", "Classification", "RegressionClassification"]] = None
    computational_load: Optional[Literal["free", "cheap", "moderate", "expensive"]] = None
    feature_keys: Optional[list[str]] = None
    use_log10: Optional[bool] = None
    use_logit: Optional[bool] = None
    split_strategy: Optional[Literal["mixed", "stratified", "leave_group_out"]] = None
    base_list: Optional[list[str]] = None
    blender_list: Optional[list[str]] = None
    red_dim_list: Optional[list[str]] = None
    ensemble_config: Optional[str] = Field(default=None, alias="model_config")
    search_type: Optional[str] = None
    randomized_iterations: Optional[int] = None
    scorer: Optional[str] = None
    # Classification options
    categorical: Optional[bool] = None
    nb_classes: Optional[list[int]] = None
    class_values: Optional[list] = None
    class_quantiles: Optional[list] = None
    # Refit control
    refit: Optional[bool] = None
    include_test_in_refit: Optional[bool] = None
    # Sample weights
    use_sample_weight: Optional[bool] = None
    sample_weight_selection: Optional[str] = None
    sample_weight_multiplier: Optional[float] = None

    model_config = {"protected_namespaces": (), "populate_by_name": True}


@router.post("/configure")
async def configure_session(req: ConfigureRequest):
    """Apply overrides or confirm training config via MCP answer_training_question."""
    args = req.model_dump(exclude_none=True, by_alias=True)
    try:
        result = await call_tool("answer_training_question", args)
    except MCPAuthError as exc:
        raise HTTPException(403, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, f"Configuration failed: {exc}")
    return result


# ── Run Pipeline (calls MCP train_and_visualize as a background task) ────────


@router.post("/run", response_model=JobCreatedResponse)
async def run_pipeline(req: RunPipelineRequest):
    """Start full pipeline via MCP train_and_visualize. Returns job_id for polling."""
    config_dict = req.config.model_dump(exclude_none=True)
    try:
        task = await call_tool_as_task("train_and_visualize", {"config": config_dict})
    except MCPAuthError as exc:
        raise HTTPException(403, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, f"Failed to start pipeline: {exc}")
    job = create_job("Train pipeline", progress_total=8)
    job._on_complete = _auto_save_dashboard
    launch_task_job(job, task)
    return JobCreatedResponse(job_id=job.id)


async def _auto_save_dashboard(job: Job) -> None:
    """Save dashboard HTML and pipeline state to output_folder/{run_id}/ if configured."""
    output_folder = get_mcp_settings().output_folder
    if not output_folder:
        return
    result = job.result
    if not isinstance(result, dict):
        return
    run_id = result.get("run_id")
    if not run_id:
        return
    dest_dir = Path(output_folder) / run_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dashboard_html = result.get("dashboard_html")
    if dashboard_html:
        (dest_dir / "dashboard.html").write_text(dashboard_html, encoding="utf-8")
    # Save a minimal pipeline_state.json so the Visualize tab can discover this run
    state = {
        "run_id": run_id,
        "pipeline_complete": True,
        "steps_completed": list(range(9)),
        "config": {
            "target_properties": result.get("properties", []),
            "task_type": result.get("task", ""),
        },
        "metrics": result.get("metrics", {}),
    }
    (dest_dir / "pipeline_state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    _logger.info("Training outputs saved to %s", dest_dir)


# ── Legacy: list/get runs (reads pipeline_state.json from disk) ─────────────


def _scan_runs() -> list[dict]:
    runs = []
    seen_ids: set[str] = set()
    output_folder = get_mcp_settings().output_folder
    sources = []
    if output_folder:
        sources.append((Path(output_folder), "remote"))
    sources.append((settings.output_root, "local"))
    for root_path, source in sources:
        if root_path is None or not root_path.exists():
            continue
        for state_path in root_path.glob("*/pipeline_state.json"):
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                rid = state.get("run_id", "")
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    state["source"] = source
                    runs.append(state)
            except (json.JSONDecodeError, OSError):
                continue
        # Also discover runs that only have a dashboard (auto-saved without state)
        for dash_path in root_path.glob("*/dashboard.html"):
            rid = dash_path.parent.name
            if rid not in seen_ids:
                seen_ids.add(rid)
                runs.append({
                    "run_id": rid,
                    "pipeline_complete": True,
                    "steps_completed": list(range(9)),
                    "config": {},
                    "source": source,
                })
    runs.sort(key=lambda r: r.get("run_id", ""), reverse=True)
    return runs


@router.get("/runs")
async def list_runs():
    return _scan_runs()


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    output_folder = get_mcp_settings().output_folder
    if output_folder:
        path = (Path(output_folder) / run_id / "pipeline_state.json").resolve()
        if not path.is_relative_to(Path(output_folder).resolve()):
            raise HTTPException(400, "Invalid run_id")
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    path = (settings.output_root / run_id / "pipeline_state.json").resolve()
    if not path.is_relative_to(settings.output_root.resolve()):
        raise HTTPException(400, "Invalid run_id")
    if not path.exists():
        raise HTTPException(404, f"Run {run_id} not found")
    return json.loads(path.read_text(encoding="utf-8"))
