#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["fastmcp[tasks]>=4", "pandas", "pydantic>=2.12"]
# ///
"""AutoMol MCP server — thirteen tools:

  list_options              — discover available features, estimators, configs
                              installed on this server (dynamic, admin-swappable).
  start_training_session    — run dataset detection, create a session, return
                              detected defaults + available options for Claude to
                              present to the user.
  answer_training_question  — accept typed field overrides (Claude parses the
                              user's natural language into typed values); return
                              the final TrainingConfig when confirmed.
  train_and_visualize       — deterministic: accepts a TrainingConfig, runs the
                              full pipeline, returns model_id + dashboard.
  list_models               — query the model registry for available models.
  predict                   — run inference on new molecules using a trained model.
  merge_models              — merge multiple registry models into one multi-property file.
  delete_model              — remove a model from registry and delete its files.
  download_model            — download a model's binary data (base64) by registry ID.
  upload_dataset            — upload a CSV dataset (base64) to the data registry.
  list_datasets             — list datasets in the data registry for the caller.
  delete_dataset            — remove a dataset from registry and delete its file.
  admin_manage              — admin token management + purge stale artifacts.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

_mcp_max_body_bytes = int(float(os.environ.get("MOLAGENT_MAX_UPLOAD_MB", "100")) * 1024 * 1024 * 2)
import mcp.server.transport_security as _mcp_ts
import mcp.server.streamable_http_manager as _mcp_shm
_mcp_ts.DEFAULT_MAX_REQUEST_BODY_SIZE = _mcp_max_body_bytes
_mcp_shm.DEFAULT_MAX_REQUEST_BODY_SIZE = _mcp_max_body_bytes
_init = _mcp_shm.StreamableHTTPSessionManager.__init__
_init.__defaults__ = _init.__defaults__[:-1] + (_mcp_max_body_bytes,)

import pandas as pd
from fastmcp import Context, FastMCP
from fastmcp.exceptions import McpError
from fastmcp_tasks import TasksExtension
from fastmcp.server.auth import AccessToken, TokenVerifier
from fastmcp.server.dependencies import Progress, get_access_token
from fastmcp.server.providers.skills import SkillProvider
from mcp.types import INTERNAL_ERROR

# Ensure sibling modules are importable when run via `uv run`
sys.path.insert(0, str(Path(__file__).parent))

from _auth import (  # noqa: E402
    bootstrap_admin_token, validate_token, create_user_token, revoke_user, rotate_user_token, list_users,
    _acquire_lock, _release_lock,
    LOCAL_USER_ID, ANONYMOUS_USER_ID,
)
from _config import TrainingConfig, TrainingResult  # noqa: E402
from _data_registry import (  # noqa: E402
    touch_registry_entry, register_dataset, update_dataset, remove_dataset,
    get_dataset, list_datasets_for_owner, data_registry_path,
    load_json_list, atomic_write_json, _lock_path as _dr_lock_path,
)
from _discovery import get_all_options, list_base_estimators, list_blender_estimators, list_dim_reduction_methods, list_feature_generator_aliases, list_feature_generators  # noqa: E402
from _pipeline import run_full_pipeline, _plugin_root, _scripts_dir, _venv_path, _output_root, _run_script_sync, _under_output_root, _active_run_folders  # noqa: E402
from _sanitize import sanitize_model_entry, sanitize_train_result, sanitize_predict_result  # noqa: E402


def _check_environment() -> None:
    """Validate the runtime environment at server startup.

    Emits a clear, actionable error rather than a deep traceback later.
    """
    venv = _venv_path()
    python = venv / "bin" / "python"
    if not python.exists():
        raise RuntimeError(
            f"Python not found at {python}.\n"
            f"Set AUTOMOL_VENV to your virtual environment path and restart.\n"
            f"Example: export AUTOMOL_VENV=$HOME/.venvs/molagent"
        )
    output_root = _output_root()
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        probe = output_root / ".write_probe"
        probe.touch()
        probe.unlink()
    except OSError as exc:
        raise RuntimeError(
            f"Output directory {output_root} is not writable: {exc}\n"
            f"Set MOLAGENT_OUTPUT_ROOT to a writable path."
        ) from exc


_check_environment()

# Warm all discovery caches at startup (before event loop starts).
# Library imports (xgboost, lightgbm, onnxruntime) write directly to fd 1 (stdout)
# on first load, which corrupts the MCP stdio transport pipe. We suppress fd 1
# during preload to prevent any C-level writes from reaching the pipe.
_saved_fd = os.dup(1)
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, 1)
os.close(_devnull)
try:
    get_all_options("Regression")
    get_all_options("Classification")
    get_all_options("RegressionClassification")
finally:
    os.dup2(_saved_fd, 1)
    os.close(_saved_fd)

def _valid_feature_keys() -> set[str]:
    """Feature keys accepted from callers: canonical listed keys plus aliases."""
    return (set(list_feature_generators()) | set(list_feature_generator_aliases())) - {"_note"}


logger = logging.getLogger(__name__)

_AUTH_REQUIRED = os.environ.get("MOLAGENT_AUTH_REQUIRED", "").lower() in ("1", "true", "yes")


# ── Auth provider for FastMCP ─────────────────────────────────────────────────

class _MolAgentTokenVerifier(TokenVerifier):
    """Validates Bearer tokens against the local token store."""

    async def verify_token(self, token: str) -> AccessToken | None:
        result = validate_token(token)
        if result is None:
            return None
        return AccessToken(
            token=token,
            client_id=result["user_id"],
            scopes=["admin"] if result["is_admin"] else [],
            claims=result,
        )


_auth_provider = _MolAgentTokenVerifier() if _AUTH_REQUIRED else None


def _should_expose_paths() -> bool:
    """True when internal paths can be shown (local plugin mode, no web backend)."""
    return not _AUTH_REQUIRED


def _get_caller(ctx: Context) -> dict | None:
    """Extract the authenticated caller from the MCP context.

    Resolution order:
      1. Auth not required → None (stdio/local superuser).
      2. Transport-verified token (streamable-http) → its claims.
      3. stdio has no transport-level auth, so a caller token supplied via the
         MOLAGENT_CALLER_TOKEN env var is validated here. This is the web app's
         "local + token" mode: it spawns the server over stdio but wants a real
         (typically non-admin) identity for per-user isolation.
      4. Otherwise → anonymous (rejected by _require_auth).
    """
    if not _AUTH_REQUIRED:
        return None

    # Transport-verified token (remote / streamable-http)
    access_token = get_access_token()
    if access_token is not None:
        return access_token.claims

    # stdio transport: accept a validated env-provided caller token
    env_token = os.environ.get("MOLAGENT_CALLER_TOKEN")
    if env_token:
        result = validate_token(env_token)
        if result:
            return result

    return {"user_id": ANONYMOUS_USER_ID, "is_admin": False}


def _require_auth(caller: dict | None) -> dict:
    """Raise McpError if caller is unauthenticated in a remote context."""
    if caller is None:
        return {"user_id": LOCAL_USER_ID, "is_admin": True}
    if caller["user_id"] == ANONYMOUS_USER_ID:
        raise McpError(
            code=INTERNAL_ERROR,
            message="Authentication required. Provide a valid Bearer token.",
        )
    return caller


def _caller_privileges(caller: dict) -> tuple[bool, bool]:
    """Return (is_local, is_admin) from a caller dict."""
    is_local = caller["user_id"] == LOCAL_USER_ID
    is_admin = caller.get("is_admin", False) or is_local
    return is_local, is_admin


def _ensure_path_access(caller: dict, *, what: str) -> None:
    """Block direct filesystem-path inputs for non-admin remote callers.

    Local/stdio callers and admins may pass absolute paths; everyone else must
    reference data via dataset_id so they cannot read arbitrary server files.
    """
    _, is_admin = _caller_privileges(caller)
    if not is_admin:
        raise McpError(
            code=INTERNAL_ERROR,
            message=(
                f"Direct {what} path access is not allowed. Upload your data "
                "with upload_dataset and reference it by dataset_id."
            ),
        )


def _resolve_data_reference(caller: dict, value: str, *, what: str) -> str:
    """Resolve a csv/smiles input to a filesystem path.

    A ``ds_``-prefixed value is always treated as a data-registry id — resolved
    and ownership-checked against the registry, never as a filesystem path — so
    a file literally named ``ds_*`` can never shadow a real dataset. Any other
    value is a direct path, permitted only for admin/local callers.
    """
    _, is_admin = _caller_privileges(caller)
    if value.startswith("ds_"):
        ds_entry = get_dataset(value)
        if ds_entry is not None:
            if not is_admin and ds_entry.get("owner") != caller.get("owner_id"):
                raise McpError(
                    code=INTERNAL_ERROR,
                    message=f"Access denied: dataset '{value}' does not belong to you.",
                )
            touch_registry_entry(data_registry_path(), ds_entry["id"])
            if ds_entry.get("type") == "3d_structure":
                # file_path for a 3d_structure entry is the raw structure
                # directory (SDF + pdbs/), not a CSV — the training CSV is the
                # prepared/data.csv that start_training_session's extract_3d_data
                # call already generated inside it.
                prepared_csv = _output_root() / ds_entry["file_path"] / "prepared" / "data.csv"
                if not prepared_csv.exists():
                    raise McpError(
                        code=INTERNAL_ERROR,
                        message=(
                            f"No extracted training CSV found for structure '{value}'. "
                            "Call start_training_session(dataset_id=...) on it first."
                        ),
                    )
                return str(prepared_csv)
            return str(_output_root() / ds_entry["file_path"])
        # Not a known dataset id. Non-admins cannot fall back to a raw path.
        if not is_admin:
            raise McpError(
                code=INTERNAL_ERROR,
                message=f"Dataset '{value}' not found in data registry.",
            )
        # Admin: may legitimately be a real path that starts with 'ds_'.
    _ensure_path_access(caller, what=what)
    return value


# Max upload payload size (decoded bytes). Override with MOLAGENT_MAX_UPLOAD_MB.
def _max_upload_bytes() -> int:
    try:
        mb = float(os.environ.get("MOLAGENT_MAX_UPLOAD_MB", "100"))
    except ValueError:
        mb = 100.0
    return int(mb * 1024 * 1024)


def _delete_model_entry_files(entry: dict, errors: list[str] | None = None) -> int:
    """Delete the on-disk files for a model registry entry.

    Prefers the run folder; falls back to individual model_file path(s) so
    nothing orphans. Returns the number of paths removed. Errors are appended
    to ``errors`` if provided, else logged.
    """
    def _note(msg: str) -> None:
        if errors is not None:
            errors.append(msg)
        else:
            logger.warning(msg)

    removed = 0
    run_folder = entry.get("run_folder")
    if run_folder and Path(run_folder).exists():
        try:
            shutil.rmtree(_under_output_root(Path(run_folder)))
            removed += 1
        except ValueError:
            _note(f"Skipped run_folder outside output root: {run_folder}")
        except OSError as exc:
            _note(f"Failed to delete {run_folder}: {exc}")
        return removed

    model_files = entry.get("model_file", [])
    if isinstance(model_files, str):
        model_files = [model_files]
    for mf in model_files:
        p = Path(mf)
        if p.exists():
            try:
                p.unlink()
                removed += 1
            except OSError as exc:
                _note(f"Failed to delete {mf}: {exc}")
    return removed


mcp = FastMCP(
    "automol-mcp",
    auth=_auth_provider,
    instructions=(
        "AutoMol molecular property prediction pipeline.\n\n"
        "TASK TYPES:\n"
        "- Regression: predict continuous numeric values\n"
        "- Classification: predict discrete classes using classification estimators\n"
        "- RegressionClassification: binary classification using REGRESSION estimators "
        "on 0/1 labels — predictions clipped to [0,1] as class probabilities. "
        "This is NOT 'run both tasks simultaneously'.\n\n"
        "DOMAIN TERMS:\n"
        "- blender_properties: auxiliary numeric columns used as extra input features "
        "(not targets)\n"
        "- feature_keys: molecular representation methods (Bottleneck, Bottleneck_chembl37_base, Bottleneck_chembl27, rdkit, fps_*), "
        "not CSV column names\n"
        "- computational_load: runtime budget "
        "(free ~2min, cheap ~10min, moderate ~1hr, expensive ~24hr)\n\n"
        "WORKFLOW:\n"
        "Discovery: list_options(category=...) for available options.\n"
        "Training: start_training_session → present config → answer_training_question "
        "to override/confirm → train_and_visualize (long-running).\n"
        "Prediction: list_models → predict(model_id=..., smiles_list/smiles_file=...).\n"
        "Merging: merge_models(model_ids=[...]) → single multi-property model.\n"
        "Download: download_model(model_id=...) → base64 binary."
    ),
)
mcp.add_extension(TasksExtension())

# ── Skill resource (provides orchestration guidance to any MCP client) ────────
_skill_dir = Path(__file__).parent / "skills" / "automol-pipeline"
if _skill_dir.exists():
    mcp.add_provider(SkillProvider(_skill_dir))


# ── Session state ─────────────────────────────────────────────────────────────

SESSION_TTL = 90 * 60  # 90 minutes in seconds
_sessions: dict[str, "_SessionState"] = {}
_sessions_lock = threading.Lock()
_cleanup_timer: threading.Timer | None = None


@dataclass
class _SessionState:
    session_id: str
    csv_file: str
    csv_columns: list[str]
    config: dict
    last_touched: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_touched = time.monotonic()


def _schedule_cleanup() -> None:
    """Start the background eviction timer. Called lazily on first session creation."""
    global _cleanup_timer
    if _cleanup_timer is not None:
        return

    def _evict() -> None:
        now = time.monotonic()
        with _sessions_lock:
            expired = [sid for sid, s in list(_sessions.items())
                       if now - s.last_touched > SESSION_TTL]
            for sid in expired:
                del _sessions[sid]
        t = threading.Timer(300, _evict)
        t.daemon = True
        t.start()
        global _cleanup_timer
        _cleanup_timer = t

    t = threading.Timer(300, _evict)
    t.daemon = True
    t.start()
    _cleanup_timer = t


def _get_session(session_id: str) -> _SessionState:
    """Retrieve a session, raising McpError if missing or expired. Thread-safe."""
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is None:
            raise McpError(
                code=INTERNAL_ERROR,
                message=(
                    f"Session '{session_id}' not found or expired. "
                    "Call start_training_session to begin a new session."
                ),
            )
        if time.monotonic() - session.last_touched > SESSION_TTL:
            del _sessions[session_id]
            raise McpError(
                code=INTERNAL_ERROR,
                message=(
                    f"Session '{session_id}' has expired (90 min timeout). "
                    "Call start_training_session to begin a new session."
                ),
            )
        session.touch()
        return session


# ── Tool 0: list_options ─────────────────────────────────────────────────────

@mcp.tool
async def list_options(
    category: Literal[
        "feature_generators", "base_estimators", "blender_estimators",
        "dim_reduction", "model_configs", "search_types", "scorers",
    ],
    ctx: Context,
    task: Literal["Regression", "Classification", "RegressionClassification"] = "Regression",
) -> dict:
    """Discover available training options installed on this server.

    Call before configuring training to see what feature generators, estimators,
    dimensionality reduction methods, model architectures, and scoring metrics
    are available. Results depend on what libraries are installed.

    Args:
        category: Which option category to list.
        task: Filter estimators and scorers by task type (default: Regression).
    """
    caller = _get_caller(ctx)
    _require_auth(caller)
    all_opts = get_all_options(task)
    return all_opts[category]


# ── Tool 1: start_training_session ────────────────────────────────────────────

@mcp.tool
async def start_training_session(
    ctx: Context,
    csv_file: Optional[str] = None,
    dataset_id: Optional[str] = None,
    property_key: Optional[str] = None,
) -> dict:
    """Start a new training configuration session for a CSV dataset or 3D structure.

    Runs dataset auto-detection and returns the detected defaults along with
    the valid options for each configurable field. Always creates a fresh
    session — safe to call multiple times with the same dataset.

    Provide EITHER csv_file (direct path) OR dataset_id (from upload_dataset,
    upload_3d_structure, or list_datasets). A dataset_id from upload_3d_structure
    is detected automatically by its registry entry type and routed through the
    same SDF-extraction path as before.

    Returns a dict with:
      - session_id: pass this to answer_training_question
      - detected: auto-detected values for all config fields
      - options: valid choices for each field (with descriptions). feature_keys
        only includes prolif/AffGraph when the resolved dataset is a 3D structure.
      - question: message to present to the user
      - structure_extraction: present only when dataset_id resolved to a 3D
        structure — the SDF property_key actually applied and how many
        molecules were successfully extracted, so the caller can tell the
        user before they confirm via answer_training_question.

    Args:
        csv_file: Absolute path to the input CSV file containing SMILES and properties.
        dataset_id: Dataset ID from the data registry (alternative to csv_file).
            Works for both plain CSV datasets and 3D structures (upload_3d_structure).
        property_key: SDF property name to extract as the training target when
            dataset_id resolves to a 3D structure (e.g. "pIC50", "Ki"). Defaults
            to "pChEMBL" if not given. Ignored for plain CSV datasets.
    """
    caller = _get_caller(ctx)
    caller = _require_auth(caller)
    _, is_admin = _caller_privileges(caller)

    # Track the registry id of the dataset so the finalized config can reference
    # it by id (remote callers) instead of an absolute server path.
    session_dataset_id: str | None = None
    # Populated only when dataset_id resolves to a 3d_structure entry, so the
    # caller can surface the detected property_key + parsed molecule count to
    # the user before they confirm.
    structure_extraction_info: dict | None = None
    # Absolute paths, populated only for a 3d_structure dataset_id — carried on
    # the session so answer_training_question can copy them into the finalized
    # TrainingConfig as sdf_file/protein_folder.
    session_sdf_path: str | None = None
    session_pdb_folder: str | None = None

    # Resolve CSV path from dataset_id or direct csv_file
    if dataset_id is not None:
        ds_entry = get_dataset(dataset_id)
        if ds_entry is None:
            raise McpError(
                code=INTERNAL_ERROR,
                message=f"Dataset '{dataset_id}' not found in data registry.",
            )
        if not is_admin and ds_entry.get("owner") != caller.get("owner_id"):
            raise McpError(
                code=INTERNAL_ERROR,
                message=f"Access denied: dataset '{dataset_id}' does not belong to you.",
            )
        session_dataset_id = dataset_id
        touch_registry_entry(data_registry_path(), dataset_id)
        if ds_entry.get("type") == "3d_structure":
            # Resolve 3D structure to CSV via extract_3d_data (same path as the
            # old structure_id branch, now selected by registry entry type).
            from _3d_tools import extract_3d_data
            struct_dir = _output_root() / ds_entry["file_path"]
            sdf_path = struct_dir / "Selected_dockings.sdf"
            if not sdf_path.exists():
                raise McpError(
                    code=INTERNAL_ERROR,
                    message=f"SDF file not found at expected path: {sdf_path}",
                )
            resolved_property_key = property_key or "pChEMBL"
            extract_result = extract_3d_data(
                sdf_file=str(sdf_path),
                property_key=resolved_property_key,
                data_dir=str(struct_dir / "prepared"),
                file_nm="data.csv",
            )
            csv_file = extract_result["data_file"]
            session_sdf_path = extract_result["sdf_file"]
            session_pdb_folder = extract_result["pdb_folder"]
            structure_extraction_info = {
                "property_key": resolved_property_key,
                "mol_count": extract_result["mol_count"],
            }
        else:
            csv_file = str(_output_root() / ds_entry["file_path"])
    elif csv_file is not None:
        _ensure_path_access(caller, what="csv_file")
        # Register the CSV in the data registry if not already tracked
        csv_path_check = Path(csv_file)
        if not csv_path_check.exists():
            raise McpError(code=INTERNAL_ERROR, message=f"CSV file not found: {csv_file}")
        owner_id = caller.get("owner_id", LOCAL_USER_ID)
        # Check if already registered (match by absolute path)
        abs_csv = str(csv_path_check.resolve())
        existing = load_json_list(data_registry_path())
        already_registered = next(
            (e for e in existing if str((_output_root() / e.get("file_path", "")).resolve()) == abs_csv),
            None,
        )
        if already_registered:
            session_dataset_id = already_registered["id"]
            touch_registry_entry(data_registry_path(), already_registered["id"])
        else:
            # Copy file into per-user uploads dir and register
            import shutil as _shutil
            uploads_dir = _output_root() / "uploads" / owner_id
            uploads_dir.mkdir(parents=True, exist_ok=True)
            dest = uploads_dir / csv_path_check.name
            if dest.resolve() != csv_path_check.resolve():
                # Deduplicate filename if target already exists
                if dest.exists():
                    stem, suffix = dest.stem, dest.suffix
                    n = 2
                    while dest.exists():
                        dest = uploads_dir / f"{stem}_{n}{suffix}"
                        n += 1
                _shutil.copy2(csv_file, dest)
            else:
                dest = csv_path_check
            rel_path = str(dest.resolve().relative_to(_output_root()))
            import csv as _csv_mod
            import io as _io_mod
            try:
                text = dest.read_text(encoding="utf-8", errors="replace")
                reader = _csv_mod.reader(_io_mod.StringIO(text))
                columns = next(reader, [])
                row_count = sum(1 for _ in reader)
            except Exception:
                columns, row_count = [], 0
            new_entry = register_dataset(
                owner_id=owner_id,
                filename=dest.name,
                file_path=rel_path.replace("\\", "/"),
                size_bytes=dest.stat().st_size,
                columns=columns,
                row_count=row_count,
            )
            session_dataset_id = new_entry["id"]
    else:
        raise McpError(
            code=INTERNAL_ERROR,
            message="Provide either csv_file or dataset_id.",
        )

    csv_path = Path(csv_file)
    if not csv_path.exists():
        raise McpError(code=INTERNAL_ERROR, message=f"CSV file not found: {csv_file}")

    # ── Read CSV columns ──────────────────────────────────────────────────────
    try:
        df_head = pd.read_csv(csv_path, nrows=1)
        csv_columns = list(df_head.columns)
    except Exception as exc:
        raise McpError(code=INTERNAL_ERROR, message=f"Could not read CSV columns: {exc}")

    # ── Run detect_dataset.py ─────────────────────────────────────────────────
    detect_script = _scripts_dir() / "detect_dataset.py"
    detection_warning: str | None = None
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["uv", "run", "--active", "--no-sync", str(detect_script), "--csv-file", csv_file],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            cwd=str(_scripts_dir()),
            env={**os.environ, "VIRTUAL_ENV": str(_venv_path()), "PYTHONPATH": str(_scripts_dir())},
        )
        if result.returncode == 0:
            detection = json.loads(result.stdout)
        else:
            detection = {}
            detection_warning = (result.stderr or "non-zero exit").strip()[:300]
    except Exception as exc:
        detection = {}
        detection_warning = str(exc)[:300]

    detected_smiles = detection.get("smiles_column", "")
    detected_targets = [t["column"] for t in detection.get("targets", [])]
    detected_task_raw = detection.get("overall_task_type", "regression")
    task_map = {
        "regression": "Regression",
        "classification": "Classification",
        "regression_classification": "RegressionClassification",
    }
    detected_task = task_map.get(detected_task_raw, "Regression")
    recs = detection.get("recommendations", {})
    rec_load = recs.get("computational_load", {}).get("value", "cheap")
    rec_transform = recs.get("target_transformations", [])
    suggest_log10 = any(t.get("transform") == "log10" for t in rec_transform)

    # Classification defaults from detection
    clf_kwargs: dict = {}
    if detected_task in ("Classification", "RegressionClassification"):
        targets_info = detection.get("targets", [])
        clf_targets = [t for t in targets_info if t.get("task_type") == "classification"]
        if clf_targets:
            all_categorical = all(t.get("suggested_categorical", False) for t in clf_targets)
            clf_kwargs["categorical"] = all_categorical
            if not all_categorical:
                clf_kwargs["nb_classes"] = [t.get("suggested_nb_classes", 2) for t in clf_targets]

    # ── Build initial config from detected defaults ───────────────────────────
    # Remote callers get a dataset_id reference (resolved + ownership-checked at
    # train time) rather than an absolute server path they aren't allowed to use.
    config_csv_ref = (
        str(csv_path.resolve())
        if (is_admin or session_dataset_id is None)
        else session_dataset_id
    )
    config = TrainingConfig(
        csv_file=config_csv_ref,
        smiles_column=detected_smiles or (csv_columns[0] if csv_columns else ""),
        properties=detected_targets or [],
        task=detected_task,
        computational_load=rec_load,
        feature_keys=["Bottleneck"],
        use_log10=suggest_log10,
        split_strategy="mixed",
        sdf_file=session_sdf_path,
        protein_folder=session_pdb_folder,
        **clf_kwargs,
    ).model_dump()

    # ── Create session (lazy-start cleanup timer on first use) ────────────────
    session_id = str(uuid.uuid4())
    session = _SessionState(
        session_id=session_id,
        csv_file=str(csv_path.resolve()),
        csv_columns=csv_columns,
        config=config,
    )
    with _sessions_lock:
        _sessions[session_id] = session
        _schedule_cleanup()

    # Surface data-quality warnings from detection
    data_warnings = recs.get("warnings", [])
    characteristics = detection.get("characteristics", {})
    smiles_validity_rate = characteristics.get("smiles_validity_rate")

    question = (
        "Here is the detected configuration. "
        "Would you like to change anything, or shall I proceed with training?"
    )
    if data_warnings:
        question += "\n\nWarnings:\n" + "\n".join(f"- {w}" for w in data_warnings)
    if smiles_validity_rate is not None and smiles_validity_rate < 50:
        question = (
            f"SMILES validity is very low ({smiles_validity_rate}%). "
            "The detected SMILES column may be incorrect. "
            f"Available columns: {', '.join(csv_columns)}. "
            "Please verify or change the smiles_column before proceeding."
        )
    if detection_warning:
        question = (
            f"Auto-detection failed ({detection_warning}). "
            f"Available columns are: {', '.join(csv_columns)}. "
            "Please specify the SMILES column and target property column(s), "
            "then confirm when ready."
        )

    # Surface the 3D-structure extraction outcome (property_key used + how many
    # molecules had a valid value for it) so the user sees it before confirming.
    if structure_extraction_info is not None:
        note = (
            f"\n\nExtracted {structure_extraction_info['mol_count']} molecule(s) from the "
            f"3D structure using property_key='{structure_extraction_info['property_key']}'."
        )
        if structure_extraction_info["mol_count"] == 0:
            note += (
                " No molecules had a valid value for this property — verify property_key "
                "matches an SDF property name (e.g. pIC50, Ki, pChEMBL) and start a new session."
            )
        question += note

    session_has_structure = structure_extraction_info is not None
    options_feature_keys = [
        k for k, v in list_feature_generators().items()
        if k != "_note" and (
            not (isinstance(v, dict) and v.get("requires_3d_structure"))
            or session_has_structure
        )
    ]

    return {
        "session_id": session_id,
        "detected": {
            "csv_file": config["csv_file"],
            "smiles_column": config["smiles_column"],
            "properties": config["properties"],
            "task": config["task"],
            "use_log10": config["use_log10"],
            "feature_keys": config["feature_keys"],
            "split_strategy": config["split_strategy"],
            "computational_load": config["computational_load"],
            "categorical": config.get("categorical", False),
            "nb_classes": config.get("nb_classes"),
            "sdf_file": config.get("sdf_file"),
            "protein_folder": config.get("protein_folder"),
        },
        "options": {
            "task": {
                "Regression": "Predict continuous values (pIC50, solubility, etc.)",
                "Classification": "Predict discrete classes using classification estimators",
                "RegressionClassification": (
                    "Binary classification via regression estimators — targets must be "
                    "0/1; predictions clipped to [0,1] as class probabilities. "
                    "NOT 'both tasks'."
                ),
            },
            "computational_load": {
                "free": "~0–2 min, single method with hyperparameter search",
                "cheap": "~2–10 min, light search",
                "moderate": "~10–360 min, stacking",
                "expensive": "1–48 hr, full hyperopt",
            },
            "feature_keys": options_feature_keys,
            "split_strategy": {
                "mixed": "stratified + scaffold + activity cliffs (recommended)",
                "stratified": "class-balanced splits",
                "leave_group_out": "scaffold-based, strict generalization test",
            },
            "smiles_column": csv_columns,
            "properties": [c for c in csv_columns if c != config["smiles_column"]],
            "base_estimators": list_base_estimators(config["task"]),
            "blender_estimators": list_blender_estimators(config["task"]),
            "dim_reduction": list_dim_reduction_methods(),
            "model_configs": get_all_options(config["task"])["model_configs"],
            "search_types": get_all_options(config["task"])["search_types"],
            "scorers": get_all_options(config["task"])["scorers"],
        },
        "question": question,
        "targets": detection.get("targets", []),
        "blender_properties": detection.get("blender_properties", []),
        "characteristics": characteristics,
        "structure_extraction": structure_extraction_info,
    }


# ── Tool 2: answer_training_question ──────────────────────────────────────────

@mcp.tool
async def answer_training_question(
    session_id: str,
    ctx: Context,
    confirm: Optional[bool] = None,
    smiles_column: Optional[str] = None,
    properties: Optional[list[str]] = None,
    task: Optional[Literal["Regression", "Classification", "RegressionClassification"]] = None,
    computational_load: Optional[Literal["free", "cheap", "moderate", "expensive"]] = None,
    feature_keys: Optional[list[str]] = None,
    use_log10: Optional[bool] = None,
    use_logit: Optional[bool] = None,
    split_strategy: Optional[Literal["mixed", "stratified", "leave_group_out"]] = None,
    base_list: Optional[list[str]] = None,
    blender_list: Optional[list[str]] = None,
    red_dim_list: Optional[list[str]] = None,
    model_config: Optional[Literal[
        "single_method", "inner_methods", "inner_stacking",
        "single_stack", "top_method", "top_stacking", "stacking_stacking",
    ]] = None,
    search_type: Optional[Literal["grid", "randomized", "hyperopt"]] = None,
    randomized_iterations: Optional[int] = None,
    scorer: Optional[str] = None,
    # Classification options
    categorical: Optional[bool] = None,
    nb_classes: Optional[list[int]] = None,
    class_values: Optional[list] = None,
    class_quantiles: Optional[list] = None,
    # Refit control
    refit: Optional[bool] = None,
    include_test_in_refit: Optional[bool] = None,
    # Sample weights
    use_sample_weight: Optional[bool] = None,
    sample_weight_selection: Optional[str] = None,
    sample_weight_multiplier: Optional[float] = None,
) -> dict:
    """Apply user overrides to the training config or confirm and finalize it.

    The LLM (Claude) parses the user's natural language into typed parameter
    values — no string parsing happens inside this tool.

    Supply any fields the user wants to change. Set confirm=True (with or
    without overrides) to finalize the config and end the session.

    Use list_options to discover valid keys for feature_keys, base_list,
    blender_list, red_dim_list, and scorer before setting them.

    Args:
        session_id: Session ID returned by start_training_session.
        confirm: True to finalize the config and return it for training.
        smiles_column: Override the SMILES column name.
        properties: Override the list of target property columns.
        task: Override the modelling task type.
        computational_load: Override the computational budget.
        feature_keys: Override the feature generators.
        use_log10: Override the log10 transform flag.
        use_logit: Override the logit transform flag (for bounded 0-1 targets).
        split_strategy: Override the validation split strategy.
        base_list: Override base estimator keys (e.g. ["xgb", "lgbm", "lasso"]).
        blender_list: Override blender/final estimator keys.
        red_dim_list: Override dimensionality reduction methods.
        model_config: Override model hierarchy configuration.
        search_type: Override hyperparameter search type.
        randomized_iterations: Override iterations for randomized/hyperopt search.
        scorer: Override scoring metric (e.g. "r2", "balanced_accuracy").
        categorical: True if targets are already categorical (0/1/2...).
        nb_classes: Number of classes per property (for continuous-to-class conversion).
        class_values: Explicit class cutoff thresholds per property.
        class_quantiles: Quantile-based class cutoffs per property.
        refit: Whether to refit model on full data after evaluation (default True).
        include_test_in_refit: Include test set in refit (default True).
        use_sample_weight: Enable sample weighting during preparation.
        sample_weight_selection: Selection criterion for weighting (e.g. '<1', '>5').
        sample_weight_multiplier: Weight multiplier for selected samples (1-1000).
    """
    caller = _get_caller(ctx)
    _require_auth(caller)
    session = _get_session(session_id)
    config = session.config
    validation_messages: list[str] = []

    # ── Apply typed overrides with column validation ──────────────────────────
    if smiles_column is not None:
        if smiles_column not in session.csv_columns:
            validation_messages.append(
                f"Column '{smiles_column}' not found in CSV. "
                f"Available columns: {', '.join(session.csv_columns)}."
            )
        else:
            config["smiles_column"] = smiles_column

    if properties is not None:
        invalid = [p for p in properties if p not in session.csv_columns]
        if invalid:
            validation_messages.append(
                f"Column(s) {invalid} not found in CSV. "
                f"Available columns: {', '.join(session.csv_columns)}."
            )
        else:
            config["properties"] = properties

    if task is not None:
        config["task"] = task
    if computational_load is not None:
        config["computational_load"] = computational_load
    if feature_keys is not None:
        import re
        _fps_pattern = re.compile(r"^fps_\d+_\d+$")
        _all_generators = list_feature_generators()
        valid_features = _valid_feature_keys()
        invalid = [k for k in feature_keys if k not in valid_features and not _fps_pattern.match(k)]
        requires_3d = [
            k for k in feature_keys
            if k in valid_features
            and isinstance(_all_generators.get(k), dict)
            and _all_generators[k].get("requires_3d_structure")
            and not config.get("sdf_file")
        ]
        if invalid:
            validation_messages.append(
                f"Unknown feature generators: {invalid}. "
                "Use list_options(category='feature_generators') to see available options."
            )
        if requires_3d:
            validation_messages.append(
                f"Feature generator(s) {requires_3d} require a 3D structure dataset "
                "(no sdf_file/protein_folder set in this session)."
            )
        if not invalid and not requires_3d:
            config["feature_keys"] = feature_keys
    if use_log10 is not None:
        config["use_log10"] = use_log10
    if use_logit is not None:
        config["use_logit"] = use_logit
    if split_strategy is not None:
        config["split_strategy"] = split_strategy

    # Classification options
    if categorical is not None:
        config["categorical"] = categorical
    if nb_classes is not None:
        config["nb_classes"] = nb_classes
    if class_values is not None:
        config["class_values"] = class_values
    if class_quantiles is not None:
        config["class_quantiles"] = class_quantiles

    # Refit control
    if refit is not None:
        config["refit"] = refit
    if include_test_in_refit is not None:
        config["include_test_in_refit"] = include_test_in_refit

    # Sample weights
    if use_sample_weight is not None:
        config["use_sample_weight"] = use_sample_weight
    if sample_weight_selection is not None:
        config["sample_weight_selection"] = sample_weight_selection
    if sample_weight_multiplier is not None:
        config["sample_weight_multiplier"] = sample_weight_multiplier

    # ── Tier 2: advanced model configuration overrides with validation ────────
    current_task = config.get("task", "Regression")
    if base_list is not None:
        valid_keys = set(list_base_estimators(current_task).keys())
        invalid = [k for k in base_list if k not in valid_keys]
        if invalid:
            validation_messages.append(
                f"Unknown base estimators: {invalid}. "
                "Use list_options(category='base_estimators') to see available options."
            )
        else:
            config["base_list"] = base_list
    if blender_list is not None:
        valid_keys = set(list_blender_estimators(current_task).keys())
        invalid = [k for k in blender_list if k not in valid_keys]
        if invalid:
            validation_messages.append(
                f"Unknown blender estimators: {invalid}. "
                "Use list_options(category='blender_estimators') to see available options."
            )
        else:
            config["blender_list"] = blender_list
    if red_dim_list is not None:
        valid_keys = set(list_dim_reduction_methods().keys())
        invalid = [k for k in red_dim_list if k not in valid_keys]
        if invalid:
            validation_messages.append(
                f"Unknown dim reduction methods: {invalid}. "
                "Use list_options(category='dim_reduction') to see available options."
            )
        else:
            config["red_dim_list"] = red_dim_list
    if model_config is not None:
        config["ensemble_config"] = model_config
    if search_type is not None:
        config["search_type"] = search_type
    if randomized_iterations is not None:
        config["randomized_iterations"] = randomized_iterations
    if scorer is not None:
        config["scorer"] = scorer

    # ── Return validation errors without finalizing ───────────────────────────
    if validation_messages:
        return {
            "session_id": session_id,
            "validation_error": True,
            "question": " ".join(validation_messages) + " Please choose valid column names.",
            "config": None,
        }

    # ── Finalize if confirmed ─────────────────────────────────────────────────
    if confirm:
        # Re-validate task-dependent fields against the final task
        final_task = config.get("task", "Regression")
        if config.get("base_list"):
            valid_keys = set(list_base_estimators(final_task).keys())
            bad = [k for k in config["base_list"] if k not in valid_keys]
            if bad:
                validation_messages.append(f"Base estimators {bad} not valid for task={final_task}.")
        if config.get("blender_list"):
            valid_keys = set(list_blender_estimators(final_task).keys())
            bad = [k for k in config["blender_list"] if k not in valid_keys]
            if bad:
                validation_messages.append(f"Blender estimators {bad} not valid for task={final_task}.")
        if validation_messages:
            return {
                "session_id": session_id,
                "validation_error": True,
                "question": " ".join(validation_messages),
                "config": None,
            }
        try:
            final = TrainingConfig.model_validate(config)
        except Exception as exc:
            raise McpError(code=INTERNAL_ERROR, message=f"Invalid config: {exc}")
        with _sessions_lock:
            _sessions.pop(session_id, None)
        return {
            "session_id": session_id,
            "validation_error": False,
            "question": None,
            "config": final.model_dump(),
        }

    # ── Overrides applied, ask for further changes ────────────────────────────
    return {
        "session_id": session_id,
        "validation_error": False,
        "question": "Overrides applied. Anything else to change, or shall I proceed with training?",
        "config": None,
    }


_STEP_LABELS: dict[int, str] = {
    1: "Preparing data",
    2: "Splitting data",
    3: "Training model",
    4: "Merging models",
    5: "Evaluating model",
    6: "Refitting model",
    7: "Merging refitted models",
    8: "Generating dashboard",
}


# ── Tool 3: train_and_visualize ───────────────────────────────────────────────

@mcp.tool(task=True)
async def train_and_visualize(config: dict, ctx: Context, progress: Progress = Progress()) -> dict:
    """Run the full AutoMol pipeline deterministically and return results.

    Accepts a config dict as returned by answer_training_question (or built
    manually). Runs prepare → split → train → evaluate → refit → dashboard.

    This is a long-running tool (minutes to hours). It supports background
    task execution — callers receive a task_id immediately and can poll for
    progress/status/result.

    Returns a TrainingResult dict containing:
      - model_id: registry ID (use download_model to fetch the binary)
      - dashboard_html: self-contained interactive HTML dashboard
      - metrics: per-property evaluation metrics
      - model_path / dashboard_path: absolute filesystem paths (local mode only)

    Args:
        config: TrainingConfig dict — use the session tools to build this.
    """
    caller = _get_caller(ctx)
    caller = _require_auth(caller)

    try:
        training_config = TrainingConfig.model_validate(config)
    except Exception as exc:
        raise McpError(code=INTERNAL_ERROR, message=f"Invalid config: {exc}")

    # Resolve csv_file: a ds_ value is a registry id (ownership checked),
    # anything else is a direct path (admin/local only).
    csv_file_val = _resolve_data_reference(caller, training_config.csv_file, what="csv_file")
    training_config.csv_file = csv_file_val

    csv_path = Path(csv_file_val)
    if not csv_path.exists():
        raise McpError(code=INTERNAL_ERROR, message=f"CSV file not found: {csv_file_val}")

    # Validate feature_keys before starting the pipeline so callers get a clear
    # error rather than an AssertionError deep inside the training script.
    if training_config.feature_keys:
        import re as _re
        _fps_pat = _re.compile(r"^fps_\d+_\d+$")
        _all_gen = list_feature_generators()
        _valid = _valid_feature_keys()
        _unknown = [
            k for k in training_config.feature_keys
            if k not in _valid and not _fps_pat.match(k)
        ]
        _needs_3d = [
            k for k in training_config.feature_keys
            if k in _all_gen
            and isinstance(_all_gen.get(k), dict)
            and _all_gen[k].get("requires_3d_structure")
            and not training_config.sdf_file
        ]
        if _unknown:
            raise McpError(
                code=INTERNAL_ERROR,
                message=(
                    f"Unknown feature generator(s): {_unknown}. "
                    "Use list_options(category='feature_generators') to see valid options."
                ),
            )
        if _needs_3d:
            raise McpError(
                code=INTERNAL_ERROR,
                message=(
                    f"Feature generator(s) {_needs_3d} require a 3D structure dataset "
                    "(upload one with upload_3d_structure first, or remove these keys)."
                ),
            )

    # output_folder is an arbitrary filesystem path written to by the pipeline —
    # gate it the same way csv_file/smiles_file/model_file are gated, so a
    # non-admin remote caller cannot direct writes outside their sandbox.
    if training_config.output_folder is not None:
        _ensure_path_access(caller, what="output_folder")

    total_steps = 8
    owner = caller.get("owner_id") if caller["user_id"] != LOCAL_USER_ID else None

    await progress.set_total(total_steps)

    async def progress_cb(step: int, total: int) -> None:
        try:
            label = _STEP_LABELS.get(step, f"Step {step}")
            await progress.set_message(label)
            await progress.increment()
        except Exception:
            logger.debug("progress reporting failed at step %d", step)

    try:
        result = await run_full_pipeline(training_config, progress_cb=progress_cb, owner=owner)
    except (RuntimeError, ValueError) as exc:
        raise McpError(code=INTERNAL_ERROR, message=str(exc))

    await progress.set_message("Complete")
    result_dict = result.model_dump()

    if not _should_expose_paths():
        result_dict = sanitize_train_result(result_dict)

    return result_dict


# ── Tool 4: list_models ──────────────────────────────────────────────────────

def _registry_path() -> Path:
    """Resolve the model registry path using the same logic as the pipeline."""
    explicit = os.environ.get("MOLAGENT_REGISTRY_PATH")
    if explicit:
        return Path(explicit)
    return _output_root() / "model_registry.json"


@mcp.tool
async def list_models(ctx: Context) -> dict:
    """List all trained models available in the registry.

    Returns a dict with:
      - models: list of model entries (id, properties, task, metrics, etc.)

    In local/stdio mode, also returns registry_path. In remote mode, models are
    filtered to only show those owned by the authenticated user (admin sees all).

    Use the returned model IDs with the predict and delete_model tools.
    """
    caller = _get_caller(ctx)
    caller = _require_auth(caller)

    registry_path = _registry_path()
    if not registry_path.exists():
        resp: dict = {"models": []}
        if _should_expose_paths():
            resp["registry_path"] = str(registry_path)
        return resp

    try:
        data = json.loads(registry_path.read_text())
        if not isinstance(data, list):
            data = []
    except (json.JSONDecodeError, OSError):
        data = []

    _, is_admin = _caller_privileges(caller)
    owner_id = caller.get("owner_id")

    models = []
    for entry in data:
        entry_owner = entry.get("owner")
        if not is_admin:
            if entry_owner is None or entry_owner != owner_id:
                continue

        model_info = {
            "id": entry.get("id"),
            "target_properties": entry.get("target_properties", []),
            "task_type": entry.get("task_type", "unknown"),
            "metrics": entry.get("metrics", {}),
            "feature_keys": entry.get("feature_keys", []),
            "is_refitted": entry.get("is_refitted", False),
            "computational_load": entry.get("computational_load"),
            "model_format": entry.get("model_format"),
            "created_at": entry.get("created_at"),
            "blender_properties": entry.get("blender_properties", []),
            "owner": entry_owner,
        }

        clf_info = entry.get("classification")
        if clf_info:
            model_info["classification"] = clf_info

        if _should_expose_paths():
            model_info["source_dataset"] = entry.get("source_dataset")

        models.append(model_info)

    resp = {"models": models}
    if _should_expose_paths():
        resp["registry_path"] = str(registry_path)
    return resp


# ── Tool 5: predict ──────────────────────────────────────────────────────────

@mcp.tool(task=True)
async def predict(
    ctx: Context,
    model_id: Optional[str] = None,
    model_file: Optional[str] = None,
    smiles_list: Optional[list[str]] = None,
    smiles_file: Optional[str] = None,
    smiles_column: str = "smiles",
    properties: Optional[list[str]] = None,
    compute_sd: bool = True,
    blender_properties: Optional[list[str]] = None,
    blender_values: Optional[dict[str, float]] = None,
    convert_log10: Optional[bool] = None,
) -> dict:
    """Make predictions on new molecules using a trained AutoMol model.

    Provide EITHER model_id (from list_models) OR model_file (direct path).
    Provide EITHER smiles_list (inline SMILES) OR smiles_file (CSV path).

    In remote/HTTP mode, model_file (direct path) is disabled for non-admin users.
    Use model_id from list_models instead.

    If neither model_id nor model_file is given, returns available models
    from the registry instead of running predictions.

    Args:
        model_id: Model ID from the registry (use list_models to find IDs).
        model_file: Direct path to a .pt model file (local/admin only).
        smiles_list: List of SMILES strings to predict on.
        smiles_file: Path to a CSV file containing SMILES.
        smiles_column: Name of the SMILES column in the CSV (default: "smiles").
        properties: Subset of properties to predict (default: all in model).
        compute_sd: Whether to compute prediction uncertainty (default: true).
        blender_properties: Column names in smiles_file that contain blender values.
        blender_values: Blender property values for smiles_list input (e.g. {"Y_noisy": 1.5}).
    """
    caller = _get_caller(ctx)
    caller = _require_auth(caller)

    _, is_admin = _caller_privileges(caller)

    # Block direct model_file paths for non-admin remote callers (same gate as
    # csv/smiles inputs); admin/local callers may still pass a path.
    if model_file is not None:
        _ensure_path_access(caller, what="model_file")

    # Resolve smiles_file up front: a ds_ value is a registry id (ownership
    # checked), anything else is a direct path (admin/local only). Done before
    # model resolution so the rejection is deterministic regardless of model_id.
    if smiles_file is not None:
        smiles_file = _resolve_data_reference(caller, smiles_file, what="smiles_file")

    # ── Resolve model file from registry if model_id given ───────────────────
    resolved_model_file: str | None = None

    if model_id is not None:
        registry_path = _registry_path()
        if not registry_path.exists():
            raise McpError(
                code=INTERNAL_ERROR,
                message=f"Registry not found at {registry_path}. No models available.",
            )
        try:
            registry = json.loads(registry_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise McpError(code=INTERNAL_ERROR, message=f"Cannot read registry: {exc}")

        entry = next((e for e in registry if e.get("id") == model_id), None)
        if entry is None:
            available = [e.get("id") for e in registry]
            raise McpError(
                code=INTERNAL_ERROR,
                message=f"Model '{model_id}' not found. Available: {available}",
            )

        if not is_admin:
            entry_owner = entry.get("owner")
            if entry_owner is None or entry_owner != caller.get("owner_id"):
                raise McpError(
                    code=INTERNAL_ERROR,
                    message=f"Access denied: model '{model_id}' does not belong to you.",
                )

        mf = entry.get("model_file")
        if isinstance(mf, list):
            resolved_model_file = mf[0]
        elif isinstance(mf, str):
            resolved_model_file = mf
        else:
            raise McpError(code=INTERNAL_ERROR, message="Invalid model_file in registry entry.")

        touch_registry_entry(registry_path, model_id)

    elif model_file is not None:
        resolved_model_file = model_file

    else:
        # No model specified — return available models as guidance
        result = await list_models(ctx)
        return {
            "error": "No model specified. Provide model_id or model_file.",
            "available_models": result["models"],
        }

    # ── Validate model file exists ───────────────────────────────────────────
    if not Path(resolved_model_file).exists():
        raise McpError(
            code=INTERNAL_ERROR,
            message=f"Model file not found: {resolved_model_file}",
        )

    # ── Validate SMILES input ────────────────────────────────────────────────
    if smiles_list is None and smiles_file is None:
        raise McpError(
            code=INTERNAL_ERROR,
            message="Provide either smiles_list or smiles_file.",
        )

    if smiles_file is not None and not Path(smiles_file).exists():
        raise McpError(
            code=INTERNAL_ERROR,
            message=f"SMILES file not found: {smiles_file}",
        )

    # ── Build predict script args ────────────────────────────────────────────
    predict_script = _plugin_root() / "skills" / "predict" / "scripts" / "predict.py"
    if not predict_script.exists():
        raise McpError(
            code=INTERNAL_ERROR,
            message=f"Predict script not found: {predict_script}",
        )

    output_folder = _output_root() / "predictions"
    output_folder.mkdir(parents=True, exist_ok=True)

    args = [
        "--model-file", resolved_model_file,
        "--output-folder", str(output_folder),
        "--verbose",
    ]

    if smiles_file is not None:
        args += ["--smiles-file", smiles_file, "--smiles-column", smiles_column]
    else:
        for smi in smiles_list:
            args += ["--smiles-list", smi]

    if properties:
        for p in properties:
            args += ["--properties", p]

    if not compute_sd:
        args.append("--no-compute-sd")

    if blender_properties:
        for bp in blender_properties:
            args += ["--blender-properties", bp]

    if blender_values:
        for prop, val in blender_values.items():
            args += ["--blender-values", f"{prop}={val}"]

    if convert_log10 is not None:
        args.append("--convert-log10" if convert_log10 else "--no-convert-log10")

    # ── Run predict script ───────────────────────────────────────────────────
    try:
        stdout = await asyncio.to_thread(
            _run_script_sync, predict_script, args, _scripts_dir()
        )
    except RuntimeError as exc:
        raise McpError(code=INTERNAL_ERROR, message=str(exc))

    # ── Find the output CSV ──────────────────────────────────────────────────
    # Prefer parsing the explicit path from stdout (most reliable)
    _SAVED_PREFIX = "Predictions saved to: "
    output_csv: Path | None = None
    for line in stdout.splitlines():
        if line.startswith(_SAVED_PREFIX):
            candidate = Path(line[len(_SAVED_PREFIX):].strip())
            if candidate.exists():
                output_csv = candidate
                break

    # Fallback: glob for most recent file in output folder
    if output_csv is None:
        prediction_csvs = sorted(output_folder.glob("*_predictions_*.csv"), key=lambda p: p.stat().st_mtime)
        if not prediction_csvs:
            prediction_csvs = sorted(output_folder.glob("*predictions*.csv"), key=lambda p: p.stat().st_mtime)
        if prediction_csvs:
            output_csv = prediction_csvs[-1]

    if output_csv is None:
        return {
            "status": "completed",
            "output_folder": str(output_folder),
            "warning": "Predictions ran but no output CSV found.",
            "stdout": stdout[:500],
        }

    # ── Read summary from CSV ────────────────────────────────────────────────
    try:
        df = pd.read_csv(output_csv)
        n_rows = len(df)
        columns = list(df.columns)
        summary_stats = {}
        for col in columns:
            if col.startswith("predicted_") or col.startswith("SD_") or col.startswith("prob_"):
                if df[col].dtype in ("float64", "float32", "int64"):
                    summary_stats[col] = {
                        "mean": round(float(df[col].mean()), 4),
                        "std": round(float(df[col].std()), 4),
                        "min": round(float(df[col].min()), 4),
                        "max": round(float(df[col].max()), 4),
                    }
    except Exception:
        n_rows = None
        columns = []
        summary_stats = {}

    result = {
        "status": "completed",
        "output_csv": str(output_csv),
        "n_predictions": n_rows,
        "columns": columns,
        "summary_stats": summary_stats,
        "model_file": resolved_model_file,
    }

    if not _should_expose_paths():
        # Include CSV content inline so remote clients can download without disk access
        try:
            result["csv_content"] = output_csv.read_text(encoding="utf-8")
            result["csv_filename"] = output_csv.name
        except OSError:
            pass
        result = sanitize_predict_result(result)

    return result


# ── Tool 6: merge_models ────────────────────────────────────────────────────

@mcp.tool(task=True)
async def merge_models(
    model_ids: list[str],
    ctx: Context,
    output_name: Optional[str] = None,
    verify_encoder: bool = True,
) -> dict:
    """Merge multiple trained models from the registry into a single model file.

    This enables training properties separately (e.g. with different feature
    configs or transforms) and combining them into one multi-property model
    afterward.

    Args:
        model_ids: Two or more model IDs from the registry to merge.
        output_name: Custom name for the merged model (default: auto-generated).
        verify_encoder: Assert that encoders are compatible across models (default True).
    """
    caller = _get_caller(ctx)
    _require_auth(caller)

    if len(model_ids) < 2:
        raise McpError(code=INTERNAL_ERROR, message="At least 2 model IDs required for merging.")

    registry_path = _registry_path()
    if not registry_path.exists():
        raise McpError(code=INTERNAL_ERROR, message="Registry not found.")

    try:
        data = json.loads(registry_path.read_text())
        if not isinstance(data, list):
            data = []
    except (json.JSONDecodeError, OSError) as exc:
        raise McpError(code=INTERNAL_ERROR, message=f"Cannot read registry: {exc}")

    if caller is None:
        is_admin, owner_id = True, None
    else:
        _, is_admin = _caller_privileges(caller)
        owner_id = caller.get("owner_id")

    # Resolve entries and validate
    entries = []
    for mid in model_ids:
        entry = next((e for e in data if e.get("id") == mid), None)
        if entry is None:
            raise McpError(code=INTERNAL_ERROR, message=f"Model '{mid}' not found in registry.")
        if not is_admin:
            entry_owner = entry.get("owner")
            if entry_owner is None or entry_owner != owner_id:
                raise McpError(code=INTERNAL_ERROR, message=f"Access denied: model '{mid}' does not belong to you.")
        entries.append(entry)

    # Validate same task type
    task_types = set(e.get("task_type", "unknown") for e in entries)
    if len(task_types) > 1:
        raise McpError(
            code=INTERNAL_ERROR,
            message=f"Cannot merge models with different task types: {task_types}",
        )
    task_type = task_types.pop()

    # Check no overlapping properties
    all_props = []
    for entry in entries:
        props = entry.get("target_properties", [])
        for p in props:
            if p in all_props:
                raise McpError(
                    code=INTERNAL_ERROR,
                    message=f"Overlapping property '{p}' — cannot merge models that predict the same property.",
                )
            all_props.append(p)

    # Resolve model file paths
    model_files = []
    for entry in entries:
        mf = entry.get("model_file")
        if isinstance(mf, list):
            model_files.extend(mf)
        elif isinstance(mf, str):
            model_files.append(mf)
        else:
            raise McpError(
                code=INTERNAL_ERROR,
                message=f"Model '{entry.get('id')}' has no model_file in registry.",
            )

    # Verify files exist
    for mf in model_files:
        if not Path(mf).exists():
            raise McpError(
                code=INTERNAL_ERROR,
                message=f"Model file not found on disk: {mf}",
            )

    # Determine output path. output_name is sanitized with Path(...).name (same
    # treatment as upload_dataset's filename) so it cannot escape output_root
    # via '..' segments or an absolute path.
    output_root = _output_root()
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "reg" if "regression" in task_type else "clf"
    merged_name = Path(output_name).name if output_name else f"merged_{timestamp}"
    if not merged_name or merged_name in (".", ".."):
        raise McpError(code=INTERNAL_ERROR, message=f"Invalid output_name: {output_name}")
    merged_folder = output_root / f"merged-{merged_name}"
    merged_folder.mkdir(parents=True, exist_ok=True)
    output_file = merged_folder / f"merged_stacking{suffix}model.pt"

    # Run merge_models.py
    merge_script = _scripts_dir() / "merge_models.py"
    merge_args = ["--output-file", str(output_file), "--verbose"]
    for mf in model_files:
        merge_args += ["--model-files", mf]
    if not verify_encoder:
        merge_args.append("--no-verify-encoder")

    try:
        stdout = await asyncio.to_thread(_run_script_sync, merge_script, merge_args)
    except RuntimeError as exc:
        raise McpError(code=INTERNAL_ERROR, message=f"Merge failed: {exc}")

    if not output_file.exists():
        raise McpError(
            code=INTERNAL_ERROR,
            message=f"Merge script completed but output file not found: {output_file}",
        )

    # Combine metrics from source models
    combined_metrics = {}
    combined_features = set()
    combined_blenders = []
    for entry in entries:
        combined_metrics.update(entry.get("metrics", {}))
        combined_features.update(entry.get("feature_keys", []))
        combined_blenders.extend(entry.get("blender_properties", []))

    # Register the merged model
    merged_id = f"merged-{merged_name}-{timestamp}"
    new_entry = {
        "id": merged_id,
        "target_properties": all_props,
        "task_type": task_type,
        "metrics": combined_metrics,
        "feature_keys": sorted(combined_features),
        "is_refitted": all(e.get("is_refitted", False) for e in entries),
        "computational_load": None,
        "model_format": "merged",
        "model_file": str(output_file),
        "run_folder": str(merged_folder),
        "source_models": model_ids,
        "blender_properties": list(set(combined_blenders)),
        "created_at": datetime.now().isoformat(),
        "owner": owner_id or LOCAL_USER_ID,
    }

    # Atomic registry update
    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    fd = _acquire_lock(lock_path)
    if fd is None:
        raise McpError(code=INTERNAL_ERROR, message="Could not acquire registry lock.")
    try:
        import tempfile as _tempfile
        current = json.loads(registry_path.read_text()) if registry_path.exists() else []
        current.append(new_entry)
        fd_tmp, tmp = _tempfile.mkstemp(
            prefix=".registry-", suffix=".json.tmp", dir=str(registry_path.parent)
        )
        try:
            with os.fdopen(fd_tmp, "w") as f:
                json.dump(current, f, indent=2)
            os.replace(tmp, registry_path)
        except Exception:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise
    finally:
        _release_lock(fd, lock_path)

    return {
        "status": "merged",
        "model_id": merged_id,
        "model_path": str(output_file) if _should_expose_paths() else None,
        "properties": all_props,
        "task_type": task_type,
        "source_models": model_ids,
        "metrics": combined_metrics,
    }


# ── Tool 7: delete_model ────────────────────────────────────────────────────

@mcp.tool
async def delete_model(model_id: str, ctx: Context) -> dict:
    """Delete a model from the registry and remove its files from disk.

    Users can only delete their own models. Admin can delete any model.

    Args:
        model_id: The model ID to delete (from list_models).
    """
    caller = _get_caller(ctx)
    caller = _require_auth(caller)

    _, is_admin = _caller_privileges(caller)

    registry_path = _registry_path()
    if not registry_path.exists():
        raise McpError(code=INTERNAL_ERROR, message="Registry not found.")

    try:
        data = json.loads(registry_path.read_text())
        if not isinstance(data, list):
            data = []
    except (json.JSONDecodeError, OSError) as exc:
        raise McpError(code=INTERNAL_ERROR, message=f"Cannot read registry: {exc}")

    entry = next((e for e in data if e.get("id") == model_id), None)
    if entry is None:
        raise McpError(
            code=INTERNAL_ERROR,
            message=f"Model '{model_id}' not found in registry.",
        )

    if not is_admin:
        entry_owner = entry.get("owner")
        if entry_owner is None or entry_owner != caller.get("owner_id"):
            raise McpError(
                code=INTERNAL_ERROR,
                message=f"Access denied: model '{model_id}' does not belong to you.",
            )

    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    fd = _acquire_lock(lock_path)
    if fd is None:
        raise McpError(code=INTERNAL_ERROR, message="Could not acquire registry lock.")

    try:
        # Re-read under lock
        data = json.loads(registry_path.read_text())
        new_data = [e for e in data if e.get("id") != model_id]
        # Atomic write
        import tempfile
        fd_tmp, tmp = tempfile.mkstemp(
            prefix=".registry-", suffix=".json.tmp", dir=str(registry_path.parent)
        )
        try:
            with os.fdopen(fd_tmp, "w") as f:
                json.dump(new_data, f, indent=2)
            os.replace(tmp, registry_path)
        except Exception:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise
    finally:
        _release_lock(fd, lock_path)

    # Delete files from disk
    files_removed = _delete_model_entry_files(entry)

    return {"status": "deleted", "id": model_id, "files_removed": files_removed}


# ── Tool 7b: download_model ───────────────────────────────────────────────


@mcp.tool
async def download_model(model_id: str, ctx: Context) -> dict:
    """Download a trained model's binary data as base64.

    Returns the base64-encoded .pt model file for a given model_id from
    the registry. Use list_models to discover available model IDs.

    Args:
        model_id: The model ID to download (from list_models or train result).
    """
    import base64

    caller = _get_caller(ctx)
    caller = _require_auth(caller)
    _, is_admin = _caller_privileges(caller)

    registry_path = _registry_path()
    if not registry_path.exists():
        raise McpError(code=INTERNAL_ERROR, message="Registry not found.")

    try:
        data = json.loads(registry_path.read_text())
        if not isinstance(data, list):
            data = []
    except (json.JSONDecodeError, OSError) as exc:
        raise McpError(code=INTERNAL_ERROR, message=f"Cannot read registry: {exc}")

    entry = next((e for e in data if e.get("id") == model_id), None)
    if entry is None:
        raise McpError(
            code=INTERNAL_ERROR,
            message=f"Model '{model_id}' not found in registry.",
        )

    if not is_admin:
        entry_owner = entry.get("owner")
        if entry_owner is None or entry_owner != caller.get("owner_id"):
            raise McpError(
                code=INTERNAL_ERROR,
                message=f"Access denied: model '{model_id}' does not belong to you.",
            )

    mf = entry.get("model_file")
    if isinstance(mf, list):
        model_path = Path(mf[0])
    elif isinstance(mf, str):
        model_path = Path(mf)
    else:
        raise McpError(code=INTERNAL_ERROR, message="Invalid model_file in registry.")

    try:
        model_path = _under_output_root(model_path)
    except ValueError:
        raise McpError(code=INTERNAL_ERROR, message=f"Model path outside output root: {model_path}")

    try:
        model_bytes = model_path.read_bytes()
    except OSError:
        raise McpError(code=INTERNAL_ERROR, message=f"Model file not found: {model_path}")
    model_b64 = base64.b64encode(model_bytes).decode()

    touch_registry_entry(registry_path, model_id)

    return {
        "model_id": model_id,
        "model_b64": model_b64,
        "model_filename": model_path.name,
        "size_bytes": len(model_bytes),
    }


# ── Tool 8: upload_dataset ─────────────────────────────────────────────────

@mcp.tool
async def upload_dataset(
    filename: str,
    file_content_b64: str,
    ctx: Context,
) -> dict:
    """Upload a CSV dataset to the data registry.

    Accepts base64-encoded file content. Creates a per-user upload directory
    and registers the dataset for later use with start_training_session.

    Args:
        filename: Original filename (e.g. "my_data.csv").
        file_content_b64: Base64-encoded CSV file content.
    """
    import base64
    import csv
    import io

    caller = _get_caller(ctx)
    caller = _require_auth(caller)

    _, is_admin = _caller_privileges(caller)
    owner_id = caller.get("owner_id", LOCAL_USER_ID)

    max_bytes = _max_upload_bytes()

    def _reject_too_large() -> None:
        raise McpError(
            code=INTERNAL_ERROR,
            message=f"Upload exceeds maximum size of {max_bytes // (1024 * 1024)} MB.",
        )

    # Reject oversized payloads before decoding (base64 inflates ~4/3, so the
    # encoded string is checked against the decoded-byte budget directly).
    if len(file_content_b64) > max_bytes * 4 // 3 + 4:
        _reject_too_large()

    # Decode base64
    try:
        raw_bytes = base64.b64decode(file_content_b64)
    except Exception as exc:
        raise McpError(code=INTERNAL_ERROR, message=f"Invalid base64 content: {exc}")

    if len(raw_bytes) > max_bytes:
        _reject_too_large()

    # Validate it's a parseable CSV
    try:
        text = raw_bytes.decode("utf-8", errors="replace")
        reader = csv.reader(io.StringIO(text))
        header = next(reader, None)
        if not header:
            raise McpError(code=INTERNAL_ERROR, message="CSV has no header row.")
        row_count = sum(1 for _ in reader)
    except McpError:
        raise
    except Exception as exc:
        raise McpError(code=INTERNAL_ERROR, message=f"Could not parse CSV: {exc}")

    import hashlib
    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    # Sanitize filename to prevent path traversal
    safe_filename = Path(filename).name
    if not safe_filename or safe_filename in (".", ".."):
        raise McpError(code=INTERNAL_ERROR, message=f"Invalid filename: {filename}")

    # Scenario 2: identical content already registered for this owner → return as-is
    owner_entries = list_datasets_for_owner(owner_id)
    for existing in owner_entries:
        if existing.get("sha256") == sha256:
            touch_registry_entry(data_registry_path(), existing["id"])
            return {
                "dataset_id": existing["id"],
                "filename": existing["filename"],
                "columns": existing["columns"],
                "row_count": existing["row_count"],
                "size_bytes": existing["size_bytes"],
            }

    # Scenario 1: same filename for this owner → overwrite file, update entry in-place
    name_match = next((e for e in owner_entries if e.get("filename") == safe_filename), None)
    if name_match:
        dest = (_output_root() / name_match["file_path"]).resolve()
        dest.write_bytes(raw_bytes)
        entry = update_dataset(
            name_match["id"],
            owner_id=owner_id,
            sha256=sha256,
            size_bytes=len(raw_bytes),
            columns=header,
            row_count=row_count,
        ) or name_match
        return {
            "dataset_id": entry["id"],
            "filename": entry["filename"],
            "columns": entry["columns"],
            "row_count": entry["row_count"],
            "size_bytes": entry["size_bytes"],
        }

    # New dataset — write file and register
    uploads_dir = _output_root() / "uploads" / owner_id
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / safe_filename
    dest.write_bytes(raw_bytes)
    rel_path = f"uploads/{owner_id}/{dest.name}"

    entry = register_dataset(
        owner_id=owner_id,
        filename=safe_filename,
        file_path=rel_path,
        size_bytes=len(raw_bytes),
        columns=header,
        row_count=row_count,
        sha256=sha256,
    )

    return {
        "dataset_id": entry["id"],
        "filename": entry["filename"],
        "columns": entry["columns"],
        "row_count": entry["row_count"],
        "size_bytes": entry["size_bytes"],
    }


# ── Tool 9: list_datasets ─────────────────────────────────────────────────

@mcp.tool
async def list_datasets(ctx: Context) -> dict:
    """List all uploaded datasets in the data registry.

    Returns datasets owned by the caller (admin sees all).
    Use the returned dataset_id with start_training_session.
    """
    caller = _get_caller(ctx)
    caller = _require_auth(caller)

    is_local, is_admin = _caller_privileges(caller)
    owner_id = caller.get("owner_id") if not is_local else None

    entries = list_datasets_for_owner(owner_id, is_admin=is_admin)

    datasets = []
    for entry in entries:
        info: dict = {
            "id": entry.get("id"),
            "filename": entry.get("filename"),
            "size_bytes": entry.get("size_bytes"),
            "columns": entry.get("columns", []),
            "row_count": entry.get("row_count"),
            "uploaded_at": entry.get("uploaded_at"),
            "last_used": entry.get("last_used"),
            "owner": entry.get("owner"),
            "type": entry.get("type", "csv"),
        }
        if _should_expose_paths():
            info["file_path"] = str(_output_root() / entry["file_path"])
        datasets.append(info)

    return {"datasets": datasets}


# ── Tool 10: delete_dataset ────────────────────────────────────────────────

@mcp.tool
async def delete_dataset(dataset_id: str, ctx: Context) -> dict:
    """Delete a dataset from the registry and remove its file from disk.

    Users can only delete their own datasets. Admin can delete any dataset.

    Args:
        dataset_id: The dataset ID to delete (from list_datasets).
    """
    caller = _get_caller(ctx)
    caller = _require_auth(caller)

    _, is_admin = _caller_privileges(caller)
    owner_id = None if is_admin else caller.get("owner_id")

    removed = remove_dataset(dataset_id, owner_id=owner_id)
    if removed is None:
        raise McpError(
            code=INTERNAL_ERROR,
            message=f"Dataset '{dataset_id}' not found or access denied.",
        )

    # Delete file(s) from disk. A 3d_structure entry's file_path is a directory
    # (SDF + pdbs/ + any generated prepared/ CSV) — unlink() raises OSError on a
    # directory, so remove the whole tree instead.
    file_path = _output_root() / removed["file_path"]
    if removed.get("type") == "3d_structure":
        shutil.rmtree(file_path, ignore_errors=True)
    elif file_path.exists():
        try:
            file_path.unlink()
        except OSError as exc:
            logger.warning("Failed to delete dataset file %s: %s", file_path, exc)

    return {"status": "deleted", "dataset_id": dataset_id, "filename": removed.get("filename")}


# ── Tool 11: admin_manage ──────────────────────────────────────────────────

@mcp.tool
async def admin_manage(
    action: Literal["create_token", "revoke_user", "rotate_token", "list_users", "purge_stale", "purge_orphans"],
    ctx: Context,
    user_id: Optional[str] = None,
    owner_id: Optional[str] = None,
    max_age_days: Optional[int] = None,
    force: bool = False,
) -> dict:
    """Admin-only management: token operations and stale artifact cleanup.

    Actions:
      - create_token: Generate a new user token (requires user_id).
      - revoke_user: Revoke a user by their unique owner_id (from list_users).
      - rotate_token: Issue a fresh token for an existing user by owner_id, keeping
        their owner_id so all their models/datasets stay accessible. Use when a user
        loses their token. The old token stops working.
      - list_users: List all registered users (owner_id, name, status, token prefix).
      - purge_stale: Remove models/datasets not used in max_age_days days.
        Dry-run by default (returns what would be deleted). Pass force=True to delete.
      - purge_orphans: Remove run folders not referenced by any registry entry
        (e.g. from failed training runs). Filters by max_age_days (folder mtime).
        Dry-run by default; pass force=True to delete.

    Args:
        action: The admin action to perform.
        user_id: Required for create_token — the username to assign.
        owner_id: Required for revoke_user / rotate_token — the unique handle from list_users.
        max_age_days: Required for purge_stale and purge_orphans — entries/folders older than this are purged.
        force: For purge_stale/purge_orphans — if False (default), dry-run only.
    """
    caller = _get_caller(ctx)
    caller = _require_auth(caller)

    _, is_admin = _caller_privileges(caller)

    if not is_admin:
        raise McpError(
            code=INTERNAL_ERROR,
            message="Access denied: admin privileges required.",
        )

    if action == "create_token":
        if not user_id:
            raise McpError(
                code=INTERNAL_ERROR,
                message="user_id is required for create_token action.",
            )
        try:
            token = create_user_token(user_id)
        except ValueError as exc:
            raise McpError(code=INTERNAL_ERROR, message=str(exc))
        return {"status": "created", "user_id": user_id, "token": token}

    elif action == "revoke_user":
        if not owner_id:
            raise McpError(
                code=INTERNAL_ERROR,
                message="owner_id is required for revoke_user action (see list_users).",
            )
        success = revoke_user(owner_id)
        if not success:
            raise McpError(
                code=INTERNAL_ERROR,
                message=f"No user with owner_id '{owner_id}' found in store.",
            )
        return {"status": "revoked", "owner_id": owner_id}

    elif action == "rotate_token":
        if not owner_id:
            raise McpError(
                code=INTERNAL_ERROR,
                message="owner_id is required for rotate_token action (see list_users).",
            )
        rotated = rotate_user_token(owner_id)
        if rotated is None:
            raise McpError(
                code=INTERNAL_ERROR,
                message=f"No user with owner_id '{owner_id}' found in store.",
            )
        new_token, user_name = rotated
        return {"status": "rotated", "owner_id": owner_id, "user_id": user_name, "token": new_token}

    elif action == "list_users":
        users = list_users()
        return {"status": "ok", "users": users}

    elif action == "purge_stale":
        if max_age_days is None or max_age_days < 1:
            raise McpError(
                code=INTERNAL_ERROR,
                message="max_age_days must be >= 1 for purge_stale action.",
            )

        from datetime import datetime, timedelta

        cutoff = datetime.now() - timedelta(days=max_age_days)
        cutoff_iso = cutoff.isoformat(timespec="seconds")

        def _is_stale(entry: dict) -> bool:
            last_used = entry.get("last_used") or entry.get("created_at") or entry.get("uploaded_at")
            if not last_used:
                return True
            return last_used < cutoff_iso

        # Collect stale models
        model_reg_path = _registry_path()
        model_entries = load_json_list(model_reg_path) if model_reg_path.exists() else []
        stale_models = [e for e in model_entries if _is_stale(e)]

        # Collect stale datasets
        data_reg_path = data_registry_path()
        data_entries = load_json_list(data_reg_path) if data_reg_path.exists() else []
        stale_datasets = [e for e in data_entries if _is_stale(e)]

        if not force:
            return {
                "dry_run": True,
                "models_to_purge": [{"id": e.get("id"), "last_used": e.get("last_used"), "created_at": e.get("created_at")} for e in stale_models],
                "datasets_to_purge": [{"id": e.get("id"), "filename": e.get("filename"), "last_used": e.get("last_used")} for e in stale_datasets],
                "total_models": len(stale_models),
                "total_datasets": len(stale_datasets),
            }

        # Force mode — actually delete
        purged_models = []
        purged_datasets = []
        errors = []

        # Purge models
        stale_model_ids = {e.get("id") for e in stale_models}
        if stale_model_ids and model_reg_path.exists():
            lock_path = model_reg_path.with_suffix(model_reg_path.suffix + ".lock")
            fd = _acquire_lock(lock_path)
            if fd is None:
                errors.append("Could not acquire model registry lock")
            else:
                try:
                    current = load_json_list(model_reg_path)
                    kept = []
                    for e in current:
                        if e.get("id") in stale_model_ids:
                            _delete_model_entry_files(e, errors)
                            purged_models.append(e.get("id"))
                        else:
                            kept.append(e)
                    atomic_write_json(model_reg_path, kept)
                finally:
                    _release_lock(fd, lock_path)

        # Purge datasets
        stale_dataset_ids = {e.get("id") for e in stale_datasets}
        if stale_dataset_ids and data_reg_path.exists():
            lock = _dr_lock_path(data_reg_path)
            fd = _acquire_lock(lock)
            if fd is None:
                errors.append("Could not acquire data registry lock")
            else:
                try:
                    current = load_json_list(data_reg_path)
                    kept = []
                    for e in current:
                        if e.get("id") in stale_dataset_ids:
                            file_path = _output_root() / e.get("file_path", "")
                            if file_path.exists():
                                try:
                                    file_path.unlink()
                                except OSError as exc:
                                    errors.append(f"Failed to delete {file_path}: {exc}")
                            purged_datasets.append(e.get("id"))
                        else:
                            kept.append(e)
                    atomic_write_json(data_reg_path, kept)
                finally:
                    _release_lock(fd, lock)

        return {
            "dry_run": False,
            "purged_models": purged_models,
            "purged_datasets": purged_datasets,
            "errors": errors,
        }

    elif action == "purge_orphans":
        if max_age_days is None or max_age_days < 1:
            raise McpError(
                code=INTERNAL_ERROR,
                message="max_age_days must be >= 1 for purge_orphans action.",
            )

        cutoff_ts = time.time() - max_age_days * 86400
        output_root = _output_root()

        # Collect all run_folder paths referenced by the model registry
        model_reg_path = _registry_path()
        model_entries = load_json_list(model_reg_path) if model_reg_path.exists() else []
        registered_folders = {
            Path(e["run_folder"]).resolve()
            for e in model_entries
            if e.get("run_folder")
        }

        # Scan output root for orphaned run directories
        skip_names = {"uploads"}
        orphans = []
        if output_root.exists():
            for child in output_root.iterdir():
                if not child.is_dir():
                    continue
                if child.name in skip_names:
                    continue
                if child.resolve() in registered_folders:
                    continue
                if child.stat().st_mtime > cutoff_ts:
                    continue
                orphans.append(child)

        if not force:
            return {
                "dry_run": True,
                "orphaned_folders": [str(f) for f in orphans],
                "total": len(orphans),
            }

        purged = []
        errors = []
        for folder in orphans:
            if str(folder.resolve()) in _active_run_folders:
                continue   # pipeline is actively writing to this folder
            try:
                shutil.rmtree(_under_output_root(folder))
                purged.append(str(folder))
            except ValueError:
                errors.append(f"Skipped orphan outside output root: {folder}")
            except OSError as exc:
                errors.append(f"Failed to delete {folder}: {exc}")

        return {
            "dry_run": False,
            "purged_folders": purged,
            "errors": errors,
        }

    else:
        raise McpError(
            code=INTERNAL_ERROR,
            message=f"Unknown action: {action}",
        )


# ── 3D Structure Tools ─────────────────────────────────────────────────────────
from _3d_tools import register_3d_tools  # noqa: E402
register_3d_tools(mcp)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AutoMol MCP server")
    parser.add_argument(
        "--transport", choices=["stdio", "streamable-http"], default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="HTTP port (default: 8001)")
    args = parser.parse_args()

    bootstrap_admin_token()
    if args.transport == "streamable-http":
        if not _AUTH_REQUIRED:
            print(
                "ERROR: --transport streamable-http requires MOLAGENT_AUTH_REQUIRED=true. "
                "Starting without auth exposes all tools to unauthenticated callers.",
                file=sys.stderr,
            )
            sys.exit(1)
        max_bytes = int(float(os.environ.get("MOLAGENT_MAX_UPLOAD_MB", "100")) * 1024 * 1024 * 2)
        mcp.run(
            transport=args.transport,
            host=args.host,
            port=args.port,
            uvicorn_config={"h11_max_incomplete_event_size": max_bytes},
        )
    else:
        mcp.run(transport=args.transport)
