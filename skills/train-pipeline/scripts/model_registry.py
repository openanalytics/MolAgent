#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["click"]
# ///
"""Model registry for trained AutoMol models.

Tracks model artifacts, metrics, and configuration in a central JSON file.

Path resolution (in priority):
  1. MOLAGENT_REGISTRY_PATH env var (full file path)
  2. --directory CLI flag / config.output_folder from pipeline state
  3. PHARMAOS_MOLAGENT_ROOT or MOLAGENT_OUTPUT_ROOT env var
  4. ./MolagentFiles
"""

from __future__ import annotations

import errno
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import click

# Local helpers — when invoked via `uv run script.py` the script directory is
# on sys.path so the bare imports work.
from _paths import (
    REGISTRY_FILENAME,
    default_output_folder,
    get_output_root,
    get_registry_path,
)


# ---------------------------------------------------------------------------
# Atomic, lock-protected I/O
# ---------------------------------------------------------------------------


def _lock_path(registry_path: Path) -> Path:
    return registry_path.with_suffix(registry_path.suffix + ".lock")


STALE_LOCK_AGE_SECONDS = 60.0  # if a lock is older than this, treat it as stale
                                # (process holding it was SIGKILL'd or crashed)


def _acquire_lock(lock_path: Path, timeout: float = 10.0) -> int | None:
    """Best-effort exclusive file lock for cross-platform concurrent safety.

    Uses O_CREAT|O_EXCL — works on Windows and POSIX. Returns the open fd
    on success; ``None`` if the lock could not be acquired within ``timeout``.

    Self-heals stale locks: if the lock file exists and is older than
    ``STALE_LOCK_AGE_SECONDS``, it is unlinked once before retry. This guards
    against locks left behind by SIGKILL'd processes.
    """
    deadline = time.monotonic() + timeout
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stale_unlink_attempted = False
    while True:
        try:
            return os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.EACCES):
                raise
            # Stale-lock detection: only unlink once per acquisition attempt
            if not stale_unlink_attempted:
                try:
                    age = time.time() - lock_path.stat().st_mtime
                    if age > STALE_LOCK_AGE_SECONDS:
                        try:
                            lock_path.unlink()
                            print(
                                f"[registry] removed stale lock {lock_path} (age {age:.0f}s)",
                                file=sys.stderr,
                            )
                        except FileNotFoundError:
                            pass
                except FileNotFoundError:
                    pass
                stale_unlink_attempted = True
                continue  # retry immediately
            if time.monotonic() > deadline:
                return None
            time.sleep(0.05)


def _release_lock(fd: int | None, lock_path: Path) -> None:
    if fd is not None:
        try:
            os.close(fd)
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def load_registry(registry_path: Path) -> list[dict]:
    if not registry_path.exists():
        return []
    try:
        with open(registry_path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, ValueError, OSError):
        return []


def save_registry(registry_path: Path, entries: list[dict]) -> None:
    """Atomic write: tempfile in same directory, then ``os.replace``."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".registry-", suffix=".json.tmp", dir=str(registry_path.parent)
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(entries, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, registry_path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _resolve_registry_path(directory: str | None) -> Path:
    """Pick the registry file path given a CLI ``--directory`` option."""
    if directory:
        return get_registry_path(directory)
    return get_registry_path()


# ---------------------------------------------------------------------------
# Model ID derivation
# ---------------------------------------------------------------------------


def derive_model_id(
    state: dict,
    dataset_stem: str,
    prop_str: str,
    existing_ids: set[str],
) -> str:
    """Prefer the run_id from pipeline state — that matches the run folder name.

    Falls back to ``{dataset}-{props}-{now}`` only when run_id is absent (older
    state files).
    """
    run_id = state.get("run_id")
    if isinstance(run_id, str) and run_id:
        base_id = run_id
    else:
        ts_str = datetime.now().strftime("%Y%m%d_%H%M")
        base_id = f"{dataset_stem}-{prop_str}-{ts_str}"

    if base_id not in existing_ids:
        return base_id
    n = 2
    while f"{base_id}-{n}" in existing_ids:
        n += 1
    return f"{base_id}-{n}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group()
def cli():
    """AutoMol model registry."""
    pass


@cli.command()
@click.option("--pipeline-state", required=True, help="Path to pipeline_state.json")
@click.option(
    "--directory",
    default=None,
    help=(
        "Directory containing the registry. Defaults to "
        "$MOLAGENT_OUTPUT_ROOT / $PHARMAOS_MOLAGENT_ROOT / ./MolagentFiles"
    ),
)
@click.option(
    "--owner",
    default=None,
    help="User ID that owns this model (for access control).",
)
def register(pipeline_state, directory, owner):
    """Register a trained model from pipeline state."""
    state_path = Path(pipeline_state)
    if not state_path.exists():
        print(f"Error: {pipeline_state} not found", file=sys.stderr)
        sys.exit(1)

    with open(state_path) as f:
        state = json.load(f)

    config = state.get("config", {})
    files = state.get("outputs", state.get("files", {}))
    metrics = state.get("metrics", {})
    detection = state.get("detection", {})

    registry_path = _resolve_registry_path(directory)

    # Detect merged model file first (preferred)
    merged_file = files.get("merged_refitted_model_file") or files.get(
        "merged_model_file"
    )
    if merged_file:
        model_file: list | str = merged_file
        model_format = "merged"
    else:
        model_format = "individual"
        refitted_files = files.get("refitted_model_files", {})
        model_files_dict = files.get("model_files", {})
        if refitted_files and isinstance(refitted_files, dict):
            model_file = list(refitted_files.values())
        elif model_files_dict and isinstance(model_files_dict, dict):
            model_file = list(model_files_dict.values())
        else:
            model_file = files.get("refitted_model") or files.get("model_file")
            if model_file:
                model_file = [model_file] if isinstance(model_file, str) else model_file

    if not model_file:
        print("Error: no model file found in pipeline state", file=sys.stderr)
        sys.exit(1)

    # Lock-protected read-modify-write
    lock = _lock_path(registry_path)
    fd = _acquire_lock(lock)
    if fd is None:
        print(
            f"Error: could not acquire registry lock at {lock} within timeout",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        registry = load_registry(registry_path)

        # Idempotency: skip if model_file already registered
        model_file_key = model_file if isinstance(model_file, list) else [model_file]
        for entry in registry:
            existing_mf = entry.get("model_file", [])
            existing_mf = (
                existing_mf if isinstance(existing_mf, list) else [existing_mf]
            )
            if set(existing_mf) == set(model_file_key):
                print(
                    json.dumps(
                        {
                            "status": "skipped",
                            "reason": "model already registered",
                            "id": entry["id"],
                        }
                    )
                )
                return

        is_refitted = (
            bool(
                files.get("refitted_model_files")
                or files.get("merged_refitted_model_file")
            )
            or files.get("refitted_model") is not None
        )

        data_file = config.get("data_file", "unknown")
        dataset_stem = Path(data_file).stem
        properties = config.get("target_properties", [])
        existing_ids = {e["id"] for e in registry}

        prop_str = "_".join(properties) if properties else "model"
        model_id = derive_model_id(state, dataset_stem, prop_str, existing_ids)

        train_info = files.get("train_info")
        if isinstance(train_info, dict):
            train_info = list(train_info.values())

        # Classification metadata: extract class_values and labelnames
        classification_info = {}
        task_type = config.get("task_type", "unknown")
        if "classification" in task_type:
            classification_info["categorical"] = config.get("categorical", False)
            classification_info["nb_classes"] = config.get("nb_classes")
            classification_info["class_values"] = config.get("class_values")
            # Read labelnames from prep info if available
            output_folder = config.get("output_folder", "")
            if output_folder:
                prep_infos = list(Path(output_folder).glob("automol_prepared_*_info.json"))
                if prep_infos:
                    try:
                        with open(prep_infos[0]) as pf:
                            prep_data = json.load(pf)
                        classification_info["labelnames"] = prep_data.get("labelnames")
                        if not classification_info["class_values"]:
                            classification_info["class_values"] = prep_data.get("class_values")
                    except (json.JSONDecodeError, OSError):
                        pass

        now = datetime.now().isoformat(timespec="seconds")
        entry = {
            "id": model_id,
            "created_at": now,
            "last_used": now,
            "model_file": model_file,
            "model_format": model_format,
            "target_properties": properties,
            "task_type": task_type,
            "metrics": metrics,
            "feature_keys": config.get("feature_keys", []),
            "smiles_column": config.get("smiles_column", "Stand_SMILES"),
            "blender_properties": config.get("blender_properties", []),
            "source_dataset": data_file,
            "is_refitted": is_refitted,
            "model_card": files.get("model_card"),
            "train_info": train_info,
            "n_samples": detection.get("n_samples") or config.get("n_samples"),
            "computational_load": config.get("computational_load", "moderate"),
            "split_strategy": config.get("split_strategy", "mixed"),
            "run_folder": config.get("output_folder"),
            "owner": owner,
        }
        if classification_info:
            entry["classification"] = classification_info

        registry.append(entry)
        save_registry(registry_path, registry)

        print(
            json.dumps(
                {
                    "status": "registered",
                    "id": model_id,
                    "registry": str(registry_path),
                }
            )
        )
    finally:
        _release_lock(fd, lock)


@cli.command("list")
@click.option(
    "--format", "fmt", type=click.Choice(["json", "table"]), default="table"
)
@click.option(
    "--directory",
    default=None,
    help="Directory containing registry (default: env-resolved output root)",
)
def list_models(fmt, directory):
    """List all registered models."""
    registry_path = _resolve_registry_path(directory)
    registry = load_registry(registry_path)
    if not registry:
        print("[]" if fmt == "json" else "No models registered.")
        return

    if fmt == "json":
        print(json.dumps(registry, indent=2))
        return

    print(
        f"{'ID':<40} {'Task':<15} {'Properties':<20} {'Metrics':<30} {'Date':<12}"
    )
    print("-" * 117)
    for e in registry:
        props = ", ".join(e.get("target_properties", []))
        metrics_parts = []
        for k, v in e.get("metrics", {}).items():
            if isinstance(v, float):
                metrics_parts.append(f"{k}={v:.3f}")
            else:
                metrics_parts.append(f"{k}={v}")
        metrics_str = ", ".join(metrics_parts) if metrics_parts else "N/A"
        date = e.get("created_at", "")[:10]
        print(
            f"{e['id']:<40} {e.get('task_type', '?'):<15} {props:<20} {metrics_str:<30} {date:<12}"
        )


@cli.command()
@click.argument("model_id")
@click.option(
    "--directory",
    default=None,
    help="Directory containing registry (default: env-resolved output root)",
)
def get(model_id, directory):
    """Get full details of a registered model."""
    registry_path = _resolve_registry_path(directory)
    registry = load_registry(registry_path)
    for entry in registry:
        if entry["id"] == model_id:
            print(json.dumps(entry, indent=2))
            return
    print(f"Error: model '{model_id}' not found", file=sys.stderr)
    sys.exit(1)


@cli.command()
@click.argument("model_id")
@click.option(
    "--directory",
    default=None,
    help="Directory containing registry (default: env-resolved output root)",
)
def remove(model_id, directory):
    """Remove a model entry from the registry (files are not deleted)."""
    registry_path = _resolve_registry_path(directory)
    lock = _lock_path(registry_path)
    fd = _acquire_lock(lock)
    if fd is None:
        print(
            f"Error: could not acquire registry lock at {lock} within timeout",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        registry = load_registry(registry_path)
        new_registry = [e for e in registry if e["id"] != model_id]
        if len(new_registry) == len(registry):
            print(f"Error: model '{model_id}' not found", file=sys.stderr)
            sys.exit(1)
        save_registry(registry_path, new_registry)
        print(json.dumps({"status": "removed", "id": model_id}))
    finally:
        _release_lock(fd, lock)


if __name__ == "__main__":
    cli()
