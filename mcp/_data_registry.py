"""Data registry for uploaded datasets.

Mirrors the model registry pattern: a JSON array file at
${MOLAGENT_OUTPUT_ROOT}/data_registry.json with per-entry ownership,
atomic lock-protected I/O, and a shared `touch_registry_entry` helper
usable by both registries.
"""
from __future__ import annotations

import json
import os
import secrets
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from _auth import _acquire_lock, _release_lock

_DATA_REGISTRY_FILENAME = "data_registry.json"


# ---------------------------------------------------------------------------
# Lock path helper (re-exported for external use)
# ---------------------------------------------------------------------------


def _lock_path(registry_path: Path) -> Path:
    return registry_path.with_suffix(registry_path.suffix + ".lock")


def load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def atomic_write_json(path: Path, data: list | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".registry-", suffix=".json.tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


# ---------------------------------------------------------------------------
# Generic touch helper (works on any registry with "id" + "last_used" fields)
# ---------------------------------------------------------------------------


def touch_registry_entry(registry_path: Path, entry_id: str) -> bool:
    """Atomically update `last_used` on a registry entry.

    Returns True if entry was found and updated, False otherwise.
    """
    lock = _lock_path(registry_path)
    fd = _acquire_lock(lock)
    if fd is None:
        return False
    try:
        data = load_json_list(registry_path)
        now = datetime.now().isoformat(timespec="seconds")
        found = False
        for entry in data:
            if entry.get("id") == entry_id:
                entry["last_used"] = now
                found = True
                break
        if found:
            atomic_write_json(registry_path, data)
        return found
    finally:
        _release_lock(fd, lock)


# ---------------------------------------------------------------------------
# Data registry path resolution
# ---------------------------------------------------------------------------


def _data_registry_output_root() -> Path:
    root = (
        os.environ.get("PHARMAOS_MOLAGENT_ROOT")
        or os.environ.get("MOLAGENT_OUTPUT_ROOT")
    )
    if root:
        return Path(root).resolve()
    plugin_root = os.environ.get("MOLAGENT_PLUGIN_ROOT")
    base = Path(plugin_root) if plugin_root else Path(__file__).resolve().parent.parent
    return (base / "MolagentFiles").resolve()


def data_registry_path() -> Path:
    return _data_registry_output_root() / _DATA_REGISTRY_FILENAME


# ---------------------------------------------------------------------------
# Data registry CRUD
# ---------------------------------------------------------------------------


def register_dataset(
    owner_id: str,
    filename: str,
    file_path: str,
    size_bytes: int,
    columns: list[str],
    row_count: int,
    sha256: Optional[str] = None,
) -> dict:
    """Register a new dataset entry. Returns the created entry."""
    registry_path = data_registry_path()
    lock = _lock_path(registry_path)
    fd = _acquire_lock(lock)
    if fd is None:
        raise RuntimeError("Could not acquire data registry lock")
    try:
        data = load_json_list(registry_path)
        now = datetime.now().isoformat(timespec="seconds")
        entry: dict = {
            "id": f"ds_{secrets.token_urlsafe(12)}",
            "filename": filename,
            "owner": owner_id,
            "file_path": file_path,
            "size_bytes": size_bytes,
            "columns": columns,
            "row_count": row_count,
            "uploaded_at": now,
            "last_used": now,
        }
        if sha256 is not None:
            entry["sha256"] = sha256
        data.append(entry)
        atomic_write_json(registry_path, data)
        return entry
    finally:
        _release_lock(fd, lock)


def update_dataset(
    entry_id: str,
    owner_id: Optional[str] = None,
    **fields,
) -> Optional[dict]:
    """Update fields of an existing dataset entry in-place. Returns updated entry or None."""
    registry_path = data_registry_path()
    lock = _lock_path(registry_path)
    fd = _acquire_lock(lock)
    if fd is None:
        raise RuntimeError("Could not acquire data registry lock")
    try:
        data = load_json_list(registry_path)
        updated = None
        for entry in data:
            if entry.get("id") != entry_id:
                continue
            if owner_id is not None and entry.get("owner") != owner_id:
                break
            entry.update(fields)
            entry["last_used"] = datetime.now().isoformat(timespec="seconds")
            updated = entry
            break
        if updated:
            atomic_write_json(registry_path, data)
        return updated
    finally:
        _release_lock(fd, lock)


def remove_dataset(entry_id: str, owner_id: Optional[str] = None) -> Optional[dict]:
    """Remove a dataset entry. Returns the removed entry or None if not found.

    If owner_id is provided, only removes if it matches (non-admin check).
    """
    registry_path = data_registry_path()
    lock = _lock_path(registry_path)
    fd = _acquire_lock(lock)
    if fd is None:
        raise RuntimeError("Could not acquire data registry lock")
    try:
        data = load_json_list(registry_path)
        removed = None
        new_data = []
        for entry in data:
            if entry.get("id") == entry_id:
                if owner_id is not None and entry.get("owner") != owner_id:
                    new_data.append(entry)
                else:
                    removed = entry
            else:
                new_data.append(entry)
        if removed:
            atomic_write_json(registry_path, new_data)
        return removed
    finally:
        _release_lock(fd, lock)


def get_dataset(entry_id: str) -> Optional[dict]:
    """Get a single dataset entry by ID."""
    data = load_json_list(data_registry_path())
    return next((e for e in data if e.get("id") == entry_id), None)


def list_datasets_for_owner(owner_id: Optional[str], is_admin: bool = False) -> list[dict]:
    """List datasets filtered by owner. Admin sees all."""
    data = load_json_list(data_registry_path())
    if is_admin or owner_id is None:
        return data
    return [e for e in data if e.get("owner") == owner_id]
