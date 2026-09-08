"""Token-based authentication for the AutoMol MCP server.

Token store lives at ${MOLAGENT_OUTPUT_ROOT}/auth_tokens.json (or overridden
via MOLAGENT_TOKEN_STORE_PATH). Uses the same atomic lock pattern as model_registry.py.

Tokens are stored as SHA-256 hashes — the plaintext is shown only once, at
creation. Lookups hash the incoming token; the admin token is compared with
secrets.compare_digest to avoid timing leaks.

Structure:
  {
    "admin_token_hash": "<sha256 hex>",
    "users": {
      "<sha256 hex>": {
        "user_id": "alice", "owner_id": "...", "created_at": "...",
        "revoked": false, "token_prefix": "molagent_usr_abc..."
      }
    }
  }
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import secrets
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

STALE_LOCK_AGE_SECONDS = 60.0
_TOKEN_STORE_FILENAME = "auth_tokens.json"

LOCAL_USER_ID = "__local__"
ANONYMOUS_USER_ID = "__anonymous__"
ADMIN_USER_ID = "__admin__"

# Sentinel identities that must never be assigned to a real token.
RESERVED_USER_IDS = frozenset({LOCAL_USER_ID, ANONYMOUS_USER_ID})


def _auth_output_root() -> Path:
    """Resolve output root without importing _pipeline (avoids circular deps)."""
    root = os.environ.get("MOLAGENT_OUTPUT_ROOT")
    if root:
        return Path(root).resolve()
    plugin_root = os.environ.get("MOLAGENT_PLUGIN_ROOT")
    base = Path(plugin_root) if plugin_root else Path(__file__).resolve().parent.parent
    return (base / "MolagentFiles").resolve()


def _admin_token_file_path(store_path: Path) -> Path:
    """Sidecar file that holds the plaintext admin token, written once on
    first-run bootstrap so it can be recovered even if stderr is swallowed."""
    return store_path.with_name("admin_token.txt")


def _write_admin_token_file(store_path: Path, token: str) -> Optional[Path]:
    """Write the admin token to a 0600 sidecar file. Best-effort: returns the
    path on success, None on failure (never raises — bootstrap must not fail)."""
    out = _admin_token_file_path(store_path)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        # Create with restrictive perms from the start (mode is a no-op on Windows).
        fd = os.open(str(out), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(token + "\n")
        try:
            os.chmod(out, 0o600)
        except OSError:
            pass
        return out
    except OSError:
        return None


def _token_store_path() -> Path:
    explicit = os.environ.get("MOLAGENT_TOKEN_STORE_PATH")
    if explicit:
        return Path(explicit)
    return _auth_output_root() / _TOKEN_STORE_FILENAME


def _lock_path(store_path: Path) -> Path:
    return store_path.with_suffix(store_path.suffix + ".lock")


def _acquire_lock(lock_path: Path, timeout: float = 10.0) -> int | None:
    deadline = time.monotonic() + timeout
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stale_unlink_attempted = False
    while True:
        try:
            return os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.EACCES):
                raise
            if not stale_unlink_attempted:
                try:
                    age = time.time() - lock_path.stat().st_mtime
                    if age > STALE_LOCK_AGE_SECONDS:
                        try:
                            lock_path.unlink()
                        except FileNotFoundError:
                            pass
                except FileNotFoundError:
                    pass
                stale_unlink_attempted = True
                continue
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


def _load_store(store_path: Path) -> dict:
    if not store_path.exists():
        return {"admin_token_hash": None, "users": {}}
    with open(store_path) as f:
        data = json.load(f)   # raises json.JSONDecodeError on corruption — do not catch
    if not isinstance(data, dict):
        raise ValueError(f"Auth store at {store_path} is not a JSON object")
    data.setdefault("admin_token_hash", None)
    data.setdefault("users", {})
    return data


def _save_store(store_path: Path, data: dict) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".auth-", suffix=".json.tmp", dir=str(store_path.parent)
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, store_path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _generate_token(prefix: str) -> str:
    return f"{prefix}{secrets.token_urlsafe(32)}"


def _hash_token(token: str) -> str:
    """SHA-256 hex digest of a token. Tokens are high-entropy random strings,
    so a plain (unsalted) hash is sufficient — there is nothing to brute-force."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _token_prefix(token: str) -> str:
    """Short display prefix stored for identifying a token in list_users."""
    return token[:20] + "..."


def bootstrap_admin_token() -> Optional[str]:
    """Generate the admin token on first run if none exists.

    Returns the plaintext token on first generation (the only time it can be
    shown), or None if an admin token already exists — the stored hash is
    one-way, so the original plaintext cannot be recovered.
    """
    store_path = _token_store_path()
    lock = _lock_path(store_path)
    fd = _acquire_lock(lock)
    if fd is None:
        raise RuntimeError("Could not acquire token store lock")
    try:
        store = _load_store(store_path)
        if store["admin_token_hash"]:
            # Already bootstrapped; the plaintext is not recoverable from the hash.
            return None
        token = _generate_token("molagent_adm_")
        store["admin_token_hash"] = _hash_token(token)
        _save_store(store_path, store)
        token_file = _write_admin_token_file(store_path, token)
        print(f"[auth] Admin token generated: {token}", file=sys.stderr)
        if token_file is not None:
            print(
                f"[auth] Admin token also written to: {token_file} "
                "(delete after copying — it cannot be recovered later).",
                file=sys.stderr,
            )
        return token
    finally:
        _release_lock(fd, lock)


def validate_token(token: str) -> Optional[dict]:
    """Validate a token. Returns {"user_id": str, "owner_id": str, "is_admin": bool} or None.

    owner_id is a unique opaque identifier per token used for registry ownership.
    """
    if not token:
        return None
    store_path = _token_store_path()
    store = _load_store(store_path)

    token_hash = _hash_token(token)

    admin_hash = store.get("admin_token_hash")
    if admin_hash and secrets.compare_digest(token_hash, admin_hash):
        return {"user_id": ADMIN_USER_ID, "owner_id": ADMIN_USER_ID, "is_admin": True}

    user_entry = store["users"].get(token_hash)
    if user_entry is None:
        return None
    if user_entry.get("revoked", False):
        return None
    return {
        "user_id": user_entry["user_id"],
        "owner_id": user_entry["owner_id"],
        "is_admin": False,
    }


def create_user_token(user_id: str) -> str:
    """Generate and store a new user token. Returns the token string."""
    if user_id in RESERVED_USER_IDS:
        raise ValueError(
            f"user_id {user_id!r} is reserved and cannot be assigned to a token"
        )
    store_path = _token_store_path()
    lock = _lock_path(store_path)
    fd = _acquire_lock(lock)
    if fd is None:
        raise RuntimeError("Could not acquire token store lock")
    try:
        store = _load_store(store_path)
        token = _generate_token("molagent_usr_")
        owner_id = secrets.token_urlsafe(16)
        store["users"][_hash_token(token)] = {
            "user_id": user_id,
            "owner_id": owner_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "revoked": False,
            "token_prefix": _token_prefix(token),
        }
        _save_store(store_path, store)
        return token
    finally:
        _release_lock(fd, lock)


def rotate_user_token(owner_id: str) -> Optional[tuple[str, str]]:
    """Issue a fresh token for an existing user, keyed by their owner_id.

    Use when a user loses their token: because ownership of models/datasets is
    keyed by owner_id (not the token), the new token unlocks all of that user's
    existing artifacts. Replace semantics — the old token stops working.

    Returns ``(new_plaintext_token, user_id)`` (token shown only once), or None
    if no user with that owner_id exists.
    """
    store_path = _token_store_path()
    lock = _lock_path(store_path)
    fd = _acquire_lock(lock)
    if fd is None:
        raise RuntimeError("Could not acquire token store lock")
    try:
        store = _load_store(store_path)
        old_hash = next(
            (h for h, e in store["users"].items() if e.get("owner_id") == owner_id),
            None,
        )
        if old_hash is None:
            return None
        entry = store["users"].pop(old_hash)
        token = _generate_token("molagent_usr_")
        entry["token_prefix"] = _token_prefix(token)
        entry["revoked"] = False
        entry["rotated_at"] = datetime.now().isoformat(timespec="seconds")
        # owner_id and user_id are preserved → existing models/datasets stay accessible.
        store["users"][_hash_token(token)] = entry
        _save_store(store_path, store)
        return token, entry["user_id"]
    finally:
        _release_lock(fd, lock)


def revoke_user(owner_id: str) -> bool:
    """Revoke a user by their unique owner_id. Returns True if found and revoked.

    owner_id is the stable, server-generated handle shown by list_users — unlike
    the plaintext token (which is unrecoverable from the hashed store) or the
    human user_id name (which is not unique).
    """
    store_path = _token_store_path()
    lock = _lock_path(store_path)
    fd = _acquire_lock(lock)
    if fd is None:
        raise RuntimeError("Could not acquire token store lock")
    try:
        store = _load_store(store_path)
        found = False
        for entry in store["users"].values():
            if entry.get("owner_id") == owner_id:
                entry["revoked"] = True
                found = True
                break
        if found:
            _save_store(store_path, store)
        return found
    finally:
        _release_lock(fd, lock)


def list_users() -> list[dict]:
    """List all users with their unique owner_id, name, status, and token prefix.

    Full plaintext tokens are never stored (only SHA-256 hashes), so they cannot
    be listed. Use owner_id (the unique handle) with revoke_user.
    """
    store_path = _token_store_path()
    store = _load_store(store_path)
    result = []
    for token_hash, entry in store["users"].items():
        # Older entries may lack a stored prefix; fall back to the hash prefix.
        prefix = entry.get("token_prefix") or (token_hash[:16] + "...")
        result.append({
            "owner_id": entry.get("owner_id"),
            "user_id": entry["user_id"],
            "created_at": entry.get("created_at"),
            "revoked": entry.get("revoked", False),
            "token_prefix": prefix,
        })
    return result
