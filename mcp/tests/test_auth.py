"""Unit tests for mcp/_auth.py — the token store that gates every remote tool.

Run with:
    uv run pytest mcp/tests/test_auth.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import _auth
from _auth import (
    ADMIN_USER_ID,
    bootstrap_admin_token,
    create_user_token,
    list_users,
    revoke_user,
    rotate_user_token,
    validate_token,
)


def _owner_id_for(token: str) -> str:
    """Resolve the owner_id assigned to a freshly created user token."""
    result = validate_token(token)
    assert result is not None
    return result["owner_id"]


@pytest.fixture(autouse=True)
def isolate_token_store(tmp_path, monkeypatch):
    """Point the token store at a temp file for each test."""
    monkeypatch.setenv("MOLAGENT_TOKEN_STORE_PATH", str(tmp_path / "auth_tokens.json"))
    # Ensure output-root fallbacks don't leak into another dir
    monkeypatch.setenv("MOLAGENT_OUTPUT_ROOT", str(tmp_path))


# ── bootstrap_admin_token ───────────────────────────────────────────────────


def test_bootstrap_generates_admin_token():
    token = bootstrap_admin_token()
    assert token.startswith("molagent_adm_")
    assert len(token) > len("molagent_adm_") + 20


def test_bootstrap_returns_none_on_second_call():
    """Re-bootstrap returns None — the plaintext is only available once, and the
    first token must still validate (no rotation)."""
    t1 = bootstrap_admin_token()
    t2 = bootstrap_admin_token()
    assert t2 is None
    # The originally-issued token still works
    assert validate_token(t1) is not None


def test_admin_token_stored_as_hash(tmp_path, monkeypatch):
    """The plaintext admin token must never be written to the token store."""
    import json

    store = tmp_path / "auth_tokens.json"
    monkeypatch.setenv("MOLAGENT_TOKEN_STORE_PATH", str(store))
    token = bootstrap_admin_token()
    raw = store.read_text()
    assert token not in raw
    data = json.loads(raw)
    assert "admin_token_hash" in data
    assert data.get("admin_token") is None  # no plaintext field


def test_admin_token_sidecar_file_written(tmp_path, monkeypatch):
    """First-run bootstrap writes the plaintext token to a sidecar file."""
    store = tmp_path / "auth_tokens.json"
    monkeypatch.setenv("MOLAGENT_TOKEN_STORE_PATH", str(store))
    token = bootstrap_admin_token()
    sidecar = tmp_path / "admin_token.txt"
    assert sidecar.exists()
    assert sidecar.read_text().strip() == token


def test_admin_token_sidecar_not_rewritten(tmp_path, monkeypatch):
    """Re-bootstrap must not touch the sidecar (token unrecoverable, returns None)."""
    store = tmp_path / "auth_tokens.json"
    monkeypatch.setenv("MOLAGENT_TOKEN_STORE_PATH", str(store))
    bootstrap_admin_token()
    sidecar = tmp_path / "admin_token.txt"
    sidecar.unlink()  # simulate user deleting it after copying
    assert bootstrap_admin_token() is None
    assert not sidecar.exists()


# ── validate_token ──────────────────────────────────────────────────────────


def test_validate_admin_token():
    admin = bootstrap_admin_token()
    result = validate_token(admin)
    assert result is not None
    assert result["is_admin"] is True
    assert result["user_id"] == ADMIN_USER_ID
    assert result["owner_id"] == ADMIN_USER_ID


def test_validate_empty_token_returns_none():
    bootstrap_admin_token()
    assert validate_token("") is None
    assert validate_token(None) is None


def test_validate_unknown_token_returns_none():
    bootstrap_admin_token()
    assert validate_token("molagent_usr_does_not_exist") is None


def test_validate_user_token():
    bootstrap_admin_token()
    token = create_user_token("alice")
    result = validate_token(token)
    assert result is not None
    assert result["is_admin"] is False
    assert result["user_id"] == "alice"
    # owner_id is opaque and distinct from the user_id
    assert result["owner_id"]
    assert result["owner_id"] != "alice"


# ── create_user_token ───────────────────────────────────────────────────────


def test_create_user_token_unique_owner_ids():
    bootstrap_admin_token()
    t1 = create_user_token("bob")
    t2 = create_user_token("bob")
    assert t1 != t2
    r1, r2 = validate_token(t1), validate_token(t2)
    # Same user_id, but isolated owner namespaces
    assert r1["user_id"] == r2["user_id"] == "bob"
    assert r1["owner_id"] != r2["owner_id"]


def test_create_user_token_prefix():
    bootstrap_admin_token()
    token = create_user_token("carol")
    assert token.startswith("molagent_usr_")


# ── revoke_user ─────────────────────────────────────────────────────────────


def test_revoke_user_invalidates():
    bootstrap_admin_token()
    token = create_user_token("dave")
    owner_id = _owner_id_for(token)
    assert validate_token(token) is not None
    assert revoke_user(owner_id) is True
    assert validate_token(token) is None


def test_revoke_unknown_owner_returns_false():
    bootstrap_admin_token()
    assert revoke_user("not-a-real-owner-id") is False


def test_revoke_user_isolates_same_name_users():
    """owner_id is the unique key: revoking one user must not affect another with
    the SAME human user_id name (the live store historically had duplicate names)."""
    bootstrap_admin_token()
    keep = create_user_token("TestDude")
    drop = create_user_token("TestDude")  # same name, different owner_id
    keep_owner = _owner_id_for(keep)
    drop_owner = _owner_id_for(drop)
    assert keep_owner != drop_owner
    assert revoke_user(drop_owner) is True
    assert validate_token(keep) is not None
    assert validate_token(drop) is None


# ── rotate_user_token ───────────────────────────────────────────────────────


def test_rotate_preserves_owner_id_and_swaps_token():
    """Rotation issues a new working token, invalidates the old one, and keeps the
    same owner_id + user_id so the user's artifacts stay accessible."""
    bootstrap_admin_token()
    old_token = create_user_token("kara")
    owner_id = _owner_id_for(old_token)

    rotated = rotate_user_token(owner_id)
    assert rotated is not None
    new_token, user_id = rotated
    assert user_id == "kara"
    assert new_token != old_token
    # Old token no longer authenticates
    assert validate_token(old_token) is None
    # New token works and maps to the SAME owner_id + name
    result = validate_token(new_token)
    assert result is not None
    assert result["owner_id"] == owner_id
    assert result["user_id"] == "kara"


def test_rotate_unknown_owner_returns_none():
    bootstrap_admin_token()
    assert rotate_user_token("not-a-real-owner-id") is None


def test_rotate_clears_revoked_flag():
    """Rotating a previously-revoked user re-activates them under a new token."""
    bootstrap_admin_token()
    token = create_user_token("liam")
    owner_id = _owner_id_for(token)
    revoke_user(owner_id)
    assert validate_token(token) is None

    new_token, _ = rotate_user_token(owner_id)
    assert validate_token(new_token) is not None
    assert next(u for u in list_users() if u["owner_id"] == owner_id)["revoked"] is False


# ── list_users ──────────────────────────────────────────────────────────────


def test_list_users_includes_owner_id_and_name():
    bootstrap_admin_token()
    token = create_user_token("grace")
    users = list_users()
    assert len(users) == 1
    entry = users[0]
    assert "token" not in entry
    assert entry["user_id"] == "grace"
    assert entry["owner_id"] == _owner_id_for(token)
    assert entry["created_at"]
    assert entry["token_prefix"].endswith("...")


def test_list_users_never_returns_plaintext_token():
    """Plaintext tokens are unrecoverable from the hashed store — only a prefix."""
    bootstrap_admin_token()
    token = create_user_token("heidi")
    users = list_users()
    assert "token" not in users[0]
    assert users[0]["token_prefix"] == token[:20] + "..."


def test_user_token_stored_as_hash(tmp_path, monkeypatch):
    """User token plaintext must never be written to disk."""
    store = tmp_path / "auth_tokens.json"
    monkeypatch.setenv("MOLAGENT_TOKEN_STORE_PATH", str(store))
    bootstrap_admin_token()
    token = create_user_token("judy")
    assert token not in store.read_text()
    # But it still validates
    assert validate_token(token) is not None


def test_list_users_reports_revoked_status():
    bootstrap_admin_token()
    token = create_user_token("ivan")
    revoke_user(_owner_id_for(token))
    users = list_users()
    assert users[0]["revoked"] is True


# ── store resilience ────────────────────────────────────────────────────────


def test_corrupt_store_raises_on_validate(tmp_path, monkeypatch):
    """A corrupt token store raises JSONDecodeError — do not silently reset."""
    import json as _json
    store = tmp_path / "auth_tokens.json"
    store.write_text("not valid json {{{")
    monkeypatch.setenv("MOLAGENT_TOKEN_STORE_PATH", str(store))
    # The store is corrupt — validate_token must surface the error, not swallow it.
    # The server's _auth_provider layer handles this by returning auth failures.
    with pytest.raises(_json.JSONDecodeError):
        validate_token("anything")
