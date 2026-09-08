"""Integration tests for per-caller authorization in mcp/server.py.

These exercise the ownership-isolation and path-restriction logic that gates
multi-tenant remote access. Rather than standing up the full auth transport,
we monkeypatch server._get_caller to simulate a specific caller identity; the
real in-tool authorization checks then run unchanged.

Run with:
    uv run pytest mcp/tests/test_auth_enforcement.py -v
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
from pathlib import Path

import pytest
from fastmcp import Client

sys.path.insert(0, str(Path(__file__).parent.parent))

import server

try:
    import rdkit  # noqa: F401
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False


@pytest.fixture(autouse=True)
def isolate_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MOLAGENT_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.delenv("MOLAGENT_AUTH_REQUIRED", raising=False)
    monkeypatch.delenv("PHARMAOS_MOLAGENT_ROOT", raising=False)
    monkeypatch.delenv("MOLAGENT_REGISTRY_PATH", raising=False)


def _as_caller(monkeypatch, *, user_id: str, owner_id: str, is_admin: bool):
    """Force server._get_caller to return a fixed caller for the next calls."""
    caller = {"user_id": user_id, "owner_id": owner_id, "is_admin": is_admin}
    monkeypatch.setattr(server, "_get_caller", lambda ctx: caller)
    # Non-admin callers must look like remote callers so path-exposure is off.
    if not is_admin:
        monkeypatch.setattr(server, "_should_expose_paths", lambda: False)
    return caller


def _b64_csv(rows: int = 5) -> tuple[str, str]:
    lines = ["smiles,logP"] + [f"CCO{i},{1.0 + i * 0.1:.2f}" for i in range(rows)]
    return "data.csv", base64.b64encode("\n".join(lines).encode()).decode()


def _upload_as(monkeypatch, *, owner_id: str) -> str:
    """Upload a dataset as a given owner; returns the dataset_id."""
    _as_caller(monkeypatch, user_id=owner_id, owner_id=owner_id, is_admin=False)
    filename, b64 = _b64_csv()

    async def _run():
        async with Client(server.mcp) as client:
            r = await client.call_tool(
                "upload_dataset", {"filename": filename, "file_content_b64": b64}
            )
            return r.data["dataset_id"]

    return asyncio.run(_run())


# ── Path restriction (the file-disclosure fix) ──────────────────────────────


def test_non_admin_cannot_pass_csv_file_path(monkeypatch, tmp_path):
    """A non-admin caller must not read arbitrary server files via csv_file."""
    secret = tmp_path / "secret.csv"
    secret.write_text("smiles,logP\nCCO,1.0\n")
    _as_caller(monkeypatch, user_id="mallory", owner_id="own_m", is_admin=False)

    async def _run():
        async with Client(server.mcp) as client:
            await client.call_tool("start_training_session", {"csv_file": str(secret)})

    with pytest.raises(Exception) as exc:
        asyncio.run(_run())
    assert "not allowed" in str(exc.value).lower()


def test_non_admin_cannot_pass_smiles_file_path(monkeypatch, tmp_path):
    """predict(smiles_file=<path>) must be rejected for non-admin callers."""
    secret = tmp_path / "secret.csv"
    secret.write_text("smiles\nCCO\n")
    _as_caller(monkeypatch, user_id="mallory", owner_id="own_m", is_admin=False)

    async def _run():
        async with Client(server.mcp) as client:
            await client.call_tool(
                "predict",
                {"model_id": "whatever", "smiles_file": str(secret)},
            )

    with pytest.raises(Exception) as exc:
        asyncio.run(_run())
    assert "not allowed" in str(exc.value).lower()


def test_non_admin_ds_named_file_does_not_bypass(monkeypatch, tmp_path):
    """A file literally named 'ds_*' must not shadow the registry: a non-admin
    passing smiles_file='ds_evil.csv' that exists on disk is still rejected,
    because ds_ values resolve registry-first, never as a path."""
    # Run from a cwd containing a real 'ds_evil.csv' so a bare relative path exists.
    evil = tmp_path / "ds_evil.csv"
    evil.write_text("smiles\nCCO\n")
    monkeypatch.chdir(tmp_path)
    _as_caller(monkeypatch, user_id="mallory", owner_id="own_m", is_admin=False)

    async def _run():
        async with Client(server.mcp) as client:
            await client.call_tool(
                "predict",
                {"model_id": "whatever", "smiles_file": "ds_evil.csv"},
            )

    with pytest.raises(Exception) as exc:
        asyncio.run(_run())
    # Resolved registry-first → unknown id → "not found", never fed as a path.
    assert "not found in data registry" in str(exc.value).lower()


def test_non_admin_cannot_pass_model_file_path(monkeypatch, tmp_path):
    """predict(model_file=<path>) remains blocked for non-admin callers."""
    _as_caller(monkeypatch, user_id="mallory", owner_id="own_m", is_admin=False)

    async def _run():
        async with Client(server.mcp) as client:
            await client.call_tool(
                "predict",
                {"model_file": str(tmp_path / "x.pt"), "smiles_list": ["CCO"]},
            )

    with pytest.raises(Exception) as exc:
        asyncio.run(_run())
    assert "not allowed" in str(exc.value).lower()


def test_admin_may_pass_csv_file_path(monkeypatch, tmp_path):
    """Admin/local callers are still allowed to use direct paths."""
    csv = tmp_path / "ok.csv"
    csv.write_text("smiles,logP\nCCO,1.0\nCCC,2.0\n")
    _as_caller(monkeypatch, user_id="__admin__", owner_id="__admin__", is_admin=True)

    async def _run():
        async with Client(server.mcp) as client:
            r = await client.call_tool("start_training_session", {"csv_file": str(csv)})
            return r.data

    data = asyncio.run(_run())
    assert "session_id" in data


# ── Remote session round-trip references dataset_id, not a path ─────────────


def test_remote_session_config_uses_dataset_id(monkeypatch):
    """A non-admin start_training_session must emit csv_file as a dataset_id.

    Regression guard: the path-restriction gate would otherwise reject the
    config that the session itself produced (it baked in an absolute path).
    """
    ds_id = _upload_as(monkeypatch, owner_id="own_alice")

    _as_caller(monkeypatch, user_id="alice", owner_id="own_alice", is_admin=False)

    async def _run():
        async with Client(server.mcp) as client:
            start = await client.call_tool(
                "start_training_session", {"dataset_id": ds_id}
            )
            sid = start.data["session_id"]
            answer = await client.call_tool(
                "answer_training_question",
                {"session_id": sid, "properties": ["logP"], "confirm": True},
            )
            return start.data, answer.data

    start_data, answer_data = asyncio.run(_run())
    # Session should reference the dataset by id, not an absolute path
    assert start_data["detected"]["csv_file"] == ds_id
    assert answer_data["config"]["csv_file"] == ds_id


def test_admin_session_config_uses_path(monkeypatch, tmp_path):
    """Admin/local sessions keep the absolute path in the config (unchanged)."""
    csv = tmp_path / "ok.csv"
    csv.write_text("smiles,logP\nCCO,1.0\nCCC,2.0\n")
    _as_caller(monkeypatch, user_id="__admin__", owner_id="__admin__", is_admin=True)

    async def _run():
        async with Client(server.mcp) as client:
            start = await client.call_tool(
                "start_training_session", {"csv_file": str(csv)}
            )
            return start.data

    data = asyncio.run(_run())
    assert data["detected"]["csv_file"].endswith("ok.csv")
    assert not data["detected"]["csv_file"].startswith("ds_")


@pytest.mark.skipif(not _HAS_RDKIT, reason="rdkit not installed")
def test_remote_3d_structure_train_resolves_extracted_csv_not_raw_directory(monkeypatch):
    """A non-admin train_and_visualize on a 3d_structure dataset_id must resolve
    to the extracted prepared/data.csv, not the raw structure directory.

    Regression guard: _resolve_data_reference previously returned the raw
    structures/<owner>/<target>/ directory for a 3d_structure entry (it only
    had a single "return the file_path" branch, same as for plain CSVs),
    which crashed pandas.read_csv() with IsADirectoryError once training
    actually started — this only surfaced for non-admin/remote callers, since
    admin/local sessions bake in the already-resolved csv path directly and
    never call _resolve_data_reference at all.
    """
    import importlib
    _3d_tools_tests = importlib.import_module("test_3d_tools")

    _as_caller(monkeypatch, user_id="joris", owner_id="own_joris", is_admin=False)

    pdb_json = json.dumps([
        {"filename": "2E2B_receptor.pdb", "content_b64": base64.b64encode(_3d_tools_tests._make_minimal_pdb()).decode()}
    ])

    async def _run():
        async with Client(server.mcp) as client:
            upload = await client.call_tool("upload_3d_structure", {
                "target_name": "ABL",
                "sdf_content_b64": base64.b64encode(_3d_tools_tests._make_minimal_sdf()).decode(),
                "pdb_files": pdb_json,
            })
            structure_id = upload.data["structure_id"]
            start = await client.call_tool("start_training_session", {
                "dataset_id": structure_id,
            })
            sid = start.data["session_id"]
            answer = await client.call_tool("answer_training_question", {
                "session_id": sid,
                "properties": ["pChEMBL"],
                "confirm": True,
            })
            return structure_id, answer.data["config"]

    structure_id, config = asyncio.run(_run())
    # csv_file in the confirmed config is the dataset_id (non-admin path-hiding rule)
    assert config["csv_file"] == structure_id

    resolved = server._resolve_data_reference(
        {"user_id": "joris", "owner_id": "own_joris", "is_admin": False},
        config["csv_file"],
        what="csv_file",
    )
    assert resolved.endswith("data.csv")
    assert Path(resolved).is_file(), (
        f"_resolve_data_reference returned a non-file path for a 3d_structure "
        f"dataset_id: {resolved!r} — this reproduces the IsADirectoryError bug."
    )


# ── Cross-user dataset isolation ─────────────────────────────────────────────


def test_user_cannot_train_on_another_users_dataset(monkeypatch):
    alice_ds = _upload_as(monkeypatch, owner_id="own_alice")

    # Bob tries to use Alice's dataset_id
    _as_caller(monkeypatch, user_id="bob", owner_id="own_bob", is_admin=False)

    async def _run():
        async with Client(server.mcp) as client:
            await client.call_tool("start_training_session", {"dataset_id": alice_ds})

    with pytest.raises(Exception) as exc:
        asyncio.run(_run())
    assert "does not belong to you" in str(exc.value).lower()


def test_user_cannot_predict_on_another_users_dataset(monkeypatch):
    alice_ds = _upload_as(monkeypatch, owner_id="own_alice")

    _as_caller(monkeypatch, user_id="bob", owner_id="own_bob", is_admin=False)

    async def _run():
        async with Client(server.mcp) as client:
            await client.call_tool(
                "predict", {"model_id": "anything", "smiles_file": alice_ds}
            )

    with pytest.raises(Exception) as exc:
        asyncio.run(_run())
    msg = str(exc.value).lower()
    assert "does not belong to you" in msg or "not found" in msg


def test_user_only_sees_own_datasets(monkeypatch):
    _upload_as(monkeypatch, owner_id="own_alice")
    _upload_as(monkeypatch, owner_id="own_bob")

    _as_caller(monkeypatch, user_id="bob", owner_id="own_bob", is_admin=False)

    async def _run():
        async with Client(server.mcp) as client:
            return (await client.call_tool("list_datasets", {})).data

    data = asyncio.run(_run())
    owners = {d["owner"] for d in data["datasets"]}
    assert owners == {"own_bob"}


def test_admin_sees_all_datasets(monkeypatch):
    _upload_as(monkeypatch, owner_id="own_alice")
    _upload_as(monkeypatch, owner_id="own_bob")

    _as_caller(monkeypatch, user_id="__admin__", owner_id="__admin__", is_admin=True)

    async def _run():
        async with Client(server.mcp) as client:
            return (await client.call_tool("list_datasets", {})).data

    data = asyncio.run(_run())
    assert len(data["datasets"]) == 2


def test_user_cannot_delete_another_users_dataset(monkeypatch):
    alice_ds = _upload_as(monkeypatch, owner_id="own_alice")

    _as_caller(monkeypatch, user_id="bob", owner_id="own_bob", is_admin=False)

    async def _run():
        async with Client(server.mcp) as client:
            await client.call_tool("delete_dataset", {"dataset_id": alice_ds})

    with pytest.raises(Exception) as exc:
        asyncio.run(_run())
    msg = str(exc.value).lower()
    assert "not found" in msg or "access denied" in msg


# ── admin_manage requires admin ──────────────────────────────────────────────


def test_non_admin_cannot_call_admin_manage(monkeypatch):
    _as_caller(monkeypatch, user_id="bob", owner_id="own_bob", is_admin=False)

    async def _run():
        async with Client(server.mcp) as client:
            await client.call_tool(
                "admin_manage", {"action": "list_users"}
            )

    with pytest.raises(Exception) as exc:
        asyncio.run(_run())
    assert "admin" in str(exc.value).lower()


def test_admin_create_list_revoke_user_by_owner_id(monkeypatch, tmp_path):
    """End-to-end admin flow: create a user, find its owner_id via list_users,
    revoke by owner_id, and confirm list_users reflects the revoked status."""
    monkeypatch.setenv("MOLAGENT_TOKEN_STORE_PATH", str(tmp_path / "auth_tokens.json"))
    _as_caller(monkeypatch, user_id="__admin__", owner_id="__admin__", is_admin=True)

    async def _run():
        async with Client(server.mcp) as client:
            await client.call_tool(
                "admin_manage", {"action": "create_token", "user_id": "carol"}
            )
            listed = await client.call_tool("admin_manage", {"action": "list_users"})
            owner_id = listed.data["users"][0]["owner_id"]
            assert owner_id  # surfaced for revocation
            revoked = await client.call_tool(
                "admin_manage", {"action": "revoke_user", "owner_id": owner_id}
            )
            after = await client.call_tool("admin_manage", {"action": "list_users"})
            return owner_id, revoked.data, after.data

    owner_id, revoked, after = asyncio.run(_run())
    assert revoked["status"] == "revoked"
    assert revoked["owner_id"] == owner_id
    assert after["users"][0]["revoked"] is True


def test_admin_revoke_unknown_owner_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("MOLAGENT_TOKEN_STORE_PATH", str(tmp_path / "auth_tokens.json"))
    _as_caller(monkeypatch, user_id="__admin__", owner_id="__admin__", is_admin=True)

    async def _run():
        async with Client(server.mcp) as client:
            await client.call_tool(
                "admin_manage", {"action": "revoke_user", "owner_id": "nope"}
            )

    with pytest.raises(Exception) as exc:
        asyncio.run(_run())
    assert "owner_id" in str(exc.value).lower() or "not" in str(exc.value).lower()


def test_rotated_token_keeps_access_to_owned_dataset(monkeypatch, tmp_path):
    """A user who loses their token gets a rotated one; because owner_id is
    preserved, their existing dataset remains accessible under the new token."""
    monkeypatch.setenv("MOLAGENT_TOKEN_STORE_PATH", str(tmp_path / "auth_tokens.json"))
    import _auth

    # Admin creates a real user with a real owner_id
    _as_caller(monkeypatch, user_id="__admin__", owner_id="__admin__", is_admin=True)
    user_token = _auth.create_user_token("nina")
    user = _auth.validate_token(user_token)
    owner_id = user["owner_id"]

    # The user uploads a dataset (as themselves)
    _as_caller(monkeypatch, user_id="nina", owner_id=owner_id, is_admin=False)
    filename, b64 = _b64_csv()

    async def _upload():
        async with Client(server.mcp) as client:
            return (await client.call_tool(
                "upload_dataset", {"filename": filename, "file_content_b64": b64}
            )).data["dataset_id"]

    ds_id = asyncio.run(_upload())

    # Admin rotates the user's token (user lost the old one)
    rotated_pair = _auth.rotate_user_token(owner_id)
    assert rotated_pair is not None
    new_token, rotated_name = rotated_pair
    assert new_token != user_token and rotated_name == "nina"
    rotated = _auth.validate_token(new_token)
    assert rotated["owner_id"] == owner_id  # same identity

    # The rotated identity (same owner_id) still sees + can train on the dataset
    _as_caller(monkeypatch, user_id="nina", owner_id=owner_id, is_admin=False)

    async def _use():
        async with Client(server.mcp) as client:
            listed = (await client.call_tool("list_datasets", {})).data
            session = (await client.call_tool(
                "start_training_session", {"dataset_id": ds_id}
            )).data
            return listed, session

    listed, session = asyncio.run(_use())
    assert any(d["id"] == ds_id for d in listed["datasets"])
    assert "session_id" in session


# ── upload size cap ──────────────────────────────────────────────────────────


# ── _get_caller resolution (local + token mode) ─────────────────────────────


def test_get_caller_none_when_auth_not_required(monkeypatch):
    monkeypatch.setattr(server, "_AUTH_REQUIRED", False)
    assert server._get_caller(ctx=None) is None


def test_get_caller_uses_env_token_on_stdio(monkeypatch):
    """With auth required but no transport token (stdio), MOLAGENT_CALLER_TOKEN
    is validated to yield a real caller identity."""
    import _auth

    monkeypatch.setenv("MOLAGENT_TOKEN_STORE_PATH", str(Path.cwd() / "_t.json"))
    # Fresh store
    store = Path.cwd() / "_t.json"
    if store.exists():
        store.unlink()
    _auth.bootstrap_admin_token()
    token = _auth.create_user_token("local_user")

    monkeypatch.setattr(server, "_AUTH_REQUIRED", True)
    monkeypatch.setattr(server, "get_access_token", lambda: None)
    monkeypatch.setenv("MOLAGENT_CALLER_TOKEN", token)
    try:
        caller = server._get_caller(ctx=None)
        assert caller is not None
        assert caller["user_id"] == "local_user"
        assert caller["is_admin"] is False
    finally:
        if store.exists():
            store.unlink()


def test_get_caller_anonymous_when_no_token(monkeypatch):
    """Auth required, no transport token, no env token → anonymous (rejected)."""
    monkeypatch.setattr(server, "_AUTH_REQUIRED", True)
    monkeypatch.setattr(server, "get_access_token", lambda: None)
    monkeypatch.delenv("MOLAGENT_CALLER_TOKEN", raising=False)
    caller = server._get_caller(ctx=None)
    assert caller["user_id"] == server.ANONYMOUS_USER_ID
    with pytest.raises(Exception):
        server._require_auth(caller)


def test_upload_rejects_oversized_payload(monkeypatch):
    monkeypatch.setenv("MOLAGENT_MAX_UPLOAD_MB", "1")
    _as_caller(monkeypatch, user_id="bob", owner_id="own_bob", is_admin=False)
    # ~2 MB of decoded content
    big = "smiles,logP\n" + ("CCO,1.0\n" * 260_000)
    b64 = base64.b64encode(big.encode()).decode()

    async def _run():
        async with Client(server.mcp) as client:
            await client.call_tool(
                "upload_dataset", {"filename": "big.csv", "file_content_b64": b64}
            )

    with pytest.raises(Exception) as exc:
        asyncio.run(_run())
    assert "maximum size" in str(exc.value).lower()
