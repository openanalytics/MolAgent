"""Tests that purge actions reject max_age_days < 1.

Run with:
    uv run pytest mcp/tests/test_purge_validation.py -v
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from fastmcp import Client

sys.path.insert(0, str(Path(__file__).parent.parent))

import server


@pytest.fixture(autouse=True)
def isolate_env(tmp_path, monkeypatch):
    """Isolate all registry files and auth tokens to a temp directory."""
    monkeypatch.setenv("MOLAGENT_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.delenv("MOLAGENT_AUTH_REQUIRED", raising=False)
    monkeypatch.delenv("MOLAGENT_REGISTRY_PATH", raising=False)


def _as_admin(monkeypatch):
    """Force server._get_caller to return an admin caller."""
    caller = {"user_id": "admin", "owner_id": "admin_owner", "is_admin": True}
    monkeypatch.setattr(server, "_get_caller", lambda ctx: caller)


@pytest.mark.asyncio
async def test_purge_stale_rejects_zero_days(monkeypatch):
    """purge_stale with max_age_days=0 must raise, not purge everything."""
    _as_admin(monkeypatch)

    async with Client(server.mcp) as client:
        with pytest.raises(Exception, match="max_age_days"):
            await client.call_tool("admin_manage", {
                "action": "purge_stale",
                "max_age_days": 0,
            })


@pytest.mark.asyncio
async def test_purge_orphans_rejects_zero_days(monkeypatch):
    """purge_orphans with max_age_days=0 must raise, not purge everything."""
    _as_admin(monkeypatch)

    async with Client(server.mcp) as client:
        with pytest.raises(Exception, match="max_age_days"):
            await client.call_tool("admin_manage", {
                "action": "purge_orphans",
                "max_age_days": 0,
            })


@pytest.mark.asyncio
async def test_purge_stale_rejects_negative_days(monkeypatch):
    """purge_stale with negative max_age_days must raise."""
    _as_admin(monkeypatch)

    async with Client(server.mcp) as client:
        with pytest.raises(Exception, match="max_age_days"):
            await client.call_tool("admin_manage", {
                "action": "purge_stale",
                "max_age_days": -5,
            })


@pytest.mark.asyncio
async def test_purge_orphans_rejects_negative_days(monkeypatch):
    """purge_orphans with negative max_age_days must raise."""
    _as_admin(monkeypatch)

    async with Client(server.mcp) as client:
        with pytest.raises(Exception, match="max_age_days"):
            await client.call_tool("admin_manage", {
                "action": "purge_orphans",
                "max_age_days": -5,
            })


@pytest.mark.asyncio
async def test_purge_stale_accepts_one_day(monkeypatch):
    """purge_stale with max_age_days=1 must succeed (dry-run)."""
    _as_admin(monkeypatch)

    async with Client(server.mcp) as client:
        result = await client.call_tool("admin_manage", {
            "action": "purge_stale",
            "max_age_days": 1,
            "force": False,
        })
        data = result.data
        assert data.get("dry_run") is True


@pytest.mark.asyncio
async def test_purge_orphans_accepts_one_day(monkeypatch):
    """purge_orphans with max_age_days=1 must succeed (dry-run)."""
    _as_admin(monkeypatch)

    async with Client(server.mcp) as client:
        result = await client.call_tool("admin_manage", {
            "action": "purge_orphans",
            "max_age_days": 1,
            "force": False,
        })
        data = result.data
        assert data.get("dry_run") is True
