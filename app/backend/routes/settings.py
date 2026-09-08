"""Settings routes — MCP connection configuration."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import socket
import sys
import urllib.parse
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..job_store import JobStatus, list_jobs
from ..mcp_client import disconnect, get_mcp_settings, update_mcp_settings

router = APIRouter(prefix="/api/settings", tags=["settings"])

# e.g. "C:\Users\me\out" or "D:/data" — absolute in Windows terms only.
_WINDOWS_ABS = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def _running_under_wsl() -> bool:
    if sys.platform != "linux":
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def _output_folder_warnings(output_folder: str | None) -> list[str]:
    """Warn when output_folder is a Windows path but the backend is not on Windows.

    On Linux/WSL a string like "C:\\Users\\me\\out" is not absolute, so it gets
    created as a *relative* directory whose name contains backslashes — outputs
    then land somewhere unexpected and diverge from the MCP server's own view.
    """
    if not output_folder or sys.platform == "win32":
        return []
    m = _WINDOWS_ABS.match(output_folder)
    if not m:
        return []
    drive, rest = m.group(1).lower(), m.group(2).replace("\\", "/")
    where = "WSL" if _running_under_wsl() else f"a non-Windows host ({sys.platform})"
    suggestion = f"/mnt/{drive}/{rest}"
    return [
        f"'{output_folder}' looks like a Windows path, but this backend runs on {where}, "
        f"where it is not absolute — it will be created as a relative folder of that "
        f"literal name instead. Use '{suggestion}' if you meant that Windows location."
    ]


class MCPSettingsResponse(BaseModel):
    mode: str
    url: str | None
    server_path: str | None
    has_auth: bool
    output_folder: str | None
    warnings: list[str] = []


class MCPSettingsUpdate(BaseModel):
    mode: Literal["local", "remote"] | None = None
    url: str | None = None
    server_path: str | None = None
    auth_token: str | None = None
    output_folder: str | None = None

    model_config = {"protected_namespaces": ()}


@router.get("", response_model=MCPSettingsResponse)
async def get_settings():
    s = get_mcp_settings()
    return MCPSettingsResponse(
        mode=s.mode,
        url=s.url,
        server_path=s.server_path,
        has_auth=s.auth_token is not None,
        output_folder=s.output_folder,
        warnings=_output_folder_warnings(s.output_folder),
    )


@router.put("", response_model=MCPSettingsResponse)
async def put_settings(req: MCPSettingsUpdate):
    # Validate before touching the live connection — a rejected request must not
    # disconnect the MCP client as a side effect.
    if req.server_path is not None:
        plugin_root_env = os.environ.get("MOLAGENT_PLUGIN_ROOT", "")
        if not plugin_root_env:
            raise HTTPException(400, "MOLAGENT_PLUGIN_ROOT not set — cannot validate server_path")
        plugin_root = Path(plugin_root_env).resolve()
        try:
            resolved = Path(req.server_path).resolve()
            resolved.relative_to(plugin_root)
        except ValueError:
            raise HTTPException(400, f"server_path must be inside the plugin root ({plugin_root})")

    if req.url is not None:
        parsed = urllib.parse.urlparse(req.url)
        if parsed.scheme not in ("http", "https"):
            raise HTTPException(400, "url scheme must be http or https")
        host = (parsed.hostname or "").rstrip(".").lower()
        if not host:
            raise HTTPException(400, "url must have a host")
        # Loopback and private ranges are ALLOWED on purpose: the documented
        # deployments are a local MCP server (http://127.0.0.1:8001/mcp) and
        # on-prem servers on a LAN. Blocking them would break normal use, and it
        # would not mitigate the actual W-H1 risk anyway — token exfiltration
        # targets an attacker's *public* endpoint. What we do block is
        # cloud-instance metadata (link-local) plus non-routable oddities.
        # getaddrinfo is blocking, so it runs off the event loop.
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            infos = await asyncio.to_thread(socket.getaddrinfo, host, port)
        except socket.gaierror:
            infos = []
        if not infos:
            # Fail closed: an unresolvable host cannot be screened, so we don't
            # let it through. If the backend reaches the MCP server through an
            # HTTPS_PROXY without resolving it directly, use the proxy-side
            # hostname or an IP literal here.
            raise HTTPException(
                400,
                f"url host {host!r} could not be resolved, so it cannot be validated. "
                "Use an IP literal if this host is only resolvable via a proxy.",
            )
        for _family, _type, _proto, _canonname, sockaddr in infos:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
                raise HTTPException(400, f"url resolves to disallowed address {ip}")

    connection_changed = any(
        x is not None for x in (req.mode, req.url, req.server_path, req.auth_token)
    )
    if connection_changed:
        running = [j for j in list_jobs() if j.status in (JobStatus.PENDING, JobStatus.RUNNING)]
        if running:
            raise HTTPException(
                409, f"Cannot change connection settings while {len(running)} job(s) are still running."
            )
        await disconnect()

    s = update_mcp_settings(
        mode=req.mode,
        url=req.url,
        server_path=req.server_path,
        auth_token=req.auth_token,
        output_folder=req.output_folder,
    )
    return MCPSettingsResponse(
        mode=s.mode,
        url=s.url,
        server_path=s.server_path,
        has_auth=s.auth_token is not None,
        output_folder=s.output_folder,
        warnings=_output_folder_warnings(s.output_folder),
    )
