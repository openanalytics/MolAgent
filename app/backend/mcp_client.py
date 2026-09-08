"""MCP client abstraction — persistent fastmcp.Client with task support.

Supports local (stdio, keep_alive=True) and remote (streamable-http) transports.
Long-running tools (train_and_visualize, predict) use call_tool_task for background
execution with progress tracking and cancellation.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal

from fastmcp import Client
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from fastmcp_tasks import call_tool_task

from .config import settings

logger = logging.getLogger(__name__)

_SETTINGS_FILE = Path.home() / ".molagent" / "app_settings.json"


@dataclass
class MCPSettings:
    mode: Literal["local", "remote"] = "local"
    url: str | None = None
    server_path: str | None = None
    auth_token: str | None = None
    output_folder: str | None = None


def _load_settings() -> MCPSettings:
    try:
        data = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
        return MCPSettings(
            mode=data.get("mode", "local"),
            url=data.get("url"),
            server_path=data.get("server_path"),
            output_folder=data.get("output_folder"),
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return MCPSettings()


# auth_token is intentionally excluded from persistence (credential safety).
_PERSIST_FIELDS = ("mode", "url", "server_path", "output_folder")


def _persist_settings(s: MCPSettings) -> None:
    try:
        _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in asdict(s).items() if k in _PERSIST_FIELDS}
        _SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        logger.debug("Failed to persist settings to %s", _SETTINGS_FILE)


_settings = _load_settings()
_client: Client | None = None


def get_mcp_settings() -> MCPSettings:
    return _settings


def update_mcp_settings(
    mode: Literal["local", "remote"] | None = None,
    url: str | None = None,
    server_path: str | None = None,
    auth_token: str | None = None,
    output_folder: str | None = None,
) -> MCPSettings:
    global _client
    connection_changed = False
    if mode is not None:
        _settings.mode = mode
        connection_changed = True
    if url is not None:
        _settings.url = url
        connection_changed = True
    if server_path is not None:
        _settings.server_path = server_path
        connection_changed = True
    if auth_token is not None:
        _settings.auth_token = auth_token
        connection_changed = True
    if output_folder is not None:
        _settings.output_folder = output_folder
    if connection_changed:
        _client = None
    _persist_settings(_settings)
    return _settings


def _default_server_path() -> str:
    return str(settings.plugin_root / "mcp" / "server.py")


def _build_transport() -> StdioTransport | StreamableHttpTransport:
    """Build the appropriate transport based on current settings."""
    if _settings.mode == "remote":
        url = _settings.url
        if not url:
            raise RuntimeError("Remote MCP URL not configured")
        headers = {}
        if _settings.auth_token:
            headers["Authorization"] = f"Bearer {_settings.auth_token}"
        return StreamableHttpTransport(url=url, headers=headers)

    # Local stdio mode — keep_alive=True so the subprocess persists
    server_path = _settings.server_path or _default_server_path()
    venv_path = os.environ.get("AUTOMOL_VENV") or str(settings.plugin_root / ".venv")
    scripts_dir = str(Path(server_path).parent)

    env = {
        **os.environ,
        "MOLAGENT_PLUGIN_ROOT": str(settings.plugin_root),
        "MOLAGENT_OUTPUT_ROOT": str(settings.output_root),
        "PYTHONPATH": scripts_dir,
        "VIRTUAL_ENV": venv_path,
    }

    if _settings.auth_token:
        env["MOLAGENT_AUTH_REQUIRED"] = "true"
        env["MOLAGENT_CALLER_TOKEN"] = _settings.auth_token

    return StdioTransport(
        command=settings.uv_path,
        args=["run", "--active", "--no-sync", server_path],
        env=env,
        cwd=scripts_dir,
        keep_alive=True,
    )


def _get_client() -> Client:
    """Get or create a persistent MCP client."""
    global _client
    if _client is None:
        transport = _build_transport()
        _client = Client(transport, timeout=3600)
    return _client


async def ensure_connected() -> Client:
    """Ensure the client is connected. Call before any tool use."""
    client = _get_client()
    if not client.is_connected():
        await client.__aenter__()
    return client


async def disconnect() -> None:
    """Disconnect the persistent client. Call during app shutdown or settings change."""
    global _client
    if _client is not None:
        try:
            await _client.__aexit__(None, None, None)
        except Exception:
            logger.debug("Error during client disconnect", exc_info=True)
        _client = None


async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
    """Call an MCP tool synchronously (blocking until result).

    Use for fast tools like list_models, start_training_session, etc.
    """
    client = await ensure_connected()
    try:
        result = await client.call_tool(name, arguments)
        return _parse_result(result)
    except Exception as exc:
        _check_auth_error(exc)
        raise RuntimeError(f"MCP call '{name}' failed: {exc}") from exc


async def call_tool_as_task(name: str, arguments: dict[str, Any]):
    """Call an MCP tool as a background task (returns immediately).

    Returns a ToolTask handle with .task_id, .status(), .result(), .cancel().
    Use for long-running tools like train_and_visualize, predict.
    """
    client = await ensure_connected()
    try:
        task = await call_tool_task(client, name, arguments)
        return task
    except Exception as exc:
        _check_auth_error(exc)
        raise RuntimeError(f"MCP task call '{name}' failed: {exc}") from exc


def _parse_result(result) -> Any:
    """Parse a CallToolResult into a Python dict/str."""
    import json

    if result is None:
        return None

    # fastmcp.Client.call_tool returns parsed content directly
    # If it's already a dict or list, return as-is
    if isinstance(result, (dict, list)):
        return result

    # If it's a string, try JSON parsing
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (json.JSONDecodeError, ValueError):
            return result

    # FastMCP 4: CallToolResult.data is the parsed Python value (preferred)
    if hasattr(result, "data") and result.data is not None:
        return result.data

    # Fallback: parse text content blocks
    if hasattr(result, "content"):
        texts = []
        for content in result.content:
            if hasattr(content, "text"):
                texts.append(content.text)
        combined = "\n".join(texts)
        try:
            return json.loads(combined)
        except (json.JSONDecodeError, ValueError):
            return combined

    return result


class MCPAuthError(RuntimeError):
    """Raised when the MCP server rejects a request due to auth."""
    pass


def _check_auth_error(exc: Exception) -> None:
    """Re-raise as MCPAuthError if the exception indicates auth failure."""
    msg = str(exc)
    if "Authentication required" in msg or "Access denied" in msg:
        raise MCPAuthError(msg) from exc
    if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
        if exc.response.status_code in (401, 403):
            raise MCPAuthError(f"Authentication failed ({exc.response.status_code})") from exc
