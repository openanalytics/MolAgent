#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["smolagents[toolkit,mcp]", "gradio", "mcp"]
# ///
"""Standalone smolagents + Gradio chat demo for the MolAgent MCP server.

Connects to an already-running `mcp/server.py --transport streamable-http`
instance and exposes its tools through a smolagents ToolCallingAgent in a
Gradio chat UI.

Prerequisites — the MCP server MUST be started with auth enabled:

    MOLAGENT_AUTH_REQUIRED=true \\
    MOLAGENT_OUTPUT_ROOT=/abs/path/to/output \\
    uv run mcp/server.py --transport streamable-http --host 127.0.0.1 --port 8001

Mint a token (once per user):

    uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp \\
        --token <ADMIN_TOKEN> create-user gradio-demo

Then run this script:

    MOLAGENT_MCP_URL=http://127.0.0.1:8001/mcp \\
    MOLAGENT_TOKEN=<USER_TOKEN> \\
    HF_TOKEN=<HF_TOKEN> \\
    uv run demos/gradio_mcp_agent.py
"""
from __future__ import annotations

import os
import sys

from smolagents import GradioUI, InferenceClientModel, ToolCallingAgent, ToolCollection


def _require_env(name: str) -> str:
    """Return the env var's value, or exit 1 with a clear message if unset."""
    value = os.environ.get(name)
    if not value:
        print(f"Error: required environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return value


def get_config() -> dict:
    """Read and validate all env vars this demo needs."""
    return {
        "mcp_url": _require_env("MOLAGENT_MCP_URL"),
        "mcp_token": _require_env("MOLAGENT_TOKEN"),  # mandatory: server rejects unauthenticated streamable-http
        "hf_model_id": os.environ.get("HF_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct"),
        "hf_token": os.environ.get("HF_TOKEN"),
    }


def build_server_parameters(config: dict) -> dict:
    """Build the streamable-http server_parameters dict for ToolCollection.from_mcp."""
    headers = {"Authorization": f"Bearer {config['mcp_token']}"}
    return {
        "url": config["mcp_url"],
        "transport": "streamable-http",
        "headers": headers,
    }


def build_agent(config: dict, tools: list) -> ToolCallingAgent:
    """Construct the ToolCallingAgent backed by an HF Inference API model."""
    model = InferenceClientModel(model_id=config["hf_model_id"], token=config["hf_token"])
    return ToolCallingAgent(tools=tools, model=model)


def main() -> None:
    config = get_config()
    server_parameters = build_server_parameters(config)
    try:
        with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
            agent = build_agent(config, list(tool_collection.tools))
            print(f"Agent ready with {len(tool_collection.tools)} MCP tools. Launching Gradio UI...")
            GradioUI(agent, file_upload_folder="./gradio_uploads").launch()
    except Exception as exc:
        print(
            f"Error: could not connect to MCP server at {config['mcp_url']}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
