#!/usr/bin/env bash
# Launch the MolAgent MCP server using the configured venv.
# Set AUTOMOL_VENV to your virtual environment path before starting Claude Code.
# Fallback: .venv in the repo root (created by the standard install instructions).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${AUTOMOL_VENV:-${SCRIPT_DIR}/../.venv}"
export AUTOMOL_VENV="${VENV}"
exec "${VENV}/bin/python" "${SCRIPT_DIR}/server.py"
