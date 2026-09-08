# CLAUDE.md

## Project Overview

MolAgent is a multi-agent system for AI-driven molecular property prediction in early-stage drug discovery. It provides an MCP server, Claude Code plugin, and web application wrapping the AutoMol ML framework.

## Key Commands

### Installation
```bash
git clone --recurse-submodules https://github.com/openanalytics/MolAgent
pip install uv

uv venv .venv --python 3.12
source .venv/bin/activate
export AUTOMOL_VENV="$(pwd)/.venv"

uv pip install -r requirements.txt
uv pip install -e "AutoMol/automol[extended]" -e AutoMol/automol_resources
```

**Windows (native):** replace the activate line with:
```bat
.venv\Scripts\activate
```

**WSL with repo on a Windows-mounted drive (`/mnt/c/...`):** create the venv on the
Linux filesystem instead — `/mnt/c` is a 9p mount (~137× slower per-file, uv cannot
hardlink from its ext4 cache, and interrupted installs leave partially-written packages):
```bash
uv venv ~/.venvs/molagent --python 3.12
source ~/.venvs/molagent/bin/activate
export AUTOMOL_VENV="$HOME/.venvs/molagent"
```

`[extended]` brings the 3D feature stack (torch-geometric, yacs, networkx,
MDAnalysis, prolif, molfeat, lightning). `automol_resources` ships the trained
model files — without it the default `Bottleneck` encoder cannot load and
`retrieve_default_offline_generators()` fails with
`No module named 'automol.trained_models'`.

Three encoder keys are available as `feature_keys`:

| Key | Encoder | When to use |
|---|---|---|
| `Bottleneck` | ChEMBL 37 E-logD (default, v6_best) | Most endpoints |
| `Bottleneck_chembl37_base` | ChEMBL 37 E-base (no logD supervision) | **logD, logP, lipophilicity** — use this to avoid label leakage |
| `Bottleneck_chembl27` | Legacy ChEMBL 27 | Models trained before August 2026 |

The `tutorials` extra (PyTDC) is deliberately separate: PyTDC requires
`rdkit<2024.3.1` while the library needs `2024.3.5`, so install it only in a
throwaway environment, or with `uv pip install --no-deps "PyTDC>=0.4.0"`.

### Running MCP Server (standalone)
```bash
source ~/.venvs/molagent/bin/activate
uv run --active --no-sync mcp/server.py
```

### Claude Code MCP integration (`.mcp.json`)

The project ships a `.mcp.json` that registers the server via `mcp/launch.sh`.
The wrapper reads `$AUTOMOL_VENV` at startup — **export it before launching Claude Code**:

```bash
export AUTOMOL_VENV="$HOME/.venvs/molagent"   # or wherever your venv lives
claude --plugin-dir .
```

Add the export to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) so it persists
across sessions. The fallback when `AUTOMOL_VENV` is unset is `.venv` in the repo root.

### Running Web App
```bash
# Backend
cd app/backend && uvicorn main:app --reload --port 8080

# Frontend
cd app/frontend && npm install && npm run dev
```

### Testing
```bash
./run_tests.sh            # all tests
./run_tests.sh -k auth    # filter
```

The recorded baseline is **156 passed, 0 failed** (as of P4).
To reproduce it, the following four environment variables must be set to POSIX paths
(Windows-style `C:/…` strings are silently treated as relative paths on Linux, causing
9 test failures):

```bash
export AUTOMOL_ROOT="/mnt/c/Users/<you>/Projects/MOSA/molagent"
export AUTOMOL_VENV="$HOME/.venvs/molagent"
export MOLAGENT_PLUGIN_ROOT="/mnt/c/Users/<you>/Projects/MOSA/molagent"
export MOLAGENT_OUTPUT_ROOT="/mnt/c/Users/<you>/Projects/MOSA/molagent/MolagentFiles"
```

These cannot be committed (`.claude/settings.local.json` is gitignored globally),
so they must be set manually or added to a local shell profile before running tests.

### Claude Code Plugin
```bash
claude --plugin-dir .
```

## Architecture

```
MolAgent/
├── AutoMol/              # ML backend (git submodule)
├── mcp/                  # Unified MCP server (14 tools)
│   ├── server.py         # FastMCP app with all tool registrations
│   ├── _auth.py          # Token-based auth (optional)
│   ├── _config.py        # TrainingConfig / TrainingResult
│   ├── _data_registry.py # Dataset/structure registry
│   ├── _discovery.py     # Feature/estimator discovery
│   ├── _pipeline.py      # Training pipeline runner
│   ├── _3d_tools.py      # 3D structure upload + extraction
│   ├── _sanitize.py      # Output sanitization
│   └── tests/            # Test suite
├── app/                  # Web application
│   ├── backend/          # FastAPI
│   └── frontend/         # SvelteKit
├── skills/               # Claude Code plugin skills
│   ├── train-pipeline/   # 8-step guided training workflow
│   ├── predict/          # Inference
│   └── visualize/        # Dashboard generation
├── hooks/                # Plugin hooks
├── commands/             # Plugin commands
├── .claude-plugin/       # Plugin manifest
├── Data/                 # Sample datasets
└── MolagentFiles/        # Pipeline outputs
```

## MCP Server Tools (14 total)

1. list_options — discover features, estimators, configs
2. start_training_session — auto-detect dataset, create config session
3. answer_training_question — accept config overrides
4. train_and_visualize — run full pipeline (long-running)
5. list_models — query model registry
6. predict — inference on new molecules
7. merge_models — combine multi-property models
8. delete_model — remove from registry
9. download_model — base64 model binary
10. upload_dataset — base64 CSV upload
11. list_datasets — list data registry
12. delete_dataset — remove from registry
13. admin_manage — token management + purge
14. upload_3d_structure — upload SDF + PDB bundle for 3D features

## Environment Variables

| Variable | Purpose |
|----------|---------|
| AUTOMOL_ROOT | Project root directory |
| MOLAGENT_PLUGIN_ROOT | Plugin root (set by SessionStart hook) |
| MOLAGENT_OUTPUT_ROOT | Output directory (default: ./MolagentFiles) |
| AUTOMOL_VENV | Virtual environment path (default: .venv; documented install sets `~/.venvs/molagent`) |
| MOLAGENT_AUTH_REQUIRED | Enable auth (default: false) |
| MOLAGENT_MAX_UPLOAD_MB | Max upload size in MB (default: 100) |

## Known Workarounds

### Background task TTL (FastMCP 4 / Docket)

**Background task TTL:** FastMCP 4 delegates task lifetime to Docket (default `execution_ttl` = 15 min).
The web app backend refreshes it by polling `tasks/get` every 5 s. Do not increase `_POLL_INTERVAL`
in `job_store.py` beyond a few minutes or add long gaps in polling.

### FastMCP 4.x — MCP body size limit patch (`mcp/server.py`)

**Affects:** fastmcp 4.0.2 (verified). `create_streamable_http_app` does not
expose `max_request_body_size` to callers. MCP SDK v2 `StreamableHTTPSessionManager`
still defaults to 4 MiB; the patch overrides the last `__defaults__` entry.

`mcp/server.py` works around this at startup by patching two things before `fastmcp` is
imported (default args are baked at import time, so the patch must run first):

```python
import mcp.server.transport_security as _mcp_ts
import mcp.server.streamable_http_manager as _mcp_shm
_mcp_ts.DEFAULT_MAX_REQUEST_BODY_SIZE = _mcp_max_body_bytes
_mcp_shm.DEFAULT_MAX_REQUEST_BODY_SIZE = _mcp_max_body_bytes
_init = _mcp_shm.StreamableHTTPSessionManager.__init__
_init.__defaults__ = _init.__defaults__[:-1] + (_mcp_max_body_bytes,)
```

**To remove this patch:** verify that FastMCP's `create_streamable_http_app` accepts a
`max_request_body_size` parameter and passes it to `StreamableHTTPSessionManager`. If it
does, delete the patch block at the top of `mcp/server.py` and remove the `uvicorn_config`
kwarg from `mcp.run()`.

## Dependencies

Key packages: fastmcp[tasks]>=4, pandas, pydantic>=2.12, scikit-learn, torch, xgboost, lightgbm, rdkit, molfeat, prolif, fastapi, uvicorn

Requires: Python 3.12, uv, Node.js (for frontend)
