# AutoMol Web App

A browser-based interface for AutoMol molecular property prediction. The app is a thin HTTP bridge to the MCP server — all pipeline logic runs in the MCP tools.

## Architecture

```
Browser (SvelteKit :5173)
  │ HTTP proxy /api → :8000
  ▼
FastAPI Backend (:8000)         ← Thin bridge, no ML logic
  │ MCP protocol (stdio or streamable-http)
  ▼
MCP Server (automol-mcp)        ← All pipeline logic lives here
  │ subprocess calls
  ▼
AutoMol Scripts (uv run)
```

The backend has **no pipeline logic**. It translates HTTP requests into MCP `call_tool` calls, manages async job state for the frontend's polling loop, and handles file uploads. The MCP server can be local (spawned via stdio) or remote (connected via URL).

## Installation (without Claude Code marketplace)

If you're running the MCP server or web app standalone (not via the Claude Code plugin/hook):

```bash
# Run from the repo root

# Create venv and install AutoMol + dependencies
uv venv .venv --python 3.12
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install -e "AutoMol/automol[extended]" -e AutoMol/automol_resources
uv pip install "fastmcp[tasks]" pandas pydantic

# Install frontend dependencies
cd app/frontend && npm install && cd ../..
```

> **WSL with repo on a Windows drive (`/mnt/c/...`):** use `~/.venvs/molagent` instead of `.venv` to avoid the slow 9p mount.

## Quick Start

A convenience script starts both backend and frontend:

```bash
# Run from the repo root/app
./start.sh
```

Or start them manually:

```bash
# Run from the repo root/app

# Terminal 1: Backend (port 8000)
uv run --with fastapi --with uvicorn --with python-multipart --with pydantic-settings --with "fastmcp[tasks]" --with pandas \
  uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Frontend (port 5173)
cd frontend && npm run dev
```

Open **http://localhost:5173** in your browser.

## MCP Connection Modes

Configure in the **Settings** tab:

| Mode | How it works |
|------|-------------|
| **Local** (default) | Backend spawns `mcp/server.py` via stdio on each tool call. Requires the `.venv` with AutoMol installed. No auth needed. |
| **Remote** | Backend connects to a URL via streamable-http. Set the auth token in settings — the backend uses it for all MCP calls. |

## Running the MCP Server Standalone (Remote Mode)

To run the MCP server as a standalone HTTP service with authentication:

```bash
# Run from the repo root

MOLAGENT_AUTH_REQUIRED=true uv run mcp/server.py \
  --transport streamable-http --host 127.0.0.1 --port 8001
```

On first run the admin token is printed to stderr and saved to `${MOLAGENT_OUTPUT_ROOT}/auth_tokens.json`.

Then configure the web app: Settings → Remote → URL: `http://<host>:8001/mcp` → Auth Token: `<admin or user token>`.

Server arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--transport` | `stdio` | `stdio` or `streamable-http` |
| `--host` | `127.0.0.1` | Bind address for HTTP mode |
| `--port` | `8001` | Port for HTTP mode |

## Authentication

Authentication lives on the **MCP server**, not the web app. The web app itself is open — anyone with network access to the backend can use it. The backend authenticates to the MCP server using the token configured in settings.

### Token Types

| Token | Prefix | Can do |
|-------|--------|--------|
| Admin | `molagent_adm_` | See all models, delete any model, create/revoke user tokens |
| User | `molagent_usr_` | See only own models, train, predict, delete own models |

Each token has a unique internal `owner_id`. Two tokens with the same display name (e.g. "TestDude") are separate identities — models trained with one token are invisible to the other.

### Managing Users (Admin CLI)

Use `admin_cli.py` to manage tokens against a running MCP server:

```bash
# Run from the repo root

# Create a user
uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp --token <ADMIN_TOKEN> create-user alice

# List all users (shows full tokens for copy-paste)
uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp --token <ADMIN_TOKEN> list-users

# Revoke a token (copy full token from list-users output)
uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp --token <ADMIN_TOKEN> revoke molagent_usr_...
```

### Managing Users (via Web App Backend)

If the backend is configured with an admin token in settings, you can also use REST:

```bash
# Create user
curl -X POST http://127.0.0.1:8000/api/admin/manage \
  -H "Content-Type: application/json" \
  -d '{"action":"create_token","user_id":"alice"}'

# List users
curl -X POST http://127.0.0.1:8000/api/admin/manage \
  -H "Content-Type: application/json" \
  -d '{"action":"list_users"}'

# Revoke a user by their unique owner_id (from list_users)
curl -X POST http://127.0.0.1:8000/api/admin/manage \
  -H "Content-Type: application/json" \
  -d '{"action":"revoke_user","owner_id":"<owner_id>"}'

# Rotate (re-issue) a user's token by owner_id — for a user who lost their token.
# Old token stops working; their models/datasets stay accessible.
curl -X POST http://127.0.0.1:8000/api/admin/manage \
  -H "Content-Type: application/json" \
  -d '{"action":"rotate_token","owner_id":"<owner_id>"}'
```

### Token Store

Tokens are stored in `${MOLAGENT_OUTPUT_ROOT}/auth_tokens.json`. Override location with `MOLAGENT_TOKEN_STORE_PATH`.

### Multi-User Setup

1. Start the MCP server with `MOLAGENT_AUTH_REQUIRED=true`
2. Note the admin token from the first-run output (or `cat MolagentFiles/auth_tokens.json`)
3. Create user tokens via `admin_cli.py` or `/api/admin/manage`
4. Give each web app instance a different user token in settings
5. Each user only sees models they trained; admin sees all
6. Revoked tokens get a 403 error on any MCP operation

## API Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/health` | GET | Health check |
| `/api/settings` | GET/PUT | MCP connection config (mode, url, auth_token) |
| `/api/train/upload` | POST | Upload CSV file |
| `/api/train/detect` | POST | Call MCP `start_training_session` |
| `/api/train/configure` | POST | Call MCP `answer_training_question` |
| `/api/train/run` | POST | Call MCP `train_and_visualize` (background job) |
| `/api/train/runs` | GET | List completed runs from disk |
| `/api/train/runs/{id}` | GET | Get pipeline state for a run |
| `/api/models` | GET | Call MCP `list_models` |
| `/api/registry` | GET | List models (via MCP, filtered by token) |
| `/api/registry/{id}` | GET | Get a single model entry |
| `/api/registry/{id}` | DELETE | Call MCP `delete_model` (removes registry + files) |
| `/api/admin/manage` | POST | Call MCP `admin_manage` (create/revoke tokens) |
| `/api/predict/run` | POST | Call MCP `predict` (background job) |
| `/api/predict/{job_id}/result` | GET | Get prediction output path |
| `/api/predict/{job_id}/download` | GET | Download predictions CSV |
| `/api/visualize/{run_id}/html` | GET | Serve dashboard HTML from disk |
| `/api/visualize/job/{job_id}/html` | GET | Serve dashboard from job result |
| `/api/jobs/{id}/status` | GET | Poll job progress/status |
| `/api/jobs/{id}/logs` | GET | Get job log lines |

## Progress Tracking

The MCP server reports progress at each pipeline step via the MCP progress protocol. The backend receives these notifications through `session.call_tool(progress_callback=...)` and updates the in-memory `Job` object. The frontend polls `/api/jobs/{id}/status` every 2 seconds and reads `progress`, `progress_total`, and `progress_label` to animate the step list.

## Frontend Structure

```
src/
├── routes/
│   ├── train/+page.svelte      Upload → Detect → Configure → Run
│   ├── predict/+page.svelte    Model picker → SMILES input → Results
│   ├── visualize/+page.svelte  Run picker → Dashboard iframe
│   └── settings/+page.svelte   MCP connection config
├── lib/
│   ├── api/client.ts           HTTP client + TypeScript types
│   ├── stores/
│   │   ├── pipeline.svelte.ts  Training state (config, detect result, job)
│   │   └── registry.svelte.ts  Model registry state
│   └── components/
│       ├── train/              DetectPanel, ConfigPanel, PipelineSteps
│       ├── predict/            ModelPicker, SmilesInput, PredictResults
│       ├── visualize/          RunPicker, DashboardFrame
│       ├── shared/             FileUpload, JobSpinner
│       └── layout/             TabBar, SettingsPanel
```

## Backend Structure

```
backend/
├── main.py                 FastAPI app, router registration
├── config.py               Settings (plugin_root, output_root, venv paths)
├── mcp_client.py           MCP client abstraction (local stdio / remote HTTP)
├── job_store.py            In-memory job tracking with progress fields
├── routes/
│   ├── train.py            Upload, detect, configure, run-pipeline
│   ├── predict.py          Predict via MCP
│   ├── visualize.py        Serve dashboards
│   ├── jobs.py             Job status/logs polling
│   ├── settings.py         MCP connection settings
│   ├── registry.py         Model registry (list, get, delete via MCP)
│   └── admin.py            Admin token management (proxies to MCP)
└── schemas/
    └── job.py              Response models
```
