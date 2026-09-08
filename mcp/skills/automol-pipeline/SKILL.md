---
description: Orchestrate the AutoMol molecular property prediction pipeline — training, prediction, model management — via MCP tools. Self-contained guide for remote and local usage.
---

# AutoMol Pipeline — MCP Skill

You have access to the AutoMol molecular property prediction pipeline via MCP tools. This skill describes how to orchestrate them.

## Available Tools

| Tool | Purpose |
|------|---------|
| `list_options` | Discover available feature generators, estimators, scorers installed on the server |
| `start_training_session` | Detect dataset defaults, start a config session. Accepts `csv_file` or `dataset_id` |
| `answer_training_question` | Apply user overrides or confirm the config |
| `train_and_visualize` | Run the full pipeline, returns model + dashboard |
| `list_models` | Discover trained models in the registry |
| `predict` | Run inference on new molecules. `smiles_file` accepts a path or `dataset_id` |
| `merge_models` | Merge per-property models into a single multi-property file |
| `delete_model` | Remove a model from the registry and disk |
| `download_model` | Download a model's binary (.pt) as base64 by registry ID |
| `upload_dataset` | Upload CSV (base64) to per-user data registry |
| `list_datasets` | List datasets owned by the caller |
| `delete_dataset` | Remove a dataset from the registry and disk |
| `admin_manage` | Token management + `purge_stale` cleanup (admin only) |
| `upload_3d_structure` | Upload SDF + PDB bundle for structure-based 3D feature extraction |

---

## Domain Terms

| Term | Meaning |
|------|---------|
| **Regression** | Predict continuous numeric values |
| **Classification** | Predict discrete classes using classification estimators |
| **RegressionClassification** | Binary classification via REGRESSION estimators on 0/1 labels — predictions clipped to [0,1] as probabilities. This is NOT "run both tasks simultaneously" |
| **blender_properties** | Auxiliary numeric columns used as extra input features alongside molecular representations — they are NOT prediction targets |
| **feature_keys** | Molecular representation methods (Bottleneck, rdkit, fps_2048_2, etc.) — NOT CSV column names |
| **computational_load** | Runtime budget: `free` ~2 min, `cheap` ~10 min, `moderate` ~1 hr, `expensive` ~24 hr |
| **dataset_id** | Registry ID (prefix `ds_`) referencing an uploaded CSV — used in place of file paths in remote mode |

---

## Transport Modes

### Local (stdio) — default for Claude Code plugin

The server is spawned automatically when the plugin loads. No authentication required. Filesystem paths are returned in all responses. This is the default when launched with `claude --plugin-dir .` from the repo root.

### Remote (streamable-http) — multi-user / web app

The server runs standalone at a URL. Every request requires a Bearer token. Filesystem paths are stripped from responses (security); prediction CSV content is returned inline instead of a path.

Start the server:
```bash
# Run from the repo root
MOLAGENT_OUTPUT_ROOT=/absolute/path/to/output \
MOLAGENT_AUTH_REQUIRED=true \
uv run --active --no-sync mcp/server.py --transport streamable-http --host 127.0.0.1 --port 8001
```

The admin token is printed to stderr on first run and saved to `${MOLAGENT_OUTPUT_ROOT}/auth_tokens.json`.

**Adding the remote server to Claude Code:**

```bash
claude mcp add automol-mcp \
  --transport http \
  --url http://host:8001/mcp \
  --header "Authorization: Bearer molagent_usr_..."
```

Or in `.mcp.json`:
```json
{
  "mcpServers": {
    "automol-mcp": {
      "type": "http",
      "url": "http://host:8001/mcp",
      "headers": { "Authorization": "Bearer molagent_usr_..." }
    }
  }
}
```

Claude Code fully supports Bearer token auth for streamable-http MCP servers — the `--header` / `headers` value is sent on every request automatically. No special handling is needed; the server validates the token on each call.

**Windows: MCP startup timeout**

On Windows, `uv` may take >30s on first start resolving dependencies. If the server times out before connecting, increase `MCP_TIMEOUT` (milliseconds, Claude env variable) in `~/.claude/settings.json`:
```json
{ "env": { "MCP_TIMEOUT": "120000" } }
```

---

## Authentication

| Token type | Prefix | Permissions |
|------------|--------|-------------|
| Admin | `molagent_adm_...` | All models, delete any, create/revoke users |
| User | `molagent_usr_...` | Own models only (train, predict, delete own) |

Manage users via `admin_manage` tool or `admin_cli.py`:
```bash
uv run mcp/admin_cli.py --url http://host:8001/mcp --token <ADMIN_TOKEN> create-user alice
uv run mcp/admin_cli.py --url http://host:8001/mcp --token <ADMIN_TOKEN> list-users
uv run mcp/admin_cli.py --url http://host:8001/mcp --token <ADMIN_TOKEN> revoke <OWNER_ID>
uv run mcp/admin_cli.py --url http://host:8001/mcp --token <ADMIN_TOKEN> rotate <OWNER_ID>
```

---

## Workflow: Upload Dataset (Remote Mode)

When connected to a remote server (streamable-http), filesystem paths are inaccessible. Upload CSV files first, then reference them by `dataset_id`.

### 1. Check existing datasets

```
list_datasets()
```

If the file was previously uploaded, reuse its `dataset_id`.

### 2. Find MCP URL and Token (Claude Code)

To run the upload snippet, you need the server URL and Bearer token. For Claude Code users:

1. Read `~/.claude.json` (Windows: `C:/Users/<username>/.claude.json`).
2. Find the key under `projects` matching the current project path.
3. Read `.mcpServers.<server-name>.url` and `.mcpServers.<server-name>.headers.Authorization`.
4. Strip the `"Bearer "` prefix from the Authorization value to get the raw token.

Example structure:
```json
{
  "projects": {
    "/home/you/project": {
      "mcpServers": {
        "automol-mcp": {
          "url": "http://host:8001/mcp",
          "headers": { "Authorization": "Bearer molagent_usr_abc123..." }
        }
      }
    }
  }
}
```

If `~/.claude.json` does not contain the server entry, also check `.claude/settings.local.json` and `.claude/settings.json` in the project root.

> **Do not grep for the token** — read the file directly with the Read tool to avoid partial matches or exposing the token in search output.

### 3. Upload via CLI

The base64-encoded file content must NOT pass through LLM I/O (it will be truncated). Use this in-process snippet:

```bash
uv run --with fastmcp python -c "
import asyncio,base64,json,sys,os
async def main():
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport
    t=StreamableHttpTransport(sys.argv[2],headers={'Authorization':f'Bearer {sys.argv[3]}'})
    async with Client(t) as c:
        b64=base64.b64encode(open(sys.argv[1],'rb').read()).decode()
        r=await c.call_tool('upload_dataset',{'filename':os.path.basename(sys.argv[1]),'file_content_b64':b64})
        print(r.content[0].text)
asyncio.run(main())
" "/path/to/data.csv" "http://host:8001/mcp" "molagent_usr_..."
```

Returns JSON with `dataset_id`, `filename`, `columns`, `row_count`.

> **Why not call `upload_dataset` directly?** The MCP tool parameter `file_content_b64` can exceed Claude's tool-call output/input limits for files >30KB. The snippet above keeps base64 in-process (Python memory only) and only the small result JSON appears in the terminal.

> **CRITICAL — never do this**: Do NOT run `base64 <file>`, `cat <file> | base64`, or `python -c "print(base64.b64encode(...))"` to display encoded file content in the terminal. The output WILL be truncated by shell output limits, producing corrupted data.

### 4. Use dataset_id

Pass the returned ID to training or prediction:
```
start_training_session(dataset_id="ds_abc123...")
predict(model_id="...", smiles_file="ds_abc123...")
```

---

## Workflow: Training a New Model

### 1. Discover options (optional)

In remote mode, servers may have different libraries installed. Call `list_options` before configuring to see what's available:

```
list_options(category="feature_generators")
list_options(category="base_estimators", task="Classification")
list_options(category="scorers", task="Regression")
```

Categories: `feature_generators`, `base_estimators`, `blender_estimators`, `dim_reduction`, `model_configs`, `search_types`, `scorers`.

### 2. Start a session

```
start_training_session(csv_file="/absolute/path/to/data.csv")
```

Returns:
- `session_id` — pass to subsequent calls
- `detected` — auto-detected config (smiles_column, properties, task, etc.)
- `options` — valid choices for each field
- `question` — message to present to the user

### 3. Present config to the user

Format the detected config clearly:

```
Dataset: {csv_file or dataset_id} ({characteristics.valid_smiles} molecules)
SMILES column: {detected.smiles_column}
Task: {detected.task}
Targets: {detected.properties}
Features: {detected.feature_keys}
Split: {detected.split_strategy}
Load: {detected.computational_load}
Log10: {detected.use_log10}

Warnings:
  {from response.question field}
```

**Domain notes for the user:**
- If task is **RegressionClassification**: explain it uses regression estimators on 0/1 labels (not "both tasks").
- If **blender_properties** are detected: explain these are auxiliary input features, not targets.
- If **feature_keys** includes unfamiliar names: explain these are molecular representations, not column names.

**SMILES column can be wrong** — datasets often have ID columns that look like SMILES. Always verify with the user.

**Always pause for user confirmation** before proceeding to training — even if the user's original request clearly asked for it. The config summary surfaces critical warnings (mixed target types, log10 flags, skewed targets, null rates) that the user should review before a long-running pipeline.

> **Windows paths:** Pass `C:/Users/...` style paths, not `/c/Users/...` (Git Bash style). The MCP server receives the path as-is and will fail if it can't find the file.

### 4. Apply overrides

When the user wants changes, parse their natural language into typed parameters:

```
answer_training_question(
    session_id="...",
    smiles_column="SMILES",         # if user corrects it
    computational_load="moderate",  # if user wants more compute
    feature_keys=["Bottleneck", "fps_2048_2"],  # add fingerprints
)
```

You can call this multiple times. Each call returns updated state and asks if there are more changes.

Use `list_options(category=...)` to discover valid values for feature_keys, estimators, scorers, etc.

### 5. Confirm

```
answer_training_question(session_id="...", confirm=True)
```

Returns `config` — a complete TrainingConfig dict ready for training. Sessions expire after 90 minutes of inactivity.

### 6. Train

```
train_and_visualize(config={...})
```

Runs the full pipeline: prepare → split → train → evaluate → refit → dashboard. This is a **long-running tool** (minutes to hours depending on `computational_load`). Progress is reported via MCP progress notifications at each step.

Returns:
- `model_id` — registry ID (use `download_model` to fetch the binary)
- `dashboard_html` — self-contained interactive Plotly.js dashboard
- `metrics` — per-property evaluation metrics (R2/RMSE/MAE for regression; Accuracy for classification)
- `model_path`, `dashboard_path`, `run_id`, `output_folder` — filesystem details (stdio mode only)

**Present the metrics to the user.** If they ask to see the dashboard, save `dashboard_html` to a `.html` file and open it in a browser. If they want the model locally, call `download_model(model_id=...)`.

---

## Workflow: Prediction

### 1. Find a model

```
list_models()
```

Returns all models accessible to the current token, with IDs, properties, task types, and metrics.

- **0 models**: Tell the user to train first. Stop.
- **1 model**: Auto-select. Show ID, properties, task type, metrics.
- **N models**: Present choices (most recent first) with properties and metrics.

### 2. Run predictions

With a model ID from the registry:
```
predict(
    model_id="Caco2_wang-Y-20260403_1121",
    smiles_list=["CCO", "c1ccccc1", "CC(=O)OC1=CC=CC=C1C(=O)O"]
)
```

Or with a CSV file:
```
predict(
    model_id="Caco2_wang-Y-20260403_1121",
    smiles_file="/path/to/new_molecules.csv",
    smiles_column="SMILES"
)
```

Or with a direct model path (no registry needed):
```
predict(
    model_file="/path/to/model.pt",
    smiles_list=["CCO"]
)
```

**Remote mode:** If the server is remote (HTTP MCP), local CSV paths won't work. Upload the file first (see Upload Dataset workflow) and pass the returned `dataset_id` as `smiles_file`.

Returns:
- `output_csv` — path to the predictions CSV (stdio) or inline content (remote)
- `n_predictions` — number of molecules predicted
- `summary_stats` — mean/std/min/max for each predicted property
- `columns` — column names in the output CSV

### Blender properties

Some models require additional input properties ("blender properties") used during training. Check `blender_properties` in the `list_models` response.

For CSV input:
```
predict(model_id="...", smiles_file="data.csv", blender_properties=["logP", "MW"])
```

For inline SMILES:
```
predict(model_id="...", smiles_list=["CCO"], blender_values={"logP": 1.2, "MW": 46.07})
```

---

## Workflow: Delete a Model

`delete_model` is **irreversible** — it removes the model from the registry and deletes the `.pt` files from disk. Always confirm with the user before calling.

```
delete_model(model_id="Caco2_wang-Y-20260403_1121")
```

Admin tokens can delete any model. User tokens can only delete their own.

---

## Workflow: Manage Users (Admin Only)

```
admin_manage(action="create_token", user_id="alice")
    → { token: "molagent_usr_..." }

admin_manage(action="list_users")
    → { users: [{owner_id, user_id, created_at, revoked, token_prefix}, ...] }

admin_manage(action="revoke_user", owner_id="<owner_id from list_users>")
    → { status: "revoked", owner_id: "..." }

admin_manage(action="rotate_token", owner_id="<owner_id from list_users>")
    → { status: "rotated", owner_id: "...", user_id: "...", token: "molagent_usr_..." }
    # Re-issues a token for a user who lost theirs; owner_id preserved so their
    # models/datasets remain accessible. Old token stops working.
```

---

## Workflow: Skip Sessions (Automation)

For automated or scripted use, skip the session tools and build config manually:

```
train_and_visualize(config={
    "csv_file": "/data/compounds.csv",
    "smiles_column": "SMILES",
    "properties": ["pIC50"],
    "task": "Regression",
    "computational_load": "cheap",
    "feature_keys": ["Bottleneck"],
    "use_log10": true,
    "split_strategy": "mixed",
    "refit": true
})
```

---

## Error Handling

**"CSV file not found"** from `start_training_session`:
- Verify the path is absolute. On Windows use `C:/Users/...` (forward slashes), NOT Git Bash style `/c/Users/...`. The MCP server passes paths directly to Python's `Path()`.

**"sent no response or progress for 300s"** from `train_and_visualize`:
- Training may still be running on the server. Call `list_models()` to check if it completed.
- To prevent: set `CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT` (ms) in `~/.claude/settings.json` under `"env"`:
  ```json
  { "env": { "CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT": "1800000" } }
  ```
  1800000 = 30 minutes. For `expensive` runs, use `172800000` (48 hr). This is a **client-side tool execution timeout** — distinct from `MCP_TIMEOUT` (server startup only).

**Training fails mid-pipeline**:
- Start a new session. The pipeline is atomic — there is no partial resume. Failed runs do not corrupt the registry.

**"Direct csv_file path access is not allowed"**:
- The server is in remote (auth-required) mode. Upload the CSV first (see Upload Dataset workflow), then use `dataset_id`.

---

## Key Behaviors

- **Always present detected config to the user** — don't assume auto-detection is correct.
- **SMILES column detection can be wrong** — datasets often have ID columns that look like SMILES. Verify with the user.
- **`computational_load` controls runtime**: `free` (~2 min), `cheap` (~10 min), `moderate` (~1 hr), `expensive` (~24 hr).
- **`train_and_visualize` is long-running** — it uses background task execution; you'll receive progress updates while it runs. Do not time out or retry while it is in progress.
- **Predictions always save to disk** (stdio mode) — the tool returns a CSV path, not inline data. Report the path and summary stats.
- **`delete_model` is permanent** — always confirm with the user before calling.
- **`model_id` vs `model_file`**: prefer `model_id` (from registry) for traceability. Use `model_file` only for models not yet registered.
- **Dashboard is HTML** — it's a self-contained Plotly.js page. Save to a `.html` file for the user to open in a browser.
- **Use `list_options` in remote mode** before training to discover what feature generators and estimators are installed on that server.
- **Store the auth token in `claude mcp add --header`**, not in the prompt or conversation — it's a secret.
- **`download_model` returns base64** — save the decoded bytes to a `.pt` file locally. Useful for backing up models from a remote server.
- **Confirmation before training is mandatory** — always present the detected config and wait for explicit user approval before calling `train_and_visualize`, even if the user said "just train it."
- **Remote mode: prefer `dataset_id` over file paths** — once uploaded, reference data by `dataset_id` in all subsequent calls. This eliminates filesystem-access errors.
