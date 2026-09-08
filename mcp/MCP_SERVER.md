# AutoMol MCP Server

An MCP server that exposes the AutoMol molecular property prediction pipeline as tools callable from any MCP client (Claude Code, Claude Desktop, smolagents, or any MCP-compatible app).

---

## Overview

The server provides fourteen tools:

| Tool | Type | Purpose |
|------|------|---------|
| `list_options` | Discovery | Discover available feature generators, estimators, configs installed on this server |
| `start_training_session` | Detection | Runs dataset auto-detection, creates a session, returns detected defaults + available options. Accepts `csv_file` or `dataset_id` |
| `answer_training_question` | Configuration | Accepts typed field overrides (Claude parses user language); returns final config when confirmed |
| `train_and_visualize` | Long-running | Takes a `TrainingConfig`, runs the full pipeline, returns model (base64) + interactive HTML dashboard. Uses `call_tool_task` from `fastmcp_tasks` (FastMCP 4 background execution via Docket) — returns a task ID immediately when called from a client that supports the tasks protocol (e.g. the web app). **Claude Code's built-in MCP client does not support tasks** — the connection drops with `-32000` when the tool is called, but training continues in the background and results land in `MolagentFiles/`. Reconnect and call `list_models` when done. |
| `list_models` | Discovery | Lists trained models visible to the authenticated user (admin sees all) |
| `predict` | Inference | Runs predictions on new SMILES using a trained model (by registry ID or direct path). `smiles_file` accepts a path or `dataset_id` |
| `merge_models` | Management | Merge multiple per-property models from the registry into a single multi-property file |
| `delete_model` | Management | Removes a model from the registry and deletes its files from disk |
| `download_model` | Management | Download a model's binary data (base64) by registry ID |
| `upload_dataset` | Data | Upload a CSV dataset (base64-encoded) to the per-user data registry |
| `list_datasets` | Data | List datasets owned by the caller (admin sees all) |
| `delete_dataset` | Data | Removes a dataset from the registry and deletes its file from disk |
| `admin_manage` | Admin | Token management + artifact cleanup (create/revoke/list users, purge_stale, purge_orphans) |
| `upload_3d_structure` | Data | Upload SDF + PDB bundle for structure-based 3D feature extraction |

### Dataset management flow

```
upload_dataset(filename="molecules.csv", file_content_b64="...")
    → { dataset_id: "ds_abc123", filename, columns, row_count, size_bytes }

list_datasets()
    → { datasets: [{id, filename, columns, row_count, uploaded_at, last_used, ...}] }

delete_dataset(dataset_id="ds_abc123")
    → { status: "deleted", dataset_id, filename }
```

### Typical conversation flow

```
start_training_session(dataset_id="ds_abc123")   # or csv_file="/path/to/data.csv"
    → { session_id, detected: {...}, options: {...}, question }

# Claude presents detected config to the user and asks if they want to change anything.
# User: "change computational budget to moderate"
# Claude parses this and calls:

answer_training_question(session_id, computational_load="moderate")
    → { session_id, question: "Anything else to change?" }

# User: "looks good"
answer_training_question(session_id, confirm=True)
    → { session_id, question: None, config: {...} }

train_and_visualize(config)
    → { model_id, dashboard_html, metrics, ... }
```

You can also skip the session tools and build the config dict manually for automation.

### Prediction flow

```
list_models()
    → { models: [{id, target_properties, task_type, metrics, ...}] }

predict(model_id="Caco2_wang-Y-20260403_1121", smiles_list=["CCO", "c1ccccc1"])
    → { status, output_csv, n_predictions, summary_stats, ... }
```

---

## File Structure

```
MolAgent/
├── .claude-plugin/plugin.json         # Plugin manifest (registers MCP server via mcpServers key)
├── mcp/
│   ├── server.py                      # FastMCP server entry point — defines tools + auth
│   ├── _auth.py                       # Token store management (validate, create, revoke)
│   ├── _data_registry.py             # Dataset registry (CRUD, last_used tracking, atomic I/O)
│   ├── _sanitize.py                   # Strips filesystem paths from remote responses
│   ├── _config.py                     # TrainingConfig + PredictConfig + TrainingResult Pydantic models
│   ├── _pipeline.py                   # Pipeline orchestrator — calls pipeline scripts as subprocesses
│   ├── _discovery.py                  # Discovery utilities (model/dataset resolution)
│   ├── admin_cli.py                   # CLI for remote user/admin management (PEP 723 script)
│   ├── test_mcp_server.py            # End-to-end integration tests against running remote server
│   ├── skills/
│   │   └── automol-pipeline/
│   │       └── SKILL.md               # MCP skill resource — orchestration guide for any LLM client
│   └── tests/
│       ├── test_pipeline.py           # Unit tests for _pipeline.py (uses synthetic CSV)
│       ├── test_server.py             # Integration tests via in-process FastMCP client
│       ├── test_data_registry.py      # Unit tests for _data_registry.py
│       └── test_dataset_tools.py      # Integration tests for dataset tools + purge_stale
```

---

## Transport & Authentication

### Stdio (default — Claude Code plugin)

```bash
uv run mcp/server.py
```

No auth. Used when spawned by the Claude Code plugin or the web app backend in local mode.

### Streamable HTTP (remote / multi-user)

```bash
# Run from the repo root
MOLAGENT_OUTPUT_ROOT=/absolute/path/to/output \
MOLAGENT_AUTH_REQUIRED=true \
uv run mcp/server.py --transport streamable-http --host 127.0.0.1 --port 8001
```

Requires a Bearer token on every request. The admin token is auto-generated on first run and printed to stderr. It's also stored in `${MOLAGENT_OUTPUT_ROOT}/auth_tokens.json`.

### Server CLI arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--transport` | `stdio` | `stdio` or `streamable-http` |
| `--host` | `127.0.0.1` | Bind address for HTTP mode |
| `--port` | `8001` | Port for HTTP mode |

### Token management (admin_cli.py)

```bash
# Create a user token
uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp --token <ADMIN_TOKEN> create-user alice

# List all users (shows owner_id, name, status, creation date, token prefix)
uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp --token <ADMIN_TOKEN> list-users

# Revoke a user by their unique owner_id (from list-users)
uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp --token <ADMIN_TOKEN> revoke <OWNER_ID>

# Rotate (re-issue) a user's token by owner_id — for a user who lost their token.
# The old token stops working; their models/datasets stay accessible.
uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp --token <ADMIN_TOKEN> rotate <OWNER_ID>

# Purge stale models/datasets (dry-run)
uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp --token <ADMIN_TOKEN> purge-stale --days 30

# Purge stale models/datasets (force delete)
uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp --token <ADMIN_TOKEN> purge-stale --days 30 --force

# Purge orphaned run folders from failed training (dry-run)
uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp --token <ADMIN_TOKEN> purge-orphans --days 7

# Purge orphaned run folders (force delete)
uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp --token <ADMIN_TOKEN> purge-orphans --days 7 --force
```

### Token types

| Token | Prefix | Permissions |
|-------|--------|-------------|
| Admin | `molagent_adm_` | All models, delete any, create/revoke users |
| User | `molagent_usr_` | Own models only (train, predict, delete) |

Each token has a unique internal `owner_id`. Two tokens with the same display name are separate identities with separate model access.

### Path sanitization (remote mode)

In HTTP mode, responses never include filesystem paths (`model_file`, `source_dataset`, `run_folder`, etc.). In stdio mode, all paths are returned normally.

### Prediction result delivery

The `predict` tool delivers results differently depending on the transport:

| Mode | `output_csv` | `csv_content` | How client gets data |
|------|-------------|---------------|---------------------|
| **Local (stdio)** | Absolute path on disk | Not included | Client reads file directly from the shared filesystem |
| **Remote (HTTP)** | Stripped | Full CSV text inline | CSV is embedded in the MCP JSON response |

**Remote mode**: The entire prediction CSV is serialized as a UTF-8 string in the `csv_content` field of the JSON response. This means predictions are fully self-contained — no shared filesystem required between client and server.

**Practical size limits**: Since the CSV travels inline in a single JSON response, very large prediction sets increase memory usage and transfer time. Recommendations:

| Compounds | Approximate payload | Guidance |
|-----------|-------------------|----------|
| < 1,000 | < 1 MB | No issues |
| 1,000 – 10,000 | 1–10 MB | Fine for most deployments |
| 10,000 – 50,000 | 10–50 MB | May be slow over high-latency connections |
| > 50,000 | > 50 MB | **Batch into multiple calls** (e.g. split input CSV into 10k chunks) |

For large-scale screening in remote mode, split the input SMILES file into batches and call `predict` multiple times. Each call is independent and can run in parallel from different clients.

In **local mode** (stdio), there is no practical limit — results are written to disk and only the path is returned.

---

## Setup

### Prerequisites

The `.venv` must exist and have AutoMol installed. This is handled automatically by the SessionStart hook when you load the plugin:

```
/plugin install MolAgentLight@molagent-marketplace
```

After installing, restart Claude Code once so the hook runs. It will:
1. Create `.venv` if missing
2. Install `automol` from the bundled copy
3. Install `fastmcp` (required by the MCP server)
4. Export `MOLAGENT_PLUGIN_ROOT` and `MOLAGENT_OUTPUT_ROOT` env vars

To verify the venv is ready:

```bash
uv run --active python -c "import automol, fastmcp; print('OK')"
```

### Manual dependency install (without the hook)

```bash
# Run from the repo root
uv venv .venv
uv pip install AutoMol/automol/ "fastmcp[tasks]" pandas pydantic
```

---

## Adding to Claude Code

### Via plugin (recommended)

The `mcpServers` key in `.claude-plugin/plugin.json` is auto-discovered when the plugin loads. No manual config needed — just install the plugin:

```
/plugin install MolAgentLight@molagent-marketplace
```

The server appears as `automol-mcp` in the MCP server list (`/mcp`).

### Manual add — stdio (without the plugin)

```bash
claude mcp add automol-mcp \
  --transport stdio \
  -- uv run --active --no-sync mcp/server.py
```

### Manual add — remote HTTP with auth

```bash
claude mcp add --transport http automol-mcp http://127.0.0.1:8001/mcp \
  --header "Authorization: Bearer <user token>"
```

Or with the `--url` flag form:

```bash
claude mcp add automol-mcp \
  --transport http \
  --url http://host:8001/mcp \
  --header "Authorization: Bearer molagent_usr_..."
```

Or in `.claude/settings.json` (or project `.claude/settings.local.json`):
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

Claude Code fully supports Bearer token auth for streamable-http MCP servers — the `--header` / `headers` value is forwarded on every request automatically.

---

## Usage

### Step 1 — Start a session

Call `start_training_session` with the path to your CSV. The server will:
1. Read the CSV column names
2. Run `detect_dataset.py` to auto-detect SMILES column, target columns, task type, and recommended settings
3. Create a fresh session (UUID) valid for 90 minutes
4. Return detected defaults plus the valid options for each field

### Step 2 — Answer questions

Claude presents the detected config to the user. For each change the user requests, Claude maps their natural language to typed parameters and calls `answer_training_question`. The server validates that any column names exist in the CSV.

Call with `confirm=True` (with or without overrides) to finalize and retrieve the `TrainingConfig` dict.

### Step 3 — Train and visualize

Pass the config dict to `train_and_visualize`. The server will:
1. Prepare data (standardize SMILES, apply transforms)
2. Split into train / validation / test
3. Train ensemble model(s)
4. Merge multi-property models into a single file (if >1 property)
5. Evaluate on the test set
6. Refit on train+valid+test (if `refit=True`)
7. Generate interactive HTML dashboard
8. Return `TrainingResult`

Progress is reported via MCP progress notifications at each step.

### Example (Python client)

```python
import asyncio, base64
from fastmcp import Client
from server import mcp  # or connect via HTTP/stdio

async def main():
    async with Client(mcp) as client:
        # Step 1 — start session
        start = await client.call_tool(
            "start_training_session",
            {"csv_file": "/data/my_compounds.csv"}
        )
        sid = start.data["session_id"]
        print(start.data["question"])  # present to user

        # Step 2 — confirm with no changes
        answer = await client.call_tool(
            "answer_training_question",
            {"session_id": sid, "confirm": True}
        )
        config = answer.data["config"]

        # Step 3 — train
        result = await client.call_tool(
            "train_and_visualize",
            {"config": config}
        )
        r = result.data

        # Download model separately if needed
        if r.get("model_id"):
            dl = await client.call_tool("download_model", {"model_id": r["model_id"]})
            with open("model.pt", "wb") as f:
                f.write(base64.b64decode(dl.data["model_b64"]))
        with open("dashboard.html", "w") as f:
            f.write(r["dashboard_html"])
        print("Metrics:", r["metrics"])

asyncio.run(main())
```

### Skipping sessions — build config manually

```python
config = {
    "csv_file": "/data/my_compounds.csv",
    "smiles_column": "SMILES",
    "properties": ["pIC50", "logD"],
    "task": "Regression",
    "computational_load": "cheap",
    "feature_keys": ["Bottleneck", "rdkit"],
    "use_log10": True,
    "split_strategy": "mixed",
    "refit": True
}

result = await client.call_tool("train_and_visualize", {"config": config})
```

---

## Data Registry

Uploaded datasets are tracked in `${MOLAGENT_OUTPUT_ROOT}/data_registry.json`. Each user's files are stored under `uploads/<owner_id>/`.

### upload_dataset Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `filename` | `str` | Yes | Original filename (e.g. `"molecules.csv"`) |
| `file_content_b64` | `str` | Yes | Base64-encoded CSV file content |

> **Claude Code CLI users:** Do NOT pass `file_content_b64` through the MCP tool call directly — base64 of files >30KB will be truncated by LLM I/O limits. Instead, use the in-process upload snippet that keeps base64 in Python memory.
>
> Retrieve the URL and token first:
> ```bash
> claude mcp get <server-name>   # prints URL and Authorization header
> ```
> Then upload:
>
> ```bash
> uv run --with fastmcp python -c "
> import asyncio,base64,json,sys,os
> async def main():
>     from fastmcp import Client
>     from fastmcp.client.transports import StreamableHttpTransport
>     t=StreamableHttpTransport(sys.argv[2],headers={'Authorization':f'Bearer {sys.argv[3]}'})
>     async with Client(t) as c:
>         b64=base64.b64encode(open(sys.argv[1],'rb').read()).decode()
>         r=await c.call_tool('upload_dataset',{'filename':os.path.basename(sys.argv[1]),'file_content_b64':b64})
>         print(r.content[0].text)
> asyncio.run(main())
> " "/path/to/data.csv" "http://host:8001/mcp" "molagent_usr_..."
> ```

### upload_3d_structure Parameters (remote mode)

For a remote server, use the in-process snippet below — same pattern as `upload_dataset`. It reads the SDF and all PDB files into memory and calls the tool in a single request.

```bash
uv run --with fastmcp python -c "
import asyncio,base64,json,sys,os,glob
async def main():
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport
    sdf_path,pdb_folder,target_name,url,token=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]
    t=StreamableHttpTransport(url,headers={'Authorization':f'Bearer {token}'})
    async with Client(t) as c:
        sdf_b64=base64.b64encode(open(sdf_path,'rb').read()).decode()
        pdbs=[{'filename':os.path.basename(p),'content_b64':base64.b64encode(open(p,'rb').read()).decode()} for p in glob.glob(os.path.join(pdb_folder,'*.pdb'))]
        r=await c.call_tool('upload_3d_structure',{'target_name':target_name,'sdf_content_b64':sdf_b64,'pdb_files':json.dumps(pdbs)})
        print(r.content[0].text)
asyncio.run(main())
" "/path/to/dockings.sdf" "/path/to/pdbs/" "ABL" "http://host:8001/mcp" "molagent_usr_..."
```

### list_datasets Response

Returns `{ datasets: [...] }` filtered by owner (admin sees all). Each entry:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Dataset ID (pass to `start_training_session(dataset_id=...)`) |
| `filename` | `str` | Original filename |
| `columns` | `list[str]` | CSV column names |
| `row_count` | `int` | Number of data rows |
| `size_bytes` | `int` | File size |
| `uploaded_at` | `str` | ISO timestamp |
| `last_used` | `str` | Last time this dataset was used for training/prediction |

### last_used Tracking

Both model and dataset registry entries have a `last_used` ISO timestamp, automatically updated when:
- **Models**: `predict(model_id=...)` is called
- **Datasets**: `start_training_session(dataset_id=...)` or `predict(smiles_file=dataset_id)` is called, or a direct `csv_file` matches a registered entry

### purge_stale (via admin_manage)

```python
admin_manage(action="purge_stale", max_age_days=30)          # dry-run
admin_manage(action="purge_stale", max_age_days=30, force=True)  # delete
```

Removes models and datasets where `last_used` (or `created_at` for legacy entries) is older than `max_age_days`. Dry-run returns what would be deleted; `force=True` actually deletes files and registry entries.

### purge_orphans (via admin_manage)

```python
admin_manage(action="purge_orphans", max_age_days=7)           # dry-run
admin_manage(action="purge_orphans", max_age_days=7, force=True)  # delete
```

Removes run folders in the output root that are not referenced by any model registry entry (e.g. leftover from failed training runs). Filters by folder modification time older than `max_age_days`. Skips `uploads/` and lock files.

---

## Session Lifecycle

- Sessions are in-memory, scoped to the MCP server process
- Each session lives for **90 minutes** of inactivity, then is evicted by a background timer
- A new `start_training_session` call always creates a fresh session — safe to call multiple times
- Using an unknown or expired `session_id` returns a clear error message
- Sessions are deleted immediately when `confirm=True` returns the final config

---

## TrainingConfig Parameters

All parameters are Pydantic-validated. Required fields are marked with *.

### Required

| Parameter | Type | Description |
|-----------|------|-------------|
| `csv_file` * | `str` | Absolute path to input CSV |
| `smiles_column` * | `str` | Column name containing SMILES strings |
| `properties` * | `list[str]` | Target property column names to predict |
| `task` * | `"Regression"` \| `"Classification"` \| `"RegressionClassification"` | Modelling task type. `RegressionClassification` = binary classification via regression estimators on 0/1 labels (predictions clipped to [0,1] as probabilities) |

### Model options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `computational_load` | `str` | `"cheap"` | `free` (0-2 min) / `cheap` (2-10 min) / `moderate` (10-360 min) / `expensive` (1-48 hr) |
| `feature_keys` | `list[str]` | `["Bottleneck"]` | Feature generators — see the encoder table below, plus `rdkit` and `fps_{nbits}_{radius}` |

#### Encoders

All three produce 250-dimensional embeddings and require only `onnxruntime`.

| Key | Checkpoint | Corpus | Use it for |
|---|---|---|---|
| `Bottleneck` | v6_best (E-logD), epoch 39 | full ChEMBL 37 | **Default.** Best predictive accuracy. |
| `Bottleneck_chembl37_base` | v5_full_best (E-base), epoch 39 | full ChEMBL 37 | logD / logP / lipophilicity (no logD supervision); embedding-driven work. |
| `Bottleneck_chembl27` | v1 (legacy) | ChEMBL 27 | Reproducing results from pre-2026 models. |

`Bottleneck_chembl37_logd` is accepted as an alias for `Bottleneck`.

**logD supervision.** `Bottleneck` is trained with a logD property head. For logD/logP/lipophilicity endpoints, use `Bottleneck_chembl37_base` to avoid label leakage.

| `sep` | `str` | `","` | CSV separator |

### Regression transforms

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_log10` | `bool` | `False` | Apply log10 to targets |
| `use_logit` | `bool` | `False` | Apply logit transform |

### Classification options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `categorical` | `bool` | `False` | Targets are already 0/1/2 integers |
| `nb_classes` | `list[int]` | `None` | Number of classes per property |
| `class_values` | `list` | `None` | Explicit cutoff thresholds |
| `class_quantiles` | `list` | `None` | Quantile-based cutoffs (0–1) |

### Validation / CV

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `split_strategy` | `str` | `"mixed"` | `mixed` / `stratified` / `leave_group_out` |
| `test_size` | `float` | `0.25` | Validation fraction (0.05–0.95) |
| `outer_folds` | `int` | `4` | Outer CV folds (2–20) |
| `random_state` | `int` | `42` | Seed for splits |
| `random_state_list` | `list[int]` | `[1,7,42,55,3]` | Seeds for base estimators |

### Parallelism

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_jobs_inner` | `int` | `2` | Inner CV loop jobs |
| `n_jobs_outer` | `int` | `None` | Outer loop jobs (`None` = serial) |

### Sample weights

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_sample_weight` | `bool` | `False` | Enable sample weighting during preparation |
| `sample_weight_selection` | `str` | `None` | Selection criterion, e.g. `"<1"`, `">5"` (applied to all properties) |
| `sample_weight_multiplier` | `float` | `None` | Weight multiplier for selected samples (1–1000) |

### Output

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_folder` | `str` | `None` | Run folder path (auto-generated if None) |
| `refit` | `bool` | `True` | Refit on train+valid after evaluation |
| `include_test_in_refit` | `bool` | `True` | Include test set in refit |

---

## TrainingResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | `str \| null` | Registry ID (use `download_model` to fetch the binary) |
| `model_filename` | `str` | Suggested filename |
| `dashboard_html` | `str` | Complete self-contained HTML dashboard |
| `metrics` | `dict` | Per-property metrics, e.g. `{"pIC50": {"R2": 0.85}}` |
| `properties` | `list[str]` | Property names modelled |
| `task` | `str` | Task type used |
| `model_path` | `str` | Absolute path to model file on disk |
| `dashboard_path` | `str` | Absolute path to dashboard HTML |
| `pipeline_state_path` | `str` | Absolute path to `pipeline_state.json` |
| `run_id` | `str` | Unique run identifier |
| `output_folder` | `str` | Absolute path to the run folder |

---

## list_models Response

| Field | Type | Description |
|-------|------|-------------|
| `models` | `list[dict]` | Array of model entries |
| `registry_path` | `str` | Absolute path to the registry JSON |

Each model entry contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique model ID — pass to `predict(model_id=...)` |
| `target_properties` | `list[str]` | Properties this model predicts |
| `task_type` | `str` | `"regression"` or `"classification"` |
| `metrics` | `dict` | Per-property metrics, e.g. `{"Y": {"R2": 0.70}}` |
| `feature_keys` | `list[str]` | Feature generators used |
| `source_dataset` | `str` | Original training CSV path |
| `is_refitted` | `bool` | Whether model includes refit on full data |
| `computational_load` | `str` | Budget used during training |
| `model_format` | `str` | `"merged"` (single file) or `"individual"` (per-property) |
| `created_at` | `str` | ISO timestamp |
| `blender_properties` | `list[str]` | Required blender inputs (empty if none) |

---

## merge_models Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_ids` | `list[str]` | Yes | — | Two or more model IDs from the registry to merge |
| `output_name` | `str` | No | auto | Custom name for the merged model |
| `verify_encoder` | `bool` | No | `true` | Assert encoders are compatible across models |

### merge_models Response

| Field | Type | Description |
|-------|------|-------------|
| `status` | `str` | `"merged"` |
| `model_id` | `str` | ID of the new merged model in the registry |
| `model_path` | `str` | Absolute path to merged `.pt` file (local mode only) |
| `properties` | `list[str]` | Combined properties from all source models |
| `task_type` | `str` | Task type of the merged model |
| `source_models` | `list[str]` | IDs of the source models |
| `metrics` | `dict` | Combined metrics from source models |

### Merge rules

- All source models must have the same task type (regression or classification)
- No overlapping properties (can't merge two models that both predict "solubility")
- Encoder compatibility is verified by default (disable with `verify_encoder=false` if models use different feature sets)
- The merged model is registered with `model_format: "merged"` and tracks its `source_models`

### merge_models Example

```python
# Train properties separately with different configs, then merge
result = await client.call_tool("merge_models", {
    "model_ids": ["Caco2_wang-Y-20260522", "ChEMBL-prop1-20260528"],
    "verify_encoder": True,
})
print(result.data["model_id"])     # merged model ID for predict
print(result.data["properties"])   # ["Y", "prop1"]
```

---

## predict Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str` | One of model_id / model_file | — | Model ID from registry |
| `model_file` | `str` | One of model_id / model_file | — | Direct path to `.pt` file |
| `smiles_list` | `list[str]` | One of smiles_list / smiles_file | — | Inline SMILES strings |
| `smiles_file` | `str` | One of smiles_list / smiles_file | — | Path to CSV with SMILES |
| `smiles_column` | `str` | No | `"smiles"` | Column name in CSV |
| `properties` | `list[str]` | No | all in model | Subset of properties to predict |
| `compute_sd` | `bool` | No | `true` | Compute uncertainty (SD) |
| `blender_properties` | `list[str]` | No | — | Column names in CSV with blender values |
| `blender_values` | `dict[str, float]` | No | — | Blender values for inline SMILES (`{"prop": 1.5}`) |

### predict Response

| Field | Type | Mode | Description |
|-------|------|------|-------------|
| `status` | `str` | Both | `"completed"` |
| `n_predictions` | `int` | Both | Number of molecules predicted |
| `columns` | `list[str]` | Both | Column names in output CSV |
| `summary_stats` | `dict` | Both | Per-column mean/std/min/max for predicted values |
| `output_csv` | `str` | Local only | Absolute path to predictions CSV on disk |
| `model_file` | `str` | Local only | Model file path that was used |
| `csv_content` | `str` | Remote only | Full CSV text (UTF-8) — the actual predictions data |
| `csv_filename` | `str` | Remote only | Suggested filename for download |

### predict Examples (Python client)

```python
# Inline SMILES with registry model
result = await client.call_tool("predict", {
    "model_id": "Caco2_wang-Y-20260403_1121",
    "smiles_list": ["CCO", "c1ccccc1", "CC(=O)OC1=CC=CC=C1C(=O)O"],
})
print(result.data["output_csv"])       # path to CSV
print(result.data["summary_stats"])    # {"predicted_Y": {"mean": -5.2, ...}}

# CSV input with direct model path
result = await client.call_tool("predict", {
    "model_file": "/path/to/merged_refitted_stackingregmodel.pt",
    "smiles_file": "/data/new_molecules.csv",
    "smiles_column": "SMILES",
})

# With blender properties (inline)
result = await client.call_tool("predict", {
    "model_id": "...",
    "smiles_list": ["CCO"],
    "blender_values": {"logP": 1.2, "MW": 46.07},
})

# With blender properties (CSV — columns contain the values)
result = await client.call_tool("predict", {
    "model_id": "...",
    "smiles_file": "/data/with_blenders.csv",
    "blender_properties": ["logP", "MW"],
})
```

---

## Skill Resource

The server exposes an MCP **skill resource** at `skill://automol-pipeline/SKILL.md`. Any MCP client that supports resources can read this to get a complete orchestration guide — when to call each tool, how to present results to the user, and how to handle blender properties.

Clients can also read `skill://automol-pipeline/_manifest` for a JSON listing of all files in the skill directory.

### Accessing skills via the FastMCP SDK

Use `fastmcp.utilities.skills` to discover and download skills from any running server:

```python
import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.utilities.skills import list_skills, get_skill_manifest, download_skill

async def main():
    t = StreamableHttpTransport("http://127.0.0.1:8001/mcp", headers={"Authorization": "Bearer <token>"})
    async with Client(t) as client:
        # List available skills
        skills = await list_skills(client)
        for skill in skills:
            print(f"{skill.name}: {skill.description}")
            # skill.uri → "skill://automol-pipeline/SKILL.md"

        # Inspect a skill's file manifest (name, size, sha256 hash)
        manifest = await get_skill_manifest(client, "automol-pipeline")
        print(manifest)
        # SkillManifest(name='automol-pipeline', files=[SkillFile(path='SKILL.md', size=18031, hash='sha256:...')])

        # Download all skill files to a local directory
        await download_skill(client, "automol-pipeline", target_dir="./skills/automol-pipeline")

asyncio.run(main())
```

---

## Testing

```bash
# Run from the repo root

# Fast smoke tests (< 30s)
uv run --active python -m pytest mcp/tests/test_server.py::test_tools_listed \
    mcp/tests/test_server.py::test_train_invalid_config_raises \
    mcp/tests/test_server.py::test_start_session_returns_detected_config \
    mcp/tests/test_server.py::test_unknown_session_id_returns_error -v

# Full server integration tests (pipeline runs, 2-10 min each)
uv run --active python -m pytest mcp/tests/test_server.py -v --timeout=600

# End-to-end test against a running remote server (all tools + auth)
uv run mcp/test_mcp_server.py \
    --url http://127.0.0.1:8001/mcp \
    --csv /path/to/data.csv \
    --token-file MolagentFiles/auth_tokens.json
```

The `test_mcp_server.py` script tests all 14 tools with valid and invalid inputs,
exercises all new parameter overrides (classification, refit, sample weights),
runs full training pipelines, prediction, model merging, dataset upload/list/delete,
and purge_stale. Requires a running server with `MOLAGENT_AUTH_REQUIRED=true`.

---

## Environment Variables

| Variable | Purpose | Set by |
|----------|---------|--------|
| `MOLAGENT_PLUGIN_ROOT` | Plugin root directory | SessionStart hook / plugin.json |
| `MOLAGENT_OUTPUT_ROOT` | Where run folders and registry live | SessionStart hook |
| `MOLAGENT_REGISTRY_PATH` | Full path override for `model_registry.json` | User |
| `MOLAGENT_DETERMINISTIC` | `true` for reproducible runs | User |
| `AUTOMOL_VENV` | Virtual environment path | User |
| `MOLAGENT_AUTH_REQUIRED` | Enable token auth (`true`/`1`/`yes`). Required for HTTP mode. | Deployer |
| `MOLAGENT_TOKEN_STORE_PATH` | Override token store location | Deployer |
| `MOLAGENT_CALLER_TOKEN` | Injected by web app backend for local-mode auth | Backend |
| `FASTMCP_DOCKET_REDELIVERY_TIMEOUT` | Docket redelivery timeout (seconds). Default 300s. Set to 86400 for long training runs. | Deployer |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'fastmcp'`**
Run: `uv pip install "fastmcp[tasks]"`

**`ModuleNotFoundError: No module named 'pandas'`**
Run: `uv pip install pandas`

**`ModuleNotFoundError: No module named 'automol'`**
Run: `uv pip install -e "AutoMol/automol[extended]" -e AutoMol/automol_resources`

**`Session 'xyz' not found or expired`**
Call `start_training_session` again to get a new session.

**`CSV file not found: ...`** / **`Model file not found: ...`**
Pass an absolute path. On Windows with Git Bash, use `C:/Users/...` not `/c/Users/...`.

**`No model specified. Provide model_id or model_file.`**
The predict tool was called without identifying which model to use. Call `list_models` first to find available IDs.

**`Registry not found`**
No models have been trained yet, or `MOLAGENT_OUTPUT_ROOT` points to the wrong directory.

**`Authentication required. Provide a valid Bearer token.`**
The MCP server has `MOLAGENT_AUTH_REQUIRED=true` but no valid token was provided. Set the auth token in the web app settings, or pass `--token` to `admin_cli.py`.

**`Access denied: model 'X' does not belong to you.`**
The token used doesn't own that model. Use `list_models` to see which models are accessible to your token, or use an admin token.

**`Authentication failed (401)`**
The token has been revoked or is invalid. Generate a new token via `admin_cli.py create-user`.

**`Direct csv_file path access is not allowed` from `start_training_session`**
The server is running in remote/HTTP mode and cannot access local filesystem paths. Upload the CSV first with `upload_dataset`, then pass the returned `dataset_id` to `start_training_session(dataset_id=...)`. See the train-pipeline SKILL.md Step 0 for the full in-process upload snippet.

**`train_and_visualize` disconnects with `-32000: Connection closed` from Claude Code**
Expected behavior. `train_and_visualize` uses FastMCP's `task=True` (SEP-1686 background tasks). Claude Code's built-in MCP client does not implement the tasks protocol, so it drops the connection when the server returns a `CreateTaskResult`. Training continues in the background — results land in `MolagentFiles/` as normal. Wait for training to finish (check `MolagentFiles/` for the run folder), reconnect, and call `list_models` to find the completed model. The `task=True` path works as intended from the web app, which uses a FastMCP client that supports SEP-1686.

**`train_and_visualize` aborts with "sent no response or progress for 300s"**
This is the Claude Code client-side tool execution idle timeout — different from server startup timeout. Training continues on the server despite the client abort. Call `list_models()` to check if the run completed. To prevent this, set `CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT` (ms) in `~/.claude/settings.json`:
```json
{ "env": { "CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT": "1800000" } }
```
Recommended: `1800000` (30 min). Alternatively, set a per-server `timeout` key in the server's MCP settings entry. Note: `MCP_TIMEOUT` (below) controls server startup only and has no effect here.

**`Task {id} not found` during long training runs**
Background tools use `call_tool_task(client, name, args)` from `fastmcp_tasks`. The web app backend refreshes the server-side task TTL by polling `tasks/get` every 5 s (see `job_store.py`). For the remote MCP server, set `FASTMCP_DOCKET_REDELIVERY_TIMEOUT=86400` to prevent Docket from assuming the task is dead during long training. Custom clients must keep polling — Docket's default `execution_ttl` is 15 min; if a client stops polling for longer than that, the task may be lost.

**Training killed with exit code 137 (OOM)**
The training process was killed by the OS out-of-memory killer. Reduce `--computational-load` to a lower level, reduce `--n-jobs-inner` to 1, or add swap space. Setting `MOLAGENT_DETERMINISTIC=true` forces single-threaded execution which halves peak memory.

**MCP server times out on startup (Windows)**
On first run, `uv` may take >30s resolving dependencies before the server is ready. Increase `MCP_TIMEOUT` (milliseconds, Claude env variable) in `~/.claude/settings.json` (global user settings). This only affects server startup — not tool execution timeouts (see above).
```json
{ "env": { "MCP_TIMEOUT": "120000" } }
```
