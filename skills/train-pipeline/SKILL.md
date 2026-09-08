---
name: train-pipeline
description: Plan and execute an AutoMol training pipeline. Analyzes dataset, auto-configures a plan, presents config for user approval, then executes. Use when the user wants to train a complete model from raw data.
allowed-tools: Read, Glob, Bash, AskUserQuestion,
  mcp__plugin_MolAgentLight_automol-mcp__list_options,
  mcp__plugin_MolAgentLight_automol-mcp__start_training_session,
  mcp__plugin_MolAgentLight_automol-mcp__answer_training_question,
  mcp__plugin_MolAgentLight_automol-mcp__train_and_visualize,
  mcp__plugin_MolAgentLight_automol-mcp__list_models,
  mcp__plugin_MolAgentLight_automol-mcp__upload_dataset,
  mcp__plugin_MolAgentLight_automol-mcp__list_datasets,
  mcp__automol-mcp__list_options,
  mcp__automol-mcp__start_training_session,
  mcp__automol-mcp__answer_training_question,
  mcp__automol-mcp__train_and_visualize,
  mcp__automol-mcp__list_models,
  mcp__automol-mcp__upload_dataset,
  mcp__automol-mcp__list_datasets
---

# AutoMol — Training Pipeline

Orchestrate a full training pipeline using the `automol-mcp` MCP tools.

---

## Workflow

### Step 0: Upload Dataset (Remote Mode)

When the MCP server is remote (added via `claude mcp add` with an HTTP URL), there is no shared filesystem. CSV files must be uploaded before training.

**How to detect remote mode**: If `start_training_session(csv_file="...")` fails with any of these messages, the server is remote and requires file upload:
- `"Direct csv_file path access is not allowed"`
- `"No such file"`
- `"file not found"`

**Upload flow**:

1. Check if already uploaded: call `list_datasets()` and look for a matching filename.
2. If not uploaded, retrieve the MCP URL and token by running:
   ```bash
   claude mcp get <server-name>
   ```
   This prints the server URL and the `Authorization` header directly. Strip `"Bearer "` from the Authorization value to get the token. Example output:
   ```
   automol-mcp:
     Type: http
     URL: http://127.0.0.1:8001/mcp
     Headers:
       Authorization: Bearer molagent_usr_abc123...
   ```
3. Run this upload command via Bash, substituting the three values:

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
" "<CSV_PATH>" "<MCP_URL>" "<TOKEN>"
```

4. Parse the JSON output to get `dataset_id`.
5. Use `start_training_session(dataset_id="<dataset_id>")` in Step 1.

> **CRITICAL — never do this**: Do NOT run `base64 <file>`, `cat <file> | base64`, or `python -c "print(base64.b64encode(...))"` to display encoded file content in the terminal. The output WILL be truncated by shell output limits, producing corrupted data. The snippet above keeps base64 entirely in-process.

**3D structure upload (SDF + PDBs)**: Use the same pattern for `upload_3d_structure`. The tool accepts `sdf_content_b64`, `pdb_files` (JSON array of `{filename, content_b64}`), and `target_name`:

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
" "<SDF_PATH>" "<PDB_FOLDER>" "<TARGET_NAME>" "<MCP_URL>" "<TOKEN>"
```

Use the returned `structure_id` as the `dataset_id` in `start_training_session`.

---

### Step 1: Start Session

Call `start_training_session(csv_file="<absolute_path>")` (or `dataset_id` for uploaded datasets).

This auto-detects: SMILES column, target properties, task type, recommended features, computational load, and split strategy. It returns a `session_id` and the detected config.

### Step 2: Present Config to User

Format the detected config clearly:

```
Dataset: {csv_file} ({characteristics.valid_smiles} molecules)
SMILES column: {detected.smiles_column}
Task: {detected.task} — {options.task[detected.task]}
Targets: {detected.properties}
Features: {detected.feature_keys}
Split: {detected.split_strategy} — {options.split_strategy[detected.split_strategy]}
Load: {detected.computational_load} — {options.computational_load[detected.computational_load]}
Log10: {detected.use_log10}

Warnings:
  {from response.question field}

Target Summary:
  {for each target: column, task_type, unique_values, null_count, key stats}
```

**Critical domain terms** (present these correctly to users):
- **RegressionClassification** is binary classification via regression estimators on 0/1 labels (predictions clipped to [0,1] as probabilities). It is NOT "train both regression and classification."
- **blender_properties** are auxiliary numeric columns used as extra input features alongside molecular representations — they are NOT targets.
- **feature_keys** are molecular representation methods — not CSV column names. Encoder table:

  | Key | Notes |
  |---|---|
  | `Bottleneck` | Default. ChEMBL 37 E-logD (v6_best), trained with logD supervision. Best general accuracy. |
  | `Bottleneck_chembl37_base` | ChEMBL 37 E-base, no logD supervision. Use for logD/logP/lipophilicity targets. |
  | `Bottleneck_chembl27` | Legacy. Use only to reproduce results from old models. |
  | `rdkit` | ~210 RDKit 2D descriptors. |
  | `fps_2048_2` | Morgan fingerprints (2048 bits, radius 2). |

### Step 3: Apply Overrides or Confirm

If the user wants changes, parse their natural language into typed parameters and call `answer_training_question(session_id=..., field=value, ...)`.

Use `list_options(category=...)` to discover valid values for feature_keys, estimators, scorers, etc.

When ready: `answer_training_question(session_id=..., confirm=True)`. This returns the finalized config dict.

**Always pause for confirmation**: after presenting the detected config, ask the user whether to proceed or make changes — even if the original request clearly asked for training. The config summary surfaces important warnings (mixed target types, log10 flags, skewed targets, null rates) that the user should review before a potentially long run.

### Step 4: Execute Training

Call `train_and_visualize(config={...})` with the confirmed config.

This is long-running (minutes to hours depending on computational_load). It runs: prepare → split → train → evaluate → refit → dashboard generation → registry update.

### Step 5: Present Results

From the response, display:
- Metrics per property (R2/RMSE/MAE for regression; accuracy/AUC for classification)
- Model ID (for future predictions via the `predict` skill)
- Dashboard path (offer to open in browser)

```
Training Complete!
  Model ID: {model_id}
  Properties: {properties}

  Metrics:
    {prop1}: R2={r2}, RMSE={rmse}
    {prop2}: R2={r2}, RMSE={rmse}

  Dashboard: {dashboard_path}
  Next: use the predict skill to run inference on new molecules
```

---

## Error Handling

- If `start_training_session` fails with "CSV file not found": verify the path is absolute (Windows: `C:/Users/...`, not `/c/Users/...`).
- If `train_and_visualize` times out with "sent no response or progress for 300s": training may still be running on the server. Call `list_models()` to check if it completed. To prevent this, set `CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT=1800000` (30 min) in `~/.claude/settings.json` under `env`, or set a per-server `timeout` in the MCP server config. This is a **client-side tool execution timeout** — distinct from `MCP_TIMEOUT` (which only affects server startup).
- If training fails mid-pipeline: start a new session. The MCP pipeline is atomic — there is no partial resume.

---

## Presentation Templates

**Computational load choices** (when user asks what to pick):
- `free` — ~0–2 min. Single LightGBM, good for fast signal checking.
- `cheap` — ~2–10 min. Light ensemble, quick prototyping.
- `moderate` — ~10–360 min. Stacking ensemble, good balance.
- `expensive` — 1–48 hrs. Full hyperopt search, best performance.

---

## Encoder options for `feature_keys`

Three transformer encoders are available. Choose by endpoint:

| Key | Encoder | When to use |
|---|---|---|
| `Bottleneck` | E-logD, ChEMBL 37 (default) | Most endpoints; best overall accuracy |
| `Bottleneck_chembl37_base` | E-base, ChEMBL 37 | **logD, logP, lipophilicity** — logD supervision would contaminate these; also for embedding-driven similarity search |
| `Bottleneck_chembl27` | Legacy, ChEMBL 27 | Legacy models already trained on this encoder |

`Bottleneck_chembl37_logd` is accepted as an alias for `Bottleneck`.

---

## Fallback (MCP unavailable)

If the MCP server is not connected (check `/mcp`), the pipeline can be run directly via scripts. See `skills/train-pipeline/steps/` for per-step instructions and `skills/train-pipeline/scripts/` for the underlying Python scripts. Each script is invoked as `uv run $MOLAGENT_PLUGIN_ROOT/skills/train-pipeline/scripts/<script>.py`.
