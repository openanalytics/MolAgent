---
name: predict
description: Make predictions using a trained AutoMol model. Auto-discovers models from the registry. Use when the user wants to run inference on new molecules.
allowed-tools: Read, Glob, Bash, AskUserQuestion,
  mcp__plugin_MolAgentLight_automol-mcp__list_models,
  mcp__plugin_MolAgentLight_automol-mcp__predict,
  mcp__plugin_MolAgentLight_automol-mcp__upload_dataset,
  mcp__plugin_MolAgentLight_automol-mcp__list_datasets,
  mcp__automol-mcp__list_models,
  mcp__automol-mcp__predict,
  mcp__automol-mcp__upload_dataset,
  mcp__automol-mcp__list_datasets
---

# AutoMol — Predict

Run inference on new molecules using a trained model via the `automol-mcp` MCP tools.

---

## Workflow

### Step 1: Discover Models

Call `list_models()`. Present available models to the user.

- **0 models**: tell the user to train first (via the `train-pipeline` skill). STOP.
- **1 model**: auto-select. Show summary (id, properties, task type, metrics).
- **N models**: present choices via AskUserQuestion (most recent first).

### Step 2: Get Input

If the user provided SMILES or a CSV path in their message, use it directly.

Otherwise ask:
- CSV file with a SMILES column → pass as `smiles_file`
- Individual SMILES strings → pass as `smiles_list`

**Remote mode (no shared filesystem):** If the server is remote (HTTP MCP — check `.claude/settings.json` or `.claude/settings.local.json`), a local CSV path won't work. Upload the file first, then pass the returned `dataset_id` as `smiles_file` (IDs starting with `ds_` are auto-resolved by the predict tool).

Upload flow:
1. Call `list_datasets()` to check if already uploaded.
2. Read `~/.claude.json` and look under `projects.<current-project-path>.mcpServers.automol-mcp` for the `url` and `headers.Authorization` (strip `Bearer ` prefix for token).
3. Run:

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

4. Parse JSON output → use `dataset_id` as `smiles_file`.

> **CRITICAL**: Never run `base64 <file>` or `python -c "print(b64encode(...))"` to display encoded content — it WILL be truncated. The snippet above keeps base64 in-process.

### Step 3: Check Blender Requirements

If the selected model has `blender_properties`, warn the user that those columns must be present in their input CSV. For inline SMILES, blender values must be provided separately.

### Step 4: Run Prediction

Call `predict(model_id="...", smiles_file="...")` or `predict(model_id="...", smiles_list=[...])`.

### Step 5: Present Results

Display:
- Number of predictions
- Summary statistics (mean/std/min/max per property)
- First 5–10 rows as a table
- Output CSV path

```
Predictions complete!
  Model: {model_id}
  Input: {n} molecules
  Output: {output_path}

  Summary:
    {prop1}: mean={mean}, std={std}, range=[{min}, {max}]

  Sample:
    {table of first 5 rows}
```

---

## Notes

- Prefer `model_id` over `model_file` for traceability.
- The predict tool auto-detects SMILES columns — don't force column names from training.
- Merged models predict all properties in one call; individual models predict one property each.

---

## Fallback (MCP unavailable)

If the MCP server is not connected, use the script directly:

```bash
uv run $MOLAGENT_PLUGIN_ROOT/skills/predict/scripts/predict.py \
    --model-file {path} \
    --smiles-file {input.csv} \
    --output-folder "${MOLAGENT_OUTPUT_ROOT:-MolagentFiles}/" \
    --verbose
```
