# AutoMol Plugin — Development Guide

## Architecture

This is a Claude Code plugin for the AutoMol molecular ML library. It has 3 skills (`train-pipeline`, `predict`, `visualize`) and a SessionStart hook.

### Key Directories

- `skills/train-pipeline/scripts/` — Python scripts executed via `uv run python`. Each is a Click CLI app with PEP 723 inline metadata.
- `skills/predict/scripts/` — Prediction script, separate from training.
- `skills/visualize/scripts/` — Dashboard generation script (`generate_dashboard.py`).
- `MolagentFiles/` — All pipeline outputs. Only `model_registry.json` is version-controlled.
- `hooks/setup-automol-env.sh` — Exports `AUTOMOL_ROOT` and `PLUGIN_ROOT` at session start.

### Execution Pattern

All Python scripts use: `uv run python <script> --option value`

Virtual env activation (`source ${AUTOMOL_VENV:-.venv}/bin/activate`) is needed when the system Python doesn't have AutoMol installed.

### Pipeline Version

Current: **2.0** — Run folder isolation, Task-based dispatch, unified `"outputs"` key.

### Run Folder Convention

Each pipeline run creates a self-contained folder: `MolagentFiles/{dataset}-{props}-{YYYYMMDD}_{HHMM}/`. All pipeline outputs (prepared CSV, models, state, model card) live inside the run folder. The global `MolagentFiles/model_registry.json` indexes all runs.

### Task-Based Dispatch

Steps 2, 3, 4, 5, and 7 are dispatched as `Task(subagent_type="general-purpose", model="sonnet")`. The orchestrator (SKILL.md) reads the step file content, combines it with the pipeline state, and sends it to a general-purpose subagent. Inline steps (0, 1, 6, 8) run in the main context for user interaction.


### Pipeline state key

The pipeline state uses `"outputs"` (not `"files"`) for file paths. Always read with fallback: `state.get("outputs", state.get("files", {}))`.

### Nested model files

Model files are stored per-property: `outputs.model_files.gamma1`, not as a flat string. Scripts must handle both `dict` and `str` formats.

### Refitted model filenames

Refitted models have `_refitted` in the filename (`gamma1_refitted_stackingregmodel.pt`). When extracting property names, strip this suffix or predictions will silently fail to produce output columns.

### SMILES column

The standardized column after preparation is `Stand_SMILES`. Raw input files may use `smiles`, `SMILES`, or others. The predict script auto-detects — don't force the training column name on new data.

### Classification: quantile-binning collapse

When preparing binary classification data, quantile-binning may collapse `Class_{prop}` to a single class if the original column is already binary (0/1). `train_classification.py` auto-detects this and falls back to the original column. When this happens, `labelnames` must be regenerated for the fallback property.

### Classification: JSON labelnames keys

`labelnames` loaded from JSON have string keys (`"0"`) but upstream scikit-learn indexes with `int`. Always convert: `{int(k): v for k, v in mapping.items()}`.

### blender_properties None vs empty list

Use `is not None` (not truthiness) when checking `blender_properties`. An empty list `[]` means "no blenders specified" but `None` means "auto-detect from prep info".

## Pipeline State Schema

```json
{
  "pipeline_version": "2.0",
  "run_id": "raw_data-gamma1_gamma2_gamma3-20260207_1430",
  "current_step": 8,
  "steps_completed": [0, 1, 2, 3, 4, 5, 6, 7, 8],
  "pipeline_complete": true,
  "config": {
    "data_file": "path/to/data.csv",
    "smiles_column": "smiles",
    "target_properties": ["prop1", "prop2"],
    "task_type": "regression|classification",
    "feature_keys": ["Bottleneck"],
    "computational_load": "free|cheap|moderate|expensive",
    "split_strategy": "mixed",
    "use_advanced": false,
    "output_folder": "MolagentFiles/raw_data-prop1_prop2-20260207_1430/"
  },
  "outputs": {
    "prepared_csv": "MolagentFiles/.../automol_prepared_X.csv",
    "split_csv": "MolagentFiles/.../automol_split_X.csv",
    "model_files": { "prop1": "MolagentFiles/.../prop1_stackingregmodel.pt" },
    "merged_model_file": "MolagentFiles/.../merged_stackingregmodel.pt",
    "refitted_model_files": { "prop1": "MolagentFiles/.../prop1_refitted_stackingregmodel.pt" },
    "merged_refitted_model_file": "MolagentFiles/.../merged_refitted_stackingregmodel.pt",
    "train_info": { "prop1": "MolagentFiles/.../prop1_train_info.json" },
    "evaluation_results": { "prop1": "MolagentFiles/.../prop1_evaluation_predictions.csv" }
  },
  "metrics": {
    "prop1": { "r2": 0.85, "mse": 0.12 }
  }
}
```

## File Naming Conventions

| Step | Pattern |
|------|---------|
| Prepare | `automol_prepared_{stem}.csv` + `_info.json` |
| Split | `automol_split_{stem}.csv` + `_info.json` |
| Train (reg) | `{property}_stackingregmodel.pt` + `{property}_train_info.json` |
| Train (clf) | `{property}_stackingclfmodel.pt` + `{property}_train_info.json` |
| Evaluate | `{property}_evaluation_predictions.csv` + `_evaluation_info.json` + `_evaluation_report.pdf` |
| Refit (reg) | `{property}_refitted_stackingregmodel.pt` |
| Refit (clf) | `{property}_refitted_stackingclfmodel.pt` |
| Merge | `merged_stackingregmodel.pt` / `merged_refitted_stackingregmodel.pt` |
| Predict | `{property}_predictions.csv` (individual) / `predictions.csv` (merged) |

## Merged Models

When training multiple properties, per-property `.pt` files each contain the full model with identical pretrained encoder (~10MB). The `merge_models.py` script uses upstream `merge_model()`/`delete_properties()` to combine them into a single file, eliminating encoder duplication.

- `merged_model_file` and `merged_refitted_model_file` are optional state fields — absent/null means individual files only
- The predict script auto-discovers properties from `model.models.keys()` — merged and individual files work transparently
- The registry uses `model_format: "merged"` (single string) or `"individual"` (list of paths)
- `merge_models.py` verifies encoder identity across all input files before merging

## Environment Variables

| Variable | Source | Purpose |
|----------|--------|---------|
| `AUTOMOL_ROOT` | SessionStart hook | Plugin root directory |
| `PLUGIN_ROOT` | SessionStart hook | Alias for AUTOMOL_ROOT |
| `AUTOMOL_VENV` | User env | Virtual environment name (default: `.venv`) |
| `CLAUDE_PLUGIN_ROOT` | Claude Code | Set when loaded as plugin |
| `CLAUDE_ENV_FILE` | Claude Code | Where hooks write env exports |

## Testing

**Regression**: `AutoMol/Tutorials/raw_data.csv` (157 molecules, 3 gamma targets) with `cheap` load.

**Classification**: `AutoMol/automol/automol/test/ChEMBL_SMILES.csv` (157 molecules, binary `prop5`) with `cheap` load. Note: `prop5` is already binary, so quantile-binning collapses — the training script falls back to the original column automatically.

**Predictions**: run after training using the same dataset or new SMILES. The predict skill auto-discovers models from the registry.

**Visualization**: run after evaluation (step 5 must be complete). The visualize skill auto-discovers runs with `metrics` and `outputs.evaluation_results` populated, then generates a self-contained `dashboard.html` in the run folder.
