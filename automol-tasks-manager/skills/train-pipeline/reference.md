# Training Pipeline Reference

## Pipeline Version

Current: **2.0** — Run folder isolation, Task-based dispatch, unified `"outputs"` key.

## Run Folder Convention

Each pipeline run creates a self-contained folder under `MolagentFiles/`:

```
MolagentFiles/
  model_registry.json                             ← global (indexes all runs)
  raw_data-gamma1_gamma2_gamma3-20260207_1430/     ← one run
    pipeline_state.json
    automol_prepared_raw_data.csv
    automol_prepared_raw_data_info.json
    automol_split_raw_data.csv
    automol_split_raw_data_info.json
    gamma1_stackingregmodel.pt
    gamma1_train_info.json
    gamma2_stackingregmodel.pt
    ...
    merged_stackingregmodel.pt
    gamma1_refitted_stackingregmodel.pt
    ...
    merged_refitted_stackingregmodel.pt
    model_card.md
```

**Run folder name**: `{dataset_stem}-{prop1}_{prop2}_{...}-{YYYYMMDD}_{HHMM}`

- `dataset_stem`: original CSV filename without extension
- Properties joined with `_`, sorted alphabetically
- Timestamp at folder creation time

## State Schema (`pipeline_state.json`)

```json
{
  "pipeline_version": "2.0",
  "run_id": "raw_data-gamma1_gamma2_gamma3-20260207_1430",
  "created_at": "2026-02-07T14:30:00Z",
  "last_updated": "2026-02-07T15:45:00Z",
  "steps_completed": [0, 1, 2, 3],
  "current_step": 4,
  "pipeline_complete": false,
  "task_ids": {
    "prepare": "1",
    "split": "2",
    "train": "3",
    "evaluate": "4",
    "review": "5",
    "refit": "6",
    "complete": "7"
  },
  "detection": {
    "smiles_column": "Drug",
    "overall_task_type": "regression",
    "targets": [],
    "blender_properties": [],
    "recommended_features": ["Bottleneck", "rdkit"],
    "n_samples": 910,
    "has_3d_data": false
  },
  "config": {
    "data_file": "Data/Caco2_wang.csv",
    "smiles_column": "Drug",
    "target_properties": ["Y"],
    "task_type": "regression",
    "blender_properties": [],
    "feature_keys": ["Bottleneck", "rdkit"],
    "computational_load": "moderate",
    "split_strategy": "mixed",
    "use_advanced": false,
    "output_folder": "MolagentFiles/Caco2_wang-Y-20260207_1430/"
  },
  "outputs": {
    "prepared_csv": null,
    "prepared_info": null,
    "split_csv": null,
    "split_info": null,
    "model_files": {},
    "merged_model_file": null,
    "train_info": {},
    "refitted_model_files": {},
    "merged_refitted_model_file": null,
    "model_card": null
  },
  "metrics": {},
  "decisions": {
    "refit_approved": null,
    "include_test_in_refit": null
  },
  "errors": []
}
```

### Key Schema Notes

- **`"outputs"`** is the canonical key for file paths (NOT `"files"`). Scripts use `state.get("outputs", state.get("files", {}))` for backward compatibility.
- **`config.output_folder`** points to the run folder (e.g., `MolagentFiles/raw_data-gamma1-20260207_1430/`), NOT flat `MolagentFiles/`.
- **`run_id`** matches the folder name and is used for registry lookups.
- **`model_files`** and **`train_info`** are dicts keyed by property name (not flat strings).
- **`merged_model_file`** / **`merged_refitted_model_file`** are optional — absent/null means individual files only.

## Subagent Dispatch

Steps 2, 3, 4, 5, and 7 are dispatched as `Task(subagent_type="general-purpose", model="sonnet")`. The orchestrator (SKILL.md) passes the step file content and pipeline state as the prompt. The subagent reads/updates `pipeline_state.json` inside the run folder.

Inline steps (0, 1, 6, 8) run in the main context because they require `AskUserQuestion` or display output directly to the user.

## Model Registry

Trained models are tracked in `MolagentFiles/model_registry.json` (global, not inside run folders). The registry is updated automatically by step 8.

### Registry Entry Schema

```json
{
  "id": "Caco2_wang-Y-20260207_1430",
  "created_at": "2026-02-07T14:30:00",
  "model_file": "MolagentFiles/Caco2_wang-Y-20260207_1430/Y_refitted_stackingregmodel.pt",
  "target_properties": ["Y"],
  "task_type": "regression",
  "metrics": {"R2": 0.85, "RMSE": 0.42, "MAE": 0.31},
  "feature_keys": ["Bottleneck", "rdkit"],
  "smiles_column": "Drug",
  "blender_properties": [],
  "source_dataset": "Data/Caco2_wang.csv",
  "is_refitted": true,
  "model_card": "MolagentFiles/Caco2_wang-Y-20260207_1430/model_card.md",
  "train_info": "MolagentFiles/Caco2_wang-Y-20260207_1430/Y_train_info.json",
  "n_samples": 910,
  "computational_load": "moderate",
  "split_strategy": "mixed"
}
```

### Model ID Format

`{dataset_stem}-{property}-{YYYYMMDD}_{HHMM}` (matches run folder name). Duplicate IDs get `-2`, `-3`, etc. appended.

### model_registry.py CLI

```
Usage: model_registry.py [COMMAND]

Commands:
  register  --pipeline-state PATH    Register model from pipeline state
  list      --format json|table      List all registered models
            --directory DIR
  get       MODEL_ID                 Get full details of one model
            --directory DIR
  remove    MODEL_ID                 Remove entry (files untouched)
            --directory DIR
```

**Idempotency**: `register` checks if `model_file` already exists in the registry and skips duplicates.

## Pipeline Scripts CLI Reference

### detect_dataset.py

```
Usage: detect_dataset.py [OPTIONS]

Options:
  --csv-file TEXT  Path to input CSV file [required]
  --sdf-file TEXT  Path to SDF file (indicates 3D data available)
  --help           Show this message and exit.
```

**Output**: JSON to stdout with detection results.

### generate_model_card.py

```
Usage: generate_model_card.py [OPTIONS]

Options:
  --pipeline-state TEXT  Path to pipeline_state.json [required]
  --help                 Show this message and exit.
```

**Output**: Creates `model_card.md` in the run folder, prints JSON with card path.

## Step Reference

| Step | Name | Mode | File | Condition |
|------|------|------|------|-----------|
| 0 | Detect | Inline | step-0-detect.md | Always |
| 1 | Configure | Inline | step-1-configure.md | Always |
| 2 | Prepare | Task (general-purpose) | step-2-prepare.md | Always |
| 3 | Split | Task (general-purpose) | step-3-split.md | Always |
| 4 | Train | Task (general-purpose) | step-4-train.md | Always |
| 5 | Evaluate | Task (general-purpose) | step-5-evaluate.md | Skip if advanced |
| 6 | Review | Inline | step-6-review.md | Always |
| 7 | Refit | Task (general-purpose) | step-7-refit.md | Skip if user declines |
| 8 | Complete | Inline | step-8-complete.md | Always |

## File Naming Conventions

| Stage | Input | Output |
|-------|-------|--------|
| Prepare | `Original.csv` | `automol_prepared_Original.csv` + `_info.json` |
| Split | `automol_prepared_Original.csv` | `automol_split_Original.csv` + `_info.json` |
| Train (reg) | `automol_split_Original.csv` | `{property}_stackingregmodel.pt` + `_train_info.json` |
| Train (clf) | `automol_split_Original.csv` | `{property}_stackingclfmodel.pt` + `_train_info.json` |
| Evaluate | model + split CSV | `{property}_evaluation_predictions.csv` + `_evaluation_info.json` |
| Refit (reg) | model + split CSV | `{property}_refitted_stackingregmodel.pt` |
| Refit (clf) | model + split CSV | `{property}_refitted_stackingclfmodel.pt` |

## Feature Keys

| Key | Type | Description |
|-----|------|-------------|
| `Bottleneck` | 2D | Pretrained transformer embeddings (250D) |
| `rdkit` | 2D | RDKit physicochemical descriptors (~200D) |
| `fps_2048_2` | 2D | Morgan fingerprints radius 2 (2048 bits) |
| `prolif` | 3D | Protein-ligand interaction features |
| `AffGraph` | 3D | GradFormer 3D structural features |

## Computational Load

| Load | Time | Best For |
|------|------|----------|
| `free` | 0-2 min | First signal test |
| `cheap` | 2-10 min | Quick prototyping |
| `moderate` | 10-360 min | Production models |
| `expensive` | 1-48 hours | Maximum performance |

## Split Strategies

| Strategy | Description |
|----------|-------------|
| `mixed` | Comprehensive validation (default) |
| `stratified` | Stratified k-fold for balanced splits |
| `leave_group_out` | Scaffold-based to prevent data leakage |

## Recommendations

The detection script generates data-driven recommendations for `computational_load`, `split_strategy`, and `use_advanced`. Each recommendation has a `value` and human-readable `reason`.

### Computational Load Thresholds

| Condition | Recommendation | Reason |
|-----------|---------------|--------|
| n_samples < 100 | `free` | Very small dataset — fast single-model signal check recommended |
| n_samples < 200 | `cheap` | Small dataset — extensive search won't improve results |
| n_samples < 500 | `cheap` | Quick iteration recommended, upgrade to moderate if needed |
| has_3d AND n_samples >= 1000 | `expensive` | Large 3D dataset benefits from full search |
| has_3d | `moderate` | 3D features add compute cost |
| n_targets > 5 AND n_samples >= 2000 | `expensive` | Multi-target large dataset |
| n_samples >= 2000 | `moderate` | Large dataset — expensive available for max performance |
| else | `moderate` | Good balance of speed and model quality |

### Split Strategy Thresholds

| Condition | Recommendation | Reason |
|-----------|---------------|--------|
| Any classification target with minority < 15% | `stratified` | Imbalanced classes need balanced splits |
| n_samples >= 1000 | `leave_group_out` | Enough data to test scaffold generalization |
| else | `mixed` | Comprehensive validation (stratified + scaffold + activity cliffs) |

### Advanced Training

Always returns `false` — advanced training is opt-in. The reason varies to educate the user:

| Condition | Reason |
|-----------|--------|
| Mixed regression + classification | Standard training recommended — consider advanced for per-task control |
| n_targets > 5 | Standard training recommended — consider advanced for custom CV |
| else | Standard training builds a strong stacking ensemble automatically |

### Warning Conditions

| Condition | Warning |
|-----------|---------|
| n_samples < 50 | Very small dataset — results may be unreliable |
| n_samples < 100 | Small dataset — consider cheap load for quick validation |
| Target correlation > 0.95 | Targets X and Y highly correlated — consider dropping one |
| Null rate > 20% | Target X has N% missing values — consider data cleaning first |
| Classification minority < 10% | Target X severely imbalanced — stratified split recommended |
| Classification n_classes > 10 | Target X has N classes — verify this is classification, not regression |
| Regression skewness > 1.0 (all positive) | Target X is skewed — consider log10 transformation |
| Regression skewness > 1.0 (has negatives) | Target X is skewed — consider Yeo-Johnson transformation |

### Target Distribution Detection

For regression targets, the detection script computes distribution metrics:

| Field | Description |
|-------|-------------|
| `skewness` | Measure of distribution asymmetry (normal ≈ 0) |
| `kurtosis` | Measure of tail heaviness (normal ≈ 0) |
| `is_skewed` | True if \|skewness\| > 1.0 |
| `suggest_log_transform` | True if skewed AND all values > 0 |
| `all_positive` | True if all target values are positive |
| `min`, `max` | Range of target values |

### Target Transformations

The `recommendations.target_transformations` field contains transformation suggestions for skewed regression targets:

```json
{
  "target_transformations": [
    {
      "column": "IC50",
      "transform": "log10",
      "reason": "Skewed distribution (skewness=2.45), all values positive"
    }
  ]
}
```

### n_jobs Configuration

The `n_jobs` setting is derived from `computational_load` when not explicitly specified:

| Load | n_jobs_outer | n_jobs_inner | n_jobs_method |
|------|--------------|--------------|---------------|
| `free` | 1 | 1 | 1 |
| `cheap` | 1 | 2 | 2 |
| `moderate` | -1 | 2 | 2 |
| `expensive` | -1 | -1 | 2 |

- `-1` means "use all available CPU cores"
- Outer jobs parallelize CV folds
- Inner jobs parallelize hyperparameter search
- Method jobs parallelize individual estimator fitting

### Example Detection Output (with recommendations)

```json
{
  "file": "Data/raw_data.csv",
  "file_stem": "raw_data",
  "n_samples": 157,
  "n_columns": 4,
  "columns": ["smiles", "gamma1", "gamma2", "gamma3"],
  "smiles_column": "smiles",
  "smiles_detected": true,
  "overall_task_type": "regression",
  "targets": [
    {
      "column": "gamma1",
      "task_type": "regression",
      "unique_values": 155,
      "null_count": 0,
      "mean": 5.1234,
      "std": 0.8765,
      "min": 3.2,
      "max": 8.9,
      "skewness": 0.345,
      "kurtosis": 0.123,
      "is_skewed": false,
      "suggest_log_transform": false,
      "all_positive": true
    }
  ],
  "blender_properties": [],
  "has_3d_data": false,
  "recommended_features": ["Bottleneck"],
  "characteristics": {
    "valid_smiles": 155,
    "total_smiles": 157,
    "class_balance": {},
    "high_correlations": [],
    "null_rates": {"gamma1": 0.0, "gamma2": 0.0, "gamma3": 0.0}
  },
  "recommendations": {
    "computational_load": {"value": "cheap", "reason": "Small dataset (157 samples) — extensive search won't improve results"},
    "split_strategy": {"value": "mixed", "reason": "Comprehensive validation (stratified + scaffold + activity cliffs)"},
    "use_advanced": {"value": false, "reason": "Standard training builds a strong stacking ensemble automatically"},
    "target_transformations": [],
    "warnings": []
  }
}
```
