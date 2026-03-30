# Step 0: Detect Dataset

**Mode**: Inline (runs in main context, needs user interaction)

## Purpose

Analyze the user's CSV file to detect:
- SMILES column name
- Numeric target columns and their task types
- Blender property candidates
- Feature recommendations based on dataset size

## Execution

Run the detection script and capture JSON output if the provided file is a CSV:

```bash
uv run python ${AUTOMOL_ROOT:-$PWD}/skills/train-pipeline/scripts/detect_dataset.py \
    --csv-file {data_file}
```

If the user provides 3D data (SDF file), use `--sdf-file {sdf_file}` instead.

```bash
uv run python ${AUTOMOL_ROOT:-$PWD}/skills/train-pipeline/scripts/detect_dataset.py \
    --sdf-file {data_file}
```

## Parse Output

The script outputs JSON with these fields:
- `smiles_column`: Detected SMILES column name (or null)
- `overall_task_type`: "regression", "classification", "regression_classification", or "unknown"
- `targets`: Array of numeric columns with task type, unique values, and stats
- `blender_properties`: Array of candidate blender columns with confidence/reasons (checked across ALL targets, deduplicated)
- `recommended_features`: Recommended feature keys for this dataset size
- `n_samples`: Row count
- `has_3d_data`: Whether 3D data is available
- `characteristics`: Dataset quality metrics
  - `valid_smiles`: Count of heuristically valid SMILES strings
  - `total_smiles`: Total row count
  - `class_balance`: Per-classification-target minority/majority class info (`minority_pct`, `n_classes`, `counts`)
  - `high_correlations`: Array of target pairs with |r| > 0.9 (`target_a`, `target_b`, `correlation`)
  - `null_rates`: Per-target null percentage
- `recommendations`: Data-driven configuration suggestions
  - `computational_load`: `{value, reason}` — free/cheap/moderate/expensive based on dataset size and complexity
  - `split_strategy`: `{value, reason}` — mixed/stratified/leave_group_out based on size and class balance
  - `use_advanced`: `{value, reason}` — always false, but reason educates user on when to consider it
  - `warnings`: Array of human-readable warning strings (empty if no issues detected)

## Think and Plan

After parsing the detection output, present reasoning to the user:

```
Analyzing {data_file}...

  Found {n_samples} molecules with {n_targets} target properties
  SMILES column: {smiles_column}
  Task type: {task_type}
  Recommended features: {feature_keys} — {reason}
  Recommended load: {load} — {reason}
```

This gives the user visibility into the detection reasoning before configuration.

## Error Handling

- If `smiles_detected` is false: ask the user to specify the SMILES column name
- If no numeric columns found: inform user and abort
- If file not found: ask user for correct path

## State Update

Store detection results in `pipeline_state.json` under a `detection` key (includes characteristics and recommendations):
```json
{
  "detection": {
    "smiles_column": "Drug",
    "overall_task_type": "regression",
    "targets": [...],
    "blender_properties": [...],
    "recommended_features": ["Bottleneck", "rdkit"],
    "n_samples": 910,
    "has_3d_data": false,
    "characteristics": {
      "valid_smiles": 905,
      "total_smiles": 910,
      "class_balance": {},
      "high_correlations": [],
      "null_rates": {"Y": 0.0}
    },
    "recommendations": {
      "computational_load": {"value": "moderate", "reason": "Good balance of speed and model quality"},
      "split_strategy": {"value": "mixed", "reason": "Comprehensive validation (stratified + scaffold + activity cliffs)"},
      "use_advanced": {"value": false, "reason": "Standard training builds a strong stacking ensemble automatically"},
      "warnings": []
    }
  }
}
```

## Next Step

Proceed to **step-1-configure.md** (user configuration checkpoint).
