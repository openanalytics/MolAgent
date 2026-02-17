# Step 4: Train Model

**Mode**: Task (general-purpose subagent) - potentially long-running

## Purpose

Train the ensemble model using the configured settings. This step can take minutes to hours depending on computational load.

## Instructions for Subagent

1. Read `pipeline_state.json` from the run folder to get config and file paths
2. Determine which training script to use
3. Build and execute the training command
4. Verify outputs exist
5. Update state with model file paths

## Script Selection

| task_type | use_advanced | Script |
|-----------|-------------|--------|
| regression | false | `skills/train-pipeline/scripts/train_regression.py` |
| classification | false | `skills/train-pipeline/scripts/train_classification.py` |
| regression_classification | false | `skills/train-pipeline/scripts/train_regression_classification.py` |
| regression | true | `skills/train-pipeline/scripts/train_regression_advanced.py` |
| classification | true | `skills/train-pipeline/scripts/train_classification_advanced.py` |
| regression_classification | true | `skills/train-pipeline/scripts/train_regression_classification_advanced.py` |

## Command Construction - Standard

```bash
uv run python ${AUTOMOL_ROOT:-$PWD}/skills/train-pipeline/scripts/train_{task_type}.py \
    --csv-file {outputs.split_csv} \
    --feature-keys {feature_key1} \
    --feature-keys {feature_key2} \
    --computational-load {computational_load} \
    --output-folder {config.output_folder} \
    --verbose
```

## Command Construction - Advanced

```bash
uv run python ${AUTOMOL_ROOT:-$PWD}/skills/train-pipeline/scripts/train_{task_type}_advanced.py \
    --csv-file {outputs.split_csv} \
    --properties {prop1} --properties {prop2} \
    --feature-keys {feature_key1} \
    --feature-keys {feature_key2} \
    --computational-load {computational_load} \
    --evaluate-after-training \
    --verbose
```

Note: Advanced training with `--evaluate-after-training` runs evaluation automatically, so step 5 can be skipped.

## Verify Outputs

For each target property, check:
- `{config.output_folder}/{property}_stackingregmodel.pt` (regression)
- `{config.output_folder}/{property}_stackingclfmodel.pt` (classification)
- `{config.output_folder}/{property}_train_info.json`

## Merge Model Files (Multi-Property)

If multiple properties were trained, merge per-property `.pt` files into a single file to eliminate encoder duplication:

```bash
uv run python ${AUTOMOL_ROOT:-$PWD}/skills/train-pipeline/scripts/merge_models.py \
    --model-files {config.output_folder}/{prop1}_stackingregmodel.pt \
    --model-files {config.output_folder}/{prop2}_stackingregmodel.pt \
    --output-file {config.output_folder}/merged_stackingregmodel.pt \
    --verify-encoder --verbose
```

Skip this step if only one property was trained (single file, nothing to merge).

## Update State

Update `pipeline_state.json` in the run folder:
```json
{
  "outputs": {
    "model_files": {
      "prop1": "{config.output_folder}/{prop1}_stackingregmodel.pt",
      "prop2": "{config.output_folder}/{prop2}_stackingregmodel.pt"
    },
    "merged_model_file": "{config.output_folder}/merged_stackingregmodel.pt",
    "train_info": {
      "prop1": "{config.output_folder}/{prop1}_train_info.json",
      "prop2": "{config.output_folder}/{prop2}_train_info.json"
    }
  },
  "steps_completed": [0, 1, 2, 3, 4],
  "current_step": 5
}
```

If only one property: omit `merged_model_file`. If advanced training with `--evaluate-after-training`, also set:
```json
{
  "steps_completed": [0, 1, 2, 3, 4, 5],
  "current_step": 6
}
```

## Summary Output

Print when done:
```
Step 4 complete: Model trained
  Task type: {task_type}
  Properties: {properties}
  Features: {feature_keys}
  Load: {computational_load}
  Model files:
    {prop1}: {model_file_1}
    {prop2}: {model_file_2}
  Merged: {merged_model_file or "N/A (single property)"}
```

## Critical Rules

- State key is `"outputs"` (NOT `"files"`)
- Read split_csv from `outputs.split_csv` (NOT `files.split_csv`)
- Paths: use `Path(dir) / filename` (NEVER f-string concatenation)
- Read/update/write `pipeline_state.json` in the run folder
- Scripts: `uv run python ${AUTOMOL_ROOT:-$PWD}/...`

## Error Handling

Training failures are common with large models. If it fails:
- Capture the full error output
- Common causes: out of memory, invalid features, missing data
- Append error to state
- Report back to orchestrator
