# Step 2: Prepare Data

**Mode**: Task (general-purpose subagent)

## Purpose

Run the data preparation script to standardize SMILES, apply transformations, and create the prepared CSV.

## Instructions for Subagent

1. Read `pipeline_state.json` from the run folder (`config.output_folder`) to get config
2. Build and execute the preparation command
3. Verify outputs exist
4. Update `pipeline_state.json` with file paths

## Command Construction

Read config values:
- `data_file`: path to input CSV
- `smiles_column`: SMILES column name
- `target_properties`: list of properties
- `task_type`: determines which script to run
- `blender_properties`: optional blender columns
- `output_folder`: run folder path (from `config.output_folder`)

### Regression

```bash
uv run $MOLAGENT_PLUGIN_ROOT/skills/train-pipeline/scripts/prepare_for_regression.py \
    --csv-file {data_file} \
    --smiles-column {smiles_column} \
    --properties {prop1} --properties {prop2} \
    --output-folder {config.output_folder} \
    --verbose
```

Add `--blender-properties {bp}` for each blender property if any.

**Target Transformations**: Check `config.target_transformations` in `pipeline_state.json`. If it contains a transform for any target:
- `"log10"` → add `--use-log10` flag. The transformed target will be `log10_{target}`.
- `"yeo_johnson"` → no script flag needed (handled internally by the training step). Note this in the summary.

For more data preparation options add `--help` to see all command-line options.

### Classification

```bash
uv run $MOLAGENT_PLUGIN_ROOT/skills/train-pipeline/scripts/prepare_for_classification.py \
    --csv-file {data_file} \
    --smiles-column {smiles_column} \
    --properties {prop1} --properties {prop2} \
    --output-folder {config.output_folder} \
    --nb-classes 2 \
    --verbose
```

### RegressionClassification (binary targets modeled with regression)

Use `prepare_for_classification.py` with `--categorical` — targets are already binary 0/1:

```bash
uv run $MOLAGENT_PLUGIN_ROOT/skills/train-pipeline/scripts/prepare_for_classification.py \
    --csv-file {data_file} \
    --smiles-column {smiles_column} \
    --properties {prop1} --properties {prop2} \
    --output-folder {config.output_folder} \
    --categorical \
    --verbose
```

Note: log10/logit transforms do NOT apply (targets are binary). The training step uses regression estimators on the 0/1 values, clipping predictions to [0,1] as probability estimates.

### Mixed target types

If detection reports mixed regression + classification targets, the pipeline defaults to the dominant task type. Training separate models for each type is recommended — running `prepare_for_regression.py` and `prepare_for_classification.py` separately for different targets in the same run is NOT supported.

## Verify Outputs

Check that these files exist:
- `{config.output_folder}/automol_prepared_{file_stem}.csv`
- `{config.output_folder}/automol_prepared_{file_stem}_info.json`

Where `{file_stem}` is the original CSV filename without extension.

## Update State

Update `pipeline_state.json` in the run folder:
```json
{
  "outputs": {
    "prepared_csv": "{config.output_folder}/automol_prepared_{file_stem}.csv",
    "prepared_info": "{config.output_folder}/automol_prepared_{file_stem}_info.json"
  },
  "steps_completed": [0, 1, 2],
  "current_step": 3
}
```

## Summary Output

Print when done:
```
Step 2 complete: Data prepared
  Input:  {data_file} ({n_rows} rows)
  Output: {prepared_csv}
  Rows after preparation: {n_prepared_rows}
```

## Critical Rules

- State key is `"outputs"` (NOT `"files"`)
- Paths: use `Path(dir) / filename` (NEVER f-string concatenation)
- Read/update/write `pipeline_state.json` in the run folder (`config.output_folder`)
- Scripts: `uv run $MOLAGENT_PLUGIN_ROOT/...` (always set via .claude/settings.local.json)

## Error Handling

If preparation fails:
- Capture the error message
- Append to `errors` array in state
- Report the error back to the orchestrator
