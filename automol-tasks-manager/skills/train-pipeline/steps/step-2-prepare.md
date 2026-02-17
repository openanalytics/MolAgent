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
uv run python ${AUTOMOL_ROOT:-$PWD}/skills/train-pipeline/scripts/prepare_for_regression.py \
    --csv-file {data_file} \
    --smiles-column {smiles_column} \
    --properties {prop1} --properties {prop2} \
    --output-folder {config.output_folder} \
    --verbose
```

Add `--blender-properties {bp}` for each blender property if any. Add `--use-log10` to transform the target using log10, the transformed target will be under `log10_{target}`, for more data preparation options use add `--help` to see all commandline options

### Classification

```bash
uv run python ${AUTOMOL_ROOT:-$PWD}/skills/train-pipeline/scripts/prepare_for_classification.py \
    --csv-file {data_file} \
    --smiles-column {smiles_column} \
    --properties {prop1} --properties {prop2} \
    --output-folder {config.output_folder} \
    --nb-classes 2 \
    --verbose
```

### Regression + Classification (mixed targets)

Run `prepare_for_regression.py` for regression targets and `prepare_for_classification.py` for classification targets separately is NOT supported. Use `prepare_for_regression.py` and handle classification targets downstream, OR choose the dominant task type.

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
- Scripts: `uv run python ${AUTOMOL_ROOT:-$PWD}/...`

## Error Handling

If preparation fails:
- Capture the error message
- Append to `errors` array in state
- Report the error back to the orchestrator
