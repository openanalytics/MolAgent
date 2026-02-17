# Step 3: Split Data

**Mode**: Task (general-purpose subagent)

## Purpose

Split the prepared data into train/validation/test sets using the configured strategy.

## Instructions for Subagent

1. Read `pipeline_state.json` from the run folder to get config and file paths
2. Build and execute the split command
3. Verify outputs exist
4. Update state with file paths

## Command Construction

Read from state:
- `outputs.prepared_csv`: input CSV path
- `config.target_properties`: properties list
- `config.split_strategy`: mixed, stratified, or leave_group_out
- `config.output_folder`: run folder path

```bash
uv run python ${AUTOMOL_ROOT:-$PWD}/skills/train-pipeline/scripts/split_data.py \
    --csv-file {outputs.prepared_csv} \
    --properties {prop1} --properties {prop2} \
    --split-strategy {split_strategy} \
    --output-folder {config.output_folder} \
    --add-test-set \
    --verbose
```

## Verify Outputs

The split script strips the `automol_prepared_` prefix, so outputs are:
- `{config.output_folder}/automol_split_{file_stem}.csv`
- `{config.output_folder}/automol_split_{file_stem}_info.json`

Where `{file_stem}` is the original filename stem (without `automol_prepared_` prefix).

## Update State

Update `pipeline_state.json` in the run folder:
```json
{
  "outputs": {
    "split_csv": "{config.output_folder}/automol_split_{file_stem}.csv",
    "split_info": "{config.output_folder}/automol_split_{file_stem}_info.json"
  },
  "steps_completed": [0, 1, 2, 3],
  "current_step": 4
}
```

## Summary Output

Print when done:
```
Step 3 complete: Data split
  Strategy: {split_strategy}
  Train: {n_train} samples
  Valid: {n_valid} samples
  Test:  {n_test} samples
```

Read the split info JSON to get counts per split.

## Critical Rules

- State key is `"outputs"` (NOT `"files"`)
- Read prepared_csv from `outputs.prepared_csv` (NOT `files.prepared_csv`)
- Paths: use `Path(dir) / filename` (NEVER f-string concatenation)
- Read/update/write `pipeline_state.json` in the run folder
- Scripts: `uv run python ${AUTOMOL_ROOT:-$PWD}/...`

## Error Handling

If splitting fails:
- Capture the error message
- Append to `errors` array
- Report back to orchestrator
