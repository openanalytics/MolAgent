# Step 7: Refit Model

**Mode**: Task (general-purpose subagent)
**Conditional**: Always runs unless user explicitly declined refit in step 6

## Purpose

Refit the trained model on the full dataset (train + validation + optionally test) for deployment.

## Skip Condition

Check `pipeline_state.json`:
- If `decisions.refit_approved` is explicitly `false`, skip to step 8
- Default behavior: always refit (treat missing/null/true as approved)

## Instructions for Subagent

1. Read `pipeline_state.json` from the run folder for model and data paths
2. Build refit command
3. Execute and verify output
4. Update state

## Command Construction

For each property, refit its model:

```bash
uv run python ${AUTOMOL_ROOT:-$PWD}/skills/train-pipeline/scripts/refit_model.py \
    --model-file {outputs.model_files[property]} \
    --csv-file {outputs.split_csv} \
    --output-folder {config.output_folder} \
    --include-test-set \
    --verbose
```

The refit script auto-detects properties from `train_info.json`.

## Verify Outputs

Check for refitted model file:
- `{config.output_folder}/{property}_refitted_stackingregmodel.pt` (regression)
- `{config.output_folder}/{property}_refitted_stackingclfmodel.pt` (classification)

## Merge Refitted Model Files (Multi-Property)

If multiple properties were refitted, merge the refitted per-property `.pt` files:

```bash
uv run python ${AUTOMOL_ROOT:-$PWD}/skills/train-pipeline/scripts/merge_models.py \
    --model-files {config.output_folder}/{prop1}_refitted_stackingregmodel.pt \
    --model-files {config.output_folder}/{prop2}_refitted_stackingregmodel.pt \
    --output-file {config.output_folder}/merged_refitted_stackingregmodel.pt \
    --verify-encoder --verbose
```

Skip if only one property was trained.

## Update State

Update `pipeline_state.json` in the run folder:
```json
{
  "outputs": {
    "refitted_model_files": {
      "prop1": "{config.output_folder}/{prop1}_refitted_stackingregmodel.pt",
      "prop2": "{config.output_folder}/{prop2}_refitted_stackingregmodel.pt"
    },
    "merged_refitted_model_file": "{config.output_folder}/merged_refitted_stackingregmodel.pt"
  },
  "steps_completed": [0, 1, 2, 3, 4, 5, 6, 7],
  "current_step": 8
}
```

## Summary Output

Print when done:
```
Step 7 complete: Model refitted for deployment
  Refitted model files:
    {prop1}: {refitted_model_file_1}
    {prop2}: {refitted_model_file_2}
  Merged: {merged_refitted_model_file or "N/A (single property)"}
```

## Critical Rules

- State key is `"outputs"` (NOT `"files"`)
- Model files are per-property dicts: `outputs.model_files.prop1`
- Paths: use `Path(dir) / filename` (NEVER f-string concatenation)
- Read/update/write `pipeline_state.json` in the run folder
- Scripts: `uv run python ${AUTOMOL_ROOT:-$PWD}/...`

## Error Handling

If refit fails:
- This is non-fatal - the original model still exists
- Log the error
- The model card will reference the original (non-refitted) model instead
