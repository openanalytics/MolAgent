# Step 5: Evaluate Model

**Mode**: Task (general-purpose subagent)
**Conditional**: Skip if advanced training was used (already evaluated in step 4)

## Purpose

Run model evaluation to generate metrics, PDF report, and prediction files.

## Skip Condition

Check `pipeline_state.json`:
- If `config.use_advanced` is true AND step 5 is already in `steps_completed`, skip this step
- Advanced training with `--evaluate-after-training` handles evaluation internally

## Instructions for Subagent

1. Read `pipeline_state.json` from the run folder to get config and file paths
2. Build evaluation command
3. Execute and verify outputs
4. Extract metrics from output
5. Update state

## Script Selection

| task_type | Script |
|-----------|--------|
| regression | `skills/train-pipeline/scripts/evaluate_regression_model.py` |
| classification | `skills/train-pipeline/scripts/evaluate_classification_model.py` |

## Command Construction

For each property, evaluate its model:

```bash
uv run python ${AUTOMOL_ROOT:-$PWD}/skills/train-pipeline/scripts/evaluate_{task_type}_model.py \
    --model-file {outputs.model_files[property]} \
    --test-file {outputs.split_csv} \
    --output-folder {config.output_folder} \
    --split-type test \
    --verbose
```

Note: Use `--split-type test` if a test set was created (we always use `--add-test-set` in step 3). Fall back to `--split-type valid` if no test set exists.

## Verify Outputs

- `{config.output_folder}/{property}_evaluation_report.pdf`
- `{config.output_folder}/{property}_evaluation_predictions.csv`

## Extract Metrics

Read the evaluation output or train_info.json to extract per-property metrics:

**Regression**: R2, RMSE, MAE
**Classification**: Accuracy, F1, AUC-ROC

## Update State

Update `pipeline_state.json` in the run folder:
```json
{
  "metrics": {
    "prop1": { "R2": 0.85, "RMSE": 0.42, "MAE": 0.31 },
    "prop2": { "R2": 0.78, "RMSE": 0.55, "MAE": 0.40 }
  },
  "steps_completed": [0, 1, 2, 3, 4, 5],
  "current_step": 6
}
```

## Summary Output

Print when done:
```
Step 5 complete: Model evaluated

  Metrics:
    {prop1}: R2={r2}, RMSE={rmse}, MAE={mae}
    {prop2}: R2={r2}, RMSE={rmse}, MAE={mae}
```

For classification:
```
Step 5 complete: Model evaluated

  Metrics:
    {prop1}: Accuracy={acc}, F1={f1}, AUC={auc}
```

## Critical Rules

- State key is `"outputs"` (NOT `"files"`)
- Model files are per-property dicts: `outputs.model_files.prop1` (NOT flat strings)
- Metrics are per-property: `metrics.prop1.R2` (NOT flat `metrics.R2`)
- Paths: use `Path(dir) / filename` (NEVER f-string concatenation)
- Read/update/write `pipeline_state.json` in the run folder
- Scripts: `uv run python ${AUTOMOL_ROOT:-$PWD}/...`

## Error Handling

If evaluation fails:
- This is non-fatal - the model was already trained successfully
- Log the error but allow pipeline to continue to review step
- User can still decide about refit based on training CV metrics
