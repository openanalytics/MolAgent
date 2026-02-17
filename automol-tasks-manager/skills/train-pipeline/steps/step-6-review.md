# Step 6: Review Results (User Checkpoint)

**Mode**: Inline (requires AskUserQuestion)

## Purpose

Present training and evaluation results to the user and get their decision on refitting.

## Display Metrics

Read `pipeline_state.json` and present a summary:

### Regression Metrics
```
Model Performance Summary:
  Property: {property}
  Model file: {model_file}

  Metrics:
  - R2:   {R2}
  - RMSE: {RMSE}
  - MAE:  {MAE}

  Evaluation report: {eval_pdf}
```

### Classification Metrics
```
Model Performance Summary:
  Property: {property}
  Model file: {model_file}

  Metrics:
  - Accuracy: {accuracy}
  - F1 Score: {f1}
  - AUC-ROC:  {auc}

  Evaluation report: {eval_pdf}
```

If metrics are empty (evaluation skipped or failed), read `train_info.json` for CV metrics instead.

## Refit Decision

Refit is the **default behavior** — always proceed to refit unless the user explicitly asks not to. Do NOT ask the user for refit approval. Simply show the metrics summary and proceed directly to step 7.

If the user proactively says "skip refit" or "don't refit", then set `decisions.refit_approved` to `false` and skip to step 8.

## Update State

```json
{
  "decisions": {
    "refit_approved": true,
    "include_test_in_refit": true
  },
  "steps_completed": [0, 1, 2, 3, 4, 5, 6],
  "current_step": 7
}
```

## Next Step

- Default: proceed to **step-7-refit.md** (always)
- Only if user explicitly declined: proceed to **step-8-complete.md**
