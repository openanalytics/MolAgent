# Step 6: Review Results (User Checkpoint)

**Mode**: Inline (requires AskUserQuestion)

## Purpose

Present training and evaluation results to the user and get their decision on next steps.

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

## User Decision

After displaying metrics, ask the user:

```
AskUserQuestion:
  question 1:
    header: "Results"
    question: "How do you want to proceed?"
    options:
      - "Refit & finish (Recommended) — Refit on all data (train+valid+test) for deployment."
      - "Refit without test set — Refit on train+valid only, keeping test set clean for future reporting."
      - "Skip refit — Keep current model as-is and finish the pipeline."
      - "Abort — Stop the pipeline here (can resume later)."
```

## Update State

Based on user answer:

**"Refit & finish"** (default):
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

**"Refit without test set"**:
```json
{
  "decisions": {
    "refit_approved": true,
    "include_test_in_refit": false
  },
  "steps_completed": [0, 1, 2, 3, 4, 5, 6],
  "current_step": 7
}
```

**"Skip refit"**:
```json
{
  "decisions": {
    "refit_approved": false,
    "include_test_in_refit": false
  },
  "steps_completed": [0, 1, 2, 3, 4, 5, 6],
  "current_step": 8
}
```

**"Abort"**: Leave state unchanged. STOP.

## Next Step

- **Refit approved**: proceed to **step-7-refit.md**
- **Refit declined**: proceed to **step-8-complete.md**
- **Abort**: STOP (user can resume later)
