# Training Pipeline Examples

## Example 1: Regression — Single Invocation

**User**: "Train a model on raw_data.csv"

The skill auto-detects, auto-configures, and auto-executes in one session:

```
Analyzing raw_data.csv...

  Found 157 molecules with 3 target properties
  SMILES column: smiles
  Task type: regression
  Recommended features: Bottleneck
  Recommended load: cheap — Small dataset

Pipeline Plan (auto-configured from detection):

  Dataset: raw_data.csv (157 samples)
  SMILES:  smiles
  Task:    regression
  Targets: gamma1, gamma2, gamma3
  Features: Bottleneck
  Split:   mixed
  Load:    cheap
```

Skill asks 1 question (computational load), then auto-executes because the user clearly requested training.

Creates run folder `MolagentFiles/raw_data-gamma1_gamma2_gamma3-20260207_1430/`, saves state, runs all 7 steps via subagents. At step 6 (review), shows metrics. At step 8, generates model card and registers.

```
Pipeline Complete!
  Run folder: MolagentFiles/raw_data-gamma1_gamma2_gamma3-20260207_1430/
  Model ID: raw_data-gamma1_gamma2_gamma3-20260207-1430
  Registry: MolagentFiles/model_registry.json
  Next: use 'predict' skill to make predictions
```

## Example 2: Classification — Binary Target

**User**: "Train a classification model on ChEMBL_SMILES.csv for prop5"

Detection finds prop5 is binary (0/1, 2 unique values). Auto-configures as classification.

```
Pipeline Plan:
  Dataset: ChEMBL_SMILES.csv (157 samples)
  Task:    classification
  Target:  prop5
  Features: Bottleneck
  Load:    cheap
```

During training (step 4), the quantile-binning preparation may collapse `Class_prop5` to a single class since prop5 is already binary. The training script auto-detects this and falls back to the original `prop5` column. Model file: `prop5_stackingclfmodel.pt`.

## Example 3: User Narrows Scope

**User**: "Train on multi_target.csv"

Detection finds 5 numeric columns (prop1-prop5). Auto-config includes all 5.

Skill asks:
1. **Load**: User picks "Cheap"
2. **Plan**: User picks "Change targets or features"

User says: "Only prop1 and prop2, both regression."

Skill updates config, presents revised plan, and proceeds with just prop1 and prop2. Per-property models are auto-merged into `merged_stackingregmodel.pt`.

## Example 4: Resume After Interruption

**User**: "Continue training" (after crash during step 4)

Skill scans `MolagentFiles/*/pipeline_state.json`, finds one incomplete run folder with `current_step: 4`. Enters execution directly.

```
Resuming from step 4 (Train). Steps 0-3 already completed.
Run folder: MolagentFiles/raw_data-gamma1-20260207_1430/
```

Creates tasks only for remaining steps 4-8.

## Example 5: Multiple Incomplete Runs

**User**: "/train-pipeline"

Skill finds 2 incomplete run folders. Asks which to resume:

```
AskUserQuestion:
  "Found 2 incomplete pipeline runs. Which to resume?"
  - "raw_data-gamma1-20260207_1430 (step 4/8)"
  - "Caco2-Y-20260207_1500 (step 2/8)"
  - "Start fresh"
```

## Example 6: Stop Hook in Action

If the LLM tries to stop without writing `pipeline_state.json`, the Stop hook blocks:

```
VALIDATION FAILED: No pipeline_state.json found in MolagentFiles/*/

ACTION REQUIRED: Use the Write tool to create pipeline_state.json in the run folder...
```

The LLM is forced to actually write the plan file before it can finish.

## Example 7: Evaluation with PDF Report Generation

Generate a detailed PDF evaluation report with custom metadata:

**Regression Model Evaluation:**
```bash
uv run python evaluate_regression_model.py \
    --model-file MolagentFiles/run-folder/Y_stackingregmodel.pt \
    --test-file MolagentFiles/run-folder/automol_split_data.csv \
    --title "Caco-2 Permeability Model - Final Evaluation" \
    --authors "Jane Doe, John Smith" \
    --mails "research@lab.edu" \
    --summary "Ensemble stacking model (LightGBM + XGBoost + Ridge meta-learner) for predicting Caco-2 permeability. Uses Bottleneck (250D) and RDKit (~200D) features. Evaluated on 20% held-out test set."
```

**Classification Model Evaluation:**
```bash
uv run python evaluate_classification_model.py \
    --model-file MolagentFiles/run-folder/Class_prop5_stackingclfmodel.pt \
    --test-file MolagentFiles/run-folder/automol_split_data.csv \
    --title "Binary Activity Classification Report" \
    --authors "Jane Doe" \
    --summary "Binary classifier for compound activity prediction. Model trained on 157 compounds with class balance 70/30."
```

**PDF Report Contents:**

| Section | Regression | Classification |
|---------|------------|----------------|
| Header | Title, authors, contact, summary | Title, authors, contact, summary |
| Model Info | Property, task type, features, sample count, metrics | Property, task type, classes, features, sample count |
| Figure 1 | Scatter plot: predicted vs true with R², RMSE, MAE | Classification report: precision, recall, F1 per class |
| Figure 2 | Scatter plot: moving average error across ranges | Confusion matrix with class-wise counts |
| Figure 3 | Bar plot: performance metrics summary | ROC curve with AUC score |
| Figure 4 | — | F1-score vs probability threshold |
| Figure 5 | — | Predicted probability distribution |
| Figure 6 | — | Precision-Recall curve |
| Figure 7 | — | Reliability diagram |

**Output Files:**
- `{property}_evaluation_report.pdf` - PDF report with figures and metrics
- `{property}_evaluation_predictions.csv` - Predictions with SMILES, true values, predicted values, uncertainties
- `{property}_evaluation_info.json` - Evaluation metadata and metrics
