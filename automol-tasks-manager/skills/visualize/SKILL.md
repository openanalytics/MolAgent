---
name: visualize
description: Generate an interactive HTML dashboard from AutoMol evaluation results. Auto-discovers completed runs with evaluation data and renders Plotly.js charts. Use when the user wants to visualize training results.
allowed-tools: Read, Glob, Grep, Bash, AskUserQuestion
---

# AutoMol — Visualize

Single-phase skill: discover evaluation runs, generate dashboard, open in browser.

---

## Step 1: Discover Runs

Glob for `MolagentFiles/*/pipeline_state.json`.

For each file found, read it and check:
- `steps_completed` includes step 5 (evaluate)
- `metrics` exists and is non-empty

Build a list of qualifying runs with: `run_id`, `config.task_type`, `config.target_properties`, `metrics`, `last_updated`.

**0 runs**: Tell the user: "No completed evaluation runs found. Use the `train-pipeline` skill to train and evaluate a model first." **STOP.**

---

## Step 2: Select Run

**1 run**: Auto-select it. Display a brief summary:

```
Run: {run_id}
  Task: {task_type}
  Properties: {target_properties}
  Metrics: {formatted_metrics}
```

**N runs**: Present choices via AskUserQuestion:

```
AskUserQuestion:
  question: "Which run do you want to visualize?"
  header: "Run"
  options: (up to 4 runs, most recent first)
    - "{run_id} — {target_properties} ({task_type}, {key_metric})"
    - ...
```

After selection, display the run summary as above.

---

## Step 3: Ensure Evaluation Results in State

The dashboard script reads `outputs.evaluation_results` from `pipeline_state.json`. MolAgent's evaluate step (step 5) writes evaluation CSVs to the run folder but may not record their paths in `outputs.evaluation_results`.

Before running the script, check if `outputs.evaluation_results` is present and populated. If not, auto-populate it:

For each property in `config.target_properties`, the expected CSV path is:
```
{config.output_folder}/{property}_evaluation_predictions.csv
```

If this file exists, add it to `outputs.evaluation_results` in the state and write the updated `pipeline_state.json` back to disk. For example:
```json
{
  "outputs": {
    "evaluation_results": {
      "prop1": "MolagentFiles/run-folder/prop1_evaluation_predictions.csv",
      "prop2": "MolagentFiles/run-folder/prop2_evaluation_predictions.csv"
    }
  }
}
```

If no evaluation CSVs are found for any property, tell the user: "Evaluation prediction files not found. Re-run step 5 (evaluate) via the `train-pipeline` skill." **STOP.**

---

## Step 4: Generate Dashboard

Run the generation script:

```bash
uv run python ${AUTOMOL_ROOT:-$PWD}/skills/visualize/scripts/generate_dashboard.py \
    --pipeline-state {pipeline_state_path} \
    --output {run_folder}/dashboard.html \
    --verbose
```

Where:
- `{pipeline_state_path}` is the full path to the selected `pipeline_state.json`
- `{run_folder}` is the parent directory of the pipeline state file

---

## Step 5: Open Dashboard

Open the generated HTML in the default browser (detect platform):

```bash
# macOS
open {run_folder}/dashboard.html
# Linux
xdg-open {run_folder}/dashboard.html
# Windows
start {run_folder}/dashboard.html
```

Report the result:

```
Dashboard generated and opened!

  Run: {run_id}
  Output: {run_folder}/dashboard.html
  Properties: {target_properties}

The dashboard includes:
  - Interactive Plotly.js charts (scatter, residuals, error distribution, etc.)
  - Metrics summary panel
  - Property selector (for multi-property runs)
  - Overview and detailed view presets
  - Copy-able findings summary
```

---

## Important Notes

- `${AUTOMOL_ROOT:-$PWD}` resolves to this plugin's root directory (set by SessionStart hook, falls back to `$PWD`)
- Python scripts: `uv run python ...` (dependencies resolved via PEP 723 inline metadata)
- The dashboard is a self-contained HTML file — no server needed, works offline after first load (Plotly.js CDN)
- Only runs with completed evaluation (step 5) and non-empty `metrics` are eligible
