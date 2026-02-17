---
name: predict
description: Make predictions using a trained AutoMol model. Auto-discovers models from the registry. Use when the user wants to run inference on new molecules.
allowed-tools: Read, Glob, Grep, Bash, AskUserQuestion
---

# AutoMol — Predict

Single-phase skill: load registry, select model, run predictions, show results.

---

## Step 1: Load Registry

Read `MolagentFiles/model_registry.json`.

- If the file does not exist or is empty → tell the user: "No trained models found. Use the `train-pipeline` skill to train a model first." **STOP.**

---

## Step 2: Select Model

**0 models**: STOP (handled above).

**1 model**: Auto-select it. Display a brief summary:

```
Using model: {id}
  Target: {target_properties}
  Task: {task_type}
  Metrics: {formatted_metrics or "No metrics available"}
  Model file: {model_file}
```

**N models**: Present choices via AskUserQuestion:

```
AskUserQuestion:
  question: "Which model do you want to use for predictions?"
  header: "Model"
  options: (up to 4 models, most recent first)
    - "{id} — {target_properties} ({task_type}, {key_metric})"
    - ...
```

After selection, display the model summary as above.

---

## Step 3: Get Input

If the user already provided input molecules (CSV path or SMILES strings) in their original message, use that directly.

Otherwise, ask:

```
AskUserQuestion:
  question: "How would you like to provide molecules for prediction?"
  header: "Input"
  options:
    - "CSV file — I have a CSV with a SMILES column"
    - "SMILES strings — I'll paste individual SMILES"
```

For **CSV file**: ask for the path if not provided. Do NOT force `--smiles-column` from the training config — let `predict.py` auto-detect the column in the user's new data (it tries `SMILES`, `smiles`, `standardized_smiles`, `Stand_SMILES`). Only add `--smiles-column` if the user explicitly specifies a non-standard column name.

For **SMILES strings**: collect the SMILES. Each becomes a `--smiles-list` argument.

---

## Step 4: Check for 3D Feature Warning

Read the registry entry's `feature_keys`. If it contains `prolif` or `AffGraph`:

```
WARNING: This model uses 3D protein-ligand features ({feature_keys}).
Predictions require corresponding protein structures for each molecule.
If you don't have matching 3D data, predictions may fail or be unreliable.
```

Ask via AskUserQuestion whether to proceed.

---

## Step 5: Run Prediction

Build the command from the registry entry. The approach depends on `model_format`:

**Merged model** (`model_format: "merged"`, `model_file` is a single string):

Single invocation predicts all properties at once:

```bash
uv run python ${AUTOMOL_ROOT:-$PWD}/skills/predict/scripts/predict.py \
    --model-file {model_file} \
    {--smiles-file INPUT_CSV | --smiles-list "SMILES1" --smiles-list "SMILES2"} \
    --output-folder MolagentFiles/ \
    --verbose
```

To predict a subset of properties: add `--properties prop1 --properties prop2`.

**Individual models** (`model_format: "individual"`, `model_file` is a list):

One invocation per model file (existing behavior):

```bash
uv run python ${AUTOMOL_ROOT:-$PWD}/skills/predict/scripts/predict.py \
    --model-file {model_file_path} \
    {--smiles-file INPUT_CSV | --smiles-list "SMILES1" --smiles-list "SMILES2"} \
    --output-folder MolagentFiles/ \
    --verbose
```

Add `--blender-properties {bp}` for each entry in the registry's `blender_properties` array (only for CSV input — the script reads values from the CSV columns).

For CLI SMILES with blender properties, add `--blender-values {prop}={value}` — ask user for values if not provided.

---

## Step 6: Show Results

1. Read the output CSV file (merged: `MolagentFiles/predictions.csv`, individual: `MolagentFiles/{property}_predictions.csv`)
2. Display a table of the first 5-10 predictions
3. Report total count and output file path

```
Predictions complete!

  Model: {id}
  Input: {n} molecules
  Output: {output_path}

Sample predictions:
  {table of first 5 rows}
```

For API serving, see the model card for deployment instructions.

---

## Important Notes

- `${AUTOMOL_ROOT:-$PWD}` resolves to this plugin's root directory (set by SessionStart hook, falls back to `$PWD`)
- Python scripts: `uv run python ...` (no venv activation needed — `uv run` handles dependencies)
- The predict script auto-detects SMILES columns — don't force the training column name on new data
- Blender properties are auto-detected from `train_info.json` if not explicitly provided

## Additional Resources

- **[examples.md](examples.md)** — Usage examples
