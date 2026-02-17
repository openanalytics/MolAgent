---
name: train-pipeline
description: Plan and execute an AutoMol training pipeline. Analyzes dataset, auto-configures a plan, asks for approval, then executes — all in one invocation. Re-invoke to resume if interrupted. Use when the user wants to train a complete model from raw data.
allowed-tools: Read, Glob, Grep, Bash, Write, Edit, AskUserQuestion,
  TaskCreate, TaskUpdate, TaskList, TaskGet, Task, TaskOutput
hooks:
  Stop:
    - hooks:
        - type: command
          command: >-
            uv run ${AUTOMOL_ROOT:-$PWD}/skills/train-pipeline/validators/validate_pipeline_state.py
            --directory MolagentFiles
---

# AutoMol V3 — Training Pipeline

Single-invocation skill: **PLAN** → **approve** → **EXECUTE** in one session.

- Phase 1 analyzes the dataset, auto-fills a plan, asks the user to confirm and approve execution, creates a run folder, saves `pipeline_state.json` inside it, then proceeds directly to execution.
- If the user declines execution ("save for later"), the plan is saved and the skill stops. Re-invoking the skill resumes from the saved state.
- **Auto-execute**: if the user's original prompt clearly requests training (e.g., "train a model on X.csv", "run the pipeline"), skip the approval question and proceed directly to execution after saving.

The `Stop` hook enforces that `pipeline_state.json` exists with valid config before the skill can finish — this prevents the LLM from stopping before the plan is actually written.

### Why not Claude Code plan mode?

Phase 1 looks like "planning" but Claude Code plan mode disables Write/Edit tools. Phase 1 *must* write `pipeline_state.json`, run the detection script via Bash, and create tasks. The skill achieves the same plan-then-approve flow using AskUserQuestion — the user sees the full plan and explicitly approves before execution begins.

---

## Routing Logic

```
1. Glob MolagentFiles/*/pipeline_state.json
2. Filter: keep only where pipeline_complete = false
3. CASE: 0 incomplete runs found → PHASE 1 (fresh start)
4. CASE: 1 incomplete run found → PHASE 2 (resume from that run folder)
5. CASE: multiple incomplete → AskUserQuestion which to resume, or start fresh
6. CASE: all runs complete → AskUserQuestion: start fresh?
```

Read [steps/continue-handler.md](steps/continue-handler.md) for detailed resume logic.

---

## PHASE 1: Think & Plan

Goal: analyze dataset, auto-configure sensible defaults, present "thinking" reasoning, get user input on what can't be auto-detected, create a run folder, save `pipeline_state.json`, then proceed to execution (or stop if user declines).

### Step 0 — Detect

Run the detection script:

```bash
uv run python ${AUTOMOL_ROOT:-$PWD}/skills/train-pipeline/scripts/detect_dataset.py \
    --csv-file {user_data_file}
```

Capture the JSON output. It contains: `smiles_column`, `overall_task_type`, `targets` (with stats), `blender_properties`, `recommended_features`, `n_samples`, `characteristics`, and `recommendations`.

**Think and Plan**: Present reasoning to the user after detection:

```
Analyzing {data_file}...

  Found {n_samples} molecules with {n_targets} target properties
  SMILES column: {smiles_column}
  Task type: {task_type}
  Recommended features: {feature_keys} — {reason}
  Recommended load: {load} — {reason}

  Target Distribution:
    {for each regression target:}
    {target}: skewness={skewness}, {if skewed: "⚠️ suggests log10 transform" or "⚠️ suggests Yeo-Johnson"}
```

### Step 1 — Auto-Configure + User Input

Read [steps/step-1-configure.md](steps/step-1-configure.md) for full instructions.

**1.1 — Auto-fill defaults from detection:**

Using the detection output, build a complete config with sensible defaults:

| Field | Auto-fill Rule |
|-------|---------------|
| `smiles_column` | From detection |
| `target_properties` | All detected numeric targets |
| `task_type` | From detection (`overall_task_type`) |
| `blender_properties` | Strong-confidence candidates only |
| `feature_keys` | From `recommended_features` |
| `split_strategy` | From `recommendations.split_strategy.value` |
| `computational_load` | From `recommendations.computational_load.value` (ask user to confirm) |
| `use_advanced` | From `recommendations.use_advanced.value` |

**1.2 — Present the auto-filled plan:**

Show the user the complete plan before asking anything:

```
Pipeline Plan (auto-configured from detection):

  Dataset: {data_file} ({n_samples} samples)
  SMILES: {smiles_column}
  Task: {task_type}
  Targets: {target_properties}
  Blender: {blender_properties or "none"}
  Features: {feature_keys}
  Split: {split_strategy} — {recommendations.split_strategy.reason}
  Load: {computational_load} — {recommendations.computational_load.reason}
  n_jobs: outer={n_jobs.outer}, inner={n_jobs.inner}, method={n_jobs.method}
  Training: standard — {recommendations.use_advanced.reason}

  Warnings:
    {each warning as bullet, or "None"}

  Target Transformations:
    {each recommendation as bullet, or "None suggested"}

  Steps: prepare → split → train → evaluate → review → refit → model card
```

**1.3 — Ask user ONE focused question:**

Use AskUserQuestion with 1-2 questions. Only ask about things that can't be auto-detected.

```
AskUserQuestion:
  question 1:
    header: "Load"
    question: "Computational budget? Detection suggests {value}: {reason}"
    options:
      # Dynamic: recommended option listed FIRST with "(Recommended)" appended
      - "{recommended_load} (Recommended) - {reason}. {time_estimate}."
      - "Free - 0-2 min. Single LightGBM model for fast signal checking."
      - "Cheap - 2-10 min. Minimal ensemble for quick prototyping."
      - "Moderate - 10-360 min. Stacking ensemble, good balance of speed and quality."
      - "Expensive - 1-48 hrs. Full ensemble with extensive hyperparameter search."

  question 2:
    header: "Plan"
    question: "Review the plan above. Ready to execute?"
    options:
      - "Looks good, run it (Recommended) - Save plan and start execution immediately."
      - "Save plan for later - Save but don't execute. Re-invoke the skill to run."
      - "Change targets or features - Modify the plan before deciding."
      - "Use advanced training - {conditional: if reason mentions 'consider advanced', show that; else show default text}"
```

**Auto-execute rule**: If the user's original prompt clearly requests training (e.g., "train a model on X.csv", "run the full pipeline"), skip Question 2 entirely — treat it as "Looks good, run it". Only ask Q2 when the user's intent is ambiguous or they explicitly asked to review/plan first.

If the user selects "Change targets or features" or uses "Other", handle their corrections inline, show the updated plan, and re-ask Q2 for execution approval.

**1.4 — Create run folder:**

```
run_id = f"{dataset_stem}-{props_joined}-{YYYYMMDD}_{HHMM}"
run_folder = f"MolagentFiles/{run_id}/"
mkdir -p {run_folder}
config.output_folder = run_folder
```

**1.5 — Create execution tasks:**

```
TaskCreate("Prepare molecular data", ...)         → id_prepare
TaskCreate("Split into train/test", ...)           → id_split
TaskCreate("Train ensemble model", ...)            → id_train
TaskCreate("Evaluate model performance", ...)      → id_evaluate
TaskCreate("Review results with user", ...)        → id_review
TaskCreate("Refit model for deployment", ...)      → id_refit
TaskCreate("Generate model card", ...)             → id_complete

Set blockedBy chains: split←prepare, train←split, evaluate←train, etc.
```

**1.6 — Save `pipeline_state.json`:**

Write the complete state to `{run_folder}/pipeline_state.json`. See [reference.md](reference.md) for the full schema. Key fields:
- `pipeline_version`: `"2.0"`
- `run_id`: matches the folder name
- `current_step`: `2`
- `steps_completed`: `[0, 1]`
- `config`: full config object (includes `output_folder`)
- `outputs`: initialized with all null values
- `task_ids`: mapping from step names to task IDs

**1.7 — Summarize and transition:**

```
Plan saved to {run_folder}/pipeline_state.json

  Run ID:   {run_id}
  Dataset:  {data_file} ({n_samples} samples)
  Targets:  {properties} ({task_type})
  Features: {feature_keys}
  Load:     {computational_load}
  n_jobs:   outer={n_jobs.outer}, inner={n_jobs.inner}, method={n_jobs.method}
```

Then, based on Q2 answer (or auto-execute):
- **"Looks good, run it"** or **auto-execute** → proceed directly to PHASE 2
- **"Save plan for later"** → display "Re-invoke the skill or say 'run the pipeline' to execute" and **STOP**

---

## PHASE 2: Execute

Entered directly after Phase 1 approval, or when re-invoked and an incomplete run folder is found.

### Dispatch Loop

For each step from `current_step` to 8:

1. Read `{run_folder}/pipeline_state.json` to get current state
2. Look up `task_ids[step_name]` for the current step
3. `TaskUpdate(taskId=task_id, status="in_progress")` — shows spinner to user

**Then dispatch based on step type:**

| Step | Mode | File | Condition |
|------|------|------|-----------|
| 2 | Task (general-purpose) | [step-2-prepare.md](steps/step-2-prepare.md) | Always |
| 3 | Task (general-purpose) | [step-3-split.md](steps/step-3-split.md) | Always |
| 4 | Task (general-purpose) | [step-4-train.md](steps/step-4-train.md) | Always |
| 5 | Task (general-purpose) | [step-5-evaluate.md](steps/step-5-evaluate.md) | Skip if `use_advanced` |
| 6 | **Inline** | [step-6-review.md](steps/step-6-review.md) | Always (user checkpoint) |
| 7 | Task (general-purpose) | [step-7-refit.md](steps/step-7-refit.md) | Always (skip only if user explicitly declines) |
| 8 | **Inline** | [step-8-complete.md](steps/step-8-complete.md) | Always |

**Subagent dispatch pattern (steps 2, 3, 4, 5, 7):**

```
step_content = Read(steps/step-{N}-{name}.md)
state_json = Read({run_folder}/pipeline_state.json)

Task(
  description="Step {N}: {name}",
  subagent_type="general-purpose",
  model="sonnet",
  prompt="You are executing step {N} of the AutoMol training pipeline.

Run folder: {run_folder}
Pipeline state:
{state_json}

Instructions:
{step_content}

Critical Rules:
- Scripts: uv run python ${AUTOMOL_ROOT:-$PWD}/skills/train-pipeline/scripts/...
- State key: 'outputs' (NOT 'files')
- Paths: Path(dir) / filename (NEVER f-string concat)
- Read/update/write pipeline_state.json in the run folder
- Print summary when done"
)
```

4. After subagent returns, read the updated `{run_folder}/pipeline_state.json`
5. Print user-facing summary (see Summary Templates below)
6. `TaskUpdate(taskId=task_id, status="completed")`
7. `TaskList()` — confirms next task is unblocked

**Inline steps** (6, 8) run in the main context because they need AskUserQuestion or display output.

### Summary Templates

Print after each step completes:

**Step 2 — Prepare:**
```
Data prepared: {prepared_csv}
  {n_rows} molecules processed
```

**Step 3 — Split:**
```
Data split ({split_strategy}):
  Train: {n_train} | Valid: {n_valid} | Test: {n_test}
```

**Step 4 — Train:**
```
Model trained:
  {prop1}: {model_file}
  {prop2}: {model_file}
  Merged: {merged or "N/A"}
```

**Step 5 — Evaluate:**
```
Evaluation results:
  {prop1}: R2={r2}, RMSE={rmse}, MAE={mae}
  {prop2}: R2={r2}, RMSE={rmse}, MAE={mae}
```

**Step 7 — Refit:**
```
Model refitted for deployment:
  {prop1}: {refitted_model_file}
  Merged: {merged or "N/A"}
```

**Step 8 — Complete:**
```
Pipeline Complete!
  Run folder: {run_folder}
  Model ID: {model_id}
  Registry: MolagentFiles/model_registry.json
  Next: use 'predict' skill to make predictions
```

### Error Handling

If a subagent step fails:

```
AskUserQuestion:
  header: "Error"
  question: "Step {N} ({name}) failed: {error_summary}. How to proceed?"
  options:
    - "Retry (Recommended) - Re-run the failed step"
    - "Skip - Mark as done and continue to next step"
    - "Abort - Stop the pipeline (can resume later)"
```

- **Retry**: re-dispatch the same `Task()`
- **Skip**: `TaskUpdate(taskId, status="completed")`, continue
- **Abort**: leave task in_progress, STOP

---

## Important Notes

- `${AUTOMOL_ROOT:-$PWD}` resolves to the plugin root directory (set by SessionStart hook, falls back to `$PWD`)
- Python scripts: `uv run python ...`
- State persists in `{run_folder}/pipeline_state.json` — each run is isolated
- The registry at `MolagentFiles/model_registry.json` is global (indexes all runs)
- Prediction is NOT part of this pipeline — use the `predict` skill after

## Additional Resources

- **[reference.md](reference.md)** — State schema, CLI reference, run folder convention
- **[examples.md](examples.md)** — Usage examples
- **[steps/](steps/)** — Detailed step instructions
