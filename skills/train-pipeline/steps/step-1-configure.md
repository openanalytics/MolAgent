# Step 1: Auto-Configure + User Input

**Mode**: Inline

## Philosophy

Auto-detect everything possible from the dataset. Present a complete plan. Ask the user only about what can't be auto-detected. The user reviews the plan and can request changes via "Other" — this is the scope refinement.

## 1.1 — Auto-Fill Defaults

Using the detection JSON from step 0, build the full config:

```python
recs = detection["recommendations"]

# Derive n_jobs from computational_load
n_jobs_map = {
    "free": {"outer": 1, "inner": 1, "method": 1},
    "cheap": {"outer": 1, "inner": 2, "method": 2},
    "moderate": {"outer": -1, "inner": 2, "method": 2},
    "expensive": {"outer": -1, "inner": -1, "method": 2},
}
load = recs["computational_load"]["value"]
n_jobs = n_jobs_map.get(load, {"outer": 2, "inner": 2, "method": 2})

config = {
    "data_file":           detection["file"],
    "smiles_column":       detection["smiles_column"],
    "target_properties":   [t["column"] for t in detection["targets"]],
    "task_type":           detection["overall_task_type"],
    "blender_properties":  [b["column"] for b in detection["blender_properties"]
                            if b["confidence"] == "strong"],
    "feature_keys":        detection["recommended_features"],
    "split_strategy":      recs["split_strategy"]["value"],
    "computational_load":  load,   # ← data-driven, ask user to confirm
    "n_jobs":              n_jobs, # ← derived from computational_load
    "use_advanced":        recs["use_advanced"]["value"],
    "output_folder":       null   # ← set after run folder creation in step 1.5
}
```

Rules:
- **Targets**: include ALL detected numeric columns by default. User can narrow.
- **Blender**: include only `strong` confidence candidates. User can add `moderate` ones.
- **Features**: use detection's recommendation (based on dataset size).
- **Split**: use detection's recommendation (based on dataset size and class balance).
- **Load**: use detection's recommendation — but ask user to confirm (depends on hardware/patience).
- **n_jobs**: derive from computational_load (shown in summary for user awareness).
- **Advanced**: use detection's recommendation. User can opt in.

## 1.2 — Present Auto-Filled Plan

Display the complete plan to the user as text output (NOT as a question):

```
Pipeline Plan (auto-configured from detection):

  Dataset:  {config.data_file} ({n_samples} samples)
  SMILES:   {config.smiles_column}
  Task:     {config.task_type}
  Targets:  {', '.join(config.target_properties)}
  Blender:  {', '.join(config.blender_properties) or 'none'}
  Features: {', '.join(config.feature_keys)}
  Split:    {split_strategy} — {recs.split_strategy.reason}
  Load:     {computational_load} — {recs.computational_load.reason}
  n_jobs:   outer={n_jobs.outer}, inner={n_jobs.inner}, method={n_jobs.method}
  Training: standard — {recs.use_advanced.reason}

  Warnings:
    {each warning as bullet, or "None"}

  Target Transformations:
    {each recommendation as bullet, or "None suggested"}
```

Show each recommendation's reason so the user understands WHY each default was chosen. Display any warnings from `recommendations.warnings` as bullets. Show target transformation suggestions from `recommendations.target_transformations`. This gives the user full visibility before they're asked anything.

## 1.3 — AskUserQuestion (1-2 questions)

Only ask about what can't be auto-detected. Use AskUserQuestion.

### Question 1 — Computational Load

The `(Recommended)` label is placed on whichever option matches `recommendations.computational_load.value`. The question text includes the detection reason. Options are reordered so the recommended option appears first.

```yaml
header: "Load"
question: "Computational budget? Detection suggests {recs.computational_load.value}: {recs.computational_load.reason}"
multiSelect: false
options:
  # Dynamic: recommended option listed FIRST with "(Recommended)" appended
  # Example when detection recommends "cheap":
  - label: "Cheap (Recommended)"
    description: "2-10 min. Minimal ensemble with few estimators and limited search. Good for quick prototyping or verifying the pipeline works before a full run."
  - label: "Free"
    description: "0-2 min. Trains a single LightGBM model for fast signal checking — no ensemble, minimal compute. Best for very small datasets (< 100 samples) or quick feasibility tests."
  - label: "Moderate"
    description: "10-360 min. Trains a stacking ensemble with multiple base estimators (XGBoost, LightGBM, etc.) and moderate hyperparameter search. Good balance of speed and model quality."
  - label: "Expensive"
    description: "1-48 hrs. Full ensemble with all estimators, extensive hyperparameter search (grid or randomized with many iterations), and more CV folds. Maximum model performance."
```

### Question 2 — Scope Review + Execution Approval

This question serves dual purpose: scope refinement AND execution gate. The default option approves and starts execution immediately.

If `recommendations.use_advanced.reason` mentions "consider advanced", surface that hint in the advanced option description.

```yaml
header: "Plan"
question: "Review the plan above. Ready to execute?"
multiSelect: false
options:
  - label: "Looks good, run it (Recommended)"
    description: "Save plan and start execution immediately."
  - label: "Save plan for later"
    description: "Save the plan but don't execute yet. Re-invoke the skill to run."
  - label: "Change targets or features"
    description: "Narrow which properties to model, or change feature set."
  - label: "Use advanced training"
    description: "Gives you control over model hierarchy, estimator lists, CV clustering, and custom Python estimators/features. {recs.use_advanced.reason if it mentions 'consider advanced', else 'Only needed if standard defaults don't fit your use case.'}"
```

**Auto-execute rule**: If the user's original prompt clearly requests training (e.g., "train a model on X.csv", "run the full pipeline on my data"), skip this question entirely — treat it as "Looks good, run it". Only ask Q2 when the user's intent is ambiguous or they explicitly asked to review/plan first. When auto-executing, do use the free computational load. 

This is also the **scope refinement** question. "Other" lets the user type any correction.

## 1.4 — Handle Corrections

If the user selected "Change targets or features" or typed a correction via "Other":

- Parse their request
- Update the config accordingly
- Present the updated plan for confirmation (just show the text again — no need for another AskUserQuestion)

Examples of corrections:
- "Only model Y, not prop2" → remove prop2 from target_properties
- "Use Bottleneck rdkit fps_2048_2" → update feature_keys
- "Use stratified split" → update split_strategy
- "Include Y_noisy as blender" → add to blender_properties

If the user said "Looks good, run it", proceed to saving and then execution.
If the user said "Save plan for later", proceed to saving and then STOP.

## 1.5 — Create Run Folder

Build the run folder name and create it:

```
# Build run ID
props_joined = "_".join(sorted(config["target_properties"]))
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
dataset_stem = Path(config["data_file"]).stem
run_id = f"{dataset_stem}-{props_joined}-{timestamp}"
run_folder = f"MolagentFiles/{run_id}/"

# Create the folder
mkdir -p {run_folder}

# Set config
config["output_folder"] = run_folder
```

## 1.6 — Create Tasks

Create the visible execution task list:

```
TaskCreate(
    subject="Prepare molecular data",
    description="Run data-prep script on {data_file} for {task_type} task",
    activeForm="Preparing molecular data"
)
→ save returned id as task_prepare

TaskCreate(
    subject="Split into train/test",
    description="Split prepared data using {split_strategy} strategy",
    activeForm="Splitting data"
)
→ save as task_split

TaskCreate(
    subject="Train ensemble model",
    description="Train {task_type} model with {feature_keys}, {computational_load} load",
    activeForm="Training ensemble model"
)
→ save as task_train

TaskCreate(
    subject="Evaluate model performance",
    description="Evaluate trained model on test set",
    activeForm="Evaluating model"
)
→ save as task_evaluate

TaskCreate(
    subject="Review results with user",
    description="Present metrics, ask about refit",
    activeForm="Reviewing results"
)
→ save as task_review

TaskCreate(
    subject="Refit model for deployment",
    description="Refit on full dataset for deployment",
    activeForm="Refitting model"
)
→ save as task_refit

TaskCreate(
    subject="Generate model card",
    description="Create model_card.md with usage instructions",
    activeForm="Generating model card"
)
→ save as task_complete
```

Set dependencies:
```
TaskUpdate(task_split,    addBlockedBy=[task_prepare])
TaskUpdate(task_train,    addBlockedBy=[task_split])
TaskUpdate(task_evaluate, addBlockedBy=[task_train])
TaskUpdate(task_review,   addBlockedBy=[task_evaluate])
TaskUpdate(task_refit,    addBlockedBy=[task_review])
TaskUpdate(task_complete, addBlockedBy=[task_refit])
```

## 1.7 — Save pipeline_state.json

Write the full state to `{run_folder}/pipeline_state.json` (inside the run folder, NOT at `MolagentFiles/pipeline_state.json`):

```json
{
  "pipeline_version": "2.0",
  "run_id": "{run_id}",
  "created_at": "<ISO timestamp>",
  "last_updated": "<ISO timestamp>",
  "steps_completed": [0, 1],
  "current_step": 2,
  "pipeline_complete": false,
  "task_ids": {
    "prepare": "<id>", "split": "<id>", "train": "<id>",
    "evaluate": "<id>", "review": "<id>", "refit": "<id>",
    "complete": "<id>"
  },
  "detection": { "<full detection JSON from step 0>" },
  "config": { "<final config after user input, including output_folder>" },
  "outputs": {
    "prepared_csv": null, "prepared_info": null,
    "split_csv": null, "split_info": null,
    "model_files": {}, "merged_model_file": null,
    "train_info": {},
    "evaluation_results": {},
    "refitted_model_files": {}, "merged_refitted_model_file": null,
    "model_card": null
  },
  "metrics": {},
  "decisions": { "refit_approved": null, "include_test_in_refit": null },
  "errors": []
}
```

## 1.8 — Present Summary and Transition

Show the final saved plan:

```
Plan saved to {run_folder}/pipeline_state.json

  Run ID:   {run_id}
  Dataset:  {data_file} ({n_samples} samples)
  Targets:  {properties} ({task_type})
  Features: {feature_keys}
  Load:     {computational_load}
```

Then, based on user's Q2 answer (or auto-execute):

**"Looks good, run it"** or **auto-execute** → show "Starting execution..." and proceed directly to Phase 2 (step 2). Do NOT stop between phases.

**"Save plan for later"** → show the following and **STOP**:
```
  7 execution steps queued. To run:
    → re-invoke this skill, or say "run the pipeline"
    → to modify: tell me what to change before executing
```

### Auto-execute detection

Skip Q2 entirely and proceed as "run it" when the user's original prompt clearly requests training. Examples:
- "train a model on X.csv" → auto-execute
- "run the full pipeline" → auto-execute
- "train my data with cheap load" → auto-execute (also pre-fills Q1 answer)

Only ask Q2 when:
- User said "plan", "configure", or "set up" (intent is planning only)
- User's prompt is ambiguous (e.g., just a file path with no verb)
- There are **warnings** in the detection output (always pause and show them)
