# Continue Handler: Execute or Resume Pipeline

**Mode**: Inline (orchestrator logic)

## Purpose

This handler covers pipeline resumption when re-invoking the skill. It scans for existing run folders and determines the right action.

## Run Folder Scanning

Scan for existing pipeline states:

```
1. Glob MolagentFiles/*/pipeline_state.json
2. For each found, read and check pipeline_complete
3. Categorize into: complete runs vs incomplete runs
```

## Routing Cases

### Case 1 — No runs found
No `MolagentFiles/*/pipeline_state.json` exists at all.
→ Start fresh: PHASE 1

### Case 2 — One incomplete run
Exactly one run has `pipeline_complete = false`.
→ Resume that run from its `current_step`

Display:
```
Found incomplete pipeline run:
  Run ID:   {run_id}
  Dataset:  {data_file}
  Steps completed: {steps_completed}
  Next step: {current_step} - {step_name}
```

AskUserQuestion:
```
header: "Resume"
question: "Resume this pipeline run?"
options:
  - "Resume (Recommended) - Continue from step {current_step}"
  - "Start fresh - Begin a new pipeline run"
```

### Case 3 — Multiple incomplete runs
More than one run has `pipeline_complete = false`.
→ Ask user which to resume

AskUserQuestion:
```
header: "Resume"
question: "Multiple incomplete runs found. Which to resume?"
options:
  - "{run_id_1} (step {current_step_1})"
  - "{run_id_2} (step {current_step_2})"
  - "Start fresh - Begin a new pipeline run"
```

### Case 4 — All runs complete
All found runs have `pipeline_complete = true`.
→ Offer to start fresh

```
All previous pipeline runs are complete.
```

AskUserQuestion:
```
header: "New run"
question: "Start a new pipeline run?"
options:
  - "Yes, start fresh (Recommended)"
  - "No, cancel"
```

## Step Name Map

| Step | Name |
|------|------|
| 0 | Detect dataset |
| 1 | Configure pipeline |
| 2 | Prepare data |
| 3 | Split data |
| 4 | Train model |
| 5 | Evaluate model |
| 6 | Review results |
| 7 | Refit model |
| 8 | Complete pipeline |

## Resume Logic

If user chooses resume:
- Load the run folder path from `config.output_folder`
- Load all existing config and file paths from state
- Recreate task list if task IDs are stale (tasks from previous session may not exist)
- Enter PHASE 2 dispatch loop starting from `current_step`

If user chooses start fresh:
- Proceed to PHASE 1 (new detection, new run folder)
- Previous run folders are NOT deleted — they remain as historical records

## Recreating Tasks on Resume

When resuming, previously completed tasks don't need to be recreated. Only create tasks for remaining steps:

```
For step in range(current_step, 9):
    if step not already completed:
        TaskCreate(step_name, ...)

Set blockedBy dependencies between the new tasks.
Update task_ids in pipeline_state.json.
```

This ensures the task list shows only remaining work.

## Error Recovery

If the state file is corrupted:
- Inform user: "Pipeline state appears corrupted"
- Offer to start fresh
- Don't attempt partial recovery from corrupt state
