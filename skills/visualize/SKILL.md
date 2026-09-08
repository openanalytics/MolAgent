---
name: visualize
description: Generate an interactive HTML dashboard from AutoMol evaluation results. Auto-discovers completed runs with evaluation data and renders Plotly.js charts. Use when the user wants to visualize training results.
allowed-tools: Read, Glob, Bash, AskUserQuestion,
  mcp__plugin_MolAgentLight_automol-mcp__list_models
---

# AutoMol — Visualize

View or regenerate interactive dashboards from completed training runs.

> **Note:** The `train_and_visualize` MCP tool already generates a dashboard as part of training. This skill is for finding and opening existing dashboards, or regenerating them from evaluation data.

---

## Workflow

### Step 1: Find Existing Dashboards

Glob for `${MOLAGENT_OUTPUT_ROOT:-MolagentFiles}/*/dashboard.html`.

- **0 found**: check if evaluation data exists (glob for `*/pipeline_state.json`). If yes → offer to regenerate (Step 3). If no → tell user to train first. STOP.
- **1 found**: auto-select.
- **N found**: present choices via AskUserQuestion (most recent first, show run_id and properties).

### Step 2: Open Dashboard

Open the HTML file in the default browser:

```bash
# Windows
start "" "{dashboard_path}"
# macOS
open "{dashboard_path}"
# Linux
xdg-open "{dashboard_path}"
```

Report the path and what's included (interactive charts, metrics, property selector, outlier detection).

### Step 3: Regenerate (fallback)

If no dashboard exists but a `pipeline_state.json` with completed evaluation (step 5) is available:

```bash
uv run $MOLAGENT_PLUGIN_ROOT/skills/visualize/scripts/generate_dashboard.py \
    --pipeline-state {pipeline_state_path} \
    --output {run_folder}/dashboard.html \
    --verbose
```

Then open the generated file (Step 2).

---

## Notes

- Dashboards are self-contained HTML files — no server needed, works offline after CDN load.
- Only runs with completed evaluation (step 5) can generate dashboards.
- The dashboard also ships as a Nexus playground artifact (`molagent_dashboard` type).
