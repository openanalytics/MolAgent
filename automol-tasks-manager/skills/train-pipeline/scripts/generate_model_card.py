#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["click"]
# ///
"""
Generate a model card from pipeline_state.json.
Creates model_card.md with model details, metrics, usage, and deployment instructions.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import click


def load_json_safe(path):
    """Load JSON file, return empty dict on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def format_metrics_table(metrics):
    """Format metrics dict as markdown table. Handles nested per-property metrics."""
    if not metrics:
        return "_No metrics available._"
    # Check if metrics are nested per-property (e.g. {"gamma1": {"r2": -0.14}})
    first_val = next(iter(metrics.values()), None)
    if isinstance(first_val, dict):
        # Per-property metrics
        all_metric_keys = set()
        for prop_metrics in metrics.values():
            all_metric_keys.update(prop_metrics.keys())
        metric_keys = sorted(all_metric_keys)
        header = "| Property | " + " | ".join(k.upper() for k in metric_keys) + " |"
        sep = "|----------|" + "|".join("-------" for _ in metric_keys) + "|"
        lines = [header, sep]
        for prop, prop_metrics in metrics.items():
            vals = []
            for k in metric_keys:
                v = prop_metrics.get(k, "N/A")
                vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            lines.append(f"| {prop} | " + " | ".join(vals) + " |")
        return "\n".join(lines)
    else:
        # Flat metrics
        lines = ["| Metric | Value |", "|--------|-------|"]
        for key, val in metrics.items():
            if isinstance(val, float):
                lines.append(f"| {key} | {val:.4f} |")
            else:
                lines.append(f"| {key} | {val} |")
        return "\n".join(lines)


@click.command()
@click.option('--pipeline-state', required=True, help='Path to pipeline_state.json')
def main(pipeline_state):
    """Generate model_card.md from pipeline state."""
    state_path = Path(pipeline_state)
    if not state_path.exists():
        print(f"Error: {pipeline_state} not found", file=sys.stderr)
        sys.exit(1)

    state = load_json_safe(state_path)
    config = state.get("config", {})
    outputs = state.get("outputs", state.get("files", {}))
    metrics = state.get("metrics", {})
    decisions = state.get("decisions", {})

    # Determine best model file (supports both flat and nested schemas)
    refitted = outputs.get("refitted_model_files", {})
    model_files_dict = outputs.get("model_files", {})
    if refitted:
        model_file = ", ".join(refitted.values()) if isinstance(refitted, dict) else refitted
        model_type = "Refitted"
    elif model_files_dict:
        model_file = ", ".join(model_files_dict.values()) if isinstance(model_files_dict, dict) else model_files_dict
        model_type = "Standard"
    else:
        model_file = outputs.get("refitted_model") or outputs.get("model_file") or "N/A"
        model_type = "Refitted" if outputs.get("refitted_model") else "Standard"
    # For backward compat, expose outputs as files
    files = outputs

    # Target properties
    properties = config.get("target_properties", ["unknown"])
    properties_str = ", ".join(properties)
    task_type = config.get("task_type", "unknown")

    # Feature keys
    feature_keys = config.get("feature_keys", [])
    features_str = ", ".join(feature_keys) if feature_keys else "default"

    # Computational load
    comp_load = config.get("computational_load", "unknown")

    # Training mode
    training_mode = "Advanced" if config.get("use_advanced") else "Standard"

    # Split strategy
    split_strategy = config.get("split_strategy", "mixed")

    # Plugin root (for path construction in usage examples)
    plugin_root = "${PLUGIN_ROOT}"

    # Build model card
    card = f"""# Model Card: {properties_str} Prediction Model

_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_

## Model Details

| Field | Value |
|-------|-------|
| Model file | `{model_file}` |
| Model type | {model_type} ensemble (stacking) |
| Task type | {task_type} |
| Target properties | {properties_str} |
| Features | {features_str} |
| Training mode | {training_mode} |
| Computational load | {comp_load} |
| Split strategy | {split_strategy} |

## Model Registry

This model is registered in `MolagentFiles/model_registry.json`. Use the `predict` skill to auto-discover and run inference with it.

## Performance Metrics

{format_metrics_table(metrics)}

## How to Use This Model

### Make Predictions (predict skill)

```bash
source ${{AUTOMOL_VENV:-.venv}}/bin/activate
uv run python {plugin_root}/skills/predict/scripts/predict.py \\
    --model-file {model_file} \\
    --csv-file your_molecules.csv \\
    --smiles-column SMILES
```

Or invoke the `predict` skill in Claude Code:
> "Use the predict skill to make predictions with {model_file} on my new data"

### Deploy as REST API

```bash
source ${{AUTOMOL_VENV:-.venv}}/bin/activate
MODEL_PATH={model_file} python -m deploy.server.app --port 8000
```

Then query via `curl` or the Python client:

```python
from deploy.client import AutoMolClient
client = AutoMolClient("http://localhost:8000")
result = client.predict("CCO")  # -> {{{properties_str}: value}}
```

### Deploy with Docker

```bash
docker run -p 8000:8000 -v $(pwd)/MolagentFiles:/models:ro \\
    -e MODEL_PATH=/models/{Path(model_file).name if model_file != 'N/A' else 'model.pt'} \\
    automol-server
```

For full deployment instructions (Docker Compose, GPU, nginx, environment variables), see **[deploy/DEPLOYMENT.md](../deploy/DEPLOYMENT.md)**.

## Training Configuration

```json
{json.dumps(config, indent=2)}
```

## Pipeline Artifacts

| Artifact | Path |
|----------|------|
| Prepared data | `{files.get('prepared_csv', 'N/A')}` |
| Split data | `{files.get('split_csv', 'N/A')}` |
| Model file | `{files.get('model_file', 'N/A')}` |
| Training info | `{files.get('train_info', 'N/A')}` |
| Evaluation report | `{files.get('eval_pdf', 'N/A')}` |
| Refitted model | `{files.get('refitted_model', 'N/A')}` |

## Decisions

| Decision | Value |
|----------|-------|
| Refit approved | {decisions.get('refit_approved', 'N/A')} |
| Include test in refit | {decisions.get('include_test_in_refit', 'N/A')} |
"""

    # Write model card
    output_folder = Path(config.get("output_folder", "MolagentFiles/"))
    output_folder.mkdir(parents=True, exist_ok=True)
    card_path = output_folder / "model_card.md"
    card_path.write_text(card)

    # Update pipeline state with card path
    outputs_key = "outputs" if "outputs" in state else "files"
    if outputs_key not in state:
        state[outputs_key] = {}
    state[outputs_key]["model_card"] = str(card_path)
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)

    print(json.dumps({"model_card": str(card_path)}, indent=2))


if __name__ == "__main__":
    main()
