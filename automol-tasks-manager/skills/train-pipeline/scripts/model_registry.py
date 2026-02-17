#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["click"]
# ///
"""
Model registry for trained AutoMol models.
Tracks model artifacts, metrics, and configuration in a central JSON file.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import click


REGISTRY_FILENAME = "model_registry.json"


def get_registry_path(directory):
    return Path(directory) / REGISTRY_FILENAME


def load_registry(directory):
    path = get_registry_path(directory)
    if not path.exists():
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return []


def save_registry(directory, entries):
    path = get_registry_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)


def generate_model_id(dataset_stem, prop, timestamp, existing_ids):
    ts_str = timestamp.strftime("%Y%m%d-%H%M")
    base_id = f"{dataset_stem}-{prop}-{ts_str}"
    if base_id not in existing_ids:
        return base_id
    n = 2
    while f"{base_id}-{n}" in existing_ids:
        n += 1
    return f"{base_id}-{n}"


@click.group()
def cli():
    """AutoMol model registry."""
    pass


@cli.command()
@click.option("--pipeline-state", required=True, help="Path to pipeline_state.json")
def register(pipeline_state):
    """Register a trained model from pipeline state."""
    state_path = Path(pipeline_state)
    if not state_path.exists():
        print(f"Error: {pipeline_state} not found", file=sys.stderr)
        sys.exit(1)

    with open(state_path) as f:
        state = json.load(f)

    config = state.get("config", {})
    files = state.get("outputs", state.get("files", {}))
    metrics = state.get("metrics", {})
    detection = state.get("detection", {})

    output_folder = config.get("output_folder", "MolagentFiles/")
    # Registry always lives at MolagentFiles/ (global), not inside run folder
    registry_dir = "MolagentFiles/"

    # Detect merged model file first (preferred)
    merged_file = files.get("merged_refitted_model_file") or files.get("merged_model_file")
    if merged_file:
        model_file = merged_file  # single string
        model_format = "merged"
    else:
        model_format = "individual"
        # Support both flat and nested model file schemas
        refitted_files = files.get("refitted_model_files", {})
        model_files_dict = files.get("model_files", {})
        if refitted_files and isinstance(refitted_files, dict):
            model_file = list(refitted_files.values())
        elif model_files_dict and isinstance(model_files_dict, dict):
            model_file = list(model_files_dict.values())
        else:
            model_file = files.get("refitted_model") or files.get("model_file")
            if model_file:
                model_file = [model_file] if isinstance(model_file, str) else model_file
    if not model_file:
        print("Error: no model file found in pipeline state", file=sys.stderr)
        sys.exit(1)

    # Check idempotency — skip if model_file already registered
    registry = load_registry(registry_dir)
    model_file_key = model_file if isinstance(model_file, list) else [model_file]
    for entry in registry:
        existing_mf = entry.get("model_file", [])
        existing_mf = existing_mf if isinstance(existing_mf, list) else [existing_mf]
        if set(existing_mf) == set(model_file_key):
            print(json.dumps({"status": "skipped", "reason": "model already registered", "id": entry["id"]}))
            return

    is_refitted = bool(files.get("refitted_model_files") or files.get("merged_refitted_model_file")) or files.get("refitted_model") is not None

    # Build entry
    data_file = config.get("data_file", "unknown")
    dataset_stem = Path(data_file).stem
    properties = config.get("target_properties", [])
    now = datetime.now()
    existing_ids = {e["id"] for e in registry}

    prop_str = "_".join(properties) if properties else "model"
    model_id = generate_model_id(dataset_stem, prop_str, now, existing_ids)

    # Flatten train_info if nested dict
    train_info = files.get("train_info")
    if isinstance(train_info, dict):
        train_info = list(train_info.values())

    entry = {
        "id": model_id,
        "created_at": now.isoformat(timespec="seconds"),
        "model_file": model_file,
        "model_format": model_format,
        "target_properties": properties,
        "task_type": config.get("task_type", "unknown"),
        "metrics": metrics,
        "feature_keys": config.get("feature_keys", []),
        "smiles_column": config.get("smiles_column", "Stand_SMILES"),
        "blender_properties": config.get("blender_properties", []),
        "source_dataset": data_file,
        "is_refitted": is_refitted,
        "model_card": files.get("model_card"),
        "train_info": train_info,
        "n_samples": detection.get("n_samples") or config.get("n_samples"),
        "computational_load": config.get("computational_load", "moderate"),
        "split_strategy": config.get("split_strategy", "mixed"),
    }

    registry.append(entry)
    save_registry(registry_dir, registry)

    print(json.dumps({"status": "registered", "id": model_id, "registry": str(get_registry_path(registry_dir))}))


@cli.command("list")
@click.option("--format", "fmt", type=click.Choice(["json", "table"]), default="table", help="Output format")
@click.option("--directory", default="MolagentFiles/", help="Directory containing registry")
def list_models(fmt, directory):
    """List all registered models."""
    registry = load_registry(directory)
    if not registry:
        if fmt == "json":
            print("[]")
        else:
            print("No models registered.")
        return

    if fmt == "json":
        print(json.dumps(registry, indent=2))
    else:
        print(f"{'ID':<40} {'Task':<15} {'Properties':<20} {'Metrics':<30} {'Date':<12}")
        print("-" * 117)
        for e in registry:
            props = ", ".join(e.get("target_properties", []))
            metrics_parts = []
            for k, v in e.get("metrics", {}).items():
                if isinstance(v, float):
                    metrics_parts.append(f"{k}={v:.3f}")
                else:
                    metrics_parts.append(f"{k}={v}")
            metrics_str = ", ".join(metrics_parts) if metrics_parts else "N/A"
            date = e.get("created_at", "")[:10]
            print(f"{e['id']:<40} {e.get('task_type', '?'):<15} {props:<20} {metrics_str:<30} {date:<12}")


@cli.command()
@click.argument("model_id")
@click.option("--directory", default="MolagentFiles/", help="Directory containing registry")
def get(model_id, directory):
    """Get full details of a registered model."""
    registry = load_registry(directory)
    for entry in registry:
        if entry["id"] == model_id:
            print(json.dumps(entry, indent=2))
            return
    print(f"Error: model '{model_id}' not found", file=sys.stderr)
    sys.exit(1)


@cli.command()
@click.argument("model_id")
@click.option("--directory", default="MolagentFiles/", help="Directory containing registry")
def remove(model_id, directory):
    """Remove a model entry from the registry (files are not deleted)."""
    registry = load_registry(directory)
    new_registry = [e for e in registry if e["id"] != model_id]
    if len(new_registry) == len(registry):
        print(f"Error: model '{model_id}' not found", file=sys.stderr)
        sys.exit(1)
    save_registry(directory, new_registry)
    print(json.dumps({"status": "removed", "id": model_id}))


if __name__ == "__main__":
    cli()
