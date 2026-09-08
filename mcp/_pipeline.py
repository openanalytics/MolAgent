"""Core pipeline logic: orchestrates MolAgentLight scripts as subprocesses.

Each step calls the same scripts used by the train-pipeline skill, so all
battle-tested data handling and model logic is reused without duplication.

Usage as CLI (for testing):
    uv run mcp/_pipeline.py --config config.json
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from _config import TrainingConfig, TrainingResult

_active_run_folders: set[str] = set()
_active_folders_lock = asyncio.Lock()


def _plugin_root() -> Path:
    """Resolve MOLAGENT_PLUGIN_ROOT → this file's grandparent as fallback."""
    env = os.environ.get("MOLAGENT_PLUGIN_ROOT")
    return Path(env).resolve() if env else Path(__file__).resolve().parent.parent


def _scripts_dir() -> Path:
    return _plugin_root() / "skills" / "train-pipeline" / "scripts"


def _visualize_script() -> Path:
    return _plugin_root() / "skills" / "visualize" / "scripts" / "generate_dashboard.py"


def _output_root() -> Path:
    root = os.environ.get("MOLAGENT_OUTPUT_ROOT")
    # Always resolve to an absolute path so subprocesses launched from any CWD
    # see the same location.
    return (Path(root) if root else _plugin_root() / "MolagentFiles").resolve()


def _under_output_root(p: Path) -> Path:
    """Resolve p and verify it is under _output_root(). Raises ValueError if not."""
    rp = Path(p).resolve()
    rp.relative_to(_output_root())   # raises ValueError if outside
    return rp


def _venv_path() -> Path:
    """Resolve the plugin venv: AUTOMOL_VENV > MOLAGENT_PLUGIN_ROOT/.venv."""
    env = os.environ.get("AUTOMOL_VENV")
    if env:
        return Path(env)
    return _plugin_root() / ".venv"


def _run_script_sync(script: Path, args: list[str], cwd: Optional[Path] = None) -> str:
    """Run a pipeline script with uv and return its stdout (blocking).

    Raises RuntimeError on non-zero exit with stderr message.
    """
    scripts_dir = _scripts_dir()
    cmd = ["uv", "run", "--active", "--no-sync", str(script)] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=str(cwd or scripts_dir),
        env={
            **os.environ,
            "VIRTUAL_ENV": str(_venv_path()),
            "PYTHONPATH": str(scripts_dir),
            # Always inject an absolute, resolved output root so PEP 723 scripts
            # (which ignore --active/--no-sync and may run with a different CWD)
            # write to the same location the MCP server reads from.
            "MOLAGENT_OUTPUT_ROOT": str(_output_root()),
        },
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Script {script.name} failed (exit {result.returncode}):\n{result.stderr}"
        )
    return result.stdout


async def _run_script(script: Path, args: list[str], cwd: Optional[Path] = None) -> str:
    """Run a pipeline script without blocking the event loop."""
    return await asyncio.to_thread(_run_script_sync, script, args, cwd)


def _make_run_folder(config: TrainingConfig) -> Path:
    if config.output_folder:
        folder = _under_output_root(Path(config.output_folder))
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    csv_stem = Path(config.csv_file).stem
    props_joined = "_".join(config.properties[:3])  # cap at 3 for readability
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_id = f"{csv_stem}-{props_joined}-{timestamp}"
    folder = _output_root() / run_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _run_id_from_folder(folder: Path) -> str:
    return folder.name


_TASK_SLUG = {
    "Regression": "regression",
    "Classification": "classification",
    "RegressionClassification": "regression_classification",
}


def _task_slug(config: TrainingConfig) -> str:
    return _TASK_SLUG[config.task]


def _task_script(config: TrainingConfig, prefix: str, suffix: str = "") -> Path:
    """Map task + prefix to the right script filename.

    E.g. prefix="train", suffix="" → train_regression.py
         prefix="evaluate", suffix="_model" → evaluate_regression_model.py
    """
    return _scripts_dir() / f"{prefix}_{_task_slug(config)}{suffix}.py"


def _build_train_args(config: TrainingConfig, split_csv: Path, run_folder: Path) -> tuple[Path, list[str]]:
    """Build the train script path and CLI args for pipeline step 3 (train)."""
    train_suffix = "_advanced" if config.use_advanced else ""
    train_script = _task_script(config, "train", suffix=train_suffix)
    train_args = [
        "--csv-file", str(split_csv),
        "--output-folder", str(run_folder),
        "--computational-load", config.computational_load,
        "--outer-folds", str(config.outer_folds),
        "--random-state", str(config.random_state),
        "--n-jobs-inner", str(config.n_jobs_inner),
        "--verbose",
    ]
    for k in config.feature_keys:
        train_args += ["--feature-keys", k]
    if config.n_jobs_outer is not None:
        train_args += ["--n-jobs-outer", str(config.n_jobs_outer)]
    if config.scorer:
        train_args += ["--scorer", config.scorer]
    if config.base_list:
        for b in config.base_list:
            train_args += ["--base-list", b]
    if config.blender_list:
        for b in config.blender_list:
            train_args += ["--blender-list", b]
    if config.red_dim_list:
        for r in config.red_dim_list:
            train_args += ["--red-dim-list", r]
    if config.ensemble_config:
        train_args += ["--model-config", config.ensemble_config]
    if config.search_type:
        train_args += ["--search-type", config.search_type]
    if config.randomized_iterations is not None:
        train_args += ["--randomized-iterations", str(config.randomized_iterations)]
    if config.computational_load == 'expensive' and config.random_state_list:
        for rs in config.random_state_list:
            train_args += ["--random-state-list", str(rs)]
    if config.sdf_file:
        train_args += ["--sdf-file", config.sdf_file]
    if config.protein_folder:
        train_args += ["--protein-folder", config.protein_folder]
    return train_script, train_args


async def run_full_pipeline(
    config: TrainingConfig,
    progress_cb: Callable[[int, int], Any] | None = None,
    owner: str | None = None,
) -> TrainingResult:
    """Run the full AutoMol pipeline and return a TrainingResult.

    Args:
        config: Training configuration.
        progress_cb: Optional callback for progress reporting.
        owner: Optional user_id to tag the model in the registry.

    Steps:
      1  prepare    → automol_prepared_*.csv
      2  split      → automol_split_*.csv
      3  train      → per-property .pt files
      4  [merge]    → merged_stackingregmodel.pt  (if >1 property)
      5  evaluate   → evaluation_predictions.csv per property
      6  [refit]    → refitted .pt files
      7  [merge]    → merged_refitted_stackingregmodel.pt
      8  dashboard  → dashboard.html
    """
    total_steps = 8
    run_folder = _make_run_folder(config)
    if config.output_folder:
        _under_output_root(run_folder)  # raises ValueError if outside output root
    folder_str = str(run_folder.resolve())
    async with _active_folders_lock:
        _active_run_folders.add(folder_str)
    try:
        run_id = _run_id_from_folder(run_folder)

        csv_stem = Path(config.csv_file).stem
        scripts = _scripts_dir()

        async def _progress(step: int, total: int) -> None:
            if progress_cb is None:
                return
            result = progress_cb(step, total)
            if asyncio.iscoroutine(result):
                await result

        # ── Step 1: prepare ──────────────────────────────────────────────────────
        await _progress(1, total_steps)
        # RegressionClassification uses classification prep (targets are already binary 0/1)
        if config.task == "RegressionClassification":
            prepare_script = _scripts_dir() / "prepare_for_classification.py"
        else:
            prepare_script = _task_script(config, "prepare_for")
        prepare_args = [
            "--csv-file", config.csv_file,
            "--smiles-column", config.smiles_column,
            "--output-folder", str(run_folder),
            "--separator", config.sep,
            "--verbose",
        ]
        for p in config.properties:
            prepare_args += ["--properties", p]
        if config.task == "Regression":
            if config.use_log10:
                prepare_args.append("--use-log10")
            if config.use_logit:
                prepare_args.append("--use-logit")
        if config.task in ("Classification", "RegressionClassification"):
            if config.categorical:
                prepare_args.append("--categorical")
            if config.nb_classes:
                for nc in config.nb_classes:
                    prepare_args += ["--nb-classes", str(nc)]
            if config.class_values:
                # class_values may arrive as a nested list (one inner list per property)
                # or as a flat list; flatten to individual float tokens for the CLI.
                flat_cv = []
                for cv in config.class_values:
                    if isinstance(cv, list):
                        flat_cv.extend(cv)
                    else:
                        flat_cv.append(cv)
                for cv in flat_cv:
                    prepare_args += ["--class-values", str(cv)]
            if config.class_quantiles:
                flat_cq = []
                for cq in config.class_quantiles:
                    if isinstance(cq, list):
                        flat_cq.extend(cq)
                    else:
                        flat_cq.append(cq)
                for cq in flat_cq:
                    prepare_args += ["--class-quantiles", str(cq)]
        if config.blender_properties:
            for bp in config.blender_properties:
                prepare_args += ["--blender-properties", bp]
        if config.use_sample_weight:
            prepare_args.append("--use-sample-weight")
            if config.sample_weight_selection:
                for _ in config.properties:
                    prepare_args += ["--property-selection", config.sample_weight_selection]
            if config.sample_weight_multiplier is not None:
                for _ in config.properties:
                    prepare_args += ["--property-weight", str(int(config.sample_weight_multiplier))]
        await _run_script(prepare_script, prepare_args)

        prepared_csv = run_folder / f"automol_prepared_{csv_stem}.csv"

        # ── Step 2: split ─────────────────────────────────────────────────────────
        await _progress(2, total_steps)
        split_args = [
            "--csv-file", str(prepared_csv),
            "--output-folder", str(run_folder),
            "--split-strategy", config.split_strategy,
            "--test-size", str(config.test_size),
            "--random-state", str(config.random_state),
            "--clustering-method", config.clustering_method,
            "--n-clusters", str(config.n_clusters),
            "--butina-cutoff", str(config.butina_cutoff),
            "--add-test-set",
            "--verbose",
        ]
        # When log10/logit transformation was applied, the prepared CSV contains
        # log10_{p} / logit_{p} columns. The split script must receive the actual names.
        if config.use_log10:
            split_properties = [f"log10_{p}" for p in config.properties]
        elif config.use_logit:
            split_properties = [f"logit_{p}" for p in config.properties]
        else:
            split_properties = list(config.properties)
        for p in split_properties:
            split_args += ["--properties", p]
        await _run_script(scripts / "split_data.py", split_args)

        split_csv = run_folder / f"automol_split_{csv_stem}.csv"

        # ── Step 3: train ─────────────────────────────────────────────────────────
        await _progress(3, total_steps)
        train_script, train_args = _build_train_args(config, split_csv, run_folder)
        await _run_script(train_script, train_args)

        # ── Locate per-property train info files (written by train script) ──────────
        # The train script may rename the property (e.g. "Y" → "Class_Y" for
        # classification binarization), so check both the bare name and Class_ prefix.
        # When use_log10 is active, the train script sees "log10_{p}" as the property
        # name, so files are written as "log10_{p}_train_info.json". We map back to the
        # original config.properties key for the rest of the pipeline.
        train_info_files: dict[str, Path] = {}
        for orig, sp in zip(config.properties, split_properties):
            for candidate in (f"{sp}_train_info.json", f"Class_{sp}_train_info.json"):
                ti = run_folder / candidate
                if ti.exists():
                    train_info_files[orig] = ti
                    break

        # ── Locate per-property model files ──────────────────────────────────────
        # Read the actual filename from train_info (most reliable); fall back to
        # the conventional naming pattern if the JSON isn't available.
        task_suffix = "reg" if config.task == "Regression" else "clf"
        if config.task == "RegressionClassification":
            task_suffix = "reg"
        model_files: dict[str, Path] = {}
        for orig, sp in zip(config.properties, split_properties):
            ti_path = train_info_files.get(orig)
            if ti_path is not None:
                ti = json.loads(ti_path.read_text())
                model_files[orig] = run_folder / ti["model_file"]
            else:
                model_files[orig] = run_folder / f"{sp}_stacking{task_suffix}model.pt"

        # Verify expected model files exist before proceeding
        for prop, mf in model_files.items():
            if not mf.exists():
                raise RuntimeError(
                    f"Expected model file not found after training: {mf}\n"
                    "The train script may have used a different filename pattern."
                )

        # ── Step 4: merge (optional) ──────────────────────────────────────────────
        await _progress(4, total_steps)
        merged_model: Optional[Path] = None
        if len(config.properties) > 1:
            merged_model = run_folder / f"merged_stacking{task_suffix}model.pt"
            merge_args = ["--output-file", str(merged_model), "--verify-encoder"]
            for p in config.properties:
                merge_args += ["--model-files", str(model_files[p])]
            await _run_script(scripts / "merge_models.py", merge_args)

        # ── Step 5: evaluate ──────────────────────────────────────────────────────
        await _progress(5, total_steps)
        eval_task = "classification" if config.task == "RegressionClassification" else _task_slug(config)
        eval_script = _scripts_dir() / f"evaluate_{eval_task}_model.py"
        # Evaluates first property (representative metrics) or iterate all
        eval_results: dict[str, Path] = {}
        for orig, sp in zip(config.properties, split_properties):
            eval_args = [
                "--model-file", str(model_files[orig]),
                "--test-file", str(split_csv),
                "--output-folder", str(run_folder),
                "--split-type", "test",
                "--no-generate-pdf",
                "--verbose",
            ]
            await _run_script(eval_script, eval_args)
            # evaluate script names the CSV after the internal property (log10_{p} when transformed)
            eval_results[orig] = run_folder / f"{sp}_evaluation_predictions.csv"

        # Extract metrics from evaluation CSVs
        metrics = _extract_metrics(eval_results, config.task)

        # ── Step 6: refit (optional) ──────────────────────────────────────────────
        await _progress(6, total_steps)
        refitted_model_files: dict[str, Path] = {}
        if config.refit:
            refit_args_base = [
                "--csv-file", str(split_csv),
                "--output-folder", str(run_folder),
            ]
            if config.include_test_in_refit:
                refit_args_base.append("--include-test-set")
            for prop in config.properties:
                refit_args = ["--model-file", str(model_files[prop])] + refit_args_base
                await _run_script(scripts / "refit_model.py", refit_args)
                # Derive the actual property stem from the model file (may have Class_ prefix)
                model_stem = model_files[prop].stem  # e.g. "Class_Y_stackingclfmodel"
                prop_stem = model_stem.split("_stacking")[0]  # e.g. "Class_Y"
                refitted_model_files[prop] = run_folder / f"{prop_stem}_refitted_stacking{task_suffix}model.pt"

        # ── Step 7: re-merge refitted (optional) ─────────────────────────────────
        await _progress(7, total_steps)
        merged_refitted: Optional[Path] = None
        if config.refit and len(config.properties) > 1:
            merged_refitted = run_folder / f"merged_refitted_stacking{task_suffix}model.pt"
            merge_args = ["--output-file", str(merged_refitted), "--verify-encoder"]
            for p in config.properties:
                merge_args += ["--model-files", str(refitted_model_files[p])]
            await _run_script(scripts / "merge_models.py", merge_args)

        # ── Build pipeline_state.json for dashboard ───────────────────────────────
        state = _build_pipeline_state(
            run_id=run_id,
            run_folder=run_folder,
            config=config,
            csv_stem=csv_stem,
            eval_results=eval_results,
            model_files=model_files,
            merged_model=merged_model,
            refitted_model_files=refitted_model_files,
            merged_refitted=merged_refitted,
            metrics=metrics,
            train_info_files=train_info_files,
        )
        state_path = run_folder / "pipeline_state.json"
        state_path.write_text(json.dumps(state, indent=2))

        # ── Step 8: dashboard ─────────────────────────────────────────────────────
        await _progress(8, total_steps)
        dashboard_path = run_folder / "dashboard.html"
        dash_args = [
            "--pipeline-state", str(state_path),
            "--output", str(dashboard_path),
            "--verbose",
        ]
        await _run_script(_visualize_script(), dash_args, cwd=_scripts_dir())

        # ── Register model in registry ─────────────────────────────────────────────
        registry_script = scripts / "model_registry.py"
        reg_args = ["register", "--pipeline-state", str(state_path)]
        if owner:
            reg_args += ["--owner", owner]
        model_id = None
        try:
            reg_stdout = await _run_script(registry_script, reg_args)
            last_line = reg_stdout.strip().split("\n")[-1]
            reg_result = json.loads(last_line)
            model_id = reg_result.get("id")
        except (RuntimeError, json.JSONDecodeError, IndexError) as exc:
            logging.getLogger(__name__).warning("Model registration failed (model still usable): %s", exc)

        # ── Assemble result ───────────────────────────────────────────────────────
        final_model = merged_refitted or merged_model
        if final_model is None:
            prop = config.properties[0]
            final_model = refitted_model_files.get(prop) or model_files[prop]

        dashboard_html = dashboard_path.read_text(encoding="utf-8")

        # Load train_info content for the result (read the JSON files we already located)
        train_info_content = {}
        for prop, ti_path in train_info_files.items():
            try:
                train_info_content[prop] = json.loads(ti_path.read_text())
            except Exception:
                pass

        return TrainingResult(
            run_id=run_id,
            output_folder=str(run_folder),
            model_id=model_id,
            model_filename=final_model.name,
            dashboard_html=dashboard_html,
            metrics=metrics,
            properties=config.properties,
            task=config.task,
            model_path=str(final_model),
            dashboard_path=str(dashboard_path),
            pipeline_state_path=str(state_path),
            train_info=train_info_content,
        )
    finally:
        async with _active_folders_lock:
            _active_run_folders.discard(folder_str)


def _extract_metrics(eval_results: dict[str, Path], task: str) -> dict:
    """Read evaluation CSVs and compute summary metrics per property."""
    import math

    metrics: dict = {}
    for prop, csv_path in eval_results.items():
        if not csv_path.exists():
            continue
        try:
            # Lazy import — pandas is available in the venv
            import pandas as pd
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        prop_metrics: dict = {}
        # Column names may use a transformed name (e.g. "true_log10_prop1" when
        # use_log10 was active). Try exact match first, then transformed variants.
        true_col = f"true_{prop}"
        pred_col = f"predicted_{prop}"
        if true_col not in df.columns:
            true_col = next((c for c in df.columns if c in (f"true_log10_{prop}", f"true_logit_{prop}")), true_col)
        if pred_col not in df.columns:
            pred_col = next((c for c in df.columns if c in (f"predicted_log10_{prop}", f"predicted_logit_{prop}")), pred_col)

        if task == "Regression" and true_col in df.columns and pred_col in df.columns:
            true = df[true_col].dropna()
            pred = df[pred_col].dropna()
            idx = true.index.intersection(pred.index)
            true, pred = true.loc[idx], pred.loc[idx]
            if len(true) > 0:
                ss_res = ((true - pred) ** 2).sum()
                ss_tot = ((true - true.mean()) ** 2).sum()
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                rmse = math.sqrt(ss_res / len(true))
                mae = (true - pred).abs().mean()
                prop_metrics = {"R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4)}

        elif task in ("Classification", "RegressionClassification"):
            true_col_clf = f"true_Class_{prop}"
            pred_col_clf = f"predicted_Class_{prop}"
            if true_col_clf not in df.columns:
                true_col_clf = true_col
                pred_col_clf = pred_col
            if true_col_clf in df.columns and pred_col_clf in df.columns:
                true = df[true_col_clf].dropna()
                pred = df[pred_col_clf].dropna()
                idx = true.index.intersection(pred.index)
                true, pred = true.loc[idx], pred.loc[idx]
                if len(true) > 0:
                    acc = (true == pred).mean()
                    prop_metrics = {"Accuracy": round(acc, 4)}

        metrics[prop] = prop_metrics
    return metrics


def _build_pipeline_state(
    *,
    run_id: str,
    run_folder: Path,
    config: TrainingConfig,
    csv_stem: str,
    eval_results: dict[str, Path],
    model_files: dict[str, Path],
    merged_model: Optional[Path],
    refitted_model_files: dict[str, Path],
    merged_refitted: Optional[Path],
    metrics: dict,
    train_info_files: dict[str, Path] | None = None,
) -> dict:
    """Construct a pipeline_state.json compatible with generate_dashboard.py."""
    task_map = {
        "Regression": "regression",
        "Classification": "classification",
        "RegressionClassification": "regression_classification",
    }
    return {
        "pipeline_version": "2.0",
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "steps_completed": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "current_step": 8,
        "pipeline_complete": True,
        "config": {
            "data_file": Path(config.csv_file).name,
            "smiles_column": config.smiles_column,
            "target_properties": config.properties,
            "task_type": task_map[config.task],
            "feature_keys": config.feature_keys,
            "computational_load": config.computational_load,
            "split_strategy": config.split_strategy,
            "use_advanced": config.use_advanced,
            "output_folder": str(run_folder) + "/",
            "categorical": config.categorical,
            "nb_classes": config.nb_classes,
            "class_values": config.class_values,
            "blender_properties": config.blender_properties,
        },
        "outputs": {
            "prepared_csv": str(run_folder / f"automol_prepared_{csv_stem}.csv"),
            "split_csv": str(run_folder / f"automol_split_{csv_stem}.csv"),
            "model_files": {p: str(f) for p, f in model_files.items()},
            "merged_model_file": str(merged_model) if merged_model else None,
            "refitted_model_files": {p: str(f) for p, f in refitted_model_files.items()},
            "merged_refitted_model_file": str(merged_refitted) if merged_refitted else None,
            "evaluation_results": {p: str(f) for p, f in eval_results.items()},
            "train_info": {p: str(f) for p, f in (train_info_files or {}).items()} or None,
        },
        "metrics": metrics,
        "decisions": {
            "refit_approved": config.refit,
            "include_test_in_refit": config.include_test_in_refit,
        },
        "errors": [],
    }


# ── CLI entry point (for manual testing) ─────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AutoMol training pipeline from a JSON config")
    parser.add_argument("--config", required=True, help="Path to TrainingConfig JSON file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = TrainingConfig.model_validate_json(f.read())

    def print_progress(step: int, total: int) -> None:
        print(f"[{step}/{total}] Running step {step}...", file=sys.stderr)

    result = asyncio.run(run_full_pipeline(cfg, progress_cb=print_progress))
    print(json.dumps(result.model_dump(exclude={"dashboard_html"}), indent=2))
    print(f"\nModel saved to: {result.model_path}", file=sys.stderr)
    print(f"Dashboard saved to: {result.dashboard_path}", file=sys.stderr)
