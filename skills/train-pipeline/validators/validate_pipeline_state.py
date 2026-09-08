#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Stop hook validator for the training pipeline skill.

Validates that pipeline_state.json exists and contains required fields.
Prevents the LLM from stopping before it has actually written a valid plan.

Behavior:
  - After Phase 1 (PLAN): enforces that plan file was written. current_step >= 2
    means plan is complete. This is the primary guard.
  - After Phase 2 (EXECUTE): file already exists with current_step >= 2,
    so the hook passes trivially. Execution completion is enforced by the task
    list, not by this hook. This allows the user to abort mid-execution.

Output root resolution:
  --directory CLI flag > MOLAGENT_OUTPUT_ROOT > PHARMAOS_MOLAGENT_ROOT > MolagentFiles

Log file lives in tempdir (writable in plugin-cache deployments) — the legacy
``SCRIPT_DIR`` location was read-only when installed via /plugin install.

Hook type: Stop
Exit codes:
  0: Validation passed (pipeline_state.json is valid)
  1: Validation failed (missing file or required fields)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

LOG_DIR_ENV = "MOLAGENT_LOG_DIR"


def _resolve_log_path() -> Path:
    """Pick a writable log location.

    Priority: ``MOLAGENT_LOG_DIR`` env > system tempdir.
    Plugin-cache directories are often read-only so the legacy
    ``SCRIPT_DIR / "validate_pipeline_state.log"`` is unsafe.
    """
    explicit = os.environ.get(LOG_DIR_ENV)
    if explicit:
        log_dir = Path(explicit)
    else:
        log_dir = Path(tempfile.gettempdir()) / "molagent"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "validate_pipeline_state.log"


def _resolve_default_directory() -> str:
    """Default --directory honors env-var output-root resolution."""
    return (
        os.environ.get("PHARMAOS_MOLAGENT_ROOT")
        or os.environ.get("MOLAGENT_OUTPUT_ROOT")
        or "MolagentFiles"
    )


LOG_FILE = _resolve_log_path()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE, mode="a")],
)
logger = logging.getLogger(__name__)

STATE_FILENAME = "pipeline_state.json"

REQUIRED_TOP_KEYS = [
    "pipeline_version",
    "steps_completed",
    "current_step",
    "config",
]

REQUIRED_CONFIG_KEYS = [
    "data_file",
    "smiles_column",
    "target_properties",
    "task_type",
]

NO_FILE_ERROR = (
    "VALIDATION FAILED: No pipeline_state.json found in {directory}/ or its subdirectories.\n\n"
    "ACTION REQUIRED: You must write pipeline_state.json to a run folder under {directory}/ "
    "with the pipeline configuration before stopping. "
    "Use the Write tool to create the file. Do not stop until it exists."
)

INVALID_JSON_ERROR = (
    "VALIDATION FAILED: {path} is not valid JSON.\n\n"
    "ACTION REQUIRED: Fix the JSON in {path}. "
    "Do not stop until the file contains valid JSON."
)

MISSING_KEYS_ERROR = (
    "VALIDATION FAILED: {path} is missing {count} required field(s).\n\n"
    "MISSING FIELDS:\n{missing_list}\n\n"
    "ACTION REQUIRED: Use the Edit tool to add the missing fields to {path}. "
    "Do not stop until all required fields are present."
)


def find_state_files(directory: str) -> list[Path]:
    """Find all pipeline_state.json files in run folders, sorted by mtime."""
    base = Path(directory)

    direct = base / STATE_FILENAME
    if direct.exists():
        return [direct]

    if not base.exists():
        return []

    # Sort by modification time so the *most recently updated* run is checked
    # last (newest sort order), which is what the user just touched. Falling
    # back to lexicographic ordering when mtimes match.
    files = list(base.glob(f"*/{STATE_FILENAME}"))
    files.sort(key=lambda p: (p.stat().st_mtime, str(p)))
    return files


def validate_state(state_path: Path) -> tuple[bool, str]:
    """Validate a single pipeline_state.json file."""
    try:
        with open(state_path) as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False, INVALID_JSON_ERROR.format(path=state_path)

    missing = []
    for key in REQUIRED_TOP_KEYS:
        if key not in state:
            missing.append(key)

    config = state.get("config", {})
    if isinstance(config, dict):
        for key in REQUIRED_CONFIG_KEYS:
            if key not in config:
                missing.append(f"config.{key}")

    if missing:
        missing_list = "\n".join(f"  - {m}" for m in missing)
        return False, MISSING_KEYS_ERROR.format(
            path=state_path, count=len(missing), missing_list=missing_list
        )

    current_step = state.get("current_step", 0)
    if current_step < 2:
        msg = (
            f"VALIDATION FAILED: current_step is {current_step}, expected >= 2.\n\n"
            f"ACTION REQUIRED: Complete the planning phase (detect + configure) "
            f"and set current_step to 2 before stopping."
        )
        return False, msg

    return True, f"Pipeline state valid: {state_path} (step {current_step})"


def validate(directory: str) -> tuple[bool, str]:
    state_files = find_state_files(directory)
    if not state_files:
        return False, NO_FILE_ERROR.format(directory=directory)
    # Validate the most-recently-modified state file
    return validate_state(state_files[-1])


def main():
    logger.info("=" * 60)
    logger.info("Validator started: validate_pipeline_state")

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--directory", default=_resolve_default_directory())
        args = parser.parse_args()

        try:
            json.load(sys.stdin)
        except (json.JSONDecodeError, EOFError):
            pass

        success, message = validate(args.directory)

        if success:
            logger.info(f"PASS: {message}")
            print(json.dumps({"result": "continue", "message": message}))
            sys.exit(0)
        else:
            logger.warning(f"BLOCK: {message}")
            print(json.dumps({"result": "block", "reason": message}))
            sys.exit(1)

    except Exception as e:
        logger.exception(f"Validation error: {e}")
        print(
            json.dumps(
                {
                    "result": "continue",
                    "message": f"Validation error (allowing through): {str(e)}",
                }
            )
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
