#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Stop hook validator for the training pipeline skill.

Validates that pipeline_state.json exists and contains required fields.
Prevents the LLM from stopping before it has actually written a valid plan.

Design: This hook fires on EVERY Stop event (both after Phase 1 and Phase 2).
- After Phase 1 (PLAN): Enforces that the plan file was actually written.
  current_step >= 2 means plan is complete. This is the primary guard.
- After Phase 2 (EXECUTE): File already exists with current_step >= 2,
  so the hook passes trivially. Execution completion is enforced by the
  task list, not by this hook. This allows the user to abort mid-execution.

v2.0: Now scans MolagentFiles/*/pipeline_state.json (run folders) when
the direct MolagentFiles/pipeline_state.json path doesn't exist.

Hook type: Stop
Exit codes:
  0: Validation passed (pipeline_state.json is valid)
  1: Validation failed (missing file or required fields)

Usage:
  uv run validate_pipeline_state.py --directory MolagentFiles
"""

import argparse
import json
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
LOG_FILE = SCRIPT_DIR / "validate_pipeline_state.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE, mode='a')]
)
logger = logging.getLogger(__name__)

DEFAULT_DIRECTORY = "MolagentFiles"
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
    """Find all pipeline_state.json files in run folders."""
    base = Path(directory)

    # Check direct path first (backward compatibility)
    direct = base / STATE_FILENAME
    if direct.exists():
        return [direct]

    # Scan run folders: MolagentFiles/*/pipeline_state.json
    state_files = sorted(base.glob(f"*/{STATE_FILENAME}"))
    return state_files


def validate_state(state_path: Path) -> tuple[bool, str]:
    """Validate a single pipeline_state.json file."""
    # Check valid JSON
    try:
        with open(state_path) as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError):
        msg = INVALID_JSON_ERROR.format(path=state_path)
        return False, msg

    # Check required top-level keys
    missing = []
    for key in REQUIRED_TOP_KEYS:
        if key not in state:
            missing.append(key)

    # Check required config keys (only if config exists)
    config = state.get("config", {})
    if isinstance(config, dict):
        for key in REQUIRED_CONFIG_KEYS:
            if key not in config:
                missing.append(f"config.{key}")

    if missing:
        missing_list = "\n".join(f"  - {m}" for m in missing)
        msg = MISSING_KEYS_ERROR.format(
            path=state_path, count=len(missing), missing_list=missing_list
        )
        return False, msg

    # Check current_step is at least 2 (plan phase complete)
    current_step = state.get("current_step", 0)
    if current_step < 2:
        msg = (
            f"VALIDATION FAILED: current_step is {current_step}, expected >= 2.\n\n"
            f"ACTION REQUIRED: Complete the planning phase (detect + configure) "
            f"and set current_step to 2 before stopping."
        )
        return False, msg

    msg = f"Pipeline state valid: {state_path} (step {current_step})"
    return True, msg


def validate(directory: str) -> tuple[bool, str]:
    """Validate pipeline_state.json exists and has required fields.

    Scans both the direct path and run folder subdirectories.
    """
    state_files = find_state_files(directory)

    # No state files found anywhere
    if not state_files:
        msg = NO_FILE_ERROR.format(directory=directory)
        return False, msg

    # Validate the most recent state file (last in sorted order = newest run folder)
    # If direct path exists, it was returned first and is the only entry
    state_path = state_files[-1]
    return validate_state(state_path)


def main():
    logger.info("=" * 60)
    logger.info("Validator started: validate_pipeline_state")

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--directory', default=DEFAULT_DIRECTORY)
        args = parser.parse_args()

        # Read hook input from stdin
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
        print(json.dumps({
            "result": "continue",
            "message": f"Validation error (allowing through): {str(e)}"
        }))
        sys.exit(0)


if __name__ == "__main__":
    main()
