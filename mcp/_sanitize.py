"""Strip internal filesystem paths from MCP responses for remote callers."""
from __future__ import annotations


_STRIP_MODEL_FIELDS = frozenset({
    "model_file",
    "source_dataset",
    "run_folder",
    "train_info",
    "model_card",
})

_STRIP_TRAIN_FIELDS = frozenset({
    "model_path",
    "dashboard_path",
    "pipeline_state_path",
    "output_folder",
})

_STRIP_PREDICT_FIELDS = frozenset({
    "output_csv",
    "model_file",
    "output_folder",
})


def sanitize_model_entry(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if k not in _STRIP_MODEL_FIELDS}


def sanitize_train_result(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in _STRIP_TRAIN_FIELDS}


def sanitize_predict_result(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in _STRIP_PREDICT_FIELDS}
