"""Shared path / output-root resolution for MolAgent pipeline scripts.

Single source of truth for "where do outputs live". Resolved in priority:

  1. PHARMAOS_MOLAGENT_ROOT   — Nexus-injected per-project root
  2. MOLAGENT_OUTPUT_ROOT     — explicit user override
  3. ./MolagentFiles          — default, relative to CWD

All scripts should use ``default_output_folder()`` for Click ``default=`` values
and ``get_output_root()`` for runtime path computation. The model registry uses
``get_registry_path()`` which honors a separate ``MOLAGENT_REGISTRY_PATH``
override but falls back to ``<output_root>/model_registry.json``.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ROOT_NAME = "MolagentFiles"
REGISTRY_FILENAME = "model_registry.json"


def get_output_root() -> Path:
    """Resolve the MolAgent output root directory."""
    root = os.environ.get("PHARMAOS_MOLAGENT_ROOT") or os.environ.get(
        "MOLAGENT_OUTPUT_ROOT"
    )
    return Path(root) if root else Path(DEFAULT_ROOT_NAME)


def default_output_folder() -> str:
    """String form for Click ``default=``, with trailing slash."""
    return str(get_output_root()).rstrip("/") + "/"


def get_registry_path(directory: str | os.PathLike[str] | None = None) -> Path:
    """Path to the global model registry JSON.

    Resolution order:
      1. ``MOLAGENT_REGISTRY_PATH`` env var (full path, takes precedence)
      2. ``directory / model_registry.json`` if ``directory`` was passed
      3. ``<output_root> / model_registry.json``
    """
    explicit = os.environ.get("MOLAGENT_REGISTRY_PATH")
    if explicit:
        return Path(explicit)
    base = Path(directory) if directory else get_output_root()
    return base / REGISTRY_FILENAME


def labelnames_from_json(raw: dict) -> dict:
    """Convert labelnames loaded from JSON back to int-keyed dicts.

    JSON serialization turns Python int keys into strings; this reverses it.
    """
    return {prop: {int(k): v for k, v in mapping.items()}
            for prop, mapping in raw.items()}


def replace_csv_suffix(path: str | os.PathLike[str], new_suffix: str) -> str:
    """Replace ``.csv`` at end of path with ``new_suffix``.

    Robust against paths containing ``.csv`` mid-string (e.g. ``data.csv.bak``)
    — only the final ``.csv`` is replaced. Mirrors ``Path.with_suffix`` for
    plain string paths but allows multi-character suffix replacements like
    ``_info.json``.
    """
    p = str(path)
    if p.lower().endswith(".csv"):
        return p[:-4] + new_suffix
    return p + new_suffix
