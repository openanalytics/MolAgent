"""Opt-in deterministic training for MolAgent.

Activated by ``MOLAGENT_DETERMINISTIC=true`` (also accepts 1/yes/on, case
insensitive). When active:

  - Seeds ``random``, ``numpy.random``, ``torch`` (CPU + CUDA)
  - Sets ``PYTHONHASHSEED`` for stable hash ordering across runs
  - Asks torch for deterministic algorithms (warn-only; some ops have no
    deterministic implementation)
  - Returns sentinel that callers use to override ``n_jobs`` to 1 (joblib
    parallel CV is the largest source of non-reproducibility)

Default (env unset / falsy): no behavior change.
"""

from __future__ import annotations

import os
import random

ENV_VAR = "MOLAGENT_DETERMINISTIC"
TRUTHY = {"1", "true", "yes", "on", "y", "t"}


def is_deterministic() -> bool:
    return os.environ.get(ENV_VAR, "").strip().lower() in TRUTHY


def maybe_seed_everything(seed: int = 42) -> bool:
    """Seed RNGs globally if MOLAGENT_DETERMINISTIC is set. Returns True if seeded."""
    if not is_deterministic():
        return False

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch  # type: ignore[import]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError):
            pass
    except ImportError:
        pass

    print(
        f"[determinism] MOLAGENT_DETERMINISTIC=true — seeded RNGs (seed={seed}); "
        f"n_jobs forced to 1 by callers.",
        flush=True,
    )
    return True


def force_serial_jobs(n_jobs: dict) -> dict:
    """If deterministic, override every job count in the dict to 1."""
    if not is_deterministic():
        return n_jobs
    return {k: 1 for k in n_jobs}
