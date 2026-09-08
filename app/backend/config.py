"""Application settings — resolved from env vars with sensible defaults."""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

from pydantic_settings import BaseSettings


def _find_plugin_root() -> Path:
    d = Path(__file__).resolve().parent
    for _ in range(5):
        d = d.parent
        if (d / "skills").is_dir():
            return d
    return Path(__file__).resolve().parents[2]


def _find_venv_python(plugin_root: Path) -> Path:
    venv = Path(os.environ.get("AUTOMOL_VENV", str(plugin_root / ".venv")))
    if platform.system() == "Windows":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


class Settings(BaseSettings):
    plugin_root: Path = _find_plugin_root()
    output_root: Path = Path(
        os.environ.get("MOLAGENT_OUTPUT_ROOT")
        or str(_find_plugin_root() / "MolagentFiles")
    )
    venv_python: Path = _find_venv_python(_find_plugin_root())
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def train_scripts_dir(self) -> Path:
        return self.plugin_root / "skills" / "train-pipeline" / "scripts"

    @property
    def predict_script(self) -> Path:
        return self.plugin_root / "skills" / "predict" / "scripts" / "predict.py"

    @property
    def visualize_script(self) -> Path:
        return self.plugin_root / "skills" / "visualize" / "scripts" / "generate_dashboard.py"

    @property
    def uploads_dir(self) -> Path:
        return self.output_root / "uploads"

    @property
    def uv_path(self) -> str:
        found = shutil.which("uv")
        return found or "uv"

    model_config = {"env_prefix": "MOLAGENT_APP_"}


settings = Settings()
