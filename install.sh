#!/bin/bash
set -euo pipefail

echo "Installing MolAgent..."

# Check prerequisites
command -v uv >/dev/null 2>&1 || { echo "Installing uv..."; pip install uv; }

# On WSL, create the venv on the Linux filesystem (ext4), NOT under /mnt/c.
# /mnt/c is a 9p mount: per-file operations are ~137x slower and uv cannot
# hardlink from its ext4 cache, so it full-copies every file.
VENV_DIR="${HOME}/.venvs/molagent"
if [ ! -d "${VENV_DIR}" ]; then
  uv venv "${VENV_DIR}" --python 3.12
fi

# Activate
source "${VENV_DIR}/bin/activate"

# Install dependencies
uv pip install -r requirements.txt

# Install AutoMol from submodule
if [ -d "AutoMol/automol" ]; then
  # [extended] brings the 3D feature stack (torch-geometric, yacs, networkx,
  # MDAnalysis, prolif, molfeat, lightning).
  # automol_resources ships trained model files required by the default encoder.
  uv pip install -e "AutoMol/automol[extended]" -e AutoMol/automol_resources
  echo "AutoMol installed from submodule."
else
  echo "WARNING: AutoMol submodule not found. Run: git submodule update --init"
fi

echo "Done. Activate with: source ${VENV_DIR}/bin/activate"
