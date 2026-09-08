@echo off
echo Installing MolAgent...

pip install uv
REM On native Windows a project-local .venv is fine (no 9p mount penalty).
uv venv .venv --python 3.12
call .venv\Scripts\activate.bat

uv pip install -r requirements.txt

if exist AutoMol\automol (
  REM [extended] brings the 3D feature stack; automol_resources ships trained model files.
  uv pip install -e "AutoMol\automol[extended]" -e AutoMol\automol_resources
  echo AutoMol installed from submodule.
) else (
  echo WARNING: AutoMol submodule not found. Run: git submodule update --init
)

echo Done. Activate with: .venv\Scripts\activate.bat
