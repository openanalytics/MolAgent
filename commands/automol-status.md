# AutoMol Status

Check the current AutoMol environment and trained models.

## Instructions

Run the following checks and report the results:

### 1. Environment

```bash
echo "AUTOMOL_ROOT: ${AUTOMOL_ROOT:-not set}"
echo "MOLAGENT_PLUGIN_ROOT: ${MOLAGENT_PLUGIN_ROOT:-not set (restart Claude Code to apply)}"
echo "AUTOMOL_VENV: ${AUTOMOL_VENV:-.venv}"
echo "Python: $(python --version 2>&1)"
echo "uv: $(uv --version 2>&1)"
```

### 2. AutoMol Installation

```bash
uv run python -c "from automol.version import __version__; print(__version__)" 2>&1 || echo "AutoMol not installed in current environment"
```

### 3. Virtual Environment

```bash
if [ -d "${AUTOMOL_VENV:-.venv}" ]; then
  echo "Virtual environment exists at ${AUTOMOL_VENV:-.venv}"
else
  echo "No virtual environment found"
fi
```

### 4. Trained Models

Check the model registry for trained models:

```bash
if [ -f "MolagentFiles/model_registry.json" ]; then
  echo "Model registry found:"
  cat MolagentFiles/model_registry.json
else
  echo "No model registry found (no models trained yet)"
fi
```

### 5. Settings

Check that MOLAGENT_PLUGIN_ROOT is persisted for subagents:

```bash
echo "=== Settings ==="
if python -c "
import json, os, sys
sf = os.path.join(os.getcwd(), '.claude', 'settings.local.json')
try:
    s = json.load(open(sf))
    env = s.get('env', {})
    print('MOLAGENT_PLUGIN_ROOT in settings.local.json:', env.get('MOLAGENT_PLUGIN_ROOT', 'NOT SET'))
    print('AUTOMOL_ROOT in settings.local.json:', env.get('AUTOMOL_ROOT', 'NOT SET'))
except: print('settings.local.json not found or invalid')
" 2>&1; then :; fi
```

### Output Format

Present the results as a concise status summary with sections for Environment, Settings, Installation, and Trained Models. Flag any issues (missing dependencies, broken venv, etc.).
