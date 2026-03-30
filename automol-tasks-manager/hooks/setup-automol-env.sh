#!/bin/bash
# SessionStart hook for AutoMol plugin
# Provides: AUTOMOL_ROOT (repo root), PLUGIN_ROOT (plugin root with skills/)
#
# Works in BOTH modes:
#   - Plugin mode: claude --plugin-dir ./automol-tasks-manager/ (uses CLAUDE_PLUGIN_ROOT)
#   - Standalone mode: cd MolAgent && claude (uses CLAUDE_PROJECT_DIR or PWD)

# 1. Compute roots (always run, not gated on CLAUDE_ENV_FILE)
if [ -n "$CLAUDE_PROJECT_DIR" ]; then AUTOMOL_ROOT="$CLAUDE_PROJECT_DIR"; else AUTOMOL_ROOT="$PWD"; fi

if [ -n "$CLAUDE_PLUGIN_ROOT" ]; then
  PLUGIN_ROOT="$CLAUDE_PLUGIN_ROOT"
elif [ -d "$AUTOMOL_ROOT/automol-tasks-manager/skills" ]; then
  PLUGIN_ROOT="$AUTOMOL_ROOT/automol-tasks-manager"
else
  PLUGIN_ROOT="$AUTOMOL_ROOT"
fi

# 2. Write to CLAUDE_ENV_FILE if set
if [ -n "$CLAUDE_ENV_FILE" ]; then
  echo "export AUTOMOL_ROOT=\"$AUTOMOL_ROOT\"" >> "$CLAUDE_ENV_FILE"
  echo "export PLUGIN_ROOT=\"$PLUGIN_ROOT\"" >> "$CLAUDE_ENV_FILE"
fi

# 3. Venv setup + AutoMol install
VENV_DIR="${AUTOMOL_VENV:-$AUTOMOL_ROOT/.venv}"

if [ ! -d "$VENV_DIR" ]; then
  echo "No virtual environment found at $VENV_DIR. Creating one ..." >&2
  pip install uv >/dev/null 2>&1
  if ! uv venv "$VENV_DIR" --python 3.12; then
    echo "ERROR: Failed to create virtual environment at $VENV_DIR" >&2
    exit 1
  fi
fi

if [ -d "$VENV_DIR/Scripts" ]; then
  echo "Windows setup found, activating venv from $VENV_DIR/Scripts/activate" >&2
  source "$VENV_DIR/Scripts/activate"
else
  echo "Linux setup found, activating venv from $VENV_DIR/bin/activate" >&2
  source "$VENV_DIR/bin/activate"
fi

# Install AutoMol (resources first, then core library)
AUTOMOL_RESOURCES="$AUTOMOL_ROOT/AutoMol/automol_resources"
AUTOMOL_LIB="$AUTOMOL_ROOT/AutoMol/automol"

if [ -d "$AUTOMOL_RESOURCES" ]; then
  echo "Installing automol_resources from $AUTOMOL_RESOURCES ..." >&2
  if ! uv pip install "$AUTOMOL_RESOURCES"; then
    echo "ERROR: Failed to install automol_resources" >&2
    exit 1
  fi
else
  echo "WARNING: automol_resources not found at $AUTOMOL_RESOURCES (skipping)" >&2
fi

echo "Installing automol from $AUTOMOL_LIB ..." >&2
if ! uv pip install "$AUTOMOL_LIB"; then
  echo "ERROR: Failed to install automol" >&2
  exit 1
fi
uv pip install -r "$CLAUDE_PROJECT_DIR/requirements.txt"
uv pip install pytdc
uv pip install rdkit==2024.3.5

# 4. Persist to settings.local.json so PLUGIN_ROOT is available in all Bash calls (including subagents)
_SETTINGS_FILE="$AUTOMOL_ROOT/.claude/settings.local.json" \
_PLUGIN_ROOT="$PLUGIN_ROOT" \
_AUTOMOL_ROOT="$AUTOMOL_ROOT" \
python -c "
import json, sys, os
sf = os.environ['_SETTINGS_FILE']
try:
    with open(sf) as f: s = json.load(f)
except (FileNotFoundError, json.JSONDecodeError): s = {}
s.setdefault('env', {})
s['env']['PLUGIN_ROOT'] = os.environ['_PLUGIN_ROOT']
s['env']['AUTOMOL_ROOT'] = os.environ['_AUTOMOL_ROOT']
with open(sf, 'w') as f: json.dump(s, f, indent=2)
print('PLUGIN_ROOT persisted to settings.local.json (takes effect on next session start)', file=sys.stderr)
" 2>&1 || echo "WARNING: Could not update settings.local.json" >&2

exit 0
