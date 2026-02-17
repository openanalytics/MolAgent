#!/bin/bash
# SessionStart hook for AutoMol plugin
# Provides: AUTOMOL_ROOT, PLUGIN_ROOT (alias)
#
# Works in BOTH modes:
#   - Plugin mode: /plugin install automol (uses CLAUDE_PLUGIN_ROOT)
#   - Standalone mode: cd automol-models && claude (uses CLAUDE_PROJECT_DIR or PWD)

if [ -n "$CLAUDE_ENV_FILE" ]; then
  # 3-level fallback: CLAUDE_PLUGIN_ROOT → CLAUDE_PROJECT_DIR → PWD
  if [ -n "$CLAUDE_PLUGIN_ROOT" ]; then
    AUTOMOL_ROOT="$CLAUDE_PLUGIN_ROOT"
  elif [ -n "$CLAUDE_PROJECT_DIR" ]; then
    AUTOMOL_ROOT="$CLAUDE_PROJECT_DIR"
  else
    AUTOMOL_ROOT="$PWD"
  fi

  echo "export AUTOMOL_ROOT=\"$AUTOMOL_ROOT\"" >> "$CLAUDE_ENV_FILE"
  echo "export PLUGIN_ROOT=\"$AUTOMOL_ROOT\"" >> "$CLAUDE_ENV_FILE"
fi

if [ ! -d ".venv" ]; then
  echo "⚠️ No virtual environment found. Using uv to make one ..." >&2
  pip install uv >/dev/null 2>&1
  uv venv --python 3.12 >/dev/null 2>&1
  uv pip install AutoMol/automol_resources/ >/dev/null 2>&1
  uv pip install AutoMol/automol/ >/dev/null 2>&1
fi

# Laat de oorspronkelijke tool-aanroep doorgaan
exit 0


exit 0
