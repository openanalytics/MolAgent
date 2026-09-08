#!/bin/bash
# SessionStart hook for the MolAgentLight plugin.
#
# Exports:
#   AUTOMOL_ROOT          — repo / project root (CLAUDE_PROJECT_DIR or PWD)
#   MOLAGENT_PLUGIN_ROOT  — plugin root (CLAUDE_PLUGIN_ROOT, or local fallback)
#   MOLAGENT_OUTPUT_ROOT  — output directory (defaults to $AUTOMOL_ROOT/MolagentFiles
#                           but honors PHARMAOS_MOLAGENT_ROOT when injected by Nexus)
#
# Works in BOTH modes:
#   - Plugin mode: /plugin install MolAgentLight (uses CLAUDE_PLUGIN_ROOT)
#   - Standalone mode: cd MolAgentLight && claude (uses CLAUDE_PROJECT_DIR or PWD)
#   - Nexus mode: drug_discovery_ui injects PHARMAOS_MOLAGENT_ROOT per-project

set -u

# 1. Compute roots (always run, not gated on CLAUDE_ENV_FILE)
if [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
  AUTOMOL_ROOT="$CLAUDE_PROJECT_DIR"
else
  AUTOMOL_ROOT="$PWD"
fi

if [ -n "${CLAUDE_PLUGIN_ROOT:-}" ]; then
  MOLAGENT_PLUGIN_ROOT="$CLAUDE_PLUGIN_ROOT"
elif [ -d "$AUTOMOL_ROOT/MolAgent-Marketplace/MolAgentLight/skills" ]; then
  MOLAGENT_PLUGIN_ROOT="$AUTOMOL_ROOT/MolAgent-Marketplace/MolAgentLight"
else
  MOLAGENT_PLUGIN_ROOT="$AUTOMOL_ROOT"
fi

# Output root: PHARMAOS_MOLAGENT_ROOT (Nexus) > MOLAGENT_OUTPUT_ROOT (user) > default
if [ -n "${PHARMAOS_MOLAGENT_ROOT:-}" ]; then
  MOLAGENT_OUTPUT_ROOT="$PHARMAOS_MOLAGENT_ROOT"
elif [ -n "${MOLAGENT_OUTPUT_ROOT:-}" ]; then
  : # already set by user
else
  MOLAGENT_OUTPUT_ROOT="$AUTOMOL_ROOT/MolagentFiles"
fi

# 2. Write to CLAUDE_ENV_FILE if set (existing behavior)
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  {
    echo "export AUTOMOL_ROOT=\"$AUTOMOL_ROOT\""
    echo "export MOLAGENT_PLUGIN_ROOT=\"$MOLAGENT_PLUGIN_ROOT\""
    echo "export MOLAGENT_OUTPUT_ROOT=\"$MOLAGENT_OUTPUT_ROOT\""
  } >> "$CLAUDE_ENV_FILE"
fi

# 3. Venv setup + AutoMol install
VENV_DIR="${AUTOMOL_VENV:-$MOLAGENT_PLUGIN_ROOT/.venv}"

if [ ! -d "$VENV_DIR" ]; then
  echo "No virtual environment found at $VENV_DIR. Creating one ..." >&2
  pip install uv >/dev/null 2>&1 || true
  if ! uv venv "$VENV_DIR" --python 3.12; then
    echo "ERROR: Failed to create virtual environment at $VENV_DIR" >&2
    exit 1
  fi
fi

# Activate venv (Linux/macOS path or Windows-Git-Bash path)
if [ -d "$VENV_DIR/Scripts" ]; then
  echo "Windows setup found, activating venv from $VENV_DIR/Scripts/activate" >&2
  # shellcheck disable=SC1090,SC1091
  source "$VENV_DIR/Scripts/activate"
else
  echo "Linux setup found, activating venv from $VENV_DIR/bin/activate" >&2
  # shellcheck disable=SC1090,SC1091
  source "$VENV_DIR/bin/activate"
fi

# When running as a marketplace plugin, install from the bundled copy
if [ -n "${CLAUDE_PLUGIN_ROOT:-}" ]; then
  AUTOMOL_LIB="$CLAUDE_PLUGIN_ROOT/AutoMol/automol"
else
  AUTOMOL_LIB="$AUTOMOL_ROOT/AutoMol/automol"
fi

if ! uv pip show automol >/dev/null 2>&1; then
  echo "Installing automol from $AUTOMOL_LIB ..." >&2
  if ! uv pip install "$AUTOMOL_LIB"; then
    echo "ERROR: Failed to install automol" >&2
    exit 1
  fi
else
  echo "automol already installed, skipping." >&2
fi

echo "Installing fastmcp (required for MCP server) ..." >&2
uv pip install "fastmcp[tasks]" >/dev/null 2>&1 || echo "WARNING: fastmcp install failed; MCP server tools may be unavailable" >&2

# 4. Persist env to settings.local.json so vars are available in all Bash calls
#    (including subagents). Uses lock + tempfile + os.replace for atomicity and
#    handles paths-with-spaces by passing values through the environment, never
#    interpolating into the python -c body.
SETTINGS_DIR="$AUTOMOL_ROOT/.claude"
mkdir -p "$SETTINGS_DIR"
SETTINGS_FILE="$SETTINGS_DIR/settings.local.json"

_MOLAGENT_PLUGIN_ROOT="$MOLAGENT_PLUGIN_ROOT" \
_AUTOMOL_ROOT="$AUTOMOL_ROOT" \
_MOLAGENT_OUTPUT_ROOT="$MOLAGENT_OUTPUT_ROOT" \
_VENV_DIR="$VENV_DIR" \
_SETTINGS_FILE="$SETTINGS_FILE" \
python - <<'PY' 2>&1 || echo "WARNING: Could not update settings.local.json" >&2
import errno, json, os, sys, tempfile, time

sf = os.environ['_SETTINGS_FILE']
sf_dir = os.path.dirname(sf) or '.'
lock = sf + '.lock'

# Tiny exclusive-create lock for cross-platform safety
deadline = time.monotonic() + 5.0
fd = None
while True:
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        break
    except OSError as e:
        if e.errno not in (errno.EEXIST, errno.EACCES):
            raise
        if time.monotonic() > deadline:
            print("WARNING: settings.local.json lock not acquired; proceeding without lock", file=sys.stderr)
            break
        time.sleep(0.05)

try:
    try:
        with open(sf) as f:
            s = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        s = {}
    if not isinstance(s, dict):
        s = {}
    s.setdefault('env', {})
    s['env']['MOLAGENT_PLUGIN_ROOT'] = os.environ['_MOLAGENT_PLUGIN_ROOT']
    s['env']['AUTOMOL_ROOT'] = os.environ['_AUTOMOL_ROOT']
    s['env']['MOLAGENT_OUTPUT_ROOT'] = os.environ['_MOLAGENT_OUTPUT_ROOT']
    s['env']['AUTOMOL_VENV'] = os.environ['_VENV_DIR']

    tmp_fd, tmp = tempfile.mkstemp(prefix='.settings-', suffix='.json.tmp', dir=sf_dir)
    with os.fdopen(tmp_fd, 'w') as f:
        json.dump(s, f, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, sf)
    print('MOLAGENT env persisted to settings.local.json (takes effect on next session start)', file=sys.stderr)
finally:
    if fd is not None:
        try:
            os.close(fd)
        finally:
            try:
                os.unlink(lock)
            except FileNotFoundError:
                pass
PY

exit 0
