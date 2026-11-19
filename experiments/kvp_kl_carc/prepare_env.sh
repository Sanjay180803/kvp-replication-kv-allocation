#!/bin/bash
# Run in your own CARC setup/CPU session with Python 3.11 or 3.12 available.
# Installs software only. Does not submit a job or download any model/agent weights.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/config.sh"
BOOTSTRAP_PYTHON=${BOOTSTRAP_PYTHON:-python3}
"$BOOTSTRAP_PYTHON" -c 'import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 12), "Use Python 3.11 or 3.12 for this environment"'
if [[ "$EXPERIMENT_ROOT" == /YOUR_* ]]; then printf 'Edit config.sh first.\n' >&2; exit 2; fi
mkdir -p "$EXPERIMENT_ROOT"
ENV_DIR=$(dirname "$(dirname "$PYTHON_BIN")")
if [[ ! -x "$PYTHON_BIN" ]]; then "$BOOTSTRAP_PYTHON" -m venv "$ENV_DIR"; fi
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel packaging
"$PYTHON_BIN" -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128
"$PYTHON_BIN" -m pip install -r "$HERE/requirements.txt"
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" -m pip freeze > "$EXPERIMENT_ROOT/environment.freeze.txt"
printf 'Environment ready: %s\nNo model or KVP weights were downloaded.\n' "$PYTHON_BIN"
