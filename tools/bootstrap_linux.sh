#!/usr/bin/env bash
set -euo pipefail

# Reproducible bootstrap for CounterBMT on Linux.
#
# Usage:
#   tools/bootstrap_linux.sh [v2|legacy|full]
#
# Env vars:
#   PYTHON_BIN=python3.10
#   VENV_DIR=.venv-v2
#   RECREATE_VENV=0|1

PROFILE="${1:-v2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEFAULT_VENV_DIR=".venv-${PROFILE}"
VENV_DIR="${VENV_DIR:-$DEFAULT_VENV_DIR}"
RECREATE_VENV="${RECREATE_VENV:-0}"

if [[ "$PROFILE" != "v2" && "$PROFILE" != "legacy" && "$PROFILE" != "full" ]]; then
  echo "Invalid profile: $PROFILE (expected v2|legacy|full)"
  exit 2
fi

if [[ -f ".gitmodules" ]] && command -v git >/dev/null 2>&1; then
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git submodule update --init --recursive
  fi
fi

if [[ "$RECREATE_VENV" == "1" && -d "$VENV_DIR" ]]; then
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

REQ_FILE="requirements.txt"
if [[ "$PROFILE" == "legacy" ]]; then
  REQ_FILE="requirements-legacy.txt"
elif [[ "$PROFILE" == "full" ]]; then
  REQ_FILE="requirements-installed-freeze.txt"
fi

"$VENV_DIR/bin/python" -m pip install -r "$REQ_FILE"
"$VENV_DIR/bin/python" tools/verify_environment.py --profile "$PROFILE"

echo "Bootstrap complete for profile '$PROFILE'."
