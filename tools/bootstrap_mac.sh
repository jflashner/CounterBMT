#!/usr/bin/env bash
set -euo pipefail

# Reproducible bootstrap for CounterBMT on macOS CPU-only local iteration.
#
# Usage:
#   tools/bootstrap_mac.sh
#
# Env vars:
#   PYTHON_BIN=python3.10
#   VENV_DIR=.venv-mac
#   RECREATE_VENV=0|1
#   INSTALL_PARITY_TOOLS=0|1

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-.venv-mac}"
RECREATE_VENV="${RECREATE_VENV:-0}"
INSTALL_PARITY_TOOLS="${INSTALL_PARITY_TOOLS:-0}"
REQ_FILE="requirements-mac-cpu.txt"
PARITY_REQ_FILE="requirements-mac-parity-tools.txt"

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
"$VENV_DIR/bin/python" -m pip install -r "$REQ_FILE"

if [[ "$INSTALL_PARITY_TOOLS" == "1" ]]; then
  "$VENV_DIR/bin/python" -m pip install -r "$PARITY_REQ_FILE"
fi

"$VENV_DIR/bin/python" tools/verify_environment.py --profile mac-cpu

mkdir -p ".cache/matplotlib"

echo "Bootstrap complete for macOS CPU profile."
echo "Activate with: source \"$VENV_DIR/bin/activate\""
echo "Recommended for local plotting/tests: export MPLCONFIGDIR=\"$PWD/.cache/matplotlib\""
if [[ "$INSTALL_PARITY_TOOLS" == "1" ]]; then
  echo "Legacy parity tools installed from: $PARITY_REQ_FILE"
fi
