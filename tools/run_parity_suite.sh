#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "No python interpreter found (checked PYTHON_BIN, python, python3)." >&2
  exit 1
fi

"$PYTHON_BIN" src/scripts/parity/parity_report.py "$@"

echo "Parity report written:"
echo "  outputs/parity_report/latest.json"
echo "  outputs/parity_report/latest.md"
