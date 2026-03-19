#!/usr/bin/env bash
set -euo pipefail

# Reproducible separate environment bootstrap for the legacy Adv-BMT trainer.
#
# This intentionally lives outside the main v2 JAX environment because the
# legacy stack pulls in a materially different dependency graph
# (PyTorch/Lightning/Hydra/PyG/MetaDrive/ScenarioNet, and optionally
# TensorFlow/Waymo).
#
# Usage:
#   tools/bootstrap_legacy_adv_bmt.sh
#
# Important env vars:
#   PYTHON_BIN=python3.10 (or leave unset for auto-detection)
#   VENV_DIR=.venv-legacy-adv-bmt
#   RECREATE_VENV=0|1
#   LEGACY_PROFILE=auto|linux-cu121|linux-cpu|mac-cpu
#   INSTALL_SIM_STACK=0|1
#   INSTALL_WAYMO_EVAL=0|1
#   EXTERNAL_DEPS_DIR=.external/legacy_deps
#   METADRIVE_SRC=/path/to/metadrive
#   SCENARIONET_SRC=/path/to/scenarionet
#   METADRIVE_REF=<git ref>
#   SCENARIONET_REF=<git ref>
#   DRY_RUN=0|1

PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR="${VENV_DIR:-.venv-legacy-adv-bmt}"
RECREATE_VENV="${RECREATE_VENV:-0}"
LEGACY_PROFILE="${LEGACY_PROFILE:-auto}"
INSTALL_SIM_STACK="${INSTALL_SIM_STACK:-1}"
INSTALL_WAYMO_EVAL="${INSTALL_WAYMO_EVAL:-0}"
EXTERNAL_DEPS_DIR="${EXTERNAL_DEPS_DIR:-.external/legacy_deps}"
METADRIVE_SRC="${METADRIVE_SRC:-}"
SCENARIONET_SRC="${SCENARIONET_SRC:-}"
METADRIVE_REF="${METADRIVE_REF:-}"
SCENARIONET_REF="${SCENARIONET_REF:-}"
DRY_RUN="${DRY_RUN:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

resolve_python_bin() {
  if [[ -n "$PYTHON_BIN" ]]; then
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
      echo "$PYTHON_BIN"
      return
    fi
    echo "Requested PYTHON_BIN not found on PATH: $PYTHON_BIN" >&2
    exit 1
  fi

  local candidate
  for candidate in python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
      return
    fi
  done

  echo "No suitable Python interpreter found. Checked: python3.10, python3, python" >&2
  exit 1
}

detect_profile() {
  local uname_s
  uname_s="$(uname -s | tr '[:upper:]' '[:lower:]')"
  if [[ "$LEGACY_PROFILE" != "auto" ]]; then
    echo "$LEGACY_PROFILE"
    return
  fi
  if [[ "$uname_s" == "darwin" ]]; then
    echo "mac-cpu"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "linux-cu121"
    return
  fi
  echo "linux-cpu"
}

clone_or_reuse_repo() {
  local name="$1"
  local repo_url="$2"
  local override_path="$3"
  local ref="$4"
  local dest

  if [[ -n "$override_path" ]]; then
    dest="$override_path"
  else
    dest="$EXTERNAL_DEPS_DIR/$name"
    if [[ ! -d "$dest/.git" ]]; then
      mkdir -p "$EXTERNAL_DEPS_DIR"
      run_cmd git clone "$repo_url" "$dest"
    fi
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    if [[ -n "$ref" && -d "$dest/.git" ]]; then
      run_cmd git -C "$dest" fetch --depth 1 origin "$ref"
      run_cmd git -C "$dest" checkout -f FETCH_HEAD
    elif [[ -n "$ref" && ! -d "$dest/.git" ]]; then
      echo "+ git -C $dest fetch --depth 1 origin $ref"
      echo "+ git -C $dest checkout -f FETCH_HEAD"
    fi
    run_cmd "$VENV_DIR/bin/python" -m pip install -e "$dest"
    return
  fi

  if [[ ! -d "$dest" ]]; then
    echo "Dependency repo not found: $dest" >&2
    exit 1
  fi

  if [[ -n "$ref" && -d "$dest/.git" ]]; then
    run_cmd git -C "$dest" fetch --depth 1 origin "$ref"
    run_cmd git -C "$dest" checkout -f FETCH_HEAD
  fi

  run_cmd "$VENV_DIR/bin/python" -m pip install -e "$dest"
}

PROFILE="$(detect_profile)"
PYTHON_BIN="$(resolve_python_bin)"

case "$PROFILE" in
  linux-cu121)
    TRAIN_REQ="requirements-legacy-train-linux-cu121.txt"
    VERIFY_PROFILE="legacy-train-linux-cu121"
    PYG_WHEEL_URL="https://data.pyg.org/whl/torch-2.5.1+cu121.html"
    ;;
  linux-cpu)
    TRAIN_REQ="requirements-legacy-train-cpu.txt"
    VERIFY_PROFILE="legacy-train-cpu"
    PYG_WHEEL_URL="https://data.pyg.org/whl/torch-2.5.1+cpu.html"
    ;;
  mac-cpu)
    TRAIN_REQ="requirements-legacy-train-cpu.txt"
    VERIFY_PROFILE="legacy-train-cpu"
    PYG_WHEEL_URL=""
    ;;
  *)
    echo "Unsupported LEGACY_PROFILE: $PROFILE" >&2
    exit 1
    ;;
esac

if [[ "$RECREATE_VENV" == "1" && -d "$VENV_DIR" ]]; then
  run_cmd rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  run_cmd "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

run_cmd "$VENV_DIR/bin/python" -m ensurepip --upgrade
run_cmd "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

if [[ "$PROFILE" == "linux-cu121" ]]; then
  # Pull CUDA-specific torch wheels from the PyTorch index while keeping the
  # rest of the dependency resolution on PyPI.
  run_cmd "$VENV_DIR/bin/python" -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r "$TRAIN_REQ"
else
  run_cmd "$VENV_DIR/bin/python" -m pip install -r "$TRAIN_REQ"
fi

if [[ "$PROFILE" == "linux-cu121" || "$PROFILE" == "linux-cpu" ]]; then
  # The legacy decoder uses torch_geometric message passing in hot paths. On
  # Linux we install the matching compiled PyG operators explicitly so training
  # speed and operator support match the legacy stack more closely.
  run_cmd "$VENV_DIR/bin/python" -m pip install \
    pyg-lib \
    torch-cluster \
    torch-scatter \
    torch-sparse \
    torch-spline-conv \
    -f "$PYG_WHEEL_URL"
else
  echo "Skipping compiled PyG extension install on macOS; torch-geometric base package only."
fi

if [[ "$INSTALL_SIM_STACK" == "1" ]]; then
  clone_or_reuse_repo "metadrive" "https://github.com/metadriverse/metadrive.git" "$METADRIVE_SRC" "$METADRIVE_REF"
  clone_or_reuse_repo "scenarionet" "https://github.com/metadriverse/scenarionet.git" "$SCENARIONET_SRC" "$SCENARIONET_REF"
fi

if [[ "$INSTALL_WAYMO_EVAL" == "1" ]]; then
  run_cmd "$VENV_DIR/bin/python" -m pip install -r requirements-legacy-waymo-eval.txt
  WAYMO_VERIFY_PROFILE="legacy-waymo-eval"
else
  WAYMO_VERIFY_PROFILE=""
fi

run_cmd "$VENV_DIR/bin/python" tools/verify_environment.py --profile "$VERIFY_PROFILE"
if [[ -n "$WAYMO_VERIFY_PROFILE" ]]; then
  run_cmd "$VENV_DIR/bin/python" tools/verify_environment.py --profile "$WAYMO_VERIFY_PROFILE"
fi

echo
echo "Legacy Adv-BMT bootstrap complete."
echo "Profile: $PROFILE"
echo "Activate with: source \"$VENV_DIR/bin/activate\""
echo "Train-only env is ready for the legacy trainer and the MidGPT learning probe."
if [[ "$INSTALL_WAYMO_EVAL" == "1" ]]; then
  echo "Waymo/TensorFlow evaluator stack is also installed."
else
  echo "Waymo/TensorFlow evaluator stack was skipped. Set INSTALL_WAYMO_EVAL=1 if you need the original legacy evaluator."
fi
