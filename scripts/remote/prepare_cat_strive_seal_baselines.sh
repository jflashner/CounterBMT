#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="${BASELINE_ROOT:-/home/grads/jflashner/CounterBMT_run/baselines}"
REPOS_DIR="${BASELINE_ROOT}/repos"
ARTIFACTS_DIR="${BASELINE_ROOT}/artifacts"
SCENARIO_BANKS_DIR="${BASELINE_ROOT}/scenario_banks"
LOG_DIR="${BASELINE_ROOT}/logs"

mkdir -p "${REPOS_DIR}" "${ARTIFACTS_DIR}" "${SCENARIO_BANKS_DIR}" "${LOG_DIR}"

clone_or_update() {
  local url="$1"
  local dest="$2"
  local extra_clone_args="${3:-}"

  if [[ -d "${dest}/.git" ]]; then
    echo "[prepare] Updating ${dest}"
    git -C "${dest}" fetch --depth 1 origin
    git -C "${dest}" reset --hard origin/HEAD
  else
    echo "[prepare] Cloning ${url} -> ${dest}"
    # shellcheck disable=SC2086
    git clone --depth 1 ${extra_clone_args} "${url}" "${dest}"
  fi
}

clone_or_update "https://github.com/metadriverse/cat.git" "${REPOS_DIR}/cat"
clone_or_update "https://github.com/nv-tlabs/STRIVE.git" "${REPOS_DIR}/STRIVE"

# SEAL vendors a large MetaDrive tree. Keep it opt-in because sparse checkout
# still needs several large blobs on some git versions.
if [[ "${CLONE_SEAL:-0}" == "1" && "${FULL_SEAL_CLONE:-0}" == "1" ]]; then
  clone_or_update "https://github.com/cmubig/SEAL.git" "${REPOS_DIR}/SEAL" "--filter=blob:none"
elif [[ "${CLONE_SEAL:-0}" == "1" ]]; then
  if [[ -d "${REPOS_DIR}/SEAL/.git" ]]; then
    echo "[prepare] Updating sparse SEAL checkout"
    git -C "${REPOS_DIR}/SEAL" fetch --depth 1 origin
    git -C "${REPOS_DIR}/SEAL" reset --hard origin/HEAD
  else
    echo "[prepare] Cloning sparse SEAL checkout -> ${REPOS_DIR}/SEAL"
    git clone --depth 1 --filter=blob:none --sparse "https://github.com/cmubig/SEAL.git" "${REPOS_DIR}/SEAL"
  fi
  git -C "${REPOS_DIR}/SEAL" sparse-checkout set --no-cone \
    readme.md requirements.txt cat_advgen.py cat_RLtrain.py cat_convert_output.py cat_metrics.py \
    advgen saferl_algo saferl_plotter decision32 reskill safeshift goose_train.py goose_models
else
  echo "[prepare] Skipping SEAL clone by default. Set CLONE_SEAL=1 when ready to stage SEAL."
fi

cat <<EOF

[prepare] Baseline repos staged under:
  ${REPOS_DIR}

[prepare] Artifact directories:
  CAT:    ${ARTIFACTS_DIR}/cat
  STRIVE: ${ARTIFACTS_DIR}/STRIVE
  SEAL:   ${ARTIFACTS_DIR}/SEAL

[prepare] Scenario bank target:
  ${SCENARIO_BANKS_DIR}

Manual / optional artifacts:
  CAT Drive folder:
    https://drive.google.com/drive/folders/1xVQ84pF5clVtKw6d4NCC-0mYbo4cIZ_a
  STRIVE checkpoints:
    https://www.dropbox.com/scl/fo/ajge853wnwgtrrxysuwim/APUPJ8UTs8mxGGNp0SHN0aw?rlkey=zfzkycbf7h1dx0a6mcj4etl1e&st=pk1tis1d&dl=0
  STRIVE generated nuScenes scenarios:
    https://www.dropbox.com/scl/fo/djjdbr4ykiwogvfdpuz0c/APryskA5DXsnRqWwvs6_-tM?rlkey=p5kvk9ns1bk6ejjnrjg7s3ndo&st=nueo2y07&dl=0

EOF

if [[ "${DOWNLOAD_CAT_ARTIFACTS:-0}" == "1" ]]; then
  echo "[prepare] Attempting CAT artifact download with gdown"
  python -m pip install --user gdown
  python -m gdown --folder "https://drive.google.com/drive/folders/1xVQ84pF5clVtKw6d4NCC-0mYbo4cIZ_a" -O "${ARTIFACTS_DIR}/cat"
fi

echo "[prepare] Done."
