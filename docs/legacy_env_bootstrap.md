# Legacy Adv-BMT Environment Bootstrap

Use a separate environment for legacy Adv-BMT work. The dependency graph is
different enough from `counter_bmt_v2` that sharing one venv is brittle.

The bootstrap script is:

- [bootstrap_legacy_adv_bmt.sh](/Users/joshuaflashner/Projects/CounterBMT/tools/bootstrap_legacy_adv_bmt.sh)

## Default Goal

The default goal is a **train-only legacy environment** suitable for:

- `src/Adv-BMT/bmt/train_motion.py`
- the paired MidGPT learning probe
- legacy checkpoint loading
- head-to-head evaluation through the repo's wrapper scripts

It does **not** install the heavy TensorFlow/Waymo evaluator stack unless you
opt in.

## Recommended H200 Usage

On a Linux H200 box:

```bash
VENV_DIR=.venv-legacy-adv-bmt \
LEGACY_PROFILE=linux-cu121 \
INSTALL_SIM_STACK=1 \
INSTALL_WAYMO_EVAL=0 \
tools/bootstrap_legacy_adv_bmt.sh
```

`PYTHON_BIN` is optional. If unset, the bootstrap auto-detects the first
available compatible interpreter from `python3.10`, `python3.11`, `python3`,
then `python`.

If the host only exposes newer interpreters such as Python `3.12`, the legacy
bootstrap will, by default, fall back to `uv` and provision a managed Python
`3.10` interpreter automatically. That avoids source builds for older pinned
packages like `Pillow 9.2.0`, which are much more fragile on Python `3.12`.

This will:

1. create a dedicated venv
2. install the pinned legacy PyTorch/Lightning stack
3. install matching PyG operator wheels
4. install `metadrive` and `scenarionet`
5. re-apply the pinned legacy base requirements so broad external dependency
   ranges do not upgrade core packages like NumPy behind our backs
6. verify the resulting environment

The dedicated legacy env is intentionally pinned to `numpy==1.26.4` because the
released editable `adv-bmt` package declares `numpy>=1.26,<2`. That keeps the
legacy stack separate from the main v2 JAX environment's newer NumPy line.

If a previous bootstrap attempt failed partway through dependency resolution,
rerun with a clean env:

```bash
RECREATE_VENV=1 tools/bootstrap_legacy_adv_bmt.sh
```

If you want to force a particular legacy Python target, set:

```bash
LEGACY_PYTHON_SPEC=3.11 tools/bootstrap_legacy_adv_bmt.sh
```

## Optional Full Legacy Evaluator

If you need the original legacy validation/evaluation stack too:

```bash
INSTALL_WAYMO_EVAL=1 tools/bootstrap_legacy_adv_bmt.sh
```

That adds:

- `tensorflow`
- `tensorflow-addons`
- `tensorflow-datasets`
- `tensorflow-probability`
- `waymo-open-dataset-tf-2-12-0`

## Profiles

- `linux-cu121`
  Best choice for the H200 box.
- `linux-cpu`
  CPU/debug environment on Linux.
- `mac-cpu`
  Best-effort local smoke environment. This skips the compiled PyG extension
  family because those wheels are not reliably available on macOS.

## Reusing Existing MetaDrive / ScenarioNet Clones

If those repos are already checked out elsewhere, point the bootstrap at them:

```bash
METADRIVE_SRC=/path/to/metadrive \
SCENARIONET_SRC=/path/to/scenarionet \
tools/bootstrap_legacy_adv_bmt.sh
```

Otherwise the script clones them under `.external/legacy_deps/`.

## Pinning External Refs

If you want to pin the external repos to specific commits or tags:

```bash
METADRIVE_REF=<git-ref> \
SCENARIONET_REF=<git-ref> \
tools/bootstrap_legacy_adv_bmt.sh
```

## Dry Run

To inspect the exact commands without changing anything:

```bash
DRY_RUN=1 tools/bootstrap_legacy_adv_bmt.sh
```
