# Linux Transfer Guide (Reproducible Setup)

This project now has pinned dependency files and verification tools to reduce
cross-machine breakage.

## Files
- `requirements.txt`: pinned v2 stack (`src/counter_bmt_v2`)
- `requirements-legacy.txt`: pinned legacy Adv-BMT stack
- `requirements-installed-freeze.txt`: full package snapshot from source machine
- `tools/bootstrap_linux.sh`: create/setup venv and install pinned deps
- `tools/verify_environment.py`: strict version/import verification

## Recommended workflow on a new Linux machine
1. Install matching Python version (recommended: Python 3.10+).
2. Clone repo.
3. Initialize submodules:
```bash
git submodule update --init --recursive
```
4. Run:
```bash
tools/bootstrap_linux.sh v2
```
5. Verify manually again if needed:
```bash
.venv-v2/bin/python tools/verify_environment.py --profile v2
```

## Legacy/Adv-BMT workflow
```bash
tools/bootstrap_linux.sh legacy
```
Default venv: `.venv-legacy`

## Exact full-environment reproduction
Use only if you want to mirror the source machine as closely as possible:
```bash
tools/bootstrap_linux.sh full
```
Default venv: `.venv-full`

## Common failure points
1. GPU stack mismatch (NVIDIA driver / CUDA runtime vs pinned wheels).
2. Different Python minor version.
3. Reusing old `.venv` across machine moves.
4. Mixing `v2` and `legacy` installs into the same environment.

## Best practices
1. Use a fresh venv per profile (`v2` and `legacy` separately).
2. Keep `RECREATE_VENV=1` for clean rebuilds:
```bash
RECREATE_VENV=1 tools/bootstrap_linux.sh v2
```
3. Commit dependency updates only after re-running verification.
4. Keep submodule pointers clean before pushing:
```bash
git submodule status
```
