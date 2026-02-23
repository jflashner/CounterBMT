# P6 Unified Parity Harness + One-Command Gate

This document defines the unified parity runner for P0-P5 checks.

## What was added

- New orchestrator: `src/scripts/parity/parity_report.py`
  - Runs P0-P5 gates.
  - Captures per-gate logs + JSON payloads.
  - Writes consolidated JSON + Markdown reports.
  - Supports profile + policy controls (`quick/full/remote`, legacy policy, P5 policy).
- New one-command wrapper: `tools/run_parity_suite.sh`
  - Defaults to `PYTHONPATH=src` if unset.
  - Uses `$PYTHON_BIN` when provided.
  - Writes canonical:
    - `outputs/parity_report/latest.json`
    - `outputs/parity_report/latest.md`
- Added `--output-json` to:
  - `src/scripts/parity/inspect_decoder_masks.py`
  - `src/scripts/parity/compare_decoder_inputs.py`
  - `src/scripts/parity/check_lr_schedule.py`
  - `src/scripts/parity/check_resume_determinism.py`
- Added `--num-train-scenarios` passthrough to:
  - `src/scripts/parity/benchmark_throughput.py`

## Default behavior

- Profile: `quick`
- Legacy policy: `required_if_available`
- P5 policy: `pass_with_waiver`

`quick` runs P0-P4 + P5 LR schedule, and records P5 throughput/resume as explicit waivers (with thresholds and evidence text).

## Commands

1. Default quick run
```bash
bash tools/run_parity_suite.sh
```

2. Quick run with explicit interpreter + existing P4 artifacts
```bash
PYTHON_BIN=.venv/bin/python \
bash tools/run_parity_suite.sh \
  --forward-artifact-dir outputs/p4_smoke_approx/forward_eval_artifacts
```

3. Legacy unavailable behavior (default policy)
```bash
bash tools/run_parity_suite.sh --legacy-root /tmp/missing_legacy
```

4. Strict legacy policy
```bash
bash tools/run_parity_suite.sh \
  --legacy-policy required \
  --legacy-root /tmp/missing_legacy
```

5. Full profile with strict P5 policy
```bash
bash tools/run_parity_suite.sh --profile full --p5-policy strict_fail
```

## Report artifacts

Each run writes:
- `outputs/parity_report/run_<timestamp>/report.json`
- `outputs/parity_report/run_<timestamp>/report.md`
- `outputs/parity_report/run_<timestamp>/logs/*.log`
- `outputs/parity_report/run_<timestamp>/results/*.json`

Canonical latest pointers:
- `outputs/parity_report/latest.json`
- `outputs/parity_report/latest.md`

## Report schema (top-level)

- `suite_version`
- `timestamp_utc`
- `config`
- `environment`
- `overall` (`pass|fail|pass_with_waiver`)
- `phase_summary`
- `gates`
- `waivers`
- `artifacts`

Per-gate fields include:
- `id`, `phase`, `name`, `required`, `status`
- `reason`, `command`, `return_code`
- `started_at_utc`, `duration_sec`
- `stdout_log`, `stderr_log`, `json_result_path`
- `metrics_excerpt`, `thresholds`, `artifact_links`

## Current smoke result

Validated command:
```bash
PYTHON_BIN=.venv/bin/python \
bash tools/run_parity_suite.sh \
  --forward-artifact-dir outputs/p4_smoke_approx/forward_eval_artifacts
```

Observed:
- `overall = pass_with_waiver`
- `latest.json` and `latest.md` written successfully.
- P5 quick-profile waivers recorded explicitly.
