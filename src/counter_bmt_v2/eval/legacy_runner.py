"""Legacy Adv-BMT subprocess runner for head-to-head eval."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Sequence

from .types import ModelRunResult, ModelSpec, ScenarioSubsetEntry, model_spec_hashable_dict


def _artifact_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.npz"))


def run_legacy_model(
    *,
    spec: ModelSpec,
    dataset_dir: Path,
    subset_file: Path,
    subset: Sequence[ScenarioSubsetEntry],
    out_dir: Path,
    run_seed: int,
    reuse_artifacts: bool,
    required: bool,
) -> ModelRunResult:
    model_dir = out_dir / str(spec.id)
    artifact_dir = model_dir / "step_eval"
    summary_path = model_dir / "runner_summary.json"
    stdout_path = model_dir / "worker.stdout.log"
    stderr_path = model_dir / "worker.stderr.log"
    cache_meta = model_dir / "artifact_meta.json"
    model_dir.mkdir(parents=True, exist_ok=True)

    subset_rel = [str(s.relative_path) for s in subset]
    subset_hash = hashlib.sha256(json.dumps(subset_rel, sort_keys=True).encode("utf-8")).hexdigest()
    spec_hash = json.dumps(model_spec_hashable_dict(spec), sort_keys=True)

    if reuse_artifacts and cache_meta.is_file() and artifact_dir.is_dir():
        try:
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        files = _artifact_files(artifact_dir)
        if (
            meta.get("subset_hash") == subset_hash
            and meta.get("spec_hash") == spec_hash
            and int(meta.get("num_artifacts", -1)) == len(subset)
            and len(files) == len(subset)
        ):
            summary_path.write_text(
                json.dumps(
                    {
                        "model_id": str(spec.id),
                        "backend": "legacy_adv_bmt",
                        "skipped_inference": True,
                        "reason": "artifact_reuse",
                        "artifact_dir": str(artifact_dir),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return ModelRunResult(
                model_id=str(spec.id),
                backend="legacy_adv_bmt",
                artifact_dir=str(artifact_dir),
                summary_path=str(summary_path),
                log_path=str(stdout_path),
                stderr_path=str(stderr_path),
            )

    worker_script = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "run_legacy_model_worker.py"
    cmd = [
        str(spec.runtime.python_bin or "python"),
        str(worker_script),
        "--legacy-root",
        str(spec.runtime.legacy_root or "src/Adv-BMT"),
        "--ckpt",
        str(spec.checkpoint),
        "--dataset-dir",
        str(dataset_dir),
        "--scenario-indices-file",
        str(subset_file),
        "--output-artifact-dir",
        str(artifact_dir),
        "--num-modes",
        str(int(spec.runtime.num_modes)),
        "--sampling-method",
        str(spec.runtime.sampling_method),
        "--temperature",
        str(float(spec.runtime.temperature)),
        "--topp",
        str(float(spec.runtime.topp)),
        "--skip-steps",
        str(int(spec.runtime.skip_steps)),
        "--seed",
        str(int(run_seed)),
        "--output-json",
        str(summary_path),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = "src" + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    proc = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[3]),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    if proc.returncode != 0:
        reason = f"legacy worker failed rc={proc.returncode}"
        if required:
            raise RuntimeError(f"{reason}. stderr: {(proc.stderr or '').strip()[-800:]}")
        return ModelRunResult(
            model_id=str(spec.id),
            backend="legacy_adv_bmt",
            artifact_dir=str(artifact_dir),
            summary_path=str(summary_path),
            skipped=True,
            reason=reason,
            log_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )

    files = _artifact_files(artifact_dir)
    cache_meta.write_text(
        json.dumps(
            {
                "subset_hash": subset_hash,
                "spec_hash": spec_hash,
                "num_artifacts": int(len(files)),
                "checkpoint": str(spec.checkpoint),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ModelRunResult(
        model_id=str(spec.id),
        backend="legacy_adv_bmt",
        artifact_dir=str(artifact_dir),
        summary_path=str(summary_path),
        log_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )
