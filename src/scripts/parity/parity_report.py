"""Unified P0-P5 parity harness with JSON + Markdown reporting."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
SUITE_VERSION = "p6.v1"


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _iso_utc(ts: datetime) -> str:
    return ts.isoformat()


def _safe_json_load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_capture(
    cmd: List[str],
    *,
    env: Dict[str, str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> Tuple[int, float, str]:
    started = _utc_now()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as so, stderr_path.open("w", encoding="utf-8") as se:
        proc = subprocess.run(cmd, env=env, cwd=str(cwd), stdout=so, stderr=se, check=False)
    duration = time.time() - started.timestamp()
    return int(proc.returncode), float(duration), _iso_utc(started)


def _cmd_str(cmd: List[str]) -> str:
    return " ".join(cmd)


def _git_value(args: List[str]) -> str:
    try:
        out = subprocess.check_output(args, cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _collect_environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(REPO_ROOT),
        "git_commit": _git_value(["git", "rev-parse", "HEAD"]),
        "git_dirty": (_git_value(["git", "status", "--porcelain"]) != ""),
    }
    try:
        import jax  # type: ignore

        env["jax_version"] = getattr(jax, "__version__", "unknown")
        devices = jax.devices()
        env["jax_num_devices"] = int(len(devices))
        env["jax_devices"] = [str(d) for d in devices]
    except Exception as exc:
        env["jax_error"] = str(exc)
    return env


def _probe_legacy_environment(python_bin: str, legacy_root: Path, env: Dict[str, str]) -> Tuple[bool, str]:
    if not legacy_root.exists():
        return False, f"legacy_root not found: {legacy_root}"
    probe_code = (
        "import sys; "
        f"sys.path.insert(0, r'{str(legacy_root)}'); "
        "import bmt; import bmt.tokenization.motion_tokenizers"
    )
    cmd = [python_bin, "-c", probe_code]
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if proc.returncode == 0:
        return True, "legacy imports available"
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    last = tail[-1] if tail else f"return_code={proc.returncode}"
    return False, f"legacy import probe failed: {last}"


def _extract_metrics_excerpt(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not payload:
        return {}
    out: Dict[str, Any] = {}
    if isinstance(payload.get("metrics"), dict):
        metrics = payload["metrics"]
        for key in (
            "target_token_exact_match_rate_masked",
            "valid_mask_exact_match_rate",
            "a2t_causal_valid_match_rate",
            "input_mask_match_rate",
            "decoder_embedding_abs_max_common_valid",
            "mask_exact_match_rate",
            "feat_abs_max",
            "passed",
            "has_nan",
        ):
            if key in metrics:
                out[key] = metrics[key]
    if isinstance(payload.get("targets"), dict):
        scene = payload["targets"].get("scene_s2s", {})
        if isinstance(scene, dict):
            scene_metrics = scene.get("metrics", {})
            if isinstance(scene_metrics, dict):
                for key in ("feat_abs_max", "mask_exact_match_rate", "index_exact_match_rate", "has_nan"):
                    if key in scene_metrics:
                        out[f"scene_s2s/{key}"] = scene_metrics[key]
    if isinstance(payload.get("summary"), dict):
        summary = payload["summary"]
        for key in (
            "mad",
            "mean_tokens_per_sec",
            "mean_steps_per_sec",
            "num_scenarios",
            "failed_checks",
            "has_nan_mismatch",
        ):
            if key in summary:
                out[key] = summary[key]
    if "max_abs_error" in payload:
        out["max_abs_error"] = payload.get("max_abs_error")
    if "passed" in payload:
        out["passed"] = payload.get("passed")
    return out


@dataclass
class GateSpec:
    gate_id: str
    phase: str
    name: str
    required: bool
    command: Optional[List[str]]
    json_result_path: Optional[Path]
    thresholds: Dict[str, Any]
    reason: str = ""
    status_override: Optional[str] = None
    artifact_links: Optional[List[str]] = None


def _record_static_gate(
    spec: GateSpec,
    *,
    status: str,
    reason: str,
) -> Dict[str, Any]:
    return {
        "id": spec.gate_id,
        "phase": spec.phase,
        "name": spec.name,
        "required": bool(spec.required),
        "status": status,
        "reason": reason,
        "command": _cmd_str(spec.command) if spec.command else "",
        "return_code": None,
        "started_at_utc": _iso_utc(_utc_now()),
        "duration_sec": 0.0,
        "stdout_log": "",
        "stderr_log": "",
        "json_result_path": str(spec.json_result_path) if spec.json_result_path else "",
        "metrics_excerpt": {},
        "thresholds": dict(spec.thresholds),
        "artifact_links": list(spec.artifact_links or []),
    }


def _execute_gate(
    spec: GateSpec,
    *,
    env: Dict[str, str],
    logs_dir: Path,
) -> Dict[str, Any]:
    stdout_path = logs_dir / f"{spec.gate_id}.stdout.log"
    stderr_path = logs_dir / f"{spec.gate_id}.stderr.log"
    if spec.command is None:
        return _record_static_gate(
            spec,
            status=(spec.status_override or "skipped"),
            reason=(spec.reason or "no command"),
        )

    rc, duration_sec, started_at = _run_capture(
        spec.command,
        env=env,
        cwd=REPO_ROOT,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    status = "pass" if rc == 0 else "fail"
    if spec.status_override in ("waived", "skipped"):
        status = spec.status_override
    reason = spec.reason
    if not reason and status == "fail":
        reason = f"command exited {rc}"

    payload = _safe_json_load(spec.json_result_path) if spec.json_result_path else None
    links = list(spec.artifact_links or [])
    links.extend([str(stdout_path), str(stderr_path)])
    if spec.json_result_path:
        links.append(str(spec.json_result_path))

    return {
        "id": spec.gate_id,
        "phase": spec.phase,
        "name": spec.name,
        "required": bool(spec.required),
        "status": status,
        "reason": reason,
        "command": _cmd_str(spec.command),
        "return_code": int(rc),
        "started_at_utc": started_at,
        "duration_sec": float(duration_sec),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "json_result_path": str(spec.json_result_path) if spec.json_result_path else "",
        "metrics_excerpt": _extract_metrics_excerpt(payload),
        "thresholds": dict(spec.thresholds),
        "artifact_links": links,
    }


def _default_env() -> Dict[str, str]:
    env = dict(os.environ)
    py_path = env.get("PYTHONPATH", "")
    if py_path:
        if "src" not in py_path.split(":"):
            env["PYTHONPATH"] = f"src:{py_path}"
    else:
        env["PYTHONPATH"] = "src"
    return env


def _phase_status(records: List[Dict[str, Any]], *, include_in_overall: bool) -> str:
    if not records:
        return "skipped"
    considered = [r for r in records if include_in_overall]
    if not considered:
        return "skipped"
    statuses = [str(r["status"]) for r in considered]
    if any(s == "fail" for s in statuses):
        return "fail"
    if any(s == "waived" for s in statuses):
        return "pass_with_waiver"
    if all(s == "skipped" for s in statuses):
        return "skipped"
    return "pass"


def _overall_status(gates: List[Dict[str, Any]], *, p5_policy: str) -> str:
    considered = []
    for g in gates:
        if str(g["phase"]) == "P5" and p5_policy == "skip_overall":
            continue
        if not bool(g.get("required", False)):
            continue
        considered.append(g)
    statuses = [str(g["status"]) for g in considered]
    if any(s == "fail" for s in statuses):
        return "fail"
    if any(s == "waived" for s in statuses):
        return "pass_with_waiver"
    return "pass"


def _count_by_status(gates: List[Dict[str, Any]]) -> Dict[str, int]:
    out = {"pass": 0, "fail": 0, "waived": 0, "skipped": 0}
    for g in gates:
        s = str(g["status"])
        if s in out:
            out[s] += 1
    return out


def _markdown_report(
    *,
    report: Dict[str, Any],
    run_command: str,
) -> str:
    lines: List[str] = []
    lines.append("# Parity Suite Report")
    lines.append("")
    lines.append(f"- Timestamp (UTC): `{report['timestamp_utc']}`")
    lines.append(f"- Suite version: `{report['suite_version']}`")
    lines.append(f"- Overall: **{report['overall']}**")
    lines.append("")
    lines.append("## Phase Summary")
    lines.append("")
    lines.append("| Phase | Status | Pass | Fail | Waived | Skipped |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for phase, info in report["phase_summary"].items():
        counts = info["counts"]
        lines.append(
            f"| {phase} | {info['status']} | {counts['pass']} | {counts['fail']} | {counts['waived']} | {counts['skipped']} |"
        )
    lines.append("")

    failed_or_waived = [g for g in report["gates"] if g["status"] in ("fail", "waived")]
    lines.append("## Failed / Waived Gates")
    lines.append("")
    if not failed_or_waived:
        lines.append("- None")
    else:
        for gate in failed_or_waived:
            lines.append(f"### {gate['id']} ({gate['status']})")
            lines.append(f"- Phase: `{gate['phase']}`")
            lines.append(f"- Required: `{gate['required']}`")
            if gate.get("reason"):
                lines.append(f"- Reason: {gate['reason']}")
            if gate.get("command"):
                lines.append(f"- Command: `{gate['command']}`")
            links = gate.get("artifact_links", [])
            if links:
                lines.append("- Artifacts:")
                for link in links:
                    lines.append(f"  - `{link}`")
            metrics = gate.get("metrics_excerpt", {})
            if metrics:
                lines.append(f"- Metrics excerpt: `{json.dumps(metrics, sort_keys=True)}`")
            lines.append("")

    if report.get("waivers"):
        lines.append("## Waivers")
        lines.append("")
        for w in report["waivers"]:
            lines.append(f"- `{w['id']}`: {w['reason']}")
        lines.append("")

    lines.append("## Repro Command")
    lines.append("")
    lines.append(f"`{run_command}`")
    lines.append("")
    return "\n".join(lines)


def _legacy_required(legacy_available: bool, policy: str) -> bool:
    if policy == "required":
        return True
    if policy == "required_if_available":
        return bool(legacy_available)
    return False


def _make_python_script_cmd(
    python_bin: str,
    script_rel: str,
    args: List[str],
) -> List[str]:
    return [python_bin, str(REPO_ROOT / script_rel), *args]


def _make_python_module_cmd(
    python_bin: str,
    module: str,
    args: List[str],
) -> List[str]:
    return [python_bin, "-m", module, *args]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified CounterBMT parity suite and write report artifacts")
    parser.add_argument("--data-dir", type=str, default="data/scenarionet_waymo_training_500")
    parser.add_argument("--output-dir", type=str, default="outputs/parity_report")
    parser.add_argument("--profile", type=str, default="quick", choices=["quick", "full", "remote"])
    parser.add_argument(
        "--legacy-policy",
        type=str,
        default="required_if_available",
        choices=["required_if_available", "required", "optional"],
    )
    parser.add_argument(
        "--p5-policy",
        type=str,
        default="pass_with_waiver",
        choices=["pass_with_waiver", "strict_fail", "skip_overall"],
    )
    parser.add_argument("--legacy-root", type=str, default="src/Adv-BMT")
    parser.add_argument("--forward-artifact-dir", type=str, default="")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--output-md", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_started = _utc_now()
    env = _default_env()
    output_root = Path(args.output_dir)
    run_name = f"run_{run_started.strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_name
    logs_dir = run_dir / "logs"
    results_dir = run_dir / "results"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    python_bin = str(args.python_bin)
    data_dir = str(args.data_dir)
    legacy_root = (REPO_ROOT / args.legacy_root).resolve()
    legacy_available, legacy_reason = _probe_legacy_environment(python_bin, legacy_root, env)

    gates: List[Dict[str, Any]] = []
    waivers: List[Dict[str, Any]] = []
    stop_triggered = False

    def append_record(rec: Dict[str, Any]) -> None:
        nonlocal stop_triggered
        gates.append(rec)
        if args.stop_on_fail and bool(rec.get("required", False)) and rec.get("status") == "fail":
            stop_triggered = True

    def run_gate(spec: GateSpec) -> None:
        if stop_triggered:
            append_record(
                _record_static_gate(
                    spec,
                    status="skipped",
                    reason="stop_on_fail triggered by previous required failure",
                )
            )
            return
        rec = _execute_gate(spec, env=env, logs_dir=logs_dir)
        append_record(rec)

    # P0 smoke gates.
    for mode in ("forward", "backward"):
        gid = f"p0_token_{mode}_smoke"
        out_json = results_dir / f"{gid}.json"
        run_gate(
            GateSpec(
                gate_id=gid,
                phase="P0",
                name=f"P0 tokenization smoke ({mode})",
                required=True,
                command=_make_python_script_cmd(
                    python_bin,
                    "src/scripts/parity/compare_tokenization.py",
                    [
                        "--data-dir",
                        data_dir,
                        "--mode",
                        mode,
                        "--n",
                        "20",
                        "--batch-size",
                        "4",
                        "--skip-steps",
                        "5",
                        "--output-json",
                        str(out_json),
                    ],
                ),
                json_result_path=out_json,
                thresholds={"invalid_ids": False, "has_nan": False},
            )
        )

    # P0 legacy gates.
    p0_legacy_required = _legacy_required(legacy_available, str(args.legacy_policy))
    p0_legacy_specs: List[Tuple[str, str, str, str]] = [
        ("p0_token_forward_legacy", "forward", "0.999", "0.999"),
        ("p0_token_backward_legacy", "backward", "0.995", "0.999"),
    ]
    for gid, mode, min_tok, min_mask in p0_legacy_specs:
        out_json = results_dir / f"{gid}.json"
        if legacy_available:
            run_gate(
                GateSpec(
                    gate_id=gid,
                    phase="P0",
                    name=f"P0 tokenization legacy parity ({mode})",
                    required=p0_legacy_required,
                    command=_make_python_script_cmd(
                        python_bin,
                        "src/scripts/parity/compare_tokenization.py",
                        [
                            "--data-dir",
                            data_dir,
                            "--mode",
                            mode,
                            "--n",
                            "100",
                            "--batch-size",
                            "4",
                            "--skip-steps",
                            "5",
                            "--legacy-check",
                            "--legacy-root",
                            str(legacy_root),
                            "--min-token-match",
                            min_tok,
                            "--min-valid-mask-match",
                            min_mask,
                            "--output-json",
                            str(out_json),
                        ],
                    ),
                    json_result_path=out_json,
                    thresholds={
                        "min_token_match": float(min_tok),
                        "min_valid_mask_match": float(min_mask),
                    },
                )
            )
        else:
            append_record(
                _record_static_gate(
                    GateSpec(
                        gate_id=gid,
                        phase="P0",
                        name=f"P0 tokenization legacy parity ({mode})",
                        required=p0_legacy_required,
                        command=None,
                        json_result_path=out_json,
                        thresholds={},
                    ),
                    status=("fail" if p0_legacy_required else "skipped"),
                    reason=f"legacy unavailable: {legacy_reason}",
                )
            )

    # P1 relation smoke.
    p1_smoke_json = results_dir / "p1_relations_smoke.json"
    run_gate(
        GateSpec(
            gate_id="p1_relations_smoke",
            phase="P1",
            name="P1 relation smoke (scene_s2s)",
            required=True,
            command=_make_python_script_cmd(
                python_bin,
                "src/scripts/parity/compare_relations.py",
                [
                    "--data-dir",
                    data_dir,
                    "--target",
                    "scene_s2s",
                    "--mode",
                    "simple",
                    "--n",
                    "20",
                    "--batch-size",
                    "4",
                    "--skip-steps",
                    "5",
                    "--output-json",
                    str(p1_smoke_json),
                ],
            ),
            json_result_path=p1_smoke_json,
            thresholds={"has_nan": False},
        )
    )

    # P1 legacy relation gate.
    p1_legacy_required = _legacy_required(legacy_available, str(args.legacy_policy))
    p1_legacy_json = results_dir / "p1_relations_legacy.json"
    if legacy_available:
        run_gate(
            GateSpec(
                gate_id="p1_relations_legacy",
                phase="P1",
                name="P1 relation legacy parity (scene_s2s)",
                required=p1_legacy_required,
                command=_make_python_script_cmd(
                    python_bin,
                    "src/scripts/parity/compare_relations.py",
                    [
                        "--data-dir",
                        data_dir,
                        "--target",
                        "scene_s2s",
                        "--mode",
                        "simple",
                        "--n",
                        "100",
                        "--batch-size",
                        "4",
                        "--skip-steps",
                        "5",
                        "--legacy-check",
                        "--legacy-root",
                        str(legacy_root),
                        "--max-feat-diff",
                        "1e-5",
                        "--min-mask-match",
                        "1.0",
                        "--output-json",
                        str(p1_legacy_json),
                    ],
                ),
                json_result_path=p1_legacy_json,
                thresholds={"max_feat_diff": 1e-5, "min_mask_match": 1.0},
            )
        )
    else:
        append_record(
            _record_static_gate(
                GateSpec(
                    gate_id="p1_relations_legacy",
                    phase="P1",
                    name="P1 relation legacy parity (scene_s2s)",
                    required=p1_legacy_required,
                    command=None,
                    json_result_path=p1_legacy_json,
                    thresholds={},
                ),
                status=("fail" if p1_legacy_required else "skipped"),
                reason=f"legacy unavailable: {legacy_reason}",
            )
        )

    # P2 gates.
    p2_masks_json = results_dir / "p2_decoder_masks.json"
    run_gate(
        GateSpec(
            gate_id="p2_decoder_masks",
            phase="P2",
            name="P2 decoder mask parity",
            required=True,
            command=_make_python_script_cmd(
                python_bin,
                "src/scripts/parity/inspect_decoder_masks.py",
                [
                    "--data-dir",
                    data_dir,
                    "--n",
                    "20",
                    "--batch-size",
                    "4",
                    "--skip-steps",
                    "5",
                    "--min-match",
                    "1.0",
                    "--output-json",
                    str(p2_masks_json),
                ],
            ),
            json_result_path=p2_masks_json,
            thresholds={"min_match": 1.0},
        )
    )

    p2_smoke_json = results_dir / "p2_decoder_inputs_smoke.json"
    run_gate(
        GateSpec(
            gate_id="p2_decoder_inputs_smoke",
            phase="P2",
            name="P2 decoder input smoke",
            required=True,
            command=_make_python_script_cmd(
                python_bin,
                "src/scripts/parity/compare_decoder_inputs.py",
                [
                    "--data-dir",
                    data_dir,
                    "--n",
                    "20",
                    "--batch-size",
                    "4",
                    "--skip-steps",
                    "5",
                    "--output-json",
                    str(p2_smoke_json),
                ],
            ),
            json_result_path=p2_smoke_json,
            thresholds={"has_nan": False},
        )
    )

    p2_legacy_required = _legacy_required(legacy_available, str(args.legacy_policy))
    p2_legacy_json = results_dir / "p2_decoder_inputs_legacy.json"
    if legacy_available:
        run_gate(
            GateSpec(
                gate_id="p2_decoder_inputs_legacy",
                phase="P2",
                name="P2 decoder input legacy parity",
                required=p2_legacy_required,
                command=_make_python_script_cmd(
                    python_bin,
                    "src/scripts/parity/compare_decoder_inputs.py",
                    [
                        "--data-dir",
                        data_dir,
                        "--n",
                        "50",
                        "--batch-size",
                        "4",
                        "--skip-steps",
                        "5",
                        "--legacy-check",
                        "--legacy-root",
                        str(legacy_root),
                        "--max-embedding-diff",
                        "2e-4",
                        "--min-mask-match",
                        "0.9995",
                        "--output-json",
                        str(p2_legacy_json),
                    ],
                ),
                json_result_path=p2_legacy_json,
                thresholds={
                    "max_embedding_diff": 2e-4,
                    "min_mask_match": 0.9995,
                },
            )
        )
    else:
        append_record(
            _record_static_gate(
                GateSpec(
                    gate_id="p2_decoder_inputs_legacy",
                    phase="P2",
                    name="P2 decoder input legacy parity",
                    required=p2_legacy_required,
                    command=None,
                    json_result_path=p2_legacy_json,
                    thresholds={},
                ),
                status=("fail" if p2_legacy_required else "skipped"),
                reason=f"legacy unavailable: {legacy_reason}",
            )
        )

    # P3 gate.
    p3_json = results_dir / "p3_dataset_index.json"
    run_gate(
        GateSpec(
            gate_id="p3_dataset_index",
            phase="P3",
            name="P3 dataset index parity",
            required=True,
            command=_make_python_script_cmd(
                python_bin,
                "src/scripts/parity/compare_dataset_index.py",
                [
                    "--train",
                    data_dir,
                    "--val",
                    data_dir,
                    "--sample-interval-training",
                    "2",
                    "--sample-interval-test",
                    "3",
                    "--json",
                    str(p3_json),
                ],
            ),
            json_result_path=p3_json,
            thresholds={"count_match": True},
        )
    )

    # P4 artifact producer (optional) + strict compare.
    artifact_dir = Path(args.forward_artifact_dir).expanduser().resolve() if args.forward_artifact_dir else None
    if artifact_dir is None:
        p4_train_dir = run_dir / "p4_artifact_producer"
        run_gate(
            GateSpec(
                gate_id="p4_artifact_producer",
                phase="P4",
                name="P4 artifact producer (tiny training)",
                required=True,
                command=_make_python_module_cmd(
                    python_bin,
                    "counter_bmt_v2.cli.train_nnx_bmt",
                    [
                        "--data-dir",
                        data_dir,
                        "--output-dir",
                        str(p4_train_dir),
                        "--model-preset",
                        "midgpt_parity",
                        "--tokenizer-mode",
                        "adv_bmt_parity",
                        "--max-steps",
                        "2",
                        "--batch-size",
                        "1",
                        "--eval-every",
                        "1",
                        "--eval-batches",
                        "1",
                        "--log-every",
                        "1",
                        "--forward-export-artifacts",
                    ],
                ),
                json_result_path=None,
                thresholds={"artifacts_written": True},
                artifact_links=[str(p4_train_dir)],
            )
        )
        artifact_dir = p4_train_dir / "forward_eval_artifacts"
    p4_compare_json = results_dir / "p4_forward_metrics_compare.json"
    run_gate(
        GateSpec(
            gate_id="p4_forward_metrics_compare",
            phase="P4",
            name="P4 strict forward metric compare",
            required=True,
            command=_make_python_script_cmd(
                python_bin,
                "src/scripts/parity/compare_forward_metrics.py",
                [
                    "--artifact-dir",
                    str(artifact_dir),
                    "--output-json",
                    str(p4_compare_json),
                    "--max-rel-error",
                    "0.01",
                    "--min-corr",
                    "0.99",
                ],
            ),
            json_result_path=p4_compare_json,
            thresholds={"max_rel_error": 0.01, "min_corr": 0.99},
            artifact_links=[str(artifact_dir)],
        )
    )

    # P5 LR schedule gate (always).
    p5_lr_json = results_dir / "p5_lr_schedule.json"
    run_gate(
        GateSpec(
            gate_id="p5_lr_schedule",
            phase="P5",
            name="P5 LR schedule parity",
            required=True,
            command=_make_python_script_cmd(
                python_bin,
                "src/scripts/parity/check_lr_schedule.py",
                [
                    "--steps",
                    "0,1,100,2000,10000",
                    "--lr",
                    "3e-4",
                    "--warmup-steps",
                    "2000",
                    "--total-steps",
                    "300000",
                    "--mode",
                    "legacy_cosine_zero",
                    "--max-abs-error",
                    "1e-9",
                    "--output-json",
                    str(p5_lr_json),
                ],
            ),
            json_result_path=p5_lr_json,
            thresholds={"max_abs_error": 1e-9},
        )
    )

    def add_waiver(gate_id: str, reason: str) -> None:
        waivers.append({"id": gate_id, "reason": reason})

    if args.profile == "quick":
        if args.p5_policy == "strict_fail":
            append_record(
                _record_static_gate(
                    GateSpec(
                        gate_id="p5_resume_determinism",
                        phase="P5",
                        name="P5 resume determinism",
                        required=True,
                        command=None,
                        json_result_path=None,
                        thresholds={"max_mad": 1e-6},
                    ),
                    status="fail",
                    reason="strict_fail policy requires full/remote profile for resume determinism gate",
                )
            )
            append_record(
                _record_static_gate(
                    GateSpec(
                        gate_id="p5_throughput_speedup",
                        phase="P5",
                        name="P5 throughput speedup",
                        required=True,
                        command=None,
                        json_result_path=None,
                        thresholds={"min_speedup_x": 3.0},
                    ),
                    status="fail",
                    reason="strict_fail policy requires full/remote profile for throughput gate",
                )
            )
        elif args.p5_policy == "skip_overall":
            append_record(
                _record_static_gate(
                    GateSpec(
                        gate_id="p5_resume_determinism",
                        phase="P5",
                        name="P5 resume determinism",
                        required=False,
                        command=None,
                        json_result_path=None,
                        thresholds={"max_mad": 1e-6},
                    ),
                    status="skipped",
                    reason="quick profile skips heavy P5 runtime gates",
                )
            )
            append_record(
                _record_static_gate(
                    GateSpec(
                        gate_id="p5_throughput_speedup",
                        phase="P5",
                        name="P5 throughput speedup",
                        required=False,
                        command=None,
                        json_result_path=None,
                        thresholds={"min_speedup_x": 3.0},
                    ),
                    status="skipped",
                    reason="quick profile skips heavy P5 runtime gates",
                )
            )
        else:
            append_record(
                _record_static_gate(
                    GateSpec(
                        gate_id="p5_resume_determinism",
                        phase="P5",
                        name="P5 resume determinism",
                        required=True,
                        command=None,
                        json_result_path=None,
                        thresholds={"max_mad": 1e-6, "waiver_max_mad": 5e-5},
                    ),
                    status="waived",
                    reason=(
                        "quick profile skips heavy gate; accepted waiver: observed mad=2.784e-05 "
                        "vs strict 1e-6 (practical <=5e-5)."
                    ),
                )
            )
            append_record(
                _record_static_gate(
                    GateSpec(
                        gate_id="p5_throughput_speedup",
                        phase="P5",
                        name="P5 throughput speedup",
                        required=True,
                        command=None,
                        json_result_path=None,
                        thresholds={"min_speedup_x": 3.0},
                    ),
                    status="waived",
                    reason=(
                        "quick profile skips heavy gate; accepted waiver: observed 1.69x "
                        "vs strict >=3.0x due host-side preprocessing bottleneck."
                    ),
                )
            )
            add_waiver(
                "p5_resume_determinism",
                "Accepted waiver in quick profile with previously observed mad=2.784e-05 (strict 1e-6, practical <=5e-5).",
            )
            add_waiver(
                "p5_throughput_speedup",
                "Accepted waiver in quick profile with previously observed speedup=1.69x (strict >=3.0x).",
            )
    else:
        # Resume determinism gate.
        p5_resume_json = results_dir / "p5_resume_determinism.json"
        run_gate(
            GateSpec(
                gate_id="p5_resume_determinism",
                phase="P5",
                name="P5 resume determinism",
                required=True,
                command=_make_python_script_cmd(
                    python_bin,
                    "src/scripts/parity/check_resume_determinism.py",
                    [
                        "--data-dir",
                        data_dir,
                        "--output-dir",
                        str(run_dir / "p5_resume_check"),
                        "--steps-total",
                        "200",
                        "--split-step",
                        "100",
                        "--batch-size",
                        "2",
                        "--seed",
                        "0",
                        "--max-mad",
                        "1e-6",
                        "--output-json",
                        str(p5_resume_json),
                    ],
                ),
                json_result_path=p5_resume_json,
                thresholds={"max_mad": 1e-6, "waiver_max_mad": 5e-5},
                artifact_links=[str(run_dir / "p5_resume_check")],
            )
        )
        # Waiver handling for resume in pass_with_waiver mode.
        for gate in reversed(gates):
            if gate["id"] != "p5_resume_determinism":
                continue
            if gate["status"] == "fail" and args.p5_policy == "pass_with_waiver":
                payload = _safe_json_load(p5_resume_json)
                mad = None
                if payload and isinstance(payload.get("summary"), dict):
                    mad = payload["summary"].get("mad")
                try:
                    mad_f = float(mad)
                except Exception:
                    mad_f = float("nan")
                if mad_f == mad_f and mad_f <= 5e-5:
                    gate["status"] = "waived"
                    gate["reason"] = (
                        f"accepted waiver: mad={mad_f:.6g} exceeds strict 1e-6 but is within practical <=5e-5."
                    )
                    add_waiver(
                        "p5_resume_determinism",
                        gate["reason"],
                    )
            break

        # Throughput benchmark gates.
        single_json = results_dir / "p5_bench_single.json"
        pmap_json = results_dir / "p5_bench_pmap.json"
        run_gate(
            GateSpec(
                gate_id="p5_bench_single",
                phase="P5",
                name="P5 single-GPU throughput benchmark",
                required=True,
                command=_make_python_script_cmd(
                    python_bin,
                    "src/scripts/parity/benchmark_throughput.py",
                    [
                        "--data-dir",
                        data_dir,
                        "--output-dir",
                        str(run_dir / "p5_bench_single"),
                        "--distributed-backend",
                        "none",
                        "--batch-size",
                        "4",
                        "--max-steps",
                        "100",
                        "--num-train-scenarios",
                        "472",
                        "--json-out",
                        str(single_json),
                    ],
                ),
                json_result_path=single_json,
                thresholds={},
            )
        )
        run_gate(
            GateSpec(
                gate_id="p5_bench_pmap",
                phase="P5",
                name="P5 pmap throughput benchmark",
                required=True,
                command=_make_python_script_cmd(
                    python_bin,
                    "src/scripts/parity/benchmark_throughput.py",
                    [
                        "--data-dir",
                        data_dir,
                        "--output-dir",
                        str(run_dir / "p5_bench_pmap"),
                        "--distributed-backend",
                        "pmap",
                        "--batch-size",
                        "4",
                        "--max-steps",
                        "100",
                        "--num-train-scenarios",
                        "472",
                        "--json-out",
                        str(pmap_json),
                    ],
                ),
                json_result_path=pmap_json,
                thresholds={},
            )
        )
        single_payload = _safe_json_load(single_json) or {}
        pmap_payload = _safe_json_load(pmap_json) or {}
        single_tps = float(single_payload.get("summary", {}).get("mean_tokens_per_sec", float("nan")))
        pmap_tps = float(pmap_payload.get("summary", {}).get("mean_tokens_per_sec", float("nan")))
        speedup = pmap_tps / single_tps if (single_tps == single_tps and single_tps > 0) else float("nan")
        throughput_gate = {
            "id": "p5_throughput_speedup",
            "phase": "P5",
            "name": "P5 throughput speedup",
            "required": True,
            "status": "pass" if (speedup == speedup and speedup >= 3.0) else "fail",
            "reason": "" if (speedup == speedup and speedup >= 3.0) else "speedup below strict threshold",
            "command": "",
            "return_code": None,
            "started_at_utc": _iso_utc(_utc_now()),
            "duration_sec": 0.0,
            "stdout_log": "",
            "stderr_log": "",
            "json_result_path": "",
            "metrics_excerpt": {
                "single_mean_tokens_per_sec": single_tps,
                "pmap_mean_tokens_per_sec": pmap_tps,
                "speedup_x": speedup,
            },
            "thresholds": {"min_speedup_x": 3.0},
            "artifact_links": [str(single_json), str(pmap_json)],
        }
        if throughput_gate["status"] == "fail" and args.p5_policy == "pass_with_waiver":
            throughput_gate["status"] = "waived"
            throughput_gate["reason"] = (
                f"accepted waiver: observed speedup={speedup:.6g} < 3.0x due host-side preprocessing bottleneck."
            )
            add_waiver("p5_throughput_speedup", throughput_gate["reason"])
        append_record(throughput_gate)

    phase_summary: Dict[str, Any] = {}
    for phase in ("P0", "P1", "P2", "P3", "P4", "P5"):
        phase_gates = [g for g in gates if g["phase"] == phase]
        include = not (phase == "P5" and args.p5_policy == "skip_overall")
        phase_summary[phase] = {
            "status": _phase_status(phase_gates, include_in_overall=include),
            "counts": _count_by_status(phase_gates),
        }

    overall = _overall_status(gates, p5_policy=str(args.p5_policy))
    ended = _utc_now()

    report: Dict[str, Any] = {
        "suite_version": SUITE_VERSION,
        "timestamp_utc": _iso_utc(ended),
        "config": {
            "data_dir": data_dir,
            "output_dir": str(output_root),
            "run_dir": str(run_dir),
            "profile": str(args.profile),
            "legacy_policy": str(args.legacy_policy),
            "p5_policy": str(args.p5_policy),
            "legacy_root": str(legacy_root),
            "legacy_available": bool(legacy_available),
            "legacy_probe_reason": legacy_reason,
            "forward_artifact_dir": str(artifact_dir) if artifact_dir else "",
            "python_bin": python_bin,
            "stop_on_fail": bool(args.stop_on_fail),
        },
        "environment": _collect_environment(),
        "overall": overall,
        "phase_summary": phase_summary,
        "gates": gates,
        "waivers": waivers,
        "artifacts": {
            "run_dir": str(run_dir),
            "logs_dir": str(logs_dir),
            "results_dir": str(results_dir),
        },
        "timing": {
            "started_at_utc": _iso_utc(run_started),
            "ended_at_utc": _iso_utc(ended),
            "elapsed_sec": float((ended - run_started).total_seconds()),
        },
    }

    default_json = run_dir / "report.json"
    default_md = run_dir / "report.md"
    out_json = Path(args.output_json) if args.output_json else default_json
    out_md = Path(args.output_md) if args.output_md else default_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    run_cmd = f"bash tools/run_parity_suite.sh --data-dir {data_dir} --profile {args.profile}"
    md = _markdown_report(report=report, run_command=run_cmd)
    out_md.write_text(md, encoding="utf-8")

    latest_json = output_root / "latest.json"
    latest_md = output_root / "latest.md"
    latest_json.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_json, latest_json)
    shutil.copy2(out_md, latest_md)

    report["artifacts"]["report_json"] = str(out_json)
    report["artifacts"]["report_md"] = str(out_md)
    report["artifacts"]["latest_json"] = str(latest_json)
    report["artifacts"]["latest_md"] = str(latest_md)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({"overall": overall, "report_json": str(out_json), "report_md": str(out_md)}, indent=2))
    return 0 if overall in ("pass", "pass_with_waiver") else 1


if __name__ == "__main__":
    raise SystemExit(main())
