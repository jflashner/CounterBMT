from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    for path in (repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual import (
    ArtifactProvenance,
    CommitmentMetrics,
    ConflictAgentRef,
    InterventionContext,
    RecoveredDecision,
    TerminalPose,
    WindowSpec,
    analyze_conflicts,
    build_local_intervention_train_view,
    compile_control_code_from_local_intervention,
    discover_scenario_pickles,
    extract_local_patch,
    load_and_normalize_scenario,
    load_motion_config,
    load_raw_scenario,
    recover_ground_truth_branch,
    select_signalized_candidates_for_scenario,
    summarize_forward_supervision_for_raw_scenario,
)
from bmt.counterfactual.path_corpus import (
    DedupConfig,
    LightCanonicalizationConfig,
    PATH_LABELS,
    annotate_light_groups,
    apply_path_only_safe_mode,
    build_path_histograms,
    build_path_manifest,
    cluster_path_rows,
    summarize_path_rows,
    write_json,
    write_jsonl,
)
from scripts.counterfactual.mine_local_interventions import (
    _build_supervision_gates,
    _derive_compliance_label,
    _derive_entry_timing,
    _enumerate_branches,
    _infer_target_agent_alignment,
    _serialize_branches,
    _track_comes_within_stop_point,
    materialize_candidate_debug_bundle,
)
ARTIFACT_MODES = ("minimal", "sampled", "full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a lightweight path-control corpus index.")
    parser.add_argument("--scenario-root", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--max-scenarios", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--write-examples-manifest", action="store_true")
    parser.add_argument("--write-histograms", action="store_true")
    parser.add_argument("--write-summary", action="store_true")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--max-agents-per-candidate", type=int, default=0)
    parser.add_argument("--max-candidates-per-scenario", type=int, default=0)
    parser.add_argument("--artifact-mode", type=str, default="minimal", choices=ARTIFACT_MODES)
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--debug-sample-total", type=int, default=0)
    parser.add_argument("--debug-sample-per-class", type=int, default=0)
    parser.add_argument("--debug-sample-per-drop-reason", type=int, default=0)
    parser.add_argument("--dedup-mode", type=str, default="overlap_cluster", choices=("none", "overlap_cluster"))
    parser.add_argument("--decision-time-merge-frames", type=int, default=5)
    parser.add_argument("--window-overlap-threshold", type=float, default=0.5)
    parser.add_argument("--anchor-dist-threshold", type=float, default=5.0)
    parser.add_argument("--anchor-heading-threshold-rad", type=float, default=0.35)
    parser.add_argument("--path-only-safe-mode", dest="path_only_safe_mode", action="store_true")
    parser.add_argument("--no-path-only-safe-mode", dest="path_only_safe_mode", action="store_false")
    parser.add_argument("--target-agent-policy", type=str, default="all_trainable", choices=("all_trainable", "sdc_only"))
    parser.set_defaults(path_only_safe_mode=True)
    parser.set_defaults(progress=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start_time = time.perf_counter()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    light_config = LightCanonicalizationConfig()
    dedup_config = DedupConfig(
        mode=str(args.dedup_mode),
        decision_time_merge_frames=int(args.decision_time_merge_frames),
        window_overlap_threshold=float(args.window_overlap_threshold),
        anchor_dist_threshold=float(args.anchor_dist_threshold),
        anchor_heading_threshold_rad=float(args.anchor_heading_threshold_rad),
    )

    scenario_paths = discover_scenario_pickles(args.scenario_root)
    selected_paths = _select_scenarios(scenario_paths, max_scenarios=int(args.max_scenarios), seed=int(args.seed))
    worker_args = [
        {
            "scenario_pkl": str(path),
            "config_path": str(args.config or ""),
            "max_agents_per_candidate": int(args.max_agents_per_candidate),
            "max_candidates_per_scenario": int(args.max_candidates_per_scenario),
        }
        for path in selected_paths
    ]

    scenario_results = _run_scenario_jobs(
        worker_args=worker_args,
        num_workers=int(args.num_workers),
        show_progress=bool(args.progress),
    )

    entries: List[Dict[str, Any]] = []
    path_filter_drop_counts: Counter[str] = Counter()
    signalized_drop_counts: Counter[str] = Counter()
    prefilter_drop_counts: Counter[str] = Counter()
    debug_candidates: List[Dict[str, Any]] = []
    num_candidate_agents_considered = 0
    num_prefilter_pass = 0
    num_full_train_views_built = 0

    for result in scenario_results:
        entries.extend(result["entries"])
        path_filter_drop_counts.update(result["path_filter_drop_counts"])
        prefilter_drop_counts.update(result["prefilter_drop_counts"])
        debug_candidates.extend(result["debug_candidates"])
        num_candidate_agents_considered += int(result["num_candidate_agents_considered"])
        num_prefilter_pass += int(result["num_prefilter_pass"])
        num_full_train_views_built += int(result["num_full_train_views_built"])
        if result["signalized_primary_drop_reason"] is not None:
            signalized_drop_counts[str(result["signalized_primary_drop_reason"])] += 1

    raw_entries = sorted(
        entries,
        key=lambda item: (
            str(item["scenario_id"]),
            str(item["branch_label"]),
            str(item["agent_id"]),
            int(item["decision_time_idx"]),
            str(item.get("light_id")),
        ),
    )
    raw_entries, light_summary, light_group_hist = annotate_light_groups(raw_entries, config=light_config)
    raw_entries = sorted(
        raw_entries,
        key=lambda item: (
            str(item["scenario_id"]),
            str(item["branch_label"]),
            str(item["agent_id"]),
            str(item.get("light_group_id")),
            int(item["decision_time_idx"]),
            str(item.get("light_id")),
        ),
    )
    curated_dedup_rows, clustered_rows, dedup_stats = cluster_path_rows(raw_entries, config=dedup_config)
    curated_rows, curated_filter_summary = apply_path_only_safe_mode(
        curated_dedup_rows,
        enabled=bool(args.path_only_safe_mode),
    )
    curated_rows = sorted(
        curated_rows,
        key=lambda item: (
            str(item["scenario_id"]),
            str(item["branch_label"]),
            str(item["agent_id"]),
            str(item.get("light_group_id")),
            int(item["decision_time_idx"]),
            str(item.get("light_id")),
        ),
    )

    path_index_raw_path = outdir / "path_index_raw.jsonl"
    path_index_curated_path = outdir / "path_index_curated.jsonl"
    path_index_path = outdir / "path_index.jsonl"
    write_jsonl(path_index_raw_path, raw_entries)
    write_jsonl(path_index_curated_path, curated_rows)
    write_jsonl(path_index_path, curated_rows)

    prefilter_stats = {
        "num_scenarios_scanned": len(selected_paths),
        "num_candidate_agents_considered": int(num_candidate_agents_considered),
        "num_prefilter_pass": int(num_prefilter_pass),
        "num_prefilter_drop": int(sum(prefilter_drop_counts.values())),
        "drop_reason_counts": dict(sorted(prefilter_drop_counts.items())),
        "num_full_train_views_built": int(num_full_train_views_built),
        "num_path_examples_written": int(len(raw_entries)),
    }
    prefilter_stats_path = outdir / "prefilter_stats.json"
    write_json(prefilter_stats_path, prefilter_stats)

    raw_summary = summarize_path_rows(
        raw_entries,
        artifact_mode=str(args.artifact_mode),
        scenario_root=str(args.scenario_root),
        outdir=str(outdir),
        max_scenarios=int(args.max_scenarios),
        seed=int(args.seed),
        num_workers=int(args.num_workers),
        num_scenarios_discovered=len(scenario_paths),
        num_scenarios_scanned=len(selected_paths),
        signalized_drop_reasons=signalized_drop_counts,
        path_filter_drop_reasons=path_filter_drop_counts,
        prefilter_stats_json=str(prefilter_stats_path),
        path_index_jsonl=str(path_index_raw_path),
    )
    raw_summary.update(
        {
            "max_agents_per_candidate": int(args.max_agents_per_candidate),
            "max_candidates_per_scenario": int(args.max_candidates_per_scenario),
            "target_agent_policy": str(args.target_agent_policy),
            "light_canonicalization_summary_json": str(outdir / "light_canonicalization_summary.json"),
            "light_group_histogram_json": str(outdir / "light_group_histogram.json"),
            "path_index_raw_jsonl": str(path_index_raw_path),
        }
    )
    curated_summary = summarize_path_rows(
        curated_rows,
        artifact_mode=str(args.artifact_mode),
        scenario_root=str(args.scenario_root),
        outdir=str(outdir),
        max_scenarios=int(args.max_scenarios),
        seed=int(args.seed),
        num_workers=int(args.num_workers),
        num_scenarios_discovered=len(scenario_paths),
        num_scenarios_scanned=len(selected_paths),
        signalized_drop_reasons=signalized_drop_counts,
        path_filter_drop_reasons=path_filter_drop_counts,
        prefilter_stats_json=str(prefilter_stats_path),
        path_index_jsonl=str(path_index_curated_path),
    )
    curated_summary.update(
        {
            "max_agents_per_candidate": int(args.max_agents_per_candidate),
            "max_candidates_per_scenario": int(args.max_candidates_per_scenario),
            "target_agent_policy": str(args.target_agent_policy),
            "path_index_raw_jsonl": str(path_index_raw_path),
            "path_index_curated_jsonl": str(path_index_curated_path),
            "default_training_facing_index_jsonl": str(path_index_curated_path),
            "light_canonicalization_summary_json": str(outdir / "light_canonicalization_summary.json"),
            "light_group_histogram_json": str(outdir / "light_group_histogram.json"),
            "curated_filter_summary_json": str(outdir / "curated_filter_summary.json"),
            "dedup_mode": str(args.dedup_mode),
            "decision_time_merge_frames": int(args.decision_time_merge_frames),
            "window_overlap_threshold": float(args.window_overlap_threshold),
            "anchor_dist_threshold": float(args.anchor_dist_threshold),
            "anchor_heading_threshold_rad": float(args.anchor_heading_threshold_rad),
            "path_only_safe_mode": bool(args.path_only_safe_mode),
            **dedup_stats,
            "num_rows_before_curated_filter": int(len(curated_dedup_rows)),
            "num_rows_after_curated_filter": int(len(curated_rows)),
        }
    )

    raw_histograms = build_path_histograms(raw_entries)
    curated_histograms = build_path_histograms(curated_rows)
    manifest = build_path_manifest(curated_rows)

    summary_raw_path = outdir / "path_support_summary_raw.json"
    summary_curated_path = outdir / "path_support_summary_curated.json"
    summary_path = outdir / "path_support_summary.json"
    hist_raw_path = outdir / "path_label_histograms_raw.json"
    hist_curated_path = outdir / "path_label_histograms_curated.json"
    hist_path = outdir / "path_label_histograms.json"
    manifest_path = outdir / "path_examples_manifest.json"
    write_json(summary_raw_path, raw_summary)
    write_json(summary_curated_path, curated_summary)
    write_json(summary_path, curated_summary)
    write_json(hist_raw_path, raw_histograms)
    write_json(hist_curated_path, curated_histograms)
    write_json(hist_path, curated_histograms)
    write_json(manifest_path, manifest)
    write_json(outdir / "light_canonicalization_summary.json", light_summary)
    write_json(outdir / "light_group_histogram.json", light_group_hist)
    write_json(outdir / "curated_filter_summary.json", curated_filter_summary)

    debug_manifest_path = None
    curated_debug_candidates = [
        _build_debug_ref(
            kind="kept",
            scenario_id=str(row["scenario_id"]),
            scenario_pkl=str(row["scenario_pkl"]),
            scenario_file_name=str(row["scenario_file_name"]),
            light_id=str(row["light_id"]),
            agent_id=str(row["agent_id"]),
            decision_time_idx=int(row["decision_time_idx"]),
            branch_label=str(row["branch_label"]),
            agent_role=str(row.get("agent_role") or ""),
            example_id=str(row["example_id"]),
        )
        for row in curated_rows
    ]
    debug_candidates = [item for item in debug_candidates if str(item.get("kind")) != "kept"] + curated_debug_candidates
    if str(args.artifact_mode) == "sampled":
        debug_manifest_path = _materialize_sampled_debug_bundles(
            debug_candidates=debug_candidates,
            outdir=outdir,
            config_path=str(args.config or ""),
            seed=int(args.seed),
            sample_total=int(args.debug_sample_total),
            sample_per_class=int(args.debug_sample_per_class),
            sample_per_drop_reason=int(args.debug_sample_per_drop_reason),
        )
    elif str(args.artifact_mode) == "full":
        debug_manifest_path = _materialize_sampled_debug_bundles(
            debug_candidates=debug_candidates,
            outdir=outdir,
            config_path=str(args.config or ""),
            seed=int(args.seed),
            sample_total=0,
            sample_per_class=0,
            sample_per_drop_reason=0,
            materialize_all_kept=True,
        )

    wall_time_sec = float(time.perf_counter() - start_time)
    io_summary = _build_io_benchmark_summary(
        outdir=outdir,
        artifact_mode=str(args.artifact_mode),
        wall_time_sec=wall_time_sec,
        num_scenarios_scanned=len(selected_paths),
        num_full_train_views_built=int(num_full_train_views_built),
        num_path_examples_written=len(curated_rows),
    )
    io_summary_path = outdir / "io_benchmark_summary.json"
    write_json(io_summary_path, io_summary)
    io_summary = _build_io_benchmark_summary(
        outdir=outdir,
        artifact_mode=str(args.artifact_mode),
        wall_time_sec=wall_time_sec,
        num_scenarios_scanned=len(selected_paths),
        num_full_train_views_built=int(num_full_train_views_built),
        num_path_examples_written=len(curated_rows),
    )
    write_json(io_summary_path, io_summary)
    if debug_manifest_path is not None:
        curated_summary["debug_bundle_manifest_jsonl"] = str(debug_manifest_path)
    curated_summary["io_benchmark_summary_json"] = str(io_summary_path)
    raw_summary["io_benchmark_summary_json"] = str(io_summary_path)
    write_json(summary_raw_path, raw_summary)
    write_json(summary_curated_path, curated_summary)
    write_json(summary_path, curated_summary)

    print(json.dumps(curated_summary, indent=2, sort_keys=True))
    return 0


def _select_scenarios(paths: List[Path], *, max_scenarios: int, seed: int) -> List[Path]:
    if max_scenarios <= 0 or len(paths) <= max_scenarios:
        return list(paths)
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(len(paths), size=int(max_scenarios), replace=False))
    return [paths[int(idx)] for idx in indices.tolist()]


def _run_scenario_jobs(
    *,
    worker_args: Sequence[Dict[str, Any]],
    num_workers: int,
    show_progress: bool,
) -> List[Dict[str, Any]]:
    tracker = _ScenarioProgressTracker(total=len(worker_args), enabled=show_progress)
    results: List[Dict[str, Any]] = []
    if num_workers > 1 and len(worker_args) > 1:
        with ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
            future_to_arg = {executor.submit(_process_one_scenario, item): item for item in worker_args}
            for future in as_completed(future_to_arg):
                result = future.result()
                results.append(result)
                tracker.update(result)
    else:
        for item in worker_args:
            result = _process_one_scenario(item)
            results.append(result)
            tracker.update(result)
    tracker.finish()
    return results


class _ScenarioProgressTracker:
    def __init__(self, *, total: int, enabled: bool, update_interval_sec: float = 2.0):
        self.total = int(total)
        self.enabled = bool(enabled) and self.total > 0
        self.update_interval_sec = float(update_interval_sec)
        self.start_time = time.perf_counter()
        self.last_print_time = self.start_time
        self.completed = 0
        self.kept_rows = 0
        self.full_train_views = 0
        self.prefilter_pass = 0
        self.tty = bool(sys.stderr.isatty())
        if self.enabled:
            self._emit(force=True)

    def update(self, result: Mapping[str, Any]) -> None:
        self.completed += 1
        self.kept_rows += int(len(result.get("entries", [])))
        self.full_train_views += int(result.get("num_full_train_views_built", 0))
        self.prefilter_pass += int(result.get("num_prefilter_pass", 0))
        if not self.enabled:
            return
        now = time.perf_counter()
        if (now - self.last_print_time) >= self.update_interval_sec or self.completed >= self.total:
            self._emit(force=self.completed >= self.total)

    def finish(self) -> None:
        if not self.enabled:
            return
        if self.completed < self.total:
            self._emit(force=True)
        elif self.tty:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def _emit(self, *, force: bool) -> None:
        now = time.perf_counter()
        elapsed = max(now - self.start_time, 1e-6)
        rate = float(self.completed) / elapsed
        remaining = max(self.total - self.completed, 0)
        eta_sec = (remaining / rate) if rate > 0 else float("inf")
        fraction = (float(self.completed) / float(self.total)) if self.total > 0 else 1.0
        bar = self._progress_bar(fraction)
        message = (
            f"[path-index] {bar} {self.completed}/{self.total} "
            f"({fraction * 100:5.1f}%) | elapsed {self._format_duration(elapsed)} "
            f"| eta {self._format_duration(eta_sec)} | rate {rate:0.2f} scen/s "
            f"| rows {self.kept_rows} | built {self.full_train_views}"
        )
        if self.tty:
            end = "\n" if force and self.completed >= self.total else ""
            sys.stderr.write("\r" + message + " " * 4 + end)
        else:
            sys.stderr.write(message + "\n")
        sys.stderr.flush()
        self.last_print_time = now

    @staticmethod
    def _progress_bar(fraction: float, width: int = 24) -> str:
        clamped = min(max(float(fraction), 0.0), 1.0)
        filled = int(round(clamped * width))
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if not np.isfinite(seconds):
            return "??:??:??"
        total_seconds = max(int(round(seconds)), 0)
        hours, rem = divmod(total_seconds, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _process_one_scenario(args: Dict[str, Any]) -> Dict[str, Any]:
    scenario_pkl = str(args["scenario_pkl"])
    scenario_file_name = Path(scenario_pkl).name
    config = load_motion_config(config_path=args.get("config_path") or None)

    signalized_result = select_signalized_candidates_for_scenario(scenario_pkl)
    if not signalized_result.candidates:
        return {
            "scenario_pkl": scenario_pkl,
            "signalized_primary_drop_reason": signalized_result.primary_drop_reason,
            "prefilter_drop_counts": {},
            "path_filter_drop_counts": {},
            "debug_candidates": [],
            "num_candidate_agents_considered": 0,
            "num_prefilter_pass": 0,
            "num_full_train_views_built": 0,
            "entries": [],
        }

    raw_scenario = load_raw_scenario(scenario_pkl)
    canonical = load_and_normalize_scenario(scenario_pkl)
    forward_summary = summarize_forward_supervision_for_raw_scenario(raw_scenario, config=config)

    path_filter_drop_counts: Counter[str] = Counter()
    prefilter_drop_counts: Counter[str] = Counter()
    debug_candidates: List[Dict[str, Any]] = []
    entries: List[Dict[str, Any]] = []
    num_candidate_agents_considered = 0
    num_prefilter_pass = 0
    num_full_train_views_built = 0

    candidates = list(signalized_result.candidates)
    max_candidates = int(args.get("max_candidates_per_scenario", 0))
    if max_candidates > 0:
        candidates = candidates[:max_candidates]

    for candidate in candidates:
        if str(candidate.light_id) not in canonical.traffic_lights:
            continue
        light = canonical.traffic_lights[str(candidate.light_id)]
        if str(args.get("target_agent_policy", "all_trainable")) == "sdc_only":
            considered_agents = [str(canonical.sdc_id)]
        else:
            considered_agents = list(getattr(forward_summary, "trainable_track_ids", []))
        max_agents = int(args.get("max_agents_per_candidate", 0))
        if max_agents > 0:
            considered_agents = considered_agents[:max_agents]

        for agent_id in considered_agents:
            if agent_id not in canonical.tracks:
                continue
            track = canonical.tracks[agent_id]
            if str(track.object_type) != "VEHICLE":
                continue
            if not _track_comes_within_stop_point(track, candidate.stop_point_xy, threshold_m=35.0):
                continue
            num_candidate_agents_considered += 1

            prefilter = _prefilter_path_candidate(
                canonical=canonical,
                candidate=candidate,
                agent_id=agent_id,
                forward_summary=forward_summary,
            )
            if not prefilter["keep"]:
                reason = str(prefilter["drop_reason"])
                prefilter_drop_counts[reason] += 1
                debug_candidates.append(
                    _build_debug_ref(
                        kind="prefilter_drop",
                        scenario_id=candidate.scenario_id,
                        scenario_pkl=scenario_pkl,
                        scenario_file_name=scenario_file_name,
                        light_id=candidate.light_id,
                        agent_id=agent_id,
                        decision_time_idx=prefilter.get("decision_time_idx"),
                        branch_label=None,
                        agent_role=prefilter.get("agent_role"),
                        drop_reason=reason,
                    )
                )
                continue

            num_prefilter_pass += 1
            num_full_train_views_built += 1
            built = _build_path_candidate_row(
                scenario_pkl=scenario_pkl,
                scenario_file_name=scenario_file_name,
                candidate=candidate,
                canonical=canonical,
                light=light,
                raw_scenario=raw_scenario,
                agent_id=agent_id,
                forward_summary=forward_summary,
                prefilter=prefilter,
            )
            if built is None:
                path_filter_drop_counts["build_failed"] += 1
                debug_candidates.append(
                    _build_debug_ref(
                        kind="path_filter_drop",
                        scenario_id=candidate.scenario_id,
                        scenario_pkl=scenario_pkl,
                        scenario_file_name=scenario_file_name,
                        light_id=candidate.light_id,
                        agent_id=agent_id,
                        decision_time_idx=prefilter.get("decision_time_idx"),
                        branch_label=None,
                        agent_role=prefilter.get("agent_role"),
                        drop_reason="build_failed",
                    )
                )
                continue

            train_view = built["train_view"]
            keep, drop_reason = _is_path_train_view_eligible(train_view)
            if not keep:
                path_filter_drop_counts[str(drop_reason)] += 1
                debug_candidates.append(
                    _build_debug_ref(
                        kind="path_filter_drop",
                        scenario_id=candidate.scenario_id,
                        scenario_pkl=scenario_pkl,
                        scenario_file_name=scenario_file_name,
                        light_id=candidate.light_id,
                        agent_id=agent_id,
                        decision_time_idx=int(train_view["decision_time_idx"]),
                        branch_label=str(train_view.get("supervised_decision", {}).get("branch_label") or ""),
                        agent_role=str(train_view.get("provenance", {}).get("agent_role") or prefilter.get("agent_role") or ""),
                        drop_reason=str(drop_reason),
                    )
                )
                continue

            entries.append(built["row"])
            debug_candidates.append(
                _build_debug_ref(
                    kind="kept",
                    scenario_id=candidate.scenario_id,
                    scenario_pkl=scenario_pkl,
                    scenario_file_name=scenario_file_name,
                    light_id=candidate.light_id,
                    agent_id=agent_id,
                    decision_time_idx=int(train_view["decision_time_idx"]),
                    branch_label=str(built["row"]["branch_label"]),
                    agent_role=str(train_view["provenance"]["agent_role"]),
                    example_id=str(built["row"]["example_id"]),
                )
            )

    return {
        "scenario_pkl": scenario_pkl,
        "signalized_primary_drop_reason": signalized_result.primary_drop_reason,
        "prefilter_drop_counts": dict(prefilter_drop_counts),
        "path_filter_drop_counts": dict(path_filter_drop_counts),
        "debug_candidates": debug_candidates,
        "num_candidate_agents_considered": int(num_candidate_agents_considered),
        "num_prefilter_pass": int(num_prefilter_pass),
        "num_full_train_views_built": int(num_full_train_views_built),
        "entries": entries,
    }


def _prefilter_path_candidate(
    *,
    canonical: Any,
    candidate: Any,
    agent_id: str,
    forward_summary: Any,
) -> Dict[str, Any]:
    from bmt.counterfactual import choose_decision_window

    agent_role = "sdc" if str(agent_id) == str(canonical.sdc_id) else "forward_loss_vehicle"
    target_alignment = _infer_target_agent_alignment(forward_summary=forward_summary, agent_id=agent_id)
    if not target_alignment.target_is_trainable:
        return _prefilter_drop("target_not_trainable", agent_role=agent_role)
    try:
        decision_window = choose_decision_window(
            canonical,
            agent_id=agent_id,
            agent_role=agent_role,
            stop_point_xy=candidate.stop_point_xy,
        )
    except Exception:
        return _prefilter_drop("no_decision_window", agent_role=agent_role)

    current_idx = int(canonical.current_time_index)
    if int(decision_window.window_end_idx) <= current_idx:
        return _prefilter_drop("no_future_horizon", agent_role=agent_role, decision_time_idx=int(decision_window.decision_time_idx))
    if int(decision_window.decision_time_idx) < current_idx:
        return _prefilter_drop("decision_before_current", agent_role=agent_role, decision_time_idx=int(decision_window.decision_time_idx))
    if decision_window.cross_time_idx is not None and int(decision_window.cross_time_idx) < current_idx:
        return _prefilter_drop("cross_before_current", agent_role=agent_role, decision_time_idx=int(decision_window.decision_time_idx))

    target_track = canonical.tracks[agent_id]
    decision_idx = int(decision_window.decision_time_idx)
    if decision_idx >= target_track.valid.shape[0]:
        return _prefilter_drop("decision_out_of_range", agent_role=agent_role, decision_time_idx=decision_idx)
    if not bool(target_track.valid[decision_idx]) or not np.isfinite(target_track.position_xy[decision_idx]).all():
        return _prefilter_drop("invalid_decision_state", agent_role=agent_role, decision_time_idx=decision_idx)

    local_patch = extract_local_patch(
        canonical,
        stop_point_xy=candidate.stop_point_xy,
        radius_m=30.0,
        time_index=decision_idx,
    )
    _route_result, branch_candidates = _enumerate_branches(
        canonical=canonical,
        agent_id=str(agent_id),
        decision_window=decision_window,
        stop_point_xy=candidate.stop_point_xy,
        local_patch=local_patch,
    )
    feasible_branches = [
        branch
        for branch in branch_candidates
        if str(getattr(branch, "branch_label", "")) in PATH_LABELS and getattr(branch, "terminal_pose", None) is not None
    ]
    if not feasible_branches:
        return _prefilter_drop("no_feasible_branch_family", agent_role=agent_role, decision_time_idx=decision_idx)
    if not _terminal_anchor_computable(target_track=target_track, decision_idx=decision_idx, feasible_branch=feasible_branches[0]):
        return _prefilter_drop("terminal_anchor_unavailable", agent_role=agent_role, decision_time_idx=decision_idx)

    return {
        "keep": True,
        "drop_reason": None,
        "agent_role": agent_role,
        "target_alignment": target_alignment,
        "decision_window": decision_window,
        "local_patch": local_patch,
        "branch_candidates": branch_candidates,
    }


def _prefilter_drop(reason: str, *, agent_role: str, decision_time_idx: Optional[int] = None) -> Dict[str, Any]:
    return {
        "keep": False,
        "drop_reason": str(reason),
        "agent_role": str(agent_role),
        "decision_time_idx": None if decision_time_idx is None else int(decision_time_idx),
    }


def _terminal_anchor_computable(*, target_track: Any, decision_idx: int, feasible_branch: Any) -> bool:
    if decision_idx >= target_track.position_xy.shape[0]:
        return False
    if not np.isfinite(target_track.position_xy[decision_idx]).all():
        return False
    terminal_pose = getattr(feasible_branch, "terminal_pose", None)
    if terminal_pose is None:
        return False
    values = [
        getattr(terminal_pose, "x", np.nan),
        getattr(terminal_pose, "y", np.nan),
        getattr(terminal_pose, "heading", np.nan),
    ]
    return bool(np.isfinite(np.asarray(values, dtype=np.float32)).all())


def _build_path_candidate_row(
    *,
    scenario_pkl: str,
    scenario_file_name: str,
    candidate: Any,
    canonical: Any,
    light: Any,
    raw_scenario: Mapping[str, Any],
    agent_id: str,
    forward_summary: Any,
    prefilter: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    decision_window = prefilter["decision_window"]
    branch_candidates = _serialize_branches(
        recover_ground_truth_branch(
            canonical,
            decision_window=decision_window,
            branch_candidates=list(prefilter["branch_candidates"]),
            agent_id=agent_id,
        )
    )
    gt_recovery = branch_candidates["gt_recovery"]
    provenance = gt_recovery.get("provenance")
    commitment_metrics = gt_recovery.get("commitment_metrics")
    if not isinstance(provenance, dict) or not isinstance(commitment_metrics, dict):
        return None
    provenance_obj = ArtifactProvenance(**provenance)
    commitment_obj = CommitmentMetrics(**commitment_metrics)

    conflict_result = analyze_conflicts(
        canonical,
        agent_id=agent_id,
        stop_point_xy=candidate.stop_point_xy,
        decision_time_idx=int(decision_window.decision_time_idx),
    )
    signal_state_at_crossing = None
    if decision_window.cross_time_idx is not None and int(decision_window.cross_time_idx) < len(light.object_state):
        signal_state_at_crossing = light.object_state[int(decision_window.cross_time_idx)]
    compliance_label = _derive_compliance_label(signal_state_at_crossing or candidate.signal_state_at_time, decision_window.crossed_stop_region)
    entry_timing = _derive_entry_timing(conflict_result)
    target_alignment = prefilter["target_alignment"]
    supervision_gates = _build_supervision_gates(
        canonical=canonical,
        candidate=candidate,
        conflict_result=conflict_result,
        provenance=provenance_obj,
        commitment_metrics=commitment_obj,
        agent_id=agent_id,
        target_agent_alignment=target_alignment,
    )

    context = InterventionContext(
        sdc_id=canonical.sdc_id,
        traffic_light_id=candidate.light_id,
        stop_point_xy=candidate.stop_point_xy,
        approach_heading=float(decision_window.approach_heading),
        signal_state_at_decision=candidate.signal_state_at_time,
        objects_of_interest=list(canonical.objects_of_interest),
        conflict_agents=[
            ConflictAgentRef(track_id=record.track_id, eta_s=record.eta_s, eta_gap_s=record.eta_gap_s)
            for record in conflict_result.conflict_agents
        ],
    )
    recovered_decision = RecoveredDecision(
        branch_id=str(gt_recovery["branch_id"]),
        branch_label=str(gt_recovery["branch_label"]),
        terminal_pose=TerminalPose(
            x=float(gt_recovery["terminal_pose"]["x"]),
            y=float(gt_recovery["terminal_pose"]["y"]),
            heading=float(gt_recovery["terminal_pose"]["heading"]),
        ),
        crossed_stop_region=bool(gt_recovery["crossed_stop_region"]),
        compliance_label=compliance_label,
        entry_timing=entry_timing,
        signal_state_at_crossing=signal_state_at_crossing,
    )
    compact_debug = {
        "source_view_type": "train_view",
        "light_id": str(candidate.light_id),
        "current_time_index": int(canonical.current_time_index),
        "num_branch_candidates": int(len(branch_candidates["branch_candidates"])),
        "num_conflict_agents": int(len(conflict_result.conflict_agents)),
        "branch_recall_hit": bool(gt_recovery.get("branch_recall_hit", False)),
        "recovered_from_existing_candidate": bool(gt_recovery.get("recovered_from_existing_candidate", False)),
    }
    train_view = build_local_intervention_train_view(
        scenario_id=candidate.scenario_id,
        agent_id=agent_id,
        decision_time_idx=int(decision_window.decision_time_idx),
        window=WindowSpec(start_idx=int(decision_window.window_start_idx), end_idx=int(decision_window.window_end_idx)),
        context=context,
        signal_qc=dict(candidate.signal_qc),
        provenance=provenance_obj,
        commitment=commitment_obj,
        supervision=supervision_gates,
        target_alignment=target_alignment,
        raw_recovered_decision=recovered_decision,
        alternatives=[],
        debug=compact_debug,
    )
    control_code = compile_control_code_from_local_intervention(
        train_view.to_dict(),
        canonical=canonical,
        source_path="",
    ).to_dict()
    row = _build_inline_path_index_row(
        scenario_pkl=scenario_pkl,
        scenario_file_name=scenario_file_name,
        candidate=candidate,
        raw_scenario=raw_scenario,
        train_view=train_view.to_dict(),
        control_code=control_code,
    )
    return {
        "train_view": train_view.to_dict(),
        "row": row,
    }
def _build_inline_path_index_row(
    *,
    scenario_pkl: str,
    scenario_file_name: str,
    candidate: Any,
    raw_scenario: Mapping[str, Any],
    train_view: Mapping[str, Any],
    control_code: Mapping[str, Any],
) -> Dict[str, Any]:
    scenario_id = str(train_view["scenario_id"])
    agent_id = str(train_view["agent_id"])
    decision_time_idx = int(train_view["decision_time_idx"])
    example_id = _example_id(scenario_id=scenario_id, light_id=str(candidate.light_id), agent_id=agent_id, decision_time_idx=decision_time_idx)
    debug = dict(control_code.get("debug", {}))
    debug["light_id"] = str(candidate.light_id)
    debug["example_id"] = example_id
    debug["scenario_file_name"] = scenario_file_name
    debug["scenario_pkl"] = str(scenario_pkl)
    debug["source_provenance"] = train_view.get("provenance", {})
    debug["source_supervision_gates"] = train_view.get("supervision", {})
    debug["source_target_agent_alignment"] = train_view.get("target_alignment", {})
    debug["source_commitment"] = train_view.get("commitment", {})
    debug["source_view_type"] = "train_view"
    debug["control_available_at_current"] = bool(train_view.get("control_available_at_current"))
    debug["conditioning_eligible"] = bool(train_view.get("conditioning_eligible"))
    debug["target_is_trainable"] = bool(train_view.get("target_is_trainable"))
    debug["control_kind"] = "factual"

    row = dict(control_code)
    row["debug"] = debug
    row["scenario_pkl"] = str(scenario_pkl)
    row["scenario_file_name"] = scenario_file_name
    row["example_id"] = example_id
    row["light_id"] = str(candidate.light_id)
    row["window_start_idx"] = int(train_view["window"]["start_idx"])
    row["window_end_idx"] = int(train_view["window"]["end_idx"])
    row["branch_label"] = str(train_view["supervised_decision"]["branch_label"])
    row["branch_id"] = str(train_view["supervised_decision"]["branch_id"])
    row["compliance_label"] = str(train_view.get("supervised_decision", {}).get("compliance_label") or "none")
    row["stop_point_xy"] = list(train_view.get("context", {}).get("stop_point_xy", []))
    row["target_is_trainable"] = bool(train_view.get("target_is_trainable"))
    row["control_available_at_current"] = bool(train_view.get("control_available_at_current"))
    row["conditioning_eligible"] = bool(train_view.get("conditioning_eligible"))
    row["path_choice_supervisable"] = bool(train_view.get("supervision", {}).get("path_choice_supervisable"))
    row["compliance_supervisable"] = bool(train_view.get("supervision", {}).get("compliance_supervisable"))
    row["timing_supervisable"] = bool(train_view.get("supervision", {}).get("timing_supervisable"))
    row["signal_state_at_decision"] = train_view.get("context", {}).get("signal_state_at_decision")
    row["agent_role"] = str(train_view.get("provenance", {}).get("agent_role"))
    row["decision_state"] = train_view.get("supervision", {}).get("decision_state")
    row["branch_margin"] = train_view.get("commitment", {}).get("branch_margin")
    row["downstream_progress_along_branch_m"] = train_view.get("commitment", {}).get("downstream_progress_along_branch_m")
    row["final_heading_error_rad"] = train_view.get("commitment", {}).get("final_heading_error_rad")
    row["mean_lateral_error_to_best_branch_m"] = train_view.get("commitment", {}).get("mean_lateral_error_to_best_branch_m")
    row["signed_stopline_progress_m"] = train_view.get("commitment", {}).get("signed_stopline_progress_m")
    row["current_time_index_global"] = train_view.get("provenance", {}).get("current_time_index_global")
    row["decision_time_index_global"] = train_view.get("provenance", {}).get("decision_time_index_global")
    row["cross_time_index_global"] = train_view.get("provenance", {}).get("cross_time_index_global")
    row["branch_commit_index_global"] = train_view.get("provenance", {}).get("branch_commit_index_global")
    row["is_sdc_target"] = bool(str(agent_id) == str(train_view.get("context", {}).get("sdc_id")))
    row["compact_debug_ids"] = {
        "light_id": str(candidate.light_id),
        "example_id": example_id,
        "raw_scenario_id": str(raw_scenario.get("id", scenario_id)),
    }
    return row


def _example_id(*, scenario_id: str, light_id: str, agent_id: str, decision_time_idx: int) -> str:
    return f"{scenario_id}__agent_{agent_id}__light_{light_id}__t_{int(decision_time_idx):03d}"


def _compact_manifest_entry(entry: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "example_id": entry.get("example_id"),
        "scenario_id": entry.get("scenario_id"),
        "scenario_file_name": entry.get("scenario_file_name"),
        "agent_id": entry.get("agent_id"),
        "light_id": entry.get("light_id"),
        "decision_time_idx": entry.get("decision_time_idx"),
        "window_start_idx": entry.get("window_start_idx"),
        "window_end_idx": entry.get("window_end_idx"),
        "branch_label": entry.get("branch_label"),
        "agent_role": entry.get("agent_role"),
        "signal_state_at_decision": entry.get("signal_state_at_decision"),
    }


def _build_debug_ref(
    *,
    kind: str,
    scenario_id: str,
    scenario_pkl: str,
    scenario_file_name: str,
    light_id: str,
    agent_id: str,
    decision_time_idx: Optional[int],
    branch_label: Optional[str],
    agent_role: Optional[str],
    drop_reason: Optional[str] = None,
    example_id: Optional[str] = None,
) -> Dict[str, Any]:
    if example_id is None and decision_time_idx is not None:
        example_id = _example_id(
            scenario_id=str(scenario_id),
            light_id=str(light_id),
            agent_id=str(agent_id),
            decision_time_idx=int(decision_time_idx),
        )
    return {
        "kind": str(kind),
        "scenario_id": str(scenario_id),
        "scenario_pkl": str(scenario_pkl),
        "scenario_file_name": str(scenario_file_name),
        "light_id": str(light_id),
        "agent_id": str(agent_id),
        "decision_time_idx": None if decision_time_idx is None else int(decision_time_idx),
        "branch_label": None if branch_label in (None, "") else str(branch_label),
        "agent_role": None if agent_role in (None, "") else str(agent_role),
        "drop_reason": None if drop_reason in (None, "") else str(drop_reason),
        "example_id": example_id,
    }


def _materialize_sampled_debug_bundles(
    *,
    debug_candidates: Sequence[Mapping[str, Any]],
    outdir: Path,
    config_path: str,
    seed: int,
    sample_total: int,
    sample_per_class: int,
    sample_per_drop_reason: int,
    materialize_all_kept: bool = False,
) -> Path:
    config = load_motion_config(config_path=config_path or None)
    rng = random.Random(int(seed))
    debug_root = outdir / "debug_samples"
    debug_root.mkdir(parents=True, exist_ok=True)

    selected: List[Dict[str, Any]] = []
    used_ids = set()

    def _maybe_take(items: Sequence[Mapping[str, Any]], limit: int) -> None:
        pool = [dict(item) for item in items if str(item.get("example_id") or id(item)) not in used_ids]
        rng.shuffle(pool)
        for item in pool[: max(0, int(limit))]:
            key = str(item.get("example_id") or id(item))
            if key in used_ids:
                continue
            used_ids.add(key)
            selected.append(item)

    kept = [item for item in debug_candidates if str(item.get("kind")) == "kept"]
    drops = [item for item in debug_candidates if str(item.get("kind")) != "kept"]

    if materialize_all_kept:
        _maybe_take(kept, len(kept))
    else:
        if sample_per_class > 0:
            for label in PATH_LABELS:
                _maybe_take([item for item in kept if str(item.get("branch_label")) == label], sample_per_class)
        if sample_per_drop_reason > 0:
            drop_reasons = sorted({str(item.get("drop_reason")) for item in drops if item.get("drop_reason")})
            for reason in drop_reasons:
                _maybe_take([item for item in drops if str(item.get("drop_reason")) == reason], sample_per_drop_reason)
        if sample_total > 0 and len(selected) < sample_total:
            _maybe_take(debug_candidates, sample_total - len(selected))

    manifest_rows: List[Dict[str, Any]] = []
    for item in selected:
        if str(item.get("kind")) == "kept":
            result = materialize_candidate_debug_bundle(
                scenario_pkl=str(item["scenario_pkl"]),
                light_id=str(item["light_id"]),
                agent_id=str(item["agent_id"]),
                outdir=debug_root,
                config=config,
                include_pngs=True,
            )
            manifest_rows.append(
                {
                    **dict(item),
                    "materialized": result is not None,
                    "artifact_dir": None if result is None else result.get("artifact_dir"),
                    "train_view_path": None if result is None else result.get("train_view_path"),
                    "factual_control_code_path": None if result is None else result.get("factual_control_code_path"),
                }
            )
        else:
            drop_dir = debug_root / "drops" / str(item.get("drop_reason", "unknown")) / str(item.get("example_id") or f"{item.get('scenario_id')}__{item.get('agent_id')}")
            drop_dir.mkdir(parents=True, exist_ok=True)
            drop_path = drop_dir / "prefilter_drop.json"
            _write_json(drop_path, dict(item))
            manifest_rows.append({**dict(item), "materialized": True, "drop_debug_path": str(drop_path)})

    manifest_path = debug_root / "debug_bundle_manifest.jsonl"
    _write_jsonl(manifest_path, manifest_rows)
    return manifest_path


def _build_io_benchmark_summary(
    *,
    outdir: Path,
    artifact_mode: str,
    wall_time_sec: float,
    num_scenarios_scanned: int,
    num_full_train_views_built: int,
    num_path_examples_written: int,
) -> Dict[str, Any]:
    num_files = 0
    num_dirs = 1
    total_bytes = 0
    for path in outdir.rglob("*"):
        if path.is_file():
            num_files += 1
            total_bytes += int(path.stat().st_size)
        elif path.is_dir():
            num_dirs += 1
    return {
        "artifact_mode": str(artifact_mode),
        "wall_time_sec": float(wall_time_sec),
        "num_files_written": int(num_files),
        "num_dirs_written": int(num_dirs),
        "total_bytes_written": int(total_bytes),
        "num_scenarios_scanned": int(num_scenarios_scanned),
        "num_full_train_views_built": int(num_full_train_views_built),
        "num_path_examples_written": int(num_path_examples_written),
        "mean_files_per_scanned_scenario": float(num_files / max(int(num_scenarios_scanned), 1)),
        "mean_files_per_path_example": float(num_files / max(int(num_path_examples_written), 1)),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _histogram(values: Iterable[Any]) -> Dict[str, int]:
    histogram: Dict[str, int] = {}
    for value in values:
        key = "null" if value is None else str(value)
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items(), key=lambda item: item[0]))


def _is_path_train_view_eligible(train_view: Dict[str, Any]) -> Tuple[bool, str]:
    supervised_decision = dict(train_view.get("supervised_decision", {}))
    branch_label = supervised_decision.get("branch_label")
    terminal_pose = supervised_decision.get("terminal_pose")
    if not bool(train_view.get("conditioning_eligible")):
        return False, "conditioning_ineligible"
    if not bool(train_view.get("target_is_trainable")):
        return False, "non_trainable_target"
    if not bool(train_view.get("control_available_at_current")):
        return False, "control_unavailable"
    if not bool(train_view.get("supervision", {}).get("path_choice_supervisable")):
        return False, "path_not_supervisable"
    if branch_label is None:
        return False, "branch_label_null"
    if str(branch_label) == "u_turn":
        return False, "u_turn_excluded"
    if str(branch_label) not in PATH_LABELS:
        return False, "unsupported_branch_label"
    if terminal_pose is None:
        return False, "missing_terminal_pose"
    return True, "kept"


if __name__ == "__main__":
    raise SystemExit(main())
