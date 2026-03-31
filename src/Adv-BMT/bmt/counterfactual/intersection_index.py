from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .contract_local_intervention import FilterReport
from .geometry import any_point_within_radius, max_contiguous_true_run, point_distance_curve
from .normalize import load_and_normalize_scenario, load_raw_scenario
from .signal_qc import evaluate_signal_qc
from .types import CanonicalScenario, stable_string_sort_key
from .visualize import plot_stop_point_distance_curve

REQUIRED_DROP_REASONS = (
    "no_traffic_light",
    "no_sdc",
    "no_valid_sdc_window",
    "no_stop_point",
    "stop_point_too_far",
    "no_lane_like_features",
    "ambiguous_light_state",
)

LANE_LIKE_PREFIX = "LANE_"


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    return value


@dataclass
class SignalizedCandidateWindow:
    scenario_id: str
    scenario_pkl: str
    sdc_id: str
    light_id: str
    stop_point_xy: Tuple[float, float]
    min_dist_stop_point_m: float
    t_min_dist: int
    first_time_under_35m: int
    sdc_speed_mps: float
    signal_state_at_time: Optional[str]
    objects_of_interest_overlap: List[str] = field(default_factory=list)
    lane_like_feature_count: int = 0
    signal_qc: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class SignalizedScenarioIndexResult:
    scenario_id: str
    scenario_pkl: str
    candidates: List[SignalizedCandidateWindow] = field(default_factory=list)
    primary_drop_reason: Optional[str] = None
    scenario_drop_reasons: Dict[str, int] = field(default_factory=dict)
    light_drop_reasons: Dict[str, int] = field(default_factory=dict)
    filter_reports: List[FilterReport] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class SignalizedIndexBuildResult:
    scenario_results: List[SignalizedScenarioIndexResult]
    candidates: List[SignalizedCandidateWindow]
    scenario_drop_reason_counts: Dict[str, int]
    light_drop_reason_counts: Dict[str, int]
    filter_reports: List[FilterReport]


def discover_scenario_pickles(data_dir: str | Path) -> List[Path]:
    root = Path(data_dir).expanduser()
    files = list(root.rglob("sd_*.pkl"))
    return sorted(set(files), key=lambda p: p.relative_to(root).as_posix())


def select_signalized_candidates_for_scenario(
    scenario_pkl: str | Path,
    *,
    distance_threshold_m: float = 35.0,
    local_patch_radius_m: float = 30.0,
    min_valid_sdc_steps: int = 10,
    ooi_overlap_radius_m: float = 15.0,
    ambiguity_threshold: float = 0.45,
) -> SignalizedScenarioIndexResult:
    raw = load_raw_scenario(scenario_pkl)
    canonical = load_and_normalize_scenario(scenario_pkl)
    scenario_id = canonical.scenario_id
    scenario_path = str(Path(scenario_pkl).expanduser())
    result = SignalizedScenarioIndexResult(scenario_id=scenario_id, scenario_pkl=scenario_path)

    if not canonical.traffic_lights:
        return _finalize_rejected_result(result, "no_traffic_light")
    result.filter_reports.append(FilterReport(stage="traffic_light_presence", input_count=1, kept_count=1, dropped_count=0, drop_reasons={}))

    sdc_track = canonical.tracks.get(canonical.sdc_id)
    if sdc_track is None:
        return _finalize_rejected_result(result, "no_sdc")
    result.filter_reports.append(FilterReport(stage="sdc_presence", input_count=1, kept_count=1, dropped_count=0, drop_reasons={}))

    max_valid_run = max_contiguous_true_run(np.asarray(sdc_track.valid, dtype=bool))
    if max_valid_run < int(min_valid_sdc_steps):
        return _finalize_rejected_result(result, "no_valid_sdc_window")
    result.filter_reports.append(FilterReport(stage="sdc_valid_window", input_count=1, kept_count=1, dropped_count=0, drop_reasons={}))

    lights_with_stop = [
        (light_id, light)
        for light_id, light in sorted(canonical.traffic_lights.items(), key=lambda item: stable_string_sort_key(item[0]))
        if light.stop_point_xy is not None
    ]
    if not lights_with_stop:
        return _finalize_rejected_result(result, "no_stop_point")
    result.filter_reports.append(
        FilterReport(
            stage="stop_point_presence",
            input_count=int(len(canonical.traffic_lights)),
            kept_count=int(len(lights_with_stop)),
            dropped_count=int(len(canonical.traffic_lights) - len(lights_with_stop)),
            drop_reasons={},
        )
    )

    raw_sdc_paths = _extract_optional_sdc_paths(raw)
    lane_like_by_light = {
        light_id: count_lane_like_features_near_stop_point(canonical, stop_point_xy=light.stop_point_xy, radius_m=local_patch_radius_m)
        for light_id, light in lights_with_stop
        if light.stop_point_xy is not None
    }

    kept_candidates: List[SignalizedCandidateWindow] = []
    light_drop_counts: Counter[str] = Counter()
    for light_id, light in lights_with_stop:
        stop_xy = light.stop_point_xy
        assert stop_xy is not None
        dist_curve = point_distance_curve(sdc_track.position_xy, stop_xy, valid_mask=sdc_track.valid)
        if not np.any(np.isfinite(dist_curve)):
            light_drop_counts["stop_point_too_far"] += 1
            continue
        t_min_dist = int(np.argmin(dist_curve))
        min_dist = float(dist_curve[t_min_dist])
        under_thresh = np.isfinite(dist_curve) & (dist_curve < float(distance_threshold_m))
        if not np.any(under_thresh):
            light_drop_counts["stop_point_too_far"] += 1
            continue
        first_under = int(np.flatnonzero(under_thresh)[0])

        lane_like_ids = lane_like_by_light.get(light_id, [])
        if len(lane_like_ids) < 2 and not raw_sdc_paths:
            light_drop_counts["no_lane_like_features"] += 1
            continue

        qc = evaluate_signal_qc(
            light.object_state,
            stop_point_present=light.stop_point_xy is not None,
            reference_time_index=first_under,
            ambiguity_threshold=ambiguity_threshold,
        )
        if qc.ambiguous_light_state:
            light_drop_counts["ambiguous_light_state"] += 1
            continue

        overlap = compute_objects_of_interest_overlap(
            canonical,
            stop_point_xy=stop_xy,
            time_index=first_under,
            radius_m=ooi_overlap_radius_m,
        )
        sdc_speed = float(np.linalg.norm(np.asarray(sdc_track.velocity_xy[first_under], dtype=np.float32)))
        kept_candidates.append(
            SignalizedCandidateWindow(
                scenario_id=scenario_id,
                scenario_pkl=scenario_path,
                sdc_id=canonical.sdc_id,
                light_id=light_id,
                stop_point_xy=(float(stop_xy[0]), float(stop_xy[1])),
                min_dist_stop_point_m=min_dist,
                t_min_dist=t_min_dist,
                first_time_under_35m=first_under,
                sdc_speed_mps=sdc_speed,
                signal_state_at_time=light.object_state[first_under] if first_under < len(light.object_state) else None,
                objects_of_interest_overlap=overlap,
                lane_like_feature_count=int(len(lane_like_ids)),
                signal_qc=qc.to_dict(),
                debug={
                    "lane_like_feature_ids": lane_like_ids,
                    "raw_sdc_paths_present": bool(raw_sdc_paths),
                },
            )
        )

    result.candidates = kept_candidates
    result.light_drop_reasons = {reason: int(light_drop_counts.get(reason, 0)) for reason in REQUIRED_DROP_REASONS if light_drop_counts.get(reason, 0) > 0}
    result.filter_reports.append(
        FilterReport(
            stage="candidate_light_filter",
            input_count=int(len(lights_with_stop)),
            kept_count=int(len(kept_candidates)),
            dropped_count=int(len(lights_with_stop) - len(kept_candidates)),
            drop_reasons=result.light_drop_reasons,
        )
    )

    if not kept_candidates:
        primary = _primary_drop_reason_from_counts(result.light_drop_reasons)
        result.primary_drop_reason = primary
        if primary:
            result.scenario_drop_reasons = {primary: 1}
    return result


def build_signalized_index(
    data_dir: str | Path,
    *,
    max_scenarios: Optional[int] = None,
    distance_threshold_m: float = 35.0,
    local_patch_radius_m: float = 30.0,
    min_valid_sdc_steps: int = 10,
    ooi_overlap_radius_m: float = 15.0,
    ambiguity_threshold: float = 0.45,
) -> SignalizedIndexBuildResult:
    files = discover_scenario_pickles(data_dir)
    if max_scenarios is not None:
        files = files[: int(max_scenarios)]

    scenario_results: List[SignalizedScenarioIndexResult] = []
    candidates: List[SignalizedCandidateWindow] = []
    scenario_drop_counts: Counter[str] = Counter()
    light_drop_counts: Counter[str] = Counter()
    filter_reports: List[FilterReport] = []

    for path in files:
        scenario_result = select_signalized_candidates_for_scenario(
            path,
            distance_threshold_m=distance_threshold_m,
            local_patch_radius_m=local_patch_radius_m,
            min_valid_sdc_steps=min_valid_sdc_steps,
            ooi_overlap_radius_m=ooi_overlap_radius_m,
            ambiguity_threshold=ambiguity_threshold,
        )
        scenario_results.append(scenario_result)
        candidates.extend(scenario_result.candidates)
        scenario_drop_counts.update(scenario_result.scenario_drop_reasons)
        light_drop_counts.update(scenario_result.light_drop_reasons)
        filter_reports.extend(scenario_result.filter_reports)

    return SignalizedIndexBuildResult(
        scenario_results=scenario_results,
        candidates=candidates,
        scenario_drop_reason_counts={reason: int(scenario_drop_counts.get(reason, 0)) for reason in REQUIRED_DROP_REASONS},
        light_drop_reason_counts={reason: int(light_drop_counts.get(reason, 0)) for reason in REQUIRED_DROP_REASONS},
        filter_reports=filter_reports,
    )


def write_signalized_index_outputs(
    build_result: SignalizedIndexBuildResult,
    *,
    data_dir: str | Path,
    outdir: str | Path,
) -> Dict[str, Path]:
    root = Path(outdir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    index_jsonl = root / "signalized_index.jsonl"
    summary_json = root / "signalized_index_summary.json"
    drop_reasons_json = root / "drop_reasons.json"
    hist_json = root / "candidate_histograms.json"

    with index_jsonl.open("w", encoding="utf-8") as f:
        for candidate in build_result.candidates:
            f.write(json.dumps(candidate.to_dict(), sort_keys=True) + "\n")

    summary_payload = {
        "data_dir": str(Path(data_dir).expanduser()),
        "scanned_scenarios": int(len(build_result.scenario_results)),
        "candidate_scenarios": int(sum(1 for result in build_result.scenario_results if result.candidates)),
        "candidate_windows": int(len(build_result.candidates)),
        "scenario_drop_reason_counts": build_result.scenario_drop_reason_counts,
        "light_drop_reason_counts": build_result.light_drop_reason_counts,
        "filter_reports": [_jsonify(asdict(report)) for report in build_result.filter_reports],
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    drop_payload = {
        "required_reasons": list(REQUIRED_DROP_REASONS),
        "scenario_drop_reasons": build_result.scenario_drop_reason_counts,
        "light_drop_reasons": build_result.light_drop_reason_counts,
        "scenario_primary_drop_reason_by_scenario": {
            result.scenario_id: result.primary_drop_reason
            for result in build_result.scenario_results
            if result.primary_drop_reason
        },
    }
    drop_reasons_json.write_text(json.dumps(drop_payload, indent=2, sort_keys=True), encoding="utf-8")
    hist_json.write_text(json.dumps(build_candidate_histograms(build_result.candidates), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "signalized_index": index_jsonl,
        "signalized_index_summary": summary_json,
        "drop_reasons": drop_reasons_json,
        "candidate_histograms": hist_json,
    }


def write_signal_qc_artifacts_for_candidate(candidate: SignalizedCandidateWindow, *, outdir: str | Path) -> Dict[str, Path]:
    canonical = load_and_normalize_scenario(candidate.scenario_pkl)
    sdc_track = canonical.tracks[candidate.sdc_id]
    curve = point_distance_curve(sdc_track.position_xy, candidate.stop_point_xy, valid_mask=sdc_track.valid)

    root = Path(outdir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    qc_path = root / "signal_qc.json"
    curve_path = root / "stop_point_distance_curve.npz"
    plot_path = root / "stop_point_distance_plot.png"

    qc_payload = {
        "scenario_id": candidate.scenario_id,
        "light_id": candidate.light_id,
        "signal_qc": candidate.signal_qc,
        "signal_state_at_time": candidate.signal_state_at_time,
        "first_time_under_35m": int(candidate.first_time_under_35m),
        "t_min_dist": int(candidate.t_min_dist),
    }
    qc_path.write_text(json.dumps(qc_payload, indent=2, sort_keys=True), encoding="utf-8")

    np.savez(
        curve_path,
        scenario_id=np.asarray(candidate.scenario_id),
        light_id=np.asarray(candidate.light_id),
        stop_point_xy=np.asarray(candidate.stop_point_xy, dtype=np.float32),
        ts=canonical.ts,
        distance_curve_m=curve,
        valid=np.asarray(sdc_track.valid, dtype=bool),
        threshold_m=np.asarray(35.0, dtype=np.float32),
        first_time_under_35m=np.asarray(candidate.first_time_under_35m, dtype=np.int32),
        t_min_dist=np.asarray(candidate.t_min_dist, dtype=np.int32),
    )
    plot_stop_point_distance_curve(
        ts=canonical.ts,
        distance_curve_m=curve,
        threshold_m=35.0,
        first_time_under_threshold_idx=candidate.first_time_under_35m,
        t_min_dist_idx=candidate.t_min_dist,
        out_path=plot_path,
    )
    return {
        "signal_qc": qc_path,
        "stop_point_distance_curve": curve_path,
        "stop_point_distance_plot": plot_path,
    }


def count_lane_like_features_near_stop_point(
    canonical: CanonicalScenario,
    *,
    stop_point_xy: Tuple[float, float],
    radius_m: float,
) -> List[str]:
    feature_ids: List[str] = []
    for feature_id, feature in sorted(canonical.map_features.items(), key=lambda item: stable_string_sort_key(item[0])):
        if not str(feature.feature_type).startswith(LANE_LIKE_PREFIX):
            continue
        if feature.polyline_xy.shape[0] == 0:
            continue
        if any_point_within_radius(feature.polyline_xy, stop_point_xy, radius_m):
            feature_ids.append(feature_id)
    return feature_ids


def compute_objects_of_interest_overlap(
    canonical: CanonicalScenario,
    *,
    stop_point_xy: Tuple[float, float],
    time_index: int,
    radius_m: float,
) -> List[str]:
    overlaps: List[str] = []
    for track_id in canonical.objects_of_interest:
        track = canonical.tracks.get(track_id)
        if track is None or time_index >= track.valid.shape[0]:
            continue
        if not bool(track.valid[time_index]) or not np.isfinite(track.position_xy[time_index]).all():
            continue
        dist = float(np.linalg.norm(np.asarray(track.position_xy[time_index], dtype=np.float32) - np.asarray(stop_point_xy, dtype=np.float32)))
        if dist <= float(radius_m):
            overlaps.append(track_id)
    return overlaps


def build_candidate_histograms(candidates: Sequence[SignalizedCandidateWindow]) -> Dict[str, Any]:
    min_dist = [candidate.min_dist_stop_point_m for candidate in candidates]
    speed = [candidate.sdc_speed_mps for candidate in candidates]
    time_idx = [candidate.first_time_under_35m for candidate in candidates]
    signal_states = Counter((candidate.signal_state_at_time or "UNKNOWN") for candidate in candidates)
    ooi_overlap = Counter(str(len(candidate.objects_of_interest_overlap)) for candidate in candidates)
    lane_like = Counter(str(candidate.lane_like_feature_count) for candidate in candidates)
    return {
        "candidate_count": int(len(candidates)),
        "min_dist_stop_point_m": _histogram(min_dist, bins=[0, 2, 5, 10, 20, 35]),
        "first_time_under_35m": _histogram(time_idx, bins=[0, 10, 20, 30, 45, 60, 90]),
        "sdc_speed_mps": _histogram(speed, bins=[0, 1, 2, 5, 10, 15, 25, 40]),
        "signal_state_at_time": dict(sorted(signal_states.items())),
        "objects_of_interest_overlap_count": dict(sorted(ooi_overlap.items(), key=lambda item: int(item[0]))),
        "lane_like_feature_count": dict(sorted(lane_like.items(), key=lambda item: int(item[0]))),
    }


def _histogram(values: Sequence[float], *, bins: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"bins": list(bins), "counts": [0] * max(0, len(bins) - 1)}
    hist, edges = np.histogram(np.asarray(values, dtype=np.float32), bins=np.asarray(bins, dtype=np.float32))
    return {"bins": [float(v) for v in edges.tolist()], "counts": [int(v) for v in hist.tolist()]}


def _finalize_rejected_result(result: SignalizedScenarioIndexResult, reason: str) -> SignalizedScenarioIndexResult:
    result.primary_drop_reason = reason
    result.scenario_drop_reasons = {reason: 1}
    result.filter_reports.append(FilterReport(stage="scenario_rejection", input_count=1, kept_count=0, dropped_count=1, drop_reasons={reason: 1}))
    return result


def _primary_drop_reason_from_counts(counts: Mapping[str, int]) -> Optional[str]:
    for reason in REQUIRED_DROP_REASONS:
        if int(counts.get(reason, 0)) > 0:
            return reason
    return None


def _extract_optional_sdc_paths(raw: Mapping[str, Any]) -> bool:
    if raw.get("sdc_paths"):
        return True
    metadata = raw.get("metadata", {})
    return bool(isinstance(metadata, Mapping) and metadata.get("sdc_paths"))
