from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .geometry import angle_delta
from .signal_qc import is_caution_signal_state, is_go_like_signal_state, is_stop_like_signal_state
from .types import stable_string_sort_key

PATH_LABELS = ("left", "straight", "right")
SAFE_COMPLIANCE_LABELS = ("obey_signal", "none")


@dataclass(frozen=True)
class LightCanonicalizationConfig:
    radius_m: float = 4.0
    require_compatible_signal_family: bool = True


@dataclass(frozen=True)
class DedupConfig:
    mode: str = "overlap_cluster"
    decision_time_merge_frames: int = 5
    window_overlap_threshold: float = 0.5
    anchor_dist_threshold: float = 5.0
    anchor_heading_threshold_rad: float = 0.35


def load_jsonl_rows(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).expanduser().write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    Path(path).expanduser().write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def histogram(values: Iterable[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        key = "null" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def summarize_path_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    artifact_mode: str,
    scenario_root: str,
    outdir: str,
    max_scenarios: int,
    seed: int,
    num_workers: int,
    num_scenarios_discovered: int,
    num_scenarios_scanned: int,
    signalized_drop_reasons: Mapping[str, int],
    path_filter_drop_reasons: Mapping[str, int],
    prefilter_stats_json: str,
    path_index_jsonl: str,
) -> Dict[str, Any]:
    label_hist = histogram(_row_branch_label(row) for row in rows)
    support = {label: int(label_hist.get(label, 0)) for label in PATH_LABELS}
    at_least_100 = {label: bool(count >= 100) for label, count in support.items()}
    below_300 = {label: bool(count < 300) for label, count in support.items()}
    warnings = [f"class_{label}_below_300" for label, flag in below_300.items() if flag]
    return {
        "artifact_mode": str(artifact_mode),
        "scenario_root": str(Path(scenario_root).expanduser()),
        "outdir": str(Path(outdir).expanduser()),
        "max_scenarios": int(max_scenarios),
        "seed": int(seed),
        "num_workers": int(num_workers),
        "num_scenarios_discovered": int(num_scenarios_discovered),
        "num_scenarios_scanned": int(num_scenarios_scanned),
        "num_scenarios_with_path_examples": int(len({str(row.get("scenario_id")) for row in rows})),
        "num_path_examples": int(len(rows)),
        "signalized_drop_reasons": dict(sorted((str(k), int(v)) for k, v in signalized_drop_reasons.items())),
        "path_filter_drop_reasons": dict(sorted((str(k), int(v)) for k, v in path_filter_drop_reasons.items())),
        "class_support": support,
        "class_support_at_least_100": at_least_100,
        "class_support_warn_below_300": below_300,
        "warnings": warnings,
        "prefilter_stats_json": str(prefilter_stats_json),
        "path_index_jsonl": str(path_index_jsonl),
        "compliance_histogram": histogram(_row_compliance_label(row) for row in rows),
    }


def build_path_histograms(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "branch_label_histogram": histogram(_row_branch_label(row) for row in rows),
        "agent_role_histogram": histogram(row.get("agent_role") for row in rows),
        "decision_state_histogram": histogram(row.get("decision_state") for row in rows),
        "signal_state_histogram": histogram(row.get("signal_state_at_decision") for row in rows),
        "target_is_sdc_histogram": histogram(row.get("is_sdc_target") for row in rows),
        "compliance_histogram": histogram(_row_compliance_label(row) for row in rows),
        "light_group_size_histogram": histogram(row.get("light_group_size") for row in rows),
        "cluster_size_histogram": histogram(row.get("cluster_size") for row in rows),
    }


def build_path_manifest(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "examples": [
            {
                "example_id": row.get("example_id"),
                "scenario_id": row.get("scenario_id"),
                "scenario_file_name": row.get("scenario_file_name"),
                "agent_id": row.get("agent_id"),
                "light_id": row.get("light_id"),
                "light_group_id": row.get("light_group_id"),
                "primary_light_id": row.get("primary_light_id"),
                "decision_time_idx": row.get("decision_time_idx"),
                "window_start_idx": row.get("window_start_idx"),
                "window_end_idx": row.get("window_end_idx"),
                "branch_label": _row_branch_label(row),
                "compliance_label": _row_compliance_label(row),
                "agent_role": row.get("agent_role"),
                "signal_state_at_decision": row.get("signal_state_at_decision"),
                "cluster_id": row.get("cluster_id"),
                "cluster_size": row.get("cluster_size"),
            }
            for row in rows
        ]
    }


def annotate_light_groups(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: Optional[LightCanonicalizationConfig] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, int]]:
    cfg = config or LightCanonicalizationConfig()
    annotated = [dict(row) for row in rows]
    group_assignments: Dict[int, Tuple[str, str, int]] = {}
    group_records: List[Dict[str, Any]] = []

    by_scenario: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for idx, row in enumerate(annotated):
        by_scenario[str(row.get("scenario_id"))].append((idx, row))

    for scenario_id in sorted(by_scenario.keys(), key=stable_string_sort_key):
        items = by_scenario[scenario_id]
        parents = list(range(len(items)))

        def _find(i: int) -> int:
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        def _union(i: int, j: int) -> None:
            root_i = _find(i)
            root_j = _find(j)
            if root_i != root_j:
                parents[root_j] = root_i

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                row_i = items[i][1]
                row_j = items[j][1]
                stop_i = _row_stop_point_xy(row_i)
                stop_j = _row_stop_point_xy(row_j)
                if stop_i is None or stop_j is None:
                    continue
                if _euclidean_distance(stop_i, stop_j) > float(cfg.radius_m):
                    continue
                if bool(cfg.require_compatible_signal_family):
                    family_i = _signal_family(row_i.get("signal_state_at_decision"))
                    family_j = _signal_family(row_j.get("signal_state_at_decision"))
                    if not _signal_families_compatible(family_i, family_j):
                        continue
                _union(i, j)

        component_members: Dict[int, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
        for local_idx, item in enumerate(items):
            component_members[_find(local_idx)].append(item)

        ordered_components = sorted(
            component_members.values(),
            key=lambda members: (
                stable_string_sort_key(min(str(member[1].get("light_id")) for member in members)),
                min(int(member[1].get("decision_time_idx", 0)) for member in members),
            ),
        )
        for component_idx, members in enumerate(ordered_components):
            unique_light_ids = sorted({str(member[1].get("light_id")) for member in members}, key=stable_string_sort_key)
            primary_light_id = unique_light_ids[0]
            light_group_id = f"{scenario_id}__lg_{component_idx:03d}"
            group_size = len(unique_light_ids)
            group_records.append(
                {
                    "scenario_id": scenario_id,
                    "light_group_id": light_group_id,
                    "primary_light_id": primary_light_id,
                    "light_group_size": int(group_size),
                    "member_light_ids": list(unique_light_ids),
                    "member_row_count": int(len(members)),
                }
            )
            for global_idx, _row in members:
                group_assignments[global_idx] = (light_group_id, primary_light_id, int(group_size))

    for idx, row in enumerate(annotated):
        light_group_id, primary_light_id, group_size = group_assignments.get(
            idx,
            (f"{row.get('scenario_id')}__lg_single_{row.get('light_id')}", str(row.get("light_id")), 1),
        )
        row["light_group_id"] = light_group_id
        row["primary_light_id"] = primary_light_id
        row["light_group_size"] = int(group_size)

    unique_light_ids = {(str(row.get("scenario_id")), str(row.get("light_id"))) for row in annotated}
    unique_groups = {(str(row.get("scenario_id")), str(row.get("light_group_id"))) for row in annotated}
    collapsible = Counter(
        (
            str(row.get("scenario_id")),
            str(row.get("agent_id")),
            _row_branch_label(row),
            int(row.get("decision_time_idx", 0)),
            str(row.get("light_group_id")),
        )
        for row in annotated
    )
    summary = {
        "light_group_radius_m": float(cfg.radius_m),
        "require_compatible_signal_family": bool(cfg.require_compatible_signal_family),
        "num_rows_total": int(len(annotated)),
        "num_unique_raw_lights": int(len(unique_light_ids)),
        "num_unique_light_groups": int(len(unique_groups)),
        "num_multi_light_groups": int(sum(1 for group in group_records if int(group["light_group_size"]) > 1)),
        "num_rows_in_multi_light_groups": int(sum(1 for row in annotated if int(row.get("light_group_size", 1)) > 1)),
        "num_rows_collapsible_by_light_group": int(sum(max(count - 1, 0) for count in collapsible.values())),
    }
    group_hist = histogram(group["light_group_size"] for group in group_records)
    return annotated, summary, group_hist


def cluster_path_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: Optional[DedupConfig] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    cfg = config or DedupConfig()
    annotated_rows = [dict(row) for row in rows]
    cluster_rows: List[Dict[str, Any]] = []
    curated_rows: List[Dict[str, Any]] = []

    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in annotated_rows:
        key = (
            str(row.get("scenario_id")),
            str(row.get("agent_id")),
            _row_branch_label(row),
            str(row.get("light_group_id") or row.get("light_id")),
        )
        grouped[key].append(row)

    for key in sorted(grouped.keys(), key=lambda item: tuple(stable_string_sort_key(value) for value in item)):
        members = sorted(
            grouped[key],
            key=lambda row: (
                int(row.get("decision_time_idx", 0)),
                int(row.get("window_start_idx", 0)),
                stable_string_sort_key(str(row.get("light_id"))),
            ),
        )
        components = _cluster_group_members(members, config=cfg)
        for component_idx, component in enumerate(components):
            ranked = sorted(component, key=_representative_sort_key)
            scenario_id = str(ranked[0].get("scenario_id"))
            agent_id = str(ranked[0].get("agent_id"))
            branch_label = _row_branch_label(ranked[0])
            light_group_id = str(ranked[0].get("light_group_id") or ranked[0].get("light_id"))
            cluster_id = f"{scenario_id}__agent_{agent_id}__{branch_label}__{light_group_id}__cluster_{component_idx:03d}"
            for rank, row in enumerate(ranked, start=1):
                annotated = dict(row)
                annotated["cluster_id"] = cluster_id
                annotated["cluster_size"] = int(len(ranked))
                annotated["representative_rank_within_cluster"] = int(rank)
                annotated["primary_light_id"] = str(annotated.get("primary_light_id") or annotated.get("light_id"))
                cluster_rows.append(annotated)
                if rank == 1:
                    curated_rows.append(annotated)

    cluster_sizes = [int(row.get("cluster_size", 1)) for row in curated_rows]
    stats = {
        "dedup_mode": str(cfg.mode),
        "decision_time_merge_frames": int(cfg.decision_time_merge_frames),
        "window_overlap_threshold": float(cfg.window_overlap_threshold),
        "anchor_dist_threshold": float(cfg.anchor_dist_threshold),
        "anchor_heading_threshold_rad": float(cfg.anchor_heading_threshold_rad),
        "unique_decision_clusters_estimate": int(len(curated_rows)),
        "mean_rows_per_cluster": float(sum(cluster_sizes) / max(len(cluster_sizes), 1)),
        "max_rows_per_cluster": int(max(cluster_sizes) if cluster_sizes else 0),
        "num_rows_before_dedup": int(len(rows)),
        "num_rows_after_dedup": int(len(curated_rows)),
        "num_rows_removed_by_dedup": int(max(len(rows) - len(curated_rows), 0)),
    }
    return curated_rows, cluster_rows, stats


def apply_path_only_safe_mode(
    rows: Sequence[Mapping[str, Any]],
    *,
    enabled: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not enabled:
        return [dict(row) for row in rows], {
            "path_only_safe_mode": False,
            "num_rows_input": int(len(rows)),
            "num_rows_kept": int(len(rows)),
            "num_rows_dropped": 0,
            "drop_reason_counts": {},
        }

    kept: List[Dict[str, Any]] = []
    drop_counts: Counter[str] = Counter()
    for row in rows:
        branch_label = _row_branch_label(row)
        compliance_label = _row_compliance_label(row)
        terminal_anchor = _coerce_mapping(row.get("terminal_anchor"))
        if branch_label not in PATH_LABELS:
            drop_counts["unsupported_branch_label"] += 1
            continue
        if not bool(row.get("conditioning_eligible")):
            drop_counts["conditioning_ineligible"] += 1
            continue
        if not bool(row.get("path_choice_supervisable")):
            drop_counts["path_not_supervisable"] += 1
            continue
        if not terminal_anchor:
            drop_counts["missing_terminal_anchor"] += 1
            continue
        if compliance_label == "red_light_violation":
            drop_counts["red_light_violation_excluded"] += 1
            continue
        if compliance_label not in SAFE_COMPLIANCE_LABELS:
            drop_counts["unsupported_compliance_label"] += 1
            continue
        kept.append(dict(row))
    return kept, {
        "path_only_safe_mode": True,
        "num_rows_input": int(len(rows)),
        "num_rows_kept": int(len(kept)),
        "num_rows_dropped": int(sum(drop_counts.values())),
        "drop_reason_counts": dict(sorted(drop_counts.items())),
    }


def split_rows_by_scenario(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 0,
    val_fraction: float = 0.2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    scenario_ids = sorted({str(row.get("scenario_id")) for row in rows}, key=stable_string_sort_key)
    rng = random.Random(int(seed))
    scenario_ids = list(scenario_ids)
    rng.shuffle(scenario_ids)
    if len(scenario_ids) <= 1:
        val_ids: set[str] = set()
    else:
        num_val = max(1, int(round(len(scenario_ids) * float(val_fraction))))
        num_val = min(num_val, len(scenario_ids) - 1)
        val_ids = set(scenario_ids[:num_val])
    train_ids = {scenario_id for scenario_id in scenario_ids if scenario_id not in val_ids}
    train_rows = [dict(row) for row in rows if str(row.get("scenario_id")) in train_ids]
    val_rows = [dict(row) for row in rows if str(row.get("scenario_id")) in val_ids]
    overlap = train_ids & val_ids
    summary = {
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "num_train_rows": int(len(train_rows)),
        "num_val_rows": int(len(val_rows)),
        "num_train_scenarios": int(len(train_ids)),
        "num_val_scenarios": int(len(val_ids)),
        "train_class_histogram": histogram(_row_branch_label(row) for row in train_rows),
        "val_class_histogram": histogram(_row_branch_label(row) for row in val_rows),
        "scenario_overlap_count": int(len(overlap)),
        "scenario_overlap_ids": sorted(overlap, key=stable_string_sort_key),
    }
    return train_rows, val_rows, summary


def analyze_path_index_redundancy(
    rows: Sequence[Mapping[str, Any]],
    *,
    dedup_config: Optional[DedupConfig] = None,
    light_config: Optional[LightCanonicalizationConfig] = None,
) -> Dict[str, Any]:
    light_annotated, light_summary, _ = annotate_light_groups(rows, config=light_config)
    _, clustered_rows, dedup_stats = cluster_path_rows(light_annotated, config=dedup_config)

    total_rows = len(light_annotated)
    by_scenario_agent_branch: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_scenario_agent_decision: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in light_annotated:
        by_scenario_agent_branch[(str(row.get("scenario_id")), str(row.get("agent_id")), _row_branch_label(row))].append(dict(row))
        by_scenario_agent_decision[
            (str(row.get("scenario_id")), str(row.get("agent_id")), int(row.get("decision_time_idx", 0)))
        ].append(dict(row))

    duplicate_groups: List[Dict[str, Any]] = []
    window_overlap_values: List[float] = []
    for key, members in by_scenario_agent_branch.items():
        if len(members) <= 1:
            continue
        overlap_values = _pairwise_window_overlaps(members)
        window_overlap_values.extend(overlap_values)
        duplicate_groups.append(
            {
                "scenario_id": key[0],
                "agent_id": key[1],
                "branch_label": key[2],
                "num_rows": int(len(members)),
                "decision_time_indices": sorted(int(row.get("decision_time_idx", 0)) for row in members),
                "light_ids": sorted({str(row.get("light_id")) for row in members}, key=stable_string_sort_key),
                "light_group_ids": sorted({str(row.get("light_group_id")) for row in members}, key=stable_string_sort_key),
                "primary_light_ids": sorted({str(row.get("primary_light_id")) for row in members}, key=stable_string_sort_key),
                "window_overlap_mean": float(sum(overlap_values) / max(len(overlap_values), 1)),
                "window_overlap_max": float(max(overlap_values) if overlap_values else 0.0),
                "compliance_histogram": histogram(_row_compliance_label(row) for row in members),
            }
        )
    duplicate_groups = sorted(
        duplicate_groups,
        key=lambda item: (-int(item["num_rows"]), stable_string_sort_key(item["scenario_id"]), stable_string_sort_key(item["agent_id"])),
    )

    compliance_hist = histogram(_row_compliance_label(row) for row in light_annotated)
    cluster_sizes = [int(row.get("cluster_size", 1)) for row in clustered_rows if int(row.get("representative_rank_within_cluster", 0)) == 1]
    summary = {
        "num_rows_total": int(total_rows),
        "num_unique_scenarios": int(len({str(row.get("scenario_id")) for row in light_annotated})),
        "num_unique_(scenario_id,agent_id)": int(len({(str(row.get("scenario_id")), str(row.get("agent_id"))) for row in light_annotated})),
        "num_unique_(scenario_id,agent_id,branch_label)": int(
            len({(str(row.get("scenario_id")), str(row.get("agent_id")), _row_branch_label(row)) for row in light_annotated})
        ),
        "num_unique_(scenario_id,agent_id,branch_label,light_id)": int(
            len(
                {
                    (str(row.get("scenario_id")), str(row.get("agent_id")), _row_branch_label(row), str(row.get("light_id")))
                    for row in light_annotated
                }
            )
        ),
        "num_duplicate_rows_by_same_(scenario_id,agent_id,branch_label)": int(
            sum(max(len(members) - 1, 0) for members in by_scenario_agent_branch.values())
        ),
        "num_duplicate_rows_by_same_(scenario_id,agent_id,decision_time_idx)": int(
            sum(max(len(members) - 1, 0) for members in by_scenario_agent_decision.values())
        ),
        "count_compliance_label_red_light_violation": int(compliance_hist.get("red_light_violation", 0)),
        "count_compliance_label_obey_signal": int(compliance_hist.get("obey_signal", 0)),
        "count_compliance_label_none": int(compliance_hist.get("none", 0)),
        "top_50_duplicated_groups": duplicate_groups[:50],
        "unique_decision_clusters_estimate": int(dedup_stats["unique_decision_clusters_estimate"]),
        "mean_rows_per_cluster": float(dedup_stats["mean_rows_per_cluster"]),
        "max_rows_per_cluster": int(dedup_stats["max_rows_per_cluster"]),
    }
    return {
        "redundancy_summary": summary,
        "duplicated_groups": duplicate_groups,
        "window_overlap_histograms": {
            "within_same_(scenario,agent,branch)": _bucket_window_overlaps(window_overlap_values),
        },
        "light_duplication_summary": light_summary,
        "compliance_histogram": compliance_hist,
        "scenario_agent_branch_counts": _scenario_agent_branch_counts(light_annotated),
    }


def _cluster_group_members(rows: Sequence[Dict[str, Any]], *, config: DedupConfig) -> List[List[Dict[str, Any]]]:
    if str(config.mode) == "none":
        return [[dict(row)] for row in rows]
    parents = list(range(len(rows)))

    def _find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def _union(i: int, j: int) -> None:
        root_i = _find(i)
        root_j = _find(j)
        if root_i != root_j:
            parents[root_j] = root_i

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if _rows_should_cluster(rows[i], rows[j], config=config):
                _union(i, j)

    components: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        components[_find(idx)].append(dict(row))
    return sorted(
        components.values(),
        key=lambda members: (
            min(int(row.get("decision_time_idx", 0)) for row in members),
            min(int(row.get("window_start_idx", 0)) for row in members),
            stable_string_sort_key(str(members[0].get("scenario_id"))),
        ),
    )


def _rows_should_cluster(row_a: Mapping[str, Any], row_b: Mapping[str, Any], *, config: DedupConfig) -> bool:
    anchor_a = _row_terminal_anchor(row_a)
    anchor_b = _row_terminal_anchor(row_b)
    if anchor_a is None or anchor_b is None:
        return False
    anchor_dist = _euclidean_distance(anchor_a[:2], anchor_b[:2])
    if anchor_dist > float(config.anchor_dist_threshold):
        return False
    heading_delta = abs(angle_delta(anchor_a[2], anchor_b[2]))
    if heading_delta > float(config.anchor_heading_threshold_rad):
        return False
    decision_close = abs(int(row_a.get("decision_time_idx", 0)) - int(row_b.get("decision_time_idx", 0))) <= int(config.decision_time_merge_frames)
    overlap = window_overlap_fraction(row_a, row_b)
    return bool(decision_close or overlap >= float(config.window_overlap_threshold))


def window_overlap_fraction(row_a: Mapping[str, Any], row_b: Mapping[str, Any]) -> float:
    start_a, end_a = _row_window(row_a)
    start_b, end_b = _row_window(row_b)
    intersection = max(0, min(end_a, end_b) - max(start_a, start_b) + 1)
    length_a = max(0, end_a - start_a + 1)
    length_b = max(0, end_b - start_b + 1)
    denom = max(min(length_a, length_b), 1)
    return float(intersection / denom)


def _pairwise_window_overlaps(rows: Sequence[Mapping[str, Any]]) -> List[float]:
    values: List[float] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            values.append(window_overlap_fraction(rows[i], rows[j]))
    return values


def _bucket_window_overlaps(values: Sequence[float]) -> Dict[str, int]:
    buckets = {
        "0.00-0.25": 0,
        "0.25-0.50": 0,
        "0.50-0.75": 0,
        "0.75-1.00": 0,
    }
    for value in values:
        if value < 0.25:
            buckets["0.00-0.25"] += 1
        elif value < 0.50:
            buckets["0.25-0.50"] += 1
        elif value < 0.75:
            buckets["0.50-0.75"] += 1
        else:
            buckets["0.75-1.00"] += 1
    return buckets


def _representative_sort_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    compliance_label = _row_compliance_label(row)
    branch_margin = _finite_float(row.get("branch_margin"), default=float("-inf"))
    downstream_progress = _finite_float(row.get("downstream_progress_along_branch_m"), default=float("-inf"))
    final_heading_error = _finite_float(row.get("final_heading_error_rad"), default=float("inf"))
    return (
        0 if bool(row.get("control_available_at_current")) else 1,
        0 if bool(row.get("path_choice_supervisable")) else 1,
        0 if compliance_label in SAFE_COMPLIANCE_LABELS else 1,
        -branch_margin,
        -downstream_progress,
        final_heading_error,
        int(row.get("decision_time_idx", 0)),
        stable_string_sort_key(str(row.get("primary_light_id") or row.get("light_id"))),
        stable_string_sort_key(str(row.get("example_id") or "")),
    )


def _scenario_agent_branch_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    per_scenario: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        scenario_id = str(row.get("scenario_id"))
        payload = per_scenario.setdefault(
            scenario_id,
            {
                "num_rows": 0,
                "unique_agents": set(),
                "unique_(agent,branch)": set(),
                "branch_histogram": Counter(),
            },
        )
        payload["num_rows"] += 1
        payload["unique_agents"].add(str(row.get("agent_id")))
        payload["unique_(agent,branch)"].add((str(row.get("agent_id")), _row_branch_label(row)))
        payload["branch_histogram"][_row_branch_label(row)] += 1
    output: Dict[str, Any] = {}
    for scenario_id, payload in sorted(per_scenario.items(), key=lambda item: stable_string_sort_key(item[0])):
        output[scenario_id] = {
            "num_rows": int(payload["num_rows"]),
            "num_unique_agents": int(len(payload["unique_agents"])),
            "num_unique_(agent,branch)": int(len(payload["unique_(agent,branch)"])),
            "branch_histogram": dict(sorted(payload["branch_histogram"].items())),
        }
    return output


def _row_window(row: Mapping[str, Any]) -> Tuple[int, int]:
    if "window_start_idx" in row and "window_end_idx" in row:
        return int(row.get("window_start_idx", 0)), int(row.get("window_end_idx", 0))
    window = _coerce_mapping(row.get("window"))
    return int(window.get("start_idx", 0)), int(window.get("end_idx", 0))


def _row_stop_point_xy(row: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    if "stop_point_xy" in row:
        value = row.get("stop_point_xy")
    else:
        value = _coerce_mapping(row.get("compliance_token")).get("stop_point_xy")
    if not isinstance(value, Sequence) or len(value) < 2:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
    except Exception:
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return float(x), float(y)


def _row_terminal_anchor(row: Mapping[str, Any]) -> Optional[Tuple[float, float, float]]:
    anchor = _coerce_mapping(row.get("terminal_anchor"))
    if not anchor:
        return None
    x = _finite_optional_float(anchor.get("target_x_rel"))
    y = _finite_optional_float(anchor.get("target_y_rel"))
    sin_h = _finite_optional_float(anchor.get("target_sin_heading_rel"))
    cos_h = _finite_optional_float(anchor.get("target_cos_heading_rel"))
    if x is None or y is None or sin_h is None or cos_h is None:
        return None
    return float(x), float(y), float(math.atan2(sin_h, cos_h))


def _row_branch_label(row: Mapping[str, Any]) -> str:
    value = row.get("branch_label")
    if value is None:
        value = _coerce_mapping(row.get("path_token")).get("branch_label")
    return "" if value is None else str(value)


def _row_compliance_label(row: Mapping[str, Any]) -> str:
    value = row.get("compliance_label")
    if value is None:
        value = _coerce_mapping(row.get("compliance_token")).get("compliance_label")
    return "" if value is None else str(value)


def _signal_family(signal_state: Any) -> str:
    if signal_state is None:
        return "unknown"
    text = str(signal_state)
    if is_stop_like_signal_state(text):
        return "stop"
    if is_go_like_signal_state(text):
        return "go"
    if is_caution_signal_state(text):
        return "caution"
    return "unknown"


def _signal_families_compatible(a: str, b: str) -> bool:
    return a == "unknown" or b == "unknown" or a == b


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _finite_optional_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def _finite_float(value: Any, default: float) -> float:
    maybe = _finite_optional_float(value)
    if maybe is None:
        return float(default)
    return float(maybe)


def _euclidean_distance(point_a: Sequence[float], point_b: Sequence[float]) -> float:
    dx = float(point_a[0]) - float(point_b[0])
    dy = float(point_a[1]) - float(point_b[1])
    return float(math.hypot(dx, dy))
