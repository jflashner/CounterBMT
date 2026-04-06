from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    vendored_scenarionet = repo_root / "scenarionet"
    vendored_metadrive = repo_root / "metadrive"
    for path in (vendored_scenarionet, vendored_metadrive, repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual import audit_gt_future_against_branch_routes_v2, build_branch_routes_v2, load_and_normalize_scenario
from bmt.counterfactual.path_eval_bundle import (
    find_bundle_checkpoint,
    find_bundle_config_yaml,
    find_materialized_eval_dir,
    load_json,
    load_materialized_controls,
    load_model_and_tokenizer_for_bundle,
    load_raw_scenario,
    preprocess_raw_scenario_for_audit,
    raw_track_world_state,
    run_control_variant,
    write_json,
)
from bmt.counterfactual.branch_routes_v2 import render_branch_routes_overlay_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit lane-sequence branch routes on the local eval bundle.")
    parser.add_argument("--bundle-root", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--selected-manifest", type=str, default="")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument(
        "--load-mode",
        type=str,
        default="forgiving_state_dict",
        choices=("forgiving_state_dict", "strict_state_dict"),
    )
    parser.add_argument("--skip-model", action="store_true")
    return parser.parse_args()


def _track_pose_at_index(state: Mapping[str, np.ndarray], *, time_index: int, fallback_heading: float = 0.0) -> Tuple[np.ndarray, float]:
    valid = np.asarray(state["valid"], dtype=bool)
    position = np.asarray(state["position"], dtype=np.float64)
    heading = np.asarray(state["heading"], dtype=np.float64)
    if valid.size == 0:
        return np.zeros((2,), dtype=np.float64), float(fallback_heading)
    idx = int(np.clip(int(time_index), 0, valid.shape[0] - 1))
    if not bool(valid[idx]) or not np.isfinite(position[idx]).all():
        before = np.flatnonzero(valid[: idx + 1])
        if before.size > 0:
            idx = int(before[-1])
        else:
            after = np.flatnonzero(valid[idx:])
            idx = int(idx + after[0]) if after.size > 0 else idx
    heading_val = float(heading[idx]) if idx < heading.shape[0] and np.isfinite(heading[idx]) else float(fallback_heading)
    return np.asarray(position[idx, :2], dtype=np.float64), heading_val


def _valid_future_xy(state: Mapping[str, np.ndarray], *, start_idx: int) -> np.ndarray:
    valid = np.asarray(state["valid"], dtype=bool)
    position = np.asarray(state["position"], dtype=np.float64)
    indices = [
        idx
        for idx in range(max(0, int(start_idx)), valid.shape[0])
        if bool(valid[idx]) and np.isfinite(position[idx]).all()
    ]
    if not indices:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(position[indices, :2], dtype=np.float64)


def _shared_stem_length_m(polyline_xy: np.ndarray) -> float:
    polyline = np.asarray(polyline_xy, dtype=np.float64)
    if polyline.ndim != 2 or polyline.shape[0] < 2:
        return 0.0
    deltas = np.diff(polyline[:, :2], axis=0)
    return float(np.sum(np.linalg.norm(deltas, axis=-1)))


def _load_selected_manifest(bundle_root: Path, explicit_path: str) -> List[Dict[str, Any]]:
    if explicit_path:
        return list(load_json(explicit_path))
    default_path = bundle_root / "audit_local" / "selected_examples_manifest.json"
    if default_path.is_file():
        return list(load_json(default_path))
    raise FileNotFoundError(f"Could not find selected_examples_manifest.json under {bundle_root}")


def _resolve_paths(
    *,
    bundle_root: Path,
    item: Mapping[str, Any],
) -> Tuple[Path, Path]:
    scenario_pkl = str(item.get("local_scenario_pkl") or "").strip()
    if not scenario_pkl:
        raise FileNotFoundError(f"{item.get('example_id')}: missing local_scenario_pkl")
    materialized_dir = str(item.get("local_materialized_eval_input") or "").strip()
    if not materialized_dir:
        found = find_materialized_eval_dir(
            bundle_root=bundle_root,
            example_id=str(item.get("example_id")),
            scenario_id=str(item.get("scenario_id")),
            agent_id=str(item.get("agent_id")),
            decision_time_idx=int(item.get("decision_time_idx", 0)),
        )
        if found is None:
            raise FileNotFoundError(f"{item.get('example_id')}: missing local_materialized_eval_input")
        materialized_dir = str(found)
    return Path(scenario_pkl).expanduser().resolve(), Path(materialized_dir).expanduser().resolve()


def _load_model_bundle(
    *,
    bundle_root: Path,
    config_path: str,
    load_mode: str,
) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Dict[str, Any]]:
    if not bundle_root.exists():
        return None, None, None, {"loaded": False, "reason": "bundle_root_missing"}
    ckpt_path = find_bundle_checkpoint(bundle_root)
    resolved_config = Path(config_path).expanduser().resolve() if config_path else find_bundle_config_yaml(bundle_root)
    if ckpt_path is None:
        return None, None, None, {"loaded": False, "reason": "checkpoint_missing"}
    try:
        config, model, tokenizer, load_report = load_model_and_tokenizer_for_bundle(
            ckpt_path=ckpt_path,
            config_path=resolved_config,
            load_mode=load_mode,
        )
        return config, model, tokenizer, {"loaded": True, "checkpoint": str(ckpt_path), "config": (None if resolved_config is None else str(resolved_config)), "load_report": load_report}
    except Exception as exc:
        return None, None, None, {"loaded": False, "reason": f"model_load_failed: {exc}"}


def main() -> int:
    args = parse_args()
    bundle_root = Path(args.bundle_root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    selected_manifest = _load_selected_manifest(bundle_root, args.selected_manifest)
    write_json(outdir / "selected_examples_manifest.json", selected_manifest)

    config = model = tokenizer = None
    model_status: Dict[str, Any] = {"loaded": False, "reason": "skip_model_requested"}
    if not bool(args.skip_model):
        config, model, tokenizer, model_status = _load_model_bundle(
            bundle_root=bundle_root,
            config_path=str(args.config or ""),
            load_mode=str(args.load_mode),
        )
    write_json(outdir / "branch_routes_v2_model_status.json", model_status)

    base_sample_cache: Dict[str, Dict[str, Any]] = {}
    split_rows: List[Dict[str, Any]] = []
    gt_rows: List[Dict[str, Any]] = []

    for item in selected_manifest:
        example_id = str(item["example_id"])
        scenario_pkl, materialized_dir = _resolve_paths(bundle_root=bundle_root, item=item)
        raw_scenario = load_raw_scenario(scenario_pkl)
        canonical = load_and_normalize_scenario(scenario_pkl)
        materialized = load_materialized_controls(materialized_dir)
        factual_control = materialized.get("factual_control_code") or {}
        local_intervention = materialized.get("local_intervention_train_view") or {}
        provenance = dict(factual_control.get("debug", {}).get("source_provenance", {})) or dict(local_intervention.get("provenance", {}))
        context = dict(local_intervention.get("context", {}))

        agent_id = str(item["agent_id"])
        current_time_idx = int(provenance.get("current_time_index_global", canonical.current_time_index))
        decision_time_idx = int(provenance.get("decision_time_index_global", item.get("decision_time_idx", canonical.current_time_index)))
        stop_point_xy = tuple(float(v) for v in context.get("stop_point_xy", [0.0, 0.0])[:2])
        approach_heading = float(context.get("approach_heading", 0.0))

        route_result = build_branch_routes_v2(
            canonical,
            agent_id=agent_id,
            current_time_idx=current_time_idx,
            decision_time_idx=decision_time_idx,
            stop_point_xy=stop_point_xy,
            approach_heading=approach_heading,
        )
        gt_audit = audit_gt_future_against_branch_routes_v2(
            canonical,
            agent_id=agent_id,
            decision_time_idx=decision_time_idx,
            route_result=route_result,
        )

        track_state = raw_track_world_state(raw_scenario, track_id=agent_id)
        current_pose_xy, current_heading = _track_pose_at_index(track_state, time_index=current_time_idx, fallback_heading=approach_heading)
        decision_pose_xy, decision_heading = _track_pose_at_index(track_state, time_index=decision_time_idx, fallback_heading=approach_heading)
        gt_future_xy = _valid_future_xy(track_state, start_idx=current_time_idx)

        factual_rollout_xy = None
        if config is not None and model is not None and tokenizer is not None and factual_control:
            scenario_key = str(item["scenario_id"])
            if scenario_key not in base_sample_cache:
                base_sample_cache[scenario_key] = preprocess_raw_scenario_for_audit(raw_scenario, config=config, tokenizer=tokenizer)
            factual_variant = run_control_variant(
                base_sample=base_sample_cache[scenario_key],
                scenario_id=scenario_key,
                mode="factual",
                control_code=factual_control,
                model=model,
                tokenizer=tokenizer,
                sampling_method="argmax",
                temperature=1.0,
                topp=1.0,
                seed=0,
                deterministic_agent_ids=True,
            )
            mask = np.asarray(factual_variant["target_valid_mask"], dtype=bool)
            factual_rollout_xy = np.asarray(factual_variant["target_positions_world"], dtype=np.float64)[mask]

        example_outdir = outdir / example_id
        example_outdir.mkdir(parents=True, exist_ok=True)
        route_payload = route_result.to_dict()
        route_payload["example_id"] = example_id
        route_payload["scenario_id"] = str(item["scenario_id"])
        route_payload["agent_id"] = agent_id
        route_payload["current_pose_world"] = {
            "x": float(current_pose_xy[0]),
            "y": float(current_pose_xy[1]),
            "heading": float(current_heading),
        }
        route_payload["decision_pose_world"] = {
            "x": float(decision_pose_xy[0]),
            "y": float(decision_pose_xy[1]),
            "heading": float(decision_heading),
        }
        route_payload["requested_branch_label"] = str(factual_control.get("path_token", {}).get("branch_label") or "")
        route_payload["gt_branch_match_audit"] = gt_audit
        write_json(example_outdir / "branch_routes_v2.json", route_payload)

        render_branch_routes_overlay_v2(
            output_path=example_outdir / "branch_routes_overlay_v2.png",
            canonical=canonical,
            route_result=route_result,
            current_pose_xy=current_pose_xy,
            decision_pose_xy=decision_pose_xy,
            gt_future_xy=gt_future_xy,
            factual_rollout_xy=factual_rollout_xy,
            title=f"{example_id} | requested={route_payload['requested_branch_label'] or 'none'}",
        )

        split_point_xy = route_result.split_point_xy
        split_rows.append(
            {
                "example_id": example_id,
                "scenario_id": str(item["scenario_id"]),
                "agent_id": agent_id,
                "requested_branch_label": route_payload["requested_branch_label"],
                "route_family_labels": [family.branch_label for family in route_result.route_families],
                "num_route_families": int(len(route_result.route_families)),
                "route_v2_training_eligible": bool(len(route_result.route_families) >= 2),
                "shared_stem_length_m": float(_shared_stem_length_m(route_result.shared_stem_xy)),
                "split_point_xy": (None if split_point_xy is None else [float(split_point_xy[0]), float(split_point_xy[1])]),
                "split_point_to_decision_m": (
                    None
                    if split_point_xy is None
                    else float(np.linalg.norm(np.asarray(split_point_xy, dtype=np.float64) - np.asarray(decision_pose_xy, dtype=np.float64)))
                ),
                "host_lane_current": (None if route_result.host_lane_current is None else route_result.host_lane_current.to_dict()),
                "host_lane_decision": (None if route_result.host_lane_decision is None else route_result.host_lane_decision.to_dict()),
                "current_to_decision_connected": bool(route_result.current_to_decision_connected),
                "local_lane_node_count": int(route_result.local_lane_node_count),
                "local_lane_edge_count": int(route_result.local_lane_edge_count),
            }
        )
        gt_rows.append(
            {
                "example_id": example_id,
                "scenario_id": str(item["scenario_id"]),
                "agent_id": agent_id,
                "requested_branch_label_factual": route_payload["requested_branch_label"],
                "num_route_families": int(len(route_result.route_families)),
                "route_v2_training_eligible": bool(len(route_result.route_families) >= 2 and not bool(gt_audit.get("drop_for_training"))),
                "gt_branch_label": gt_audit.get("gt_branch_label"),
                "gt_branch_id": gt_audit.get("gt_branch_id"),
                "best_score": gt_audit.get("best_score"),
                "score_margin": gt_audit.get("score_margin"),
                "ambiguous": bool(gt_audit.get("ambiguous")),
                "drop_for_training": bool(gt_audit.get("drop_for_training")),
                "matches_factual_requested": bool(gt_audit.get("gt_branch_label") == route_payload["requested_branch_label"]) if route_payload["requested_branch_label"] else None,
                "scores": gt_audit.get("scores", []),
            }
        )

    split_distances = [row["split_point_to_decision_m"] for row in split_rows if row["split_point_to_decision_m"] is not None]
    split_summary = {
        "num_examples": int(len(split_rows)),
        "num_examples_with_split_point": int(sum(row["split_point_xy"] is not None for row in split_rows)),
        "num_examples_with_connected_current_to_decision": int(sum(bool(row["current_to_decision_connected"]) for row in split_rows)),
        "num_examples_with_ge2_route_families": int(sum(int(row["num_route_families"]) >= 2 for row in split_rows)),
        "mean_split_point_to_decision_m": (None if not split_distances else float(np.mean(np.asarray(split_distances, dtype=np.float64)))),
        "rows": split_rows,
        "model_status": model_status,
    }
    gt_summary = {
        "num_examples": int(len(gt_rows)),
        "num_unambiguous_examples": int(sum(not bool(row["ambiguous"]) for row in gt_rows)),
        "num_drop_for_training": int(sum(bool(row["drop_for_training"]) for row in gt_rows)),
        "num_route_v2_training_eligible": int(sum(bool(row["route_v2_training_eligible"]) for row in gt_rows)),
        "gt_matches_factual_requested_rate": (
            None
            if not gt_rows
            else float(
                sum(bool(row["matches_factual_requested"]) for row in gt_rows if row["matches_factual_requested"] is not None)
                / max(1, sum(row["matches_factual_requested"] is not None for row in gt_rows))
            )
        ),
        "rows": gt_rows,
        "model_status": model_status,
    }
    write_json(outdir / "split_point_audit.json", split_summary)
    write_json(outdir / "gt_branch_match_audit.json", gt_summary)

    print(json.dumps({"outdir": str(outdir), "num_examples": len(selected_manifest)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
