from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    modern_src = repo_root / "src"
    legacy_src = repo_root / "src" / "Adv-BMT"
    vendored_scenarionet = repo_root / "scenarionet"
    vendored_metadrive = repo_root / "metadrive"
    for path in (vendored_scenarionet, vendored_metadrive, modern_src, repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual.forward_supervision import (  # noqa: E402
    preprocess_raw_scenario_for_forward_supervision,
    summarize_forward_supervision_for_sample,
)
from bmt.counterfactual.normalize import load_raw_scenario  # noqa: E402
from bmt.counterfactual.sdc_path_control import normalize_semantic_label  # noqa: E402
from bmt.counterfactual.scenarionet_waymo_export_source import (  # noqa: E402
    DEFAULT_WOD_131_TRAIN_PATH,
    materialize_scenarionet_waymo_sources,
)
from bmt.counterfactual.sdc_semantic_control import extract_model_frame  # noqa: E402
from counter_bmt.scenario_export import (  # noqa: E402
    create_dataset_summary,
    export_victim_centric_ground_truth_scenario,
    export_victim_centric_scenario,
)
from scripts.counterfactual.probe_agent_semantic_rollout import (  # noqa: E402
    _build_control_sample,
    _build_eval_module,
    _build_time_window_mask,
    _extract_all_reference_world_from_sample,
    _extract_target_action_tokens_from_output_np,
    _extract_target_rollout_world_from_output_np,
    _load_config,
    _load_model,
    _normalize_track_id,
    _path_length_m,
    _resolve_device,
    _run_rollout,
    _save_overlay_plot,
    _save_victim_centric_overlay_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an offline victim-centric CounterBMT dataset for Table 4 style agent evaluation."
    )
    parser.add_argument("--control-index", type=str, required=True)
    parser.add_argument("--scenario-root", type=str, default="")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default="src/Adv-BMT/cfgs/motion_forward_sdc_semantic_only_progresssoft_topomcpo_dag_trafficcap_progresson.yaml",
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--split-name", type=str, default="train")
    parser.add_argument("--num-scenes", type=int, default=500)
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--load-mode", type=str, default="forgiving_state_dict")
    parser.add_argument("--teacher-ckpt", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--semantic-label", action="append", dest="semantic_labels", default=[])
    parser.add_argument("--semantic-confidence", type=float, default=1.0)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int, default=-1)
    parser.add_argument("--rollout-sampling-method", type=str, default="argmax")
    parser.add_argument("--rollout-temperature", type=float, default=-1.0)
    parser.add_argument("--rollout-topp", type=float, default=-1.0)
    parser.add_argument("--min-moving-speed-mps", type=float, default=0.5)
    parser.add_argument("--max-distance-to-sdc-m", type=float, default=40.0)
    parser.add_argument("--max-adversary-candidates", type=int, default=3)
    parser.add_argument("--min-final-position-delta-m", type=float, default=1.0)
    parser.add_argument("--min-changed-action-steps", type=int, default=1)
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument(
        "--export-source-mode",
        type=str,
        choices=("auto", "raw_scenario_pkl", "scenarionet_waymo"),
        default="auto",
    )
    parser.add_argument("--waymo-raw-path", type=str, default=DEFAULT_WOD_131_TRAIN_PATH)
    parser.add_argument("--waymo-source-version", type=str, default="v1.2")
    parser.add_argument("--export-source-cache-dir", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _safe_name(value: Any) -> str:
    text = _normalize_track_id(value).strip()
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    cleaned = cleaned.strip("_")
    return cleaned or "unknown"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_default(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_default(v) for v in value]
    return value


def _read_index_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("rt", encoding="utf-8") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _select_unique_scene_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        scenario_id = str(row.get("scenario_id") or "").strip()
        if not scenario_id:
            continue
        grouped.setdefault(scenario_id, []).append(row)

    selected: List[Dict[str, Any]] = []
    for scenario_id in sorted(grouped):
        scene_rows = grouped[scenario_id]
        gt_row = next((row for row in scene_rows if str(row.get("selected_slot_id") or "") == "gt"), None)
        chosen = gt_row if gt_row is not None else scene_rows[0]
        selected.append(dict(chosen))
    return selected


def _resolve_scenario_pkl(row: Mapping[str, Any], *, scenario_root: Path | None) -> Path:
    raw_path = str(row.get("scenario_pkl") or "").strip()
    if raw_path:
        candidate = Path(raw_path).expanduser()
        if candidate.is_file():
            return candidate
    scenario_id = str(row.get("scenario_id") or "").strip()
    if not scenario_id:
        raise ValueError("Row is missing scenario_id and scenario_pkl could not be resolved.")
    if scenario_root is None:
        raise FileNotFoundError(
            f"Could not resolve scenario pickle for scene '{scenario_id}'. "
            "Provide --scenario-root or fix scenario_pkl paths in the index."
        )
    exact = scenario_root / f"sd_waymo_v1.3.1_{scenario_id}.pkl"
    if exact.is_file():
        return exact
    matches = sorted(scenario_root.glob(f"*{scenario_id}*.pkl"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find scenario pickle for scene '{scenario_id}' under {scenario_root}.")


def _first_valid_speed_mps(sample: Mapping[str, Any], *, slot: int) -> float:
    velocity = np.asarray(sample["decoder/agent_velocity"], dtype=np.float32)[:, slot, :]
    valid_mask = np.asarray(sample["decoder/agent_valid_mask"], dtype=bool)[:, slot]
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size == 0:
        return 0.0
    vel = velocity[int(valid_idx[0])]
    return float(np.linalg.norm(vel[:2]))


def _distance_at_first_valid(all_reference_world: Mapping[str, np.ndarray], *, track_id: str, sdc_id: str) -> float:
    target_xy = np.asarray(all_reference_world.get(str(track_id), np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    sdc_xy = np.asarray(all_reference_world.get(str(sdc_id), np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    if target_xy.shape[0] == 0 or sdc_xy.shape[0] == 0:
        return float("inf")
    return float(np.linalg.norm(target_xy[0] - sdc_xy[0]))


def _rank_adversary_candidates(
    *,
    base_sample: Mapping[str, Any],
    forward_summary: Any,
    all_reference_world: Mapping[str, np.ndarray],
    min_moving_speed_mps: float,
    max_distance_to_sdc_m: float,
    max_candidates: int,
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    sdc_id = str(forward_summary.sdc_id)
    for agent_summary in forward_summary.agents:
        track_id = str(agent_summary.raw_track_id)
        if not track_id or track_id == sdc_id:
            continue
        if not bool(agent_summary.receives_motion_loss):
            continue
        slot = int(agent_summary.model_agent_slot)
        speed_mps = _first_valid_speed_mps(base_sample, slot=slot)
        distance_to_sdc_m = _distance_at_first_valid(
            all_reference_world,
            track_id=track_id,
            sdc_id=sdc_id,
        )
        if speed_mps < float(min_moving_speed_mps):
            continue
        if distance_to_sdc_m > float(max_distance_to_sdc_m):
            continue
        ranked.append(
            {
                "agent_id": track_id,
                "slot": slot,
                "speed_mps": float(speed_mps),
                "distance_to_sdc_m": float(distance_to_sdc_m),
                "num_loss_steps": int(agent_summary.num_loss_steps),
            }
        )
    ranked.sort(key=lambda row: (row["distance_to_sdc_m"], -row["speed_mps"], row["agent_id"]))
    return ranked[: max(0, int(max_candidates))]


def _score_intervention(record: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(record["victim_min_distance_m"]),
        -float(record["effect"]["final_position_delta_m"]),
        -float(record["effect"]["num_changed_action_steps"]),
    )


def _evaluate_candidate_label(
    *,
    module: Any,
    tokenizer: Any,
    raw_scenario: Mapping[str, Any],
    base_sample: Mapping[str, Any],
    forward_summary: Any,
    candidate: Mapping[str, Any],
    semantic_label: str,
    semantic_confidence: float,
    time_window_mask: np.ndarray,
    map_center_world: np.ndarray,
    map_heading_world: float,
    baseline_output_np: Mapping[str, Any],
    all_reference_world: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    target_slot = int(candidate["slot"])
    decision_agent_mask = np.zeros((len(forward_summary.modeled_agent_ids),), dtype=np.float32)
    decision_agent_mask[target_slot] = 1.0
    controlled_sample = _build_control_sample(
        base_sample=base_sample,
        semantic_label=semantic_label,
        semantic_confidence=float(semantic_confidence),
        time_window_mask=time_window_mask,
        decision_agent_mask=decision_agent_mask,
    )
    controlled = _run_rollout(module, tokenizer, raw_sample=controlled_sample)
    baseline_world_xy = _extract_target_rollout_world_from_output_np(
        baseline_output_np,
        target_slot=target_slot,
        map_center_world=map_center_world,
        map_heading_world=map_heading_world,
    )
    controlled_world_xy = _extract_target_rollout_world_from_output_np(
        controlled["output_np"],
        target_slot=target_slot,
        map_center_world=map_center_world,
        map_heading_world=map_heading_world,
    )
    baseline_tokens = _extract_target_action_tokens_from_output_np(
        baseline_output_np,
        target_slot=target_slot,
    )
    controlled_tokens = _extract_target_action_tokens_from_output_np(
        controlled["output_np"],
        target_slot=target_slot,
    )
    compare_len = int(min(baseline_world_xy.shape[0], controlled_world_xy.shape[0]))
    rollout_delta = (
        np.linalg.norm(controlled_world_xy[:compare_len] - baseline_world_xy[:compare_len], axis=-1)
        if compare_len > 0
        else np.zeros((0,), dtype=np.float32)
    )
    sdc_reference_world_xy = np.asarray(
        all_reference_world.get(str(forward_summary.sdc_id), np.zeros((0, 2), dtype=np.float32)),
        dtype=np.float32,
    )
    victim_compare_len = int(min(controlled_world_xy.shape[0], sdc_reference_world_xy.shape[0]))
    victim_pairwise_distance = (
        np.linalg.norm(controlled_world_xy[:victim_compare_len] - sdc_reference_world_xy[:victim_compare_len], axis=-1)
        if victim_compare_len > 0
        else np.zeros((0,), dtype=np.float32)
    )
    changed_len = min(len(baseline_tokens), len(controlled_tokens))
    num_changed_action_steps = int(
        (baseline_tokens[:changed_len] != controlled_tokens[:changed_len]).sum()
    )
    return {
        "agent_id": str(candidate["agent_id"]),
        "target_agent_slot": int(target_slot),
        "semantic_label": str(semantic_label),
        "semantic_confidence": float(semantic_confidence),
        "candidate": dict(candidate),
        "victim_agent_id": str(forward_summary.sdc_id),
        "victim_selection_mode": "sdc",
        "victim_min_distance_m": float(victim_pairwise_distance.min()) if victim_pairwise_distance.size > 0 else float("inf"),
        "victim_min_distance_step": int(victim_pairwise_distance.argmin()) if victim_pairwise_distance.size > 0 else -1,
        "victim_mean_distance_m": float(victim_pairwise_distance.mean()) if victim_pairwise_distance.size > 0 else float("inf"),
        "baseline": {
            "num_points": int(baseline_world_xy.shape[0]),
            "path_length_m": _path_length_m(baseline_world_xy),
            "final_world_xy": baseline_world_xy[-1].tolist() if baseline_world_xy.shape[0] > 0 else None,
            "action_tokens": baseline_tokens.tolist(),
        },
        "controlled": {
            "num_points": int(controlled_world_xy.shape[0]),
            "path_length_m": _path_length_m(controlled_world_xy),
            "final_world_xy": controlled_world_xy[-1].tolist() if controlled_world_xy.shape[0] > 0 else None,
            "action_tokens": controlled_tokens.tolist(),
        },
        "effect": {
            "compare_len": int(compare_len),
            "final_position_delta_m": (
                float(np.linalg.norm(controlled_world_xy[compare_len - 1] - baseline_world_xy[compare_len - 1]))
                if compare_len > 0
                else 0.0
            ),
            "mean_position_delta_m": float(rollout_delta.mean()) if rollout_delta.size > 0 else 0.0,
            "max_position_delta_m": float(rollout_delta.max()) if rollout_delta.size > 0 else 0.0,
            "num_changed_action_steps": int(num_changed_action_steps),
        },
        "artifacts": {
            "controlled_output_np": controlled["output_np"],
            "controlled_world_xy": controlled_world_xy,
            "baseline_world_xy": baseline_world_xy,
            "sdc_reference_world_xy": sdc_reference_world_xy,
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_default(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")


def _should_materialize_scenarionet_sources(*, scene_rows: Sequence[Mapping[str, Any]], export_source_mode: str) -> bool:
    mode = str(export_source_mode).strip()
    if mode == "scenarionet_waymo":
        return True
    return False


def main() -> int:
    args = parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    control_index_path = Path(args.control_index).expanduser()
    if not control_index_path.is_file():
        raise FileNotFoundError(f"Control index not found: {control_index_path}")
    scenario_root = Path(args.scenario_root).expanduser() if str(args.scenario_root).strip() else None
    outdir = Path(args.outdir).expanduser()
    if outdir.exists() and any(outdir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists and is not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    labels = [normalize_semantic_label(label) for label in (args.semantic_labels or ["left", "right", "stop"])]
    labels = list(dict.fromkeys(labels))

    config_args = SimpleNamespace(
        config=args.config,
        ckpt=args.ckpt,
        teacher_ckpt=args.teacher_ckpt,
    )
    config = _load_config(config_args)
    model, load_report = _load_model(config=config, ckpt_path=args.ckpt, load_mode=args.load_mode)
    device = _resolve_device(args.device)
    model = model.to(device)
    module, tokenizer = _build_eval_module(
        config=config,
        ckpt_path=args.ckpt,
        device=device,
        save_path=outdir / "unused_eval_metrics",
        model=model,
    )
    if str(args.rollout_sampling_method).strip():
        module.config.SAMPLING.SAMPLING_METHOD = str(args.rollout_sampling_method)
    if float(args.rollout_temperature) > 0.0:
        module.config.SAMPLING.TEMPERATURE = float(args.rollout_temperature)
    if float(args.rollout_topp) > 0.0:
        module.config.SAMPLING.TOPP = float(args.rollout_topp)

    scene_rows = _select_unique_scene_rows(_read_index_rows(control_index_path))
    if args.scene_offset > 0:
        scene_rows = scene_rows[int(args.scene_offset) :]
    if args.num_scenes > 0:
        scene_rows = scene_rows[: int(args.num_scenes)]

    natural_dir = outdir / "natural_scenarios"
    adversarial_dir = outdir / "adversarial_scenarios"
    analysis_dir = outdir / "scene_analysis"
    natural_dir.mkdir(parents=True, exist_ok=True)
    adversarial_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    natural_paths: List[Path] = []
    adversarial_paths: List[Path] = []
    scene_results: List[Dict[str, Any]] = []
    skipped_scenes: List[Dict[str, Any]] = []

    export_source_by_scene: Dict[str, Path] = {}
    export_source_cache_dir = (
        Path(args.export_source_cache_dir).expanduser()
        if str(args.export_source_cache_dir).strip()
        else outdir / "_scenarionet_export_sources"
    )
    if _should_materialize_scenarionet_sources(
        scene_rows=scene_rows,
        export_source_mode=str(args.export_source_mode),
    ):
        export_source_by_scene = materialize_scenarionet_waymo_sources(
            scenario_ids=[str(row.get("scenario_id") or "").strip() for row in scene_rows],
            cache_root=export_source_cache_dir,
            waymo_raw_path=str(args.waymo_raw_path),
            version=str(args.waymo_source_version),
        )

    for scene_idx, row in enumerate(scene_rows):
        scenario_id = str(row.get("scenario_id") or "").strip()
        scene_safe = _safe_name(scenario_id)
        scene_dir = analysis_dir / scene_safe
        scene_dir.mkdir(parents=True, exist_ok=True)
        try:
            scenario_pkl = _resolve_scenario_pkl(row, scenario_root=scenario_root)
            export_source_pkl = export_source_by_scene.get(scenario_id, scenario_pkl)
            raw_scenario = load_raw_scenario(scenario_pkl)
            base_sample = preprocess_raw_scenario_for_forward_supervision(
                raw_scenario,
                config=config,
                in_evaluation=True,
            )
            base_sample["metadata/scenario_id"] = str(
                raw_scenario.get("id") or base_sample.get("metadata/scenario_id", scenario_id)
            )
            forward_summary = summarize_forward_supervision_for_sample(base_sample, raw_scenario=raw_scenario)
            map_center_world, map_heading_world = extract_model_frame(raw_scenario)
            all_reference_world = _extract_all_reference_world_from_sample(
                base_sample,
                map_center_world=map_center_world,
                map_heading_world=map_heading_world,
                modeled_agent_ids=forward_summary.modeled_agent_ids,
            )
            candidates = _rank_adversary_candidates(
                base_sample=base_sample,
                forward_summary=forward_summary,
                all_reference_world=all_reference_world,
                min_moving_speed_mps=float(args.min_moving_speed_mps),
                max_distance_to_sdc_m=float(args.max_distance_to_sdc_m),
                max_candidates=int(args.max_adversary_candidates),
            )
            if not candidates:
                skipped = {
                    "scene_index": int(scene_idx),
                    "scenario_id": scenario_id,
                    "scenario_pkl": str(scenario_pkl),
                    "export_source_pkl": str(export_source_pkl),
                    "reason": "no_candidate_adversaries",
                }
                skipped_scenes.append(skipped)
                _write_json(scene_dir / "scene_skip.json", skipped)
                continue

            baseline = _run_rollout(module, tokenizer, raw_sample=base_sample)
            horizon = int(np.asarray(base_sample["decoder/target_action_valid_mask"]).shape[0])
            time_window_mask = _build_time_window_mask(
                horizon=horizon,
                start_step=int(args.start_step),
                end_step=int(args.end_step),
            )

            candidate_records: List[Dict[str, Any]] = []
            best_record: Dict[str, Any] | None = None
            for candidate in candidates:
                for label in labels:
                    record = _evaluate_candidate_label(
                        module=module,
                        tokenizer=tokenizer,
                        raw_scenario=raw_scenario,
                        base_sample=base_sample,
                        forward_summary=forward_summary,
                        candidate=candidate,
                        semantic_label=label,
                        semantic_confidence=float(args.semantic_confidence),
                        time_window_mask=time_window_mask,
                        map_center_world=np.asarray(map_center_world, dtype=np.float32),
                        map_heading_world=float(map_heading_world),
                        baseline_output_np=baseline["output_np"],
                        all_reference_world=all_reference_world,
                    )
                    candidate_records.append(record)
                    passes = (
                        record["effect"]["final_position_delta_m"] >= float(args.min_final_position_delta_m)
                        and record["effect"]["num_changed_action_steps"] >= int(args.min_changed_action_steps)
                        and math.isfinite(float(record["victim_min_distance_m"]))
                    )
                    if not passes:
                        continue
                    if best_record is None or _score_intervention(record) < _score_intervention(best_record):
                        best_record = record

            scene_summary = {
                "scene_index": int(scene_idx),
                "scenario_id": scenario_id,
                "scenario_pkl": str(scenario_pkl),
                "export_source_pkl": str(export_source_pkl),
                "sdc_id": str(forward_summary.sdc_id),
                "candidate_adversaries": [
                    {
                        "agent_id": str(candidate["agent_id"]),
                        "slot": int(candidate["slot"]),
                        "speed_mps": float(candidate["speed_mps"]),
                        "distance_to_sdc_m": float(candidate["distance_to_sdc_m"]),
                        "num_loss_steps": int(candidate["num_loss_steps"]),
                    }
                    for candidate in candidates
                ],
                "evaluated_interventions": [
                    {
                        "agent_id": str(record["agent_id"]),
                        "semantic_label": str(record["semantic_label"]),
                        "victim_min_distance_m": float(record["victim_min_distance_m"]),
                        "victim_min_distance_step": int(record["victim_min_distance_step"]),
                        "effect": dict(record["effect"]),
                    }
                    for record in sorted(candidate_records, key=_score_intervention)
                ],
            }

            if best_record is None:
                scene_summary["selected_intervention"] = None
                scene_summary["reason"] = "no_intervention_passed_filters"
                skipped_scenes.append(scene_summary)
                _write_json(scene_dir / "scene_summary.json", scene_summary)
                continue

            selected_agent_id = str(best_record["agent_id"])
            selected_label = str(best_record["semantic_label"])
            victim_id = str(forward_summary.sdc_id)
            natural_path = export_victim_centric_ground_truth_scenario(
                raw_scenario,
                natural_dir / f"sd_counterfactual_1.0_{scene_safe}_ground_truth.pkl",
                victim_track_id=victim_id,
                adversary_track_id=selected_agent_id,
                original_file_path=export_source_pkl,
                intervention_name=f"{selected_label}_ground_truth",
            )
            adversarial_path = export_victim_centric_scenario(
                raw_scenario,
                adversarial_dir / (
                    f"sd_counterfactual_1.0_{scene_safe}_victim_{_safe_name(victim_id)}_adv_{_safe_name(selected_agent_id)}_{_safe_name(selected_label)}.pkl"
                ),
                victim_track_id=victim_id,
                adversary_track_id=selected_agent_id,
                adversary_trajectory_world_xy=np.asarray(best_record["artifacts"]["controlled_world_xy"], dtype=np.float32),
                intervention_name=f"{selected_label}_semantic_probe",
                original_file_path=export_source_pkl,
            )
            natural_paths.append(natural_path)
            adversarial_paths.append(adversarial_path)

            if args.save_plots:
                _save_overlay_plot(
                    out_path=scene_dir / "overlay.png",
                    raw_scenario=raw_scenario,
                    reference_world_xy=np.asarray(
                        all_reference_world.get(selected_agent_id, np.zeros((0, 2), dtype=np.float32)),
                        dtype=np.float32,
                    ),
                    baseline_world_xy=np.asarray(best_record["artifacts"]["baseline_world_xy"], dtype=np.float32),
                    controlled_world_xy=np.asarray(best_record["artifacts"]["controlled_world_xy"], dtype=np.float32),
                    scenario_id=scenario_id,
                    agent_id=selected_agent_id,
                    semantic_label=selected_label,
                )
                _save_victim_centric_overlay_plot(
                    out_path=scene_dir / "victim_centric_overlay.png",
                    raw_scenario=raw_scenario,
                    all_reference_world=all_reference_world,
                    victim_reference_world_xy=np.asarray(best_record["artifacts"]["sdc_reference_world_xy"], dtype=np.float32),
                    adversary_reference_world_xy=np.asarray(
                        all_reference_world.get(selected_agent_id, np.zeros((0, 2), dtype=np.float32)),
                        dtype=np.float32,
                    ),
                    adversary_baseline_world_xy=np.asarray(best_record["artifacts"]["baseline_world_xy"], dtype=np.float32),
                    adversary_controlled_world_xy=np.asarray(best_record["artifacts"]["controlled_world_xy"], dtype=np.float32),
                    victim_agent_id=victim_id,
                    adversary_agent_id=selected_agent_id,
                    semantic_label=selected_label,
                    scenario_id=scenario_id,
                )

            scene_result = {
                **scene_summary,
                "selected_intervention": {
                    "agent_id": selected_agent_id,
                    "semantic_label": selected_label,
                    "victim_agent_id": victim_id,
                    "victim_min_distance_m": float(best_record["victim_min_distance_m"]),
                    "victim_min_distance_step": int(best_record["victim_min_distance_step"]),
                    "effect": dict(best_record["effect"]),
                    "natural_scenario_pkl": str(natural_path),
                    "adversarial_scenario_pkl": str(adversarial_path),
                },
            }
            scene_results.append(scene_result)
            _write_json(scene_dir / "scene_summary.json", scene_result)
        except Exception as exc:
            skipped = {
                "scene_index": int(scene_idx),
                "scenario_id": scenario_id,
                "reason": "exception",
                "error": repr(exc),
            }
            skipped_scenes.append(skipped)
            _write_json(scene_dir / "scene_skip.json", skipped)

    natural_summary_path = create_dataset_summary(natural_paths, natural_dir) if natural_paths else None
    adversarial_summary_path = create_dataset_summary(adversarial_paths, adversarial_dir) if adversarial_paths else None

    builder_summary = {
        "split_name": str(args.split_name),
        "control_index": str(control_index_path),
        "scenario_root": None if scenario_root is None else str(scenario_root),
        "ckpt": str(Path(args.ckpt).expanduser()),
        "config": str(Path(args.config).expanduser()),
        "device": str(device),
        "seed": int(args.seed),
        "labels": labels,
        "num_requested_scenes": int(args.num_scenes),
        "scene_offset": int(args.scene_offset),
        "num_processed_scenes": int(len(scene_rows)),
        "num_exported_pairs": int(len(scene_results)),
        "num_skipped_scenes": int(len(skipped_scenes)),
        "natural_dir": str(natural_dir),
        "adversarial_dir": str(adversarial_dir),
        "export_source_mode": str(args.export_source_mode),
        "waymo_raw_path": str(args.waymo_raw_path),
        "waymo_source_version": str(args.waymo_source_version),
        "export_source_cache_dir": str(export_source_cache_dir),
        "natural_dataset_summary_pkl": None if natural_summary_path is None else str(natural_summary_path),
        "adversarial_dataset_summary_pkl": None if adversarial_summary_path is None else str(adversarial_summary_path),
        "checkpoint_load_report": load_report,
    }
    _write_json(outdir / "builder_summary.json", builder_summary)
    _write_json(outdir / "dataset_manifest.json", {"scenes": scene_results, "skipped_scenes": skipped_scenes})

    print(json.dumps(_json_default(builder_summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
