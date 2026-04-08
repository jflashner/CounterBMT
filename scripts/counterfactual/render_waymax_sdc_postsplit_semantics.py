from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    for path in (repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from scripts.counterfactual.label_waymax_sdc_path_semantics import (
    ALT_COLORS,
    CONTEXT_SELECTION_RADIUS_M,
    GT_COLOR,
    _finite_xy_rows,  # type: ignore[attr-defined]
    _path_to_rows,  # type: ignore[attr-defined]
    _render_single_image,  # type: ignore[attr-defined]
    _select_map_context,  # type: ignore[attr-defined]
    _select_nearby_agents,  # type: ignore[attr-defined]
    _select_traffic_lights,  # type: ignore[attr-defined]
    write_json,
    write_jsonl,
)
from scripts.counterfactual.plot_waymax_sdc_path_grids import _plot_scene_grid  # type: ignore[attr-defined]
from bmt.counterfactual.sdc_path_control import (
    ResampledLocalPath,
    compute_path_separability_profile,
    polyline_arc_lengths,
    polyline_headings,
    polyline_length_m,
    resample_polyline_xy,
    split_polyline_on_discontinuities,
    trim_polyline_from_point,
    world_to_sdc_up_frame,
)
from bmt.counterfactual.vlm_semantics.sdc_path_contract import SLOT_IDS, sdc_path_semantic_json_schema
from bmt.counterfactual.vlm_semantics.sdc_path_prompt import build_single_sdc_path_postsplit_prompt
from bmt.counterfactual.waymax_adapter import (
    raw_scenario_from_waymax_state,
    resolve_waymax_config,
    waymax_available,
)
from waymax.dataloader import womd_dataloader

DEFAULT_WOD_131_TRAIN_PATH = "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/training/training_tfexample.tfrecord-00000-of-01000"
DEFAULT_RESAMPLE_SPACING_M = 2.0
DEFAULT_SEPARABILITY_SCALE_M = 6.0
DEFAULT_SEPARABILITY_HEADING_WEIGHT_M = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render split-aware SDC path semantics images without calling the VLM.")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--path", type=str, default=DEFAULT_WOD_131_TRAIN_PATH)
    parser.add_argument("--config-name", type=str, default="WOD_1_3_1_TRAINING")
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--candidate-scenes", type=int, default=100)
    parser.add_argument("--num-selected-scenes", type=int, default=20)
    parser.add_argument("--current-time-index", type=int, default=-1)
    parser.add_argument("--num-paths", type=int, default=45)
    parser.add_argument("--num-points-per-path", type=int, default=800)
    parser.add_argument("--min-route-length-m", type=float, default=15.0)
    parser.add_argument("--min-gt-length-m", type=float, default=10.0)
    parser.add_argument("--gt-relative-threshold-m", type=float, default=10.0)
    parser.add_argument("--alt-diversity-weight", type=float, default=1.0)
    parser.add_argument("--include-off-route-paths", action="store_true")
    parser.add_argument("--diversity-top-k", type=int, default=0)
    parser.add_argument("--gradient-display-reference", type=float, default=0.30)
    parser.add_argument("--gradient-display-gamma", type=float, default=0.80)
    parser.add_argument("--save-scene-grid", action="store_true")
    parser.add_argument("--scene-grid-columns", type=int, default=4)
    parser.add_argument("--scene-grid-padding-m", type=float, default=18.0)
    parser.add_argument("--resample-spacing-m", type=float, default=DEFAULT_RESAMPLE_SPACING_M)
    parser.add_argument("--separability-scale-m", type=float, default=DEFAULT_SEPARABILITY_SCALE_M)
    parser.add_argument("--separability-heading-weight-m", type=float, default=DEFAULT_SEPARABILITY_HEADING_WEIGHT_M)
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--image-detail", type=str, default="original", choices=("low", "high", "original", "auto"))
    parser.add_argument("--save-pkls", action="store_true")
    parser.add_argument("--render-workers", type=int, default=1)
    return parser.parse_args()


def _resampled_local_path_from_world_segments(
    segments_xy_world: Sequence[np.ndarray],
    *,
    center_xy_world: np.ndarray,
    origin_heading_world: float,
    spacing_m: float,
) -> Tuple[ResampledLocalPath, List[np.ndarray], List[np.ndarray]]:
    local_segments: List[np.ndarray] = []
    world_resampled_segments: List[np.ndarray] = []
    heading_chunks: List[np.ndarray] = []
    arc_chunks: List[np.ndarray] = []
    arc_offset = 0.0
    for segment_world in segments_xy_world:
        segment_world_xy = _finite_xy_rows(np.asarray(segment_world, dtype=np.float32))
        if segment_world_xy.shape[0] < 2:
            continue
        segment_world_resampled = resample_polyline_xy(segment_world_xy, spacing_m=float(spacing_m))
        segment_local = world_to_sdc_up_frame(
            segment_world_resampled,
            origin_xy_world=np.asarray(center_xy_world, dtype=np.float32),
            origin_heading_world=float(origin_heading_world),
        )
        if segment_local.shape[0] < 2:
            continue
        world_resampled_segments.append(np.asarray(segment_world_resampled, dtype=np.float32))
        local_segments.append(np.asarray(segment_local, dtype=np.float32))
        seg_headings = polyline_headings(segment_local).astype(np.float32)
        seg_arc = polyline_arc_lengths(segment_local).astype(np.float32) + np.float32(arc_offset)
        heading_chunks.append(seg_headings)
        arc_chunks.append(seg_arc)
        arc_offset = float(seg_arc[-1]) if seg_arc.size > 0 else float(arc_offset)

    if not local_segments:
        empty = ResampledLocalPath(
            waypoints_xy=np.zeros((0, 2), dtype=np.float32),
            headings=np.zeros((0,), dtype=np.float32),
            arc_lengths_m=np.zeros((0,), dtype=np.float32),
        )
        return empty, [], []

    waypoints_xy = np.concatenate(local_segments, axis=0).astype(np.float32)
    headings = np.concatenate(heading_chunks, axis=0).astype(np.float32)
    arc_lengths_m = np.concatenate(arc_chunks, axis=0).astype(np.float32)
    return (
        ResampledLocalPath(waypoints_xy=waypoints_xy, headings=headings, arc_lengths_m=arc_lengths_m),
        local_segments,
        world_resampled_segments,
    )


def _integrated_separability_score(separability: np.ndarray, arc_lengths_m: np.ndarray, *, gt_length_m: float) -> float:
    sep = np.asarray(separability, dtype=np.float32).reshape(-1)
    arc = np.asarray(arc_lengths_m, dtype=np.float32).reshape(-1)
    if sep.size == 0 or arc.size == 0:
        return 0.0
    if sep.size == 1 or arc.size == 1:
        total = float(sep[0])
    else:
        total = float(np.trapezoid(sep, arc))
    return float(total / max(float(gt_length_m), 1e-3))


def _trim_and_split_world_path(points_xy_world: Any, *, current_xy_world: np.ndarray) -> List[np.ndarray]:
    trimmed = trim_polyline_from_point(points_xy_world, current_xy_world, prepend_point=True)
    return [
        np.asarray(segment, dtype=np.float32)
        for segment in split_polyline_on_discontinuities(trimmed)
        if np.asarray(segment).shape[0] >= 2
    ]


def _display_gradient_values(values: np.ndarray, *, reference: float, gamma: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    ref = max(float(reference), 1e-3)
    gam = max(float(gamma), 1e-3)
    scaled = np.clip(arr / ref, 0.0, 1.0)
    return np.power(scaled, gam, dtype=np.float32)


def _symmetric_pairwise_diversity_score(
    path_a: ResampledLocalPath,
    path_b: ResampledLocalPath,
    *,
    scale_m: float,
    heading_weight_m: float,
) -> float:
    if path_a.waypoints_xy.shape[0] < 2 or path_b.waypoints_xy.shape[0] < 2:
        return 0.0
    sep_a = compute_path_separability_profile(
        path_a,
        {"other": path_b},
        scale_m=float(scale_m),
        heading_weight_m=float(heading_weight_m),
    )
    score_a = _integrated_separability_score(
        np.asarray(sep_a["separability"], dtype=np.float32),
        np.asarray(path_a.arc_lengths_m, dtype=np.float32),
        gt_length_m=float(path_a.arc_lengths_m[-1]) if path_a.arc_lengths_m.size > 0 else 1.0,
    )
    sep_b = compute_path_separability_profile(
        path_b,
        {"other": path_a},
        scale_m=float(scale_m),
        heading_weight_m=float(heading_weight_m),
    )
    score_b = _integrated_separability_score(
        np.asarray(sep_b["separability"], dtype=np.float32),
        np.asarray(path_b.arc_lengths_m, dtype=np.float32),
        gt_length_m=float(path_b.arc_lengths_m[-1]) if path_b.arc_lengths_m.size > 0 else 1.0,
    )
    return float(0.5 * (score_a + score_b))


def _select_top_separable_alternates(
    *,
    raw_scenario: Mapping[str, Any],
    sdc_id: str,
    current_time_index: int,
    gt_future_xy: np.ndarray,
    path_rows: Sequence[Mapping[str, Any]],
    spacing_m: float,
    separability_scale_m: float,
    separability_heading_weight_m: float,
    gt_relative_threshold_m: float,
    alt_diversity_weight: float,
    include_off_route_paths: bool,
    diversity_top_k: int,
) -> Optional[Dict[str, Any]]:
    track_state = dict(dict(raw_scenario["tracks"]).get(str(sdc_id), {}).get("state", {}))
    current_position = np.asarray(track_state.get("position", []), dtype=np.float32)
    current_heading_seq = np.asarray(track_state.get("heading", []), dtype=np.float32).reshape(-1)
    valid = np.asarray(track_state.get("valid", []), dtype=bool).reshape(-1)
    idx = int(np.clip(int(current_time_index), 0, max(0, current_position.shape[0] - 1)))
    while idx > 0 and valid.shape[0] > idx and not bool(valid[idx]):
        idx -= 1
    current_xy = _finite_xy_rows(current_position[idx])[0]
    current_heading = float(current_heading_seq[idx]) if current_heading_seq.shape[0] > idx and np.isfinite(current_heading_seq[idx]) else 0.0

    gt_segments_world = [np.asarray(gt_future_xy, dtype=np.float32)]
    gt_local_path, gt_local_segments, gt_world_resampled_segments = _resampled_local_path_from_world_segments(
        gt_segments_world,
        center_xy_world=current_xy,
        origin_heading_world=current_heading,
        spacing_m=float(spacing_m),
    )
    gt_length_m = float(gt_local_path.arc_lengths_m[-1]) if gt_local_path.arc_lengths_m.size > 0 else 0.0
    if gt_local_path.waypoints_xy.shape[0] < 2 or gt_length_m < 1.0:
        return None

    scored_rows: List[Dict[str, Any]] = []
    for path_row in path_rows:
        if (not bool(include_off_route_paths)) and (not bool(path_row.get("on_route", False))):
            continue
        path_id = str(path_row.get("path_id") or "")
        trimmed_segments_world = _trim_and_split_world_path(
            path_row.get("polyline_xy", []),
            current_xy_world=current_xy,
        )
        alt_local_path, alt_local_segments, alt_world_resampled_segments = _resampled_local_path_from_world_segments(
            trimmed_segments_world,
            center_xy_world=current_xy,
            origin_heading_world=current_heading,
            spacing_m=float(spacing_m),
        )
        if alt_local_path.waypoints_xy.shape[0] < 2:
            continue
        alt_sep = compute_path_separability_profile(
            alt_local_path,
            {"gt": gt_local_path},
            scale_m=float(separability_scale_m),
            heading_weight_m=float(separability_heading_weight_m),
        )
        alt_relative_score = _integrated_separability_score(
            np.asarray(alt_sep["separability"], dtype=np.float32),
            np.asarray(alt_local_path.arc_lengths_m, dtype=np.float32),
            gt_length_m=gt_length_m,
        )
        gt_vs_alt_sep = compute_path_separability_profile(
            gt_local_path,
            {path_id: alt_local_path},
            scale_m=float(separability_scale_m),
            heading_weight_m=float(separability_heading_weight_m),
        )
        gt_relative_score = _integrated_separability_score(
            np.asarray(gt_vs_alt_sep["separability"], dtype=np.float32),
            np.asarray(gt_local_path.arc_lengths_m, dtype=np.float32),
            gt_length_m=gt_length_m,
        )
        use_gt_relative = bool(gt_length_m >= float(gt_relative_threshold_m))
        selection_score = float(gt_relative_score if use_gt_relative else alt_relative_score)
        scored_rows.append(
            {
                "path_row": dict(path_row),
                "path_id": path_id,
                "local_path": alt_local_path,
                "local_segments": [np.asarray(seg, dtype=np.float32) for seg in alt_local_segments],
                "world_resampled_segments": [np.asarray(seg, dtype=np.float32) for seg in alt_world_resampled_segments],
                "separability": np.asarray(alt_sep["separability"], dtype=np.float32),
                "min_distance_m": np.asarray(alt_sep["min_distance_m"], dtype=np.float32),
                "nearest_competing_path_id": list(alt_sep["nearest_competing_path_id"]),
                "gt_vs_alt_separability": np.asarray(gt_vs_alt_sep["separability"], dtype=np.float32),
                "gt_vs_alt_min_distance_m": np.asarray(gt_vs_alt_sep["min_distance_m"], dtype=np.float32),
                "alt_relative_score": float(alt_relative_score),
                "gt_relative_score": float(gt_relative_score),
                "score_kind": ("gt_relative" if use_gt_relative else "alt_relative"),
                "score": selection_score,
                "trimmed_route_length_m": float(polyline_length_m(alt_local_path.waypoints_xy)),
            }
        )

    if len(scored_rows) < 3:
        return None
    scored_rows = sorted(scored_rows, key=lambda row: (-float(row["score"]), str(row["path_id"])))
    shortlist_top_k = max(0, int(diversity_top_k))
    if shortlist_top_k > 0:
        scored_rows = scored_rows[: max(3, shortlist_top_k)]
    pairwise_diversity: Dict[Tuple[int, int], float] = {}
    for left_idx, right_idx in itertools.combinations(range(len(scored_rows)), 2):
        pairwise_diversity[(left_idx, right_idx)] = _symmetric_pairwise_diversity_score(
            scored_rows[left_idx]["local_path"],
            scored_rows[right_idx]["local_path"],
            scale_m=float(separability_scale_m),
            heading_weight_m=float(separability_heading_weight_m),
        )

    best_combo: Optional[Tuple[int, int, int]] = None
    best_combo_score = -float("inf")
    best_combo_gt_component = -float("inf")
    best_combo_diversity_component = -float("inf")
    for combo in itertools.combinations(range(len(scored_rows)), 3):
        gt_component = float(sum(float(scored_rows[idx]["score"]) for idx in combo))
        diversity_component = float(
            sum(float(pairwise_diversity.get((min(i, j), max(i, j)), 0.0)) for i, j in itertools.combinations(combo, 2))
        )
        combo_score = gt_component + (float(alt_diversity_weight) * diversity_component)
        if (
            combo_score > best_combo_score
            or (
                np.isclose(combo_score, best_combo_score)
                and (gt_component > best_combo_gt_component or diversity_component > best_combo_diversity_component)
            )
        ):
            best_combo = tuple(int(idx) for idx in combo)
            best_combo_score = float(combo_score)
            best_combo_gt_component = float(gt_component)
            best_combo_diversity_component = float(diversity_component)

    if best_combo is None:
        return None
    top_three = [scored_rows[idx] for idx in best_combo]
    gt_sep = compute_path_separability_profile(
        gt_local_path,
        {str(row["path_id"]): row["local_path"] for row in scored_rows},
        scale_m=float(separability_scale_m),
        heading_weight_m=float(separability_heading_weight_m),
    )
    return {
        "gt_current_xy_world": current_xy.astype(np.float32),
        "gt_current_heading_world": float(current_heading),
        "gt_local_path": gt_local_path,
        "gt_local_segments": gt_local_segments,
        "gt_world_resampled_segments": gt_world_resampled_segments,
        "gt_separability": np.asarray(gt_sep["separability"], dtype=np.float32),
        "gt_min_distance_m": np.asarray(gt_sep["min_distance_m"], dtype=np.float32),
        "gt_nearest_competing_path_id": list(gt_sep["nearest_competing_path_id"]),
        "gt_length_m": float(gt_length_m),
        "selected_alternates": top_three,
        "scene_score": float(best_combo_score),
        "scene_gt_component": float(best_combo_gt_component),
        "scene_alt_diversity_component": float(best_combo_diversity_component),
        "scene_score_kind": ("gt_relative" if gt_length_m >= float(gt_relative_threshold_m) else "alt_relative"),
    }


def _build_payload_with_postsplit_gradients(
    *,
    raw_scenario: Mapping[str, Any],
    example_id: str,
    scenario_id: str,
    sdc_id: str,
    current_time_index: int,
    gt_future_xy: np.ndarray,
    gt_past_xy: np.ndarray,
    selected_bundle: Mapping[str, Any],
    image_dir: Path,
    gradient_display_reference: float,
    gradient_display_gamma: float,
) -> Dict[str, Any]:
    def _concat_segments(segments: Sequence[np.ndarray]) -> np.ndarray:
        kept = [np.asarray(segment, dtype=np.float64) for segment in segments if np.asarray(segment).shape[0] >= 2]
        if not kept:
            return np.zeros((0, 2), dtype=np.float64)
        return np.concatenate(kept, axis=0).astype(np.float64)

    current_state = dict(raw_scenario["tracks"][str(sdc_id)]["state"])
    current_position = np.asarray(current_state["position"], dtype=np.float64)
    current_heading_seq = np.asarray(current_state["heading"], dtype=np.float64)
    current_valid = np.asarray(current_state["valid"], dtype=bool)
    idx = int(np.clip(int(current_time_index), 0, current_position.shape[0] - 1))
    while idx > 0 and not bool(current_valid[idx]):
        idx -= 1
    current_xy = _finite_xy_rows(current_position[idx])[0]
    current_heading = float(current_heading_seq[idx]) if idx < current_heading_seq.shape[0] and np.isfinite(current_heading_seq[idx]) else 0.0
    map_context = _select_map_context(raw_scenario, center_xy=current_xy, radius_m=CONTEXT_SELECTION_RADIUS_M)
    traffic_lights = _select_traffic_lights(raw_scenario, center_xy=current_xy, radius_m=CONTEXT_SELECTION_RADIUS_M, time_index=idx)
    nearby_agents = _select_nearby_agents(raw_scenario, sdc_id=sdc_id, center_xy=current_xy, current_idx=idx, radius_m=CONTEXT_SELECTION_RADIUS_M)

    slot_metadata = [
        {
            "slot_id": "gt",
            "source_kind": "ground_truth",
            "path_id": None,
            "on_route": True,
            "route_length_m": float(polyline_length_m(gt_future_xy.astype(np.float32))) if gt_future_xy.shape[0] >= 2 else 0.0,
            "separability_score": float(selected_bundle["scene_score"]),
        }
    ]
    images: Dict[str, str] = {}
    image_pixel_size: Dict[str, Dict[str, int]] = {}

    images["gt"] = str((image_dir / "gt.png").resolve())
    gt_render = _render_single_image(
        output_path=Path(images["gt"]),
        title=f"{example_id} | GT",
        sidebar_lines=[],
        center_xy=current_xy,
        current_xy=current_xy,
        current_heading=current_heading,
        gt_past_xy=gt_past_xy,
        highlighted_xy=_concat_segments(selected_bundle.get("gt_world_resampled_segments", [])),
        highlighted_segments_xy=[np.asarray(segment, dtype=np.float64) for segment in list(selected_bundle.get("gt_world_resampled_segments") or [])],
        highlighted_color=GT_COLOR,
        highlighted_label="GT",
        highlighted_gradient_values=_display_gradient_values(
            np.asarray(selected_bundle["gt_separability"], dtype=np.float32),
            reference=float(gradient_display_reference),
            gamma=float(gradient_display_gamma),
        ),
        gradient_label_low="shared",
        gradient_label_high="distinct",
        map_context=map_context,
        traffic_lights=traffic_lights,
        nearby_agents=nearby_agents,
    )
    image_pixel_size["gt"] = dict(gt_render["pixel_size"])

    selected_alternates = list(selected_bundle["selected_alternates"])
    for alt_index, alt_bundle in enumerate(selected_alternates, start=1):
        slot_id = f"alt_{alt_index}"
        path_row = dict(alt_bundle["path_row"])
        slot_metadata.append(
            {
                "slot_id": slot_id,
                "source_kind": "sdc_path",
                "path_id": str(path_row["path_id"]),
                "on_route": bool(path_row["on_route"]),
                "route_length_m": float(path_row["route_length_m"]),
                "trimmed_route_length_m": float(alt_bundle.get("trimmed_route_length_m") or 0.0),
                "separability_score": float(alt_bundle["score"]),
                "score_kind": str(alt_bundle.get("score_kind") or "unknown"),
                "alt_relative_score": float(alt_bundle.get("alt_relative_score") or 0.0),
                "gt_relative_score": float(alt_bundle.get("gt_relative_score") or 0.0),
            }
        )
        images[slot_id] = str((image_dir / f"{slot_id}.png").resolve())
        render_info = _render_single_image(
            output_path=Path(images[slot_id]),
            title=f"{example_id} | {slot_id}",
            sidebar_lines=[],
            center_xy=current_xy,
            current_xy=current_xy,
            current_heading=current_heading,
            gt_past_xy=gt_past_xy,
            highlighted_xy=_concat_segments(list(alt_bundle.get("world_resampled_segments") or [])),
            highlighted_segments_xy=[np.asarray(segment, dtype=np.float64) for segment in list(alt_bundle.get("world_resampled_segments") or [])],
            highlighted_color=ALT_COLORS[(alt_index - 1) % len(ALT_COLORS)],
            highlighted_label=slot_id.upper(),
            highlighted_gradient_values=_display_gradient_values(
                np.asarray(alt_bundle["separability"], dtype=np.float32),
                reference=float(gradient_display_reference),
                gamma=float(gradient_display_gamma),
            ),
            gradient_label_low="shared",
            gradient_label_high="distinct",
            map_context=map_context,
            traffic_lights=traffic_lights,
            nearby_agents=nearby_agents,
        )
        image_pixel_size[slot_id] = dict(render_info["pixel_size"])

    return {
        "example_id": str(example_id),
        "scenario_id": str(scenario_id),
        "sdc_id": str(sdc_id),
        "current_time_index": int(idx),
        "slot_metadata": slot_metadata,
        "images": images,
        "image_pixel_size": image_pixel_size,
        "scene_score": float(selected_bundle["scene_score"]),
        "scene_score_kind": str(selected_bundle.get("scene_score_kind") or "unknown"),
        "scene_gt_component": float(selected_bundle.get("scene_gt_component") or 0.0),
        "scene_alt_diversity_component": float(selected_bundle.get("scene_alt_diversity_component") or 0.0),
        "gt_length_m": float(selected_bundle["gt_length_m"]),
        "selected_alternates": [
            {
                "slot_id": f"alt_{alt_index}",
                "path_id": str(dict(bundle["path_row"]).get("path_id") or ""),
                "route_length_m": float(dict(bundle["path_row"]).get("route_length_m") or 0.0),
                "separability_score": float(bundle["score"]),
                "score_kind": str(bundle.get("score_kind") or "unknown"),
                "alt_relative_score": float(bundle.get("alt_relative_score") or 0.0),
                "gt_relative_score": float(bundle.get("gt_relative_score") or 0.0),
            }
            for alt_index, bundle in enumerate(selected_alternates, start=1)
        ],
        "gt_separability_summary": {
            "min": float(np.min(np.asarray(selected_bundle["gt_separability"], dtype=np.float32))) if np.asarray(selected_bundle["gt_separability"]).size > 0 else 0.0,
            "mean": float(np.mean(np.asarray(selected_bundle["gt_separability"], dtype=np.float32))) if np.asarray(selected_bundle["gt_separability"]).size > 0 else 0.0,
            "max": float(np.max(np.asarray(selected_bundle["gt_separability"], dtype=np.float32))) if np.asarray(selected_bundle["gt_separability"]).size > 0 else 0.0,
        },
    }


def _slot_request_row(
    *,
    payload: Mapping[str, Any],
    slot_row: Mapping[str, Any],
    example_dir: Path,
    model_name: str,
    image_detail: str,
) -> Dict[str, Any]:
    slot_id = str(slot_row.get("slot_id") or "")
    prompt_text = build_single_sdc_path_postsplit_prompt(payload, slot_row=slot_row)
    prompt_path = example_dir / f"prompt_{slot_id}.txt"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    request_payload = {
        "example_id": str(payload.get("example_id")),
        "slot_id": slot_id,
        "model": str(model_name),
        "image_detail": str(image_detail),
        "image_paths": [str(dict(payload.get("images", {}) or {}).get(slot_id) or "")],
        "prompt": prompt_text,
        "json_schema": sdc_path_semantic_json_schema(),
    }
    request_json_path = example_dir / f"request_{slot_id}.json"
    write_json(request_json_path, request_payload)
    return {
        "example_id": str(payload.get("example_id") or ""),
        "scenario_id": str(payload.get("scenario_id") or ""),
        "sdc_id": str(payload.get("sdc_id") or ""),
        "slot_id": slot_id,
        "prompt_path": str(prompt_path.resolve()),
        "request_json": str(request_json_path.resolve()),
        "image_paths": request_payload["image_paths"],
    }


def _render_selected_entry(
    *,
    row: Mapping[str, Any],
    rank_idx: int,
    outdir: Path,
    save_scene_grid: bool,
    scene_grid_padding_m: float,
    scene_grid_columns: int,
    save_pkls: bool,
    model_name: str,
    image_detail: str,
    gradient_display_reference: float,
    gradient_display_gamma: float,
) -> Dict[str, Any]:
    scenario_id = str(row["scenario_id"])
    sdc_id = str(row["sdc_id"])
    current_time_index = int(row["current_time_index"])
    example_id = f"{scenario_id}__sdc_{sdc_id}__t_{current_time_index:03d}"
    example_dir = outdir / "examples" / example_id
    image_dir = example_dir / "images"
    payload = _build_payload_with_postsplit_gradients(
        raw_scenario=row["raw_scenario"],
        example_id=example_id,
        scenario_id=scenario_id,
        sdc_id=sdc_id,
        current_time_index=current_time_index,
        gt_future_xy=np.asarray(row["gt_future_xy"], dtype=np.float32),
        gt_past_xy=np.asarray(row["gt_past_xy"], dtype=np.float32),
        selected_bundle=row["selected_bundle"],
        image_dir=image_dir,
        gradient_display_reference=float(gradient_display_reference),
        gradient_display_gamma=float(gradient_display_gamma),
    )
    payload["scene_index"] = int(row["scene_index"])
    payload["selection_rank"] = int(rank_idx)
    payload["selection_score"] = float(row["scene_score"])
    if bool(save_scene_grid):
        scene_grid_path = example_dir / "all_sdc_paths_grid.png"
        scene_grid_summary = _plot_scene_grid(
            row["raw_scenario"],
            out_path=scene_grid_path,
            padding_m=float(scene_grid_padding_m),
            columns=int(scene_grid_columns),
        )
        payload["all_sdc_paths_grid_png"] = str(scene_grid_path.resolve())
        payload["all_sdc_paths_grid_summary"] = scene_grid_summary
    write_json(example_dir / "render_metadata.json", payload)
    if bool(save_pkls):
        write_json(example_dir / "raw_scenario.json", row["raw_scenario"])

    prompt_manifest_rows: List[Dict[str, Any]] = []
    for slot_row in list(payload.get("slot_metadata") or []):
        slot_id = str(slot_row.get("slot_id") or "")
        if slot_id not in SLOT_IDS:
            continue
        prompt_manifest_rows.append(
            _slot_request_row(
                payload=payload,
                slot_row=slot_row,
                example_dir=example_dir,
                model_name=str(model_name),
                image_detail=str(image_detail),
            )
        )

    aggregate_row = {
        "example_id": example_id,
        "scenario_id": scenario_id,
        "sdc_id": sdc_id,
        "scene_index": int(row["scene_index"]),
        "current_time_index": int(current_time_index),
        "selection_rank": int(rank_idx),
        "selection_score": float(row["scene_score"]),
        "selection_score_kind": str(row["selected_bundle"].get("scene_score_kind") or "unknown"),
        "selection_gt_component": float(row["selected_bundle"].get("scene_gt_component") or 0.0),
        "selection_alt_diversity_component": float(row["selected_bundle"].get("scene_alt_diversity_component") or 0.0),
        "gt_length_m": float(row["gt_length_m"]),
        "selected_alt_path_ids": list(row["selected_alt_path_ids"]),
        "slot_metadata": payload["slot_metadata"],
        "images": payload["images"],
        "all_sdc_paths_grid_png": payload.get("all_sdc_paths_grid_png"),
        "all_sdc_paths_grid_summary": payload.get("all_sdc_paths_grid_summary"),
        "prompt_paths": {
            str(slot_row.get("slot_id") or ""): str((example_dir / f"prompt_{slot_row.get('slot_id')}.txt").resolve())
            for slot_row in list(payload.get("slot_metadata") or [])
            if str(slot_row.get("slot_id") or "") in SLOT_IDS
        },
        "request_jsons": {
            str(slot_row.get("slot_id") or ""): str((example_dir / f"request_{slot_row.get('slot_id')}.json").resolve())
            for slot_row in list(payload.get("slot_metadata") or [])
            if str(slot_row.get("slot_id") or "") in SLOT_IDS
        },
    }
    return {
        "render_payload": payload,
        "prompt_manifest_rows": prompt_manifest_rows,
        "aggregate_row": aggregate_row,
    }


def main() -> int:
    args = parse_args()
    if not waymax_available():
        raise SystemExit("waymax is not installed in this environment")

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    config = resolve_waymax_config(
        config_name=str(args.config_name),
        path=str(args.path),
        include_sdc_paths=True,
        num_paths=int(args.num_paths),
        num_points_per_path=int(args.num_points_per_path),
    )
    if dataclasses.is_dataclass(config):
        if hasattr(config, "num_shards"):
            config = dataclasses.replace(config, num_shards=1, deterministic=True)

    candidate_rows: List[Dict[str, Any]] = []
    scene_iter = itertools.islice(
        womd_dataloader.simulator_state_generator(config=config),
        int(args.scene_offset),
        int(args.scene_offset) + int(args.candidate_scenes),
    )

    for local_idx, state in enumerate(scene_iter, start=int(args.scene_offset)):
        fallback_scenario_id = f"waymax_scene_{local_idx:05d}"
        raw = raw_scenario_from_waymax_state(
            state,
            scenario_id=fallback_scenario_id,
            current_time_index=(None if int(args.current_time_index) < 0 else int(args.current_time_index)),
        )
        scenario_id = str(raw.get("id") or fallback_scenario_id)
        sdc_id = str(dict(raw.get("metadata", {})).get("sdc_id") or "")
        if not sdc_id or str(sdc_id) not in dict(raw.get("tracks", {})):
            continue
        current_time_index = int(dict(raw.get("metadata", {})).get("current_time_index") or 0)
        gt_future_xy, gt_past_xy, path_rows = _path_to_rows(
            raw,
            sdc_id=sdc_id,
            current_idx=current_time_index,
            min_route_length_m=float(args.min_route_length_m),
        )
        if gt_future_xy.shape[0] < 5:
            continue

        selected_bundle = _select_top_separable_alternates(
            raw_scenario=raw,
            sdc_id=sdc_id,
            current_time_index=current_time_index,
            gt_future_xy=gt_future_xy,
            path_rows=path_rows,
            spacing_m=float(args.resample_spacing_m),
            separability_scale_m=float(args.separability_scale_m),
            separability_heading_weight_m=float(args.separability_heading_weight_m),
            gt_relative_threshold_m=float(args.gt_relative_threshold_m),
            alt_diversity_weight=float(args.alt_diversity_weight),
            include_off_route_paths=bool(args.include_off_route_paths),
            diversity_top_k=int(args.diversity_top_k),
        )
        if selected_bundle is None:
            continue
        if float(selected_bundle["gt_length_m"]) < float(args.min_gt_length_m):
            continue

        candidate_rows.append(
            {
                "scene_index": int(local_idx),
                "scenario_id": scenario_id,
                "sdc_id": sdc_id,
                "current_time_index": int(current_time_index),
                "raw_scenario": raw,
                "gt_future_xy": np.asarray(gt_future_xy, dtype=np.float32),
                "gt_past_xy": np.asarray(gt_past_xy, dtype=np.float32),
                "selected_bundle": selected_bundle,
                "selected_alt_path_ids": [str(dict(bundle["path_row"]).get("path_id") or "") for bundle in selected_bundle["selected_alternates"]],
                "scene_score": float(selected_bundle["scene_score"]),
                "gt_length_m": float(selected_bundle["gt_length_m"]),
            }
        )

    ranked_rows = sorted(
        candidate_rows,
        key=lambda row: (-float(row["scene_score"]), int(row["scene_index"]), str(row["scenario_id"]), str(row["sdc_id"])),
    )
    selected_rows = ranked_rows[: int(args.num_selected_scenes)]

    render_rows: List[Dict[str, Any]] = []
    prompt_manifest_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []

    render_jobs = [(int(rank_idx), dict(row)) for rank_idx, row in enumerate(selected_rows)]
    render_results: List[Dict[str, Any]] = []
    if int(args.render_workers) > 1 and len(render_jobs) > 1:
        with ProcessPoolExecutor(max_workers=int(args.render_workers)) as executor:
            future_map = {
                executor.submit(
                    _render_selected_entry,
                    row=row,
                    rank_idx=rank_idx,
                    outdir=outdir,
                    save_scene_grid=bool(args.save_scene_grid),
                    scene_grid_padding_m=float(args.scene_grid_padding_m),
                    scene_grid_columns=int(args.scene_grid_columns),
                    save_pkls=bool(args.save_pkls),
                    model_name=str(args.model),
                    image_detail=str(args.image_detail),
                    gradient_display_reference=float(args.gradient_display_reference),
                    gradient_display_gamma=float(args.gradient_display_gamma),
                ): rank_idx
                for rank_idx, row in render_jobs
            }
            for future in as_completed(future_map):
                render_results.append(dict(future.result()))
    else:
        for rank_idx, row in render_jobs:
            render_results.append(
                _render_selected_entry(
                    row=row,
                    rank_idx=rank_idx,
                    outdir=outdir,
                    save_scene_grid=bool(args.save_scene_grid),
                    scene_grid_padding_m=float(args.scene_grid_padding_m),
                    scene_grid_columns=int(args.scene_grid_columns),
                    save_pkls=bool(args.save_pkls),
                    model_name=str(args.model),
                    image_detail=str(args.image_detail),
                    gradient_display_reference=float(args.gradient_display_reference),
                    gradient_display_gamma=float(args.gradient_display_gamma),
                )
            )

    render_results = sorted(
        render_results,
        key=lambda row: int(dict(row.get("render_payload") or {}).get("selection_rank") or 0),
    )
    for result in render_results:
        render_rows.append(dict(result["render_payload"]))
        prompt_manifest_rows.extend([dict(row) for row in list(result.get("prompt_manifest_rows") or [])])
        aggregate_rows.append(dict(result["aggregate_row"]))

    ranking_rows = []
    for row in ranked_rows:
        ranking_rows.append(
            {
                "scene_index": int(row["scene_index"]),
                "scenario_id": str(row["scenario_id"]),
                "sdc_id": str(row["sdc_id"]),
                "current_time_index": int(row["current_time_index"]),
                "selection_score": float(row["scene_score"]),
                "selection_score_kind": str(row["selected_bundle"].get("scene_score_kind") or "unknown"),
                "selection_gt_component": float(row["selected_bundle"].get("scene_gt_component") or 0.0),
                "selection_alt_diversity_component": float(row["selected_bundle"].get("scene_alt_diversity_component") or 0.0),
                "gt_length_m": float(row["gt_length_m"]),
                "selected_alt_path_ids": list(row["selected_alt_path_ids"]),
                "selected_alt_scores": [float(bundle["score"]) for bundle in row["selected_bundle"]["selected_alternates"]],
                "selected_alt_score_kinds": [str(bundle.get("score_kind") or "unknown") for bundle in row["selected_bundle"]["selected_alternates"]],
                "selected_alt_gt_relative_scores": [float(bundle.get("gt_relative_score") or 0.0) for bundle in row["selected_bundle"]["selected_alternates"]],
                "selected_alt_alt_relative_scores": [float(bundle.get("alt_relative_score") or 0.0) for bundle in row["selected_bundle"]["selected_alternates"]],
            }
        )

    render_manifest_path = outdir / "postsplit_render_manifest.json"
    write_json(
        render_manifest_path,
        {
            "path": str(args.path),
            "candidate_scenes_scanned": int(args.candidate_scenes),
            "scene_offset": int(args.scene_offset),
            "num_selected_scenes": int(len(render_rows)),
            "rows": render_rows,
        },
    )
    request_manifest_path = outdir / "postsplit_request_manifest.jsonl"
    write_jsonl(request_manifest_path, prompt_manifest_rows)
    selection_summary_path = outdir / "postsplit_scene_selection.json"
    write_json(
        selection_summary_path,
        {
            "path": str(args.path),
            "candidate_scenes_requested": int(args.candidate_scenes),
            "num_candidate_scenes_kept": int(len(candidate_rows)),
            "num_selected_scenes": int(len(selected_rows)),
            "selection_metric": "choose best 3-alt set by sum(base GT disagreement scores) + alt_diversity_weight * sum(pairwise alt-alt diversity); keep gradients relative to GT",
            "min_gt_length_m": float(args.min_gt_length_m),
            "gt_relative_threshold_m": float(args.gt_relative_threshold_m),
            "alt_diversity_weight": float(args.alt_diversity_weight),
            "diversity_top_k": int(args.diversity_top_k),
            "include_off_route_paths": bool(args.include_off_route_paths),
            "gradient_display_reference": float(args.gradient_display_reference),
            "gradient_display_gamma": float(args.gradient_display_gamma),
            "rows": ranking_rows,
        },
    )
    aggregate_index_path = outdir / "postsplit_selected_scene_index.jsonl"
    write_jsonl(aggregate_index_path, aggregate_rows)

    summary = {
        "path": str(args.path),
        "candidate_scenes_requested": int(args.candidate_scenes),
        "num_candidate_scenes_kept": int(len(candidate_rows)),
        "num_selected_scenes": int(len(selected_rows)),
        "model": str(args.model),
        "image_detail": str(args.image_detail),
        "min_gt_length_m": float(args.min_gt_length_m),
        "gt_relative_threshold_m": float(args.gt_relative_threshold_m),
        "alt_diversity_weight": float(args.alt_diversity_weight),
        "diversity_top_k": int(args.diversity_top_k),
        "include_off_route_paths": bool(args.include_off_route_paths),
        "gradient_display_reference": float(args.gradient_display_reference),
        "gradient_display_gamma": float(args.gradient_display_gamma),
        "render_workers": int(args.render_workers),
        "render_manifest_json": str(render_manifest_path.resolve()),
        "request_manifest_jsonl": str(request_manifest_path.resolve()),
        "selection_summary_json": str(selection_summary_path.resolve()),
        "selected_scene_index_jsonl": str(aggregate_index_path.resolve()),
    }
    write_json(outdir / "postsplit_render_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
