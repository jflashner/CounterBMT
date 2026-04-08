from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    for path in (repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from scripts.counterfactual.label_waymax_sdc_path_semantics import _path_to_rows, write_json, write_jsonl
from scripts.counterfactual.render_waymax_sdc_postsplit_semantics import (
    DEFAULT_SEPARABILITY_HEADING_WEIGHT_M,
    DEFAULT_SEPARABILITY_SCALE_M,
    DEFAULT_WOD_131_TRAIN_PATH,
    _build_payload_with_postsplit_gradients,
    _pairwise_distance_matrix,
    _resampled_local_path_from_world_segments,
    _single_competitor_separability_profile_from_distances,
    _slot_request_row,
    _trim_and_split_world_path,
)
from bmt.counterfactual.sdc_path_control import compute_path_separability_profile
from bmt.counterfactual.waymax_adapter import raw_scenario_from_waymax_state, resolve_waymax_config
from waymax.dataloader import womd_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerender an existing postsplit bundle from its selected-scene index.")
    parser.add_argument("--bundle-root", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="")
    parser.add_argument("--path", type=str, default=DEFAULT_WOD_131_TRAIN_PATH)
    parser.add_argument("--config-name", type=str, default="WOD_1_3_1_TRAINING")
    parser.add_argument("--current-time-index", type=int, default=-1)
    parser.add_argument("--num-paths", type=int, default=45)
    parser.add_argument("--num-points-per-path", type=int, default=800)
    parser.add_argument("--min-route-length-m", type=float, default=15.0)
    parser.add_argument("--resample-spacing-m", type=float, default=2.0)
    parser.add_argument("--separability-scale-m", type=float, default=DEFAULT_SEPARABILITY_SCALE_M)
    parser.add_argument("--separability-heading-weight-m", type=float, default=DEFAULT_SEPARABILITY_HEADING_WEIGHT_M)
    parser.add_argument("--gradient-display-reference", type=float, default=0.75)
    parser.add_argument("--gradient-display-gamma", type=float, default=1.10)
    parser.add_argument("--show-traffic-lights", action="store_true")
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--image-detail", type=str, default="original", choices=("low", "high", "original", "auto"))
    parser.add_argument("--scene-indices", type=str, default="")
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(dict(json.loads(line)))
    return rows


def _parse_scene_indices(text: str) -> Set[int]:
    out: Set[int] = set()
    raw = str(text or "").strip()
    if not raw:
        return out
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.add(int(chunk))
    return out


def _selection_row_to_bundle(
    *,
    raw_scenario: Mapping[str, Any],
    selection_row: Mapping[str, Any],
    gt_future_xy: np.ndarray,
    path_rows: Sequence[Mapping[str, Any]],
    spacing_m: float,
    separability_scale_m: float,
    separability_heading_weight_m: float,
    include_off_route_paths: bool,
) -> Dict[str, Any]:
    sdc_id = str(selection_row.get("sdc_id") or "")
    current_time_index = int(selection_row.get("current_time_index") or 0)
    track_state = dict(dict(raw_scenario["tracks"]).get(sdc_id, {}).get("state", {}))
    current_position = np.asarray(track_state.get("position", []), dtype=np.float32)
    current_heading_seq = np.asarray(track_state.get("heading", []), dtype=np.float32).reshape(-1)
    valid = np.asarray(track_state.get("valid", []), dtype=bool).reshape(-1)
    idx = int(np.clip(int(current_time_index), 0, max(0, current_position.shape[0] - 1)))
    while idx > 0 and valid.shape[0] > idx and not bool(valid[idx]):
        idx -= 1
    current_xy = np.asarray(current_position[idx, :2], dtype=np.float32)
    current_heading = float(current_heading_seq[idx]) if current_heading_seq.shape[0] > idx and np.isfinite(current_heading_seq[idx]) else 0.0

    gt_local_path, gt_local_segments, gt_world_resampled_segments = _resampled_local_path_from_world_segments(
        [np.asarray(gt_future_xy, dtype=np.float32)],
        center_xy_world=current_xy,
        origin_heading_world=current_heading,
        spacing_m=float(spacing_m),
    )
    gt_length_m = float(gt_local_path.arc_lengths_m[-1]) if gt_local_path.arc_lengths_m.size > 0 else 0.0

    path_row_by_id = {str(row.get("path_id") or ""): dict(row) for row in path_rows}
    slot_meta_by_path_id = {
        str(row.get("path_id") or ""): dict(row)
        for row in list(selection_row.get("slot_metadata") or [])
        if str(row.get("path_id") or "").strip()
    }
    selected_alt_path_ids = [str(path_id) for path_id in list(selection_row.get("selected_alt_path_ids") or []) if str(path_id).strip()]
    competitor_paths: Dict[str, Any] = {}

    selected_alternates: List[Dict[str, Any]] = []
    for path_row in path_rows:
        if (not bool(include_off_route_paths)) and (not bool(path_row.get("on_route", False))):
            continue
        path_id = str(path_row.get("path_id") or "")
        if not path_id:
            continue
        trimmed_segments_world = _trim_and_split_world_path(path_row.get("polyline_xy", []), current_xy_world=current_xy)
        alt_local_path, _, _ = _resampled_local_path_from_world_segments(
            trimmed_segments_world,
            center_xy_world=current_xy,
            origin_heading_world=current_heading,
            spacing_m=float(spacing_m),
        )
        if alt_local_path.waypoints_xy.shape[0] < 2:
            continue
        competitor_paths[path_id] = alt_local_path

    for path_id in selected_alt_path_ids:
        path_row = dict(path_row_by_id[path_id])
        trimmed_segments_world = _trim_and_split_world_path(path_row.get("polyline_xy", []), current_xy_world=current_xy)
        alt_local_path, alt_local_segments, alt_world_resampled_segments = _resampled_local_path_from_world_segments(
            trimmed_segments_world,
            center_xy_world=current_xy,
            origin_heading_world=current_heading,
            spacing_m=float(spacing_m),
        )
        alt_gt_distance = _pairwise_distance_matrix(alt_local_path.waypoints_xy, gt_local_path.waypoints_xy)
        alt_sep = _single_competitor_separability_profile_from_distances(
            alt_local_path,
            "gt",
            gt_local_path,
            alt_gt_distance,
            scale_m=float(separability_scale_m),
            heading_weight_m=float(separability_heading_weight_m),
        )
        gt_vs_alt_sep = _single_competitor_separability_profile_from_distances(
            gt_local_path,
            path_id,
            alt_local_path,
            alt_gt_distance.T,
            scale_m=float(separability_scale_m),
            heading_weight_m=float(separability_heading_weight_m),
        )
        slot_meta = dict(slot_meta_by_path_id.get(path_id) or {})
        selected_alternates.append(
            {
                "path_row": path_row,
                "path_id": path_id,
                "local_path": alt_local_path,
                "local_segments": [np.asarray(seg, dtype=np.float32) for seg in alt_local_segments],
                "world_resampled_segments": [np.asarray(seg, dtype=np.float32) for seg in alt_world_resampled_segments],
                "separability": np.asarray(alt_sep["separability"], dtype=np.float32),
                "min_distance_m": np.asarray(alt_sep["min_distance_m"], dtype=np.float32),
                "nearest_competing_path_id": list(alt_sep["nearest_competing_path_id"]),
                "gt_vs_alt_separability": np.asarray(gt_vs_alt_sep["separability"], dtype=np.float32),
                "gt_vs_alt_min_distance_m": np.asarray(gt_vs_alt_sep["min_distance_m"], dtype=np.float32),
                "alt_relative_score": float(slot_meta.get("alt_relative_score") or 0.0),
                "gt_relative_score": float(slot_meta.get("gt_relative_score") or 0.0),
                "score_kind": str(slot_meta.get("score_kind") or "unknown"),
                "score": float(slot_meta.get("separability_score") or 0.0),
                "trimmed_route_length_m": float(slot_meta.get("trimmed_route_length_m") or 0.0),
            }
        )

    gt_sep = compute_path_separability_profile(
        gt_local_path,
        competitor_paths,
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
        "gt_length_m": float(selection_row.get("gt_length_m") or gt_length_m),
        "selected_alternates": selected_alternates,
        "scene_score": float(selection_row.get("selection_score") or 0.0),
        "scene_gt_component": float(selection_row.get("selection_gt_component") or 0.0),
        "scene_alt_diversity_component": float(selection_row.get("selection_alt_diversity_component") or 0.0),
        "scene_score_kind": str(selection_row.get("selection_score_kind") or "unknown"),
    }


def _aggregate_row_from_payload(
    payload: Mapping[str, Any],
    selection_row: Mapping[str, Any],
    *,
    example_dir: Path,
) -> Dict[str, Any]:
    slot_ids = ["gt", "alt_1", "alt_2", "alt_3"]
    return {
        "example_id": str(payload.get("example_id") or ""),
        "scenario_id": str(payload.get("scenario_id") or ""),
        "sdc_id": str(payload.get("sdc_id") or ""),
        "scene_index": int(selection_row.get("scene_index") or 0),
        "current_time_index": int(payload.get("current_time_index") or 0),
        "selection_rank": int(selection_row.get("selection_rank") or 0),
        "selection_score": float(selection_row.get("selection_score") or 0.0),
        "selection_score_kind": str(selection_row.get("selection_score_kind") or "unknown"),
        "selection_gt_component": float(selection_row.get("selection_gt_component") or 0.0),
        "selection_alt_diversity_component": float(selection_row.get("selection_alt_diversity_component") or 0.0),
        "gt_length_m": float(selection_row.get("gt_length_m") or 0.0),
        "selected_alt_path_ids": list(selection_row.get("selected_alt_path_ids") or []),
        "slot_metadata": payload.get("slot_metadata"),
        "images": payload.get("images"),
        "all_sdc_paths_grid_png": payload.get("all_sdc_paths_grid_png"),
        "all_sdc_paths_grid_summary": payload.get("all_sdc_paths_grid_summary"),
        "prompt_paths": {
            slot_id: str((example_dir / f"prompt_{slot_id}.txt").resolve())
            for slot_id in slot_ids
            if (example_dir / f"prompt_{slot_id}.txt").is_file()
        },
        "request_jsons": {
            slot_id: str((example_dir / f"request_{slot_id}.json").resolve())
            for slot_id in slot_ids
            if (example_dir / f"request_{slot_id}.json").is_file()
        },
    }


def main() -> int:
    args = parse_args()
    bundle_root = Path(args.bundle_root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if str(args.outdir).strip() else bundle_root
    outdir.mkdir(parents=True, exist_ok=True)

    selection_index_path = bundle_root / "postsplit_selected_scene_index.jsonl"
    render_summary_path = bundle_root / "postsplit_render_summary.json"
    selection_summary_path = bundle_root / "postsplit_scene_selection.json"
    render_manifest_path = bundle_root / "postsplit_render_manifest.json"

    selection_rows = _read_jsonl(selection_index_path)
    if not selection_rows:
        raise RuntimeError(f"No rows found in {selection_index_path}")

    scene_filter = _parse_scene_indices(args.scene_indices)
    if scene_filter:
        selection_rows = [row for row in selection_rows if int(row.get("scene_index") or -1) in scene_filter]
    if int(args.limit_scenes) > 0:
        selection_rows = selection_rows[: int(args.limit_scenes)]
    if not selection_rows:
        raise RuntimeError("No selected scenes remain after filtering.")

    old_render_summary = _read_json(render_summary_path) if render_summary_path.is_file() else {}
    old_selection_summary = _read_json(selection_summary_path) if selection_summary_path.is_file() else {}
    old_render_manifest = _read_json(render_manifest_path) if render_manifest_path.is_file() else {}

    selection_by_scene_index = {int(row.get("scene_index") or 0): dict(row) for row in selection_rows}
    wanted_scene_indices = set(selection_by_scene_index.keys())
    max_scene_index = max(wanted_scene_indices)

    config = resolve_waymax_config(
        config_name=str(args.config_name),
        path=str(args.path),
        include_sdc_paths=True,
        num_paths=int(args.num_paths),
        num_points_per_path=int(args.num_points_per_path),
    )
    if dataclasses.is_dataclass(config) and hasattr(config, "num_shards"):
        config = dataclasses.replace(config, num_shards=1, deterministic=True)

    render_rows: List[Dict[str, Any]] = []
    prompt_manifest_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []

    processed = 0
    start_t = time.time()
    scene_iter = itertools.islice(womd_dataloader.simulator_state_generator(config=config), 0, max_scene_index + 1)
    for local_idx, state in enumerate(scene_iter):
        if local_idx not in wanted_scene_indices:
            continue
        selection_row = dict(selection_by_scene_index[local_idx])
        fallback_scenario_id = f"waymax_scene_{local_idx:05d}"
        raw = raw_scenario_from_waymax_state(
            state,
            scenario_id=fallback_scenario_id,
            current_time_index=(None if int(args.current_time_index) < 0 else int(args.current_time_index)),
        )
        scenario_id = str(raw.get("id") or fallback_scenario_id)
        sdc_id = str(dict(raw.get("metadata", {})).get("sdc_id") or "")
        current_time_index = int(dict(raw.get("metadata", {})).get("current_time_index") or 0)
        gt_future_xy, gt_past_xy, path_rows = _path_to_rows(
            raw,
            sdc_id=sdc_id,
            current_idx=current_time_index,
            min_route_length_m=float(args.min_route_length_m),
        )
        selected_bundle = _selection_row_to_bundle(
            raw_scenario=raw,
            selection_row=selection_row,
            gt_future_xy=np.asarray(gt_future_xy, dtype=np.float32),
            path_rows=path_rows,
            spacing_m=float(args.resample_spacing_m),
            separability_scale_m=float(args.separability_scale_m),
            separability_heading_weight_m=float(args.separability_heading_weight_m),
            include_off_route_paths=bool(old_render_summary.get("include_off_route_paths", False)),
        )
        example_id = str(selection_row.get("example_id") or f"{scenario_id}__sdc_{sdc_id}__t_{current_time_index:03d}")
        example_dir = outdir / "examples" / example_id
        image_dir = example_dir / "images"
        payload = _build_payload_with_postsplit_gradients(
            raw_scenario=raw,
            example_id=example_id,
            scenario_id=scenario_id,
            sdc_id=sdc_id,
            current_time_index=current_time_index,
            gt_future_xy=np.asarray(gt_future_xy, dtype=np.float32),
            gt_past_xy=np.asarray(gt_past_xy, dtype=np.float32),
            selected_bundle=selected_bundle,
            image_dir=image_dir,
            gradient_display_reference=float(args.gradient_display_reference),
            gradient_display_gamma=float(args.gradient_display_gamma),
            show_traffic_lights=bool(args.show_traffic_lights),
        )
        payload["scene_index"] = int(selection_row.get("scene_index") or local_idx)
        payload["selection_rank"] = int(selection_row.get("selection_rank") or 0)
        payload["selection_score"] = float(selection_row.get("selection_score") or 0.0)
        payload["all_sdc_paths_grid_png"] = selection_row.get("all_sdc_paths_grid_png")
        payload["all_sdc_paths_grid_summary"] = selection_row.get("all_sdc_paths_grid_summary")
        write_json(example_dir / "render_metadata.json", payload)

        for slot_row in list(payload.get("slot_metadata") or []):
            slot_id = str(slot_row.get("slot_id") or "")
            if slot_id not in {"gt", "alt_1", "alt_2", "alt_3"}:
                continue
            prompt_manifest_rows.append(
                _slot_request_row(
                    payload=payload,
                    slot_row=slot_row,
                    example_dir=example_dir,
                    model_name=str(args.model),
                    image_detail=str(args.image_detail),
                )
            )

        render_rows.append(dict(payload))
        aggregate_rows.append(_aggregate_row_from_payload(payload, selection_row, example_dir=example_dir))
        processed += 1
        if processed % max(1, int(args.progress_every)) == 0 or processed == len(selection_rows):
            elapsed = time.time() - start_t
            print(f"Rerendered {processed}/{len(selection_rows)} scenes | elapsed {elapsed:.1f}s")

    render_rows = sorted(render_rows, key=lambda row: int(row.get("selection_rank") or 0))
    aggregate_rows = sorted(aggregate_rows, key=lambda row: int(row.get("selection_rank") or 0))

    write_json(
        outdir / "postsplit_render_manifest.json",
        {
            "path": str(old_render_manifest.get("path") or args.path),
            "candidate_scenes_scanned": int(old_render_manifest.get("candidate_scenes_scanned") or old_render_summary.get("candidate_scenes_requested") or 0),
            "scene_offset": int(old_render_manifest.get("scene_offset") or 0),
            "num_selected_scenes": int(len(render_rows)),
            "rows": render_rows,
        },
    )
    write_jsonl(outdir / "postsplit_request_manifest.jsonl", prompt_manifest_rows)
    write_jsonl(outdir / "postsplit_selected_scene_index.jsonl", aggregate_rows)

    if old_selection_summary:
        updated_selection_summary = dict(old_selection_summary)
        updated_selection_summary["gradient_display_reference"] = float(args.gradient_display_reference)
        updated_selection_summary["gradient_display_gamma"] = float(args.gradient_display_gamma)
        updated_selection_summary["show_traffic_lights"] = bool(args.show_traffic_lights)
        write_json(outdir / "postsplit_scene_selection.json", updated_selection_summary)

    summary = dict(old_render_summary)
    summary.update(
        {
            "path": str(summary.get("path") or args.path),
            "model": str(args.model),
            "image_detail": str(args.image_detail),
            "gradient_display_reference": float(args.gradient_display_reference),
            "gradient_display_gamma": float(args.gradient_display_gamma),
            "show_traffic_lights": bool(args.show_traffic_lights),
            "num_selected_scenes": int(len(render_rows)),
            "render_manifest_json": str((outdir / "postsplit_render_manifest.json").resolve()),
            "request_manifest_jsonl": str((outdir / "postsplit_request_manifest.jsonl").resolve()),
            "selection_summary_json": str((outdir / "postsplit_scene_selection.json").resolve()),
            "selected_scene_index_jsonl": str((outdir / "postsplit_selected_scene_index.jsonl").resolve()),
            "rerendered_from_selection_bundle": str(bundle_root),
        }
    )
    write_json(outdir / "postsplit_render_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
