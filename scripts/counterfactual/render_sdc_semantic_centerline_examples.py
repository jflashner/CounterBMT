from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
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

from bmt.dataset.dataset import InfgenDataset
from bmt.counterfactual.sdc_semantic_control import load_raw_scenario_from_row
from bmt.utils.config import global_config
from scripts.counterfactual.plot_sdc_semantic_progress_cap_examples import (
    _draw_scene_context,
    _extract_scene_render_context,
)
from scripts.counterfactual.render_sdc_semantic_eval_examples import (
    _read_jsonl,
    _save_grid,
    _scenario_sort_key,
)


RAW_PATH_COLOR = "#2563eb"
PROGRESS_CENTERLINE_COLOR = "#be185d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render scene-context plots showing the indexed selected raw route and "
            "indexed selected progress centerline for semantic-control rows."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/Adv-BMT/cfgs/motion_forward_sdc_semantic_only_strict_local.yaml",
    )
    parser.add_argument("--control-index", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--teacher-ckpt", type=str, default="")
    parser.add_argument("--sampling-method", type=str, default="argmax")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topp", type=float, default=1.0)
    parser.add_argument("--num-scenes", type=int, default=20)
    parser.add_argument("--grid-columns", type=int, default=2)
    parser.add_argument("--tile-size", type=int, default=900)
    return parser.parse_args()


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _world_to_sdc_up_frame(
    points_world_xy: Any,
    *,
    center_xy_world: Sequence[float],
    heading_world_rad: float,
) -> np.ndarray:
    points = _to_numpy(points_world_xy).astype(np.float32).reshape(-1, 2)
    if points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    center_xy = np.asarray(center_xy_world, dtype=np.float32).reshape(2)
    centered = points - center_xy[None, :]
    rot = (math.pi / 2.0) - float(heading_world_rad)
    c = math.cos(rot)
    s = math.sin(rot)
    return np.stack(
        [c * centered[:, 0] - s * centered[:, 1], s * centered[:, 0] + c * centered[:, 1]],
        axis=-1,
    ).astype(np.float32)


def _valid_points(points_xy: Any, point_mask: Any) -> np.ndarray:
    points = _to_numpy(points_xy).astype(np.float32).reshape(-1, 2)
    mask = _to_numpy(point_mask).reshape(-1) > 0.5
    if points.shape[0] == 0 or mask.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    count = min(points.shape[0], mask.shape[0])
    return np.asarray(points[:count][mask[:count]], dtype=np.float32)


def _split_by_segment_mask(points_xy: Any, point_mask: Any, segment_mask: Any) -> List[np.ndarray]:
    points = _to_numpy(points_xy).astype(np.float32).reshape(-1, 2)
    point_mask_arr = _to_numpy(point_mask).reshape(-1) > 0.5
    seg_mask_arr = _to_numpy(segment_mask).reshape(-1) > 0.5
    if points.shape[0] == 0 or point_mask_arr.shape[0] == 0:
        return []

    count = min(points.shape[0], point_mask_arr.shape[0])
    points = points[:count]
    point_mask_arr = point_mask_arr[:count]
    if seg_mask_arr.shape[0] < max(count - 1, 0):
        padded = np.zeros((max(count - 1, 0),), dtype=bool)
        padded[: seg_mask_arr.shape[0]] = seg_mask_arr
        seg_mask_arr = padded
    else:
        seg_mask_arr = seg_mask_arr[: max(count - 1, 0)]

    valid_indices = np.flatnonzero(point_mask_arr)
    if valid_indices.size < 2:
        return []

    segments: List[List[np.ndarray]] = []
    current_segment: List[np.ndarray] = [points[int(valid_indices[0])]]
    for idx in valid_indices[1:]:
        idx = int(idx)
        prev_idx = int(valid_indices[np.where(valid_indices == idx)[0][0] - 1])
        connected = (idx == prev_idx + 1) and bool(seg_mask_arr[prev_idx])
        if connected:
            current_segment.append(points[idx])
            continue
        if len(current_segment) >= 2:
            segments.append(current_segment)
        current_segment = [points[idx]]
    if len(current_segment) >= 2:
        segments.append(current_segment)
    return [np.asarray(seg, dtype=np.float32) for seg in segments]


def _polyline_length(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(points_xy[1:] - points_xy[:-1], axis=-1).sum())


def _build_dataset_config(args: argparse.Namespace):
    cfg = copy.deepcopy(global_config)
    data_dir = str(Path(args.data_dir).expanduser().resolve())
    control_index = str(Path(args.control_index).expanduser().resolve())
    cfg.DATA.TRAINING_DATA_DIR = data_dir
    cfg.DATA.TEST_DATA_DIR = data_dir
    cfg.DATA.COUNTERFACTUAL_CONTROL_INDEX = control_index
    cfg.DATA.COUNTERFACTUAL_CONTROL_INDEX_TRAIN = control_index
    cfg.DATA.COUNTERFACTUAL_CONTROL_INDEX_VAL = control_index
    cfg.DATA.COUNTERFACTUAL_MODE = "sdc_semantic_only"
    cfg.DATA.COUNTERFACTUAL_WEIGHTED_SAMPLER = False
    return cfg


def _draw_centerline_panel(
    *,
    ax: plt.Axes,
    row: Mapping[str, Any],
    data_dict: Mapping[str, Any],
    raw_scenario: Mapping[str, Any],
) -> Dict[str, Any]:
    render_context = _extract_scene_render_context(raw_scenario, row)
    current_xy = np.asarray(render_context["current_xy"], dtype=np.float32)
    current_heading = float(render_context["current_heading"])

    raw_segments_world = _split_by_segment_mask(
        data_dict["cf/sdc_selected_raw_path_world"],
        data_dict["cf/sdc_selected_raw_path_mask"],
        data_dict["cf/sdc_selected_raw_path_segment_mask"],
    )
    progress_segments_world = _split_by_segment_mask(
        data_dict["cf/sdc_selected_progress_centerline_world"],
        data_dict["cf/sdc_selected_progress_centerline_mask"],
        data_dict["cf/sdc_selected_progress_centerline_segment_mask"],
    )

    raw_world = _valid_points(
        data_dict["cf/sdc_selected_raw_path_world"],
        data_dict["cf/sdc_selected_raw_path_mask"],
    )
    progress_world = _valid_points(
        data_dict["cf/sdc_selected_progress_centerline_world"],
        data_dict["cf/sdc_selected_progress_centerline_mask"],
    )

    info_box = (
        f"scene={row['scenario_id']}\n"
        f"slot={row.get('selected_slot_id') or row.get('slot_id')}\n"
        f"label={row.get('requested_semantic_label') or row.get('semantic_label')}\n"
        f"source={row.get('source_kind')}\n"
        f"raw_len={_polyline_length(raw_world):.1f}m\n"
        f"progress_len={_polyline_length(progress_world):.1f}m"
    )
    _draw_scene_context(ax=ax, render_context=render_context, info_box_text=info_box)

    for seg_world in raw_segments_world:
        seg_local = _world_to_sdc_up_frame(
            seg_world,
            center_xy_world=current_xy,
            heading_world_rad=current_heading,
        )
        if seg_local.shape[0] >= 2:
            ax.plot(
                seg_local[:, 0],
                seg_local[:, 1],
                color=RAW_PATH_COLOR,
                linewidth=3.4,
                alpha=0.85,
                zorder=8.8,
                solid_capstyle="round",
            )

    for seg_world in progress_segments_world:
        seg_local = _world_to_sdc_up_frame(
            seg_world,
            center_xy_world=current_xy,
            heading_world_rad=current_heading,
        )
        if seg_local.shape[0] >= 2:
            ax.plot(
                seg_local[:, 0],
                seg_local[:, 1],
                color=PROGRESS_CENTERLINE_COLOR,
                linewidth=2.6,
                alpha=0.95,
                zorder=9.2,
                linestyle="-.",
                solid_capstyle="round",
            )

    ax.plot([], [], color=RAW_PATH_COLOR, linewidth=3.2, label="Indexed raw route")
    ax.plot([], [], color=PROGRESS_CENTERLINE_COLOR, linewidth=2.4, linestyle="-.", label="Indexed progress centerline")
    ax.legend(
        loc="lower right",
        fontsize=8,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#cbd5e1",
    )

    return {
        "scenario_id": str(row.get("scenario_id") or ""),
        "selected_slot_id": str(row.get("selected_slot_id") or row.get("slot_id") or ""),
        "requested_semantic_label": str(row.get("requested_semantic_label") or row.get("semantic_label") or ""),
        "source_kind": str(row.get("source_kind") or ""),
        "selected_path_id": row.get("selected_path_id"),
        "raw_num_points": int(raw_world.shape[0]),
        "progress_num_points": int(progress_world.shape[0]),
        "raw_length_m": _polyline_length(raw_world),
        "progress_length_m": _polyline_length(progress_world),
    }


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(Path(args.control_index).expanduser().resolve())
    selected_sids: List[str] = []
    for row in rows:
        sid = str(row.get("scenario_id") or "")
        if sid and sid not in selected_sids:
            selected_sids.append(sid)
        if len(selected_sids) >= int(args.num_scenes):
            break
    selected_sid_set = set(selected_sids)

    dataset_config = _build_dataset_config(args)
    dataset = InfgenDataset(dataset_config, "test", backward_prediction=False)

    scenario_to_rows: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {sid: [] for sid in selected_sids}
    for idx, row in enumerate(rows):
        sid = str(row.get("scenario_id") or "")
        if sid in selected_sid_set:
            scenario_to_rows[sid].append((idx, row))

    manifest: List[Dict[str, Any]] = []
    for scenario_id in selected_sids:
        row_entries = sorted(scenario_to_rows[scenario_id], key=lambda item: _scenario_sort_key(item[1]))
        if not row_entries:
            continue
        scenario_dir = outdir / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        panels = []
        panel_meta: List[Dict[str, Any]] = []
        for row_idx, row in row_entries:
            data_dict = dataset[row_idx]
            raw_scenario = load_raw_scenario_from_row(row)
            fig, ax = plt.subplots(figsize=(8.2, 8.2), dpi=180)
            meta = _draw_centerline_panel(ax=ax, row=row, data_dict=data_dict, raw_scenario=raw_scenario)
            slot_id = str(row.get("selected_slot_id") or row.get("slot_id") or f"row_{int(row_idx):04d}")
            title = f"{slot_id} | {meta['requested_semantic_label']}"
            png_path = scenario_dir / f"{scenario_id}__{slot_id}__centerlines.png"
            fig.savefig(png_path, bbox_inches="tight")
            plt.close(fig)
            meta["png"] = str(png_path)
            meta["row_index"] = int(row_idx)
            panel_meta.append(meta)
            panels.append((title, png_path))

        grid_items = []
        for title, png_path in panels:
            from PIL import Image

            grid_items.append((title, Image.open(png_path).convert("RGB")))
        grid_path = scenario_dir / f"{scenario_id}__centerlines_grid.png"
        _save_grid(grid_items, grid_path, columns=int(args.grid_columns), tile_size=int(args.tile_size))
        for _, image in grid_items:
            image.close()

        manifest.append(
            {
                "scenario_id": scenario_id,
                "grid_png": str(grid_path),
                "rows": panel_meta,
            }
        )

    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "num_scenes": int(len(manifest)),
                "control_index": str(Path(args.control_index).expanduser().resolve()),
                "examples": manifest,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"manifest_json": str(manifest_path), "num_scenes": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
