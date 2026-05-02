from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

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
    _build_payload_with_postsplit_gradients,
    _slot_request_row,
)
from scripts.counterfactual.rerender_postsplit_bundle_from_selection import (
    _aggregate_row_from_payload,
    _parse_scene_indices,
    _read_json,
    _read_jsonl,
    _selection_row_to_bundle,
)


DEFAULT_SCENARIO_ROOT = "outputs/pr10_1_sdc_semantic_top859_full/scenario_root"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerender an existing postsplit bundle from local raw scenario pickles."
    )
    parser.add_argument("--bundle-root", type=str, required=True)
    parser.add_argument("--scenario-root", type=str, default=DEFAULT_SCENARIO_ROOT)
    parser.add_argument("--outdir", type=str, default="")
    parser.add_argument("--min-route-length-m", type=float, default=15.0)
    parser.add_argument("--resample-spacing-m", type=float, default=2.0)
    parser.add_argument("--separability-scale-m", type=float, default=DEFAULT_SEPARABILITY_SCALE_M)
    parser.add_argument(
        "--separability-heading-weight-m",
        type=float,
        default=DEFAULT_SEPARABILITY_HEADING_WEIGHT_M,
    )
    parser.add_argument("--gradient-display-reference", type=float, default=0.75)
    parser.add_argument("--gradient-display-gamma", type=float, default=1.10)
    parser.add_argument("--show-traffic-lights", action="store_true")
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument(
        "--image-detail",
        type=str,
        default="original",
        choices=("low", "high", "original", "auto"),
    )
    parser.add_argument("--scene-indices", type=str, default="")
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def _resolve_scenario_pickle(scenario_root: Path, scenario_id: str) -> Path:
    matches = sorted(scenario_root.glob(f"*{scenario_id}.pkl"))
    if not matches:
        raise FileNotFoundError(f"No scenario pickle found for {scenario_id} under {scenario_root}")
    return matches[0]


def _load_raw_scenario(scenario_root: Path, scenario_id: str) -> Mapping[str, Any]:
    scenario_path = _resolve_scenario_pickle(scenario_root, scenario_id)
    with scenario_path.open("rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, Mapping):
        raise TypeError(f"Scenario pickle for {scenario_id} did not contain a mapping")
    return raw


def main() -> int:
    args = parse_args()
    bundle_root = Path(args.bundle_root).expanduser().resolve()
    scenario_root = Path(args.scenario_root).expanduser().resolve()
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
        selection_rows = [
            row for row in selection_rows if int(row.get("scene_index") or -1) in scene_filter
        ]
    if int(args.limit_scenes) > 0:
        selection_rows = selection_rows[: int(args.limit_scenes)]
    if not selection_rows:
        raise RuntimeError("No selected scenes remain after filtering.")

    old_render_summary = _read_json(render_summary_path) if render_summary_path.is_file() else {}
    old_selection_summary = _read_json(selection_summary_path) if selection_summary_path.is_file() else {}
    old_render_manifest = _read_json(render_manifest_path) if render_manifest_path.is_file() else {}

    render_rows: List[Dict[str, Any]] = []
    prompt_manifest_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []

    start_t = time.time()
    for processed, selection_row in enumerate(selection_rows, start=1):
        selection_row = dict(selection_row)
        scenario_id = str(selection_row.get("scenario_id") or "")
        sdc_id = str(selection_row.get("sdc_id") or "")
        current_time_index = int(selection_row.get("current_time_index") or 0)
        example_id = str(
            selection_row.get("example_id") or f"{scenario_id}__sdc_{sdc_id}__t_{current_time_index:03d}"
        )

        raw = _load_raw_scenario(scenario_root, scenario_id)
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
        payload["scene_index"] = int(selection_row.get("scene_index") or 0)
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
        aggregate_rows.append(
            _aggregate_row_from_payload(payload, selection_row, example_dir=example_dir)
        )
        if processed % max(1, int(args.progress_every)) == 0 or processed == len(selection_rows):
            elapsed = time.time() - start_t
            print(f"Rerendered {processed}/{len(selection_rows)} scenes | elapsed {elapsed:.1f}s")

    render_rows = sorted(render_rows, key=lambda row: int(row.get("selection_rank") or 0))
    aggregate_rows = sorted(aggregate_rows, key=lambda row: int(row.get("selection_rank") or 0))

    write_json(
        outdir / "postsplit_render_manifest.json",
        {
            "path": str(old_render_manifest.get("path") or ""),
            "candidate_scenes_scanned": int(
                old_render_manifest.get("candidate_scenes_scanned")
                or old_render_summary.get("candidate_scenes_requested")
                or 0
            ),
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
            "rerendered_from_local_scenario_root": str(scenario_root),
        }
    )
    write_json(outdir / "postsplit_render_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
