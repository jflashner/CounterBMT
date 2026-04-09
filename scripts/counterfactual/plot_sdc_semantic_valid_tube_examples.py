from __future__ import annotations

import argparse
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

from bmt.counterfactual.sdc_path_control import polyline_length_m, split_polyline_on_discontinuities
from bmt.counterfactual.sdc_semantic_control import load_raw_scenario_from_row
from scripts.counterfactual.eval_sdc_semantic_action_projections import (
    _draw_vlm_style_scene_ax,
    _extract_scene_render_context,
    _read_jsonl,
    _select_row,
)
from scripts.counterfactual.sdc_semantic_tube_utils import (
    segment_distance_field_in_sdc_frame,
    selected_raw_route_world,
    world_to_sdc_up_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 3m valid-tube overlays around raw semantic route geometry for selected scene slots."
    )
    parser.add_argument("--control-index", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument(
        "--example",
        action="append",
        default=[],
        help="Example to plot, formatted as scenario_id:slot_id . Repeatable.",
    )
    parser.add_argument("--tube-radius-m", type=float, default=3.0)
    parser.add_argument("--grid-step-m", type=float, default=0.35)
    parser.add_argument("--jump-threshold-m", type=float, default=6.0)
    return parser.parse_args()


def _parse_example_spec(spec: str) -> Tuple[str, str]:
    text = str(spec).strip()
    if ":" not in text:
        raise ValueError(f"Example spec must be scenario_id:slot_id, got {spec!r}")
    scenario_id, slot_id = text.split(":", 1)
    return str(scenario_id).strip(), str(slot_id).strip()


def _plot_single_example(
    *,
    ax,
    fig,
    row: Mapping[str, Any],
    raw_scenario: Mapping[str, Any],
    tube_radius_m: float,
    grid_step_m: float,
    jump_threshold_m: float,
) -> Dict[str, Any]:
    render_context = _extract_scene_render_context(raw_scenario, row)
    current_xy = np.asarray(render_context["current_xy"], dtype=np.float32)
    current_heading = float(render_context["current_heading"])
    path_world = np.asarray(selected_raw_route_world(raw_scenario, row), dtype=np.float32)
    segments_world = [
        np.asarray(seg, dtype=np.float32)
        for seg in split_polyline_on_discontinuities(path_world, jump_threshold_m=float(jump_threshold_m))
        if np.asarray(seg).shape[0] >= 2
    ]

    xx, yy, distance_field = segment_distance_field_in_sdc_frame(
        polyline_world_xy=path_world,
        center_xy_world=current_xy,
        heading_world_rad=current_heading,
        grid_step_m=float(grid_step_m),
    )

    _draw_vlm_style_scene_ax(
        fig=fig,
        ax=ax,
        render_context=render_context,
        highlighted_segments_world=[],
        highlighted_gradient_values=None,
        representative_route_world=path_world,
        info_box_text=(
            f"scene={row['scenario_id']}\n"
            f"slot={row['selected_slot_id']}\n"
            f"requested={row['requested_semantic_label']}\n"
            f"path={row.get('selected_path_id')}\n"
            f"tube_radius={float(tube_radius_m):.1f}m\n"
            f"segments={len(segments_world)}"
        ),
        show_colorbar=False,
    )

    inside = np.ma.masked_where(distance_field > float(tube_radius_m), distance_field)
    ax.contourf(
        xx,
        yy,
        inside,
        levels=np.linspace(0.0, float(tube_radius_m), num=8),
        cmap="Blues_r",
        alpha=0.24,
        zorder=6.1,
        antialiased=True,
    )
    ax.contour(
        xx,
        yy,
        distance_field,
        levels=[float(tube_radius_m)],
        colors=["#f59e0b"],
        linewidths=1.8,
        linestyles=["--"],
        zorder=11.5,
    )

    for seg_idx, seg_world in enumerate(segments_world):
        seg_local = world_to_sdc_up_frame(
            seg_world,
            center_xy_world=current_xy,
            heading_world_rad=current_heading,
        )
        if seg_local.shape[0] < 2:
            continue
        ax.plot(
            seg_local[:, 0],
            seg_local[:, 1],
            color="#2563eb",
            linewidth=4.6,
            alpha=0.98,
            zorder=10.0,
            solid_capstyle="round",
        )
        if seg_idx > 0:
            ax.scatter(
                [seg_local[0, 0]],
                [seg_local[0, 1]],
                c="#111827",
                s=28,
                marker="x",
                linewidths=1.1,
                zorder=10.8,
            )

    return {
        "scenario_id": str(row.get("scenario_id") or ""),
        "selected_slot_id": str(row.get("selected_slot_id") or ""),
        "requested_semantic_label": str(row.get("requested_semantic_label") or ""),
        "selected_path_id": row.get("selected_path_id"),
        "tube_radius_m": float(tube_radius_m),
        "jump_threshold_m": float(jump_threshold_m),
        "num_segments": int(len(segments_world)),
        "path_length_m": float(polyline_length_m(path_world)),
    }


def main() -> int:
    args = parse_args()
    if not list(args.example):
        raise ValueError("Provide at least one --example scenario_id:slot_id")

    rows = _read_jsonl(Path(args.control_index).expanduser().resolve())
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    example_specs = [_parse_example_spec(spec) for spec in list(args.example)]
    cols = min(2, max(1, len(example_specs)))
    grid_rows = int(math.ceil(float(len(example_specs)) / float(cols)))
    fig, axes = plt.subplots(grid_rows, cols, figsize=(7.8 * cols, 7.8 * grid_rows), dpi=180)
    axes = np.asarray(axes).reshape(-1)

    manifest: List[Dict[str, Any]] = []
    for ax, (scenario_id, slot_id) in zip(axes, example_specs):
        row_index = _select_row(rows, row_index=-1, scenario_id=scenario_id, slot_id=slot_id)
        row = dict(rows[row_index])
        raw_scenario = load_raw_scenario_from_row(row)
        meta = _plot_single_example(
            ax=ax,
            fig=fig,
            row=row,
            raw_scenario=raw_scenario,
            tube_radius_m=float(args.tube_radius_m),
            grid_step_m=float(args.grid_step_m),
            jump_threshold_m=float(args.jump_threshold_m),
        )
        slug = f"{scenario_id}__{slot_id}".replace("/", "_")
        single_path = outdir / f"{slug}_tube.png"
        single_fig = plt.figure(figsize=(7.8, 7.8), dpi=180)
        single_ax = single_fig.add_axes([0.02, 0.02, 0.96, 0.96])
        _plot_single_example(
            ax=single_ax,
            fig=single_fig,
            row=row,
            raw_scenario=raw_scenario,
            tube_radius_m=float(args.tube_radius_m),
            grid_step_m=float(args.grid_step_m),
            jump_threshold_m=float(args.jump_threshold_m),
        )
        single_fig.savefig(single_path, bbox_inches="tight")
        plt.close(single_fig)
        meta["png"] = str(single_path)
        meta["row_index"] = int(row_index)
        manifest.append(meta)

    for ax in axes[len(example_specs) :]:
        ax.axis("off")

    fig.suptitle(
        f"Raw semantic-route valid tube overlays (tube radius = {float(args.tube_radius_m):.1f}m)",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.975])
    grid_path = outdir / "tube_examples_grid.png"
    fig.savefig(grid_path, bbox_inches="tight")
    plt.close(fig)

    manifest_path = outdir / "tube_examples_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "tube_radius_m": float(args.tube_radius_m),
                "grid_step_m": float(args.grid_step_m),
                "jump_threshold_m": float(args.jump_threshold_m),
                "grid_png": str(grid_path),
                "examples": manifest,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "tube_examples_manifest_json": str(manifest_path),
                "tube_examples_grid_png": str(grid_path),
                "examples": manifest,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
