from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    for path in (repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual.sdc_path_control import (
    ResampledLocalPath,
    _sanitize_polyline,
    polyline_arc_lengths,
    polyline_headings,
    resample_polyline_xy,
    world_to_sdc_up_frame,
)
from scripts.counterfactual.path_semantics_plot_utils import _finite_xy_rows

DEFAULT_RESAMPLE_SPACING_M = 2.0
DEFAULT_SEPARABILITY_SCALE_M = 6.0
DEFAULT_SEPARABILITY_HEADING_WEIGHT_M = 2.0


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
        segment_world_resampled = _sanitize_polyline(
            resample_polyline_xy(segment_world_xy, spacing_m=float(spacing_m))
        ).astype(np.float32)
        if segment_world_resampled.shape[0] < 2:
            continue
        segment_local = world_to_sdc_up_frame(
            segment_world_resampled,
            origin_xy_world=np.asarray(center_xy_world, dtype=np.float32),
            origin_heading_world=float(origin_heading_world),
        )
        segment_local = _sanitize_polyline(segment_local).astype(np.float32)
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


def _display_gradient_values(values: np.ndarray, *, reference: float, gamma: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    ref = max(float(reference), 1e-3)
    gam = max(float(gamma), 1e-3)
    scaled = np.clip(arr / ref, 0.0, 1.0)
    return np.power(scaled, gam, dtype=np.float32)
