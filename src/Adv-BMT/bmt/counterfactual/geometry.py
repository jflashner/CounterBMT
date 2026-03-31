from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence

import numpy as np


def wrap_to_pi(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def angle_delta(target: float, source: float) -> float:
    return wrap_to_pi(float(target) - float(source))


def heading_from_points(p0: Sequence[float], p1: Sequence[float]) -> float:
    v = np.asarray(p1, dtype=np.float32)[:2] - np.asarray(p0, dtype=np.float32)[:2]
    if float(np.linalg.norm(v)) < 1e-6:
        return 0.0
    return float(math.atan2(float(v[1]), float(v[0])))


def polyline_length(polyline_xy: np.ndarray) -> float:
    if polyline_xy.shape[0] < 2:
        return 0.0
    diffs = np.diff(np.asarray(polyline_xy, dtype=np.float32), axis=0)
    return float(np.linalg.norm(diffs, axis=-1).sum())


def cumulative_polyline_lengths(polyline_xy: np.ndarray) -> np.ndarray:
    if polyline_xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if polyline_xy.shape[0] == 1:
        return np.zeros((1,), dtype=np.float32)
    diffs = np.diff(np.asarray(polyline_xy, dtype=np.float32), axis=0)
    seg = np.linalg.norm(diffs, axis=-1)
    return np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(seg, dtype=np.float32)], axis=0)


def nearest_point_index(polyline_xy: np.ndarray, point_xy: Sequence[float]) -> int:
    if polyline_xy.shape[0] == 0:
        return 0
    d = np.linalg.norm(np.asarray(polyline_xy, dtype=np.float32) - np.asarray(point_xy, dtype=np.float32)[:2], axis=-1)
    return int(np.argmin(d))


def min_distance_to_points(points_xy: np.ndarray, point_xy: Sequence[float]) -> float:
    if points_xy.shape[0] == 0:
        return float("inf")
    d = np.linalg.norm(np.asarray(points_xy, dtype=np.float32) - np.asarray(point_xy, dtype=np.float32)[:2], axis=-1)
    return float(np.min(d))


def any_point_within_radius(points_xy: np.ndarray, center_xy: Sequence[float], radius_m: float) -> bool:
    return bool(min_distance_to_points(points_xy, center_xy) <= float(radius_m))


def point_distance_curve(points_xy: np.ndarray, point_xy: Sequence[float], valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    if points_xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    center = np.asarray(point_xy, dtype=np.float32)[:2]
    d = np.linalg.norm(np.asarray(points_xy, dtype=np.float32) - center[None, :], axis=-1).astype(np.float32)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        out = np.full((d.shape[0],), np.inf, dtype=np.float32)
        rows = min(out.shape[0], mask.shape[0], d.shape[0])
        out[:rows] = np.where(mask[:rows], d[:rows], np.float32(np.inf))
        return out
    return d


def contiguous_true_run_lengths(mask: np.ndarray) -> List[int]:
    runs: List[int] = []
    run = 0
    for value in np.asarray(mask, dtype=bool).tolist():
        if value:
            run += 1
        elif run > 0:
            runs.append(int(run))
            run = 0
    if run > 0:
        runs.append(int(run))
    return runs


def max_contiguous_true_run(mask: np.ndarray) -> int:
    runs = contiguous_true_run_lengths(mask)
    return max(runs) if runs else 0


def heading_from_track_window(position_xy: np.ndarray, valid_mask: np.ndarray, *, end_idx: int, lookback_steps: int = 8) -> Optional[float]:
    if position_xy.shape[0] == 0:
        return None
    end_idx = int(np.clip(end_idx, 0, position_xy.shape[0] - 1))
    start_idx = max(0, end_idx - int(lookback_steps))
    idx = [
        t
        for t in range(start_idx, end_idx + 1)
        if bool(valid_mask[t]) and np.isfinite(position_xy[t]).all()
    ]
    if len(idx) < 2:
        return None
    return heading_from_points(position_xy[idx[0]], position_xy[idx[-1]])


def classify_heading_delta(heading_delta_rad: float) -> str:
    deg = math.degrees(wrap_to_pi(heading_delta_rad))
    abs_deg = abs(deg)
    if abs_deg <= 30.0:
        return "straight"
    if 30.0 < deg < 150.0:
        return "left"
    if -150.0 < deg < -30.0:
        return "right"
    return "u_turn"


def cluster_heading_values(headings: Sequence[float], *, threshold_deg: float = 25.0) -> List[List[int]]:
    if not headings:
        return []
    threshold_rad = math.radians(float(threshold_deg))
    ordered = sorted(range(len(headings)), key=lambda idx: wrap_to_pi(headings[idx]))
    clusters: List[List[int]] = []
    for idx in ordered:
        placed = False
        for cluster in clusters:
            ref = headings[cluster[0]]
            if abs(angle_delta(headings[idx], ref)) <= threshold_rad:
                cluster.append(int(idx))
                placed = True
                break
        if not placed:
            clusters.append([int(idx)])
    return clusters


def finite_mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float(default)
