from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class PlotValidationError(ValueError):
    pass


@dataclass(frozen=True)
class TaggedXYSeries:
    name: str
    xy: np.ndarray
    frame: str
    draw_style: str = "line"
    color: str = "#000000"
    label: Optional[str] = None
    alpha: float = 1.0
    linewidth: float = 1.5
    linestyle: str = "-"
    marker: str = "o"
    markersize: float = 24.0
    zorder: int = 1
    annotate: Optional[str] = None
    annotate_index: int = -1
    fill_alpha: float = 0.16


def _coerce_xy(xy: Any) -> np.ndarray:
    array = np.asarray(xy, dtype=np.float64)
    if array.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise PlotValidationError(f"Expected xy shaped [N,2+], got {array.shape}")
    return np.asarray(array[:, :2], dtype=np.float64)


def validate_tagged_series(
    series: TaggedXYSeries,
    *,
    expected_frame: str,
    example_id: str,
    plot_name: str,
    failures: List[Dict[str, Any]],
    local_limit_abs_m: float = 200.0,
) -> np.ndarray:
    if str(series.frame) != str(expected_frame):
        failure = {
            "example_id": str(example_id),
            "plot_name": str(plot_name),
            "series_name": str(series.name),
            "reason": "frame_mismatch",
            "expected_frame": str(expected_frame),
            "series_frame": str(series.frame),
        }
        failures.append(failure)
        raise PlotValidationError(
            f"{example_id}::{plot_name}::{series.name} frame mismatch "
            f"{series.frame!r} vs {expected_frame!r}"
        )
    xy = _coerce_xy(series.xy)
    if str(expected_frame) == "agent_relative_at_decision" and xy.size > 0:
        over_limit = np.argwhere(np.abs(xy) > float(local_limit_abs_m))
        if over_limit.size > 0:
            point_idx, axis_idx = over_limit[0].tolist()
            failure = {
                "example_id": str(example_id),
                "plot_name": str(plot_name),
                "series_name": str(series.name),
                "reason": "local_frame_sanity_limit_exceeded",
                "expected_frame": str(expected_frame),
                "axis": "x" if int(axis_idx) == 0 else "y",
                "point_index": int(point_idx),
                "value_m": float(xy[int(point_idx), int(axis_idx)]),
                "limit_abs_m": float(local_limit_abs_m),
            }
            failures.append(failure)
            raise PlotValidationError(
                f"{example_id}::{plot_name}::{series.name} exceeds local sanity bound "
                f"{local_limit_abs_m}m"
            )
    return xy


def plot_tagged_series(
    ax: Any,
    series: TaggedXYSeries,
    xy: np.ndarray,
) -> None:
    if xy.size == 0:
        return
    label = series.label
    if series.draw_style == "line":
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=series.color,
            linewidth=series.linewidth,
            linestyle=series.linestyle,
            alpha=series.alpha,
            zorder=series.zorder,
            label=label,
        )
    elif series.draw_style == "scatter":
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            color=series.color,
            alpha=series.alpha,
            zorder=series.zorder,
            s=series.markersize,
            marker=series.marker,
            label=label,
        )
    elif series.draw_style == "polygon":
        ax.fill(
            xy[:, 0],
            xy[:, 1],
            color=series.color,
            alpha=series.fill_alpha,
            zorder=series.zorder,
            label=label,
        )
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=series.color,
            linewidth=max(1.0, float(series.linewidth)),
            linestyle=series.linestyle,
            alpha=min(1.0, float(series.alpha) + 0.1),
            zorder=series.zorder,
        )
    else:
        raise PlotValidationError(f"Unknown draw_style={series.draw_style!r} for {series.name}")

    if series.annotate:
        index = int(series.annotate_index)
        if index < 0:
            index = max(0, xy.shape[0] + index)
        index = min(max(index, 0), xy.shape[0] - 1)
        ax.text(
            float(xy[index, 0]),
            float(xy[index, 1]),
            str(series.annotate),
            color=series.color,
            fontsize=8,
            zorder=max(series.zorder, 10),
        )


def render_tagged_series_collection(
    ax: Any,
    *,
    series_list: Sequence[TaggedXYSeries],
    expected_frame: str,
    example_id: str,
    plot_name: str,
    failures: List[Dict[str, Any]],
    local_limit_abs_m: float = 200.0,
) -> np.ndarray:
    plotted_points: List[np.ndarray] = []
    for series in series_list:
        xy = validate_tagged_series(
            series,
            expected_frame=expected_frame,
            example_id=example_id,
            plot_name=plot_name,
            failures=failures,
            local_limit_abs_m=local_limit_abs_m,
        )
        plot_tagged_series(ax, series, xy)
        if xy.size > 0:
            plotted_points.append(xy)
    if not plotted_points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.concatenate(plotted_points, axis=0)


def set_axes_from_points(
    ax: Any,
    points: np.ndarray,
    *,
    padding: float = 10.0,
    fixed_half_extent: Optional[float] = None,
) -> Dict[str, float]:
    if fixed_half_extent is not None:
        half_extent = float(fixed_half_extent)
        ax.set_xlim(-half_extent, half_extent)
        ax.set_ylim(-half_extent, half_extent)
        return {
            "x_min": -half_extent,
            "x_max": half_extent,
            "y_min": -half_extent,
            "y_max": half_extent,
        }
    if points.size == 0:
        half_extent = float(padding)
        ax.set_xlim(-half_extent, half_extent)
        ax.set_ylim(-half_extent, half_extent)
        return {
            "x_min": -half_extent,
            "x_max": half_extent,
            "y_min": -half_extent,
            "y_max": half_extent,
        }
    min_xy = np.min(points, axis=0)
    max_xy = np.max(points, axis=0)
    center = (min_xy + max_xy) / 2.0
    half_extent = max(float(np.max(max_xy - min_xy)) / 2.0 + float(padding), float(padding))
    ax.set_xlim(float(center[0] - half_extent), float(center[0] + half_extent))
    ax.set_ylim(float(center[1] - half_extent), float(center[1] + half_extent))
    return {
        "x_min": float(center[0] - half_extent),
        "x_max": float(center[0] + half_extent),
        "y_min": float(center[1] - half_extent),
        "y_max": float(center[1] + half_extent),
    }
