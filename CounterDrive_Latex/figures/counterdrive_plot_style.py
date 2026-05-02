from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


TEXT_COLOR = "#111827"
MUTED_TEXT = "#4B5563"
GRID_COLOR = "#E5E7EB"
AXIS_COLOR = "#D1D5DB"
COUNTERDRIVE_BLUE = "#2563EB"
COMPARATOR_ORANGE = "#D97706"
RISK_GREEN = "#059669"
RISK_AMBER = "#D97706"
RISK_ROSE = "#E11D48"
METHOD_GRAYS = ["#D1D5DB", "#AEB8C4", "#8592A3", "#64748B"]


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelcolor": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _normalize(values: Sequence[float], y_min: float, y_max: float) -> np.ndarray:
    return (np.asarray(values, dtype=float) - float(y_min)) / max(float(y_max) - float(y_min), 1e-9)


def draw_metric_group_bars(
    *,
    metrics: Sequence[Mapping[str, object]],
    series_names: Sequence[str],
    colors: Sequence[str],
    output_prefix: Path,
    figsize: tuple[float, float],
    legend_ncol: int,
    annotate: bool = False,
    note: str | None = None,
) -> None:
    setup_plot_style()
    figure, axis = plt.subplots(figsize=figsize, dpi=300)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")

    group_centers = np.arange(len(metrics), dtype=float)
    bar_width = min(0.18, 0.70 / max(len(series_names), 1))
    offsets = (np.arange(len(series_names), dtype=float) - (len(series_names) - 1) / 2.0) * bar_width

    for metric_index, metric in enumerate(metrics):
        y_min, y_max = metric["display_ylim"]
        raw_values = np.asarray(metric["values"], dtype=float)
        scaled_values = _normalize(raw_values, float(y_min), float(y_max))
        x_values = group_centers[metric_index] + offsets

        bars = axis.bar(
            x_values,
            scaled_values,
            width=bar_width * 0.86,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for bar in bars:
            bar.set_clip_on(False)

        errors = metric.get("errors")
        if errors is not None:
            raw_errors = np.asarray(errors, dtype=float)
            for x_value, y_value, err in zip(x_values, scaled_values, raw_errors):
                if not np.isfinite(err) or float(err) <= 0.0:
                    continue
                axis.errorbar(
                    x_value,
                    y_value,
                    yerr=float(err) / max(float(y_max) - float(y_min), 1e-9),
                    color=TEXT_COLOR,
                    capsize=1.8,
                    elinewidth=0.7,
                    capthick=0.7,
                    zorder=5,
                )

        tick_values = metric.get("tick_values")
        if tick_values is None:
            tick_values = [y_min, (float(y_min) + float(y_max)) / 2.0, y_max]
        local_axis_x = group_centers[metric_index] - 0.48
        axis.vlines(local_axis_x, 0.0, 1.0, colors=AXIS_COLOR, linewidth=0.65, zorder=2)
        for tick_value in tick_values:
            tick_y = float(_normalize([float(tick_value)], float(y_min), float(y_max))[0])
            if tick_y < -0.03 or tick_y > 1.03:
                continue
            axis.hlines(tick_y, local_axis_x, local_axis_x + 0.035, colors=AXIS_COLOR, linewidth=0.65, zorder=2)
            axis.text(
                local_axis_x - 0.025,
                tick_y,
                metric.get("tick_fmt", "{:.2f}").format(float(tick_value)),
                ha="right",
                va="center",
                fontsize=5.3,
                color=MUTED_TEXT,
                zorder=4,
            )

        if annotate:
            y_span = float(y_max) - float(y_min)
            for bar, raw_value, scaled_value in zip(bars, raw_values, scaled_values):
                y_text = min(float(scaled_value) + 0.045, 1.075)
                va = "bottom"
                if y_text > 1.03:
                    y_text = max(float(scaled_value) - 0.055, 0.04)
                    va = "top"
                axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y_text,
                    metric.get("label_fmt", "{:.2f}").format(float(raw_value)),
                    ha="center",
                    va=va,
                    fontsize=5.6,
                    fontweight="semibold",
                    color=TEXT_COLOR,
                    zorder=6,
                )

    axis.set_xlim(group_centers[0] - 0.62, group_centers[-1] + 0.56)
    axis.set_ylim(0.0, 1.12)
    axis.set_xticks(group_centers, [str(metric["label"]) for metric in metrics])
    axis.set_yticks([0.0, 0.5, 1.0])
    axis.set_yticklabels([])
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.65)
    axis.grid(axis="x", visible=False)
    axis.set_axisbelow(True)

    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color(AXIS_COLOR)
    axis.spines["bottom"].set_linewidth(0.65)
    axis.tick_params(axis="x", labelsize=6.3, colors=TEXT_COLOR, length=0, pad=5)
    axis.tick_params(axis="y", length=0, pad=0)

    legend_handles = [
        Patch(facecolor=color, edgecolor="white", linewidth=0.7, label=name)
        for name, color in zip(series_names, colors)
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=legend_ncol,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        fontsize=6.2,
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.8,
    )

    bottom = 0.19 if note else 0.14
    if note:
        figure.text(0.5, 0.025, note, ha="center", va="bottom", fontsize=5.9, color=MUTED_TEXT)
    figure.subplots_adjust(top=0.86, bottom=bottom, left=0.08, right=0.985)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(output_prefix.with_suffix(f".{suffix}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def draw_metric_grid_bars(
    *,
    metrics: Sequence[Mapping[str, object]],
    series_names: Sequence[str],
    colors: Sequence[str],
    output_prefix: Path,
    figsize: tuple[float, float],
    legend_ncol: int,
    annotate: bool = True,
) -> None:
    setup_plot_style()
    n_metrics = len(metrics)
    ncols = 2
    nrows = int(np.ceil(n_metrics / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=300)
    figure.patch.set_facecolor("white")
    axes_array = np.asarray(axes, dtype=object).reshape(-1)

    legend_handles = [
        Patch(facecolor=color, edgecolor="white", linewidth=0.7, label=name)
        for name, color in zip(series_names, colors)
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=legend_ncol,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=6.2,
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.9,
    )

    for axis, metric in zip(axes_array, metrics):
        axis.set_facecolor("white")
        raw_values = np.asarray(metric["values"], dtype=float)
        x_values = np.arange(len(series_names), dtype=float)
        y_min, y_max = metric["display_ylim"]
        y_span = max(float(y_max) - float(y_min), 1e-9)

        bars = axis.bar(
            x_values,
            raw_values,
            width=0.62,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        errors = metric.get("errors")
        if errors is not None:
            raw_errors = np.asarray(errors, dtype=float)
            for x_value, y_value, err in zip(x_values, raw_values, raw_errors):
                if not np.isfinite(err) or float(err) <= 0.0:
                    continue
                axis.errorbar(
                    x_value,
                    y_value,
                    yerr=float(err),
                    color=TEXT_COLOR,
                    capsize=1.8,
                    elinewidth=0.7,
                    capthick=0.7,
                    zorder=5,
                )

        if annotate:
            for bar, raw_value in zip(bars, raw_values):
                y_text = float(raw_value) + 0.035 * y_span
                va = "bottom"
                if y_text > float(y_max) - 0.01 * y_span:
                    y_text = float(raw_value) - 0.045 * y_span
                    va = "top"
                axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y_text,
                    metric.get("label_fmt", "{:.2f}").format(float(raw_value)),
                    ha="center",
                    va=va,
                    fontsize=5.6,
                    fontweight="semibold",
                    color=TEXT_COLOR,
                    zorder=6,
                )

        axis.set_title(str(metric["label"]), loc="left", fontsize=6.5, fontweight="semibold", pad=6.0)
        axis.set_ylim(float(y_min), float(y_max))
        axis.set_xlim(-0.55, len(series_names) - 0.45)
        axis.set_xticks([])
        tick_values = metric.get("tick_values")
        if tick_values is not None:
            axis.set_yticks([float(value) for value in tick_values])
        tick_fmt = str(metric.get("tick_fmt", "{:.2f}"))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _, fmt=tick_fmt: fmt.format(float(value))))
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.65)
        axis.grid(axis="x", visible=False)
        axis.set_axisbelow(True)

        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.spines["left"].set_color(AXIS_COLOR)
        axis.spines["bottom"].set_color(AXIS_COLOR)
        axis.spines["left"].set_linewidth(0.65)
        axis.spines["bottom"].set_linewidth(0.65)
        axis.tick_params(axis="y", labelsize=5.3, colors=MUTED_TEXT, length=0, pad=1.5)
        axis.tick_params(axis="x", length=0)

    for axis in axes_array[n_metrics:]:
        axis.axis("off")

    figure.subplots_adjust(top=0.78, bottom=0.08, left=0.13, right=0.985, wspace=0.28, hspace=0.60)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(output_prefix.with_suffix(f".{suffix}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
