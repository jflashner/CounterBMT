from pathlib import Path

from counterdrive_plot_style import (
    COMPARATOR_ORANGE,
    COUNTERDRIVE_BLUE,
    METHOD_GRAYS,
    draw_metric_group_bars,
)


METHODS = ["Waymo", "CAT", "STRIVE", "SEAL", "Adv-BMT", "Adv-BMT Ref.", "CounterDrive"]
COLORS = [
    METHOD_GRAYS[0],
    METHOD_GRAYS[1],
    METHOD_GRAYS[2],
    METHOD_GRAYS[3],
    COMPARATOR_ORANGE,
    "#F59E0B",
    COUNTERDRIVE_BLUE,
]

METRICS = [
    {
        "label": "Reward $\\uparrow$",
        "values": [32.03, 30.37, 31.30, 29.94, 31.47, 33.22, 34.81],
        "errors": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.28],
        "display_ylim": (28.5, 37.0),
        "tick_values": [29, 33, 37],
        "tick_fmt": "{:.0f}",
    },
    {
        "label": "Cost $\\downarrow$",
        "values": [0.39, 0.39, 0.40, 0.39, 0.38, 0.36, 0.304],
        "errors": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.010],
        "display_ylim": (0.28, 0.42),
        "tick_values": [0.28, 0.35, 0.42],
        "tick_fmt": "{:.2f}",
    },
    {
        "label": "Completion $\\uparrow$",
        "values": [0.72, 0.71, 0.73, 0.71, 0.73, 0.74, 0.746],
        "errors": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.027],
        "display_ylim": (0.66, 0.80),
        "tick_values": [0.66, 0.73, 0.80],
        "tick_fmt": "{:.2f}",
    },
    {
        "label": "Collision $\\downarrow$",
        "values": [0.14, 0.14, 0.13, 0.12, 0.11, 0.12, 0.066],
        "errors": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0215],
        "display_ylim": (0.04, 0.16),
        "tick_values": [0.04, 0.10, 0.16],
        "tick_fmt": "{:.2f}",
    },
]


def main():
    output_dir = Path(__file__).resolve().parent
    draw_metric_group_bars(
        metrics=METRICS,
        series_names=METHODS,
        colors=COLORS,
        output_prefix=output_dir / "downstream_procedure_comparison",
        figsize=(7.1, 2.35),
        legend_ncol=7,
        annotate=False,
        note="Each metric group uses its own scale; CounterDrive error bars show random-100 evaluation standard deviation.",
    )


if __name__ == "__main__":
    main()
