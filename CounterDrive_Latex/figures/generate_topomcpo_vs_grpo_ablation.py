from pathlib import Path

from counterdrive_plot_style import COMPARATOR_ORANGE, COUNTERDRIVE_BLUE, draw_metric_grid_bars


METHODS = ["GRPO", "Topo-MCPO"]
COLORS = [COMPARATOR_ORANGE, COUNTERDRIVE_BLUE]

METRICS = [
    {
        "label": "SADE $\\downarrow$",
        "values": [3.86, 3.02],
        "display_ylim": (2.8, 4.05),
        "tick_values": [2.8, 3.4, 4.0],
        "tick_fmt": "{:.1f}",
        "label_fmt": "{:.2f}",
    },
    {
        "label": "SFDE $\\downarrow$",
        "values": [8.40, 7.00],
        "display_ylim": (6.6, 8.7),
        "tick_values": [6.8, 7.6, 8.6],
        "tick_fmt": "{:.1f}",
        "label_fmt": "{:.2f}",
    },
    {
        "label": "ADD $\\uparrow$",
        "values": [39.44, 41.65],
        "display_ylim": (38.5, 42.2),
        "tick_values": [39, 40.5, 42],
        "tick_fmt": "{:.0f}",
        "label_fmt": "{:.1f}",
    },
    {
        "label": "FDD $\\uparrow$",
        "values": [85.68, 90.84],
        "display_ylim": (84.0, 92.0),
        "tick_values": [84, 88, 92],
        "tick_fmt": "{:.0f}",
        "label_fmt": "{:.1f}",
    },
]


def main():
    output_dir = Path(__file__).resolve().parent
    draw_metric_grid_bars(
        metrics=METRICS,
        series_names=METHODS,
        colors=COLORS,
        output_prefix=output_dir / "topomcpo_vs_grpo_ablation",
        figsize=(3.65, 2.6),
        legend_ncol=2,
        annotate=True,
    )


if __name__ == "__main__":
    main()
