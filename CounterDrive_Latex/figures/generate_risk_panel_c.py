from pathlib import Path

from counterdrive_plot_style import RISK_AMBER, RISK_GREEN, RISK_ROSE, draw_metric_grid_bars


RISK_LEVELS = ["Low", "Medium", "High"]
RISK_COLORS = [RISK_GREEN, RISK_AMBER, RISK_ROSE]

METRICS = [
    {
        "label": "Reward $\\uparrow$",
        "values": [42.74, 48.50, 46.18],
        "display_ylim": (41.0, 49.5),
        "tick_values": [42, 46, 49],
        "tick_fmt": "{:.0f}",
        "label_fmt": "{:.1f}",
    },
    {
        "label": "Completion $\\uparrow$",
        "values": [0.706, 0.781, 0.737],
        "display_ylim": (0.69, 0.80),
        "tick_values": [0.70, 0.75, 0.80],
        "tick_fmt": "{:.2f}",
        "label_fmt": "{:.2f}",
    },
    {
        "label": "Cost $\\downarrow$",
        "values": [0.54, 0.39, 0.39],
        "display_ylim": (0.36, 0.56),
        "tick_values": [0.38, 0.47, 0.56],
        "tick_fmt": "{:.2f}",
        "label_fmt": "{:.2f}",
    },
    {
        "label": "Crash $\\downarrow$",
        "values": [0.05, 0.04, 0.02],
        "display_ylim": (0.015, 0.055),
        "tick_values": [0.02, 0.04, 0.055],
        "tick_fmt": "{:.2f}",
        "label_fmt": "{:.2f}",
    },
]


def main():
    output_dir = Path(__file__).resolve().parent
    draw_metric_grid_bars(
        metrics=METRICS,
        series_names=RISK_LEVELS,
        colors=RISK_COLORS,
        output_prefix=output_dir / "risk_ablation_4panel",
        figsize=(3.65, 2.6),
        legend_ncol=3,
        annotate=True,
    )


if __name__ == "__main__":
    main()
