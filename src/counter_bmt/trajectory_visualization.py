"""
Trajectory Visualization Module for CounterBMT

Provides visualization functions for comparing baseline vs counterfactual trajectories
and generating comprehensive scenario reports.

Author: CounterBMT Project
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

import numpy as np

logger = logging.getLogger(__name__)


# Attempt to import optional visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Polygon, Circle, Rectangle, FancyArrowPatch
    from matplotlib.collections import PatchCollection
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available - visualization will be limited")


# Color scheme for visualizations
COLORS = {
    'baseline': '#2196F3',      # Blue
    'counterfactual': '#FF5722', # Orange
    'ego_vehicle': '#4CAF50',    # Green
    'other_agents': '#9E9E9E',   # Gray
    'road': '#E0E0E0',           # Light gray
    'map_boundary': '#424242',   # Dark gray
    'collision': '#F44336',      # Red
    'safe': '#4CAF50',           # Green
}

# Intervention effect colors
EFFECT_COLORS = {
    'decrease': '#4CAF50',  # Green - desired reduction
    'increase': '#F44336',  # Red - undesired increase
    'none': '#FFC107',      # Yellow - no effect
}


@dataclass
class VisualizationConfig:
    """Configuration for trajectory visualizations."""
    figure_size: Tuple[int, int] = (12, 10)
    dpi: int = 150
    show_grid: bool = True
    show_legend: bool = True
    show_timestamps: bool = False
    trajectory_linewidth: float = 2.0
    marker_size: float = 8.0
    font_size: int = 10
    title_size: int = 14
    boundary_padding: float = 10.0


def visualize_trajectory_comparison(
    baseline: np.ndarray,
    counterfactuals: Dict[str, List[np.ndarray]],
    output_path: Path,
    scenario_id: str = "",
    config: Optional[VisualizationConfig] = None,
    map_data: Optional[Dict] = None,
    other_agents: Optional[np.ndarray] = None,
) -> bool:
    """
    Create visualization comparing baseline and counterfactual trajectories.
    
    Args:
        baseline: (T, 2) baseline trajectory
        counterfactuals: Dict mapping intervention names to lists of trajectory samples
        output_path: Path to save the visualization
        scenario_id: Scenario identifier for title
        config: Visualization configuration
        map_data: Optional map data for background
        other_agents: Optional (N, T, 2) positions of other agents
        
    Returns:
        True if successful
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib required for visualization")
        return False
    
    config = config or VisualizationConfig()
    baseline = np.asarray(baseline)
    
    # Calculate common bounds first to determine aspect ratio
    all_points = [baseline]
    for samples in counterfactuals.values():
        all_points.extend(samples)
    all_points = np.concatenate(all_points, axis=0)
    
    x_min, y_min = all_points.min(axis=0) - config.boundary_padding
    x_max, y_max = all_points.max(axis=0) + config.boundary_padding
    
    # Calculate data aspect ratio
    x_range = x_max - x_min
    y_range = y_max - y_min
    data_aspect = y_range / x_range if x_range > 0 else 1.0
    
    # Create figure with subplots - use 2 columns max for better visibility
    n_interventions = len(counterfactuals)
    n_cols = min(2, n_interventions + 1)
    n_rows = (n_interventions + n_cols) // n_cols
    
    # Calculate figure size based on data aspect ratio
    subplot_width = 6  # inches per subplot
    subplot_height = max(5, subplot_width * data_aspect)  # maintain aspect, min 5 inches
    fig_width = subplot_width * n_cols
    fig_height = subplot_height * n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), 
                             constrained_layout=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot overview in first subplot
    ax = axes.flat[0]
    _plot_trajectory(ax, baseline, COLORS['baseline'], 'Baseline', config, marker='o')
    
    # Plot all counterfactual samples with different colors
    # Use colormaps API (compatible with matplotlib 3.7+)
    try:
        cmap = plt.colormaps['Set2']
    except (AttributeError, KeyError):
        # Fallback for older matplotlib
        cmap = plt.cm.get_cmap('Set2')
    for idx, (name, samples) in enumerate(counterfactuals.items()):
        color = cmap(idx)
        for i, sample in enumerate(samples):
            label = name if i == 0 else None
            _plot_trajectory(ax, sample, color, label, config, alpha=0.6, marker='x')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title("All Trajectories Overview", fontsize=config.title_size)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    if config.show_legend:
        ax.legend(loc='best', fontsize=config.font_size - 2)
    if config.show_grid:
        ax.grid(True, alpha=0.3)
    # Use 'equal' aspect only if data is reasonably square, otherwise use 'auto'
    if 0.5 < data_aspect < 2.0:
        ax.set_aspect('equal', adjustable='box')
    else:
        ax.set_aspect('auto')
    
    # Plot individual interventions
    for idx, (name, samples) in enumerate(counterfactuals.items()):
        if idx + 1 >= len(axes.flat):
            break
        ax = axes.flat[idx + 1]
        
        # Plot baseline
        _plot_trajectory(ax, baseline, COLORS['baseline'], 'Baseline', config, marker='o')
        
        # Plot counterfactual samples
        for i, sample in enumerate(samples):
            alpha = 0.8 if i == 0 else 0.4
            label = f'CF Sample {i+1}' if i < 3 else None
            _plot_trajectory(ax, sample, COLORS['counterfactual'], label, config, 
                           alpha=alpha, marker='x')
        
        # Compute and display metrics
        baseline_dist = _compute_distance(baseline)
        mean_cf_dist = np.mean([_compute_distance(s) for s in samples])
        change = (mean_cf_dist - baseline_dist) / baseline_dist * 100
        
        effect_color = EFFECT_COLORS['decrease'] if change < -5 else (
            EFFECT_COLORS['increase'] if change > 5 else EFFECT_COLORS['none']
        )
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Title with metrics
        title = f"{name}\n"
        title += f"Distance: {baseline_dist:.1f}m → {mean_cf_dist:.1f}m ({change:+.1f}%)"
        ax.set_title(title, fontsize=config.title_size - 2, color=effect_color)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        if config.show_legend:
            ax.legend(loc='best', fontsize=config.font_size - 2)
        if config.show_grid:
            ax.grid(True, alpha=0.3)
        # Use 'equal' aspect only if data is reasonably square
        if 0.5 < data_aspect < 2.0:
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.set_aspect('auto')
    
    # Hide unused subplots
    for idx in range(n_interventions + 1, len(axes.flat)):
        axes.flat[idx].set_visible(False)
    
    fig.suptitle(f"Counterfactual Trajectory Comparison: {scenario_id}", 
                 fontsize=config.title_size + 2, fontweight='bold')
    
    # Note: using constrained_layout, so no tight_layout needed
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved trajectory comparison to {output_path}")
    return True


def visualize_single_comparison(
    baseline: np.ndarray,
    counterfactual: np.ndarray,
    intervention_name: str,
    output_path: Path,
    config: Optional[VisualizationConfig] = None,
    other_agents: Optional[np.ndarray] = None,
) -> bool:
    """
    Create detailed visualization for a single intervention comparison.
    
    Args:
        baseline: (T, 2) baseline trajectory
        counterfactual: (T, 2) counterfactual trajectory
        intervention_name: Name of the intervention
        output_path: Path to save the visualization
        config: Visualization configuration
        other_agents: Optional (N, T, 2) positions of other agents
        
    Returns:
        True if successful
    """
    if not HAS_MATPLOTLIB:
        return False
    
    config = config or VisualizationConfig()
    baseline = np.asarray(baseline)
    counterfactual = np.asarray(counterfactual)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Common bounds
    all_points = np.concatenate([baseline, counterfactual], axis=0)
    x_min, y_min = all_points.min(axis=0) - config.boundary_padding
    x_max, y_max = all_points.max(axis=0) + config.boundary_padding
    
    # Panel 1: Baseline trajectory
    ax1 = axes[0]
    _plot_trajectory(ax1, baseline, COLORS['baseline'], 'Baseline', config, marker='o')
    _add_start_end_markers(ax1, baseline, COLORS['baseline'])
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_title(f"Baseline\nDistance: {_compute_distance(baseline):.1f}m", 
                  fontsize=config.title_size)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Panel 2: Counterfactual trajectory
    ax2 = axes[1]
    _plot_trajectory(ax2, counterfactual, COLORS['counterfactual'], 'Counterfactual', 
                    config, marker='x')
    _add_start_end_markers(ax2, counterfactual, COLORS['counterfactual'])
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_title(f"Counterfactual: {intervention_name}\n"
                  f"Distance: {_compute_distance(counterfactual):.1f}m", 
                  fontsize=config.title_size)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Panel 3: Overlay comparison
    ax3 = axes[2]
    _plot_trajectory(ax3, baseline, COLORS['baseline'], 'Baseline', config, 
                    marker='o', alpha=0.7)
    _plot_trajectory(ax3, counterfactual, COLORS['counterfactual'], 'Counterfactual', 
                    config, marker='x', alpha=0.7)
    
    # Draw divergence arrows
    min_len = min(len(baseline), len(counterfactual))
    step = max(1, min_len // 10)
    for i in range(0, min_len, step):
        if np.linalg.norm(baseline[i] - counterfactual[i]) > 1.0:
            ax3.annotate('', xy=counterfactual[i], xytext=baseline[i],
                        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=0.5))
    
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    
    # Compute change
    baseline_dist = _compute_distance(baseline)
    cf_dist = _compute_distance(counterfactual)
    change = (cf_dist - baseline_dist) / baseline_dist * 100
    effect_color = EFFECT_COLORS['decrease'] if change < -5 else (
        EFFECT_COLORS['increase'] if change > 5 else EFFECT_COLORS['none']
    )
    
    ax3.set_title(f"Comparison\nChange: {change:+.1f}%", 
                  fontsize=config.title_size, color=effect_color)
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved single comparison to {output_path}")
    return True


def visualize_intervention_summary(
    results: Dict[str, Any],
    output_path: Path,
    config: Optional[VisualizationConfig] = None
) -> bool:
    """
    Create bar chart summary of intervention effects.
    
    Args:
        results: Results dictionary from pipeline
        output_path: Path to save the visualization
        config: Visualization configuration
        
    Returns:
        True if successful
    """
    if not HAS_MATPLOTLIB:
        return False
    
    config = config or VisualizationConfig()
    
    # Extract data
    interventions = []
    changes = []
    colors = []
    
    if 'generation_results' in results:
        gen_results = results['generation_results']
        cf_results = gen_results.get('counterfactuals', gen_results.get('counterfactual_results', {}))
        baseline_dist = gen_results.get('baseline_travel_distance', 0)
        
        # Handle list format (from pipeline)
        if isinstance(cf_results, list):
            for item in cf_results:
                intervention = item.get('intervention', {})
                name = intervention.get('description') or f"do({intervention.get('variable', '?')})"
                comparison = item.get('comparison', {})
                if comparison:
                    cf_dist = comparison.get('counterfactual_travel', 0)
                    if baseline_dist and baseline_dist > 0:
                        change = (cf_dist - baseline_dist) / baseline_dist * 100
                    else:
                        change = 0
                    interventions.append(name[:20])
                    changes.append(change)
                    colors.append(EFFECT_COLORS['decrease'] if change < -5 else (
                        EFFECT_COLORS['increase'] if change > 5 else EFFECT_COLORS['none']
                    ))
        # Handle dict format
        elif isinstance(cf_results, dict):
            for name, data in cf_results.items():
                if isinstance(data, dict) and 'mean_travel_distance' in data:
                    interventions.append(name[:20])  # Truncate long names
                    cf_dist = data['mean_travel_distance']
                    change = (cf_dist - baseline_dist) / baseline_dist * 100 if baseline_dist > 0 else 0
                    changes.append(change)
                    colors.append(EFFECT_COLORS['decrease'] if change < -5 else (
                        EFFECT_COLORS['increase'] if change > 5 else EFFECT_COLORS['none']
                    ))
    
    if not interventions:
        logger.warning("No intervention data to visualize")
        return False
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(8, len(interventions) * 1.2), 6))
    
    x = np.arange(len(interventions))
    bars = ax.bar(x, changes, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, change in zip(bars, changes):
        height = bar.get_height()
        ax.annotate(f'{change:+.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3 if height >= 0 else -15),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=config.font_size)
    
    # Reference line at 0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Threshold lines
    ax.axhline(y=-5, color=EFFECT_COLORS['decrease'], linestyle='--', 
               linewidth=1, alpha=0.5, label='Effective threshold')
    ax.axhline(y=5, color=EFFECT_COLORS['increase'], linestyle='--', 
               linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Intervention', fontsize=config.font_size + 2)
    ax.set_ylabel('Travel Distance Change (%)', fontsize=config.font_size + 2)
    ax.set_title('Intervention Effects on Trajectory', fontsize=config.title_size)
    ax.set_xticks(x)
    ax.set_xticklabels(interventions, rotation=45, ha='right', fontsize=config.font_size - 1)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=EFFECT_COLORS['decrease'], label='Effective (↓)'),
        mpatches.Patch(facecolor=EFFECT_COLORS['increase'], label='Increased (↑)'),
        mpatches.Patch(facecolor=EFFECT_COLORS['none'], label='No effect'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved intervention summary to {output_path}")
    return True


def create_scenario_report(
    scenario_id: str,
    output_dir: Path,
    baseline_trajectory: np.ndarray,
    counterfactual_results: Dict[str, Any],
    dag_data: Optional[Dict] = None,
    llm_logs: Optional[List[Dict]] = None,
    config: Optional[VisualizationConfig] = None
) -> Path:
    """
    Create comprehensive HTML report for a scenario.
    
    Args:
        scenario_id: Scenario identifier
        output_dir: Directory to save report
        baseline_trajectory: Baseline trajectory
        counterfactual_results: Counterfactual generation results
        dag_data: Optional DAG structure data
        llm_logs: Optional LLM interaction logs
        config: Visualization configuration
        
    Returns:
        Path to generated report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations only if they don't already exist
    # (they are typically generated earlier in the pipeline)
    if HAS_MATPLOTLIB:
        # Trajectory comparison
        traj_viz_path = output_dir / "trajectory_comparison.png"
        if not traj_viz_path.exists():
            cf_data = counterfactual_results.get('counterfactuals', 
                                                  counterfactual_results.get('counterfactual_results', {}))
            cf_trajs = {}
            
            # Handle list format (from pipeline)
            if isinstance(cf_data, list):
                for item in cf_data:
                    intervention = item.get('intervention', {})
                    name = intervention.get('description') or f"do({intervention.get('variable', 'unknown')})"
                    trajs = item.get('trajectories', [])
                    if trajs:
                        cf_trajs[name] = [np.array(t) for t in trajs]
            # Handle dict format
            elif isinstance(cf_data, dict):
                for name, data in cf_data.items():
                    if isinstance(data, dict) and 'trajectories' in data:
                        cf_trajs[name] = [np.array(t) for t in data['trajectories']]
            
            if cf_trajs:
                visualize_trajectory_comparison(
                    baseline_trajectory, cf_trajs, traj_viz_path, scenario_id, config
                )
        
        # Summary chart - skip if already exists
        summary_viz_path = output_dir / "intervention_summary.png"
        if not summary_viz_path.exists():
            visualize_intervention_summary(
                {'generation_results': counterfactual_results}, 
                summary_viz_path, config
            )
    
    # Generate HTML report
    report_path = output_dir / "scenario_report.html"
    html_content = _generate_html_report(
        scenario_id=scenario_id,
        counterfactual_results=counterfactual_results,
        dag_data=dag_data,
        llm_logs=llm_logs,
        has_images=HAS_MATPLOTLIB
    )
    
    report_path.write_text(html_content)
    logger.info(f"Generated scenario report at {report_path}")
    
    return report_path


# ==================== Private Helper Functions ====================

def _plot_trajectory(
    ax,
    trajectory: np.ndarray,
    color: str,
    label: Optional[str],
    config: VisualizationConfig,
    marker: str = 'o',
    alpha: float = 1.0
):
    """Plot a single trajectory on an axis."""
    ax.plot(trajectory[:, 0], trajectory[:, 1], 
            color=color, linewidth=config.trajectory_linewidth,
            label=label, alpha=alpha)
    # Start and end markers
    ax.scatter(trajectory[0, 0], trajectory[0, 1], 
              color=color, marker='s', s=config.marker_size * 15, 
              zorder=5, alpha=alpha)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], 
              color=color, marker=marker, s=config.marker_size * 10, 
              zorder=5, alpha=alpha)


def _add_start_end_markers(ax, trajectory: np.ndarray, color: str):
    """Add labeled start/end markers to trajectory."""
    ax.annotate('Start', xy=(trajectory[0, 0], trajectory[0, 1]),
               xytext=(5, 5), textcoords='offset points',
               fontsize=8, color=color)
    ax.annotate('End', xy=(trajectory[-1, 0], trajectory[-1, 1]),
               xytext=(5, 5), textcoords='offset points',
               fontsize=8, color=color)


def _compute_distance(trajectory: np.ndarray) -> float:
    """Compute total travel distance."""
    if len(trajectory) < 2:
        return 0.0
    diffs = np.diff(trajectory, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))


def _generate_html_report(
    scenario_id: str,
    counterfactual_results: Dict,
    dag_data: Optional[Dict],
    llm_logs: Optional[List[Dict]],
    has_images: bool
) -> str:
    """Generate HTML content for scenario report."""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>CounterBMT Report: {scenario_id}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #1976D2; border-bottom: 3px solid #1976D2; padding-bottom: 10px; }}
        h2 {{ color: #424242; margin-top: 30px; }}
        h3 {{ color: #616161; }}
        .metric {{ display: inline-block; padding: 10px 20px; margin: 5px; background: #E3F2FD; border-radius: 4px; }}
        .metric-label {{ font-size: 12px; color: #757575; }}
        .metric-value {{ font-size: 18px; font-weight: bold; color: #1976D2; }}
        .decrease {{ background: #E8F5E9; }}
        .decrease .metric-value {{ color: #4CAF50; }}
        .increase {{ background: #FFEBEE; }}
        .increase .metric-value {{ color: #F44336; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #E0E0E0; }}
        th {{ background: #F5F5F5; font-weight: 600; }}
        .log-entry {{ background: #FAFAFA; padding: 15px; margin: 10px 0; border-radius: 4px; font-family: monospace; font-size: 12px; white-space: pre-wrap; overflow-x: auto; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #E0E0E0; border-radius: 4px; }}
        .section {{ margin: 30px 0; padding: 20px; background: #FAFAFA; border-radius: 8px; }}
        .dag-node {{ display: inline-block; padding: 8px 12px; margin: 4px; border-radius: 4px; font-size: 12px; }}
        .layer-0 {{ background: #BBDEFB; }}
        .layer-1 {{ background: #C8E6C9; }}
        .layer-2 {{ background: #FFE0B2; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗 CounterBMT Scenario Report</h1>
        <p><strong>Scenario ID:</strong> {scenario_id}</p>
"""
    
    # Summary metrics
    if counterfactual_results:
        baseline_dist = counterfactual_results.get('baseline_travel_distance', None)
        baseline_dist_str = f"{baseline_dist:.1f}m" if isinstance(baseline_dist, (int, float)) else "N/A"
        html += f"""
        <div class="section">
            <h2>📊 Summary</h2>
            <div class="metric">
                <div class="metric-label">Baseline Distance</div>
                <div class="metric-value">{baseline_dist_str}</div>
            </div>
"""
        
        # Add intervention metrics - handle both dict format and list format
        cf_data = counterfactual_results.get('counterfactuals', counterfactual_results.get('counterfactual_results', {}))
        
        # Handle list format (from pipeline)
        if isinstance(cf_data, list):
            for cf_item in cf_data:
                name = cf_item.get('intervention', {}).get('description', 'Unknown')[:25]
                comparison = cf_item.get('comparison', {})
                if comparison:
                    mean_dist = comparison.get('counterfactual_travel', 0)
                    if baseline_dist and isinstance(baseline_dist, (int, float)) and baseline_dist > 0:
                        reduction = (1 - mean_dist / baseline_dist) * 100
                    else:
                        reduction = 0
                    css_class = 'decrease' if reduction > 5 else ('increase' if reduction < -5 else '')
                    mean_dist_val = mean_dist if isinstance(mean_dist, (int, float)) else 0
                    html += f"""
            <div class="metric {css_class}">
                <div class="metric-label">{name}</div>
                <div class="metric-value">{mean_dist_val:.1f}m ({reduction:+.1f}%)</div>
            </div>
"""
        # Handle dict format
        elif isinstance(cf_data, dict):
            for name, data in cf_data.items():
                if isinstance(data, dict):
                    mean_dist = data.get('mean_travel_distance', 0)
                    reduction = data.get('reduction_percent', 0)
                    # Ensure numeric types
                    mean_dist = float(mean_dist) if mean_dist else 0
                    reduction = float(reduction) if reduction else 0
                    css_class = 'decrease' if reduction > 5 else ('increase' if reduction < -5 else '')
                    html += f"""
            <div class="metric {css_class}">
                <div class="metric-label">{name[:25]}</div>
                <div class="metric-value">{mean_dist:.1f}m ({reduction:+.1f}%)</div>
            </div>
"""
        html += "</div>"
    
    # Images
    if has_images:
        html += """
        <div class="section">
            <h2>📈 Visualizations</h2>
            <img src="trajectory_comparison.png" alt="Trajectory Comparison">
            <img src="intervention_summary.png" alt="Intervention Summary">
        </div>
"""
    
    # DAG Information
    if dag_data:
        html += """
        <div class="section">
            <h2>🔗 Causal DAG</h2>
"""
        if 'nodes' in dag_data:
            html += "<h3>Nodes</h3>"
            for layer in [0, 1, 2]:
                layer_nodes = [n for n in dag_data['nodes'] if n.get('layer') == layer]
                if layer_nodes:
                    html += f"<p><strong>Layer {layer}:</strong> "
                    for node in layer_nodes:
                        html += f'<span class="dag-node layer-{layer}">{node.get("name", "")}</span>'
                    html += "</p>"
        
        if 'edges' in dag_data:
            html += "<h3>Edges</h3><table><tr><th>From</th><th>To</th><th>Confidence</th></tr>"
            for edge in dag_data['edges'][:20]:  # Limit to 20 edges
                html += f"<tr><td>{edge.get('parent', '')}</td><td>{edge.get('child', '')}</td><td>{edge.get('confidence', 0):.2f}</td></tr>"
            html += "</table>"
        
        html += "</div>"
    
    # LLM Logs
    if llm_logs:
        html += """
        <div class="section">
            <h2>🤖 LLM Interaction Logs</h2>
"""
        for i, log in enumerate(llm_logs[:10]):  # Limit to 10 logs
            html += f"""
            <h3>Query {i+1}: {log.get('type', 'unknown')}</h3>
            <div class="log-entry">
<strong>Input:</strong>
{log.get('input', '')[:500]}...

<strong>Output:</strong>
{log.get('output', '')[:500]}...
            </div>
"""
        html += "</div>"
    
    # Detailed results table
    cf_data = counterfactual_results.get('counterfactuals', counterfactual_results.get('counterfactual_results', [])) if counterfactual_results else []
    if cf_data:
        html += """
        <div class="section">
            <h2>📋 Detailed Results</h2>
            <table>
                <tr>
                    <th>Intervention</th>
                    <th>Mean Distance</th>
                    <th>Change</th>
                    <th>Samples</th>
                </tr>
"""
        baseline_dist = counterfactual_results.get('baseline_travel_distance', 0) if counterfactual_results else 0
        baseline_dist = float(baseline_dist) if isinstance(baseline_dist, (int, float)) else 0
        
        # Handle list format (from pipeline)
        if isinstance(cf_data, list):
            for cf_item in cf_data:
                name = cf_item.get('intervention', {}).get('description', 'Unknown')
                comparison = cf_item.get('comparison', {})
                n_samples = cf_item.get('n_samples', len(cf_item.get('trajectories', [])))
                if comparison:
                    mean_dist = float(comparison.get('counterfactual_travel', 0) or 0)
                    if baseline_dist > 0:
                        reduction = (1 - mean_dist / baseline_dist) * 100
                    else:
                        reduction = 0
                else:
                    mean_dist = 0
                    reduction = 0
                html += f"<tr><td>{name}</td><td>{mean_dist:.1f}m</td><td>{reduction:+.1f}%</td><td>{n_samples}</td></tr>"
        # Handle dict format
        elif isinstance(cf_data, dict):
            for name, data in cf_data.items():
                if isinstance(data, dict):
                    mean_dist = float(data.get('mean_travel_distance', 0) or 0)
                    reduction = float(data.get('reduction_percent', 0) or 0)
                    n_samples = data.get('n_samples', 0)
                    html += f"<tr><td>{name}</td><td>{mean_dist:.1f}m</td><td>{reduction:+.1f}%</td><td>{n_samples}</td></tr>"
        html += "</table></div>"
    
    html += """
    </div>
</body>
</html>
"""
    
    return html

