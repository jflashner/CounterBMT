"""
DAG Visualization Utilities

Provides visualization and export functions for ScenarioDAG objects.

Usage:
    from dag_visualization import visualize_dag, export_dag_to_dot
    
    visualize_dag(dag, Path("output.png"))
    export_dag_to_dot(dag, Path("output.dot"))

Author: CounterBMT Project
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def visualize_dag(dag, output_path: Path, title: str = None) -> bool:
    """
    Visualize DAG with layer structure.
    
    Args:
        dag: ScenarioDAG object
        output_path: Path to save PNG
        title: Optional title
        
    Returns:
        True if successful
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import networkx as nx
    except ImportError:
        logger.warning("matplotlib/networkx not available for visualization")
        return False
    
    G = nx.DiGraph()
    
    # Colors by layer and type
    layer_colors = {
        0: {'ego_state': '#87CEEB', 'agent_state': '#DDA0DD', 'environmental': '#90EE90'},
        1: {'maneuver': '#4ECDC4', 'decision': '#FF6B6B', 'interaction': '#FFE66D'},
        2: {'outcome': '#FF8C00', 'severity': '#DC143C'}
    }
    
    colors = []
    labels = {}
    
    for node_id, node in dag.nodes.items():
        G.add_node(node_id)
        layer_map = layer_colors.get(node.layer, {})
        color = layer_map.get(node.node_type.value, '#D3D3D3')
        colors.append(color)
        
        # Compact label
        name = node.name.replace("Ego Initial ", "Ego ").replace("Decision: ", "D:")
        name = name.replace("Maneuver: ", "M:").replace("Collision ", "")
        if node.value is not None:
            if isinstance(node.value, dict):
                if 'x' in node.value and 'y' in node.value:
                    val_str = f"({node.value['x']:.0f},{node.value['y']:.0f})"
                else:
                    val_str = str(node.value)[:15]
            elif isinstance(node.value, float):
                val_str = f"{node.value:.1f}"
            else:
                val_str = str(node.value)[:15]
            labels[node_id] = f"{name}\n[{val_str}]"
        else:
            labels[node_id] = name
    
    for edge in dag.edges:
        G.add_edge(edge.parent_id, edge.child_id)
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Position by layer
    layer_nodes = {0: [], 1: [], 2: []}
    for node_id, node in dag.nodes.items():
        layer_nodes[node.layer].append(node_id)
    
    pos = {}
    y_positions = {0: 4, 1: 2, 2: 0}
    
    for layer, nodes in layer_nodes.items():
        n = len(nodes)
        for i, node_id in enumerate(nodes):
            x = (i - (n - 1) / 2) * 2.5
            pos[node_id] = (x, y_positions[layer])
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=3000, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='#555555', arrows=True,
                           arrowsize=15, arrowstyle='-|>', ax=ax,
                           connectionstyle="arc3,rad=0.1", width=1.2)
    
    # Edge labels (confidence)
    edge_labels = {(e.parent_id, e.child_id): f"{e.confidence:.1f}" for e in dag.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, ax=ax)
    
    # Layer labels
    ax.text(-8, 4, "Layer 0\n(Initial States)", fontsize=10, fontweight='bold', va='center')
    ax.text(-8, 2, "Layer 1\n(Events)", fontsize=10, fontweight='bold', va='center')
    ax.text(-8, 0, "Layer 2\n(Outcome)", fontsize=10, fontweight='bold', va='center')
    
    # Legend
    legend_items = [
        ('Ego State', '#87CEEB'), ('Agent State', '#DDA0DD'),
        ('Maneuver', '#4ECDC4'), ('Decision', '#FF6B6B'), ('Outcome', '#FF8C00')
    ]
    patches = [mpatches.Patch(color=c, label=l) for l, c in legend_items]
    ax.legend(handles=patches, loc='upper right', fontsize=8)
    
    ax.set_title(title or f"Grounded Causal DAG: {dag.scenario_id}", fontsize=14, fontweight='bold')
    ax.axis('off')
    ax.set_xlim(-10, 10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved visualization to {output_path}")
    return True


def export_dag_to_dot(dag, output_path: Path) -> bool:
    """
    Export DAG to DOT format for Graphviz.
    
    Args:
        dag: ScenarioDAG object
        output_path: Path to save .dot file
        
    Returns:
        True if successful
    """
    layer_colors = {
        0: {"ego_state": "skyblue", "agent_state": "plum", "environmental": "lightgreen"},
        1: {"maneuver": "turquoise", "decision": "coral", "interaction": "yellow"},
        2: {"outcome": "orange", "severity": "crimson"}
    }
    
    lines = [
        "digraph GroundedDAG {",
        "    rankdir=TB;",
        "    node [shape=box, style=filled];",
        "",
        "    // Layer 0: Initial States",
        "    subgraph cluster_0 {",
        '        label="Layer 0: Initial States";',
        "        style=dashed;",
    ]
    
    for node in dag.get_nodes_by_layer(0):
        color = layer_colors[0].get(node.node_type.value, "white")
        val_str = str(node.value)[:20] if node.value else ""
        label = f"{node.name}\\n{val_str}" if val_str else node.name
        lines.append(f'        "{node.id}" [label="{label}", fillcolor={color}];')
    
    lines.extend([
        "    }",
        "",
        "    // Layer 1: Events",
        "    subgraph cluster_1 {",
        '        label="Layer 1: Events";',
        "        style=dashed;",
    ])
    
    for node in dag.get_nodes_by_layer(1):
        color = layer_colors[1].get(node.node_type.value, "white")
        label = f"{node.name}\\n{node.value}" if node.value else node.name
        lines.append(f'        "{node.id}" [label="{label}", fillcolor={color}];')
    
    lines.extend([
        "    }",
        "",
        "    // Layer 2: Outcome",
        "    subgraph cluster_2 {",
        '        label="Layer 2: Outcome";',
        "        style=dashed;",
    ])
    
    for node in dag.get_nodes_by_layer(2):
        color = layer_colors[2].get(node.node_type.value, "white")
        lines.append(f'        "{node.id}" [label="{node.name}", fillcolor={color}];')
    
    lines.extend(["    }", ""])
    
    for edge in dag.edges:
        lines.append(f'    "{edge.parent_id}" -> "{edge.child_id}" [label="{edge.confidence:.2f}"];')
    
    lines.append("}")
    output_path.write_text('\n'.join(lines))
    logger.info(f"Exported to DOT: {output_path}")
    return True


def print_dag_summary(dag) -> str:
    """
    Print detailed DAG summary to console.
    
    Args:
        dag: ScenarioDAG object
        
    Returns:
        Summary string
    """
    lines = [
        "=" * 60,
        f"DAG Summary: {dag.scenario_id}",
        "=" * 60,
        f"Total Nodes: {len(dag.nodes)}",
        f"Total Edges: {len(dag.edges)}",
        ""
    ]
    
    for layer in [0, 1, 2]:
        layer_names = {0: "Initial States", 1: "Events", 2: "Outcome"}
        nodes = dag.get_nodes_by_layer(layer)
        lines.append(f"Layer {layer} ({layer_names[layer]}): {len(nodes)} nodes")
        for n in nodes:
            val_str = str(n.value)[:40] if n.value else "None"
            lines.append(f"  - {n.id}: {n.name} = {val_str}")
        lines.append("")
    
    lines.append(f"Edges ({len(dag.edges)}):")
    for e in dag.edges:
        lines.append(f"  {e.parent_id} -> {e.child_id} ({e.confidence:.2f}): {e.mechanism[:50]}")
    
    summary = "\n".join(lines)
    print(summary)
    return summary