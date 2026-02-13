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
import json
import html
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _format_alternatives(alternatives, max_items: int = 4) -> Optional[str]:
    if not alternatives:
        return None
    alt_strs = [str(a) for a in alternatives]
    shown = alt_strs[:max_items]
    suffix = f", +{len(alt_strs) - max_items} more" if len(alt_strs) > max_items else ""
    return ", ".join(shown) + suffix


def _build_node_label(node) -> str:
    name = node.name.replace("Ego Initial ", "Ego ").replace("Decision: ", "D:")
    name = name.replace("Maneuver: ", "M:").replace("Collision ", "")
    lines = [name]

    if node.value is not None:
        if isinstance(node.value, dict):
            if "x" in node.value and "y" in node.value:
                val_str = f"({node.value['x']:.0f},{node.value['y']:.0f})"
            else:
                val_str = str(node.value)[:30]
        elif isinstance(node.value, float):
            val_str = f"{node.value:.2f}"
        else:
            val_str = str(node.value)[:30]
        lines.append(f"[{val_str}]")

    confidence = node.metadata.get("confidence")
    if isinstance(confidence, (int, float)):
        lines.append(f"p={confidence:.2f}")

    alts = _format_alternatives(node.metadata.get("alternatives"))
    if alts:
        lines.append(f"alts: {alts}")

    return "\n".join(lines)


def _compute_node_size(label: str) -> int:
    # Rough size heuristic based on label length
    return max(1800, min(5200, 1600 + len(label) * 35))


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
    sizes = []
    labels = {}
    
    for node_id, node in dag.nodes.items():
        G.add_node(node_id)
        layer_map = layer_colors.get(node.layer, {})
        color = layer_map.get(node.node_type.value, '#D3D3D3')
        colors.append(color)
        
        label = _build_node_label(node)
        labels[node_id] = label
        sizes.append(_compute_node_size(label))
    
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
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='#555555', arrows=True,
                           arrowsize=15, arrowstyle='-|>', ax=ax,
                           connectionstyle="arc3,rad=0.1", width=1.2)
    
    # Edge labels — skip confidence when CPTs are present
    has_cpts = any(node.metadata.get("cpt") for node in dag.nodes.values())
    if not has_cpts:
        edge_labels = {(e.parent_id, e.child_id): f"p={e.confidence:.2f}" for e in dag.edges}
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
        label = _build_node_label(node).replace("\n", "\\n")
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
        label = _build_node_label(node).replace("\n", "\\n")
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
        label = _build_node_label(node).replace("\n", "\\n")
        lines.append(f'        "{node.id}" [label="{label}", fillcolor={color}];')
    
    lines.extend(["    }", ""])
    
    has_cpts = any(node.metadata.get("cpt") for node in dag.nodes.values())
    for edge in dag.edges:
        if has_cpts:
            lines.append(f'    "{edge.parent_id}" -> "{edge.child_id}";')
        else:
            lines.append(f'    "{edge.parent_id}" -> "{edge.child_id}" [label="p={edge.confidence:.2f}"];')
    
    lines.append("}")
    output_path.write_text('\n'.join(lines))
    logger.info(f"Exported to DOT: {output_path}")
    return True


def _build_cpt_html_table(cpt: dict) -> str:
    """Build a formatted HTML table from a CPT dict."""
    values = cpt.get("values", [])
    parents = cpt.get("parents", [])
    table = cpt.get("cpt", {})
    observed_context = cpt.get("observed_context", "")

    if not values or not table:
        return ""

    parts = []

    # Show observed continuous context if present
    if observed_context:
        parts.append(
            f"<div style='font-size:11px;color:#555;margin-bottom:4px'>"
            f"<b>Conditioned on:</b> {html.escape(str(observed_context))}</div>"
        )

    rows = []
    # Header row
    th_style = "padding:3px 8px;border:1px solid #999;background:#e8e8e8"
    header_cells = "".join(
        f"<th style='{th_style}'>{html.escape(str(v))}</th>" for v in values
    )
    parent_header = "Parents" if parents else "Distribution"
    rows.append(
        f"<tr><th style='padding:3px 8px;border:1px solid #999;background:#ddd'>"
        f"{parent_header}</th>{header_cells}</tr>"
    )

    # Data rows
    for key, probs in table.items():
        if key == "*":
            label = "<i>marginal</i>" if not parents else "<i>all</i>"
        else:
            label = html.escape(str(key))
        prob_cells = "".join(
            f"<td style='padding:3px 8px;border:1px solid #ccc;text-align:center'>"
            f"{probs.get(str(v), '—')}</td>"
            for v in values
        )
        rows.append(
            f"<tr><td style='padding:3px 8px;border:1px solid #ccc;font-size:11px'>"
            f"{label}</td>{prob_cells}</tr>"
        )

    parts.append(
        f"<table style='border-collapse:collapse;font-size:12px;margin-top:4px'>"
        f"{''.join(rows)}</table>"
    )

    return "".join(parts)


def export_dag_to_html(dag, output_path: Path, title: str = None) -> bool:
    """
    Export DAG to interactive HTML using vis.js directly (no PyVis).
    Tooltips render as proper HTML with CPT tables.
    """
    title = title or f"Grounded DAG: {dag.scenario_id}"

    layer_colors = {
        0: {'ego_state': '#87CEEB', 'agent_state': '#DDA0DD', 'environmental': '#90EE90'},
        1: {'maneuver': '#4ECDC4', 'decision': '#FF6B6B', 'interaction': '#FFE66D'},
        2: {'outcome': '#FF8C00', 'severity': '#DC143C'}
    }

    # Build tooltip HTML per node (stored in a JS dict, rendered via DOM)
    tooltip_map = {}
    nodes_js = []
    for node_id, node in dag.nodes.items():
        label = _build_node_label(node)
        color = layer_colors.get(node.layer, {}).get(node.node_type.value, '#D3D3D3')
        size = _compute_node_size(label) / 120

        # Build tooltip content
        label_html = html.escape(label).replace("\n", "<br>")
        tooltip_parts = [f"<b>{label_html}</b>"]
        cpt = node.metadata.get("cpt")
        if cpt and isinstance(cpt, dict):
            cpt_table = _build_cpt_html_table(cpt)
            if cpt_table:
                tooltip_parts.append(f"<br><b>CPT</b>{cpt_table}")
            else:
                note = cpt.get("note", "")
                if note:
                    tooltip_parts.append(f"<br><i>{html.escape(note)}</i>")

        tooltip_map[node_id] = "".join(tooltip_parts)

        node_json = json.dumps({
            "id": node_id,
            "label": label,
            "color": color,
            "shape": "box",
            "size": size,
            "font": {"size": 13, "multi": False},
        })
        nodes_js.append(node_json)

    has_cpts = any(node.metadata.get("cpt") for node in dag.nodes.values())

    edges_js = []
    for edge in dag.edges:
        edge_obj = {
            "from": edge.parent_id,
            "to": edge.child_id,
            "arrows": "to",
            "font": {"size": 11, "align": "top"},
        }
        if has_cpts:
            # With CPTs, edge confidence is redundant; show mechanism only
            if edge.mechanism:
                edge_obj["title"] = edge.mechanism
        else:
            edge_obj["label"] = f"p={edge.confidence:.2f}"
        edges_js.append(json.dumps(edge_obj))

    tooltip_map_json = json.dumps(tooltip_map)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" />
<style>
body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
h1 {{ text-align: center; padding: 12px 0 4px; margin: 0; font-size: 18px; }}
#network {{ width: 100%; height: calc(100vh - 50px); border: 1px solid #ccc; }}
#tooltip {{
  display: none;
  position: absolute;
  background: #fff;
  border: 1px solid #888;
  border-radius: 6px;
  padding: 10px 14px;
  box-shadow: 2px 2px 8px rgba(0,0,0,0.25);
  max-width: 600px;
  max-height: 400px;
  overflow-y: auto;
  z-index: 9999;
  font-size: 13px;
  line-height: 1.5;
  pointer-events: auto;
}}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div id="network"></div>
<div id="tooltip"></div>
<script>
var tooltipMap = {tooltip_map_json};
var nodes = new vis.DataSet([{",".join(nodes_js)}]);
var edges = new vis.DataSet([{",".join(edges_js)}]);
var container = document.getElementById("network");
var data = {{ nodes: nodes, edges: edges }};
var options = {{
  layout: {{
    hierarchical: {{
      enabled: true,
      direction: "UD",
      sortMethod: "directed",
      levelSeparation: 200,
      nodeSpacing: 240
    }}
  }},
  interaction: {{
    hover: true,
    navigationButtons: true,
    zoomView: true,
    tooltipDelay: 0
  }},
  physics: {{ enabled: false }},
  edges: {{
    arrows: {{ to: {{ enabled: true }} }},
    font: {{ size: 11, align: "top" }}
  }},
  nodes: {{
    shape: "box",
    font: {{ size: 13, multi: false }},
    borderWidth: 1,
    widthConstraint: {{ maximum: 220 }}
  }}
}};
var network = new vis.Network(container, data, options);
var tooltip = document.getElementById("tooltip");

network.on("hoverNode", function(params) {{
  var nodeId = params.node;
  var content = tooltipMap[nodeId];
  if (content) {{
    tooltip.innerHTML = content;
    tooltip.style.display = "block";
    tooltip.style.left = (params.event.center.x + 15) + "px";
    tooltip.style.top = (params.event.center.y + 15) + "px";
  }}
}});
network.on("blurNode", function() {{
  tooltip.style.display = "none";
}});
container.addEventListener("mousemove", function(e) {{
  if (tooltip.style.display === "block") {{
    tooltip.style.left = (e.pageX + 15) + "px";
    tooltip.style.top = (e.pageY + 15) + "px";
  }}
}});
</script>
</body>
</html>"""

    output_path.write_text(html_content, encoding="utf-8")
    logger.info(f"Saved interactive DAG to {output_path}")
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