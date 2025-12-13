"""
Test script for DAG Constructor integration with VLM Extractor.

This script tests:
1. DAG construction from VLM-extracted features
2. Graph operations (nodes, edges, paths)
3. do-calculus interventions
4. Counterfactual enumeration and evaluation
5. Integration with real Waymo scenario data (if available)
6. DAG visualization and export

Usage:
    # Basic test with mock data
    python test_dag_constructor.py
    
    # Test with real Waymo data
    python test_dag_constructor.py --use-waymo
    
    # Test with real GPT-4o API
    python test_dag_constructor.py --use-api

Author: CounterBMT Project
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try package import first, fall back to local
try:
    from counter_bmt.dag_constructor import (
        DAGConstructor,
        MockDAGClient,
        GPT4oDAGClient,
        ScenarioDAG,
        DAGNode,
        DAGEdge,
        NodeType,
        EdgeType,
        Intervention,
        CounterfactualResult,
    )
except ImportError:
    # Local import for standalone testing
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dag_constructor import (
        DAGConstructor,
        MockDAGClient,
        GPT4oDAGClient,
        ScenarioDAG,
        DAGNode,
        DAGEdge,
        NodeType,
        EdgeType,
        Intervention,
        CounterfactualResult,
    )


# =============================================================================
# DAG Visualization
# =============================================================================

def visualize_dag(dag: ScenarioDAG, output_path: Path, title: str = None) -> bool:
    """
    Visualize the DAG and save to file.
    
    Args:
        dag: The ScenarioDAG to visualize
        output_path: Path to save the image
        title: Optional title for the graph
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
        return False
    
    try:
        import networkx as nx
        return _visualize_with_networkx(dag, output_path, title)
    except ImportError:
        logger.info("networkx not available, using simple visualization")
        return _visualize_simple(dag, output_path, title)


def _visualize_with_networkx(dag: ScenarioDAG, output_path: Path, title: str = None) -> bool:
    """Visualize DAG using networkx and matplotlib."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx
    
    G = nx.DiGraph()
    
    # Color scheme for node types
    node_colors_map = {
        NodeType.ENVIRONMENTAL: '#90EE90',
        NodeType.VEHICLE_STATE: '#87CEEB',
        NodeType.AGENT_STATE: '#DDA0DD',
        NodeType.PERCEPTION: '#F0E68C',
        NodeType.DECISION: '#FF6B6B',
        NodeType.MANEUVER: '#4ECDC4',
        NodeType.INTERACTION: '#FFE66D',
        NodeType.OUTCOME: '#FF8C00',
        NodeType.SEVERITY: '#DC143C',
        NodeType.CONFOUNDER: '#808080',
        NodeType.CONTEXT: '#D3D3D3',
    }
    
    colors = []
    labels = {}
    for node_id, node in dag.nodes.items():
        G.add_node(node_id)
        colors.append(node_colors_map.get(node.node_type, '#FFFFFF'))
        label = node.name.replace("Decision: ", "D:").replace("Maneuver: ", "M:")
        if node.value:
            label += f"\n({node.value})"
        labels[node_id] = label
    
    edge_labels = {}
    for edge in dag.edges:
        G.add_edge(edge.parent_id, edge.child_id)
        edge_labels[(edge.parent_id, edge.child_id)] = f"{edge.confidence:.1f}"
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Hierarchical layout
    try:
        topo_order = dag.topological_sort()
        layers = {}
        for node_id in topo_order:
            parents = dag.get_parents(node_id)
            layers[node_id] = 0 if not parents else max(layers[p] for p in parents) + 1
        
        layer_nodes = {}
        for node_id, layer in layers.items():
            layer_nodes.setdefault(layer, []).append(node_id)
        
        pos = {}
        max_layer = max(layers.values()) if layers else 0
        for layer, nodes in layer_nodes.items():
            n_nodes = len(nodes)
            for i, node_id in enumerate(nodes):
                x = (i - (n_nodes - 1) / 2) * 2.5
                y = (max_layer - layer) * 2
                pos[node_id] = (x, y)
    except:
        pos = nx.spring_layout(G, k=2, iterations=50)
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=3500, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='#555555', arrows=True, 
                           arrowsize=20, arrowstyle='-|>', ax=ax,
                           connectionstyle="arc3,rad=0.1", width=1.5)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)
    
    # Legend
    legend_patches = []
    for node_type, color in node_colors_map.items():
        if any(n.node_type == node_type for n in dag.nodes.values()):
            legend_patches.append(mpatches.Patch(color=color, label=node_type.value))
    ax.legend(handles=legend_patches, loc='upper left', fontsize=9)
    
    title_text = title if title else f"Causal DAG: {dag.scenario_id}"
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved DAG visualization to {output_path}")
    return True


def _visualize_simple(dag: ScenarioDAG, output_path: Path, title: str = None) -> bool:
    """Simple text-based DAG visualization."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 8))
    lines = [f"Causal DAG: {dag.scenario_id}", f"Nodes: {len(dag.nodes)}, Edges: {len(dag.edges)}", "", "Nodes:"]
    for node_id, node in dag.nodes.items():
        lines.append(f"  [{node.node_type.value}] {node.name}")
        if node.value:
            lines.append(f"       Value: {node.value}")
    lines.extend(["", "Edges:"])
    for edge in dag.edges:
        lines.append(f"  {dag.nodes[edge.parent_id].name} -> {dag.nodes[edge.child_id].name}")
    
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontfamily='monospace', fontsize=10, verticalalignment='top')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def export_dag_to_dot(dag: ScenarioDAG, output_path: Path) -> bool:
    """Export DAG to DOT format for Graphviz."""
    type_colors = {
        "environmental": "lightgreen", "vehicle_state": "skyblue", "agent_state": "plum",
        "perception": "khaki", "decision": "coral", "maneuver": "turquoise",
        "interaction": "yellow", "outcome": "orange", "severity": "crimson",
        "confounder": "gray", "context": "lightgray",
    }
    
    lines = ["digraph CausalDAG {", "    rankdir=TB;", "    node [shape=box, style=filled];", ""]
    for node_id, node in dag.nodes.items():
        color = type_colors.get(node.node_type.value, "white")
        label = node.name + (f"\\n({node.value})" if node.value else "")
        lines.append(f'    "{node_id}" [label="{label}", fillcolor={color}];')
    lines.append("")
    for edge in dag.edges:
        lines.append(f'    "{edge.parent_id}" -> "{edge.child_id}" [label="{edge.confidence:.2f}"];')
    lines.append("}")
    
    output_path.write_text('\n'.join(lines))
    logger.info(f"Exported DAG to DOT: {output_path}")
    return True


# =============================================================================
# Mock ScenarioFeatures
# =============================================================================

@dataclass
class MockManeuverSegment:
    type: str
    start_timestep: int
    end_timestep: int
    start_timestamp: float
    end_timestamp: float
    aggressiveness: str = "normal"
    description: str = ""
    confidence: float = 1.0


@dataclass
class MockCriticalDecisionPoint:
    timestep: int
    timestamp: float
    type: str
    choice: str
    alternatives: List[str] = field(default_factory=list)
    description: str = ""
    confidence: float = 1.0


@dataclass
class MockScenarioFeatures:
    scenario_id: str
    maneuvers: List[MockManeuverSegment] = field(default_factory=list)
    decisions: List[MockCriticalDecisionPoint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Test Scenarios
# =============================================================================

def create_intersection_scenario() -> MockScenarioFeatures:
    return MockScenarioFeatures(
        scenario_id="test_intersection_001",
        maneuvers=[
            MockManeuverSegment(type="left_turn", start_timestep=0, end_timestep=5,
                start_timestamp=0.0, end_timestamp=0.5, aggressiveness="normal",
                description="Ego initiates left turn at intersection"),
            MockManeuverSegment(type="straight", start_timestep=5, end_timestep=10,
                start_timestamp=0.5, end_timestamp=1.0, aggressiveness="normal",
                description="Ego continues straight after turn")
        ],
        decisions=[
            MockCriticalDecisionPoint(timestep=2, timestamp=0.2, type="proceed_or_yield",
                choice="proceed", alternatives=["proceed", "yield"],
                description="Chose to proceed while oncoming vehicle approaching", confidence=0.9),
            MockCriticalDecisionPoint(timestep=5, timestamp=0.5, type="evasive_action",
                choice="maintain_course", alternatives=["maintain_course", "brake", "swerve"],
                description="Maintained course without evasive action", confidence=0.8),
            MockCriticalDecisionPoint(timestep=6, timestamp=0.7, type="gap_acceptance",
                choice="accept_gap", alternatives=["accept_gap", "reject_gap"],
                description="Accepted gap to complete turn", confidence=0.85)
        ],
        metadata={"n_images": 8, "time_range": [0.0, 1.0]}
    )


def create_lane_change_scenario() -> MockScenarioFeatures:
    return MockScenarioFeatures(
        scenario_id="test_lane_change_002",
        maneuvers=[
            MockManeuverSegment(type="straight", start_timestep=0, end_timestep=3,
                start_timestamp=0.0, end_timestamp=0.3, description="Traveling in right lane"),
            MockManeuverSegment(type="lane_change_left", start_timestep=3, end_timestep=7,
                start_timestamp=0.3, end_timestamp=0.7, aggressiveness="aggressive",
                description="Lane change to left"),
            MockManeuverSegment(type="straight", start_timestep=7, end_timestep=10,
                start_timestamp=0.7, end_timestamp=1.0, description="Continues in left lane")
        ],
        decisions=[
            MockCriticalDecisionPoint(timestep=3, timestamp=0.3, type="gap_acceptance",
                choice="accept_gap", alternatives=["accept_gap", "reject_gap"],
                description="Accepted gap for lane change", confidence=0.75),
            MockCriticalDecisionPoint(timestep=5, timestamp=0.5, type="speed_choice",
                choice="accelerate", alternatives=["accelerate", "maintain", "decelerate"],
                description="Accelerated during lane change", confidence=0.9)
        ]
    )


def create_pedestrian_scenario() -> MockScenarioFeatures:
    return MockScenarioFeatures(
        scenario_id="test_pedestrian_003",
        maneuvers=[
            MockManeuverSegment(type="straight", start_timestep=0, end_timestep=4,
                start_timestamp=0.0, end_timestamp=0.4, description="Approaching crosswalk"),
            MockManeuverSegment(type="decelerate", start_timestep=4, end_timestep=7,
                start_timestamp=0.4, end_timestamp=0.7, description="Decelerating for pedestrian"),
            MockManeuverSegment(type="stop", start_timestep=7, end_timestep=10,
                start_timestamp=0.7, end_timestamp=1.0, aggressiveness="passive",
                description="Stopped for pedestrian")
        ],
        decisions=[
            MockCriticalDecisionPoint(timestep=4, timestamp=0.4, type="proceed_or_yield",
                choice="yield", alternatives=["proceed", "yield"],
                description="Yielded to pedestrian in crosswalk", confidence=0.95)
        ]
    )


# =============================================================================
# Test Functions
# =============================================================================

def test_dag_construction(client, output_dir: Path):
    """Test basic DAG construction from features."""
    print("\n" + "=" * 60)
    print("TEST: DAG Construction")
    print("=" * 60)
    
    constructor = DAGConstructor(client)
    features = create_intersection_scenario()
    dag = constructor.construct_from_features(features)
    
    print(f"\nScenario: {features.scenario_id}")
    print(f"Input: {len(features.maneuvers)} maneuvers, {len(features.decisions)} decisions")
    print(f"Output: {len(dag.nodes)} nodes, {len(dag.edges)} edges")
    
    if "llm_reasoning" in dag.metadata:
        print(f"\nLLM Reasoning: {dag.metadata['llm_reasoning']}")
    
    assert len(dag.nodes) >= len(features.maneuvers) + len(features.decisions)
    assert len(dag.edges) > 0
    assert not dag._has_cycle()
    
    node_types = {n.node_type for n in dag.nodes.values()}
    print(f"Node types: {[t.value for t in node_types]}")
    
    # Visualize
    viz_path = output_dir / f"{features.scenario_id}_dag.png"
    visualize_dag(dag, viz_path, title=f"Causal DAG: {features.scenario_id}")
    
    dot_path = output_dir / f"{features.scenario_id}_dag.dot"
    export_dag_to_dot(dag, dot_path)
    
    print("\n✓ DAG construction test passed")
    return dag


def test_graph_operations(dag: ScenarioDAG):
    """Test graph query operations."""
    print("\n" + "=" * 60)
    print("TEST: Graph Operations")
    print("=" * 60)
    
    topo_order = dag.topological_sort()
    print(f"\nTopological order: {topo_order}")
    
    roots = dag.get_roots()
    leaves = dag.get_leaves()
    print(f"Roots: {roots}")
    print(f"Leaves: {leaves}")
    
    if roots and leaves:
        paths = dag.find_all_paths(roots[0], leaves[0])
        print(f"\nPaths from {roots[0]} to {leaves[0]}: {len(paths)}")
        for i, path in enumerate(paths[:3]):
            path_names = [dag.nodes[n].name for n in path]
            print(f"  {i+1}: {' -> '.join(path_names)}")
    
    print("\n✓ Graph operations test passed")


def test_do_calculus(dag: ScenarioDAG):
    """Test do-calculus interventions."""
    print("\n" + "=" * 60)
    print("TEST: Do-Calculus Interventions")
    print("=" * 60)
    
    decision_nodes = [n for n in dag.nodes.values() if n.node_type == NodeType.DECISION]
    if not decision_nodes:
        print("No decision nodes, skipping")
        return
    
    target = decision_nodes[0]
    original = target.value
    new_val = "yield" if original != "yield" else "proceed"
    
    print(f"\nOriginal: {target.id} = {original}, Parents: {dag.get_parents(target.id)}")
    
    modified = dag.do({target.id: new_val})
    
    print(f"After do({target.id}={new_val}): Parents: {modified.get_parents(target.id)}")
    print(f"Is intervened: {modified.nodes[target.id].is_intervened}")
    
    assert modified.nodes[target.id].value == new_val
    assert modified.nodes[target.id].is_intervened
    assert dag.nodes[target.id].value == original  # Original unchanged
    
    print("\n✓ Do-calculus test passed")


def test_intervention_enumeration(dag: ScenarioDAG):
    """Test intervention enumeration."""
    print("\n" + "=" * 60)
    print("TEST: Intervention Enumeration")
    print("=" * 60)
    
    interventions = dag.enumerate_interventions()
    print(f"\nFound {len(interventions)} interventions:")
    for i, intv in enumerate(interventions, 1):
        print(f"  {i}. {intv.description}")
    
    assert len(interventions) > 0
    print("\n✓ Intervention enumeration test passed")
    return interventions


def test_counterfactual_evaluation(client, dag: ScenarioDAG, output_dir: Path):
    """Test counterfactual evaluation."""
    print("\n" + "=" * 60)
    print("TEST: Counterfactual Evaluation")
    print("=" * 60)
    
    constructor = DAGConstructor(client)
    results = constructor.evaluate_counterfactuals(dag, outcome_var="collision_risk")
    
    print(f"\nEvaluated {len(results)} counterfactuals:")
    for r in results:
        print(f"\n  do({r.intervention.variable_id}={r.intervention.value}):")
        print(f"    {r.original_outcome} -> {r.counterfactual_outcome} ({r.effect_direction})")
        print(f"    Confidence: {r.confidence:.2f}, Paths: {len(r.affected_paths)}")
        print(f"    Reasoning: {r.reasoning[:80]}...")
    
    # Save results
    cf_path = output_dir / "counterfactual_analysis.json"
    with open(cf_path, 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"\nSaved counterfactual analysis to {cf_path}")
    
    print("\n✓ Counterfactual evaluation test passed")
    return results


def test_serialization(dag: ScenarioDAG):
    """Test DAG serialization."""
    print("\n" + "=" * 60)
    print("TEST: Serialization")
    print("=" * 60)
    
    json_str = dag.to_json()
    print(f"\nSerialized: {len(json_str)} chars")
    
    restored = ScenarioDAG.from_json(json_str)
    assert len(restored.nodes) == len(dag.nodes)
    assert len(restored.edges) == len(dag.edges)
    
    print("✓ Serialization test passed")


def test_multiple_scenarios(client, output_dir: Path):
    """Test multiple scenario types."""
    print("\n" + "=" * 60)
    print("TEST: Multiple Scenarios")
    print("=" * 60)
    
    scenarios = [
        ("Intersection", create_intersection_scenario()),
        ("Lane Change", create_lane_change_scenario()),
        ("Pedestrian", create_pedestrian_scenario()),
    ]
    
    constructor = DAGConstructor(client)
    
    for name, features in scenarios:
        dag = constructor.construct_from_features(features)
        print(f"\n{name}: {len(dag.nodes)} nodes, {len(dag.edges)} edges")
        
        viz_path = output_dir / f"{features.scenario_id}_dag.png"
        visualize_dag(dag, viz_path)
    
    print("\n✓ Multiple scenarios test passed")


def test_with_waymo_data(client, data_dir: Path, output_dir: Path):
    """Test with real Waymo scenario data."""
    print("\n" + "=" * 60)
    print("TEST: Waymo Integration")
    print("=" * 60)

    try:
        from counter_bmt.scenarionet_visualizer import prepare_for_vlm
        from counter_bmt.vlm_extractor import (
            VLMSafetyCriticalExtractor, 
            MockGPT4oClient,
            TimestampedImage
        )
    except ImportError as e:
        print(f"Cannot import: {e}")
        print("Skipping Waymo test")
        return
    
    print(f"\nLoading from: {data_dir}")
    saved_images, trajectory, scenario_id = prepare_for_vlm(
        data_dir=str(data_dir),
        scenario_index=0,
        output_dir=str(output_dir),
        num_frames=5
    )
    
    print(f"Generated {len(saved_images)} images for scenario {scenario_id}")
    
    images = [TimestampedImage(path=p, timestamp=t) for p, t in saved_images]
    
    vlm_client = MockGPT4oClient()
    extractor = VLMSafetyCriticalExtractor(vlm_client)
    features = extractor.extract(images, scenario_id, trajectory)
    
    # Handle both attribute naming conventions
    maneuvers = getattr(features, 'maneuvers', None) or getattr(features, 'maneuver_segments', [])
    decisions = getattr(features, 'decisions', None) or getattr(features, 'critical_decisions', [])
    print(f"Extracted: {len(maneuvers)} maneuvers, {len(decisions)} decisions")
    
    # Build DAG
    constructor = DAGConstructor(client)
    dag = constructor.construct_from_features(features)
    
    print(f"\n{dag.summary()}")
    
    # Log LLM reasoning
    if "llm_reasoning" in dag.metadata:
        print(f"\nLLM Reasoning: {dag.metadata['llm_reasoning']}")
    
    # Visualize
    viz_path = output_dir / f"{scenario_id}_dag.png"
    visualize_dag(dag, viz_path, title=f"Waymo Scenario: {scenario_id}")
    
    dot_path = output_dir / f"{scenario_id}_dag.dot"
    export_dag_to_dot(dag, dot_path)
    
    # Evaluate counterfactuals
    results = constructor.evaluate_counterfactuals(dag)
    print(f"\nCounterfactual results: {len(results)}")
    
    for r in results:
        print(f"  do({r.intervention.variable_id}={r.intervention.value}): {r.effect_direction}")
    
    # Save comprehensive results
    output_file = output_dir / "dag_result.json"
    with open(output_file, 'w') as f:
        json.dump({
            "scenario_id": scenario_id,
            "extraction": {
                "n_maneuvers": len(maneuvers),
                "n_decisions": len(decisions),
            },
            "dag": dag.to_dict(),
            "counterfactuals": [r.to_dict() for r in results],
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "llm_reasoning": dag.metadata.get("llm_reasoning", ""),
            }
        }, f, indent=2)
    print(f"\nSaved to: {output_file}")
    
    print("\n✓ Waymo integration test passed")


def test_edge_cases(client):
    """Test edge cases."""
    print("\n" + "=" * 60)
    print("TEST: Edge Cases")
    print("=" * 60)
    
    constructor = DAGConstructor(client)
    
    # Empty features
    empty = MockScenarioFeatures(scenario_id="empty")
    dag = constructor.construct_from_features(empty)
    print(f"\nEmpty: {len(dag.nodes)} nodes")
    assert len(dag.nodes) >= 1
    
    # Cycle prevention
    dag = ScenarioDAG("cycle_test")
    dag.add_node(DAGNode(id="a", name="A", node_type=NodeType.DECISION))
    dag.add_node(DAGNode(id="b", name="B", node_type=NodeType.MANEUVER))
    dag.add_node(DAGNode(id="c", name="C", node_type=NodeType.OUTCOME))
    dag.add_edge(DAGEdge(parent_id="a", child_id="b"))
    dag.add_edge(DAGEdge(parent_id="b", child_id="c"))
    result = dag.add_edge(DAGEdge(parent_id="c", child_id="a"))
    print(f"Cycle prevention: {not result}")
    assert not result
    
    print("\n✓ Edge cases test passed")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test DAG Constructor")
    parser.add_argument("--use-api", action="store_true", help="Use GPT-4o API")
    parser.add_argument("--use-waymo", action="store_true", help="Test with Waymo data")
    parser.add_argument("--data-dir", type=str, default="./exp_converted")
    parser.add_argument("--output-dir", type=str, default="./outputs/dag_test")
    args = parser.parse_args()
    
    if args.use_api:
        print("Using GPT-4o API")
        try:
            client = GPT4oDAGClient()
        except ValueError as e:
            print(f"Error: {e}, falling back to mock")
            client = MockDAGClient()
    else:
        print("Using mock client")
        client = MockDAGClient()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_passed = True
    
    try:
        dag = test_dag_construction(client, output_dir)
        test_graph_operations(dag)
        test_do_calculus(dag)
        test_intervention_enumeration(dag)
        test_counterfactual_evaluation(client, dag, output_dir)
        test_serialization(dag)
        test_multiple_scenarios(client, output_dir)
        test_edge_cases(client)
        
        if args.use_waymo:
            data_dir = Path(args.data_dir)
            if data_dir.exists():
                test_with_waymo_data(client, data_dir, output_dir)
            else:
                print(f"\nWaymo data not found at {data_dir}")
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        all_passed = False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓" if all_passed else "SOME TESTS FAILED ✗")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())