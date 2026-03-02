"""Visualize DAG cache JSON files as a PNG diagram.

Supports both cache payloads (`cache/<scenario_id>.json`) and example DAG
payloads (`examples/<scenario_id>/dag.json`) produced by build_dag_cache_v2.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


TYPE_ORDER = ("context", "ego_state", "interaction", "maneuver", "decision", "risk", "outcome")
TYPE_TO_TIER = {t: i for i, t in enumerate(TYPE_ORDER)}

NODE_COLORS = {
    "context": "#8c8c8c",
    "ego_state": "#2a9d8f",
    "interaction": "#457b9d",
    "maneuver": "#264653",
    "decision": "#1d3557",
    "risk": "#f4a261",
    "outcome": "#e76f51",
    "unknown": "#6d6875",
}


def _to_str(x: Any) -> str:
    return str(x).strip()


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        y = float(x)
    except Exception:
        return None
    if not np.isfinite(y):
        return None
    return float(y)


def _short_text(x: Any, limit: int = 30) -> str:
    s = _to_str(x)
    if len(s) <= int(limit):
        return s
    return s[: max(1, int(limit) - 3)] + "..."


def _load_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "nodes" in payload and "edges" in payload:
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("dag"), dict):
        dag = payload["dag"]
        if "nodes" in dag and "edges" in dag:
            return dag
    raise ValueError(f"Unsupported DAG JSON format: {path}")


def _normalize_nodes(payload: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    raw_nodes = payload.get("nodes", [])
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    if not isinstance(raw_nodes, list):
        raw_nodes = []
    for i, rec in enumerate(raw_nodes):
        if not isinstance(rec, Mapping):
            continue
        node_id = _to_str(rec.get("node_id", f"node_{i}")) or f"node_{i}"
        if node_id in seen:
            node_id = f"{node_id}_{i}"
        seen.add(node_id)
        node_type = _to_str(rec.get("node_type", "unknown")).lower().replace("-", "_")
        ts = _to_float(rec.get("timestamp_s"))
        out.append(
            {
                "node_id": node_id,
                "node_type": node_type if node_type else "unknown",
                "value": rec.get("value", ""),
                "timestamp_s": ts,
                "metadata": dict(rec.get("metadata", {})) if isinstance(rec.get("metadata", {}), Mapping) else {},
            }
        )
    by_id = {n["node_id"]: n for n in out}
    return out, by_id


def _normalize_edges(payload: Mapping[str, Any], node_by_id: Mapping[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    raw_edges = payload.get("edges", [])
    edges: List[Dict[str, Any]] = []
    if not isinstance(raw_edges, list):
        return edges
    for rec in raw_edges:
        if not isinstance(rec, Mapping):
            continue
        src = _to_str(rec.get("parent_id", ""))
        dst = _to_str(rec.get("child_id", ""))
        if not src or not dst:
            continue
        if src not in node_by_id or dst not in node_by_id:
            continue
        conf = _to_float(rec.get("confidence"))
        if conf is None:
            conf = 0.7
        edges.append(
            {
                "parent_id": src,
                "child_id": dst,
                "confidence": float(np.clip(conf, 0.0, 1.0)),
                "mechanism": _to_str(rec.get("mechanism", "")),
            }
        )
    return edges


def _node_sort_key(node: Mapping[str, Any]) -> Tuple[int, float, str]:
    t = _to_str(node.get("node_type", "unknown"))
    tier = int(TYPE_TO_TIER.get(t, len(TYPE_ORDER)))
    ts = node.get("timestamp_s")
    ts_v = float(ts) if ts is not None else 1e9
    return tier, ts_v, _to_str(node.get("node_id", ""))


def _compute_positions(nodes: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for node in sorted(nodes, key=_node_sort_key):
        tier = int(TYPE_TO_TIER.get(_to_str(node["node_type"]), len(TYPE_ORDER)))
        grouped.setdefault(tier, []).append(node)

    pos: Dict[str, Tuple[float, float]] = {}
    for tier, tier_nodes in sorted(grouped.items(), key=lambda kv: kv[0]):
        n = len(tier_nodes)
        if n == 1:
            ys = [0.0]
        else:
            ys = np.linspace(0.9, -0.9, n).tolist()
        x = float(tier) * 2.6
        for node, y in zip(tier_nodes, ys):
            pos[_to_str(node["node_id"])] = (x, float(y))
    return pos


def _format_node_label(node: Mapping[str, Any], mode: str) -> str:
    node_id = _to_str(node.get("node_id", ""))
    node_type = _to_str(node.get("node_type", "unknown"))
    value = _short_text(node.get("value", ""), limit=28)
    ts = node.get("timestamp_s")
    if ts is None:
        ts_str = ""
    else:
        ts_str = f"\nt={float(ts):.1f}s"

    if mode == "id":
        return node_id
    if mode == "id_type":
        return f"{node_id}\n[{node_type}]"
    if mode == "id_value":
        return f"{node_id}\n{value}{ts_str}"
    return f"{node_id}\n[{node_type}] {value}{ts_str}"


def _draw_graph(
    *,
    payload: Mapping[str, Any],
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    pos: Mapping[str, Tuple[float, float]],
    output: Path,
    title: str,
    label_mode: str,
    show_edge_labels: bool,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_facecolor("#f8f9fb")

    node_w = 1.8
    node_h = 0.38

    for e in edges:
        u = _to_str(e["parent_id"])
        v = _to_str(e["child_id"])
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        start = (x1 + node_w * 0.48, y1)
        end = (x2 - node_w * 0.48, y2)
        alpha = 0.45 + 0.45 * float(e.get("confidence", 0.7))
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.3,
            color="#4d4d4d",
            alpha=alpha,
            connectionstyle="arc3,rad=0.04",
        )
        ax.add_patch(arrow)
        if show_edge_labels:
            xm = 0.5 * (start[0] + end[0])
            ym = 0.5 * (start[1] + end[1])
            conf = float(e.get("confidence", 0.0))
            mech = _short_text(e.get("mechanism", ""), limit=14)
            ax.text(
                xm,
                ym + 0.05,
                f"{conf:.2f} {mech}".strip(),
                fontsize=7,
                color="#555555",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )

    for node in nodes:
        node_id = _to_str(node["node_id"])
        x, y = pos[node_id]
        ntype = _to_str(node.get("node_type", "unknown"))
        color = NODE_COLORS.get(ntype, NODE_COLORS["unknown"])
        patch = FancyBboxPatch(
            (x - node_w / 2.0, y - node_h / 2.0),
            node_w,
            node_h,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            linewidth=1.0,
            edgecolor="#1f2937",
            facecolor=color,
            alpha=0.92,
        )
        ax.add_patch(patch)
        ax.text(
            x,
            y,
            _format_node_label(node, label_mode),
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            linespacing=1.1,
        )

    if pos:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
        ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    ax.axis("off")

    scenario_id = _to_str(payload.get("scenario_id", "unknown"))
    schema = _to_str(payload.get("schema_version", ""))
    subtitle = f"scenario_id={scenario_id} | nodes={len(nodes)} edges={len(edges)} | {schema}"
    ax.set_title(title, fontsize=14, pad=16, weight="bold")
    fig.text(0.5, 0.02, subtitle, ha="center", va="bottom", fontsize=9, color="#374151")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.01, 0.05, 0.99, 0.95])
    fig.savefig(output, dpi=int(max(72, dpi)))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize a DAG cache JSON file as an image.")
    p.add_argument("--dag-json", type=str, required=True, help="Path to dag.json or cache/<scenario_id>.json.")
    p.add_argument("--output", type=str, default="", help="Output image path (default: <dag-json-stem>_viz.png).")
    p.add_argument("--title", type=str, default="DAG Visualization")
    p.add_argument(
        "--label-mode",
        type=str,
        default="id_type_value",
        choices=["id", "id_type", "id_value", "id_type_value"],
    )
    p.add_argument("--hide-edge-labels", action="store_true", help="Hide edge confidence/mechanism labels.")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--show", action="store_true", help="Open an interactive window after saving.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dag_json = Path(args.dag_json)
    if not dag_json.exists():
        raise FileNotFoundError(f"DAG JSON not found: {dag_json}")

    payload = _load_payload(dag_json)
    nodes, node_by_id = _normalize_nodes(payload)
    edges = _normalize_edges(payload, node_by_id)
    pos = _compute_positions(nodes)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = dag_json.with_name(f"{dag_json.stem}_viz.png")

    _draw_graph(
        payload=payload,
        nodes=nodes,
        edges=edges,
        pos=pos,
        output=out_path,
        title=str(args.title),
        label_mode=str(args.label_mode),
        show_edge_labels=not bool(args.hide_edge_labels),
        dpi=int(args.dpi),
    )
    print(f"Saved DAG visualization: {out_path}")

    if bool(args.show):
        image = plt.imread(out_path)
        plt.figure(figsize=(11, 6))
        plt.imshow(image)
        plt.axis("off")
        plt.title(str(args.title))
        plt.tight_layout()
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

