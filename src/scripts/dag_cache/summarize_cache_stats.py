"""Summarize DAG cache statistics for quick latent-quality sanity checks.

This is intended as a lightweight pre-training audit tool for cache builds.
Point it at either:
- a cache directory containing `*.json`, or
- a build output root containing `cache/`, `manifest.json`, etc.

It prints high-signal summary stats and can optionally write JSON / Markdown.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

REQUIRED_OUTCOMES = (
    "collision_outcome",
    "progress_outcome",
    "compliance_outcome",
)


def _safe_mean(xs: Sequence[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def _safe_median(xs: Sequence[float]) -> float:
    return float(median(xs)) if xs else 0.0


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _stats(xs: Sequence[float | int]) -> Dict[str, float]:
    vals = [float(x) for x in xs]
    if not vals:
        return {"count": 0.0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(vals)),
        "mean": _safe_mean(vals),
        "median": _safe_median(vals),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def _top_items(counter: Counter[str], top_k: int) -> List[Dict[str, Any]]:
    return [{"key": str(k), "count": int(v)} for k, v in counter.most_common(int(top_k))]


def _resolve_paths(input_path: Path) -> Tuple[Path, Path]:
    cache_dir = input_path / "cache"
    if cache_dir.is_dir():
        return cache_dir, input_path
    if input_path.is_dir() and any(input_path.glob("*.json")):
        return input_path, input_path.parent
    raise FileNotFoundError(
        f"Could not resolve cache dir from {input_path}. Expected either a cache dir with *.json "
        "or an output root containing cache/."
    )


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_payload_light(payload: Mapping[str, Any]) -> bool:
    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version not in {
        "counter_bmt_v2_dag_cache_v2_compact10",
        "counter_bmt_v2_dag_cache_v3_maneuver_outcome",
    }:
        return False
    if not str(payload.get("scenario_id", "")).strip():
        return False
    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return False

    node_ids = set()
    for node in nodes:
        if not isinstance(node, Mapping):
            return False
        node_id = str(node.get("node_id", "")).strip()
        if not node_id:
            return False
        node_ids.add(node_id)

    for edge in edges:
        if not isinstance(edge, Mapping):
            return False
        parent_id = str(edge.get("parent_id", "")).strip()
        child_id = str(edge.get("child_id", "")).strip()
        if not parent_id or not child_id:
            return False
        if parent_id not in node_ids or child_id not in node_ids:
            return False

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return False
    report = metadata.get("contract_report", {})
    if not isinstance(report, Mapping):
        return False
    return bool(report.get("passed", False))


def _interval_complete(node: Mapping[str, Any]) -> bool:
    md = node.get("metadata", {})
    if not isinstance(md, Mapping):
        return False
    return all(k in md for k in ("start_s", "end_s", "duration_s", "mid_s"))


def _graph_maneuver_classes(nodes: Iterable[Mapping[str, Any]]) -> List[str]:
    out: List[str] = []
    for node in nodes:
        if str(node.get("node_type", "")).strip().lower() != "maneuver":
            continue
        out.append(str(node.get("value", "")).strip())
    return out


def _summarize_cache(cache_dir: Path, root_dir: Path, *, top_k: int) -> Dict[str, Any]:
    files = sorted(cache_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No cache json files found in {cache_dir}")

    schema_counts: Counter[str] = Counter()
    contract_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    renderer_counts: Counter[str] = Counter()
    scenario_ids: List[str] = []
    valid_payloads = 0

    node_counts: List[int] = []
    edge_counts: List[int] = []
    maneuver_counts: List[int] = []
    unique_maneuver_class_counts: List[int] = []
    interval_complete_rates: List[float] = []
    edge_confidences: List[float] = []

    maneuver_class_node_counts: Counter[str] = Counter()
    maneuver_class_graph_counts: Counter[str] = Counter()
    maneuver_pair_graph_counts: Counter[str] = Counter()
    outcome_value_counts: Dict[str, Counter[str]] = {k: Counter() for k in REQUIRED_OUTCOMES}
    outcome_presence_counts: Counter[str] = Counter()
    outcome_edge_counts: Counter[str] = Counter()
    maneuver_to_outcome_mechanisms: Counter[str] = Counter()
    maneuver_count_hist: Counter[str] = Counter()

    densest_graphs: List[Tuple[str, int, int]] = []

    for fp in files:
        payload = _read_json(fp)
        sid = str(payload.get("scenario_id", fp.stem))
        scenario_ids.append(sid)

        schema_counts[str(payload.get("schema_version", ""))] += 1
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        contract_counts[str(metadata.get("contract_name", ""))] += 1
        model = str(metadata.get("model", "")).strip()
        if model:
            model_counts[model] += 1
        renderer = str(metadata.get("frame_renderer", "")).strip()
        if renderer:
            renderer_counts[renderer] += 1

        if _validate_payload_light(payload):
            valid_payloads += 1

        nodes = payload.get("nodes", [])
        edges = payload.get("edges", [])
        if not isinstance(nodes, list):
            nodes = []
        if not isinstance(edges, list):
            edges = []

        node_counts.append(len(nodes))
        edge_counts.append(len(edges))
        densest_graphs.append((sid, len(nodes), len(edges)))

        maneuvers = [n for n in nodes if str(n.get("node_type", "")).strip().lower() == "maneuver"]
        outcomes = [n for n in nodes if str(n.get("node_type", "")).strip().lower() == "outcome"]
        maneuver_counts.append(len(maneuvers))
        maneuver_count_hist[str(len(maneuvers))] += 1

        class_list = _graph_maneuver_classes(maneuvers)
        class_set = sorted(set(class_list))
        unique_maneuver_class_counts.append(len(class_set))
        for cls in class_list:
            maneuver_class_node_counts[cls] += 1
        for cls in class_set:
            maneuver_class_graph_counts[cls] += 1
        for i, cls_a in enumerate(class_set):
            for cls_b in class_set[i + 1 :]:
                maneuver_pair_graph_counts[f"{cls_a} + {cls_b}"] += 1

        if maneuvers:
            complete = sum(1 for node in maneuvers if _interval_complete(node))
            interval_complete_rates.append(float(complete / max(1, len(maneuvers))))
        else:
            interval_complete_rates.append(0.0)

        outcome_ids_in_graph = {str(n.get("node_id", "")) for n in outcomes}
        for out_id in REQUIRED_OUTCOMES:
            if out_id in outcome_ids_in_graph:
                outcome_presence_counts[out_id] += 1

        for node in outcomes:
            node_id = str(node.get("node_id", ""))
            if node_id in outcome_value_counts:
                outcome_value_counts[node_id][str(node.get("value", ""))] += 1

        for edge in edges:
            child = str(edge.get("child_id", ""))
            mech = str(edge.get("mechanism", "")).strip() or "unknown"
            conf = _to_float(edge.get("confidence"))
            if conf is not None:
                edge_confidences.append(conf)
            if child in REQUIRED_OUTCOMES:
                outcome_edge_counts[child] += 1
            maneuver_to_outcome_mechanisms[mech] += 1

    total_graphs = len(files)
    densest_graphs = sorted(densest_graphs, key=lambda x: (x[2], x[1], x[0]), reverse=True)[: min(top_k, 10)]

    maneuver_node_total = sum(maneuver_class_node_counts.values())
    dominant_class = maneuver_class_node_counts.most_common(1)[0] if maneuver_class_node_counts else ("", 0)
    dominant_class_fraction = float(dominant_class[1] / max(1, maneuver_node_total))

    advisories: List[str] = []
    if total_graphs < 200:
        advisories.append(
            "Pilot-scale cache only. Good for auditing and Stage B smoke tests, but not enough to trust latent quality."
        )
    if dominant_class_fraction > 0.6:
        advisories.append(
            f"Maneuver class skew is high: {dominant_class[0]!r} accounts for {100.0 * dominant_class_fraction:.1f}% of maneuver nodes."
        )
    if _safe_mean(maneuver_counts) < 1.5:
        advisories.append("Average maneuvers per graph is low; graphs may be too simple to teach rich latent structure.")
    if _safe_mean(interval_complete_rates) < 0.9:
        advisories.append("Interval metadata completeness is below 90%; timing signal may be noisy.")
    missing_outcome_ids = [
        out_id for out_id in REQUIRED_OUTCOMES if outcome_presence_counts.get(out_id, 0) < total_graphs
    ]
    if missing_outcome_ids:
        advisories.append(f"Some required outcome anchors are missing in at least one graph: {missing_outcome_ids}.")
    if valid_payloads < total_graphs:
        advisories.append(
            f"{total_graphs - valid_payloads} payload(s) do not pass cache schema validation; inspect before training."
        )
    if not advisories:
        advisories.append("No obvious contract-quality red flags in the current cache sample.")

    manifest_path = root_dir / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else None

    return {
        "cache_dir": str(cache_dir),
        "root_dir": str(root_dir),
        "total_graphs": int(total_graphs),
        "valid_payloads": int(valid_payloads),
        "schema_versions": dict(schema_counts),
        "contract_names": dict(contract_counts),
        "models": dict(model_counts),
        "frame_renderers": dict(renderer_counts),
        "node_count_stats": _stats(node_counts),
        "edge_count_stats": _stats(edge_counts),
        "maneuvers_per_graph_stats": _stats(maneuver_counts),
        "unique_maneuver_classes_per_graph_stats": _stats(unique_maneuver_class_counts),
        "interval_complete_rate_stats": _stats(interval_complete_rates),
        "edge_confidence_stats": _stats(edge_confidences),
        "maneuver_count_histogram": {k: int(v) for k, v in sorted(maneuver_count_hist.items(), key=lambda kv: int(kv[0]))},
        "maneuver_class_distribution": {
            "node_counts": dict(maneuver_class_node_counts),
            "graph_presence_counts": dict(maneuver_class_graph_counts),
            "top_pairs_by_graph_presence": _top_items(maneuver_pair_graph_counts, top_k),
        },
        "outcome_summary": {
            "presence_counts": dict(outcome_presence_counts),
            "value_counts": {k: dict(v) for k, v in outcome_value_counts.items()},
            "incoming_edge_counts": dict(outcome_edge_counts),
        },
        "edge_mechanisms": dict(maneuver_to_outcome_mechanisms),
        "densest_graphs": [
            {"scenario_id": sid, "n_nodes": int(n_nodes), "n_edges": int(n_edges)}
            for sid, n_nodes, n_edges in densest_graphs
        ],
        "advisories": advisories,
        "manifest_counts": (manifest.get("counts", {}) if isinstance(manifest, Mapping) else {}),
        "manifest_failure_reasons": (manifest.get("failure_reasons", {}) if isinstance(manifest, Mapping) else {}),
    }


def _to_markdown(summary: Mapping[str, Any], *, top_k: int) -> str:
    lines: List[str] = []
    lines.append("# DAG Cache Summary")
    lines.append("")
    lines.append(f"- cache_dir: `{summary['cache_dir']}`")
    lines.append(f"- total_graphs: `{summary['total_graphs']}`")
    lines.append(f"- valid_payloads: `{summary['valid_payloads']}`")
    lines.append(f"- schema_versions: `{json.dumps(summary['schema_versions'], sort_keys=True)}`")
    lines.append(f"- contract_names: `{json.dumps(summary['contract_names'], sort_keys=True)}`")
    lines.append(f"- models: `{json.dumps(summary['models'], sort_keys=True)}`")
    lines.append("")

    for key in (
        "node_count_stats",
        "edge_count_stats",
        "maneuvers_per_graph_stats",
        "unique_maneuver_classes_per_graph_stats",
        "interval_complete_rate_stats",
    ):
        stats = dict(summary.get(key, {}))
        lines.append(f"## {key}")
        lines.append("")
        lines.append(f"- mean: `{stats.get('mean', 0.0):.3f}`")
        lines.append(f"- median: `{stats.get('median', 0.0):.3f}`")
        lines.append(f"- min: `{stats.get('min', 0.0):.3f}`")
        lines.append(f"- max: `{stats.get('max', 0.0):.3f}`")
        lines.append("")

    lines.append("## Maneuver Classes")
    lines.append("")
    class_dist = summary.get("maneuver_class_distribution", {})
    node_counts = class_dist.get("node_counts", {})
    graph_counts = class_dist.get("graph_presence_counts", {})
    for cls, count in sorted(node_counts.items(), key=lambda kv: (-int(kv[1]), kv[0])):
        lines.append(
            f"- `{cls}`: node_count=`{int(count)}` graph_presence=`{int(graph_counts.get(cls, 0))}`"
        )
    lines.append("")

    lines.append("## Outcome Values")
    lines.append("")
    outcome_summary = summary.get("outcome_summary", {})
    for out_id, counts in sorted(outcome_summary.get("value_counts", {}).items()):
        lines.append(f"- `{out_id}`: `{json.dumps(counts, sort_keys=True)}`")
    lines.append("")

    lines.append(f"## Top {top_k} Maneuver Pairs")
    lines.append("")
    for item in class_dist.get("top_pairs_by_graph_presence", [])[:top_k]:
        lines.append(f"- `{item['key']}`: `{item['count']}`")
    lines.append("")

    lines.append("## Advisories")
    lines.append("")
    for line in summary.get("advisories", []):
        lines.append(f"- {line}")
    lines.append("")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize DAG cache statistics.")
    p.add_argument("--cache-dir", type=str, required=True, help="Cache dir or cache build root containing cache/")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--output-json", type=str, default="")
    p.add_argument("--output-md", type=str, default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir, root_dir = _resolve_paths(Path(args.cache_dir))
    summary = _summarize_cache(cache_dir, root_dir, top_k=int(args.top_k))

    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Saved JSON summary: {out_json}")

    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_to_markdown(summary, top_k=int(args.top_k)), encoding="utf-8")
        print(f"Saved Markdown summary: {out_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
