"""Build v2-native DAG cache files from ScenarioNet scenes."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
import traceback
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Allow standalone execution from repo root.
import sys

if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[2]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

from counter_bmt_v2.causal import PromptBNDAGBuilder
from counter_bmt_v2.causal.dag_contract import DAGContractConfig, enforce_dag_contract
from counter_bmt_v2.contracts import ScenarioInput, TimestampedFrame
from counter_bmt_v2.data import ScenarioNetNNXLoader, build_vlm_frame_pack, render_scenario_frames
from counter_bmt_v2.perception import GPT4oPerceptionModel
from counter_bmt_v2.training.dag_cache_schema import (
    dag_to_cache_payload,
    schema_version_for_contract,
    validate_cache_payload,
)


def _normalize_sid(sid: str) -> str:
    text = str(sid).strip()
    if not text:
        return text
    stem = Path(text).stem
    if stem.startswith("sd_"):
        parts = stem.split("_")
        if len(parts) >= 2:
            return parts[-1]
    return stem


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if is_dataclass(obj):
        return _jsonify(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    return obj


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _parse_indices_file(path: Path) -> List[int]:
    out: List[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        for tok in text.split(","):
            t = tok.strip()
            if not t:
                continue
            out.append(int(t))
    return out


def _resolve_indices(loader_len: int, args: argparse.Namespace) -> List[int]:
    if args.indices_file:
        indices = _parse_indices_file(Path(args.indices_file))
    elif args.start_index is not None or args.end_index is not None:
        start = int(args.start_index if args.start_index is not None else 0)
        end = int(args.end_index if args.end_index is not None else loader_len)
        indices = list(range(max(0, start), min(loader_len, end)))
    else:
        n = min(int(args.n_scenarios), int(loader_len))
        rng = np.random.default_rng(int(args.seed))
        indices = [int(x) for x in rng.choice(loader_len, size=n, replace=False).tolist()]

    dedup: List[int] = []
    seen = set()
    for idx in indices:
        if idx < 0 or idx >= loader_len:
            continue
        if idx in seen:
            continue
        seen.add(idx)
        dedup.append(int(idx))
    return dedup


def _ego_trajectory(sample: Any) -> Optional[np.ndarray]:
    pos = np.asarray(sample.agent_position_xy, dtype=np.float32)
    valid = np.asarray(sample.agent_valid_mask, dtype=bool)
    if pos.ndim != 3 or valid.ndim != 2 or pos.shape[1] == 0:
        return None
    v = valid[:, 0]
    if not np.any(v):
        return None
    return pos[:, 0, :2].copy()


def _dag_summary_text(payload: Dict[str, Any]) -> str:
    nodes = list(payload.get("nodes", []))
    edges = list(payload.get("edges", []))
    cpts = payload.get("cpts", {})

    node_type_counts: Dict[str, int] = {}
    out_degree: Dict[str, int] = {}
    in_degree: Dict[str, int] = {}
    for n in nodes:
        t = str(n.get("node_type", "unknown"))
        nid = str(n.get("node_id", ""))
        node_type_counts[t] = node_type_counts.get(t, 0) + 1
        out_degree.setdefault(nid, 0)
        in_degree.setdefault(nid, 0)

    for e in edges:
        p = str(e.get("parent_id", ""))
        c = str(e.get("child_id", ""))
        out_degree[p] = out_degree.get(p, 0) + 1
        in_degree[c] = in_degree.get(c, 0) + 1

    top_parents = sorted(out_degree.items(), key=lambda kv: kv[1], reverse=True)[:5]
    top_children = sorted(in_degree.items(), key=lambda kv: kv[1], reverse=True)[:5]

    lines = []
    lines.append(f"scenario_id: {payload.get('scenario_id')}")
    lines.append(f"schema_version: {payload.get('schema_version')}")
    lines.append(f"nodes: {len(nodes)}")
    lines.append(f"edges: {len(edges)}")
    lines.append("")
    lines.append("node_type_counts:")
    for k, v in sorted(node_type_counts.items()):
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("top_out_degree:")
    for nid, deg in top_parents:
        lines.append(f"  - {nid}: {deg}")
    lines.append("")
    lines.append("top_in_degree:")
    for nid, deg in top_children:
        lines.append(f"  - {nid}: {deg}")
    lines.append("")
    lines.append("cpt_nodes:")
    if isinstance(cpts, dict):
        for k in sorted(cpts.keys()):
            lines.append(f"  - {k}")
    lines.append("")
    lines.append("edges:")
    for e in edges:
        lines.append(
            "  - "
            f"{e.get('parent_id')} -> {e.get('child_id')} "
            f"[confidence={float(e.get('confidence', 0.0)):.3f}, mechanism={e.get('mechanism', '')}]"
        )
    return "\n".join(lines) + "\n"


def _write_preview(
    out_dir: Path,
    *,
    preview_ids: Sequence[str],
    scenario_results: Dict[str, Dict[str, Any]],
) -> Path:
    md_path = out_dir / "preview.md"
    lines: List[str] = []
    lines.append("# DAG Preview")
    lines.append("")
    if not preview_ids:
        lines.append("No successful scenarios available for preview.")
    for sid in preview_ids:
        r = scenario_results.get(sid, {})
        ex_rel = f"examples/{sid}"
        cache_rel = f"cache/{sid}.json"
        lines.append(f"## {sid}")
        lines.append(f"- status: `{r.get('status', 'unknown')}`")
        lines.append(f"- nodes: `{r.get('n_nodes', 0)}`")
        lines.append(f"- edges: `{r.get('n_edges', 0)}`")
        lines.append(f"- frames_raw: `{r.get('n_frames_raw', 0)}`")
        lines.append(f"- frames_vlm: `{r.get('n_frames_vlm', 0)}`")
        lines.append(f"- cache: `{cache_rel}`")
        lines.append(f"- summary: `{ex_rel}/dag_summary.txt`")
        lines.append(f"- features: `{ex_rel}/features.json`")
        lines.append(f"- frames_raw: `{ex_rel}/frames_raw/`")
        lines.append(f"- frames_vlm: `{ex_rel}/frames_vlm/`")
        lines.append(f"- frame_manifest: `{ex_rel}/frame_manifest.json`")
        lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build v2 DAG cache directly from ScenarioNet with PromptBN.")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--n-scenarios", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--indices-file", type=str, default="")
    p.add_argument("--start-index", type=int, default=None)
    p.add_argument("--end-index", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--max-agents-render", type=int, default=64)
    p.add_argument("--annotate-vlm-frames", dest="annotate_vlm_frames", action="store_true")
    p.add_argument("--no-annotate-vlm-frames", dest="annotate_vlm_frames", action="store_false")
    p.set_defaults(annotate_vlm_frames=True)
    p.add_argument(
        "--annotation-style",
        type=str,
        default="banner+legend",
        choices=["banner", "banner+legend"],
    )
    p.add_argument("--ego-color-hint", type=str, default="green")
    p.add_argument("--include-ego-context-text", dest="include_ego_context_text", action="store_true")
    p.add_argument("--no-include-ego-context-text", dest="include_ego_context_text", action="store_false")
    p.set_defaults(include_ego_context_text=True)
    p.add_argument("--dual-view", dest="dual_view", action="store_true")
    p.add_argument("--no-dual-view", dest="dual_view", action="store_false")
    p.set_defaults(dual_view=False)
    p.add_argument("--add-ego-inset", dest="add_ego_inset", action="store_true")
    p.add_argument("--no-add-ego-inset", dest="add_ego_inset", action="store_false")
    p.set_defaults(add_ego_inset=True)
    p.add_argument(
        "--dual-view-mode",
        type=str,
        default="global_plus_ego_tensor",
        choices=["global_plus_ego_tensor"],
    )
    p.add_argument(
        "--frame-renderer",
        type=str,
        default="scenarionet",
        choices=["scenarionet", "tensor", "auto"],
        help="Frame renderer backend for VLM inputs.",
    )
    p.add_argument("--render-film-size", type=int, default=1200)
    p.add_argument("--render-screen-size", type=int, default=800)
    p.add_argument("--model", type=str, default="gpt-4o")
    p.add_argument("--api-key", type=str, default="")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-backoff-sec", type=float, default=2.0)
    p.add_argument("--continue-on-error", dest="continue_on_error", action="store_true")
    p.add_argument("--no-continue-on-error", dest="continue_on_error", action="store_false")
    p.set_defaults(continue_on_error=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--preview-count", type=int, default=3)
    p.add_argument("--strict-promptbn", dest="strict_promptbn", action="store_true")
    p.add_argument("--no-strict-promptbn", dest="strict_promptbn", action="store_false")
    p.set_defaults(strict_promptbn=True)
    p.add_argument("--save-raw-llm", action="store_true")
    p.add_argument(
        "--dag-contract",
        type=str,
        default="maneuver_outcome_v1",
        choices=["maneuver_outcome_v1", "compact10"],
    )
    p.add_argument("--dag-contract-mode", type=str, default="hard", choices=["hard"])
    return p.parse_args()


def _setup_scenarionet_renderer(
    *,
    data_dir: str,
    frame_renderer: str,
    film_size: int,
    screen_size: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "available": False,
        "visualizer": None,
        "sid_to_index": {},
        "sid_to_indices": {},
        "path_to_index": {},
        "name_to_indices": {},
        "env_index_by_name": {},
        "env_index_by_relpath": {},
        "env_indices_by_sid": {},
        "error": "",
    }
    if frame_renderer not in {"scenarionet", "auto"}:
        return out
    try:
        # Reduce very verbose file listing logs emitted by the legacy helper.
        logging.getLogger("counter_bmt.scenarionet_visualizer").setLevel(logging.WARNING)
        from counter_bmt.scenarionet_visualizer import ScenarioNetVisualizer

        visualizer = ScenarioNetVisualizer(
            data_dir=data_dir,
            film_size=(int(film_size), int(film_size)),
            screen_size=(int(screen_size), int(screen_size)),
        )
        db = visualizer.db
        sid_to_index: Dict[str, int] = {}
        sid_to_indices: Dict[str, List[int]] = {}
        path_to_index: Dict[str, int] = {}
        name_to_indices: Dict[str, List[int]] = {}
        for i, sid in enumerate(db.scenario_ids):
            sid_norm = _normalize_sid(str(sid))
            if sid_norm not in sid_to_index:
                sid_to_index[sid_norm] = int(i)
            sid_to_indices.setdefault(sid_norm, []).append(int(i))
        for i, p in enumerate(db.scenario_files):
            rp = str(Path(p).resolve())
            path_to_index[rp] = int(i)
            nm = Path(p).name
            name_to_indices.setdefault(nm, []).append(int(i))
        out["available"] = True
        out["visualizer"] = visualizer
        out["sid_to_index"] = sid_to_index
        out["sid_to_indices"] = sid_to_indices
        out["path_to_index"] = path_to_index
        out["name_to_indices"] = name_to_indices

        # ScenarioEnv indexing follows dataset_summary key order, which can differ
        # from visualizer.db.scenario_files sorted order.
        summary_path = Path(data_dir) / "dataset_summary.pkl"
        if summary_path.exists():
            try:
                summary = pickle.loads(summary_path.read_bytes())
                if isinstance(summary, dict):
                    env_index_by_name: Dict[str, int] = {}
                    env_index_by_relpath: Dict[str, int] = {}
                    env_indices_by_sid: Dict[str, List[int]] = {}
                    for env_idx, key in enumerate(summary.keys()):
                        key_str = str(key)
                        key_path = Path(key_str)
                        key_name = key_path.name
                        if key_name not in env_index_by_name:
                            env_index_by_name[key_name] = int(env_idx)
                        rel_norm = key_path.as_posix()
                        if rel_norm not in env_index_by_relpath:
                            env_index_by_relpath[rel_norm] = int(env_idx)
                        sid_norm = _normalize_sid(key_name)
                        env_indices_by_sid.setdefault(sid_norm, []).append(int(env_idx))
                    out["env_index_by_name"] = env_index_by_name
                    out["env_index_by_relpath"] = env_index_by_relpath
                    out["env_indices_by_sid"] = env_indices_by_sid
            except Exception as exc:
                logging.getLogger(__name__).warning("Failed to parse dataset_summary.pkl: %s", exc)
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _render_frames_for_scene(
    *,
    args: argparse.Namespace,
    sample: Any,
    sample_path: Path,
    loader_index: int,
    frames_dir: Path,
    scn_state: Dict[str, Any],
) -> tuple[List[TimestampedFrame], Optional[np.ndarray], str, str, int]:
    requested = str(args.frame_renderer)
    sid = str(sample.scenario_id)
    sid_norm = _normalize_sid(sid)

    if requested in {"scenarionet", "auto"}:
        if not bool(scn_state.get("available")):
            if requested == "scenarionet":
                raise RuntimeError(
                    "ScenarioNet renderer requested but unavailable: "
                    f"{scn_state.get('error', 'unknown import error')}"
                )
        else:
            sid_to_index = dict(scn_state.get("sid_to_index", {}))
            sid_to_indices = dict(scn_state.get("sid_to_indices", {}))
            path_to_index = dict(scn_state.get("path_to_index", {}))
            name_to_indices = dict(scn_state.get("name_to_indices", {}))
            env_index_by_name = dict(scn_state.get("env_index_by_name", {}))
            env_index_by_relpath = dict(scn_state.get("env_index_by_relpath", {}))
            env_indices_by_sid = dict(scn_state.get("env_indices_by_sid", {}))
            visualizer = scn_state["visualizer"]
            data_root = Path(args.data_dir).resolve()

            rp = str(sample_path.resolve())
            candidates: List[int] = []
            # Preferred: ScenarioEnv index space from dataset_summary order.
            try:
                rel = sample_path.resolve().relative_to(data_root).as_posix()
            except Exception:
                rel = sample_path.name
            if rel in env_index_by_relpath:
                candidates.append(int(env_index_by_relpath[rel]))
            if sample_path.name in env_index_by_name:
                candidates.append(int(env_index_by_name[sample_path.name]))
            for idx in env_indices_by_sid.get(sid_norm, []):
                candidates.append(int(idx))
            # Fallbacks from visualizer db ordering.
            for idx in sid_to_indices.get(sid_norm, []):
                candidates.append(int(idx))
            rp_idx = path_to_index.get(rp)
            if rp_idx is not None:
                candidates.append(int(rp_idx))
            name_matches = name_to_indices.get(sample_path.name, [])
            for idx in name_matches:
                candidates.append(int(idx))
            if sid_norm in sid_to_index:
                candidates.append(int(sid_to_index[sid_norm]))
            candidates.append(int(loader_index))
            # Preserve order while removing duplicates.
            dedup: List[int] = []
            seen = set()
            for idx in candidates:
                if idx in seen:
                    continue
                seen.add(idx)
                dedup.append(int(idx))

            mismatch_details: List[str] = []
            try:
                for render_idx in dedup:
                    saved_images, trajectory, sid_render = visualizer.render_scenario(
                        scenario_index=int(render_idx),
                        num_frames=int(args.num_frames),
                        output_dir=str(frames_dir),
                    )
                    sid_render_norm = _normalize_sid(str(sid_render))
                    if sid_render_norm != sid_norm:
                        mismatch_details.append(
                            f"idx={int(render_idx)} rendered_sid={sid_render_norm} expected_sid={sid_norm}"
                        )
                        continue
                    frames = [TimestampedFrame(path=str(p), timestamp_s=float(t)) for p, t in saved_images]
                    trj = (
                        np.asarray(trajectory, dtype=np.float32)
                        if trajectory is not None
                        else np.zeros((0, 4), dtype=np.float32)
                    )
                    ego_xy = (
                        trj[:, :2].copy()
                        if trj.ndim == 2 and trj.shape[1] >= 2 and trj.shape[0] > 0
                        else _ego_trajectory(sample)
                    )
                    return frames, ego_xy, "scenarionet", str(sid_render), int(render_idx)
                raise RuntimeError(
                    "ScenarioNet render scenario_id mismatch across all candidate indices. "
                    f"expected_sid={sid_norm}; candidates={dedup}; details={mismatch_details[:5]}"
                )
            except Exception:
                if requested == "scenarionet":
                    raise

    # Tensor renderer path (explicit or auto fallback).
    frames = render_scenario_frames(
        sample,
        frames_dir,
        num_frames=int(args.num_frames),
        max_agents=int(args.max_agents_render),
    )
    return frames, _ego_trajectory(sample), "tensor", sid, int(loader_index)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    examples_dir = out_dir / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = out_dir / "results.jsonl"

    loader = ScenarioNetNNXLoader(args.data_dir)
    indices = _resolve_indices(len(loader), args)
    if not indices:
        raise ValueError("No valid scenario indices resolved from inputs.")
    scn_state = _setup_scenarionet_renderer(
        data_dir=str(args.data_dir),
        frame_renderer=str(args.frame_renderer),
        film_size=int(args.render_film_size),
        screen_size=int(args.render_screen_size),
    )
    if str(args.frame_renderer) == "scenarionet" and not bool(scn_state.get("available")):
        raise RuntimeError(
            "frame_renderer=scenarionet but ScenarioNet renderer is unavailable: "
            f"{scn_state.get('error', 'unknown error')}"
        )
    if str(args.frame_renderer) == "auto" and not bool(scn_state.get("available")):
        print(
            "[dag-cache-v2] ScenarioNet renderer unavailable; auto mode will use tensor renderer. "
            f"reason={scn_state.get('error', 'unknown')}",
            flush=True,
        )

    max_frames_for_perception = int(max(1, int(args.num_frames) * (2 if bool(args.dual_view) else 1)))
    perception = GPT4oPerceptionModel(
        model=str(args.model),
        api_key=(args.api_key or None),
        max_frames=max_frames_for_perception,
        use_mock_fallback=False,
    )
    dag_builder = PromptBNDAGBuilder(
        model=str(args.model),
        api_key=(args.api_key or None),
        max_retries=4,
        use_simple_fallback=not bool(args.strict_promptbn),
        dag_contract=str(args.dag_contract),
        dag_contract_mode=str(args.dag_contract_mode),
    )
    dag_contract_cfg = DAGContractConfig(
        name=str(args.dag_contract),
        mode=str(args.dag_contract_mode),
    )
    expected_cache_schema_version = schema_version_for_contract(str(args.dag_contract))

    started_ts = time.time()
    successful_ids: List[str] = []
    failed_ids: List[str] = []
    skipped_existing_ids: List[str] = []
    scenario_results: Dict[str, Dict[str, Any]] = {}
    failure_reasons: Dict[str, int] = {}
    contract_pass_count = 0
    contract_fail_count = 0
    contract_nodes_after: List[int] = []
    contract_edges_after: List[int] = []
    contract_norm_counts: Dict[str, int] = {}
    maneuver_node_counts: List[int] = []
    maneuver_interval_complete_counts: List[float] = []

    print(
        f"[dag-cache-v2] dataset={args.data_dir} total={len(loader)} selected={len(indices)} "
        f"strict_promptbn={bool(args.strict_promptbn)} frame_renderer={args.frame_renderer}"
    , flush=True)

    try:
        for row_i, idx in enumerate(indices, start=1):
            row_start = time.time()
            status = "failed"
            sid = f"idx_{idx:06d}"
            last_error = ""
            attempts = 0
            n_nodes = 0
            n_edges = 0
            n_frames_raw = 0
            n_frames_vlm = 0
            renderer_used = ""
            rendered_sid = ""
            rendered_index = -1
            contract_pass = False
            contract_violation_counts: Dict[str, int] = {}
            contract_report_summary: Dict[str, Any] = {}
            maneuver_nodes = 0
            maneuver_interval_complete_rate = 0.0

            for attempt in range(1, max(1, int(args.max_retries)) + 1):
                attempts = attempt
                ex_dir: Optional[Path] = None
                try:
                    sample = loader.load(int(idx))
                    sample_path = Path(loader.files[int(idx)])
                    sid = str(sample.scenario_id)
                    cache_path = cache_dir / f"{sid}.json"
                    if cache_path.is_file() and not bool(args.overwrite):
                        status = "skipped_existing"
                        skipped_existing_ids.append(sid)
                        break

                    ex_dir = examples_dir / sid
                    ex_dir.mkdir(parents=True, exist_ok=True)
                    raw_frames_dir = ex_dir / "frames_raw"
                    vlm_frames_dir = ex_dir / "frames_vlm"
                    raw_frames, ego_xy, renderer_used, rendered_sid, rendered_index = _render_frames_for_scene(
                        args=args,
                        sample=sample,
                        sample_path=sample_path,
                        loader_index=int(idx),
                        frames_dir=raw_frames_dir,
                        scn_state=scn_state,
                    )
                    n_frames_raw = int(len(raw_frames))
                    vlm_frames, frame_manifest, vlm_context_text = build_vlm_frame_pack(
                        sample=sample,
                        raw_frames=raw_frames,
                        out_dir=vlm_frames_dir,
                        max_agents=int(args.max_agents_render),
                        annotate_vlm_frames=bool(args.annotate_vlm_frames),
                        annotation_style=str(args.annotation_style),
                        ego_color_hint=str(args.ego_color_hint),
                        include_ego_context_text=bool(args.include_ego_context_text),
                        dual_view=bool(args.dual_view),
                        dual_view_mode=str(args.dual_view_mode),
                        add_ego_inset=bool(args.add_ego_inset),
                    )
                    n_frames_vlm = int(len(vlm_frames))
                    frame_manifest_path = ex_dir / "frame_manifest.json"
                    frame_manifest_path.write_text(json.dumps(frame_manifest, indent=2), encoding="utf-8")
                    (ex_dir / "vlm_prompt_context.txt").write_text((vlm_context_text or "") + "\n", encoding="utf-8")

                    scene = ScenarioInput(
                        scenario_id=sid,
                        frames=vlm_frames,
                        ego_trajectory_xy=ego_xy,
                        metadata={
                            "source": "scenarionet_v2",
                            "data_dir": str(args.data_dir),
                            "loader_index": int(idx),
                            "loader_file": str(sample_path),
                            "frame_renderer": renderer_used,
                            "rendered_scenario_id": str(rendered_sid),
                            "rendered_scenario_index": int(rendered_index),
                            "ego_color_hint": str(args.ego_color_hint),
                            "dual_view_enabled": bool(args.dual_view),
                            "dual_view_mode": str(args.dual_view_mode),
                            "add_ego_inset": bool(args.add_ego_inset),
                            "annotation_enabled": bool(args.annotate_vlm_frames),
                            "annotation_style": str(args.annotation_style),
                            "frame_manifest_path": str(frame_manifest_path),
                            "frame_manifest": frame_manifest,
                            "vlm_context_text": vlm_context_text if bool(args.include_ego_context_text) else "",
                        },
                    )

                    features = perception.extract(scene)
                    dag = dag_builder.build(scene, features)
                    payload = dag_to_cache_payload(dag)
                    payload["schema_version"] = expected_cache_schema_version
                    meta = payload.get("metadata", {})
                    if not isinstance(meta, dict):
                        meta = {"metadata_raw": str(meta)}
                    meta.update(
                        {
                        "source": "counter_bmt_v2_promptbn",
                        "model": str(args.model),
                        "strict_promptbn": bool(args.strict_promptbn),
                        "dag_contract": str(args.dag_contract),
                        "dag_contract_mode": str(args.dag_contract_mode),
                        "loader_index": int(idx),
                        "generated_at_unix_s": float(time.time()),
                        "frame_renderer": renderer_used,
                        "rendered_scenario_id": str(rendered_sid),
                        "rendered_scenario_index": int(rendered_index),
                        "annotate_vlm_frames": bool(args.annotate_vlm_frames),
                        "annotation_style": str(args.annotation_style),
                        "dual_view": bool(args.dual_view),
                        "dual_view_mode": str(args.dual_view_mode),
                        "add_ego_inset": bool(args.add_ego_inset),
                        "ego_color_hint": str(args.ego_color_hint),
                        "n_frames_raw": int(n_frames_raw),
                        "n_frames_vlm": int(n_frames_vlm),
                        }
                    )
                    payload["metadata"] = meta
                    contract_ok, payload, contract_report = enforce_dag_contract(payload, config=dag_contract_cfg)
                    contract_report_summary = contract_report.summary()
                    contract_violation_counts = dict(contract_report.violation_counts)
                    contract_report_path = ex_dir / "dag_contract_report.json"
                    contract_report_path.write_text(json.dumps(contract_report_summary, indent=2), encoding="utf-8")
                    if not contract_ok:
                        raise RuntimeError(
                            "DAG contract hard-enforcement failed: "
                            f"violations={contract_violation_counts}"
                        )
                    contract_pass = True
                    contract_pass_count += 1
                    contract_nodes_after.append(int(contract_report.after_nodes))
                    contract_edges_after.append(int(contract_report.after_edges))
                    for k, v in contract_report.normalization_counts.items():
                        contract_norm_counts[str(k)] = int(contract_norm_counts.get(str(k), 0) + int(v))

                    if not validate_cache_payload(payload):
                        raise RuntimeError("Generated DAG payload failed schema validation.")

                    node_list = payload.get("nodes", []) if isinstance(payload.get("nodes", []), list) else []
                    maneuver_nodes = int(
                        sum(1 for n in node_list if str(n.get("node_type", "")).strip().lower() == "maneuver")
                    )
                    interval_complete = 0
                    if maneuver_nodes > 0:
                        for n in node_list:
                            if str(n.get("node_type", "")).strip().lower() != "maneuver":
                                continue
                            md = n.get("metadata", {})
                            if not isinstance(md, dict):
                                continue
                            has_all = all(k in md for k in ("start_s", "end_s", "duration_s", "mid_s"))
                            interval_complete += int(has_all)
                        maneuver_interval_complete_rate = float(interval_complete / max(1, maneuver_nodes))

                    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

                    feat_dict = _jsonify(features)
                    if not bool(args.save_raw_llm):
                        raw = feat_dict.get("raw", {})
                        if isinstance(raw, dict):
                            raw.pop("raw_response", None)
                    feat_dict["prompt_context"] = {
                        "ego_color_hint": str(args.ego_color_hint),
                        "dual_view_enabled": bool(args.dual_view),
                        "dual_view_mode": str(args.dual_view_mode),
                        "add_ego_inset": bool(args.add_ego_inset),
                        "annotation_enabled": bool(args.annotate_vlm_frames),
                        "annotation_style": str(args.annotation_style),
                        "include_ego_context_text": bool(args.include_ego_context_text),
                        "frame_manifest_path": str(frame_manifest_path),
                        "n_frames_raw": int(n_frames_raw),
                        "n_frames_vlm": int(n_frames_vlm),
                        "dag_contract": str(args.dag_contract),
                        "dag_contract_mode": str(args.dag_contract_mode),
                        "contract_report_path": str(ex_dir / "dag_contract_report.json"),
                    }
                    (ex_dir / "features.json").write_text(json.dumps(feat_dict, indent=2), encoding="utf-8")
                    if bool(args.save_raw_llm):
                        raw = feat_dict.get("raw", {})
                        raw_text = ""
                        if isinstance(raw, dict):
                            raw_text = str(raw.get("raw_response", ""))
                        if raw_text:
                            (ex_dir / "perception_raw_response.txt").write_text(raw_text, encoding="utf-8")

                    (ex_dir / "dag.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    (ex_dir / "dag_summary.txt").write_text(_dag_summary_text(payload), encoding="utf-8")

                    n_nodes = int(len(payload.get("nodes", [])))
                    n_edges = int(len(payload.get("edges", [])))
                    maneuver_node_counts.append(int(maneuver_nodes))
                    maneuver_interval_complete_counts.append(float(maneuver_interval_complete_rate))
                    status = "success"
                    successful_ids.append(sid)
                    break
                except Exception as exc:
                    last_error = str(exc)
                    if ex_dir is not None:
                        try:
                            (ex_dir / f"attempt_{attempt}_error.txt").write_text(
                                traceback.format_exc(),
                                encoding="utf-8",
                            )
                            raw_text = str(getattr(exc, "raw_response", "") or "")
                            if raw_text:
                                (ex_dir / f"attempt_{attempt}_raw_response.txt").write_text(
                                    raw_text,
                                    encoding="utf-8",
                                )
                            raw_excerpt = str(getattr(exc, "raw_excerpt", "") or "")
                            if raw_excerpt:
                                (ex_dir / f"attempt_{attempt}_error_excerpt.txt").write_text(
                                    raw_excerpt + "\n",
                                    encoding="utf-8",
                                )
                        except Exception:
                            pass
                    if attempt < int(args.max_retries):
                        time.sleep(float(args.retry_backoff_sec) * float(2 ** (attempt - 1)))
                        continue
                    status = "failed"

            duration = float(time.time() - row_start)
            if status == "failed":
                failed_ids.append(sid)
                key = (last_error or "unknown_error").splitlines()[0][:200]
                failure_reasons[key] = int(failure_reasons.get(key, 0) + 1)
                if "DAG contract hard-enforcement failed" in (last_error or ""):
                    contract_fail_count += 1

            record = {
                "row": int(row_i),
                "loader_index": int(idx),
                "scenario_id": sid,
                "status": status,
                "attempts": int(attempts),
                "duration_sec": duration,
                "n_nodes": int(n_nodes),
                "n_edges": int(n_edges),
                "n_frames_raw": int(n_frames_raw),
                "n_frames_vlm": int(n_frames_vlm),
                "frame_renderer": renderer_used,
                "rendered_scenario_id": str(rendered_sid),
                "rendered_scenario_index": int(rendered_index),
                "error": last_error,
                "contract_pass": bool(contract_pass),
                "contract_violation_counts": contract_violation_counts,
                "contract_report": contract_report_summary,
                "schema_version": str(expected_cache_schema_version),
                "maneuver_nodes": int(maneuver_nodes),
                "maneuver_interval_complete_rate": float(maneuver_interval_complete_rate),
            }
            _append_jsonl(results_jsonl, record)
            scenario_results[sid] = record

            msg = (
                f"[{row_i}/{len(indices)}] idx={idx} sid={sid} status={status} "
                f"attempts={attempts} nodes={n_nodes} edges={n_edges} "
                f"frames_raw={n_frames_raw} frames_vlm={n_frames_vlm} "
                f"renderer={renderer_used or 'n/a'} "
                f"rendered_sid={rendered_sid or 'n/a'} dt={duration:.1f}s"
            )
            if last_error and status == "failed":
                msg += f" error={last_error[:160]}"
            print(msg, flush=True)

            if status == "failed" and not bool(args.continue_on_error):
                break
    finally:
        vis = scn_state.get("visualizer")
        if vis is not None:
            try:
                vis.close()
            except Exception:
                pass

    # Preview markdown for random successful scenarios.
    rng = np.random.default_rng(int(args.seed) + 1337)
    k = min(int(args.preview_count), len(successful_ids))
    preview_ids = [str(x) for x in rng.choice(successful_ids, size=k, replace=False).tolist()] if k > 0 else []
    preview_path = _write_preview(out_dir, preview_ids=preview_ids, scenario_results=scenario_results)

    elapsed = float(time.time() - started_ts)
    manifest = {
        "schema_version": "counter_bmt_v2_dag_cache_build_manifest_v1",
        "config": {
            "data_dir": str(args.data_dir),
            "out_dir": str(args.out_dir),
            "n_scenarios": int(args.n_scenarios),
            "seed": int(args.seed),
            "indices_file": str(args.indices_file or ""),
            "start_index": args.start_index,
            "end_index": args.end_index,
            "num_frames": int(args.num_frames),
            "max_agents_render": int(args.max_agents_render),
            "annotate_vlm_frames": bool(args.annotate_vlm_frames),
            "annotation_style": str(args.annotation_style),
            "ego_color_hint": str(args.ego_color_hint),
            "include_ego_context_text": bool(args.include_ego_context_text),
            "dual_view": bool(args.dual_view),
            "dual_view_mode": str(args.dual_view_mode),
            "add_ego_inset": bool(args.add_ego_inset),
            "frame_renderer": str(args.frame_renderer),
            "render_film_size": int(args.render_film_size),
            "render_screen_size": int(args.render_screen_size),
            "model": str(args.model),
            "perception_max_frames": int(max_frames_for_perception),
            "strict_promptbn": bool(args.strict_promptbn),
            "max_retries": int(args.max_retries),
            "retry_backoff_sec": float(args.retry_backoff_sec),
            "continue_on_error": bool(args.continue_on_error),
            "overwrite": bool(args.overwrite),
            "save_raw_llm": bool(args.save_raw_llm),
            "dag_contract": str(args.dag_contract),
            "dag_contract_mode": str(args.dag_contract_mode),
            "cache_schema_version": str(expected_cache_schema_version),
        },
        "selected_loader_indices": [int(i) for i in indices],
        "counts": {
            "selected": int(len(indices)),
            "success": int(len(successful_ids)),
            "failed": int(len(failed_ids)),
            "skipped_existing": int(len(skipped_existing_ids)),
            "contract_pass_count": int(contract_pass_count),
            "contract_fail_count": int(contract_fail_count),
        },
        "contract_summary": {
            "avg_nodes_after": float(np.mean(contract_nodes_after)) if contract_nodes_after else 0.0,
            "avg_edges_after": float(np.mean(contract_edges_after)) if contract_edges_after else 0.0,
            "normalization_stats": contract_norm_counts,
            "avg_maneuver_nodes": float(np.mean(maneuver_node_counts)) if maneuver_node_counts else 0.0,
            "avg_maneuver_interval_complete_rate": (
                float(np.mean(maneuver_interval_complete_counts)) if maneuver_interval_complete_counts else 0.0
            ),
        },
        "failure_reasons": failure_reasons,
        "latency": {
            "total_elapsed_sec": elapsed,
            "avg_sec_per_selected": float(elapsed / max(1, len(indices))),
        },
        "artifacts": {
            "cache_dir": str(cache_dir),
            "examples_dir": str(examples_dir),
            "results_jsonl": str(results_jsonl),
            "preview_md": str(preview_path),
        },
        "preview_scenarios": preview_ids,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("", flush=True)
    print(json.dumps(manifest["counts"], indent=2), flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Preview:  {preview_path}", flush=True)
    print("", flush=True)
    print("Training command (cache-only Stage B/C):", flush=True)
    print(
        "python -m counter_bmt_v2.cli.train_nnx_bmt_dag_latent "
        "--dag-source-mode cache "
        f"--dag-cache-dir {cache_dir} "
        f"--dag-expected-schema {'v3_maneuver_outcome' if str(args.dag_contract) == 'maneuver_outcome_v1' else 'v2_compact10'} "
        "--dag-cache-strict ..."
    , flush=True)

    return 0 if len(successful_ids) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
