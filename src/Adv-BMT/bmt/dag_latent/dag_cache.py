"""Cache-backed DAG tensor attachment for legacy DAG-latent training."""

from __future__ import annotations

import math
import pathlib
import sys
from typing import Any, Dict, Iterable, List

import torch

from bmt.utils import REPO_ROOT

from .config import get_dag_latent_block

# The legacy launcher typically sets `PYTHONPATH=src/Adv-BMT`, which does not
# include the workspace-level `src/` package root where `counter_bmt_v2` lives.
_WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[4]
_SRC_ROOT = _WORKSPACE_ROOT / "src"
if _SRC_ROOT.is_dir():
    src_root_str = str(_SRC_ROOT)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

from counter_bmt_v2.training.dag_cache import DAGCacheReader
from counter_bmt_v2.training.dag_cache_schema import (
    SCHEMA_VERSION_V2_COMPACT10,
    SCHEMA_VERSION_V3_MANEUVER_OUTCOME,
)
from counter_bmt_v2.training.dag_tensorize import tensorize_dag_batch


_GRAPH_TENSOR_DTYPES = {
    "dag_node_feat": torch.float32,
    "dag_node_mask": torch.bool,
    "dag_edge_src": torch.long,
    "dag_edge_dst": torch.long,
    "dag_edge_feat": torch.float32,
    "dag_edge_mask": torch.bool,
    "dag_global_feat": torch.float32,
}

_TIME_GUIDANCE_MANEUVER_ORDER = (
    "straight",
    "left_turn",
    "right_turn",
    "lane_change_left",
    "lane_change_right",
    "accelerate",
    "decelerate",
    "stop",
)
_TIME_GUIDANCE_MANEUVER_TO_ID = {
    name: idx for idx, name in enumerate(_TIME_GUIDANCE_MANEUVER_ORDER)
}
_TIME_GUIDANCE_PHASE_OFFSET = len(_TIME_GUIDANCE_MANEUVER_ORDER)
_TIME_GUIDANCE_SUMMARY_OFFSET = 2 * len(_TIME_GUIDANCE_MANEUVER_ORDER)
_TIME_GUIDANCE_OUTCOME_ORDER = (
    ("collision_outcome", "collision_avoided"),
    ("collision_outcome", "collision_possible"),
    ("progress_outcome", "progress_good"),
    ("progress_outcome", "progress_limited"),
    ("compliance_outcome", "compliant"),
    ("compliance_outcome", "violation_possible"),
)
_TIME_GUIDANCE_OUTCOME_TO_ID = {
    key: idx for idx, key in enumerate(_TIME_GUIDANCE_OUTCOME_ORDER)
}
_TIME_GUIDANCE_SUMMARY_DIM = 2
_TIME_GUIDANCE_BASE_DIM = (
    len(_TIME_GUIDANCE_MANEUVER_ORDER)
    + len(_TIME_GUIDANCE_MANEUVER_ORDER)
    + _TIME_GUIDANCE_SUMMARY_DIM
)
_MANEUVER_TOKEN_TIMING_DIM = 4
_MANEUVER_TOKEN_SUMMARY_DIM = 2
_MANEUVER_TOKEN_BASE_DIM = (
    len(_TIME_GUIDANCE_MANEUVER_ORDER)
    + _MANEUVER_TOKEN_TIMING_DIM
    + len(_TIME_GUIDANCE_OUTCOME_ORDER)
    + _MANEUVER_TOKEN_SUMMARY_DIM
)


def resolve_expected_schema_name(mode: str) -> str:
    key = str(mode).strip().lower()
    if key in {"", "any"}:
        return "any"
    if key in {"v2_compact10", SCHEMA_VERSION_V2_COMPACT10}:
        return SCHEMA_VERSION_V2_COMPACT10
    if key in {"v3_maneuver_outcome", SCHEMA_VERSION_V3_MANEUVER_OUTCOME}:
        return SCHEMA_VERSION_V3_MANEUVER_OUTCOME
    raise ValueError(
        "Unsupported DAG expected schema value: "
        f"{mode!r}. Expected one of: any, v2_compact10, v3_maneuver_outcome."
    )


def _resolve_cache_dir(cache_dir: str) -> str:
    path = pathlib.Path(str(cache_dir)).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return str(path)


def _empty_payload(scenario_id: str, expected_schema: str) -> Dict[str, Any]:
    schema_version = (
        SCHEMA_VERSION_V3_MANEUVER_OUTCOME
        if expected_schema in {"any", SCHEMA_VERSION_V3_MANEUVER_OUTCOME}
        else SCHEMA_VERSION_V2_COMPACT10
    )
    return {
        "schema_version": schema_version,
        "scenario_id": str(scenario_id),
        "nodes": [],
        "edges": [],
        "cpts": {},
        "metadata": {"source": "null"},
    }


def _normalize_token(text: Any) -> str:
    return str(text).strip().lower().replace("-", "_").replace(" ", "_")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _default_time_guidance_step_dt(config: Any) -> float:
    token_block = config.get("TOKENIZATION", {}) if hasattr(config, "get") else {}
    skipped = int(token_block.get("NUM_SKIPPED_STEPS", 5))
    return max(1e-3, 0.1 * float(skipped))


def _build_time_guidance_one(
    payload: Dict[str, Any],
    *,
    feature_dim: int,
    step_dt: float,
    min_steps: int,
    active_agg: str,
) -> Dict[str, torch.Tensor]:
    if feature_dim <= 0:
        raise ValueError(f"TIME_GUIDANCE_FEATURE_DIM must be positive, got {feature_dim}")

    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []

    maneuver_nodes: List[Dict[str, Any]] = []
    outcome_values: Dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("node_id", "")).strip()
        node_type = _normalize_token(node.get("node_type", ""))
        value = _normalize_token(node.get("value", ""))
        if node_type == "maneuver":
            maneuver_nodes.append(node)
        elif node_type == "outcome" and node_id:
            outcome_values[node_id] = value

    base_horizon_s = max(0.0, float(min_steps - 1) * float(step_dt))
    maneuver_end_vals: List[float] = []
    for node in maneuver_nodes:
        metadata = node.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        maneuver_end_vals.append(_to_float(metadata.get("end_s", 0.0), default=0.0))
    horizon_s = max([base_horizon_s, *maneuver_end_vals]) if maneuver_end_vals else base_horizon_s
    num_steps = max(int(min_steps), int(round(horizon_s / max(step_dt, 1e-3))) + 1)

    feat = torch.zeros((num_steps, feature_dim), dtype=torch.float32)
    active_mask = torch.zeros((num_steps,), dtype=torch.bool)

    if not maneuver_nodes:
        return {"dag_time_feat": feat, "dag_time_mask": active_mask}

    if str(active_agg).strip().lower() not in {"sum", "max", "mean"}:
        raise ValueError(
            "Unsupported DAG TIME_GUIDANCE_ACTIVE_AGG="
            f"{active_agg!r}. Expected one of: sum, max, mean."
        )

    supported_dim = min(feature_dim, _TIME_GUIDANCE_BASE_DIM)
    maneuver_slice = slice(0, min(len(_TIME_GUIDANCE_MANEUVER_ORDER), supported_dim))
    phase_start = min(_TIME_GUIDANCE_PHASE_OFFSET, supported_dim)
    phase_stop = min(_TIME_GUIDANCE_SUMMARY_OFFSET, supported_dim)
    phase_slice = slice(phase_start, phase_stop)
    summary_start = phase_slice.stop
    summary_stop = min(summary_start + _TIME_GUIDANCE_SUMMARY_DIM, supported_dim)

    active_counts = torch.zeros((num_steps,), dtype=torch.float32)
    edge_conf_sum = torch.zeros((num_steps,), dtype=torch.float32)
    edge_conf_count = torch.zeros((num_steps,), dtype=torch.float32)
    class_counts = torch.zeros((num_steps, len(_TIME_GUIDANCE_MANEUVER_ORDER)), dtype=torch.float32)

    edge_outcomes_by_parent: Dict[str, List[tuple[str, float]]] = {}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        parent_id = str(edge.get("parent_id", "")).strip()
        child_id = str(edge.get("child_id", "")).strip()
        if not parent_id or child_id not in outcome_values:
            continue
        conf = _to_float(edge.get("confidence", 0.5), default=0.5)
        edge_outcomes_by_parent.setdefault(parent_id, []).append((child_id, conf))

    time_axis = torch.arange(num_steps, dtype=torch.float32) * float(step_dt)
    half_step = 0.5 * float(step_dt)
    agg_mode = str(active_agg).strip().lower()
    default_outcome_edges = [(node_id, 1.0) for node_id in outcome_values.keys()]

    for node in maneuver_nodes:
        node_id = str(node.get("node_id", "")).strip()
        maneuver_value = _normalize_token(node.get("value", ""))
        class_idx = _TIME_GUIDANCE_MANEUVER_TO_ID.get(maneuver_value, None)
        metadata = node.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        start_s = _to_float(metadata.get("start_s", node.get("timestamp_s", 0.0)), default=0.0)
        end_s = _to_float(metadata.get("end_s", start_s), default=start_s)
        end_s = max(start_s, end_s)
        duration_s = max(float(step_dt), _to_float(metadata.get("duration_s", end_s - start_s), default=end_s - start_s))
        active = (time_axis >= float(start_s) - half_step) & (time_axis <= float(end_s) + half_step)
        if not bool(active.any()):
            nearest = int(torch.clamp(torch.round(torch.tensor(start_s / max(step_dt, 1e-3))), 0, num_steps - 1).item())
            active[nearest] = True

        active_idx = torch.where(active)[0]
        active_counts[active_idx] += 1.0
        active_mask[active_idx] = True
        progress_vals = ((time_axis[active_idx] - float(start_s)) / float(duration_s)).clamp(0.0, 1.0)

        if class_idx is not None and class_idx < maneuver_slice.stop:
            class_counts[active_idx, class_idx] += 1.0
            if agg_mode == "max":
                feat[active_idx, class_idx] = torch.maximum(
                    feat[active_idx, class_idx],
                    torch.ones_like(active_idx, dtype=torch.float32),
                )
                phase_col = phase_start + int(class_idx)
                if phase_col < phase_stop:
                    feat[active_idx, phase_col] = torch.maximum(
                        feat[active_idx, phase_col],
                        progress_vals,
                    )
            else:
                feat[active_idx, class_idx] += 1.0
                phase_col = phase_start + int(class_idx)
                if phase_col < phase_stop:
                    feat[active_idx, phase_col] += progress_vals

        outcome_edges = edge_outcomes_by_parent.get(node_id, default_outcome_edges)
        for child_id, confidence in outcome_edges:
            value = outcome_values.get(child_id, "")
            outcome_idx = _TIME_GUIDANCE_OUTCOME_TO_ID.get((child_id, value), None)
            if outcome_idx is None:
                continue
            edge_conf_sum[active_idx] += float(confidence)
            edge_conf_count[active_idx] += 1.0

    if agg_mode == "mean":
        class_denom = class_counts.clamp_min(1.0)
        if maneuver_slice.stop > maneuver_slice.start:
            feat[:, maneuver_slice] = feat[:, maneuver_slice] / class_denom[:, : maneuver_slice.stop]
        if phase_stop > phase_start:
            phase_width = phase_stop - phase_start
            feat[:, phase_slice] = feat[:, phase_slice] / class_denom[:, :phase_width]

    if summary_stop > summary_start:
        summary_values: List[torch.Tensor] = [
            active_mask.float(),
            torch.where(
                edge_conf_count > 0.0,
                edge_conf_sum / edge_conf_count.clamp_min(1.0),
                torch.zeros_like(edge_conf_sum),
            ),
        ]
        for offset, values in enumerate(summary_values[: summary_stop - summary_start]):
            feat[:, summary_start + offset] = values

    feat[~active_mask] = 0.0
    return {"dag_time_feat": feat, "dag_time_mask": active_mask}


def _build_maneuver_tokens_one(
    payload: Dict[str, Any],
    *,
    feature_dim: int,
    step_dt: float,
    min_steps: int,
) -> Dict[str, torch.Tensor]:
    if feature_dim <= 0:
        raise ValueError(f"MANEUVER_TOKEN_FEATURE_DIM must be positive, got {feature_dim}")

    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []

    maneuver_nodes: List[Dict[str, Any]] = []
    outcome_values: Dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = _normalize_token(node.get("node_type", ""))
        node_id = str(node.get("node_id", "")).strip()
        if node_type == "maneuver":
            maneuver_nodes.append(node)
        elif node_type == "outcome" and node_id:
            outcome_values[node_id] = _normalize_token(node.get("value", ""))

    if not maneuver_nodes:
        return {
            "dag_maneuver_feat": torch.zeros((0, feature_dim), dtype=torch.float32),
            "dag_maneuver_mask": torch.zeros((0,), dtype=torch.bool),
        }

    base_horizon_s = max(0.0, float(min_steps - 1) * float(step_dt))
    horizon_s = max(
        base_horizon_s,
        *[
            _to_float(
                (node.get("metadata", {}) if isinstance(node.get("metadata", {}), dict) else {}).get("end_s", node.get("timestamp_s", 0.0)),
                default=0.0,
            )
            for node in maneuver_nodes
        ],
    )
    horizon_s = max(horizon_s, max(step_dt, 1e-3))

    edge_outcomes_by_parent: Dict[str, List[tuple[str, float]]] = {}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        parent_id = str(edge.get("parent_id", "")).strip()
        child_id = str(edge.get("child_id", "")).strip()
        if not parent_id or child_id not in outcome_values:
            continue
        conf = _to_float(edge.get("confidence", 0.5), default=0.5)
        edge_outcomes_by_parent.setdefault(parent_id, []).append((child_id, conf))

    maneuver_nodes.sort(
        key=lambda node: (
            _to_float(
                (node.get("metadata", {}) if isinstance(node.get("metadata", {}), dict) else {}).get(
                    "start_s",
                    node.get("timestamp_s", 0.0),
                ),
                default=0.0,
            ),
            str(node.get("node_id", "")),
        )
    )

    feat = torch.zeros((len(maneuver_nodes), feature_dim), dtype=torch.float32)
    mask = torch.ones((len(maneuver_nodes),), dtype=torch.bool)
    supported_dim = min(feature_dim, _MANEUVER_TOKEN_BASE_DIM)
    class_stop = min(len(_TIME_GUIDANCE_MANEUVER_ORDER), supported_dim)
    timing_start = class_stop
    timing_stop = min(timing_start + _MANEUVER_TOKEN_TIMING_DIM, supported_dim)
    outcome_start = timing_stop
    outcome_stop = min(outcome_start + len(_TIME_GUIDANCE_OUTCOME_ORDER), supported_dim)
    summary_start = outcome_stop
    summary_stop = min(summary_start + _MANEUVER_TOKEN_SUMMARY_DIM, supported_dim)

    for token_idx, node in enumerate(maneuver_nodes):
        metadata = node.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        maneuver_value = _normalize_token(node.get("value", ""))
        class_idx = _TIME_GUIDANCE_MANEUVER_TO_ID.get(maneuver_value, None)
        start_s = _to_float(metadata.get("start_s", node.get("timestamp_s", 0.0)), default=0.0)
        end_s = max(start_s, _to_float(metadata.get("end_s", start_s), default=start_s))
        duration_s = max(0.0, _to_float(metadata.get("duration_s", end_s - start_s), default=end_s - start_s))
        mid_s = _to_float(metadata.get("mid_s", 0.5 * (start_s + end_s)), default=0.5 * (start_s + end_s))

        if class_idx is not None and class_idx < class_stop:
            feat[token_idx, class_idx] = 1.0

        timing_values = [
            start_s / horizon_s,
            end_s / horizon_s,
            duration_s / horizon_s,
            mid_s / horizon_s,
        ]
        for offset, value in enumerate(timing_values[: max(0, timing_stop - timing_start)]):
            feat[token_idx, timing_start + offset] = float(max(0.0, min(1.0, value)))

        outcome_edges = edge_outcomes_by_parent.get(str(node.get("node_id", "")).strip(), [])
        conf_values: List[float] = []
        for child_id, confidence in outcome_edges:
            value = outcome_values.get(child_id, "")
            outcome_idx = _TIME_GUIDANCE_OUTCOME_TO_ID.get((child_id, value), None)
            if outcome_idx is None:
                continue
            feat_col = outcome_start + int(outcome_idx)
            if feat_col >= outcome_stop:
                continue
            feat[token_idx, feat_col] = max(feat[token_idx, feat_col].item(), float(confidence))
            conf_values.append(float(confidence))

        summary_values = [
            float(token_idx) / float(max(1, len(maneuver_nodes) - 1)),
            sum(conf_values) / float(len(conf_values)) if conf_values else 0.0,
        ]
        for offset, value in enumerate(summary_values[: max(0, summary_stop - summary_start)]):
            feat[token_idx, summary_start + offset] = float(value)

    return {"dag_maneuver_feat": feat, "dag_maneuver_mask": mask}


class DAGCacheBatchBuilder:
    """Resolve DAG cache payloads by scenario id and tensorize them for legacy batches."""

    def __init__(
        self,
        config: Any,
        *,
        cache_dir_override: str | None = None,
        cache_strict_override: bool | None = None,
        expected_schema_override: str | None = None,
    ):
        dag_block = get_dag_latent_block(config)
        self.enabled = bool(dag_block.get("ENABLED", False))
        self.source_mode = str(dag_block.get("SOURCE_MODE", "")).strip().lower()
        cache_dir_value = dag_block.get("CACHE_DIR", "")
        if cache_dir_override is not None:
            cache_dir_value = cache_dir_override
        self.cache_dir = str(cache_dir_value).strip()

        cache_strict_value = dag_block.get("CACHE_STRICT", False)
        if cache_strict_override is not None:
            cache_strict_value = cache_strict_override
        self.cache_strict = bool(cache_strict_value)

        expected_schema_value = dag_block.get("EXPECTED_SCHEMA", "any")
        if expected_schema_override is not None and str(expected_schema_override).strip():
            expected_schema_value = expected_schema_override
        self.expected_schema = resolve_expected_schema_name(str(expected_schema_value))
        self.max_nodes = int(dag_block.get("MAX_NODES", 64))
        self.max_edges = int(dag_block.get("MAX_EDGES", 256))
        self.d_node_in = int(dag_block.get("D_NODE_IN", 24))
        self.d_edge_in = int(dag_block.get("D_EDGE_IN", 8))
        self.use_time_guidance = bool(dag_block.get("USE_TIME_GUIDANCE", False))
        self.time_guidance_feature_dim = int(dag_block.get("TIME_GUIDANCE_FEATURE_DIM", _TIME_GUIDANCE_BASE_DIM))
        step_dt_raw = _to_float(dag_block.get("TIME_GUIDANCE_STEP_DT", 0.0), default=0.0)
        self.time_guidance_step_dt = (
            step_dt_raw if step_dt_raw > 0.0 else _default_time_guidance_step_dt(config)
        )
        self.time_guidance_min_steps = max(1, int(dag_block.get("TIME_GUIDANCE_MIN_STEPS", 19)))
        self.time_guidance_active_agg = str(dag_block.get("TIME_GUIDANCE_ACTIVE_AGG", "sum"))
        self.use_maneuver_tokens = bool(dag_block.get("USE_MANEUVER_TOKENS", False))
        self.maneuver_token_feature_dim = int(dag_block.get("MANEUVER_TOKEN_FEATURE_DIM", _MANEUVER_TOKEN_BASE_DIM))

        self._enabled_for_batch = self.enabled and self.source_mode == "cache" and bool(self.cache_dir)
        if self.source_mode not in {"", "cache"}:
            raise NotImplementedError(
                "Legacy additive DAG Stage B/C currently supports cache-backed DAG inputs only. "
                f"Got DAG_LATENT.SOURCE_MODE={self.source_mode!r}."
            )

        self.cache_reader = None
        if self._enabled_for_batch:
            self.cache_reader = DAGCacheReader(cache_dir=_resolve_cache_dir(self.cache_dir))

    def enabled_for_batch(self) -> bool:
        return bool(self._enabled_for_batch and self.cache_reader is not None)

    def _scenario_ids_from_batch(self, batch_list: Iterable[Dict[str, Any]]) -> List[str]:
        scenario_ids: List[str] = []
        for sample in batch_list:
            sid = sample.get("scenario_id", sample.get("metadata/scenario_id", ""))
            sid = str(sid).strip()
            if not sid:
                raise KeyError("Missing `scenario_id` while attaching DAG cache tensors.")
            scenario_ids.append(sid)
        return scenario_ids

    def build_batch_tensors(self, batch_list: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not self.enabled_for_batch():
            return {}

        assert self.cache_reader is not None
        scenario_ids = self._scenario_ids_from_batch(batch_list)
        dags: List[Dict[str, Any]] = []
        source_used: List[float] = []

        for sid in scenario_ids:
            payload = self.cache_reader.get(sid)
            if payload is None:
                if self.cache_strict:
                    raise ValueError(
                        "DAG cache strict mode enabled and cache lookup failed for "
                        f"scenario_id={sid}. cache_dir={self.cache_dir}."
                    )
                dags.append(_empty_payload(sid, self.expected_schema))
                source_used.append(0.0)
                continue

            schema_version = str(payload.get("schema_version", ""))
            if self.expected_schema != "any" and schema_version != self.expected_schema:
                if self.cache_strict:
                    raise ValueError(
                        "DAG cache schema mismatch for "
                        f"scenario_id={sid}. expected={self.expected_schema} got={schema_version}."
                    )
                dags.append(_empty_payload(sid, self.expected_schema))
                source_used.append(0.0)
                continue

            dags.append(payload)
            source_used.append(1.0)

        dag_t = tensorize_dag_batch(
            dags,
            max_nodes=self.max_nodes,
            max_edges=self.max_edges,
            d_node_in=self.d_node_in,
            d_edge_in=self.d_edge_in,
        )

        out: Dict[str, torch.Tensor] = {}
        for key, dtype in _GRAPH_TENSOR_DTYPES.items():
            out[key] = torch.as_tensor(dag_t[key], dtype=dtype)

        if self.use_time_guidance:
            time_items = [
                _build_time_guidance_one(
                    payload,
                    feature_dim=self.time_guidance_feature_dim,
                    step_dt=self.time_guidance_step_dt,
                    min_steps=self.time_guidance_min_steps,
                    active_agg=self.time_guidance_active_agg,
                )
                for payload in dags
            ]
            max_steps = max(int(item["dag_time_feat"].shape[0]) for item in time_items)
            dag_time_feat = torch.zeros(
                (len(time_items), max_steps, self.time_guidance_feature_dim),
                dtype=torch.float32,
            )
            dag_time_mask = torch.zeros((len(time_items), max_steps), dtype=torch.bool)
            for i, item in enumerate(time_items):
                steps = int(item["dag_time_feat"].shape[0])
                dag_time_feat[i, :steps] = item["dag_time_feat"]
                dag_time_mask[i, :steps] = item["dag_time_mask"]
            out["dag_time_feat"] = dag_time_feat
            out["dag_time_mask"] = dag_time_mask

        if self.use_maneuver_tokens:
            maneuver_items = [
                _build_maneuver_tokens_one(
                    payload,
                    feature_dim=self.maneuver_token_feature_dim,
                    step_dt=self.time_guidance_step_dt,
                    min_steps=self.time_guidance_min_steps,
                )
                for payload in dags
            ]
            max_tokens = max((int(item["dag_maneuver_feat"].shape[0]) for item in maneuver_items), default=0)
            dag_maneuver_feat = torch.zeros(
                (len(maneuver_items), max_tokens, self.maneuver_token_feature_dim),
                dtype=torch.float32,
            )
            dag_maneuver_mask = torch.zeros((len(maneuver_items), max_tokens), dtype=torch.bool)
            for i, item in enumerate(maneuver_items):
                num_tokens = int(item["dag_maneuver_feat"].shape[0])
                if num_tokens <= 0:
                    continue
                dag_maneuver_feat[i, :num_tokens] = item["dag_maneuver_feat"]
                dag_maneuver_mask[i, :num_tokens] = item["dag_maneuver_mask"]
            out["dag_maneuver_feat"] = dag_maneuver_feat
            out["dag_maneuver_mask"] = dag_maneuver_mask

        out["dag_source_used"] = torch.as_tensor(source_used, dtype=torch.float32)
        out["dag/cache_hit_rate"] = out["dag_source_used"].mean()
        return out
