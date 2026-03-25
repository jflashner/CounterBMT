"""Cache-backed DAG tensor attachment for legacy DAG-latent training."""

from __future__ import annotations

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


class DAGCacheBatchBuilder:
    """Resolve DAG cache payloads by scenario id and tensorize them for legacy batches."""

    def __init__(self, config: Any):
        dag_block = get_dag_latent_block(config)
        self.enabled = bool(dag_block.get("ENABLED", False))
        self.source_mode = str(dag_block.get("SOURCE_MODE", "")).strip().lower()
        self.cache_dir = str(dag_block.get("CACHE_DIR", "")).strip()
        self.cache_strict = bool(dag_block.get("CACHE_STRICT", False))
        self.expected_schema = resolve_expected_schema_name(str(dag_block.get("EXPECTED_SCHEMA", "any")))
        self.max_nodes = int(dag_block.get("MAX_NODES", 64))
        self.max_edges = int(dag_block.get("MAX_EDGES", 256))
        self.d_node_in = int(dag_block.get("D_NODE_IN", 24))
        self.d_edge_in = int(dag_block.get("D_EDGE_IN", 8))

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
        out["dag_source_used"] = torch.as_tensor(source_used, dtype=torch.float32)
        out["dag/cache_hit_rate"] = out["dag_source_used"].mean()
        return out
