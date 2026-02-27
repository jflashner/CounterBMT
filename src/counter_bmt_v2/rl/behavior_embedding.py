"""Behavior-manifold embeddings for RL rollouts.

Implements four embedding modes:
- risk_vector
- dag_gnn
- topology_zpi
- hybrid
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from counter_bmt_v2.config import BehaviorEmbeddingConfig
from counter_bmt_v2.contracts import BayesianDAG, Intervention, TrajectoryRollout
from counter_bmt_v2.rl.topology import TopologyEmbeddingRunner


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


def _stable_project(vec: np.ndarray, out_dim: int, seed: int = 11) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0 / np.sqrt(max(1, vec.size)), size=(vec.size, out_dim)).astype(np.float32)
    out = vec @ w
    n = float(np.linalg.norm(out))
    if n > 0.0:
        out = out / n
    return out.astype(np.float32)


def _text_hash_feature(text: str, n: int = 8) -> np.ndarray:
    out = np.zeros((n,), dtype=np.float32)
    b = text.encode("utf-8")
    if not b:
        return out
    for i, v in enumerate(b):
        out[i % n] += (float(v % 29) / 28.0) - 0.5
    return out


def extract_rollout_risk_features(rollout: TrajectoryRollout) -> Dict[str, float]:
    traj = np.asarray(rollout.trajectory_xy, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[0] < 2:
        return {
            "progress_delta": 0.0,
            "path_length": 0.0,
            "avg_speed": 0.0,
            "avg_acc": 0.0,
            "jerk": 0.0,
            "turn_rate": 0.0,
            "stop_ratio": 1.0,
            "collision_risk_proxy": 0.5,
            "rule_violation_proxy": 0.0,
        }

    vel = np.diff(traj, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    acc = np.diff(vel, axis=0) if vel.shape[0] > 1 else np.zeros((0, 2), dtype=np.float32)
    acc_mag = np.linalg.norm(acc, axis=1) if acc.size else np.zeros((0,), dtype=np.float32)
    jerk = np.diff(acc, axis=0) if acc.shape[0] > 1 else np.zeros((0, 2), dtype=np.float32)
    jerk_mag = np.linalg.norm(jerk, axis=1) if jerk.size else np.zeros((0,), dtype=np.float32)

    heading = np.arctan2(vel[:, 1], vel[:, 0]) if vel.size else np.zeros((0,), dtype=np.float32)
    d_heading = np.diff(heading) if heading.size > 1 else np.zeros((0,), dtype=np.float32)

    progress_delta = float(traj[-1, 0] - traj[0, 0])
    path_length = float(np.sum(speed))
    avg_speed = float(np.mean(speed)) if speed.size else 0.0
    avg_acc = float(np.mean(acc_mag)) if acc_mag.size else 0.0
    jerk_mean = float(np.mean(jerk_mag)) if jerk_mag.size else 0.0
    turn_rate = float(np.mean(np.abs(d_heading))) if d_heading.size else 0.0
    stop_ratio = float(np.mean(speed < 0.05)) if speed.size else 1.0

    # Proxy risk/violation scores: bounded [0, 1], intentionally simple.
    collision_risk_proxy = float(np.clip(_sigmoid(3.0 * avg_acc + 4.0 * jerk_mean - 1.5), 0.0, 1.0))
    rule_violation_proxy = float(np.clip(_sigmoid(2.0 * turn_rate + 1.5 * stop_ratio - 1.0), 0.0, 1.0))

    return {
        "progress_delta": progress_delta,
        "path_length": path_length,
        "avg_speed": avg_speed,
        "avg_acc": avg_acc,
        "jerk": jerk_mean,
        "turn_rate": turn_rate,
        "stop_ratio": stop_ratio,
        "collision_risk_proxy": collision_risk_proxy,
        "rule_violation_proxy": rule_violation_proxy,
    }


def _risk_vector_from_features(risk: Dict[str, float]) -> np.ndarray:
    keys = [
        "progress_delta",
        "path_length",
        "avg_speed",
        "avg_acc",
        "jerk",
        "turn_rate",
        "stop_ratio",
        "collision_risk_proxy",
        "rule_violation_proxy",
    ]
    return np.asarray([float(risk.get(k, 0.0)) for k in keys], dtype=np.float32)


def _dag_graph_embedding(
    dag: BayesianDAG,
    *,
    risk_vec: np.ndarray,
    intervention: Intervention,
    out_dim: int,
) -> np.ndarray:
    node_ids = sorted(dag.nodes.keys())
    if not node_ids:
        return _stable_project(risk_vec, out_dim=out_dim, seed=41)

    idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    # Node feature: [type one-hot(4), in/out degree(2), timestamp(1), value_hash(4)].
    x = np.zeros((n, 11), dtype=np.float32)
    type_map = {"ego_state": 0, "maneuver": 1, "decision": 2, "outcome": 3}
    in_deg = np.zeros((n,), dtype=np.float32)
    out_deg = np.zeros((n,), dtype=np.float32)
    a = np.zeros((n, n), dtype=np.float32)

    for e in dag.edges:
        if e.parent_id in idx and e.child_id in idx:
            u = idx[e.parent_id]
            v = idx[e.child_id]
            a[u, v] = 1.0
            out_deg[u] += 1.0
            in_deg[v] += 1.0

    for nid, node in dag.nodes.items():
        i = idx[nid]
        t_i = int(type_map.get(str(node.node_type), 0))
        x[i, t_i] = 1.0
        x[i, 4] = float(in_deg[i] / max(1.0, float(n)))
        x[i, 5] = float(out_deg[i] / max(1.0, float(n)))
        x[i, 6] = float(node.timestamp_s) if node.timestamp_s is not None else 0.0
        x[i, 7:] = _text_hash_feature(str(node.value), n=4)

    # Lightweight message passing.
    a = a + np.eye(n, dtype=np.float32)
    row_sum = np.sum(a, axis=1, keepdims=True) + 1e-6
    a = a / row_sum
    h = x
    h = 0.5 * h + 0.5 * (a @ h)
    h = np.tanh(0.6 * h + 0.4 * (a @ h))
    pooled = np.mean(h, axis=0)

    iv = _text_hash_feature(f"{intervention.variable}:{intervention.value}", n=8)
    fused = np.concatenate([pooled, risk_vec.astype(np.float32), iv], axis=0)
    return _stable_project(fused, out_dim=out_dim, seed=97)


@dataclass
class BehaviorManifoldEncoder:
    cfg: BehaviorEmbeddingConfig
    topology_runner: Optional[TopologyEmbeddingRunner] = None

    def __post_init__(self) -> None:
        if self.topology_runner is None:
            self.topology_runner = TopologyEmbeddingRunner(out_dim=max(8, self.cfg.dim // 2))

    @property
    def mode(self) -> Literal["risk_vector", "dag_gnn", "topology_zpi", "hybrid"]:
        return self.cfg.mode

    def encode(
        self,
        *,
        dag: BayesianDAG,
        intervention: Intervention,
        rollout: TrajectoryRollout,
        scenario_id: str,
        rollout_id: str,
    ) -> Tuple[np.ndarray, Dict[str, float], Dict[str, Any]]:
        risk = extract_rollout_risk_features(rollout)
        risk_vec = _risk_vector_from_features(risk)
        mode = str(self.cfg.mode)
        if self.cfg.use_topology_branch and mode == "dag_gnn":
            mode = "hybrid"
        meta: Dict[str, Any] = {"mode": mode}

        if mode == "risk_vector":
            emb = _stable_project(risk_vec, out_dim=int(self.cfg.dim), seed=53)
            meta["backend"] = "risk_vector"
            return emb, risk, meta

        dag_emb = _dag_graph_embedding(
            dag,
            risk_vec=risk_vec,
            intervention=intervention,
            out_dim=int(self.cfg.dim),
        )
        if mode == "dag_gnn":
            meta["backend"] = "dag_gnn"
            return dag_emb, risk, meta

        topo_emb, topo_meta = self.topology_runner.encode(
            scenario_id=scenario_id,
            rollout_id=rollout_id,
            rollout=rollout,
            use_cache=True,
        )
        topo_emb = np.asarray(topo_emb, dtype=np.float32).reshape(-1)
        meta.update({f"topology_{k}": v for k, v in topo_meta.items()})

        if mode == "topology_zpi":
            emb = _stable_project(topo_emb, out_dim=int(self.cfg.dim), seed=59)
            meta["backend"] = "topology_zpi"
            return emb, risk, meta

        # hybrid: concat graph + topology + risk and project.
        fused = np.concatenate([dag_emb.reshape(-1), topo_emb, risk_vec], axis=0)
        emb = _stable_project(fused, out_dim=int(self.cfg.dim), seed=61)
        meta["backend"] = "hybrid"
        return emb, risk, meta
