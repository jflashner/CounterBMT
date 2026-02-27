"""Configuration dataclasses for CounterBMT v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ConditioningConfig:
    signal_dim: int = 16


@dataclass
class TrajectoryModelConfig:
    hidden_dim: int = 64
    horizon_steps: int = 19
    dt_s: float = 0.5


@dataclass
class RewardConfig:
    w_alignment: float = 0.7
    w_safety: float = 0.2
    w_realism: float = 0.1
    # RL-only optional terms (used when manifold/consensus rewards are enabled).
    w_novelty: float = 0.0
    w_consensus: float = 0.0


@dataclass
class RLTrainConfig:
    group_size: int = 8
    rl_algo: Literal["grpo"] = "grpo"
    entropy_target: float = 1.2
    eta0: float = 0.2
    alpha0: float = 0.3
    k_eta: float = 0.1
    k_alpha: float = 0.1


@dataclass
class BehaviorEmbeddingConfig:
    mode: Literal["risk_vector", "dag_gnn", "topology_zpi", "hybrid"] = "dag_gnn"
    dim: int = 64
    use_topology_branch: bool = False


@dataclass
class NoveltyConfig:
    density: Literal["ema_gaussian", "knn"] = "ema_gaussian"
    ema_decay: float = 0.99


@dataclass
class ConsensusConfig:
    clusterer: Literal["kmeans", "hdbscan"] = "kmeans"
    k_clusters: int = 4


@dataclass
class RLConfig:
    train: RLTrainConfig = field(default_factory=RLTrainConfig)
    embedding: BehaviorEmbeddingConfig = field(default_factory=BehaviorEmbeddingConfig)
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)


@dataclass
class PipelineConfig:
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    trajectory: TrajectoryModelConfig = field(default_factory=TrajectoryModelConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rl: RLConfig = field(default_factory=RLConfig)
