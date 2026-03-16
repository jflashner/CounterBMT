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
class RLPolicyConfig:
    backend: Literal["nnx_checkpoint", "scaffold"] = "nnx_checkpoint"
    checkpoint: str = ""
    model_preset: str = ""
    tokenizer_mode: str = "adv_bmt_parity"
    skip_steps: int = 5
    dag_source_mode: Literal["dual", "cache", "scene_derived"] = "dual"
    dag_cache_dir: str = ""
    dag_cache_strict: bool = False
    dag_expected_schema: str = "any"
    clip_eps: float = 0.2
    kl_beta: float = 0.02
    policy_lr: float = 1e-5
    trainable_scope: Literal["decoder_dag", "all"] = "decoder_dag"
    ppo_epochs: int = 1
    candidate_multiplier: int = 2
    feasible_max_speed_mps: float = 40.0
    feasible_max_accel_delta: float = 4.0
    feasible_max_yaw_delta: float = 0.75
    enable_feasibility_mask: bool = True
    sampling_method: Literal["topp", "topk", "softmax", "argmax"] = "topp"
    sampling_temperature: float = 1.0
    sampling_topp: float = 0.95
    sampling_topk: int = 5


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
class VLMAlignmentConfig:
    enabled: bool = False
    source_mode: Literal["judge", "vlm_replace"] = "judge"
    backend: Literal["gpt4o", "mock"] = "gpt4o"
    model: str = "gpt-4o"
    api_key: str | None = None
    sample_rate: float = 0.15
    every_n_steps: int = 5
    max_calls_per_step: int = 2
    max_concurrency: int = 2
    per_call_timeout_s: float = 8.0
    step_wait_budget_s: float = 6.0
    neutral_score: float = 0.0
    match_threshold: float = 0.6
    unscored_policy: Literal["step_mean_fill"] = "step_mean_fill"
    cache_dir: str = "outputs/rl_vlm_alignment_cache"
    save_evidence_artifacts: bool = True
    evidence_subdir: str = "vlm_alignment_evidence"
    num_frames: int = 6
    max_agents_render: int = 48
    prompt_version: str = "vlm_alignment_topdown_dag_v1"


@dataclass
class RLConfig:
    train: RLTrainConfig = field(default_factory=RLTrainConfig)
    policy: RLPolicyConfig = field(default_factory=RLPolicyConfig)
    embedding: BehaviorEmbeddingConfig = field(default_factory=BehaviorEmbeddingConfig)
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    vlm_alignment: VLMAlignmentConfig = field(default_factory=VLMAlignmentConfig)


@dataclass
class PipelineConfig:
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    trajectory: TrajectoryModelConfig = field(default_factory=TrajectoryModelConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rl: RLConfig = field(default_factory=RLConfig)
