"""Configuration dataclasses for CounterBMT v2."""

from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass
class PipelineConfig:
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    trajectory: TrajectoryModelConfig = field(default_factory=TrajectoryModelConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
