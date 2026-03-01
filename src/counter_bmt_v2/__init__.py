"""CounterBMT v2: clean, modular stack for DAG-conditioned trajectory generation."""

from .config import (
    BehaviorEmbeddingConfig,
    ConditioningConfig,
    ConsensusConfig,
    NoveltyConfig,
    PipelineConfig,
    RLConfig,
    RLTrainConfig,
    RewardConfig,
    TrajectoryModelConfig,
    VLMAlignmentConfig,
)
from .contracts import *
from .data import NNXBMTSceneSample, ScenarioNetNNXLoader, collate_nnx_scene_samples
from .orchestration import CounterBMTPipeline
from .training import (
    DAGLatentTrainConfig,
    ForwardPassEvalConfig,
    SupervisedTrainConfig,
    train_supervised,
    train_supervised_dag_latent,
)

__all__ = [
    "ConditioningConfig",
    "BehaviorEmbeddingConfig",
    "ConsensusConfig",
    "NoveltyConfig",
    "PipelineConfig",
    "RLConfig",
    "RLTrainConfig",
    "RewardConfig",
    "TrajectoryModelConfig",
    "VLMAlignmentConfig",
    "NNXBMTSceneSample",
    "ScenarioNetNNXLoader",
    "collate_nnx_scene_samples",
    "ForwardPassEvalConfig",
    "DAGLatentTrainConfig",
    "SupervisedTrainConfig",
    "train_supervised",
    "train_supervised_dag_latent",
    "CounterBMTPipeline",
]
