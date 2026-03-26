"""CounterBMT v2 public package surface.

Keep package import light-weight so utility submodules (for example DAG cache
schema readers used by the legacy Torch path) do not eagerly pull in optional
dependencies such as JAX.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

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


_LAZY_ATTRS = {
    "ConditioningConfig": ("counter_bmt_v2.config", "ConditioningConfig"),
    "BehaviorEmbeddingConfig": ("counter_bmt_v2.config", "BehaviorEmbeddingConfig"),
    "ConsensusConfig": ("counter_bmt_v2.config", "ConsensusConfig"),
    "NoveltyConfig": ("counter_bmt_v2.config", "NoveltyConfig"),
    "PipelineConfig": ("counter_bmt_v2.config", "PipelineConfig"),
    "RLConfig": ("counter_bmt_v2.config", "RLConfig"),
    "RLTrainConfig": ("counter_bmt_v2.config", "RLTrainConfig"),
    "RewardConfig": ("counter_bmt_v2.config", "RewardConfig"),
    "TrajectoryModelConfig": ("counter_bmt_v2.config", "TrajectoryModelConfig"),
    "VLMAlignmentConfig": ("counter_bmt_v2.config", "VLMAlignmentConfig"),
    "NNXBMTSceneSample": ("counter_bmt_v2.data", "NNXBMTSceneSample"),
    "ScenarioNetNNXLoader": ("counter_bmt_v2.data", "ScenarioNetNNXLoader"),
    "collate_nnx_scene_samples": ("counter_bmt_v2.data", "collate_nnx_scene_samples"),
    "CounterBMTPipeline": ("counter_bmt_v2.orchestration", "CounterBMTPipeline"),
    "ForwardPassEvalConfig": ("counter_bmt_v2.training", "ForwardPassEvalConfig"),
    "DAGLatentTrainConfig": ("counter_bmt_v2.training", "DAGLatentTrainConfig"),
    "SupervisedTrainConfig": ("counter_bmt_v2.training", "SupervisedTrainConfig"),
    "train_supervised": ("counter_bmt_v2.training", "train_supervised"),
    "train_supervised_dag_latent": ("counter_bmt_v2.training", "train_supervised_dag_latent"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
