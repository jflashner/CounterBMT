from .behavior_embedding import BehaviorManifoldEncoder, extract_rollout_risk_features
from .consensus import ConsensusScorer
from .grpo import GRPOTrainer, compute_group_advantages as compute_advantages_from_rewards
from .loop import (
    GroupedRolloutBatch,
    collect_group_rollouts,
    compute_group_advantages,
    grpo_update,
    summarize_reward_breakdown,
)
from .novelty import EMAGaussianNovelty, KNNNovelty, build_novelty_estimator
from .reward import compose_reward
from .thermostat import EntropyThermostat
from .topology import (
    BehaviorImageBuilder,
    PHPersistenceFallbackEncoder,
    TopologyEmbeddingRunner,
    ZigzagTopologyEncoder,
)
from .vlm_alignment import AlignmentBatchResult, VLMAlignmentVerifier

__all__ = [
    "BehaviorImageBuilder",
    "BehaviorManifoldEncoder",
    "ConsensusScorer",
    "EMAGaussianNovelty",
    "EntropyThermostat",
    "GRPOTrainer",
    "compose_reward",
    "GroupedRolloutBatch",
    "KNNNovelty",
    "PHPersistenceFallbackEncoder",
    "TopologyEmbeddingRunner",
    "AlignmentBatchResult",
    "VLMAlignmentVerifier",
    "ZigzagTopologyEncoder",
    "build_novelty_estimator",
    "collect_group_rollouts",
    "compute_advantages_from_rewards",
    "compute_group_advantages",
    "extract_rollout_risk_features",
    "grpo_update",
    "summarize_reward_breakdown",
]
