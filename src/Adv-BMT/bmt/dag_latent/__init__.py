"""Additive DAG-latent modules for legacy Adv-BMT.

This package is intentionally isolated from the released legacy model path.
Import these modules directly when you want DAG-conditioned behavior without
modifying the original `bmt.models` implementation.
"""

from .config import build_dag_latent_config, dag_latent_config_as_dict, get_dag_latent_block
from .encoder import DAGLatentConfig, TorchDAGGraphEncoder

__all__ = [
    "DAGLatentConfig",
    "TorchDAGGraphEncoder",
    "build_dag_latent_config",
    "dag_latent_config_as_dict",
    "get_dag_latent_block",
    "MotionLMDAGLatentLightning",
    "MotionLMDAGLatent",
]


def __getattr__(name: str):
    if name == "MotionLMDAGLatent":
        from .model import MotionLMDAGLatent

        return MotionLMDAGLatent
    if name == "MotionLMDAGLatentLightning":
        from .lightning import MotionLMDAGLatentLightning

        return MotionLMDAGLatentLightning
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
