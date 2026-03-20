"""Additive DAG-latent modules for legacy Adv-BMT.

This package is intentionally isolated from the released legacy model path.
Import these modules directly when you want DAG-conditioned behavior without
modifying the original `bmt.models` implementation.
"""

from .encoder import DAGLatentConfig, TorchDAGGraphEncoder

__all__ = [
    "DAGLatentConfig",
    "TorchDAGGraphEncoder",
    "MotionLMDAGLatent",
]


def __getattr__(name: str):
    if name == "MotionLMDAGLatent":
        from .model import MotionLMDAGLatent

        return MotionLMDAGLatent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
