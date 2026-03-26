"""CounterBMT v2 training entrypoints.

Import lazily so schema/cache helpers can be used without requiring the full
JAX-based training stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ForwardPassEvalConfig",
    "SupervisedTrainConfig",
    "train_supervised",
    "DAGLatentTrainConfig",
    "train_supervised_dag_latent",
]


_LAZY_ATTRS = {
    "ForwardPassEvalConfig": ("counter_bmt_v2.training.forward_metrics", "ForwardPassEvalConfig"),
    "SupervisedTrainConfig": ("counter_bmt_v2.training.supervised", "SupervisedTrainConfig"),
    "train_supervised": ("counter_bmt_v2.training.supervised", "train_supervised"),
    "DAGLatentTrainConfig": ("counter_bmt_v2.training.supervised_dag_latent", "DAGLatentTrainConfig"),
    "train_supervised_dag_latent": (
        "counter_bmt_v2.training.supervised_dag_latent",
        "train_supervised_dag_latent",
    ),
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
