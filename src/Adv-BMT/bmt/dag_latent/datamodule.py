"""Additive datamodule wrapper for DAG-latent training.

This keeps the legacy dataset logic intact but makes batching tolerant of
non-array object fields such as `original_SD`, which some local project
preprocessing paths attach to each sample.

For Stage-A training we do not need those raw python objects in the model
batch, and preserving them breaks DDP device transfer on multi-GPU hosts. The
wrapper therefore drops unsupported object-valued fields instead of trying to
forward them through Lightning.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from bmt.dataset.dataset import InfgenDataset

from .dag_cache import DAGCacheBatchBuilder


class DAGLatentInfgenDataset(InfgenDataset):
    """Legacy dataset with a safer collate for object-valued metadata fields."""

    def __init__(self, *args, dag_cache_builder: DAGCacheBatchBuilder | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dag_cache_builder = dag_cache_builder

    def collate_batch(self, batch_list: List[Dict[str, Any]]):
        if not batch_list:
            return super().collate_batch(batch_list)

        # Some local preprocessing paths attach raw python objects (for example
        # `original_SD`) that the legacy collate does not whitelist. Those
        # objects are not used by the Stage-A trainer and also break DDP's
        # recursive `.to(device)` walk, so we drop them here and let the parent
        # collate handle the regular tensor/scalar fields exactly as before.
        sanitized_batch = [dict(sample) for sample in batch_list]
        sample0 = sanitized_batch[0]
        for key, value in list(sample0.items()):
            is_supported_scalar = isinstance(value, (int, float, bool, str))
            is_supported_array = isinstance(value, np.ndarray)
            if is_supported_scalar or is_supported_array:
                continue
            for sample in sanitized_batch:
                sample.pop(key, None)

        batch = super().collate_batch(sanitized_batch)
        if self.dag_cache_builder is not None and self.dag_cache_builder.enabled_for_batch():
            batch.update(self.dag_cache_builder.build_batch_tensors(batch_list))
        return batch


class DAGLatentInfgenDataModule(pl.LightningDataModule):
    """Legacy datamodule that swaps in the safer additive dataset wrapper."""

    def __init__(
        self,
        config,
        train_batch_size,
        train_num_workers,
        train_prefetch_factor,
        val_batch_size,
        val_num_workers,
        val_prefetch_factor,
    ):
        super().__init__()
        self.config = config
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.train_prefetch_factor = train_prefetch_factor
        self.val_batch_size = val_batch_size
        self.val_num_workers = val_num_workers
        self.val_prefetch_factor = val_prefetch_factor

    def setup(self, stage: str):
        train_builder = DAGCacheBatchBuilder(self.config)
        val_builder = DAGCacheBatchBuilder(self.config)
        self.train_dataset = DAGLatentInfgenDataset(
            config=self.config,
            mode="training",
            dag_cache_builder=train_builder,
        )
        self.val_dataset = DAGLatentInfgenDataset(
            config=self.config,
            mode="test",
            dag_cache_builder=val_builder,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            pin_memory=True,
            num_workers=self.train_num_workers,
            shuffle=True,
            persistent_workers=True if self.train_num_workers > 0 else False,
            collate_fn=self.train_dataset.collate_batch,
            prefetch_factor=self.train_prefetch_factor if self.train_num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            pin_memory=True,
            num_workers=self.val_num_workers,
            shuffle=False,
            collate_fn=self.val_dataset.collate_batch,
            prefetch_factor=self.val_prefetch_factor if self.val_num_workers > 0 else None,
        )
