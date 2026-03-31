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

import pathlib
from typing import Any, Dict, List

import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from bmt.dataset.dataset import InfgenDataset
from bmt.utils import REPO_ROOT
from bmt.utils import utils

from .config import get_dag_latent_block
from .dag_cache import DAGCacheBatchBuilder


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value).strip()
        if text and text.lower() not in {"none", "null"}:
            return text
    return ""


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(value)


def _normalize_scenario_id(text: str) -> str:
    raw = str(text).strip()
    if not raw:
        return raw
    stem = pathlib.Path(raw).stem
    if stem.startswith("sd_"):
        parts = stem.split("_")
        if len(parts) >= 2:
            return parts[-1]
    return stem


def _resolve_cache_payload_dir(cache_dir: str) -> pathlib.Path:
    path = pathlib.Path(str(cache_dir)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if (path / "cache").is_dir():
        path = path / "cache"
    return path


def _list_cached_scenario_ids(cache_dir: str) -> set[str]:
    payload_dir = _resolve_cache_payload_dir(cache_dir)
    if not payload_dir.is_dir():
        raise ValueError(f"DAG cache directory does not exist: {payload_dir}")
    out: set[str] = set()
    for path in payload_dir.glob("*.json"):
        sid = _normalize_scenario_id(path.stem)
        if not sid or sid == "manifest":
            continue
        out.add(sid)
    return out


class DAGLatentInfgenDataset(InfgenDataset):
    """Legacy dataset with a safer collate for object-valued metadata fields."""

    def __init__(
        self,
        *args,
        dag_cache_builder: DAGCacheBatchBuilder | None = None,
        restrict_to_cache_ids: bool = False,
        cache_dir: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dag_cache_builder = dag_cache_builder
        if bool(restrict_to_cache_ids):
            self._filter_to_cache_ids(str(cache_dir))

    def _packed_filenames(self) -> List[str]:
        base_length = int(self.real_length) if hasattr(self, "real_length") else int(self.length)
        out: List[str] = []
        for idx in range(base_length):
            seq = utils.unpack_sequence(self.strings_v, self.strings_o, idx)
            out.append(utils.sequence_to_string(seq))
        return out

    def _reset_filenames(self, filenames: List[str]) -> None:
        self.data_mapping = {k: self.data_mapping[k] for k in filenames}
        seqs = [utils.string_to_sequence(s) for s in filenames]
        self.strings_v, self.strings_o = utils.pack_sequences(seqs)
        self.length = len(filenames)
        if hasattr(self, "real_length"):
            self.real_length = len(filenames)
            if self.config.BACKWARD_PREDICTION and self.mode == "training":
                self.length = self.real_length * 2

    def _filter_to_cache_ids(self, cache_dir: str) -> None:
        cache_ids = _list_cached_scenario_ids(cache_dir)
        filenames = self._packed_filenames()
        filtered = [name for name in filenames if _normalize_scenario_id(name) in cache_ids]
        if not filtered:
            raise ValueError(
                "No dataset examples overlap the DAG cache ids. "
                f"dataset_dir={self.data_dir} cache_dir={_resolve_cache_payload_dir(cache_dir)}"
            )
        self._reset_filenames(filtered)
        print(
            f"[dag-latent] filtered {self.mode} dataset to {len(filtered)} cached scenarios "
            f"from cache_dir={_resolve_cache_payload_dir(cache_dir)}",
            flush=True,
        )

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
        keep_object_keys = {"cf/debug_meta"}
        for key, value in list(sample0.items()):
            if key in keep_object_keys:
                continue
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
        dag_block = get_dag_latent_block(self.config)
        base_cache_dir = str(dag_block.get("CACHE_DIR", ""))
        base_cache_strict = dag_block.get("CACHE_STRICT", False)
        base_expected_schema = str(dag_block.get("EXPECTED_SCHEMA", "any"))
        base_only_cache_ids = bool(dag_block.get("ONLY_CACHE_IDS", False))

        train_cache_dir = _first_non_empty(dag_block.get("TRAIN_CACHE_DIR", ""), base_cache_dir)
        val_cache_dir = _first_non_empty(dag_block.get("VAL_CACHE_DIR", ""), base_cache_dir)
        train_cache_strict = _optional_bool(dag_block.get("TRAIN_CACHE_STRICT", None))
        val_cache_strict = _optional_bool(dag_block.get("VAL_CACHE_STRICT", None))
        train_expected_schema = _first_non_empty(
            dag_block.get("TRAIN_EXPECTED_SCHEMA", ""),
            base_expected_schema,
        )
        val_expected_schema = _first_non_empty(
            dag_block.get("VAL_EXPECTED_SCHEMA", ""),
            base_expected_schema,
        )
        train_only_cache_ids = _optional_bool(dag_block.get("ONLY_CACHE_IDS_TRAIN", None))
        val_only_cache_ids = _optional_bool(dag_block.get("ONLY_CACHE_IDS_VAL", None))
        restrict_train_to_cache_ids = (
            base_only_cache_ids if train_only_cache_ids is None else train_only_cache_ids
        )
        restrict_val_to_cache_ids = (
            base_only_cache_ids if val_only_cache_ids is None else val_only_cache_ids
        )

        train_builder = DAGCacheBatchBuilder(
            self.config,
            cache_dir_override=train_cache_dir,
            cache_strict_override=train_cache_strict,
            expected_schema_override=train_expected_schema,
        )
        val_builder = DAGCacheBatchBuilder(
            self.config,
            cache_dir_override=val_cache_dir,
            cache_strict_override=val_cache_strict,
            expected_schema_override=val_expected_schema,
        )
        self.train_dataset = DAGLatentInfgenDataset(
            config=self.config,
            mode="training",
            dag_cache_builder=train_builder,
            restrict_to_cache_ids=restrict_train_to_cache_ids,
            cache_dir=train_cache_dir,
        )
        self.val_dataset = DAGLatentInfgenDataset(
            config=self.config,
            mode="test",
            dag_cache_builder=val_builder,
            restrict_to_cache_ids=restrict_val_to_cache_ids,
            cache_dir=val_cache_dir,
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
