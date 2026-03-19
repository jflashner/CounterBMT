#!/usr/bin/env python3
"""Profile legacy Adv-BMT MidGPT training memory on a short controlled run.

This helper answers one concrete question:

    "How much GPU memory does the released legacy MidGPT stack actually use
    on real training batches?"

Why this exists:
- `nvidia-smi` alone is too coarse and misses batch-local shape context.
- The legacy trainer does not emit peak memory stats by default.
- We want a reproducible baseline before tuning the v2 parity path.

The script intentionally does not patch the legacy code. Instead it composes the
same Hydra config, builds the same Lightning module/datamodule, and attaches a
small callback that records:
- model parameter counts
- memory after the model is moved onto GPU
- memory after each profiled batch is transferred
- peak memory after backward
- peak memory at train-batch end
- batch-local tensor shapes such as padded/active agents and token steps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import numpy as np
import torch


def _ensure_legacy_import_path(legacy_root: Path) -> None:
    legacy_root = legacy_root.resolve()
    legacy_root_str = str(legacy_root)
    if legacy_root_str not in sys.path:
        sys.path.insert(0, legacy_root_str)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile legacy Adv-BMT MidGPT GPU memory on short training runs.")
    p.add_argument("--legacy-root", type=str, default="src/Adv-BMT")
    p.add_argument("--config-name", type=str, default="0202_midgpt")
    p.add_argument("--train-data-dir", type=str, required=True)
    p.add_argument("--val-data-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="outputs/legacy_midgpt_memory_profile")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--val-batch-size", type=int, default=1)
    p.add_argument("--limit-train-batches", type=int, default=5)
    p.add_argument("--limit-val-batches", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--val-num-workers", type=int, default=0)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--max-agents", type=int, default=-1)
    p.add_argument("--max-map-features", type=int, default=-1)
    p.add_argument("--max-traffic-lights", type=int, default=-1)
    p.add_argument("--profile-batches", type=int, default=5, help="number of train batches to record in detail")
    p.add_argument(
        "--trainer-precision",
        type=str,
        default="",
        help="optional Lightning precision string, e.g. bf16-mixed; leave empty to match legacy default",
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="extra Hydra override in KEY=VALUE form; may be passed multiple times",
    )
    p.add_argument(
        "--write-memory-summary",
        action="store_true",
        help="write torch.cuda.memory_summary() for the peak batch to a text file",
    )
    return p.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _safe_int_list(values: Any) -> List[int]:
    try:
        return [int(v) for v in values]
    except Exception:
        return []


def _patch_legacy_collate_for_profiling(dataset_cls: Any) -> List[Dict[str, str]]:
    """Wrap legacy collate globally so profiling is resilient to object metadata.

    The released legacy `collate_batch` asserts that every non-array value is a
    Python scalar or string. Some real ScenarioNet samples carry metadata-like
    values that violate that assumption even though the model never reads them.
    For profiling we would rather preserve training behavior on model inputs and
    sanitize only those non-model extras.
    """

    patched: List[Dict[str, str]] = []
    if getattr(dataset_cls, "_counterbmt_profile_collate_patched", False):
        return patched

    original = dataset_cls.collate_batch
    seen_keys: set[str] = set()

    def wrapped(self: Any, batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        sanitized_batch: List[Dict[str, Any]] = []
        split = str(getattr(self, "mode", "unknown"))
        for sample in batch_list:
            if not isinstance(sample, dict):
                sanitized_batch.append(sample)
                continue
            out = dict(sample)
            for key, value in list(out.items()):
                if isinstance(value, np.generic):
                    out[key] = value.item()
                    if key not in seen_keys:
                        patched.append(
                            {
                                "split": split,
                                "key": str(key),
                                "action": "numpy_scalar_to_python",
                                "type": type(value).__name__,
                            }
                        )
                        seen_keys.add(str(key))
                elif isinstance(value, (int, float, bool, str, np.ndarray)):
                    continue
                else:
                    out[key] = repr(value)
                    if key not in seen_keys:
                        patched.append(
                            {
                                "split": split,
                                "key": str(key),
                                "action": "object_repr_fallback",
                                "type": type(value).__name__,
                            }
                        )
                        seen_keys.add(str(key))
            sanitized_batch.append(out)
        return original(self, sanitized_batch)

    dataset_cls.collate_batch = wrapped
    dataset_cls._counterbmt_profile_collate_patched = True
    return patched


class LegacyMemoryTraceCallback(pl.Callback):
    """Capture per-batch CUDA peaks together with batch-local shape metadata."""

    def __init__(self, *, output_dir: Path, profile_batches: int, write_memory_summary: bool) -> None:
        self.output_dir = output_dir
        self.profile_batches = max(1, int(profile_batches))
        self.write_memory_summary = bool(write_memory_summary)
        self.device_index: Optional[int] = None
        self.batch_records: List[Dict[str, Any]] = []
        self._active_record: Optional[Dict[str, Any]] = None
        self._peak_batch_end_bytes: int = -1
        self._peak_memory_summary: Optional[str] = None

    def _cuda_stats(self) -> Dict[str, int]:
        if self.device_index is None:
            return {}
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(self.device_index)),
            "reserved_bytes": int(torch.cuda.memory_reserved(self.device_index)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device_index)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device_index)),
        }

    def _describe_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        agent_feat = batch.get("encoder/agent_feature")
        agent_valid = batch.get("encoder/agent_valid_mask")
        map_valid = batch.get("encoder/map_valid_mask")
        tl_valid = batch.get("encoder/traffic_light_valid_mask")
        input_action = batch.get("decoder/input_action")
        input_action_valid = batch.get("decoder/input_action_valid_mask")
        modeled_pos = batch.get("decoder/modeled_agent_position")

        summary: Dict[str, Any] = {}
        if agent_feat is not None:
            summary["encoder_agent_feature_shape"] = list(agent_feat.shape)
            summary["batch_size"] = int(agent_feat.shape[0])
            summary["encoder_time_steps"] = int(agent_feat.shape[1])
            summary["padded_agents"] = int(agent_feat.shape[2])
        if agent_valid is not None:
            active_agents = agent_valid.any(dim=1).sum(dim=1).detach().cpu().tolist()
            valid_tokens = agent_valid.sum(dim=(1, 2)).detach().cpu().tolist()
            summary["active_agents_per_sample"] = _safe_int_list(active_agents)
            summary["agent_valid_cells_per_sample"] = _safe_int_list(valid_tokens)
        if modeled_pos is not None:
            summary["decoder_modeled_agent_position_shape"] = list(modeled_pos.shape)
            summary["modeled_agents"] = int(modeled_pos.shape[2])
        if input_action is not None:
            summary["decoder_input_action_shape"] = list(input_action.shape)
            summary["decoder_token_steps"] = int(input_action.shape[1])
        if input_action_valid is not None:
            active_modeled = input_action_valid.any(dim=1).sum(dim=1).detach().cpu().tolist()
            valid_decoder_cells = input_action_valid.sum(dim=(1, 2)).detach().cpu().tolist()
            summary["active_modeled_agents_per_sample"] = _safe_int_list(active_modeled)
            summary["decoder_valid_cells_per_sample"] = _safe_int_list(valid_decoder_cells)
        if map_valid is not None:
            summary["encoder_map_valid_mask_shape"] = list(map_valid.shape)
            summary["valid_map_tokens_per_sample"] = _safe_int_list(map_valid.sum(dim=1).detach().cpu().tolist())
        if tl_valid is not None:
            summary["encoder_traffic_light_valid_mask_shape"] = list(tl_valid.shape)
            tl_counts = tl_valid.any(dim=1).sum(dim=1).detach().cpu().tolist()
            summary["active_traffic_lights_per_sample"] = _safe_int_list(tl_counts)
        return summary

    def on_fit_start(self, trainer, pl_module) -> None:  # pragma: no cover - runtime hook
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for legacy MidGPT memory profiling.")
        device = trainer.strategy.root_device
        if device.type != "cuda":
            raise RuntimeError(f"Expected CUDA root device, got: {device}")
        self.device_index = int(torch.cuda.current_device() if device.index is None else device.index)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device_index)

        total_params = int(sum(p.numel() for p in pl_module.parameters()))
        trainable_params = int(sum(p.numel() for p in pl_module.parameters() if p.requires_grad))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fit_start = {
            "device": str(device),
            "device_index": int(self.device_index),
            "model_after_move": self._cuda_stats(),
            "parameter_count": {
                "total": total_params,
                "trainable": trainable_params,
            },
        }
        (self.output_dir / "fit_start.json").write_text(json.dumps(_jsonable(fit_start), indent=2), encoding="utf-8")
        print(
            "[legacy-memory] fit_start "
            f"device={fit_start['device']} total_params={total_params} "
            f"alloc={fit_start['model_after_move'].get('allocated_bytes', 0)}"
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> None:  # pragma: no cover - runtime hook
        if self.device_index is None or batch_idx >= self.profile_batches:
            self._active_record = None
            return
        record = {
            "batch_idx": int(batch_idx),
            "batch_summary": self._describe_batch(batch),
            "after_transfer": self._cuda_stats(),
        }
        # Reset here so the subsequent peak reflects forward+backward+step for
        # this concrete batch rather than earlier warmup allocation history.
        torch.cuda.reset_peak_memory_stats(self.device_index)
        self._active_record = record
        print(
            "[legacy-memory] batch_start "
            f"idx={batch_idx} "
            f"padded_agents={record['batch_summary'].get('padded_agents')} "
            f"token_steps={record['batch_summary'].get('decoder_token_steps')} "
            f"alloc={record['after_transfer'].get('allocated_bytes', 0)}"
        )

    def on_after_backward(self, trainer, pl_module) -> None:  # pragma: no cover - runtime hook
        if self._active_record is None:
            return
        self._active_record["after_backward_peak"] = self._cuda_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:  # pragma: no cover
        if self._active_record is None:
            return
        end_stats = self._cuda_stats()
        self._active_record["batch_end_peak"] = end_stats
        self.batch_records.append(self._active_record)
        if end_stats.get("max_allocated_bytes", -1) > self._peak_batch_end_bytes:
            self._peak_batch_end_bytes = int(end_stats["max_allocated_bytes"])
            if self.write_memory_summary and self.device_index is not None:
                self._peak_memory_summary = torch.cuda.memory_summary(device=self.device_index, abbreviated=False)
        print(
            "[legacy-memory] batch_end "
            f"idx={batch_idx} peak_alloc={end_stats.get('max_allocated_bytes', 0)} "
            f"peak_reserved={end_stats.get('max_reserved_bytes', 0)}"
        )
        self._active_record = None

    def on_fit_end(self, trainer, pl_module) -> None:  # pragma: no cover - runtime hook
        summary = {
            "profiled_batches": self.batch_records,
            "peak_batch_max_allocated_bytes": max(
                (int(r.get("batch_end_peak", {}).get("max_allocated_bytes", 0)) for r in self.batch_records),
                default=0,
            ),
            "peak_batch_max_reserved_bytes": max(
                (int(r.get("batch_end_peak", {}).get("max_reserved_bytes", 0)) for r in self.batch_records),
                default=0,
            ),
        }
        (self.output_dir / "memory_profile.json").write_text(
            json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
        )
        if self.write_memory_summary and self._peak_memory_summary:
            (self.output_dir / "peak_memory_summary.txt").write_text(self._peak_memory_summary, encoding="utf-8")
        print(
            "[legacy-memory] fit_end "
            f"peak_alloc={summary['peak_batch_max_allocated_bytes']} "
            f"peak_reserved={summary['peak_batch_max_reserved_bytes']}"
        )


def _build_overrides(args: argparse.Namespace) -> List[str]:
    overrides = [
        f"DATA.TRAINING_DATA_DIR={Path(args.train_data_dir).resolve()}",
        f"DATA.TEST_DATA_DIR={Path(args.val_data_dir).resolve()}",
        f"batch_size={int(args.batch_size)}",
        f"val_batch_size={int(args.val_batch_size)}",
        f"epochs={int(args.epochs)}",
        f"seed={int(args.seed)}",
        f"wandb=False",
        f"log_dir={Path(args.output_dir).resolve()}",
        f"num_workers={int(args.num_workers)}",
        f"val_num_workers={int(args.val_num_workers)}",
        f"prefetch_factor={int(args.prefetch_factor)}",
        f"num_sanity_val_steps=0",
        f"limit_train_batches={int(args.limit_train_batches)}",
        f"limit_val_batches={int(args.limit_val_batches)}",
    ]
    if int(args.max_agents) > 0:
        overrides.append(f"PREPROCESSING.MAX_AGENTS={int(args.max_agents)}")
    if int(args.max_map_features) > 0:
        overrides.append(f"PREPROCESSING.MAX_MAP_FEATURES={int(args.max_map_features)}")
    if int(args.max_traffic_lights) > 0:
        overrides.append(f"PREPROCESSING.MAX_TRAFFIC_LIGHTS={int(args.max_traffic_lights)}")
    overrides.extend(str(v) for v in args.override)
    return overrides


def main() -> int:
    args = _parse_args()
    legacy_root = Path(args.legacy_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_legacy_import_path(legacy_root)

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from bmt.dataset import dataset as legacy_dataset_module
    from bmt.dataset.datamodule import InfgenDataModule
    from bmt.models.motionlm_lightning import MotionLMLightning

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(int(args.seed))

    config_dir = legacy_root / "cfgs"
    overrides = _build_overrides(args)
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name=str(args.config_name), overrides=overrides)

    OmegaConf.set_struct(config, False)
    config.ROOT_DIR = legacy_root
    OmegaConf.set_struct(config, True)

    collate_patches = _patch_legacy_collate_for_profiling(legacy_dataset_module.InfgenDataset)

    datamodule = InfgenDataModule(
        config,
        train_batch_size=int(args.batch_size),
        train_num_workers=int(args.num_workers),
        train_prefetch_factor=int(args.prefetch_factor),
        val_batch_size=int(args.val_batch_size),
        val_num_workers=int(args.val_num_workers),
        val_prefetch_factor=int(args.prefetch_factor),
    )
    model = MotionLMLightning(config=config)
    callback = LegacyMemoryTraceCallback(
        output_dir=output_dir,
        profile_batches=int(args.profile_batches),
        write_memory_summary=bool(args.write_memory_summary),
    )

    trainer_kwargs: Dict[str, Any] = {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": int(args.devices) if torch.cuda.is_available() else 1,
        "max_epochs": int(args.epochs),
        "num_sanity_val_steps": 0,
        "limit_train_batches": int(args.limit_train_batches),
        "limit_val_batches": int(args.limit_val_batches),
        "logger": False,
        "enable_checkpointing": False,
        "enable_model_summary": False,
        "callbacks": [callback],
        "gradient_clip_val": float(config.OPTIMIZATION.GRAD_NORM_CLIP),
        "log_every_n_steps": 1,
        "deterministic": bool(getattr(config, "deterministic", False)),
    }
    if args.trainer_precision:
        trainer_kwargs["precision"] = str(args.trainer_precision)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and int(args.devices) > 1:
        trainer_kwargs["strategy"] = "ddp"

    trainer_meta = dict(trainer_kwargs)
    trainer_meta["callbacks"] = ["LegacyMemoryTraceCallback"]

    run_meta = {
        "legacy_root": str(legacy_root),
        "config_name": str(args.config_name),
        "overrides": overrides,
        "trainer_kwargs": _jsonable(trainer_meta),
        "output_dir": str(output_dir),
        "collate_sanitization": collate_patches,
    }
    (output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model=model, datamodule=datamodule)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
