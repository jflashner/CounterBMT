"""Lightning wrapper for additive DAG-latent legacy training."""

from __future__ import annotations

import copy

import numpy as np
import torch
from omegaconf import OmegaConf

from bmt.models.motionlm_lightning import MotionLMLightning
from bmt.utils import lr_schedule
from bmt.utils import utils

from .config import build_dag_latent_config, dag_latent_config_as_dict
from .model import MotionLMDAGLatent


class MotionLMDAGLatentLightning(MotionLMLightning):
    """Reuse legacy MotionLMLightning and swap in the DAG-latent model."""

    _DAG_GRAPH_KEYS = (
        "dag_node_feat",
        "dag_node_mask",
        "dag_edge_src",
        "dag_edge_dst",
        "dag_edge_feat",
        "dag_edge_mask",
        "dag_global_feat",
        "dag_time_feat",
        "dag_time_mask",
        "dag_maneuver_feat",
        "dag_maneuver_mask",
        "dag_latent",
        "dag/latent",
        "dag_source_used",
    )

    def __init__(self, config):
        # Checkpoint loading merges saved hyperparameters back into the live
        # config. `DAG_LATENT_RESOLVED` is checkpoint metadata only; if left in
        # place it can preserve stale Stage-A values (for example dropout=1.0)
        # and make config equality checks fail when branching into Stage B/C.
        if "DAG_LATENT_RESOLVED" in config:
            OmegaConf.set_struct(config, False)
            config.pop("DAG_LATENT_RESOLVED", None)
            OmegaConf.set_struct(config, True)
        super().__init__(config=config)
        self.dag_latent_cfg = build_dag_latent_config(self.config)
        self.model = MotionLMDAGLatent(config=self.config, dag_config=self.dag_latent_cfg)
        self._apply_stage_parameter_policy()

        # Preserve the legacy hparam payload but attach the resolved DAG block
        # so the checkpoint is self-describing.
        hparams = OmegaConf.to_container(self.config, resolve=True)
        if isinstance(hparams, dict):
            hparams["DAG_LATENT_RESOLVED"] = dag_latent_config_as_dict(self.config)
        self.save_hyperparameters(hparams)

    def _dag_block(self):
        return self.config.get("DAG_LATENT", {})

    def _dag_stage(self) -> str:
        return str(self._dag_block().get("STAGE", "")).strip().upper()

    def _iter_dag_named_parameters(self):
        for name, param in self.model.named_parameters():
            if (
                name.startswith("dag_encoder")
                or name.startswith("dag_latent_proj")
                or name.startswith("dag_gate_proj")
                or name.startswith("dag_time_proj")
                or name.startswith("dag_time_gate_proj")
                or name.startswith("dag_maneuver_token_proj")
                or name == "null_dag_latent"
            ):
                yield name, param

    def _apply_stage_parameter_policy(self) -> None:
        stage = self._dag_stage()
        if stage != "A":
            if stage == "B" and bool(self._dag_block().get("STAGE_B_FREEZE_NON_DAG", True)):
                for param in self.model.parameters():
                    param.requires_grad_(False)
                for _, param in self._iter_dag_named_parameters():
                    param.requires_grad_(True)
            return

        for _, param in self._iter_dag_named_parameters():
            param.requires_grad_(False)

    def _clone_batch(self, obj):
        if torch.is_tensor(obj):
            return obj.clone()
        if isinstance(obj, np.ndarray):
            return obj.copy()
        if isinstance(obj, dict):
            return {k: self._clone_batch(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._clone_batch(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._clone_batch(v) for v in obj)
        return copy.deepcopy(obj)

    def _strip_dag_inputs(self, batch):
        stripped = {}
        for key, value in batch.items():
            if key in self._DAG_GRAPH_KEYS:
                continue
            if key.startswith("dag/"):
                continue
            stripped[key] = value
        return stripped

    def _shuffle_dag_inputs(self, batch):
        if "dag_node_feat" not in batch:
            return batch, False

        dag_node_feat = batch["dag_node_feat"]
        batch_size = int(dag_node_feat.shape[0]) if hasattr(dag_node_feat, "shape") else 0
        if batch_size <= 1:
            return batch, False

        perm = torch.arange(batch_size, device=dag_node_feat.device).roll(1)
        shuffled = self._clone_batch(batch)
        for key in self._DAG_GRAPH_KEYS:
            if key not in shuffled:
                continue
            value = shuffled[key]
            if torch.is_tensor(value) and value.ndim >= 1 and int(value.shape[0]) == batch_size:
                shuffled[key] = value.index_select(0, perm)
            elif isinstance(value, np.ndarray) and value.ndim >= 1 and int(value.shape[0]) == batch_size:
                shuffled[key] = value[np.asarray(perm.cpu(), dtype=np.int64)]
        return shuffled, True

    def _dag_present_rate(self, batch) -> float:
        present = batch.get("dag_source_used", batch.get("dag/source_used", None))
        if present is not None:
            if not torch.is_tensor(present):
                present = torch.as_tensor(present)
            return float(present.float().mean().item())
        if "dag_node_mask" in batch:
            mask = batch["dag_node_mask"]
            if not torch.is_tensor(mask):
                mask = torch.as_tensor(mask)
            return float(mask.bool().any(dim=1).float().mean().item())
        if "dag_latent" in batch or "dag/latent" in batch:
            latent = batch.get("dag_latent", batch.get("dag/latent"))
            if latent is None:
                return 0.0
            if not torch.is_tensor(latent):
                latent = torch.as_tensor(latent)
            return 1.0 if int(latent.shape[0]) > 0 else 0.0
        return 0.0

    def _alignment_enabled(self) -> bool:
        stage = self._dag_stage()
        if stage not in {"B", "C"}:
            return False
        return bool(self._dag_block().get("EVAL_ALIGNMENT", False))

    def _train_alignment_every_n_steps(self) -> int:
        raw = self._dag_block().get("TRAIN_ALIGNMENT_EVERY_N_STEPS", 0)
        try:
            value = int(raw)
        except Exception:
            value = 0
        return max(0, value)

    @staticmethod
    def _batch_size_from_batch(batch) -> int:
        map_feature = batch.get("encoder/map_feature")
        if torch.is_tensor(map_feature):
            return int(map_feature.shape[0])
        return 1

    def _run_forward_and_loss(self, data_dict):
        data_dict = self._prepare_validation_batch(data_dict)
        data_dict = self(data_dict)
        loss, loss_stat = self.get_loss(data_dict)
        return loss, loss_stat, data_dict

    def _log_validation_loss_stat(self, loss_stat, *, batch_size):
        self.log_dict(
            {f"val/{k}": float(v) for k, v in loss_stat.items()},
            batch_size=batch_size,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        for src_key, pbar_key in (
            ("total_loss", "val_loss"),
            ("accuracy", "val_acc"),
            ("entropy", "val_entropy"),
        ):
            if src_key in loss_stat:
                self.log(
                    pbar_key,
                    float(loss_stat[src_key]),
                    batch_size=batch_size,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

    def _log_alignment_metrics(
        self,
        batch,
        *,
        batch_size,
        prefix: str = "dag_alignment",
        on_step: bool = False,
        on_epoch: bool = True,
        force_eval: bool = False,
    ):
        if not self._alignment_enabled():
            return
        if not any(key in batch for key in self._DAG_GRAPH_KEYS):
            return

        with_dag = self._clone_batch(batch)
        without_dag = self._strip_dag_inputs(self._clone_batch(batch))
        shuffled_dag, shuffle_available = self._shuffle_dag_inputs(self._clone_batch(batch))

        was_training = bool(self.training)
        try:
            if force_eval and was_training:
                self.eval()
            with torch.no_grad():
                _, with_stat, _ = self._run_forward_and_loss(with_dag)
                _, without_stat, _ = self._run_forward_and_loss(without_dag)
                shuffled_stat = with_stat
                if shuffle_available:
                    _, shuffled_stat, _ = self._run_forward_and_loss(shuffled_dag)
        finally:
            if force_eval and was_training:
                self.train()

        loss_with = float(with_stat.get("total_loss", 0.0))
        loss_without = float(without_stat.get("total_loss", 0.0))
        loss_shuffled = float(shuffled_stat.get("total_loss", loss_with))
        acc_with = float(with_stat.get("accuracy", 0.0))
        acc_without = float(without_stat.get("accuracy", 0.0))
        acc_shuffled = float(shuffled_stat.get("accuracy", acc_with))
        denom_no = max(1e-6, abs(loss_without))
        denom_shuf = max(1e-6, abs(loss_shuffled))

        metrics = {
            f"{prefix}/present_rate": float(self._dag_present_rate(batch)),
            f"{prefix}/shuffle_available": float(1.0 if shuffle_available else 0.0),
            f"{prefix}/loss_with_dag": loss_with,
            f"{prefix}/loss_without_dag": loss_without,
            f"{prefix}/loss_gain_vs_without_dag": float(loss_without - loss_with),
            f"{prefix}/loss_gain_ratio_vs_without_dag": float((loss_without - loss_with) / denom_no),
            f"{prefix}/loss_with_shuffled_dag": loss_shuffled,
            f"{prefix}/loss_gain_vs_shuffled_dag": float(loss_shuffled - loss_with),
            f"{prefix}/loss_gain_ratio_vs_shuffled_dag": float((loss_shuffled - loss_with) / denom_shuf),
            f"{prefix}/accuracy_with_dag": acc_with,
            f"{prefix}/accuracy_without_dag": acc_without,
            f"{prefix}/accuracy_gain_vs_without_dag": float(acc_with - acc_without),
            f"{prefix}/accuracy_with_shuffled_dag": acc_shuffled,
            f"{prefix}/accuracy_gain_vs_shuffled_dag": float(acc_with - acc_shuffled),
        }
        self.log_dict(
            metrics,
            batch_size=batch_size,
            prog_bar=False,
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
        )

    def training_step(self, data_dict, batch_idx):
        train_alignment_interval = self._train_alignment_every_n_steps()
        alignment_probe_batch = None
        if (
            train_alignment_interval > 0
            and self._alignment_enabled()
            and int(self.global_step) % train_alignment_interval == 0
        ):
            alignment_probe_batch = self._clone_batch(data_dict)

        loss = super().training_step(data_dict, batch_idx)

        if alignment_probe_batch is not None:
            self._log_alignment_metrics(
                alignment_probe_batch,
                batch_size=self._batch_size_from_batch(alignment_probe_batch),
                prefix="train_dag_alignment",
                on_step=True,
                on_epoch=False,
                force_eval=True,
            )
        return loss

    def _prepare_validation_batch(self, data_dict):
        # The legacy decoder expects a randomized modeled-agent id to already be
        # present when running in evaluation mode. Legacy autoregressive eval
        # paths populate this explicitly, but our Stage-A loss-only validation
        # path calls the model forward directly.
        if (
            self.config.REMOVE_AGENT_FROM_SCENE_ENCODER
            and "decoder/randomized_modeled_agent_id" not in data_dict
        ):
            data_dict["decoder/randomized_modeled_agent_id"] = (
                self.model.motion_decoder.randomize_modeled_agent_id(
                    data_dict,
                    clip_agent_id=True,
                )
            )
        return data_dict

    def validation_step(self, data_dict, batch_idx):
        # This repository snapshot's training path does not construct the
        # scenario-level evaluator that the legacy validation_step expects.
        # For Stage-A pretraining, surface normal validation loss metrics in
        # W&B/TensorBoard instead of routing through the separate evaluator
        # stack.
        if not hasattr(self, "evaluator"):
            batch = self._clone_batch(data_dict)
            loss, loss_stat, output_batch = self._run_forward_and_loss(batch)

            batch_size = int(output_batch["encoder/map_feature"].shape[0])

            self._log_validation_loss_stat(loss_stat, batch_size=batch_size)
            self._log_alignment_metrics(data_dict, batch_size=batch_size)
            return loss
        return super().validation_step(data_dict, batch_idx)

    def on_validation_epoch_end(self):
        if not hasattr(self, "evaluator"):
            self.log("monitoring_step", float(self.global_step), sync_dist=True)
            return None
        return super().on_validation_epoch_end()

    def configure_optimizers(self):
        if self._dag_stage() != "C":
            return super().configure_optimizers()

        opt_cfg = self.config.OPTIMIZATION
        if opt_cfg.OPTIMIZER != "AdamW":
            raise ValueError(f"Unsupported optimizer for Stage C: {opt_cfg.OPTIMIZER!r}")

        decoder_scale = float(self._dag_block().get("STAGE_C_DECODER_LR_SCALE", 0.1))
        dag_scale = float(self._dag_block().get("STAGE_C_DAG_LR_SCALE", 1.0))

        decoder_params = []
        dag_params = []
        other_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("model.motion_decoder"):
                decoder_params.append(param)
            elif (
                name.startswith("model.dag_encoder")
                or name.startswith("model.dag_latent_proj")
                or name.startswith("model.dag_gate_proj")
                or name.startswith("model.dag_time_proj")
                or name.startswith("model.dag_time_gate_proj")
                or name.startswith("model.dag_maneuver_token_proj")
                or name == "model.null_dag_latent"
            ):
                dag_params.append(param)
            else:
                other_params.append(param)

        param_groups = []
        if other_params:
            param_groups.append({"params": other_params, "lr": opt_cfg.LR})
        if decoder_params:
            param_groups.append({"params": decoder_params, "lr": opt_cfg.LR * decoder_scale})
        if dag_params:
            param_groups.append({"params": dag_params, "lr": opt_cfg.LR * dag_scale})
        if not param_groups:
            raise ValueError("No trainable parameters found for Stage C optimizer.")

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=opt_cfg.LR,
            weight_decay=opt_cfg.get("WEIGHT_DECAY", 0),
            betas=(0.9, 0.95),
            eps=1e-5,
        )

        utils.rank_zero_print("=====================================")
        if self.trainer.train_dataloader is not None:
            num_steps_per_epoch = len(self.trainer.train_dataloader)
        elif self.trainer.datamodule is not None and self.trainer.datamodule.train_dataset is not None:
            utils.rank_zero_print(
                "Finding num_steps_per_epoch from datamodule...",
                len(self.trainer.datamodule.train_dataset),
                self.trainer.datamodule.train_batch_size,
                self.trainer.world_size,
            )
            num_steps_per_epoch = len(self.trainer.datamodule.train_dataset) // (
                self.trainer.datamodule.train_batch_size * self.trainer.world_size
            )
        else:
            raise ValueError("Can't find num_steps_per_epoch")

        num_epochs = self.config.epochs
        total_steps = num_steps_per_epoch * num_epochs
        utils.rank_zero_print("Configuring cosine scheduler")
        utils.rank_zero_print("Num Steps per epoch: ", num_steps_per_epoch)
        utils.rank_zero_print("Num Epochs: ", num_epochs)
        utils.rank_zero_print("Total Steps: ", total_steps)
        utils.rank_zero_print("Stage C decoder LR scale: ", decoder_scale)
        utils.rank_zero_print("Stage C DAG LR scale: ", dag_scale)
        utils.rank_zero_print("=====================================")

        scheduler = lr_schedule.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=opt_cfg.WARMUP_STEPS,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
