"""Lightning wrapper for additive DAG-latent legacy training."""

from __future__ import annotations

from omegaconf import OmegaConf

from bmt.models.motionlm_lightning import MotionLMLightning

from .config import build_dag_latent_config, dag_latent_config_as_dict
from .model import MotionLMDAGLatent


class MotionLMDAGLatentLightning(MotionLMLightning):
    """Reuse legacy MotionLMLightning and swap in the DAG-latent model."""

    def __init__(self, config):
        super().__init__(config=config)
        self.dag_latent_cfg = build_dag_latent_config(self.config)
        self.model = MotionLMDAGLatent(config=self.config, dag_config=self.dag_latent_cfg)
        self._freeze_stage_a_dag_parameters()

        # Preserve the legacy hparam payload but attach the resolved DAG block
        # so the checkpoint is self-describing.
        hparams = OmegaConf.to_container(self.config, resolve=True)
        if isinstance(hparams, dict):
            hparams["DAG_LATENT_RESOLVED"] = dag_latent_config_as_dict(self.config)
        self.save_hyperparameters(hparams)

    def _freeze_stage_a_dag_parameters(self) -> None:
        dag_block = self.config.get("DAG_LATENT", {})
        stage = str(dag_block.get("STAGE", "")).strip().upper()
        if stage != "A":
            return

        for module_name in ("dag_encoder", "dag_latent_proj", "dag_gate_proj"):
            module = getattr(self.model, module_name, None)
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad_(False)

        if getattr(self.model, "null_dag_latent", None) is not None:
            self.model.null_dag_latent.requires_grad_(False)

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
            data_dict = self._prepare_validation_batch(data_dict)
            data_dict = self(data_dict)
            loss, loss_stat = self.get_loss(data_dict)

            batch_size = int(data_dict["encoder/map_feature"].shape[0])

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
            return loss
        return super().validation_step(data_dict, batch_idx)

    def on_validation_epoch_end(self):
        if not hasattr(self, "evaluator"):
            self.log("monitoring_step", float(self.global_step))
            return None
        return super().on_validation_epoch_end()
