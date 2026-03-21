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

        # Preserve the legacy hparam payload but attach the resolved DAG block
        # so the checkpoint is self-describing.
        hparams = OmegaConf.to_container(self.config, resolve=True)
        if isinstance(hparams, dict):
            hparams["DAG_LATENT_RESOLVED"] = dag_latent_config_as_dict(self.config)
        self.save_hyperparameters(hparams)
