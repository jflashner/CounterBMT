"""Paper-oriented configuration presets for the NNX Adv-BMT rewrite.

These presets are explicit and versioned so experiment configs can reference a
known architecture target instead of ad-hoc command-line overrides.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

from counter_bmt_v2.trajectory_jax.nnx_bmt import (
    BMTTokenSpaceConfig,
    NNXBMTConfig,
    NNXDAGConditioningConfig,
    NNXDecoderParityConfig,
    NNXDAGEncoderConfig,
    NNXRelationParityConfig,
    NNXSceneEncoderConfig,
)


@dataclass(frozen=True)
class RuntimeTrainPreset:
    """Training/runtime knobs paired with a model preset."""

    model_preset: str
    tokenizer_mode: str
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    grad_clip_norm: float
    skip_steps: int
    lr_schedule_mode: str
    num_epochs: int = 3
    mode: str = "mixed"
    reverse_probability: float = 0.5
    collate_padding_mode: str = "fixed"


def runtime_preset_none() -> RuntimeTrainPreset:
    """No-op runtime preset; keeps CLI defaults."""
    return RuntimeTrainPreset(
        model_preset="paper_like_small",
        tokenizer_mode="paper_simple",
        learning_rate=3e-4,
        warmup_steps=200,
        weight_decay=0.0,
        grad_clip_norm=1.0,
        skip_steps=5,
        lr_schedule_mode="v2_cosine_minlr",
        num_epochs=3,
        mode="mixed",
        reverse_probability=0.5,
    )


def adv_bmt_runtime_parity_preset() -> RuntimeTrainPreset:
    """Runtime preset aligned to supervised Adv-BMT parity defaults.

    This preserves the historical v2 behavior of the preset. Use
    `legacy_midgpt_recipe` when you want the closest match to the released
    `cfgs/0202_midgpt.yaml` training recipe, including forward-only training.
    """
    return RuntimeTrainPreset(
        model_preset="midgpt_parity",
        tokenizer_mode="adv_bmt_parity",
        learning_rate=3e-4,
        warmup_steps=2000,
        weight_decay=0.0,
        grad_clip_norm=1.0,
        skip_steps=5,
        lr_schedule_mode="legacy_cosine_zero",
        num_epochs=3,
        mode="mixed",
        reverse_probability=0.5,
    )


def legacy_midgpt_recipe_preset() -> RuntimeTrainPreset:
    """Closest v2 runtime match to legacy `cfgs/0202_midgpt.yaml`.

    Important note:
    - Legacy Lightning DDP treats `batch_size` as a per-process batch size.
    - v2 `pmap` treats `batch_size` as a global batch size.
    - This preset intentionally does not set `batch_size`; choose it explicitly
      for the device count you are using.
    """
    return RuntimeTrainPreset(
        model_preset="midgpt_parity",
        tokenizer_mode="adv_bmt_parity",
        learning_rate=3e-4,
        warmup_steps=2000,
        weight_decay=0.0,
        grad_clip_norm=1.0,
        skip_steps=5,
        lr_schedule_mode="legacy_cosine_zero",
        num_epochs=30,
        mode="forward",
        reverse_probability=0.0,
        collate_padding_mode="batch_local",
    )


def legacy_midgpt_speed_recipe_preset() -> RuntimeTrainPreset:
    """Speed-oriented MidGPT runtime for JAX/XLA training.

    This keeps the same optimization recipe as the parity preset, but swaps the
    collate policy to `bucketed` so batches reuse a small set of compiled
    shapes. It is intentionally a training-throughput preset, not the most
    literal legacy padding match.
    """
    return RuntimeTrainPreset(
        model_preset="midgpt_parity",
        tokenizer_mode="adv_bmt_parity",
        learning_rate=3e-4,
        warmup_steps=2000,
        weight_decay=0.0,
        grad_clip_norm=1.0,
        skip_steps=5,
        lr_schedule_mode="legacy_cosine_zero",
        num_epochs=30,
        mode="forward",
        reverse_probability=0.0,
        collate_padding_mode="bucketed",
    )


def get_runtime_preset(name: str) -> Dict[str, object]:
    if name == "adv_bmt_runtime_parity":
        return asdict(adv_bmt_runtime_parity_preset())
    if name == "legacy_midgpt_recipe":
        return asdict(legacy_midgpt_recipe_preset())
    if name == "legacy_midgpt_speed_recipe":
        return asdict(legacy_midgpt_speed_recipe_preset())
    return asdict(runtime_preset_none())


def paper_like_small_config() -> NNXBMTConfig:
    """Fast-turnaround preset for local development.

    Paper reference:
    - Keeps Adv-BMT token-space assumptions intact (33x33 action bins, dt=0.5s)
      while using a smaller hidden width/layer stack for faster iteration.
    """

    return NNXBMTConfig(
        d_model=128,
        n_layers=4,
        n_heads=8,
        ff_mult=4,
        scene_encoder=NNXSceneEncoderConfig(
            map_feature_dim=27,
            traffic_light_feature_dim=7,
            max_scene_tokens=384,
        ),
        token_space=BMTTokenSpaceConfig(
            n_acc_bins=33,
            n_yaw_bins=33,
            acc_min=-10.0,
            acc_max=10.0,
            yaw_min=-1.57079632679,
            yaw_max=1.57079632679,
            dt_s=0.5,
        ),
    )


def paper_like_full_config() -> NNXBMTConfig:
    """Closer-to-paper training preset for full runs.

    Paper reference:
    - Uses Adv-BMT style model width/depth intent from released configs
      (`motion_default.yaml` / `0202_midgpt.yaml`) with relation-aware decoding.
    """

    return NNXBMTConfig(
        d_model=256,
        n_layers=6,
        n_heads=8,
        ff_mult=4,
        n_agent_types=5,
        max_agent_id=128,
        scene_encoder=NNXSceneEncoderConfig(
            map_feature_dim=27,
            traffic_light_feature_dim=7,
            max_scene_tokens=576,
            map_encoder_style="legacy_pointnet",
            legacy_polyline_hidden_dim=64,
            legacy_polyline_num_layers=2,
            legacy_polyline_num_pre_layers=1,
            norm_style="layernorm",
            use_post_proj_head=True,
        ),
        token_space=BMTTokenSpaceConfig(
            n_acc_bins=33,
            n_yaw_bins=33,
            acc_min=-10.0,
            acc_max=10.0,
            yaw_min=-1.57079632679,
            yaw_max=1.57079632679,
            dt_s=0.5,
        ),
    )


def midgpt_parity_config() -> NNXBMTConfig:
    """Parity-target preset aligned to `cfgs/0202_midgpt.yaml`.

    This preset is opt-in and enables scene relation parity toggles used in the
    MidGPT training recipe.
    """

    return NNXBMTConfig(
        d_model=128,
        n_layers=6,
        n_heads=8,
        ff_mult=4,
        n_agent_types=5,
        max_agent_id=128,
        a2a_rel_dim=12,
        a2t_rel_dim=12,
        a2s_rel_dim=3,
        scene_encoder=NNXSceneEncoderConfig(
            map_feature_dim=27,
            traffic_light_feature_dim=7,
            max_scene_tokens=576,
            map_encoder_style="legacy_pointnet",
            legacy_polyline_hidden_dim=64,
            legacy_polyline_num_layers=2,
            legacy_polyline_num_pre_layers=1,
            norm_style="layernorm",
            use_post_proj_head=True,
        ),
        relation=NNXRelationParityConfig(
            enabled=True,
            simple_relation=True,
            simple_relation_factor=1,
            remove_traffic_light_state=True,
            per_contour_point_relation=False,
            add_relation_to_v=False,
            remove_rel_norm=False,
            update_relation=False,
            s2s_knn=128,
            s2s_distance=None,
            a2s_knn=128,
            a2s_distance=None,
            a2a_knn=64,
            a2a_distance=50.0,
            heading_placeholder=-100.0,
            scene_num_layers=3,
        ),
        decoder=NNXDecoderParityConfig(
            enabled=True,
            use_legacy_motion_embed=True,
            add_pe_for_token=True,
            randomize_agent_id=True,
            use_backward_indicator_embed=False,
            dense_masked_relation_attn=False,
        ),
        token_space=BMTTokenSpaceConfig(
            n_acc_bins=33,
            n_yaw_bins=33,
            acc_min=-10.0,
            acc_max=10.0,
            yaw_min=-1.57079632679,
            yaw_max=1.57079632679,
            dt_s=0.5,
        ),
    )


def midgpt_dag_latent_config() -> NNXBMTConfig:
    """MidGPT parity + opt-in DAG latent conditioning."""
    cfg = midgpt_parity_config()
    cfg.dag_encoder = NNXDAGEncoderConfig(
        enabled=True,
        d_node_in=24,
        d_edge_in=8,
        d_hidden=128,
        n_layers=3,
        dropout=0.0,
        max_nodes=64,
        max_edges=256,
    )
    cfg.dag_conditioning = NNXDAGConditioningConfig(
        enabled=True,
        injection_mode="global_gated_residual",
        dag_dropout_prob=0.0,
        use_null_latent=True,
        null_latent_init_std=0.02,
    )
    return cfg
