"""Paper-oriented configuration presets for the NNX Adv-BMT rewrite.

These presets are explicit and versioned so experiment configs can reference a
known architecture target instead of ad-hoc command-line overrides.
"""

from __future__ import annotations

from counter_bmt_v2.trajectory_jax.nnx_bmt import BMTTokenSpaceConfig, NNXBMTConfig, NNXSceneEncoderConfig


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
        scene_encoder=NNXSceneEncoderConfig(
            map_feature_dim=27,
            traffic_light_feature_dim=7,
            max_scene_tokens=576,
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
