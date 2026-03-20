from __future__ import annotations

import unittest

import numpy as np

from counter_bmt_v2.data import NNXBMTSceneSample, collate_nnx_scene_samples
from counter_bmt_v2.training.supervised import (
    SupervisedTrainConfig,
    _resolve_collate_padding_limits,
)


def _make_sample(
    *,
    scenario_id: str,
    horizon: int,
    num_agents: int,
    num_map_features: int,
    num_traffic_lights: int,
    vectors_per_map_feature: int = 4,
) -> NNXBMTSceneSample:
    t = int(horizon)
    n = int(num_agents)
    m = int(num_map_features)
    l = int(num_traffic_lights)
    v = int(vectors_per_map_feature)

    agent_pos = np.zeros((t, n, 2), dtype=np.float32)
    agent_heading = np.zeros((t, n), dtype=np.float32)
    agent_vel = np.zeros((t, n, 2), dtype=np.float32)
    agent_valid = np.ones((t, n), dtype=bool)

    for agent_idx in range(n):
        agent_pos[:, agent_idx, 0] = np.linspace(0.0, 1.0 + agent_idx, t, dtype=np.float32)
        agent_vel[:, agent_idx, 0] = 1.0 + 0.1 * agent_idx

    map_feature = np.zeros((m, v, 27), dtype=np.float32)
    map_feature_valid = np.zeros((m, v), dtype=bool)
    for map_idx in range(m):
        span = min(v, map_idx + 1)
        map_feature_valid[map_idx, :span] = True
        map_feature[map_idx, :span, 0] = float(map_idx + 1)
    map_pos = np.stack(
        [
            np.arange(m, dtype=np.float32),
            np.zeros((m,), dtype=np.float32),
            np.zeros((m,), dtype=np.float32),
        ],
        axis=-1,
    )

    tl_feat = np.zeros((t, l, 7), dtype=np.float32)
    tl_valid = np.zeros((t, l), dtype=bool)
    tl_pos = np.zeros((l, 3), dtype=np.float32)
    for light_idx in range(l):
        tl_feat[:, light_idx, :3] = np.array([float(light_idx), 0.5, 0.0], dtype=np.float32)
        tl_feat[:, light_idx, 3] = 1.0
        tl_valid[:, light_idx] = True
        tl_pos[light_idx] = np.array([float(light_idx), 0.5, 0.0], dtype=np.float32)

    return NNXBMTSceneSample(
        scenario_id=scenario_id,
        current_time_index=min(5, max(0, t - 1)),
        dt_s=0.1,
        map_center_xyz=np.zeros((3,), dtype=np.float32),
        agent_ids=np.arange(n, dtype=np.int32),
        agent_type_ids=np.ones((n,), dtype=np.int32),
        agent_shape=np.ones((n, 3), dtype=np.float32),
        agent_position_xy=agent_pos,
        agent_heading=agent_heading,
        agent_velocity_xy=agent_vel,
        agent_valid_mask=agent_valid,
        map_feature=map_feature,
        map_feature_valid_mask=map_feature_valid,
        map_position=map_pos,
        traffic_light_feature=tl_feat,
        traffic_light_valid_mask=tl_valid,
        traffic_light_position=tl_pos,
    )


class BatchLocalCollateTests(unittest.TestCase):
    def test_collate_batch_local_shapes_follow_batch_maxima(self) -> None:
        samples = [
            _make_sample(scenario_id="a", horizon=10, num_agents=3, num_map_features=5, num_traffic_lights=2),
            _make_sample(scenario_id="b", horizon=10, num_agents=7, num_map_features=9, num_traffic_lights=4),
        ]

        batch = collate_nnx_scene_samples(
            samples,
            max_time_steps=91,
            max_agents=None,
            max_map_features=None,
            max_vectors_per_map_feature=4,
            max_traffic_lights=None,
        )

        self.assertEqual(batch["agent_position_xy"].shape, (2, 91, 7, 2))
        self.assertEqual(batch["map_feature"].shape, (2, 9, 4, 27))
        self.assertEqual(batch["traffic_light_feature"].shape, (2, 91, 4, 7))
        self.assertEqual(
            batch["collate_shape"],
            {
                "batch_size": 2,
                "time_steps": 91,
                "agents": 7,
                "map_features": 9,
                "vectors_per_map_feature": 4,
                "traffic_lights": 4,
            },
        )

    def test_fixed_collate_mode_keeps_configured_ceilings(self) -> None:
        cfg = SupervisedTrainConfig(
            max_time_steps=91,
            max_agents=128,
            max_map_features=512,
            max_vectors_per_map_feature=4,
            max_traffic_lights=64,
            collate_padding_mode="fixed",
        )

        limits = _resolve_collate_padding_limits(cfg)
        self.assertEqual(limits["max_agents"], 128)
        self.assertEqual(limits["max_map_features"], 512)
        self.assertEqual(limits["max_traffic_lights"], 64)

    def test_batch_local_collate_mode_uses_local_counts_under_loader_ceilings(self) -> None:
        cfg = SupervisedTrainConfig(
            max_time_steps=91,
            max_agents=128,
            max_map_features=512,
            max_vectors_per_map_feature=128,
            max_traffic_lights=64,
            collate_padding_mode="batch_local",
        )

        limits = _resolve_collate_padding_limits(cfg)
        self.assertEqual(limits["max_time_steps"], 91)
        self.assertIsNone(limits["max_agents"])
        self.assertIsNone(limits["max_map_features"])
        self.assertEqual(limits["max_vectors_per_map_feature"], 128)
        self.assertIsNone(limits["max_traffic_lights"])


if __name__ == "__main__":
    unittest.main()
