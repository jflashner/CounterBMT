from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from counter_bmt_v2.trajectory_jax import AdvBMTParityTokenizer, ParityTokenizerConfig
from scripts.parity.compare_tokenization import (
    LegacyTokenizerRunner,
    _map_legacy_input_to_model_ids,
    _map_legacy_targets,
)


def _synthetic_batch() -> dict[str, np.ndarray]:
    batch_size = 1
    time_steps = 11
    n_agents = 2

    pos = np.zeros((batch_size, time_steps, n_agents, 2), dtype=np.float32)
    vel = np.zeros((batch_size, time_steps, n_agents, 2), dtype=np.float32)
    heading = np.zeros((batch_size, time_steps, n_agents), dtype=np.float32)
    valid = np.zeros((batch_size, time_steps, n_agents), dtype=bool)

    # Agent 0: always valid, straight motion along +x.
    valid[0, :, 0] = True
    pos[0, :, 0, 0] = np.arange(time_steps, dtype=np.float32) * 0.5
    vel[0, :, 0, 0] = 1.0
    heading[0, :, 0] = 0.0

    # Agent 1: appears at t=5 and moves along +y, exercising GPT-style add-agent semantics.
    valid[0, 5:, 1] = True
    pos[0, 5:, 1, 1] = np.arange(time_steps - 5, dtype=np.float32) * 0.5
    vel[0, 5:, 1, 1] = 1.0
    heading[0, 5:, 1] = np.pi / 2.0

    return {
        "agent_position_xy": pos,
        "agent_velocity_xy": vel,
        "agent_heading": heading,
        "agent_valid_mask": valid,
        "agent_shape": np.asarray([[[4.5, 2.0, 1.5], [0.8, 0.6, 1.7]]], dtype=np.float32),
        "agent_type_ids": np.asarray([[1, 2]], dtype=np.int32),
    }


class TokenizerParityExactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v2 = AdvBMTParityTokenizer(ParityTokenizerConfig(num_skipped_steps=5))
        cls.legacy = LegacyTokenizerRunner(legacy_root=Path("src/Adv-BMT"), skip_steps=5)

    def _assert_mode(self, mode: str) -> None:
        batch = _synthetic_batch()
        backward_prediction = mode == "backward"

        v2 = self.v2.tokenize_batch(batch, backward_prediction=backward_prediction)
        legacy = self.legacy.tokenize_batch(batch, backward_prediction=backward_prediction)
        legacy_prev = _map_legacy_input_to_model_ids(legacy["input_action"], self.v2)
        legacy_tgt = _map_legacy_targets(legacy["target_action"], legacy["target_mask"], self.v2)

        np.testing.assert_array_equal(v2.prev_token_ids, legacy_prev)
        np.testing.assert_array_equal(v2.targets, legacy_tgt["targets"])
        np.testing.assert_array_equal(v2.target_mask > 0.5, legacy_tgt["target_mask"] > 0.5)
        np.testing.assert_array_equal(v2.input_mask, legacy["input_mask"])
        np.testing.assert_allclose(v2.modeled_agent_delta, legacy["modeled_agent_delta"], atol=1e-6)

    def test_forward_matches_legacy_exactly(self) -> None:
        self._assert_mode("forward")

    def test_backward_matches_legacy_exactly(self) -> None:
        self._assert_mode("backward")


if __name__ == "__main__":
    unittest.main()
