from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

import counter_bmt_v2.runtime_guards as runtime_guards
from counter_bmt_v2.causal.promptbn import PromptBNDAGBuilder
from counter_bmt_v2.cli.train_rl_topo_mcpo import _build_parser as build_rl_parser
from counter_bmt_v2.cli.train_rl_topo_mcpo import _validate_runtime_args
from counter_bmt_v2.config import RewardConfig, VLMAlignmentConfig
from counter_bmt_v2.contracts import ConditioningSignal, JudgeResult, TrajectoryRollout
from counter_bmt_v2.perception import OpenAIPerceptionModel
from counter_bmt_v2.rl.reward import compose_reward
from counter_bmt_v2.rl.vlm_alignment import VLMAlignmentVerifier
from counter_bmt_v2.runtime_guards import normalize_openai_backend


def _rollout_with_risk(*, collision: float, violation: float) -> TrajectoryRollout:
    traj = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )
    return TrajectoryRollout(
        trajectory_xy=traj,
        conditioning=ConditioningSignal(vector=np.zeros((1,), dtype=np.float32), metadata={}),
        sample_index=0,
        metadata={
            "risk_features": {
                "collision_risk_proxy": float(collision),
                "rule_violation_proxy": float(violation),
            }
        },
    )


class RealRunHardeningTests(unittest.TestCase):
    def test_debug_settings_are_rejected_without_allow_flag(self) -> None:
        args = build_rl_parser().parse_args(
            [
                "--policy-backend",
                "scaffold",
                "--alignment-source-mode",
                "judge",
                "--vlm-alignment-backend",
                "mock",
                "--perception-backend",
                "mock",
                "--dag-backend",
                "simple",
                "--no-vlm-alignment-enabled",
            ]
        )
        with self.assertRaisesRegex(ValueError, "allow-debug-fallbacks"):
            _validate_runtime_args(args)

    def test_debug_settings_are_allowed_with_explicit_override(self) -> None:
        args = build_rl_parser().parse_args(
            [
                "--allow-debug-fallbacks",
                "--policy-backend",
                "scaffold",
                "--alignment-source-mode",
                "judge",
                "--vlm-alignment-backend",
                "mock",
                "--perception-backend",
                "mock",
                "--dag-backend",
                "simple",
                "--no-vlm-alignment-enabled",
            ]
        )
        _validate_runtime_args(args)

    def test_openai_backend_alias_warns_once(self) -> None:
        runtime_guards._GPT4O_ALIAS_WARNED = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(normalize_openai_backend("gpt4o", field_name="backend"), "openai")
            self.assertEqual(normalize_openai_backend("gpt4o", field_name="backend"), "openai")
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(len(deprecations), 1)

    def test_openai_perception_strict_init_fails_without_api_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OpenAI perception initialization failed"):
                OpenAIPerceptionModel(model="gpt-5-mini", api_key=None, allow_debug_fallbacks=False)

    def test_promptbn_strict_init_fails_without_api_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "PromptBN DAG initialization failed"):
                PromptBNDAGBuilder(model="gpt-5-mini", api_key=None, allow_debug_fallbacks=False)

    def test_vlm_alignment_strict_init_fails_without_api_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with tempfile.TemporaryDirectory() as td:
                cfg = VLMAlignmentConfig(
                    enabled=True,
                    source_mode="vlm_replace",
                    backend="openai",
                    model="gpt-5-mini",
                    api_key=None,
                    cache_dir=str(Path(td) / "cache"),
                    save_evidence_artifacts=False,
                )
                with self.assertRaisesRegex(RuntimeError, "VLM alignment initialization failed"):
                    VLMAlignmentVerifier(cfg=cfg, output_dir=Path(td) / "out", allow_debug_fallbacks=False)

    def test_safety_proxy_is_bounded_and_tracks_risk(self) -> None:
        judge = JudgeResult(reward=0.5, matched=False)
        low_risk = compose_reward(
            judge,
            _rollout_with_risk(collision=0.1, violation=0.0),
            RewardConfig(),
        )
        high_risk = compose_reward(
            judge,
            _rollout_with_risk(collision=0.9, violation=0.8),
            RewardConfig(),
        )
        self.assertGreaterEqual(low_risk.safety, 0.0)
        self.assertLessEqual(low_risk.safety, 1.0)
        self.assertGreaterEqual(high_risk.safety, 0.0)
        self.assertLessEqual(high_risk.safety, 1.0)
        self.assertGreater(low_risk.safety, high_risk.safety)


if __name__ == "__main__":
    unittest.main()
