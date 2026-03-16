from __future__ import annotations

import unittest

from counter_bmt_v2.training.supervised import SupervisedTrainConfig, _validate_resume_compatibility


class ResumeCompatibilityTests(unittest.TestCase):
    def test_precision_mismatch_fails_strict_resume(self) -> None:
        cfg = SupervisedTrainConfig(
            model_preset="midgpt_parity",
            tokenizer_mode="adv_bmt_parity",
            skip_steps=5,
            precision="bf16-mixed",
        )
        with self.assertRaisesRegex(ValueError, r"train_cfg\[precision\] mismatch"):
            _validate_resume_compatibility(
                train_cfg=cfg,
                split_hashes={"train": "abc", "val": "def"},
                resume_runtime_state={"split_hashes": {"train": "abc", "val": "def"}},
                resume_payload={
                    "train_cfg": {
                        "model_preset": "midgpt_parity",
                        "tokenizer_mode": "adv_bmt_parity",
                        "skip_steps": 5,
                        "precision": "fp32",
                    }
                },
            )

    def test_precision_match_passes_strict_resume(self) -> None:
        cfg = SupervisedTrainConfig(
            model_preset="midgpt_parity",
            tokenizer_mode="adv_bmt_parity",
            skip_steps=5,
            precision="fp32",
        )
        _validate_resume_compatibility(
            train_cfg=cfg,
            split_hashes={"train": "abc", "val": "def"},
            resume_runtime_state={"split_hashes": {"train": "abc", "val": "def"}},
            resume_payload={
                "train_cfg": {
                    "model_preset": "midgpt_parity",
                    "tokenizer_mode": "adv_bmt_parity",
                    "skip_steps": 5,
                    "precision": "fp32",
                }
            },
        )


if __name__ == "__main__":
    unittest.main()
