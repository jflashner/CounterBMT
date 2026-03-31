from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGACY_SRC = ROOT / "src" / "Adv-BMT"
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))

from bmt.counterfactual.runtime_probe import classify_probe_behavior, summarize_probe_stages


class RuntimeProbeTests(unittest.TestCase):
    def test_classifies_direct_leakage_bug(self) -> None:
        inside_mask = np.array([[True, False], [False, False]])
        outside_mask = np.array([[False, True], [True, True]])
        zero = np.zeros((1, 2, 2, 3), dtype=np.float32)
        after_control_add = zero.copy()
        after_control_add[0, 0, 1, 0] = 0.25

        stage_summaries = summarize_probe_stages(
            controlled_probes={
                "after_control_add": after_control_add,
                "after_next_shared_block": after_control_add,
                "after_full_decoder": after_control_add,
            },
            baseline_probes={
                "after_control_add": zero,
                "after_next_shared_block": zero,
                "after_full_decoder": zero,
            },
            selected_example_idx=0,
            inside_mask=inside_mask,
            outside_mask=outside_mask,
            tolerance=1e-6,
        )

        behavior = classify_probe_behavior(stage_summaries)
        self.assertTrue(stage_summaries["after_control_add"]["outside_mask_has_delta"])
        self.assertTrue(behavior["direct_leakage_bug"])
        self.assertFalse(behavior["propagated_to_non_target_positions"])

    def test_classifies_later_propagation_without_direct_leakage(self) -> None:
        inside_mask = np.array([[True, False], [False, False]])
        outside_mask = np.array([[False, True], [True, True]])
        zero = np.zeros((1, 2, 2, 3), dtype=np.float32)
        after_control_add = zero.copy()
        after_control_add[0, 0, 0, 0] = 1.0
        after_next_shared_block = after_control_add.copy()
        after_next_shared_block[0, 0, 1, 0] = 0.5
        after_full_decoder = after_next_shared_block.copy()

        stage_summaries = summarize_probe_stages(
            controlled_probes={
                "after_control_add": after_control_add,
                "after_next_shared_block": after_next_shared_block,
                "after_full_decoder": after_full_decoder,
            },
            baseline_probes={
                "after_control_add": zero,
                "after_next_shared_block": zero,
                "after_full_decoder": zero,
            },
            selected_example_idx=0,
            inside_mask=inside_mask,
            outside_mask=outside_mask,
            tolerance=1e-6,
        )

        behavior = classify_probe_behavior(stage_summaries)
        self.assertFalse(stage_summaries["after_control_add"]["outside_mask_has_delta"])
        self.assertTrue(stage_summaries["after_next_shared_block"]["outside_mask_has_delta"])
        self.assertFalse(behavior["direct_leakage_bug"])
        self.assertTrue(behavior["propagated_to_non_target_positions"])


if __name__ == "__main__":
    unittest.main()
