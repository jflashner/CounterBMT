from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np


def summarize_stage_delta(
    *,
    controlled_stage: np.ndarray,
    baseline_stage: np.ndarray,
    inside_mask: np.ndarray,
    outside_mask: np.ndarray,
    tolerance: float,
) -> Dict[str, Any]:
    delta = controlled_stage - baseline_stage
    delta_norm = np.linalg.norm(delta, axis=-1)

    def _mask_stats(mask: np.ndarray, prefix: str) -> Dict[str, Any]:
        if not bool(mask.any()):
            return {
                f"{prefix}_count": 0,
                f"{prefix}_has_delta": False,
                f"{prefix}_max_abs": 0.0,
                f"{prefix}_mean_l2": 0.0,
                f"{prefix}_max_l2": 0.0,
            }
        masked_delta = delta[mask]
        masked_norm = delta_norm[mask]
        return {
            f"{prefix}_count": int(mask.sum()),
            f"{prefix}_has_delta": bool(np.any(masked_norm > tolerance)),
            f"{prefix}_max_abs": float(np.max(np.abs(masked_delta))),
            f"{prefix}_mean_l2": float(np.mean(masked_norm)),
            f"{prefix}_max_l2": float(np.max(masked_norm)),
        }

    summary: Dict[str, Any] = {}
    summary.update(_mask_stats(inside_mask, "inside_mask"))
    summary.update(_mask_stats(outside_mask, "outside_mask"))
    return summary


def summarize_probe_stages(
    *,
    controlled_probes: Mapping[str, np.ndarray],
    baseline_probes: Mapping[str, np.ndarray],
    selected_example_idx: int,
    inside_mask: np.ndarray,
    outside_mask: np.ndarray,
    tolerance: float,
) -> Dict[str, Dict[str, Any]]:
    summaries: Dict[str, Dict[str, Any]] = {}
    for stage_name, controlled_stage in controlled_probes.items():
        baseline_stage = baseline_probes[stage_name]
        summaries[stage_name] = summarize_stage_delta(
            controlled_stage=np.asarray(controlled_stage)[selected_example_idx],
            baseline_stage=np.asarray(baseline_stage)[selected_example_idx],
            inside_mask=inside_mask,
            outside_mask=outside_mask,
            tolerance=tolerance,
        )
    return summaries


def classify_probe_behavior(stage_summaries: Mapping[str, Mapping[str, Any]]) -> Dict[str, bool]:
    after_control_add = stage_summaries["after_control_add"]
    after_next_shared_block = stage_summaries["after_next_shared_block"]
    after_full_decoder = stage_summaries["after_full_decoder"]
    direct_leakage_bug = bool(after_control_add["outside_mask_has_delta"])
    propagated_to_non_target_positions = bool(
        (after_next_shared_block["outside_mask_has_delta"] or after_full_decoder["outside_mask_has_delta"])
        and not direct_leakage_bug
    )
    return {
        "direct_leakage_bug": direct_leakage_bug,
        "propagated_to_non_target_positions": propagated_to_non_target_positions,
    }
