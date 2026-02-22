"""Inspect decoder causal-valid mask semantics for MidGPT parity mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[2]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

from counter_bmt_v2.data import ScenarioNetNNXLoader, collate_nnx_scene_samples
from counter_bmt_v2.trajectory_jax import (
    AdvBMTParityTokenizer,
    ParityTokenizerConfig,
    RelationBundleConfig,
    build_relation_bundle,
    build_scene_token_relation_inputs_np,
)


def create_causal_mask_legacy_semantics(T: int, N: int) -> np.ndarray:
    """Legacy equivalent of create_causal_mask(T, N, is_valid_mask=True)."""
    block = np.ones((N, N), dtype=bool)
    tril = np.tril(np.ones((T, T), dtype=bool))
    return np.kron(tril, block)


def _iter_batches(indices: np.ndarray, batch_size: int) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for i in range(0, len(indices), max(1, int(batch_size))):
        out.append(indices[i : i + max(1, int(batch_size))])
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect decoder A2T causal-valid masks")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--skip-steps", type=int, default=5)
    p.add_argument("--min-match", type=float, default=1.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    loader = ScenarioNetNNXLoader(data_dir=args.data_dir)
    if len(loader) == 0:
        raise RuntimeError(f"empty dataset: {args.data_dir}")

    n = min(int(args.n), len(loader))
    indices = np.arange(n, dtype=np.int32)
    tokenizer = AdvBMTParityTokenizer(ParityTokenizerConfig(num_skipped_steps=int(args.skip_steps)))
    rel_cfg = RelationBundleConfig(
        simple_relation=True,
        per_contour_point_relation=False,
        include_contour=True,
        s2s_knn=128,
        a2a_knn=64,
        a2a_distance=50.0,
        a2s_knn=128,
        strict_non_agent_relation=False,
    )

    total = 0
    match = 0
    causal_total = 0
    causal_match = 0
    has_nan = False

    for idx_batch in _iter_batches(indices, int(args.batch_size)):
        samples = [loader.load(int(i)) for i in idx_batch]
        batch = collate_nnx_scene_samples(samples)
        tok = tokenizer.tokenize_batch(batch, backward_prediction=False)

        scene_inputs = build_scene_token_relation_inputs_np(
            map_feature=np.asarray(batch["map_feature"], dtype=np.float32),
            map_feature_valid_mask=np.asarray(batch["map_feature_valid_mask"], dtype=bool),
            map_position=np.asarray(batch["map_position"], dtype=np.float32),
            traffic_light_feature=np.asarray(batch["traffic_light_feature"], dtype=np.float32),
            traffic_light_valid_mask=np.asarray(batch["traffic_light_valid_mask"], dtype=bool),
            traffic_light_position=np.asarray(batch["traffic_light_position"], dtype=np.float32),
            remove_traffic_light_state=True,
            heading_placeholder=-100.0,
        )
        bundle = build_relation_bundle(
            agent_position_xy=np.asarray(batch["agent_position_xy"], dtype=np.float32),
            agent_heading=np.asarray(batch["agent_heading"], dtype=np.float32),
            agent_valid_mask=np.asarray(batch["agent_valid_mask"], dtype=bool),
            decoder_valid_mask=np.asarray(tok.input_mask, dtype=bool),
            agent_shape=np.asarray(batch["agent_shape"], dtype=np.float32),
            sample_steps=np.asarray(tok.sample_steps, dtype=np.int32),
            scene_position=scene_inputs["scene_position"],
            scene_heading=scene_inputs["scene_heading"],
            scene_valid_mask=scene_inputs["scene_valid_mask"],
            cfg=rel_cfg,
        )

        a2t_mask = np.asarray(bundle["a2t_mask"], dtype=bool)  # [B,N,T,T]
        B, N, T, _ = a2t_mask.shape
        has_nan = bool(has_nan or np.isnan(a2t_mask.astype(np.float32)).any())

        # Legacy create_causal_mask(T, N=1, is_valid_mask=True) semantics.
        causal_t = create_causal_mask_legacy_semantics(T=T, N=1).reshape(T, T)
        causal = np.broadcast_to(causal_t[None, None, :, :], (B, N, T, T))

        input_valid_bnt = np.transpose(np.asarray(tok.input_mask, dtype=bool), (0, 2, 1))
        input_valid_bnt = input_valid_bnt[:, :, :T]
        expected = (
            input_valid_bnt[:, :, :, None]
            & input_valid_bnt[:, :, None, :]
            & causal
        )
        observed = a2t_mask & causal

        total += int(expected.size)
        match += int(np.sum(expected == observed))

        causal_total += int(causal.size)
        causal_match += int(np.sum(causal == np.broadcast_to(causal_t[None, None, :, :], (B, N, T, T))))

    metrics: Dict[str, float] = {
        "num_scenarios": float(n),
        "a2t_causal_valid_match_rate": float(match / total) if total > 0 else float("nan"),
        "causal_semantics_match_rate": float(causal_match / causal_total) if causal_total > 0 else float("nan"),
        "has_nan": float(has_nan),
    }
    payload = {
        "config": {
            "data_dir": args.data_dir,
            "n": int(args.n),
            "batch_size": int(args.batch_size),
            "skip_steps": int(args.skip_steps),
        },
        "metrics": metrics,
    }
    print(json.dumps(payload, indent=2))

    ok = True
    if np.isfinite(metrics["a2t_causal_valid_match_rate"]) and metrics["a2t_causal_valid_match_rate"] < float(args.min_match):
        ok = False
    if has_nan:
        ok = False
    if not ok:
        print("FAILED: decoder causal-valid mask parity check did not meet threshold", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
