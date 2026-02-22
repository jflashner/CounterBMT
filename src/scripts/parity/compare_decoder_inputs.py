"""Compare decoder input parity intermediates with optional legacy tokenizer outputs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[2]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

from counter_bmt_v2.data import ScenarioNetNNXLoader, collate_nnx_scene_samples
from counter_bmt_v2.trajectory_jax import (
    AdvBMTParityTokenizer,
    NNXBidirectionalMotionTransformer,
    ParityTokenizerConfig,
    RelationBundleConfig,
    build_relation_bundle,
    build_scene_token_relation_inputs_np,
    midgpt_parity_config,
)


@dataclass
class CompareStats:
    batches: int = 0
    mask_total: int = 0
    mask_match: int = 0
    special_total: int = 0
    special_match: int = 0
    delta_abs_max: float = 0.0
    delta_abs_sum: float = 0.0
    delta_abs_count: int = 0
    emb_abs_max: float = 0.0
    emb_abs_sum: float = 0.0
    emb_abs_count: int = 0
    emb_abs_max_common_valid: float = 0.0
    emb_abs_sum_common_valid: float = 0.0
    emb_abs_count_common_valid: int = 0
    edge_rel_delta_pct_max: float = 0.0
    has_nan: bool = False

    def to_metrics(self) -> Dict[str, float]:
        return {
            "batches": float(self.batches),
            "input_mask_match_rate": float(self.mask_match / self.mask_total) if self.mask_total > 0 else float("nan"),
            "special_token_class_match_rate": (
                float(self.special_match / self.special_total) if self.special_total > 0 else float("nan")
            ),
            "modeled_agent_delta_abs_max": float(self.delta_abs_max),
            "modeled_agent_delta_abs_mean": (
                float(self.delta_abs_sum / self.delta_abs_count) if self.delta_abs_count > 0 else float("nan")
            ),
            "decoder_embedding_abs_max": float(self.emb_abs_max),
            "decoder_embedding_abs_mean": (
                float(self.emb_abs_sum / self.emb_abs_count) if self.emb_abs_count > 0 else float("nan")
            ),
            "decoder_embedding_abs_max_common_valid": float(self.emb_abs_max_common_valid),
            "decoder_embedding_abs_mean_common_valid": (
                float(self.emb_abs_sum_common_valid / self.emb_abs_count_common_valid)
                if self.emb_abs_count_common_valid > 0
                else float("nan")
            ),
            "relation_edge_delta_pct_max": float(self.edge_rel_delta_pct_max),
            "has_nan": float(self.has_nan),
        }


def _iter_batches(indices: np.ndarray, batch_size: int) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for i in range(0, len(indices), max(1, int(batch_size))):
        out.append(indices[i : i + max(1, int(batch_size))])
    return out


def _legacy_input_to_model_ids(actions: np.ndarray, tokenizer: AdvBMTParityTokenizer) -> np.ndarray:
    out = np.full(actions.shape, tokenizer.PAD_MODEL_ID, dtype=np.int32)
    valid = actions >= 0
    out[valid] = actions[valid].astype(np.int32)
    out[actions == tokenizer.START_ACTION] = tokenizer.START_MODEL_ID
    out[actions == tokenizer.END_ACTION] = tokenizer.END_MODEL_ID
    out[actions == tokenizer.INVALID_ACTION] = tokenizer.PAD_MODEL_ID
    return out


def _special_class(prev_token_ids: np.ndarray, n_tokens: int) -> np.ndarray:
    cls = np.zeros_like(prev_token_ids, dtype=np.int32)  # normal
    cls[prev_token_ids == n_tokens] = 1  # start
    cls[prev_token_ids == (n_tokens + 1)] = 2  # end
    cls[(prev_token_ids == (n_tokens + 2)) | (prev_token_ids == (n_tokens + 3))] = 3  # pad/mask
    return cls


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare decoder input parity intermediates")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--skip-steps", type=int, default=5)
    p.add_argument("--legacy-check", action="store_true")
    p.add_argument("--legacy-root", type=str, default="src/Adv-BMT")
    p.add_argument("--max-embedding-diff", type=float, default=2e-4)
    p.add_argument("--min-mask-match", type=float, default=1.0)
    p.add_argument("--dump-artifacts", action="store_true")
    p.add_argument("--out-dir", type=str, default="outputs/parity")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    loader = ScenarioNetNNXLoader(data_dir=args.data_dir)
    if len(loader) == 0:
        raise RuntimeError(f"empty dataset: {args.data_dir}")

    tokenizer = AdvBMTParityTokenizer(ParityTokenizerConfig(num_skipped_steps=int(args.skip_steps)))
    model_cfg = midgpt_parity_config()
    model = NNXBidirectionalMotionTransformer(model_cfg, rngs=nnx.Rngs(0))

    legacy_runner: Optional[object] = None
    if args.legacy_check:
        # Import from sibling script to reuse shim-first legacy loading.
        script_dir = Path(__file__).resolve().parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        from compare_tokenization import LegacyTokenizerRunner  # type: ignore

        legacy_runner = LegacyTokenizerRunner(Path(args.legacy_root), int(args.skip_steps))

    n = min(int(args.n), len(loader))
    indices = np.arange(n, dtype=np.int32)
    stats = CompareStats()

    dump_dir = Path(args.out_dir)
    if args.dump_artifacts:
        dump_dir.mkdir(parents=True, exist_ok=True)

    for batch_id, idx_batch in enumerate(_iter_batches(indices, int(args.batch_size))):
        samples = [loader.load(int(i)) for i in idx_batch]
        batch = collate_nnx_scene_samples(samples)
        tok = tokenizer.tokenize_batch(batch, backward_prediction=False)
        stats.batches += 1

        input_mask_v2 = np.asarray(tok.input_mask, dtype=bool)
        modeled_delta_v2 = np.asarray(tok.modeled_agent_delta, dtype=np.float32)
        prev_v2 = np.asarray(tok.prev_token_ids, dtype=np.int32)
        special_v2 = _special_class(prev_v2, int(tokenizer.cfg.n_tokens))

        rel_cfg_fast = RelationBundleConfig(
            simple_relation=True,
            per_contour_point_relation=False,
            include_contour=True,
            s2s_knn=128,
            a2s_knn=128,
            a2a_knn=64,
            a2a_distance=50.0,
            strict_non_agent_relation=False,
        )
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
        bundle_fast = build_relation_bundle(
            agent_position_xy=np.asarray(batch["agent_position_xy"], dtype=np.float32),
            agent_heading=np.asarray(batch["agent_heading"], dtype=np.float32),
            agent_valid_mask=np.asarray(batch["agent_valid_mask"], dtype=bool),
            decoder_valid_mask=input_mask_v2,
            agent_shape=np.asarray(batch["agent_shape"], dtype=np.float32),
            sample_steps=np.asarray(tok.sample_steps, dtype=np.int32),
            scene_position=scene_inputs["scene_position"],
            scene_heading=scene_inputs["scene_heading"],
            scene_valid_mask=scene_inputs["scene_valid_mask"],
            cfg=rel_cfg_fast,
        )
        bundle_strict = bundle_fast
        if args.legacy_check:
            rel_cfg_strict = RelationBundleConfig(
                simple_relation=True,
                per_contour_point_relation=False,
                include_contour=True,
                s2s_knn=128,
                a2s_knn=128,
                a2a_knn=64,
                a2a_distance=50.0,
                strict_non_agent_relation=True,
            )
            bundle_strict = build_relation_bundle(
                agent_position_xy=np.asarray(batch["agent_position_xy"], dtype=np.float32),
                agent_heading=np.asarray(batch["agent_heading"], dtype=np.float32),
                agent_valid_mask=np.asarray(batch["agent_valid_mask"], dtype=bool),
                decoder_valid_mask=input_mask_v2,
                agent_shape=np.asarray(batch["agent_shape"], dtype=np.float32),
                sample_steps=np.asarray(tok.sample_steps, dtype=np.int32),
                scene_position=scene_inputs["scene_position"],
                scene_heading=scene_inputs["scene_heading"],
                scene_valid_mask=scene_inputs["scene_valid_mask"],
                cfg=rel_cfg_strict,
            )
        for name in ("a2a_mask", "a2t_mask", "a2s_mask"):
            fast_cnt = float(np.sum(bundle_fast[name]))
            strict_cnt = float(np.sum(bundle_strict[name]))
            delta = abs(fast_cnt - strict_cnt) / max(1.0, strict_cnt)
            stats.edge_rel_delta_pct_max = max(stats.edge_rel_delta_pct_max, float(delta * 100.0))

        reverse = np.zeros((prev_v2.shape[0],), dtype=np.int32)
        h_v2, _ = model._compose_decoder_tokens_parity(
            prev_token_ids=jnp.asarray(prev_v2, dtype=jnp.int32),
            input_action_valid_mask=jnp.asarray(input_mask_v2, dtype=bool),
            modeled_agent_delta=jnp.asarray(modeled_delta_v2, dtype=jnp.float32),
            agent_type_ids=jnp.asarray(batch["agent_type_ids"], dtype=jnp.int32),
            agent_shape=jnp.asarray(batch["agent_shape"], dtype=jnp.float32),
            agent_ids=jnp.asarray(batch["agent_ids"], dtype=jnp.int32),
            reverse_indicator=jnp.asarray(reverse, dtype=jnp.int32),
        )
        h_v2_np = np.asarray(jax.device_get(h_v2), dtype=np.float32)
        stats.has_nan = bool(stats.has_nan or np.isnan(h_v2_np).any())

        if legacy_runner is not None:
            legacy = legacy_runner.tokenize_batch(batch, backward_prediction=False)  # type: ignore[attr-defined]
            prev_legacy = _legacy_input_to_model_ids(np.asarray(legacy["input_action"], dtype=np.int32), tokenizer)
            mask_legacy = np.asarray(legacy["input_mask"], dtype=bool)
            delta_legacy = np.asarray(legacy["modeled_agent_delta"], dtype=np.float32)
            special_legacy = _special_class(prev_legacy, int(tokenizer.cfg.n_tokens))

            shape_t = min(prev_v2.shape[1], prev_legacy.shape[1])
            v2_m = input_mask_v2[:, :shape_t]
            lg_m = mask_legacy[:, :shape_t]
            v2_s = special_v2[:, :shape_t]
            lg_s = special_legacy[:, :shape_t]
            v2_d = modeled_delta_v2[:, :shape_t]
            lg_d = delta_legacy[:, :shape_t]

            stats.mask_total += int(v2_m.size)
            stats.mask_match += int(np.sum(v2_m == lg_m))
            stats.special_total += int(v2_s.size)
            stats.special_match += int(np.sum(v2_s == lg_s))

            d_abs = np.abs(v2_d - lg_d)
            stats.delta_abs_max = max(stats.delta_abs_max, float(np.max(d_abs)))
            stats.delta_abs_sum += float(np.sum(d_abs))
            stats.delta_abs_count += int(d_abs.size)

            h_lg, _ = model._compose_decoder_tokens_parity(
                prev_token_ids=jnp.asarray(prev_legacy[:, :shape_t], dtype=jnp.int32),
                input_action_valid_mask=jnp.asarray(mask_legacy[:, :shape_t], dtype=bool),
                modeled_agent_delta=jnp.asarray(delta_legacy[:, :shape_t], dtype=jnp.float32),
                agent_type_ids=jnp.asarray(batch["agent_type_ids"], dtype=jnp.int32),
                agent_shape=jnp.asarray(batch["agent_shape"], dtype=jnp.float32),
                agent_ids=jnp.asarray(batch["agent_ids"], dtype=jnp.int32),
                reverse_indicator=jnp.asarray(reverse, dtype=jnp.int32),
            )
            h_lg_np = np.asarray(jax.device_get(h_lg), dtype=np.float32)
            h_v2_cmp = h_v2_np[:, :shape_t]
            emb_abs = np.abs(h_v2_cmp - h_lg_np)
            stats.emb_abs_max = max(stats.emb_abs_max, float(np.max(emb_abs)))
            stats.emb_abs_sum += float(np.sum(emb_abs))
            stats.emb_abs_count += int(emb_abs.size)

            common_valid = np.logical_and(v2_m, lg_m)
            if np.any(common_valid):
                common_valid_4d = np.broadcast_to(common_valid[..., None], emb_abs.shape)
                emb_abs_common = emb_abs[common_valid_4d]
                stats.emb_abs_max_common_valid = max(
                    stats.emb_abs_max_common_valid,
                    float(np.max(emb_abs_common)),
                )
                stats.emb_abs_sum_common_valid += float(np.sum(emb_abs_common))
                stats.emb_abs_count_common_valid += int(emb_abs_common.size)
            stats.has_nan = bool(stats.has_nan or np.isnan(h_lg_np).any())

            if args.dump_artifacts and batch_id == 0:
                np.savez_compressed(
                    dump_dir / "decoder_compare_batch0.npz",
                    prev_v2=prev_v2[:, :shape_t],
                    prev_legacy=prev_legacy[:, :shape_t],
                    input_mask_v2=v2_m,
                    input_mask_legacy=lg_m,
                    modeled_delta_v2=v2_d,
                    modeled_delta_legacy=lg_d,
                    special_v2=v2_s,
                    special_legacy=lg_s,
                    emb_v2=h_v2_cmp,
                    emb_legacy=h_lg_np,
                    a2a_mask_fast=np.asarray(bundle_fast["a2a_mask"], dtype=bool),
                    a2t_mask_fast=np.asarray(bundle_fast["a2t_mask"], dtype=bool),
                    a2s_mask_fast=np.asarray(bundle_fast["a2s_mask"], dtype=bool),
                )

    payload = {
        "config": {
            "data_dir": args.data_dir,
            "n": int(args.n),
            "batch_size": int(args.batch_size),
            "skip_steps": int(args.skip_steps),
            "legacy_check": bool(args.legacy_check),
            "legacy_root": str(args.legacy_root),
        },
        "metrics": stats.to_metrics(),
    }
    print(json.dumps(payload, indent=2))

    ok = True
    metrics = payload["metrics"]
    if args.legacy_check:
        m = float(metrics["input_mask_match_rate"])
        if np.isfinite(m) and m < float(args.min_mask_match):
            ok = False
        emax = float(metrics["decoder_embedding_abs_max_common_valid"])
        if not np.isfinite(emax):
            emax = float(metrics["decoder_embedding_abs_max"])
        if np.isfinite(emax) and emax > float(args.max_embedding_diff):
            ok = False
    if bool(metrics["has_nan"]):
        ok = False
    if not ok:
        print("FAILED: decoder input parity check did not meet thresholds", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
