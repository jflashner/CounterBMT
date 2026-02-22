"""Export decoder-parity tensors for one scenario batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export decoder parity tensors")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--skip-steps", type=int, default=5)
    p.add_argument("--out", type=str, default="outputs/parity/decoder_batch_0")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    loader = ScenarioNetNNXLoader(data_dir=args.data_dir)
    if len(loader) == 0:
        raise RuntimeError(f"empty dataset: {args.data_dir}")
    idx = max(0, min(int(args.index), len(loader) - 1))

    sample = loader.load(idx)
    batch = collate_nnx_scene_samples([sample])
    tokenizer = AdvBMTParityTokenizer(ParityTokenizerConfig(num_skipped_steps=int(args.skip_steps)))
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
    rel_cfg = RelationBundleConfig(
        simple_relation=True,
        per_contour_point_relation=False,
        include_contour=True,
        s2s_knn=128,
        a2s_knn=128,
        a2a_knn=64,
        a2a_distance=50.0,
        strict_non_agent_relation=True,
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

    model = NNXBidirectionalMotionTransformer(midgpt_parity_config(), rngs=nnx.Rngs(0))
    reverse = np.zeros((1,), dtype=np.int32)
    dec_inp, _ = model._compose_decoder_tokens_parity(
        prev_token_ids=jnp.asarray(tok.prev_token_ids, dtype=jnp.int32),
        input_action_valid_mask=jnp.asarray(tok.input_mask, dtype=bool),
        modeled_agent_delta=jnp.asarray(tok.modeled_agent_delta, dtype=jnp.float32),
        agent_type_ids=jnp.asarray(batch["agent_type_ids"], dtype=jnp.int32),
        agent_shape=jnp.asarray(batch["agent_shape"], dtype=jnp.float32),
        agent_ids=jnp.asarray(batch["agent_ids"], dtype=jnp.int32),
        reverse_indicator=jnp.asarray(reverse, dtype=jnp.int32),
    )
    dec_inp = np.asarray(jax.device_get(dec_inp), dtype=np.float32)

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = out_prefix.with_suffix(".npz")
    json_path = out_prefix.with_suffix(".json")

    bundle_npz = dict(bundle)
    bundle_npz.pop("sample_steps", None)
    np.savez_compressed(
        npz_path,
        scenario_ids=np.asarray(batch["scenario_ids"], dtype=object),
        sample_steps=np.asarray(tok.sample_steps, dtype=np.int32),
        prev_token_ids=np.asarray(tok.prev_token_ids, dtype=np.int32),
        input_mask=np.asarray(tok.input_mask, dtype=bool),
        modeled_agent_delta=np.asarray(tok.modeled_agent_delta, dtype=np.float32),
        targets=np.asarray(tok.targets, dtype=np.int32),
        target_mask=np.asarray(tok.target_mask, dtype=np.float32),
        decoder_input_embedding=dec_inp,
        scene_position=np.asarray(scene_inputs["scene_position"], dtype=np.float32),
        scene_heading=np.asarray(scene_inputs["scene_heading"], dtype=np.float32),
        scene_valid_mask=np.asarray(scene_inputs["scene_valid_mask"], dtype=bool),
        **bundle_npz,
    )

    meta = {
        "data_dir": str(args.data_dir),
        "index": int(idx),
        "scenario_id": str(batch["scenario_ids"][0]),
        "skip_steps": int(args.skip_steps),
        "npz_path": str(npz_path),
    }
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote decoder parity bundle: {npz_path}")
    print(f"Wrote metadata: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
