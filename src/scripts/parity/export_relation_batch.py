"""Export relation parity tensors for one scenario batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root without editable install.
if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[2]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

from counter_bmt_v2.data import ScenarioNetNNXLoader, collate_nnx_scene_samples
from counter_bmt_v2.trajectory_jax import (
    RelationBundleConfig,
    build_relation_bundle,
    build_scene_token_relation_inputs_np,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export relation parity debug bundle")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--skip-steps", type=int, default=5)
    parser.add_argument("--out", type=str, default="outputs/parity/relation_batch")
    args = parser.parse_args()

    loader = ScenarioNetNNXLoader(data_dir=args.data_dir)
    if args.index < 0 or args.index >= len(loader):
        raise ValueError(f"index out of range: {args.index} (dataset size={len(loader)})")

    sample = loader.load(int(args.index))
    batch = collate_nnx_scene_samples([sample])

    sample_steps = np.arange(0, batch["agent_position_xy"].shape[1], max(1, int(args.skip_steps)), dtype=np.int32)
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

    bundle_cfg = RelationBundleConfig(
        simple_relation=True,
        per_contour_point_relation=False,
        include_contour=True,
        heading_placeholder=-100.0,
        s2s_knn=128,
        s2s_distance=None,
        a2s_knn=128,
        a2s_distance=None,
        a2a_knn=64,
        a2a_distance=50.0,
        remove_traffic_light_state=True,
    )
    bundle = build_relation_bundle(
        agent_position_xy=np.asarray(batch["agent_position_xy"], dtype=np.float32),
        agent_heading=np.asarray(batch["agent_heading"], dtype=np.float32),
        agent_valid_mask=np.asarray(batch["agent_valid_mask"], dtype=bool),
        agent_shape=np.asarray(batch["agent_shape"], dtype=np.float32),
        sample_steps=sample_steps,
        scene_position=scene_inputs["scene_position"],
        scene_heading=scene_inputs["scene_heading"],
        scene_valid_mask=scene_inputs["scene_valid_mask"],
        cfg=bundle_cfg,
    )

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    npz_path = out_base.with_suffix(".npz")
    json_path = out_base.with_suffix(".json")

    np.savez_compressed(
        npz_path,
        scenario_ids=np.asarray(batch["scenario_ids"], dtype=object),
        scene_position=scene_inputs["scene_position"],
        scene_heading=scene_inputs["scene_heading"],
        scene_valid_mask=scene_inputs["scene_valid_mask"],
        **bundle,
    )

    meta = {
        "scenario_id": str(sample.scenario_id),
        "index": int(args.index),
        "skip_steps": int(args.skip_steps),
        "relation_cfg": {
            "simple_relation": True,
            "per_contour_point_relation": False,
            "remove_traffic_light_state": True,
            "s2s_knn": 128,
            "a2s_knn": 128,
            "a2a_knn": 64,
            "a2a_distance": 50.0,
        },
        "paths": {"npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {npz_path}")
    print(f"Wrote: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
