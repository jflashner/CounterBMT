"""Export parity tokenizer inputs/outputs for a single scenario batch."""

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
from counter_bmt_v2.trajectory_jax import AdvBMTParityTokenizer, ParityTokenizerConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Export parity tokenization debug bundle")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--mode", type=str, default="forward", choices=["forward", "backward"])
    parser.add_argument("--skip-steps", type=int, default=5)
    parser.add_argument("--out", type=str, default="outputs/parity/export_batch")
    args = parser.parse_args()

    loader = ScenarioNetNNXLoader(data_dir=args.data_dir)
    if args.index < 0 or args.index >= len(loader):
        raise ValueError(f"index out of range: {args.index} (dataset size={len(loader)})")

    sample = loader.load(int(args.index))
    batch = collate_nnx_scene_samples([sample])
    tokenizer = AdvBMTParityTokenizer(ParityTokenizerConfig(num_skipped_steps=int(args.skip_steps)))

    legacy_like = tokenizer.build_legacy_like_inputs(batch)
    tok = tokenizer.tokenize_batch(batch, backward_prediction=(args.mode == "backward"))

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    npz_path = out_base.with_suffix(".npz")
    json_path = out_base.with_suffix(".json")

    np.savez_compressed(
        npz_path,
        sample_steps=legacy_like["sample_steps"],
        agent_position_xyz=legacy_like["decoder/agent_position"],
        agent_heading=legacy_like["decoder/agent_heading"],
        agent_valid_mask=legacy_like["decoder/agent_valid_mask"],
        agent_velocity_xy=legacy_like["decoder/agent_velocity"],
        current_agent_shape=legacy_like["decoder/current_agent_shape"],
        agent_type_ids=legacy_like["decoder/agent_type"],
        prev_token_ids=tok.prev_token_ids,
        targets=tok.targets,
        target_mask=tok.target_mask,
        continuous_motion=tok.continuous_motion,
    )

    meta = {
        "scenario_id": str(sample.scenario_id),
        "index": int(args.index),
        "mode": str(args.mode),
        "skip_steps": int(args.skip_steps),
        "n_tokens": int(tokenizer.num_actions),
        "special_ids": {
            "start_model_id": int(tokenizer.START_MODEL_ID),
            "end_model_id": int(tokenizer.END_MODEL_ID),
            "pad_model_id": int(tokenizer.PAD_MODEL_ID),
            "mask_model_id": int(tokenizer.MASK_MODEL_ID),
        },
        "paths": {
            "npz": str(npz_path),
        },
    }
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {npz_path}")
    print(f"Wrote: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
