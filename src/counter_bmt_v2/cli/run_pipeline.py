"""CLI entrypoint for CounterBMT v2 vertical-slice pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from counter_bmt_v2.contracts import ScenarioInput, TimestampedFrame, make_demo_frames
from counter_bmt_v2.orchestration import CounterBMTPipeline


def build_demo_scene(scenario_id: str, num_frames: int) -> ScenarioInput:
    timestamps = np.linspace(0.0, 4.0, num=num_frames).tolist()
    frames = make_demo_frames(prefix=f"{scenario_id}_frame", timestamps=timestamps)

    # Synthetic trajectory used only as a seed state carrier for now.
    t = np.linspace(0.0, 1.0, num=30, dtype=np.float32)
    x = 5.0 * t
    y = 0.5 * np.sin(2.0 * np.pi * t)
    traj = np.stack([x, y], axis=1)

    return ScenarioInput(
        scenario_id=scenario_id,
        frames=frames,
        ego_trajectory_xy=traj,
        metadata={"source": "demo"},
    )


def build_scenarionet_scene(
    data_dir: Path,
    scenario_index: int,
    num_frames: int,
    frame_output_dir: Path,
) -> ScenarioInput:
    """Build a ScenarioInput from a real ScenarioNet scene."""
    from counter_bmt.scenarionet_visualizer import prepare_for_vlm

    frame_output_dir.mkdir(parents=True, exist_ok=True)
    saved_images, trajectory, scenario_id = prepare_for_vlm(
        data_dir=str(data_dir),
        scenario_index=scenario_index,
        output_dir=str(frame_output_dir),
        num_frames=num_frames,
    )

    frames = [
        TimestampedFrame(path=str(path), timestamp_s=float(ts))
        for path, ts in saved_images
    ]
    ego_xy = None
    if trajectory is not None and len(trajectory) > 0:
        arr = np.asarray(trajectory)
        # prepare_for_vlm returns [x, y, heading, speed]; keep xy only.
        ego_xy = arr[:, :2].astype(np.float32)

    return ScenarioInput(
        scenario_id=scenario_id,
        frames=frames,
        ego_trajectory_xy=ego_xy,
        metadata={
            "source": "scenarionet",
            "data_dir": str(data_dir),
            "scenario_index": int(scenario_index),
            "frame_output_dir": str(frame_output_dir),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CounterBMT v2 pipeline (vertical slice)")
    parser.add_argument("--scenario-id", type=str, default="demo_000")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rare", action="store_true", help="sample tail-style intervention")
    parser.add_argument(
        "--perception-backend",
        type=str,
        default="mock",
        choices=["mock", "gpt4o"],
        help="perception extractor backend",
    )
    parser.add_argument(
        "--dag-backend",
        type=str,
        default="simple",
        choices=["simple", "promptbn"],
        help="DAG builder backend",
    )
    parser.add_argument("--llm-model", type=str, default="gpt-4o")
    parser.add_argument("--api-key", type=str, default=None, help="optional OpenAI API key")
    parser.add_argument("--dag-retries", type=int, default=4, help="max PromptBN retry count")
    parser.add_argument(
        "--scene-source",
        type=str,
        default="demo",
        choices=["demo", "scenarionet"],
        help="scene source: synthetic demo or real ScenarioNet dataset",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="ScenarioNet dataset directory (required when --scene-source scenarionet)",
    )
    parser.add_argument(
        "--scenario-index",
        type=int,
        default=0,
        help="Scenario index in data-dir (used when --scene-source scenarionet)",
    )
    parser.add_argument(
        "--frame-output-dir",
        type=str,
        default="",
        help="Optional output directory for extracted frames",
    )
    parser.add_argument("--json-out", type=str, default="", help="optional output JSON path")
    args = parser.parse_args()

    if args.scene_source == "scenarionet":
        if not args.data_dir:
            parser.error("--data-dir is required when --scene-source scenarionet")
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            parser.error(f"data directory not found: {data_dir}")

        if args.frame_output_dir:
            frame_output_dir = Path(args.frame_output_dir)
        else:
            frame_output_dir = Path("outputs") / "counter_bmt_v2_frames" / f"idx_{args.scenario_index:06d}"

        scene = build_scenarionet_scene(
            data_dir=data_dir,
            scenario_index=args.scenario_index,
            num_frames=args.num_frames,
            frame_output_dir=frame_output_dir,
        )
    else:
        scene = build_demo_scene(args.scenario_id, args.num_frames)

    pipeline = CounterBMTPipeline.from_backends(
        perception_backend=args.perception_backend,
        dag_backend=args.dag_backend,
        llm_model=args.llm_model,
        api_key=args.api_key,
        dag_retries=args.dag_retries,
    )
    result = pipeline.run(scene, n_samples=args.n_samples, seed=args.seed, rare=args.rare)

    mean_reward = float(np.mean([r.total for r in result.rewards])) if result.rewards else 0.0
    print(f"scenario={result.scenario_id}")
    print(f"intervention={result.intervention.variable}:{result.intervention.value}")
    print(f"samples={len(result.rollouts)}")
    print(f"mean_reward={mean_reward:.3f}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"saved={out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
