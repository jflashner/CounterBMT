"""
rl_vlm_judge.py

Lightweight, hackable VLM-judge script for CounterBMT.

Goal:
    Render a scenario (including counterfactuals), run VLM extraction,
    and score whether a desired intervention happened.

This is intentionally minimal so it can be adapted into a full RL loop:
    controller -> BMT rollout -> render -> VLM judge -> reward
"""

import argparse
import json
import logging
import pickle
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from counter_bmt.scenarionet_visualizer import ScenarioNetVisualizer, ScenarioNetDatabase
from counter_bmt.vlm_extractor import (
    VLMSafetyCriticalExtractor,
    TimestampedImage,
    GPT4oClient,
    MockGPT4oClient,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rl_vlm_judge")


# =============================================================================
# Target parsing + reward
# =============================================================================

def _normalize_choice(s: str) -> str:
    return s.lower().strip().replace(" ", "_").replace("-", "_")


def parse_target(target: Optional[str], intervention_name: Optional[str]) -> Tuple[str, str]:
    """
    Parse desired intervention target.

    Supported formats:
        --target "maneuver:lane_change_right"
        --target "decision:proceed"

    If target not provided, infer from intervention_name heuristics.
    """
    if target:
        parts = target.split(":", 1)
        if len(parts) == 2:
            return parts[0].strip().lower(), _normalize_choice(parts[1])
        return "maneuver", _normalize_choice(target)

    if not intervention_name:
        return "unknown", "unknown"

    name = _normalize_choice(intervention_name)

    # Maneuver heuristics
    for key in [
        "lane_change_left",
        "lane_change_right",
        "left_turn",
        "right_turn",
        "accelerate",
        "decelerate",
        "stop",
        "straight",
    ]:
        if key in name:
            return "maneuver", key

    # Decision heuristics
    if "proceed" in name:
        return "decision", "proceed"
    if "yield" in name:
        return "decision", "yield"
    if "left" in name and "turn" in name:
        return "maneuver", "left_turn"
    if "right" in name and "turn" in name:
        return "maneuver", "right_turn"

    return "unknown", name


def score_features(features, target_type: str, target_value: str) -> Tuple[float, Dict]:
    """
    Very simple reward function.

    Returns:
        reward (0..1), details dict
    """
    details = {"target_type": target_type, "target_value": target_value}

    if target_type == "maneuver":
        best = 0.0
        best_match = None
        for m in features.maneuver_sequence:
            if m.maneuver_type.value == target_value:
                conf = float(getattr(m, "confidence", 1.0) or 1.0)
                if conf > best:
                    best = conf
                    best_match = m
        details["match"] = asdict(best_match) if best_match else None
        return best, details

    if target_type == "decision":
        best = 0.0
        best_match = None
        for d in features.critical_decisions:
            choice = _normalize_choice(d.ground_truth_choice)
            if choice == target_value:
                conf = float(getattr(d, "confidence", 1.0) or 1.0)
                if conf > best:
                    best = conf
                    best_match = d
        details["match"] = asdict(best_match) if best_match else None
        return best, details

    details["match"] = None
    return 0.0, details


# =============================================================================
# Main
# =============================================================================

def load_intervention_name(scenario_path: Path) -> Optional[str]:
    try:
        with open(scenario_path, "rb") as f:
            scenario = pickle.load(f)
        return scenario.get("metadata", {}).get("intervention")
    except Exception as e:
        logger.warning(f"Failed to read intervention name: {scenario_path} ({e})")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLM judge for CounterBMT interventions (lightweight RL reward)"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory with ScenarioNet/MetaDrive .pkl scenarios (counterfactual replays)",
    )
    parser.add_argument("--scenario", type=int, default=None, help="Scenario index to evaluate")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames to render")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for logs")
    parser.add_argument("--target", type=str, default=None, help="Override target (e.g., maneuver:lane_change_right)")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    parser.add_argument("--vlm-model", type=str, default="gpt-4o", help="VLM model name")
    parser.add_argument("--mock", action="store_true", help="Use mock VLM client")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / "rl_vlm_judge" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # VLM client
    if args.mock:
        client = MockGPT4oClient()
    else:
        if args.api_key:
            import os
            os.environ["OPENAI_API_KEY"] = args.api_key
        client = GPT4oClient(model=args.vlm_model)

    extractor = VLMSafetyCriticalExtractor(
        client=client,
        debug=True,
        debug_output_dir=str(output_dir),
    )

    # Scenario DB for indexing
    db = ScenarioNetDatabase(str(data_dir))
    indices = [args.scenario] if args.scenario is not None else list(range(len(db)))

    # Visualizer (reused across scenarios)
    visualizer = ScenarioNetVisualizer(data_dir=str(data_dir))

    results = []

    try:
        for idx in indices:
            scenario_path = db.get_scenario_path(idx)
            intervention_name = load_intervention_name(scenario_path)
            target_type, target_value = parse_target(args.target, intervention_name)

            logger.info(f"[{idx}] target={target_type}:{target_value} (intervention='{intervention_name}')")

            saved_images, trajectory, scenario_id = visualizer.render_scenario(
                scenario_index=idx,
                num_frames=args.num_frames,
                output_dir=str(output_dir / f"frames_{idx}"),
            )

            images = [TimestampedImage(path=p, timestamp=t) for p, t in saved_images]
            features = extractor.extract(images, scenario_id=scenario_id, trajectory=trajectory)

            reward, details = score_features(features, target_type, target_value)

            result = {
                "scenario_index": idx,
                "scenario_id": scenario_id,
                "scenario_path": str(scenario_path),
                "intervention_name": intervention_name,
                "target_type": target_type,
                "target_value": target_value,
                "reward": reward,
                "details": details,
                "n_maneuvers": len(features.maneuver_sequence),
                "n_decisions": len(features.critical_decisions),
            }
            results.append(result)

            with open(output_dir / "vlm_judge_results.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")

            logger.info(f"  -> reward={reward:.2f}")

    finally:
        visualizer.close()

    # Summary
    summary = {
        "n_scenarios": len(results),
        "mean_reward": sum(r["reward"] for r in results) / max(1, len(results)),
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
