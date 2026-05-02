from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in (repo_root, repo_root / "src", repo_root / "src" / "Adv-BMT"):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))

# Match the TD3 trainer import order. Importing MetaDrive/TensorFlow-adjacent
# modules before torch can segfault in this legacy environment.
import torch  # noqa: F401


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_default(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_default(v) for v in value]
    return value


def _mean_std(values: list[Any]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved TD3 checkpoint on a ScenarioNet eval directory.")
    parser.add_argument("--ckpt", required=True, help="TD3 .zip checkpoint, e.g. seed_0_1000000_steps.zip.")
    parser.add_argument("--eval-data-dir", required=True, help="TD3-ready ScenarioNet eval directory.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--eval-ep", type=int, default=None, help="Number of eval episodes. Defaults to scenario count.")
    parser.add_argument("--eval-horizon", type=int, default=100)
    parser.add_argument("--num-eval-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda", help="Torch device for loading the TD3 policy.")
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    from metadrive.policy.env_input_policy import EnvInputPolicy
    from metadrive.scenario.utils import get_number_of_scenarios
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from bmt.rl_train.train.customized_td3 import CustomizedTD3
    from bmt.rl_train.train.train_td3 import create_eval_env

    eval_data_dir = Path(args.eval_data_dir).expanduser().resolve()
    ckpt = Path(args.ckpt).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()

    num_scenarios = int(get_number_of_scenarios(str(eval_data_dir)))
    eval_ep = int(args.eval_ep or num_scenarios)

    config_eval = dict(
        store_map=False,
        use_render=False,
        manual_control=False,
        show_interface=False,
        data_directory=str(eval_data_dir),
        agent_policy=EnvInputPolicy,
        start_scenario_index=0,
        num_scenarios=num_scenarios,
        sequential_seed=True,
        horizon=int(args.eval_horizon),
        reactive_traffic=False,
        no_static_vehicles=True,
        no_light=True,
        crash_vehicle_done=False,
        out_of_route_done=False,
        crash_object_done=False,
        crash_human_done=False,
        relax_out_of_road_done=False,
    )

    episode_infos: list[dict[str, Any]] = []

    def log_callback(locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        if not locals_.get("done"):
            return
        info = dict(locals_.get("info", {}))
        completion = float(info.get("route_completion", 0.0) or 0.0)
        info["route_completion"] = min(1.0, max(0.0, completion))
        info["crash"] = bool(info.get("crash", False))
        episode_infos.append(info)

    eval_env = SubprocVecEnv(
        [lambda: create_eval_env(config_eval) for _ in range(int(args.num_eval_envs))],
        start_method="spawn",
    )
    try:
        model = CustomizedTD3.load(str(ckpt), env=eval_env, device=str(args.device))
        episode_rewards, episode_lengths = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=eval_ep,
            deterministic=bool(args.deterministic),
            return_episode_rewards=True,
            callback=log_callback,
        )
    finally:
        eval_env.close()

    metrics: dict[str, Any] = {
        "ckpt": str(ckpt),
        "eval_data_dir": str(eval_data_dir),
        "num_scenarios": num_scenarios,
        "eval_ep": eval_ep,
        "episode_reward_mean": float(np.mean(episode_rewards)),
        "episode_reward_std": float(np.std(episode_rewards)),
        "episode_length_mean": float(np.mean(episode_lengths)),
        "episode_length_std": float(np.std(episode_lengths)),
        "episodes": episode_infos,
    }

    for info_key, output_prefix in [
        ("cost", "cost"),
        ("route_completion", "route_completion"),
        ("crash", "crash"),
        ("arrive_dest", "arrive_dest"),
        ("out_of_road", "out_of_road"),
    ]:
        values = []
        for info in episode_infos:
            if info_key not in info:
                continue
            value = info[info_key]
            if isinstance(value, (bool, np.bool_)):
                value = float(bool(value))
            values.append(value)
        if values:
            mean, std = _mean_std(values)
            metrics[f"{output_prefix}_mean"] = mean
            metrics[f"{output_prefix}_std"] = std

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    print(json.dumps({k: v for k, v in metrics.items() if k != "episodes"}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
