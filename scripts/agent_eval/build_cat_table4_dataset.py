from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    modern_src = repo_root / "src"
    legacy_src = repo_root / "src" / "Adv-BMT"
    vendored_scenarionet = repo_root / "scenarionet"
    vendored_metadrive = repo_root / "metadrive"
    for path in (vendored_scenarionet, vendored_metadrive, modern_src, repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from counter_bmt.scenario_export import create_dataset_summary, normalize_scenario_for_metadrive  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a source-matched CAT scenario bank from an existing natural ScenarioNet root. "
            "This is intended for Table-4-style baseline rows on the same index as CounterDrive."
        )
    )
    parser.add_argument("--source-natural-dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--cat-repo", type=str, required=True)
    parser.add_argument(
        "--candidate-manifest",
        type=str,
        default="",
        help="Optional dataset_manifest.json with selected_intervention/candidate_adversaries for source-matched opponent ids.",
    )
    parser.add_argument("--num-scenes", type=int, default=0)
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--ov-traj-num", type=int, default=32)
    parser.add_argument("--av-traj-num", type=int, default=1)
    parser.add_argument("--track-length", type=int, default=91)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-natural", action="store_true", help="Also copy normalized natural scenarios into outdir.")
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_default(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_default(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_default(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as fp:
        return pickle.load(fp)


def _save_pickle(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        pickle.dump(dict(payload), fp)
    return path


def _load_summary(source_dir: Path) -> Dict[str, Dict[str, Any]]:
    summary_path = source_dir / "dataset_summary.pkl"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing dataset_summary.pkl under {source_dir}")
    summary = _load_pickle(summary_path)
    if not isinstance(summary, dict):
        raise TypeError(f"{summary_path} must contain a dict, got {type(summary)}")
    return {str(k): dict(v) if isinstance(v, dict) else {} for k, v in summary.items()}


def _natural_files(source_dir: Path) -> List[str]:
    summary = _load_summary(source_dir)
    filenames = [name for name in sorted(summary) if name.startswith("sd_") and name.endswith(".pkl")]
    return [name for name in filenames if (source_dir / name).is_file()]


def _load_candidate_manifest(path: Path | None) -> Dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, str] = {}
    for item in payload.get("scenes", []):
        if not isinstance(item, dict):
            continue
        scene_id = str(item.get("scenario_id") or "").strip()
        if not scene_id:
            continue
        selected = item.get("selected_intervention")
        if isinstance(selected, dict):
            agent_id = str(selected.get("agent_id") or "").strip()
            if agent_id:
                out[scene_id] = agent_id
                continue
        candidates = item.get("candidate_adversaries")
        if isinstance(candidates, list) and candidates:
            first = candidates[0]
            if isinstance(first, dict):
                agent_id = str(first.get("agent_id") or "").strip()
                if agent_id:
                    out[scene_id] = agent_id
    return out


def _slice(items: Sequence[str], *, offset: int, num_items: int) -> List[str]:
    selected = list(items[int(max(0, offset)) :])
    if int(num_items) > 0:
        selected = selected[: int(num_items)]
    return selected


def _scenario_id(scenario: Mapping[str, Any], *, fallback: str) -> str:
    metadata = dict(scenario.get("metadata", {}))
    for value in (scenario.get("id"), metadata.get("source_scenario"), metadata.get("scenario_id"), fallback):
        text = str(value or "").strip()
        if text:
            return text.replace(".pkl", "")
    return fallback.replace(".pkl", "")


def _pick_fallback_adversary(scenario: Mapping[str, Any], *, sdc_id: str) -> str:
    tracks = dict(scenario.get("tracks", {}))
    for track_id, track in tracks.items():
        track_id_text = str(track_id)
        if track_id_text == str(sdc_id):
            continue
        if not isinstance(track, dict):
            continue
        if str(track.get("type") or "").upper() != "VEHICLE":
            continue
        valid = np.asarray(track.get("state", {}).get("valid", []), dtype=bool).reshape(-1)
        if valid.size and bool(np.any(valid)):
            return track_id_text
    for track_id in tracks:
        track_id_text = str(track_id)
        if track_id_text != str(sdc_id):
            return track_id_text
    raise RuntimeError(f"Could not find a non-SDC adversary candidate for scenario {scenario.get('id')}")


def _track_to_predict_entry(*, scenario: Mapping[str, Any], track_id: str) -> Dict[str, Any]:
    tracks = dict(scenario.get("tracks", {}))
    track_keys = [str(key) for key in tracks]
    try:
        track_index = track_keys.index(str(track_id))
    except ValueError:
        track_index = 0
    track = tracks.get(str(track_id), {})
    if not isinstance(track, dict):
        track = {}
    return {
        "track_index": int(track_index),
        "track_id": str(track_id),
        "difficulty": 0,
        "object_type": str(track.get("type") or "VEHICLE"),
    }


def _stage_cat_source_scenario(
    *,
    scenario: Dict[str, Any],
    source_path: Path,
    staged_path: Path,
    scene_id: str,
    adversary_by_scene: Mapping[str, str],
) -> tuple[Path, str]:
    staged = copy.deepcopy(scenario)
    metadata = staged.setdefault("metadata", {})
    sdc_id = str(metadata.get("sdc_id") or "").strip()
    if not sdc_id:
        raise RuntimeError(f"Scenario {scene_id} is missing metadata['sdc_id']")
    adv_id = str(adversary_by_scene.get(scene_id) or "").strip()
    if not adv_id or adv_id not in {str(key) for key in staged.get("tracks", {})}:
        adv_id = _pick_fallback_adversary(staged, sdc_id=sdc_id)

    metadata["objects_of_interest"] = [str(adv_id), str(sdc_id)]
    tracks_to_predict = dict(metadata.get("tracks_to_predict", {}))
    tracks_to_predict[str(adv_id)] = _track_to_predict_entry(scenario=staged, track_id=str(adv_id))
    tracks_to_predict[str(sdc_id)] = _track_to_predict_entry(scenario=staged, track_id=str(sdc_id))
    metadata["tracks_to_predict"] = tracks_to_predict
    metadata["source_scenario"] = str(scene_id)
    metadata["source_file"] = str(source_path)
    _save_pickle(staged_path, staged)
    return staged_path, str(adv_id)


def _step_env(env: Any, action: np.ndarray) -> tuple[Any, float, bool, Dict[str, Any]]:
    result = env.step(action)
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward), bool(terminated or truncated), dict(info)
    if isinstance(result, tuple) and len(result) == 4:
        obs, reward, done, info = result
        return obs, float(reward), bool(done), dict(info)
    raise TypeError(f"Unexpected env.step return: {type(result)} length={len(result) if isinstance(result, tuple) else 'n/a'}")


def _reset_env(env: Any, *, force_seed: int) -> Any:
    result = env.reset(force_seed=force_seed)
    if isinstance(result, tuple) and len(result) == 2:
        return result[0]
    return result


def _copy_state_array(value: Any, *, track_length: int, trailing_dim: int | None = None) -> np.ndarray:
    arr = np.asarray(value)
    if trailing_dim is not None:
        arr = arr.reshape(-1, trailing_dim)
    if arr.shape[0] >= track_length:
        return np.array(arr[:track_length], copy=True)
    if arr.shape[0] == 0:
        shape = (track_length, *arr.shape[1:])
        return np.zeros(shape, dtype=arr.dtype if arr.dtype != object else np.float32)
    pad = np.repeat(arr[-1:], track_length - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def _overwrite_cat_adversary(
    *,
    scenario: Dict[str, Any],
    adv_agent: Any,
    adv_traj: np.ndarray,
    source_scene_id: str,
    track_length: int,
) -> Dict[str, Any]:
    out = copy.deepcopy(scenario)
    tracks = out.setdefault("tracks", {})
    adv_key = str(adv_agent)
    if adv_key not in tracks and adv_agent in tracks:
        adv_key = adv_agent
    if adv_key not in tracks:
        raise KeyError(f"CAT adversary {adv_agent!r} is not present in scenario tracks")

    traj = np.asarray(adv_traj, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[1] < 5:
        raise ValueError(f"Expected CAT adv_traj shape [T, >=5], got {traj.shape}")
    traj = _copy_state_array(traj[:, :5], track_length=track_length, trailing_dim=5)

    state = tracks[adv_key].setdefault("state", {})
    old_pos = _copy_state_array(state.get("position", np.zeros((track_length, 3), dtype=np.float32)), track_length=track_length)
    if old_pos.ndim == 1:
        old_pos = old_pos.reshape(track_length, -1)
    if old_pos.shape[1] < 2:
        old_pos = np.zeros((track_length, 3), dtype=np.float32)
    new_pos = np.array(old_pos, copy=True)
    new_pos[:, :2] = traj[:, :2]
    if new_pos.shape[1] >= 3:
        new_pos[:, 2] = 0.0

    state["position"] = new_pos.astype(np.float32)
    state["velocity"] = traj[:, 2:4].astype(np.float32)
    state["heading"] = traj[:, 4].astype(np.float32)

    old_valid = np.asarray(state.get("valid", np.ones((track_length,), dtype=bool)), dtype=bool).reshape(-1)
    if old_valid.shape[0] < track_length:
        old_valid = np.pad(old_valid, (0, track_length - old_valid.shape[0]), constant_values=True)
    old_valid = old_valid[:track_length]
    history = min(11, track_length)
    future_valid = np.ones((track_length - history,), dtype=bool)
    state["valid"] = np.concatenate([old_valid[:history], future_valid], axis=0)

    out["id"] = f"{source_scene_id}_CAT"
    out["length"] = int(track_length)
    metadata = out.setdefault("metadata", {})
    metadata["id"] = out["id"]
    metadata["scenario_id"] = out["id"]
    metadata["source_scenario"] = str(source_scene_id)
    metadata["counterfactual"] = True
    metadata["intervention"] = "CAT"
    metadata["dataset"] = "waymo_CAT_source_matched"
    metadata["selected_adv_id"] = str(adv_key)
    metadata["generator_name"] = "CAT"
    metadata["track_length"] = int(track_length)
    return out


def _configure_cat_imports(cat_repo: Path) -> None:
    cat_repo = cat_repo.resolve()
    for path in (cat_repo, cat_repo / "metadrive"):
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


def _build_cat_generator(*, cat_repo: Path, ov_traj_num: int, av_traj_num: int) -> Any:
    import argparse as _argparse

    from advgen.adv_generator import AdvGenerator  # type: ignore

    cat_parser = _argparse.ArgumentParser(add_help=False)
    cat_parser.add_argument("--OV_traj_num", type=int, default=int(ov_traj_num))
    cat_parser.add_argument("--AV_traj_num", type=int, default=int(av_traj_num))
    old_argv = sys.argv
    old_cwd = Path.cwd()
    try:
        os.chdir(cat_repo)
        sys.argv = [old_argv[0]]
        return AdvGenerator(cat_parser)
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


def _run_cat_episode_logging(env: Any, adv_generator: Any) -> None:
    done = False
    while not done:
        adv_generator.log_AV_history()
        _, _, done, _ = _step_env(env, np.asarray([1.0, 0.0], dtype=np.float32))
    adv_generator.after_episode()


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_natural_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    cat_repo = Path(args.cat_repo).expanduser().resolve()
    candidate_manifest = Path(args.candidate_manifest).expanduser().resolve() if args.candidate_manifest else None
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing source natural dir: {source_dir}")
    if not cat_repo.is_dir():
        raise FileNotFoundError(f"Missing CAT repo: {cat_repo}")
    if candidate_manifest is not None and not candidate_manifest.is_file():
        raise FileNotFoundError(f"Missing candidate manifest: {candidate_manifest}")
    if outdir.exists() and any(outdir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{outdir} already exists and is not empty. Pass --overwrite.")
    outdir.mkdir(parents=True, exist_ok=True)

    _configure_cat_imports(cat_repo)

    from metadrive.envs.real_data_envs.waymo_env import WaymoEnv  # type: ignore
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy  # type: ignore

    filenames = _slice(_natural_files(source_dir), offset=args.scene_offset, num_items=args.num_scenes)
    if not filenames:
        raise RuntimeError(f"No source scenarios selected from {source_dir}")
    adversary_by_scene = _load_candidate_manifest(candidate_manifest)

    natural_dir = outdir / "natural_scenarios"
    adversarial_dir = outdir / "adversarial_scenarios"
    staged_source_dir = outdir / "_cat_source_scenarios"
    analysis_dir = outdir / "scene_analysis"
    if args.save_natural:
        natural_dir.mkdir(parents=True, exist_ok=True)
    adversarial_dir.mkdir(parents=True, exist_ok=True)
    staged_source_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    staged_records: List[Dict[str, Any]] = []
    staged_paths: List[Path] = []
    for filename in filenames:
        source_path = source_dir / filename
        source_scenario = normalize_scenario_for_metadrive(_load_pickle(source_path), original_file_path=source_path)
        source_scene_id = _scenario_id(source_scenario, fallback=filename)
        staged_path, staged_adv_id = _stage_cat_source_scenario(
            scenario=source_scenario,
            source_path=source_path,
            staged_path=staged_source_dir / filename,
            scene_id=source_scene_id,
            adversary_by_scene=adversary_by_scene,
        )
        staged_paths.append(staged_path)
        staged_records.append(
            {
                "filename": filename,
                "source_path": str(source_path),
                "staged_path": str(staged_path),
                "scenario_id": source_scene_id,
                "staged_adv_id": staged_adv_id,
            }
        )
    create_dataset_summary(staged_paths, staged_source_dir)

    adv_generator = _build_cat_generator(
        cat_repo=cat_repo,
        ov_traj_num=int(args.ov_traj_num),
        av_traj_num=int(args.av_traj_num),
    )

    env = WaymoEnv(
        {
            "agent_policy": ReplayEgoCarPolicy,
            "reactive_traffic": False,
            "use_render": False,
            "data_directory": str(staged_source_dir),
            "num_scenarios": len(staged_paths),
            "force_reuse_object_name": True,
            "sequential_seed": True,
            "vehicle_config": dict(show_navi_mark=False, show_dest_mark=False),
        }
    )

    natural_paths: List[Path] = []
    adversarial_paths: List[Path] = []
    scene_results: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    try:
        for local_index, record in enumerate(staged_records):
            filename = str(record["filename"])
            force_seed = local_index
            scene_dir = analysis_dir / filename.replace(".pkl", "")
            scene_dir.mkdir(parents=True, exist_ok=True)
            try:
                source_path = Path(str(record["source_path"]))
                source_scenario = normalize_scenario_for_metadrive(_load_pickle(source_path), original_file_path=source_path)
                source_scene_id = str(record["scenario_id"])

                if args.save_natural:
                    natural = copy.deepcopy(source_scenario)
                    natural.setdefault("metadata", {})["counterfactual"] = False
                    natural["metadata"]["intervention"] = "ground_truth"
                    natural_path = _save_pickle(natural_dir / f"sd_{source_scene_id}_ground_truth.pkl", natural)
                    natural_paths.append(natural_path)

                _reset_env(env, force_seed=force_seed)
                adv_generator.before_episode(env)
                _run_cat_episode_logging(env, adv_generator)

                _reset_env(env, force_seed=force_seed)
                adv_generator.before_episode(env)
                adv_generator.generate(mode="train")

                adv_agent = adv_generator.adv_agent
                adv_traj = getattr(adv_generator, "adv_traj", None)
                if adv_agent is None or adv_traj is None or len(adv_traj) == 0:
                    raise RuntimeError(f"CAT did not produce adversary trajectory for {source_scene_id}")

                cat_scenario = _overwrite_cat_adversary(
                    scenario=source_scenario,
                    adv_agent=adv_agent,
                    adv_traj=np.asarray(adv_traj, dtype=np.float32),
                    source_scene_id=source_scene_id,
                    track_length=int(args.track_length),
                )
                cat_path = _save_pickle(
                    adversarial_dir / f"sd_adv_reconstructed_v0_{source_scene_id}_CAT.pkl",
                    cat_scenario,
                )
                adversarial_paths.append(cat_path)

                scene_summary = {
                    "source_filename": filename,
                    "source_scenario_pkl": str(source_path),
                    "scenario_id": source_scene_id,
                    "force_seed": force_seed,
                    "cat_scenario_pkl": str(cat_path),
                    "selected_adv_id": str(adv_agent),
                    "staged_adv_id": str(record["staged_adv_id"]),
                    "adv_traj_shape": list(np.asarray(adv_traj).shape),
                }
                scene_results.append(scene_summary)
                _write_json(scene_dir / "scene_summary.json", scene_summary)
            except Exception as exc:
                failure = {
                    "source_filename": filename,
                    "force_seed": force_seed,
                    "reason": "exception",
                    "error": repr(exc),
                }
                skipped.append(failure)
                _write_json(scene_dir / "scene_skip.json", failure)
    finally:
        env.close()

    natural_summary_path = create_dataset_summary(natural_paths, natural_dir) if natural_paths else None
    adversarial_summary_path = create_dataset_summary(adversarial_paths, adversarial_dir) if adversarial_paths else None

    builder_summary = {
        "source_natural_dir": str(source_dir),
        "outdir": str(outdir),
        "cat_repo": str(cat_repo),
        "candidate_manifest": None if candidate_manifest is None else str(candidate_manifest),
        "num_requested_scenes": int(len(filenames)),
        "num_exported_pairs": int(len(scene_results)),
        "num_skipped_scenes": int(len(skipped)),
        "natural_dir": None if not args.save_natural else str(natural_dir),
        "adversarial_dir": str(adversarial_dir),
        "staged_source_dir": str(staged_source_dir),
        "natural_dataset_summary_pkl": None if natural_summary_path is None else str(natural_summary_path),
        "adversarial_dataset_summary_pkl": None if adversarial_summary_path is None else str(adversarial_summary_path),
        "ov_traj_num": int(args.ov_traj_num),
        "av_traj_num": int(args.av_traj_num),
        "track_length": int(args.track_length),
    }
    _write_json(outdir / "builder_summary.json", builder_summary)
    _write_json(outdir / "dataset_manifest.json", {"scenes": scene_results, "skipped_scenes": skipped})
    print(json.dumps(_json_default(builder_summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
