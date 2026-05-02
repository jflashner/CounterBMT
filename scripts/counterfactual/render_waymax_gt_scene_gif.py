#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]

if __package__ is None or __package__ == "":
    legacy_src = REPO_ROOT / "src" / "Adv-BMT"
    vendored_scenarionet = REPO_ROOT / "scenarionet"
    vendored_metadrive = REPO_ROOT / "metadrive"
    for path in (vendored_scenarionet, vendored_metadrive, REPO_ROOT, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual.normalize import load_raw_scenario, normalize_scenario
from scripts.counterfactual.path_semantics_plot_utils import (
    AGENT_COLOR,
    CONTEXT_SELECTION_RADIUS_M,
    CROSSWALK_FACE,
    LANE_COLOR,
    PAST_STEPS,
    PLOT_RADIUS_M,
    ROAD_COLOR,
    SDC_VERTICAL_FRACTION,
    _finite_xy_rows,
    _select_map_context,
    _select_nearby_agents,
    _select_traffic_lights,
    _world_to_sdc_up_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a ground-truth Waymax scene GIF with map context, nearby agents, "
            "and per-frame SDC GT speed."
        )
    )
    parser.add_argument("scene", nargs="?", default="", help="Scene id like 71 or waymax_scene_00071")
    parser.add_argument("--scenario-pkl", "-p", type=str, default="")
    parser.add_argument("--scene-id", "-s", type=str, default="")
    parser.add_argument(
        "--scenario-root",
        type=str,
        default=str((REPO_ROOT / "outputs" / "pr10_1_sdc_semantic_top859_full" / "scenario_root").resolve()),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str((REPO_ROOT / "outputs" / "debug_waymax_gt_scene_gifs").resolve()),
    )
    parser.add_argument("--current-time-index", type=int, default=-1)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=120)
    parser.set_defaults(play=True)
    parser.add_argument("--play", dest="play", action="store_true", help="Open the GIF after rendering (default)")
    parser.add_argument("--no-play", dest="play", action="store_false", help="Do not open the GIF after rendering")
    parser.add_argument("--print-speeds", action="store_true")
    return parser.parse_args()


def _sanitize_token(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text).strip()) or "item"


def _resolve_scenario_pkl(*, scenario_pkl: str, scene_id: str, scenario_root: str) -> Path:
    if str(scenario_pkl).strip():
        path = Path(str(scenario_pkl)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Scenario pickle not found: {path}")
        return path

    scene_token = str(scene_id).strip()
    if not scene_token:
        raise ValueError("Provide either --scenario-pkl or --scene-id")
    if scene_token.isdigit():
        scene_token = f"waymax_scene_{int(scene_token):05d}"
    elif "waymax_scene_" not in scene_token:
        scene_token = f"waymax_scene_{scene_token}"

    root = Path(str(scenario_root)).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Scenario root not found: {root}")

    candidates = sorted(root.rglob("*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No .pkl scenarios found under: {root}")

    exact: List[Path] = []
    suffix: List[Path] = []
    contains: List[Path] = []
    for path in candidates:
        stem = path.stem
        if stem == scene_token:
            exact.append(path)
        elif stem.endswith(scene_token):
            suffix.append(path)
        elif scene_token in stem:
            contains.append(path)

    matches = exact or suffix or contains
    if not matches:
        raise FileNotFoundError(f"Could not find a scenario pickle matching scene id '{scene_token}' under {root}")
    return matches[0].resolve()


def _backtrack_valid_index(valid: np.ndarray, idx: int) -> int:
    if valid.size == 0:
        return 0
    clamped = int(np.clip(int(idx), 0, max(0, valid.shape[0] - 1)))
    while clamped > 0 and not bool(valid[clamped]):
        clamped -= 1
    return clamped


def _build_render_context(raw_scenario: Mapping[str, Any], *, sdc_id: str, current_time_index: int) -> Dict[str, Any]:
    track_state = dict(raw_scenario["tracks"][str(sdc_id)]["state"])
    position = np.asarray(track_state.get("position", []), dtype=np.float64)
    heading = np.asarray(track_state.get("heading", []), dtype=np.float64).reshape(-1)
    valid = np.asarray(track_state.get("valid", []), dtype=bool).reshape(-1)

    idx = _backtrack_valid_index(valid, int(current_time_index))
    current_xy = _finite_xy_rows(position[idx])[0]
    current_heading = float(heading[idx]) if heading.shape[0] > idx and np.isfinite(heading[idx]) else 0.0
    start_idx = max(0, idx - int(PAST_STEPS))
    gt_past_xy = _finite_xy_rows(position[start_idx : idx + 1][valid[start_idx : idx + 1]])

    map_context = _select_map_context(raw_scenario, center_xy=current_xy, radius_m=CONTEXT_SELECTION_RADIUS_M)
    traffic_lights = _select_traffic_lights(
        raw_scenario,
        center_xy=current_xy,
        radius_m=CONTEXT_SELECTION_RADIUS_M,
        time_index=idx,
    )
    nearby_agents = _select_nearby_agents(
        raw_scenario,
        sdc_id=str(sdc_id),
        center_xy=current_xy,
        current_idx=idx,
        radius_m=CONTEXT_SELECTION_RADIUS_M,
    )
    return {
        "current_time_index": int(idx),
        "current_xy": np.asarray(current_xy, dtype=np.float64),
        "current_heading": float(current_heading),
        "gt_past_xy": np.asarray(gt_past_xy, dtype=np.float64),
        "map_context": map_context,
        "traffic_lights": traffic_lights,
        "nearby_agents": nearby_agents,
    }


def _draw_local_scene_background(ax, render_context: Dict[str, Any]) -> None:
    center_xy = np.asarray(render_context["current_xy"], dtype=np.float64)
    current_heading = float(render_context["current_heading"])
    map_context = render_context["map_context"]
    traffic_lights = render_context["traffic_lights"]

    ax.set_facecolor("#f8fafc")
    for feature in map_context.get("crosswalks", []):
        xy = _world_to_sdc_up_frame(
            np.asarray(feature["xy_world"], dtype=np.float64),
            center_xy=center_xy,
            heading_rad=current_heading,
        )
        if xy.shape[0] >= 3:
            ax.fill(xy[:, 0], xy[:, 1], color=CROSSWALK_FACE, alpha=0.35, zorder=1)
    for feature in map_context.get("road_boundaries", []):
        xy = _world_to_sdc_up_frame(
            np.asarray(feature["xy_world"], dtype=np.float64),
            center_xy=center_xy,
            heading_rad=current_heading,
        )
        if xy.shape[0] >= 2:
            ax.plot(xy[:, 0], xy[:, 1], color=ROAD_COLOR, linewidth=2.8, alpha=0.98, zorder=2)
    for feature in map_context.get("lane_centerlines", []):
        xy = _world_to_sdc_up_frame(
            np.asarray(feature["xy_world"], dtype=np.float64),
            center_xy=center_xy,
            heading_rad=current_heading,
        )
        if xy.shape[0] >= 2:
            ax.plot(xy[:, 0], xy[:, 1], color=LANE_COLOR, linewidth=1.1, alpha=0.30, zorder=3)
    for light in traffic_lights:
        stop_xy = _world_to_sdc_up_frame(
            np.asarray([light["stop_point_xy_world"]], dtype=np.float64),
            center_xy=center_xy,
            heading_rad=current_heading,
        )
        if stop_xy.shape[0] == 0:
            continue
        state = str(light.get("state") or "unknown")
        color = "#ef4444" if ("STOP" in state or "RED" in state) else ("#22c55e" if ("GO" in state or "GREEN" in state) else "#eab308")
        ax.scatter([stop_xy[0, 0]], [stop_xy[0, 1]], c=color, marker="s", s=70, edgecolors="black", linewidths=0.8, zorder=4)

    half_extent = float(PLOT_RADIUS_M)
    vertical_span = 2.0 * half_extent
    y_min = -float(SDC_VERTICAL_FRACTION) * vertical_span
    y_max = y_min + vertical_span
    ax.set_xlim(-half_extent, half_extent)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def _figure_to_image(fig) -> Image.Image:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return Image.fromarray(np.ascontiguousarray(rgba[..., :3]))


def _choose_visible_agents(
    positions_world: np.ndarray,
    valid_mask: np.ndarray,
    *,
    center_xy: np.ndarray,
    radius_m: float,
    sdc_index: int,
) -> np.ndarray:
    num_agents = int(positions_world.shape[1])
    keep: List[int] = [int(sdc_index)]
    for agent_idx in range(num_agents):
        if agent_idx == int(sdc_index):
            continue
        mask = np.asarray(valid_mask[:, agent_idx], dtype=bool)
        if not np.any(mask):
            continue
        agent_xy = np.asarray(positions_world[:, agent_idx, :], dtype=np.float32)[mask]
        if agent_xy.shape[0] == 0:
            continue
        if float(np.min(np.linalg.norm(agent_xy - center_xy.reshape(1, 2), axis=-1))) <= float(radius_m):
            keep.append(int(agent_idx))
    return np.asarray(sorted(set(keep)), dtype=np.int64)


def _extract_agent_arrays(raw_scenario: Mapping[str, Any], canonical) -> Dict[str, Any]:
    track_ids = sorted(canonical.tracks.keys(), key=str)
    track_index = {track_id: idx for idx, track_id in enumerate(track_ids)}
    time_steps = int(canonical.length)
    num_agents = int(len(track_ids))
    positions_world = np.zeros((time_steps, num_agents, 2), dtype=np.float32)
    headings = np.zeros((time_steps, num_agents), dtype=np.float32)
    velocities = np.zeros((time_steps, num_agents, 2), dtype=np.float32)
    valid_mask = np.zeros((time_steps, num_agents), dtype=bool)

    for track_id, agent_idx in track_index.items():
        track = canonical.tracks[track_id]
        positions_world[:, agent_idx, :] = np.asarray(track.position_xy, dtype=np.float32)
        headings[:, agent_idx] = np.asarray(track.heading, dtype=np.float32)
        velocities[:, agent_idx, :] = np.asarray(track.velocity_xy, dtype=np.float32)
        valid_mask[:, agent_idx] = np.asarray(track.valid, dtype=bool)

    sdc_index = int(track_index[str(canonical.sdc_id)])
    return {
        "track_ids": track_ids,
        "track_index": track_index,
        "positions_world": positions_world,
        "headings": headings,
        "velocities": velocities,
        "valid_mask": valid_mask,
        "sdc_index": sdc_index,
    }


def _compute_sdc_speed_mps(*, positions_world: np.ndarray, velocities: np.ndarray, valid_mask: np.ndarray, sdc_index: int, dt_s: float) -> np.ndarray:
    direct_speed = np.linalg.norm(np.asarray(velocities[:, sdc_index, :], dtype=np.float32), axis=-1).astype(np.float32)
    finite_speed = np.isfinite(direct_speed)
    if np.any(valid_mask[:, sdc_index] & finite_speed):
        speed = direct_speed
    else:
        speed = np.zeros((positions_world.shape[0],), dtype=np.float32)

    if positions_world.shape[0] > 1:
        delta = np.linalg.norm(
            np.asarray(positions_world[1:, sdc_index, :], dtype=np.float32)
            - np.asarray(positions_world[:-1, sdc_index, :], dtype=np.float32),
            axis=-1,
        ) / max(float(dt_s), 1e-3)
        delta = np.concatenate([delta[:1], delta], axis=0).astype(np.float32)
        fallback = (~finite_speed) | (~np.asarray(valid_mask[:, sdc_index], dtype=bool))
        speed = np.where(fallback, delta, speed).astype(np.float32)

    speed = np.where(np.asarray(valid_mask[:, sdc_index], dtype=bool), speed, np.nan).astype(np.float32)
    return speed


def _write_speed_sidecars(*, out_stem: Path, sdc_speed_mps: np.ndarray, timestamps_s: np.ndarray, valid_mask: np.ndarray, current_idx: int) -> Dict[str, str]:
    rows: List[Dict[str, Any]] = []
    for step_idx in range(int(current_idx), int(len(sdc_speed_mps))):
        rows.append(
            {
                "step_index": int(step_idx),
                "relative_step": int(step_idx - int(current_idx)),
                "timestamp_s": float(timestamps_s[step_idx]) if step_idx < len(timestamps_s) else None,
                "sdc_speed_mps": None if not np.isfinite(sdc_speed_mps[step_idx]) else float(sdc_speed_mps[step_idx]),
                "valid": bool(valid_mask[step_idx]),
            }
        )

    json_path = out_stem.with_suffix(".sdc_speed.json")
    csv_path = out_stem.with_suffix(".sdc_speed.csv")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with csv_path.open("wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step_index", "relative_step", "timestamp_s", "sdc_speed_mps", "valid"])
        writer.writeheader()
        writer.writerows(rows)
    return {"json": str(json_path), "csv": str(csv_path)}


def _render_gt_gif(
    *,
    raw_scenario: Mapping[str, Any],
    canonical,
    positions_world: np.ndarray,
    valid_mask: np.ndarray,
    sdc_index: int,
    sdc_speed_mps: np.ndarray,
    out_path: Path,
    current_time_index: int,
    fps: float,
    dpi: int,
) -> None:
    render_context = _build_render_context(raw_scenario, sdc_id=str(canonical.sdc_id), current_time_index=int(current_time_index))
    center_xy = np.asarray(render_context["current_xy"], dtype=np.float64)
    current_heading = float(render_context["current_heading"])
    visible_agents = _choose_visible_agents(
        positions_world,
        valid_mask,
        center_xy=np.asarray(center_xy, dtype=np.float32),
        radius_m=float(PLOT_RADIUS_M) * 1.15,
        sdc_index=int(sdc_index),
    )
    palette = plt.cm.tab20(np.linspace(0.0, 1.0, max(20, int(len(visible_agents)) + 1)))[:, :3]

    frames: List[Image.Image] = []
    history_start = max(0, int(current_time_index) - int(PAST_STEPS))
    total_steps = int(positions_world.shape[0] - int(current_time_index))
    for absolute_step in range(int(current_time_index), int(positions_world.shape[0])):
        fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
        _draw_local_scene_background(ax, render_context)

        for color_idx, agent_idx in enumerate(visible_agents.tolist()):
            mask = np.asarray(valid_mask[history_start : absolute_step + 1, agent_idx], dtype=bool)
            if not np.any(mask):
                continue
            history_world = np.asarray(positions_world[history_start : absolute_step + 1, agent_idx, :], dtype=np.float64)[mask]
            local_xy = _world_to_sdc_up_frame(history_world, center_xy=center_xy, heading_rad=current_heading)
            if local_xy.shape[0] == 0:
                continue

            is_sdc = int(agent_idx) == int(sdc_index)
            color = "#111827" if is_sdc else tuple(float(v) for v in palette[color_idx % len(palette)])
            if local_xy.shape[0] >= 2:
                ax.plot(
                    local_xy[:, 0],
                    local_xy[:, 1],
                    color=color,
                    linewidth=(2.8 if is_sdc else 1.4),
                    alpha=(0.96 if is_sdc else 0.78),
                    zorder=(8 if is_sdc else 6),
                )
            ax.scatter(
                [local_xy[-1, 0]],
                [local_xy[-1, 1]],
                c=[color],
                s=(52 if is_sdc else 24),
                edgecolors="white",
                linewidths=0.7,
                alpha=0.98,
                zorder=(9 if is_sdc else 7),
            )

        speed_text = "n/a"
        if np.isfinite(sdc_speed_mps[absolute_step]):
            speed_text = f"{float(sdc_speed_mps[absolute_step]):.2f} m/s"

        info_box_text = "\n".join(
            [
                str(canonical.scenario_id),
                f"GT SDC | current_idx={int(current_time_index)}",
                f"frame {int(absolute_step - int(current_time_index) + 1)}/{max(total_steps, 1)}",
                f"sdc gt speed: {speed_text}",
            ]
        )
        ax.text(
            0.02,
            0.975,
            info_box_text,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.90, "edgecolor": "#cbd5e1"},
            zorder=20,
        )
        fig.tight_layout(pad=0.05)
        frames.append(_figure_to_image(fig))
        plt.close(fig)

    if not frames:
        raise RuntimeError("No frames were produced for the GIF")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_duration_ms = int(round(1000.0 / max(float(fps), 1.0)))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
    )


def _play_file(path: Path) -> None:
    player_cmd: List[str] | None = None
    if sys.platform == "darwin":
        player_cmd = ["open", str(path)]
    elif shutil.which("xdg-open"):
        player_cmd = ["xdg-open", str(path)]
    if player_cmd is None:
        raise RuntimeError("No supported GIF opener found for this platform")
    subprocess.Popen(player_cmd)


def main() -> None:
    args = parse_args()
    scenario_pkl = _resolve_scenario_pkl(
        scenario_pkl=str(args.scenario_pkl),
        scene_id=str(args.scene_id or args.scene),
        scenario_root=str(args.scenario_root),
    )
    raw_scenario = load_raw_scenario(scenario_pkl)
    canonical = normalize_scenario(raw_scenario)
    current_time_index = int(canonical.current_time_index if int(args.current_time_index) < 0 else args.current_time_index)
    current_time_index = int(np.clip(current_time_index, 0, max(0, int(canonical.length) - 1)))

    arrays = _extract_agent_arrays(raw_scenario, canonical)
    sdc_valid_mask = np.asarray(arrays["valid_mask"][:, arrays["sdc_index"]], dtype=bool)
    current_time_index = _backtrack_valid_index(sdc_valid_mask, current_time_index)

    dt_s = 0.1
    if int(canonical.ts.shape[0]) >= 2:
        diffs = np.diff(np.asarray(canonical.ts, dtype=np.float32))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size > 0:
            dt_s = float(np.median(diffs))

    sdc_speed_mps = _compute_sdc_speed_mps(
        positions_world=np.asarray(arrays["positions_world"], dtype=np.float32),
        velocities=np.asarray(arrays["velocities"], dtype=np.float32),
        valid_mask=np.asarray(arrays["valid_mask"], dtype=bool),
        sdc_index=int(arrays["sdc_index"]),
        dt_s=float(dt_s),
    )

    outdir = Path(str(args.outdir)).expanduser().resolve() / _sanitize_token(str(canonical.scenario_id))
    outdir.mkdir(parents=True, exist_ok=True)
    gif_path = outdir / f"{_sanitize_token(str(canonical.scenario_id))}__ground_truth.gif"

    _render_gt_gif(
        raw_scenario=raw_scenario,
        canonical=canonical,
        positions_world=np.asarray(arrays["positions_world"], dtype=np.float32),
        valid_mask=np.asarray(arrays["valid_mask"], dtype=bool),
        sdc_index=int(arrays["sdc_index"]),
        sdc_speed_mps=np.asarray(sdc_speed_mps, dtype=np.float32),
        out_path=gif_path,
        current_time_index=int(current_time_index),
        fps=float(args.fps),
        dpi=int(args.dpi),
    )

    sidecars = _write_speed_sidecars(
        out_stem=gif_path,
        sdc_speed_mps=np.asarray(sdc_speed_mps, dtype=np.float32),
        timestamps_s=np.asarray(canonical.ts, dtype=np.float32),
        valid_mask=np.asarray(arrays["valid_mask"][:, arrays["sdc_index"]], dtype=bool),
        current_idx=int(current_time_index),
    )

    result = {
        "scene_id": str(canonical.scenario_id),
        "scenario_pkl": str(scenario_pkl),
        "gif": str(gif_path),
        "sdc_speed_json": sidecars["json"],
        "sdc_speed_csv": sidecars["csv"],
        "current_time_index": int(current_time_index),
        "sdc_id": str(canonical.sdc_id),
    }
    print(json.dumps(result, indent=2))

    if bool(args.print_speeds):
        for step_idx in range(int(current_time_index), int(len(sdc_speed_mps))):
            speed = sdc_speed_mps[step_idx]
            speed_text = "nan" if not np.isfinite(speed) else f"{float(speed):.3f}"
            ts = float(canonical.ts[step_idx]) if step_idx < len(canonical.ts) else float("nan")
            print(f"step={step_idx:03d}  t={ts:7.3f}s  sdc_speed_mps={speed_text}")

    if bool(args.play):
        _play_file(gif_path)


if __name__ == "__main__":
    main()
