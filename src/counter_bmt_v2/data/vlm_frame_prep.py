"""Prepare VLM-oriented frame packs with annotations and dual-view ordering."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from counter_bmt_v2.contracts import TimestampedFrame
from counter_bmt_v2.data.scenarionet import NNXBMTSceneSample


def _safe_dt(sample: NNXBMTSceneSample) -> float:
    dt = float(sample.dt_s)
    if not np.isfinite(dt) or dt <= 0.0:
        return 0.1
    return dt


def _timestamp_to_t_index(sample: NNXBMTSceneSample, timestamp_s: float) -> int:
    t_steps = int(sample.agent_position_xy.shape[0]) if sample.agent_position_xy.ndim >= 3 else 0
    if t_steps <= 0:
        return 0
    dt = _safe_dt(sample)
    idx = int(round(float(timestamp_s) / dt))
    return int(np.clip(idx, 0, t_steps - 1))


def _frame_span_summary(sample: NNXBMTSceneSample, raw_frames: Sequence[TimestampedFrame]) -> Dict[str, Any]:
    t_steps = int(sample.agent_position_xy.shape[0]) if sample.agent_position_xy.ndim >= 3 else 0
    dt = _safe_dt(sample)
    horizon_end_s = float(max(0.0, (t_steps - 1) * dt))
    t_indices = [_timestamp_to_t_index(sample, float(f.timestamp_s)) for f in raw_frames]
    timestamps = [float(f.timestamp_s) for f in raw_frames]
    start_idx = int(min(t_indices)) if t_indices else 0
    end_idx = int(max(t_indices)) if t_indices else 0
    start_s = float(min(timestamps)) if timestamps else 0.0
    end_s = float(max(timestamps)) if timestamps else 0.0
    coverage_ratio = float(end_s / max(horizon_end_s, 1e-6)) if horizon_end_s > 0 else 1.0
    covers_terminal_frame = bool(end_idx >= max(0, t_steps - 1))
    return {
        "scenario_t_steps": int(t_steps),
        "dt_s": float(dt),
        "horizon_end_s": float(horizon_end_s),
        "sampled_t_indices": [int(x) for x in t_indices],
        "sampled_timestamps_s": [float(x) for x in timestamps],
        "start_t_index": int(start_idx),
        "end_t_index": int(end_idx),
        "start_s": float(start_s),
        "end_s": float(end_s),
        "coverage_ratio": float(coverage_ratio),
        "covers_terminal_frame": bool(covers_terminal_frame),
    }


def _mean_abs_heading_delta(heading: np.ndarray, valid: np.ndarray) -> float:
    if heading.ndim != 1 or valid.ndim != 1 or heading.shape[0] < 2:
        return 0.0
    m = valid[1:] & valid[:-1]
    if not np.any(m):
        return 0.0
    dh = heading[1:] - heading[:-1]
    wrapped = np.arctan2(np.sin(dh), np.cos(dh))
    return float(np.mean(np.abs(wrapped[m])))


def _maneuver_proxy(speed: np.ndarray, heading: np.ndarray, valid: np.ndarray) -> str:
    if speed.ndim != 1 or valid.ndim != 1:
        return "unknown"
    v = speed[valid]
    if v.size == 0:
        return "unknown"
    stop_ratio = float(np.mean(v < 0.8))
    if stop_ratio > 0.6:
        return "stop_and_go"
    mean_turn = _mean_abs_heading_delta(heading, valid)
    if mean_turn > 0.07:
        return "turning_or_lane_change"
    if float(np.mean(v)) > 8.0:
        return "cruising_fast"
    return "steady_following"


def _build_ego_context_text(sample: NNXBMTSceneSample, raw_frames: Sequence[TimestampedFrame]) -> str:
    pos = np.asarray(sample.agent_position_xy, dtype=np.float32)
    vel = np.asarray(sample.agent_velocity_xy, dtype=np.float32)
    heading = np.asarray(sample.agent_heading, dtype=np.float32)
    valid = np.asarray(sample.agent_valid_mask, dtype=bool)

    if pos.ndim != 3 or pos.shape[1] == 0 or valid.ndim != 2 or not np.any(valid[:, 0]):
        return "Ego context unavailable: missing valid ego trajectory."

    dt = _safe_dt(sample)
    ego_valid = valid[:, 0]
    if vel.ndim == 3 and vel.shape[2] >= 2 and vel.shape[:2] == valid.shape:
        speed = np.linalg.norm(vel[:, 0, :2], axis=-1)
    else:
        speed = np.zeros((valid.shape[0],), dtype=np.float32)
        if pos.shape[0] >= 2:
            dxy = pos[1:, 0, :2] - pos[:-1, 0, :2]
            speed[1:] = np.linalg.norm(dxy, axis=-1) / max(dt, 1e-6)
            speed[0] = speed[1] if speed.shape[0] > 1 else 0.0

    ego_heading = heading[:, 0] if heading.ndim == 2 and heading.shape[:2] == valid.shape else np.zeros_like(speed)
    mean_heading_delta = _mean_abs_heading_delta(ego_heading, ego_valid)

    speed_valid = speed[ego_valid]
    speed_min = float(np.min(speed_valid)) if speed_valid.size else 0.0
    speed_mean = float(np.mean(speed_valid)) if speed_valid.size else 0.0
    speed_max = float(np.max(speed_valid)) if speed_valid.size else 0.0

    ts_list = ", ".join(f"{float(f.timestamp_s):.2f}" for f in raw_frames) if raw_frames else "none"
    span = _frame_span_summary(sample, raw_frames)
    keyframe_lines: List[str] = []
    for f in raw_frames:
        ti = _timestamp_to_t_index(sample, float(f.timestamp_s))
        if ti < valid.shape[0] and bool(valid[ti, 0]):
            sp = float(speed[ti])
            hd = float(np.degrees(ego_heading[ti]))
            keyframe_lines.append(
                f"  t={float(f.timestamp_s):.2f}s speed={sp:.2f} world_heading_deg={hd:.1f}"
            )

    lines = [
        "Known context from tensors (not inferred from images):",
        f"- View type: top-down traffic scene sequence",
        f"- Ego id: agent_0, ego valid ratio: {float(np.mean(ego_valid)):.3f}",
        f"- Ego speed m/s: min={speed_min:.2f}, mean={speed_mean:.2f}, max={speed_max:.2f}",
        f"- Ego mean absolute world-heading delta per step (rad): {mean_heading_delta:.4f}",
        (
            "- world_heading_deg values below are simulator/world-frame angles. "
            "They are auxiliary only and do not directly encode screen-left or screen-right."
        ),
        (
            "- Determine left_turn vs right_turn from the ego vehicle's own perspective in the images, "
            "not from the sign of world_heading_deg or the side of the screen reached."
        ),
        f"- Frame timestamps (s): {ts_list}",
        f"- Frame dt (s): {dt:.3f}",
        (
            f"- Frame span coverage: start={float(span['start_s']):.2f}s "
            f"end={float(span['end_s']):.2f}s horizon_end={float(span['horizon_end_s']):.2f}s "
            f"coverage_ratio={float(span['coverage_ratio']):.3f} "
            f"covers_terminal_frame={bool(span['covers_terminal_frame'])}"
        ),
        f"- Sampled frame indices: {span['sampled_t_indices']}",
    ]
    if keyframe_lines:
        lines.append("- Ego keyframe states:")
        lines.extend(keyframe_lines)
    return "\n".join(lines)


def annotate_global_frame(
    *,
    in_path: str | Path,
    out_path: str | Path,
    frame_index: int,
    frame_count: int,
    timestamp_s: float,
    annotation_style: str = "banner+legend",
    ego_color_hint: str = "green",
    ego_inset_path: str | Path | None = None,
    ego_state_text: str = "",
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("matplotlib is required for VLM frame annotations.") from exc

    img = plt.imread(str(in_path))
    h = int(img.shape[0]) if img.ndim >= 2 else 800
    w = int(img.shape[1]) if img.ndim >= 2 else 800

    fig = plt.figure(figsize=(max(4.0, w / 140.0), max(4.0, h / 140.0)), dpi=140)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(img)
    ax.axis("off")

    top_text = (
        f"Top-down traffic scene | Ego={ego_color_hint.upper()} | "
        f"Frame {int(frame_index) + 1}/{int(frame_count)} | t={float(timestamp_s):.2f}s"
    )
    ax.text(
        0.01,
        0.985,
        top_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.70, "pad": 4},
    )
    if annotation_style == "banner+legend":
        legend_text = (
            "Sequence is chronological left-to-right in index order. "
            "Use this frame's timestamp label for temporal grounding. "
            "Analyze early/mid/late sequence; do not ignore late-frame ego maneuvers. "
            "Judge left/right turns from the ego vehicle's perspective, not from the side of the image."
        )
        ax.text(
            0.01,
            0.94,
            legend_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
        )
    if ego_state_text:
        ax.text(
            0.01,
            0.875,
            ego_state_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
        )

    if ego_inset_path:
        try:
            inset_img = plt.imread(str(ego_inset_path))
            inset_ax = fig.add_axes([0.66, 0.02, 0.32, 0.32])
            inset_ax.imshow(inset_img)
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_title("EGO ZOOM (same timestamp)", fontsize=8, color="white", pad=2)
            for spine in inset_ax.spines.values():
                spine.set_edgecolor("#22C55E")
                spine.set_linewidth(1.8)
            inset_ax.patch.set_alpha(1.0)
        except Exception:
            pass

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_map_centered(ax: Any, sample: NNXBMTSceneSample, center_xy: np.ndarray, window_m: float) -> None:
    mf = np.asarray(sample.map_feature, dtype=np.float32)
    mv = np.asarray(sample.map_feature_valid_mask, dtype=bool)
    if mf.ndim != 3 or mv.ndim != 2 or mf.shape[2] < 6:
        return
    x0 = mf[:, :, 0] - float(center_xy[0])
    y0 = mf[:, :, 1] - float(center_xy[1])
    x1 = mf[:, :, 3] - float(center_xy[0])
    y1 = mf[:, :, 4] - float(center_xy[1])
    bound = float(window_m) * 1.2
    for i in range(mf.shape[0]):
        valid_i = mv[i]
        if not np.any(valid_i):
            continue
        for j in np.where(valid_i)[0].tolist():
            if (
                abs(float(x0[i, j])) > bound
                and abs(float(y0[i, j])) > bound
                and abs(float(x1[i, j])) > bound
                and abs(float(y1[i, j])) > bound
            ):
                continue
            ax.plot(
                [float(x0[i, j]), float(x1[i, j])],
                [float(y0[i, j]), float(y1[i, j])],
                color="#A3A3A3",
                linewidth=0.8,
                alpha=0.45,
            )


def _plot_traffic_lights_centered(
    ax: Any,
    sample: NNXBMTSceneSample,
    t_index: int,
    center_xy: np.ndarray,
    window_m: float,
) -> None:
    tl_pos = np.asarray(sample.traffic_light_position, dtype=np.float32)
    tl_valid = np.asarray(sample.traffic_light_valid_mask, dtype=bool)
    if tl_pos.ndim != 2 or tl_pos.shape[1] < 2:
        return
    if tl_valid.ndim != 2 or t_index >= tl_valid.shape[0]:
        return
    v = tl_valid[t_index]
    if not np.any(v):
        return
    pos = tl_pos[v, :2] - center_xy[None, :2]
    in_win = (np.abs(pos[:, 0]) <= float(window_m)) & (np.abs(pos[:, 1]) <= float(window_m))
    if np.any(in_win):
        p = pos[in_win]
        ax.scatter(
            p[:, 0],
            p[:, 1],
            s=18,
            marker="s",
            c="#F59E0B",
            alpha=0.9,
            linewidths=0.2,
            edgecolors="#111827",
        )


def render_ego_tensor_view(
    *,
    sample: NNXBMTSceneSample,
    t_index: int,
    out_path: str | Path,
    max_agents: int = 64,
    window_m: float = 55.0,
    trail_steps: int = 12,
    ego_color_hint: str = "green",
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("matplotlib is required for ego-view frame rendering.") from exc

    pos = np.asarray(sample.agent_position_xy, dtype=np.float32)
    valid = np.asarray(sample.agent_valid_mask, dtype=bool)
    heading = np.asarray(sample.agent_heading, dtype=np.float32)
    if pos.ndim != 3 or valid.ndim != 2 or pos.shape[1] == 0:
        raise ValueError("Sample is missing agent tensors required for ego view rendering.")

    t_steps = int(pos.shape[0])
    n_agents = int(pos.shape[1])
    t = int(np.clip(int(t_index), 0, max(0, t_steps - 1)))
    n_render = min(int(max_agents), n_agents)
    dt = _safe_dt(sample)
    if not bool(valid[t, 0]):
        fallback = np.where(valid[t, :n_render])[0]
        center_agent = int(fallback[0]) if fallback.size else 0
    else:
        center_agent = 0
    center = pos[t, center_agent, :2].copy()

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    _plot_map_centered(ax, sample, center_xy=center, window_m=float(window_m))
    _plot_traffic_lights_centered(ax, sample, t_index=t, center_xy=center, window_m=float(window_m))

    t0 = max(0, t - int(trail_steps) + 1)
    for a in range(n_render):
        vmask = valid[t0 : t + 1, a]
        if not np.any(vmask):
            continue
        trail = pos[t0 : t + 1, a, :2][vmask] - center[None, :]
        if trail.shape[0] >= 2:
            color = "#10B981" if a == 0 else "#60A5FA"
            alpha = 0.78 if a == 0 else 0.35
            lw = 1.8 if a == 0 else 1.0
            ax.plot(trail[:, 0], trail[:, 1], color=color, linewidth=lw, alpha=alpha)

    now_valid = valid[t, :n_render]
    if np.any(now_valid):
        cur = pos[t, :n_render, :2][now_valid] - center[None, :]
        idx = np.where(now_valid)[0]
        colors = ["#16A34A" if int(i) == 0 else "#2563EB" for i in idx.tolist()]
        sizes = [70 if int(i) == 0 else 22 for i in idx.tolist()]
        ax.scatter(cur[:, 0], cur[:, 1], c=colors, s=sizes, alpha=0.92, edgecolors="#111827", linewidths=0.2)

    if heading.ndim == 2 and heading.shape[:2] == valid.shape and bool(valid[t, 0]):
        h = float(heading[t, 0])
        arrow_len = 6.0
        dx = arrow_len * float(np.cos(h))
        dy = arrow_len * float(np.sin(h))
        ax.arrow(
            0.0,
            0.0,
            dx,
            dy,
            color="#14532D",
            linewidth=1.8,
            head_width=1.8,
            head_length=2.5,
            length_includes_head=True,
            alpha=0.95,
        )

    ax.set_xlim(-float(window_m), float(window_m))
    ax.set_ylim(-float(window_m), float(window_m))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("ego-centric x (m)")
    ax.set_ylabel("ego-centric y (m)")
    ax.set_title(
        f"Ego-focused tensor view | t={float(t) * dt:.2f}s | "
        f"Ego={ego_color_hint.upper()} | traffic lights shown as orange squares"
    )
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def build_vlm_frame_pack(
    *,
    sample: NNXBMTSceneSample,
    raw_frames: Sequence[TimestampedFrame],
    out_dir: str | Path,
    max_agents: int = 64,
    annotate_vlm_frames: bool = True,
    annotation_style: str = "banner+legend",
    ego_color_hint: str = "green",
    include_ego_context_text: bool = True,
    dual_view: bool = True,
    dual_view_mode: str = "global_plus_ego_tensor",
    add_ego_inset: bool = True,
) -> Tuple[List[TimestampedFrame], List[Dict[str, Any]], str]:
    if dual_view and dual_view_mode != "global_plus_ego_tensor":
        raise ValueError(f"Unsupported dual_view_mode: {dual_view_mode}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stable temporal order.
    ordered = sorted(raw_frames, key=lambda f: (float(f.timestamp_s), str(f.path)))
    frames_for_vlm: List[TimestampedFrame] = []
    manifest: List[Dict[str, Any]] = []
    span = _frame_span_summary(sample, ordered)

    for i, raw in enumerate(ordered):
        global_path = out_dir / f"global_t{i:03d}.png"
        t_idx = _timestamp_to_t_index(sample, float(raw.timestamp_s))
        ego_path = out_dir / f"ego_t{i:03d}.png"
        ego_render_ok = False
        if add_ego_inset or dual_view:
            try:
                render_ego_tensor_view(
                    sample=sample,
                    t_index=t_idx,
                    out_path=ego_path,
                    max_agents=max_agents,
                    ego_color_hint=ego_color_hint,
                )
                ego_render_ok = True
            except Exception:
                ego_render_ok = False

        ego_state_text = ""
        try:
            pos = np.asarray(sample.agent_position_xy, dtype=np.float32)
            vel = np.asarray(sample.agent_velocity_xy, dtype=np.float32)
            valid = np.asarray(sample.agent_valid_mask, dtype=bool)
            heading = np.asarray(sample.agent_heading, dtype=np.float32)
            if (
                pos.ndim == 3
                and vel.ndim == 3
                and valid.ndim == 2
                and heading.ndim == 2
                and t_idx < pos.shape[0]
                and pos.shape[1] > 0
                and bool(valid[t_idx, 0])
            ):
                ego_xy = pos[t_idx, 0, :2]
                ego_speed = float(np.linalg.norm(vel[t_idx, 0, :2]))
                ego_heading_deg = float(np.degrees(heading[t_idx, 0]))
                ego_state_text = (
                    f"Ego@t: x={ego_xy[0]:.1f}m y={ego_xy[1]:.1f}m "
                    f"speed={ego_speed:.1f}m/s world_heading={ego_heading_deg:.1f}deg"
                )
        except Exception:
            ego_state_text = ""

        if annotate_vlm_frames:
            try:
                annotate_global_frame(
                    in_path=raw.path,
                    out_path=global_path,
                    frame_index=i,
                    frame_count=len(ordered),
                    timestamp_s=float(raw.timestamp_s),
                    annotation_style=annotation_style,
                    ego_color_hint=ego_color_hint,
                    ego_inset_path=(ego_path if (add_ego_inset and ego_render_ok) else None),
                    ego_state_text=ego_state_text,
                )
            except Exception:
                shutil.copy2(raw.path, global_path)
        else:
            shutil.copy2(raw.path, global_path)

        sequence_index = len(frames_for_vlm)
        frames_for_vlm.append(TimestampedFrame(path=str(global_path), timestamp_s=float(raw.timestamp_s)))
        manifest.append(
            {
                "frame_id": f"global_t{i:03d}",
                "role": "global",
                "timestamp_s": float(raw.timestamp_s),
                "t_index": int(t_idx),
                "path": str(global_path),
                "sequence_index": int(sequence_index),
                "has_ego_inset": bool(add_ego_inset and ego_render_ok),
                "scenario_t_steps": int(span["scenario_t_steps"]),
                "scenario_horizon_end_s": float(span["horizon_end_s"]),
                "covers_terminal_frame": bool(span["covers_terminal_frame"]),
            }
        )

        if dual_view:
            if not ego_render_ok:
                render_ego_tensor_view(
                    sample=sample,
                    t_index=t_idx,
                    out_path=ego_path,
                    max_agents=max_agents,
                    ego_color_hint=ego_color_hint,
                )
            sequence_index = len(frames_for_vlm)
            frames_for_vlm.append(TimestampedFrame(path=str(ego_path), timestamp_s=float(raw.timestamp_s)))
            manifest.append(
                {
                    "frame_id": f"ego_t{i:03d}",
                    "role": "ego_view",
                    "timestamp_s": float(raw.timestamp_s),
                    "t_index": int(t_idx),
                    "path": str(ego_path),
                    "sequence_index": int(sequence_index),
                    "derived_from": f"global_t{i:03d}",
                    "scenario_t_steps": int(span["scenario_t_steps"]),
                    "scenario_horizon_end_s": float(span["horizon_end_s"]),
                    "covers_terminal_frame": bool(span["covers_terminal_frame"]),
                }
            )

    context_text = _build_ego_context_text(sample, ordered) if include_ego_context_text else ""
    return frames_for_vlm, manifest, context_text
