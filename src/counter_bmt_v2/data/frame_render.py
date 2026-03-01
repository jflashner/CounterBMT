"""Render lightweight scene frames from v2 ScenarioNet tensors."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from counter_bmt_v2.contracts import TimestampedFrame
from counter_bmt_v2.data.scenarionet import NNXBMTSceneSample


def _time_indices(t_steps: int, num_frames: int) -> np.ndarray:
    if t_steps <= 0:
        return np.zeros((0,), dtype=np.int32)
    k = max(1, int(num_frames))
    if t_steps == 1:
        return np.zeros((k,), dtype=np.int32)
    idx = np.linspace(0, t_steps - 1, num=k, dtype=np.int32)
    return np.clip(idx, 0, t_steps - 1)


def _plot_map(ax: any, sample: NNXBMTSceneSample) -> None:
    mf = np.asarray(sample.map_feature, dtype=np.float32)
    mv = np.asarray(sample.map_feature_valid_mask, dtype=bool)
    if mf.ndim != 3 or mv.ndim != 2:
        return
    m, v, d = mf.shape
    if d < 6:
        return
    for i in range(m):
        valid_i = mv[i]
        if not np.any(valid_i):
            continue
        seg = mf[i, valid_i]
        x0 = seg[:, 0]
        y0 = seg[:, 1]
        x1 = seg[:, 3]
        y1 = seg[:, 4]
        for j in range(seg.shape[0]):
            ax.plot([x0[j], x1[j]], [y0[j], y1[j]], color="#BBBBBB", linewidth=0.6, alpha=0.6)


def render_scenario_frames(
    sample: NNXBMTSceneSample,
    out_dir: str | Path,
    *,
    num_frames: int = 8,
    max_agents: int = 64,
) -> List[TimestampedFrame]:
    """Render map+agents snapshots and return frame metadata."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for frame rendering; install matplotlib to use render_scenario_frames."
        ) from exc

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pos = np.asarray(sample.agent_position_xy, dtype=np.float32)
    valid = np.asarray(sample.agent_valid_mask, dtype=bool)
    if pos.ndim != 3 or valid.ndim != 2:
        return []

    t_steps, n_agents = pos.shape[:2]
    n_render = min(int(max_agents), int(n_agents))
    idx_t = _time_indices(t_steps, num_frames)
    frames: List[TimestampedFrame] = []
    dt_s = float(sample.dt_s if np.isfinite(sample.dt_s) and sample.dt_s > 0 else 0.1)

    for k, t in enumerate(idx_t.tolist()):
        fig, ax = plt.subplots(figsize=(8, 8))
        _plot_map(ax, sample)

        # Plot recent trails for the first few agents to provide motion context.
        trail_len = 12
        t0 = max(0, int(t) - trail_len + 1)
        for a in range(min(n_render, 16)):
            vmask = valid[t0 : t + 1, a]
            if np.any(vmask):
                p = pos[t0 : t + 1, a]
                p = p[vmask]
                if p.shape[0] >= 2:
                    ax.plot(p[:, 0], p[:, 1], color="#6FA8DC", linewidth=1.0, alpha=0.45)

        v_now = valid[t, :n_render]
        if np.any(v_now):
            p_now = pos[t, :n_render][v_now]
            ax.scatter(p_now[:, 0], p_now[:, 1], s=10, c="#2563EB", alpha=0.75, label="agents")

        # Ego (agent 0) in red if present.
        if n_render > 0 and bool(valid[t, 0]):
            e = pos[t, 0]
            ax.scatter([e[0]], [e[1]], s=36, c="#DC2626", alpha=0.95, label="ego")

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{sample.scenario_id}  t={t} ({t * dt_s:.2f}s)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()

        frame_path = out_path / f"frame_{k:03d}.png"
        fig.savefig(frame_path, dpi=120)
        plt.close(fig)

        frames.append(TimestampedFrame(path=str(frame_path), timestamp_s=float(t * dt_s)))

    return frames

