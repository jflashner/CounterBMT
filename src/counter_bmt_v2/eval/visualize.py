"""Trajectory visualization helpers for head-to-head eval."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


def select_scenarios_by_spread(
    per_scenario_rows: Sequence[Mapping[str, Any]],
    *,
    metric_key: str = "approx/sfde_min",
    max_scenarios: int = 8,
) -> List[str]:
    by_sid: Dict[str, List[float]] = {}
    for r in per_scenario_rows:
        sid = str(r["scenario_id"])
        val = float(r.get(metric_key, float("nan")))
        if not np.isfinite(val):
            continue
        by_sid.setdefault(sid, []).append(val)
    scored: List[Tuple[str, float]] = []
    for sid, vals in by_sid.items():
        if len(vals) < 2:
            continue
        arr = np.asarray(vals, dtype=np.float32)
        scored.append((sid, float(np.max(arr) - np.min(arr))))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scored[: max(0, int(max_scenarios))]]


def _best_mode_idx(pred_pos_ktn2: np.ndarray, gt_pos_tn2: np.ndarray, gt_valid_tn: np.ndarray) -> int:
    k_modes = int(pred_pos_ktn2.shape[0])
    if k_modes <= 1:
        return 0
    last_idx = gt_valid_tn.shape[0] - 1 - np.argmax(gt_valid_tn[::-1, :], axis=0)
    has_valid = np.any(gt_valid_tn, axis=0)
    valid_agents = np.where(has_valid)[0]
    if valid_agents.size == 0:
        return 0
    fde_vals: List[float] = []
    for k in range(k_modes):
        err = []
        for n in valid_agents.tolist():
            li = int(last_idx[n])
            err.append(float(np.linalg.norm(pred_pos_ktn2[k, li, n] - gt_pos_tn2[li, n])))
        fde_vals.append(float(np.mean(err)) if err else float("inf"))
    return int(np.argmin(np.asarray(fde_vals, dtype=np.float32)))


def save_overlay_plots(
    *,
    selected_scenarios: Sequence[str],
    artifact_index: Mapping[str, Mapping[str, Path]],
    out_dir: Path,
    max_agents: int = 10,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    model_ids = sorted(artifact_index.keys())
    colors = plt.get_cmap("tab10")
    saved: List[str] = []

    for sid in selected_scenarios:
        loaded: Dict[str, Dict[str, np.ndarray]] = {}
        for i, mid in enumerate(model_ids):
            p = artifact_index.get(mid, {}).get(str(sid))
            if p is None:
                continue
            with np.load(p, allow_pickle=True) as d:
                loaded[mid] = {
                    "pred_pos_ktn2": np.asarray(d["pred_pos_ktn2"], dtype=np.float32),
                    "gt_pos_tn2": np.asarray(d["gt_pos_tn2"], dtype=np.float32),
                    "gt_valid_tn": np.asarray(d["gt_valid_tn"], dtype=bool),
                }
        if not loaded:
            continue

        ref = next(iter(loaded.values()))
        gt_pos = ref["gt_pos_tn2"]
        gt_valid = ref["gt_valid_tn"]
        n_agents = int(gt_pos.shape[1])

        valid_counts = np.sum(gt_valid, axis=0)
        ranked = np.argsort(-valid_counts)
        picked: List[int] = []
        if n_agents > 0:
            picked.append(0)
        for idx in ranked.tolist():
            if idx in picked:
                continue
            picked.append(int(idx))
            if len(picked) >= max(1, int(max_agents)):
                break

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        for n in picked:
            mask = gt_valid[:, n]
            if np.any(mask):
                xy = gt_pos[mask, n, :]
                label = "GT (SDC)" if n == 0 else None
                ax.plot(xy[:, 0], xy[:, 1], color="black", linewidth=2.0, alpha=0.9, label=label)

        for i, mid in enumerate(model_ids):
            if mid not in loaded:
                continue
            pred = loaded[mid]["pred_pos_ktn2"]
            bm = _best_mode_idx(pred, gt_pos, gt_valid)
            color = colors(i % 10)
            for n in picked:
                xy = pred[bm, :, n, :]
                label = f"{mid} (mode={bm})" if n == 0 else None
                ax.plot(
                    xy[:, 0],
                    xy[:, 1],
                    linestyle="--",
                    color=color,
                    linewidth=1.4,
                    alpha=0.85,
                    label=label,
                )
                ax.scatter(xy[-1, 0], xy[-1, 1], color=color, s=10, alpha=0.8)

        ax.set_title(f"Head-to-Head Trajectory Overlay | {sid}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.25)
        ax.axis("equal")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

        out_path = out_dir / f"overlay_{sid}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        saved.append(str(out_path))

    return saved
