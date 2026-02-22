"""Forward-pass validation metrics for the NNX Adv-BMT rewrite.

Paper alignment notes:
- This module implements the scenario-level forward-pass metrics used in
  Adv-BMT evaluation (`bmt/eval/scenario_evaluator.py`) using NumPy/JAX-only
  primitives so it can run directly inside CounterBMT v2 training.
- Metrics include supervised trajectory fit (minSFDE/minSADE/minSSDE),
  diversity (FDD/ADD/SDD), realism (Vel/Acc/TTC JSD), and safety/comfort
  summaries (collision and SDC accel/jerk statistics).
- Original Adv-BMT collision/TTC metrics use Waymo metric operators. Here we
  provide a dependency-light approximation that preserves metric intent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import numpy as np

import jax
import jax.numpy as jnp

from counter_bmt_v2.trajectory_jax import (
    BidirectionalMotionTokenizer,
    NNXBidirectionalMotionTransformer,
    sample_motion_tokens,
)


@dataclass
class ForwardPassEvalConfig:
    """Configuration for Adv-BMT-style forward-pass validation.

    Defaults track the histogram ranges and sampling style used by the
    original Adv-BMT evaluator where possible.
    """

    enabled: bool = True
    num_modes: int = 6

    sampling_method: str = "topp"
    temperature: float = 1.0
    topp: float = 0.95
    topk: int = 5

    vel_hist_min: float = 0.0
    vel_hist_max: float = 50.0
    vel_hist_bins: int = 100

    acc_hist_min: float = -10.0
    acc_hist_max: float = 10.0
    acc_hist_bins: int = 200

    # Adv-BMT uses WOD TTC bins in [0, 5].
    ttc_hist_min: float = 0.0
    ttc_hist_max: float = 5.0
    ttc_hist_bins: int = 10

    # Circle approximation radius floor for collision checks.
    collision_radius_floor_m: float = 0.5

    # P4 scope lock: only the core + realism metric family is tracked for parity.
    metric_scope: Literal["core_realism"] = "core_realism"

    # Export per-scenario eval artifacts for strict offline parity checks.
    export_artifacts: bool = True
    artifact_output_subdir: str = "forward_eval_artifacts"
    artifact_max_scenarios_per_eval: int = 32

    # Eval-time visualization controls.
    save_visualizations: bool = True
    viz_max_scenarios: int = 2
    viz_max_agents: int = 10
    viz_output_subdir: str = "forward_eval_viz"


CORE_REALISM_METRIC_KEYS: Tuple[str, ...] = (
    "sfde_min",
    "sfde_avg",
    "sade_min",
    "sade_avg",
    "ssde_min",
    "ssde_avg",
    "fdd",
    "add",
    "sdd",
    "vel_jsd",
    "acc_jsd",
    "ttc_jsd",
)


def _safe_mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _sanitize_name(name: str) -> str:
    safe = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "scenario"


def _save_rollout_vs_gt_plot(
    *,
    out_file: Path,
    scenario_id: str,
    gt_pos_tn2: np.ndarray,    # [T,N,2]
    gt_valid_tn: np.ndarray,   # [T,N]
    pred_pos_ktn2: np.ndarray, # [K,T,N,2]
    max_agents: int,
) -> bool:
    """Save simple XY rollout-vs-GT plot for one scenario.

    Uses best-SFDE mode among sampled trajectories.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    t_steps = gt_pos_tn2.shape[0]
    n_agents = gt_pos_tn2.shape[1]
    n_modes = pred_pos_ktn2.shape[0]
    if n_modes <= 0 or t_steps <= 0 or n_agents <= 0:
        return False

    has_valid, _, last_idx = _first_last_valid_indices(gt_valid_tn)
    valid_agents = np.where(has_valid)[0]
    if valid_agents.size == 0:
        return False

    # Pick mode with minimum SFDE for this scenario.
    fde_mode = []
    for k in range(n_modes):
        fde_vals = []
        for n in valid_agents.tolist():
            l = int(last_idx[n])
            fde_vals.append(float(np.linalg.norm(pred_pos_ktn2[k, l, n] - gt_pos_tn2[l, n])))
        fde_mode.append(_safe_mean(fde_vals))
    best_mode = int(np.nanargmin(np.asarray(fde_mode, dtype=np.float32)))

    # Keep SDC + most-observed agents.
    valid_counts = np.sum(gt_valid_tn, axis=0)
    rank = np.argsort(-valid_counts)
    picked: List[int] = []
    if n_agents > 0:
        picked.append(0)
    for idx in rank.tolist():
        if idx in picked or not has_valid[idx]:
            continue
        picked.append(int(idx))
        if len(picked) >= int(max_agents):
            break

    if not picked:
        return False

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    cmap = plt.get_cmap("tab20")

    for i, n in enumerate(picked):
        color = cmap(i % 20)
        gt_mask = gt_valid_tn[:, n]
        if np.any(gt_mask):
            gt_xy = gt_pos_tn2[gt_mask, n, :]
            label_gt = f"gt_sdc_{n}" if n == 0 else None
            ax.plot(gt_xy[:, 0], gt_xy[:, 1], color=color, linewidth=1.8, alpha=0.9, label=label_gt)
            ax.scatter(gt_xy[-1, 0], gt_xy[-1, 1], color=color, s=10, alpha=0.9)

        pred_xy = pred_pos_ktn2[best_mode, :, n, :]
        label_pred = f"pred_sdc_{n}" if n == 0 else None
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], linestyle="--", color=color, linewidth=1.4, alpha=0.85, label=label_pred)
        ax.scatter(pred_xy[-1, 0], pred_xy[-1, 1], marker="x", color=color, s=14, alpha=0.9)

    ax.set_title(f"Forward Eval Rollout vs GT | {scenario_id} | best_mode={best_mode}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25)
    ax.axis("equal")
    if any("sdc" in (h.get_label() or "") for h in ax.get_lines()):
        ax.legend(loc="best", fontsize=8)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_file, dpi=130)
    plt.close(fig)
    return True


def nanmean_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """NaN-aware metric aggregation across scenarios."""
    if not metrics_list:
        return {}

    keys = sorted({k for m in metrics_list for k in m.keys()})
    out: Dict[str, float] = {}
    for k in keys:
        vals = np.asarray([m.get(k, float("nan")) for m in metrics_list], dtype=np.float32)
        if np.all(np.isnan(vals)):
            out[k] = float("nan")
        else:
            out[k] = float(np.nanmean(vals))
    return out


def _filter_core_realism_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    return {k: float(metrics[k]) for k in CORE_REALISM_METRIC_KEYS if k in metrics}


def _append_step_manifest(
    *,
    manifest_path: Path,
    seed: int,
    n_mode: int,
    skip_steps: int,
    cfg: ForwardPassEvalConfig,
    scenario_records: List[Dict[str, Any]],
) -> None:
    payload: Dict[str, Any]
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    else:
        payload = {}

    if not payload:
        payload = {
            "version": "p4_core_realism_v1",
            "metric_scope": str(cfg.metric_scope),
            "seed": int(seed),
            "num_modes": int(n_mode),
            "skip_steps": int(skip_steps),
            "histogram_config": {
                "vel": {
                    "min": float(cfg.vel_hist_min),
                    "max": float(cfg.vel_hist_max),
                    "bins": int(cfg.vel_hist_bins),
                },
                "acc": {
                    "min": float(cfg.acc_hist_min),
                    "max": float(cfg.acc_hist_max),
                    "bins": int(cfg.acc_hist_bins),
                },
                "ttc": {
                    "min": float(cfg.ttc_hist_min),
                    "max": float(cfg.ttc_hist_max),
                    "bins": int(cfg.ttc_hist_bins),
                },
            },
            "core_realism_metric_keys": list(CORE_REALISM_METRIC_KEYS),
            "scenarios": [],
        }

    existing = {str(r.get("artifact_file", "")) for r in payload.get("scenarios", [])}
    for rec in scenario_records:
        af = str(rec.get("artifact_file", ""))
        if af and af in existing:
            continue
        payload.setdefault("scenarios", []).append(rec)
        if af:
            existing.add(af)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _histogram_jsd(
    gt_values: np.ndarray,
    pred_values: np.ndarray,
    *,
    min_val: float,
    max_val: float,
    num_bins: int,
    eps: float = 1e-10,
) -> float:
    gt = np.asarray(gt_values, dtype=np.float32)
    pred = np.asarray(pred_values, dtype=np.float32)

    gt = gt[np.isfinite(gt)]
    pred = pred[np.isfinite(pred)]
    if gt.size == 0 or pred.size == 0:
        return float("nan")

    gt = np.clip(gt, min_val, max_val)
    pred = np.clip(pred, min_val, max_val)

    gt_hist, _ = np.histogram(gt, bins=int(num_bins), range=(float(min_val), float(max_val)))
    pred_hist, _ = np.histogram(pred, bins=int(num_bins), range=(float(min_val), float(max_val)))

    if gt_hist.sum() <= 0 or pred_hist.sum() <= 0:
        return float("nan")

    p = gt_hist.astype(np.float64) / float(gt_hist.sum())
    q = pred_hist.astype(np.float64) / float(pred_hist.sum())

    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)

    kl_pm = np.sum(p * (np.log(p) - np.log(m)))
    kl_qm = np.sum(q * (np.log(q) - np.log(m)))
    return float(0.5 * (kl_pm + kl_qm))


def _first_last_valid_indices(mask_tn: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return first/last valid timestep indices per agent.

    Args:
        mask_tn: [T,N] boolean mask.
    Returns:
        has_valid: [N] bool
        first_idx: [N] int
        last_idx: [N] int
    """
    has_valid = np.any(mask_tn, axis=0)
    first_idx = np.argmax(mask_tn, axis=0)
    last_idx = mask_tn.shape[0] - 1 - np.argmax(mask_tn[::-1, :], axis=0)
    return has_valid, first_idx.astype(np.int32), last_idx.astype(np.int32)


def _pairwise_max_dist(points_k2: np.ndarray) -> float:
    """Maximum pairwise distance across K trajectory modes for one state."""
    if points_k2.shape[0] < 2:
        return float("nan")
    diffs = points_k2[:, None, :] - points_k2[None, :, :]
    d = np.linalg.norm(diffs, axis=-1)
    return float(np.max(d))


def _central_diff_2d(values_tn: np.ndarray, valid_tn: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Central difference on [T,N] with validity-aware mask."""
    t_steps = int(values_tn.shape[0])
    out = np.full_like(values_tn, np.nan, dtype=np.float32)
    out_mask = np.zeros_like(valid_tn, dtype=bool)

    if t_steps < 3 or dt <= 0.0:
        return out, out_mask

    out[1:-1, :] = (values_tn[2:, :] - values_tn[:-2, :]) / float(2.0 * dt)
    out_mask[1:-1, :] = valid_tn[2:, :] & valid_tn[1:-1, :] & valid_tn[:-2, :]
    return out, out_mask


def _central_diff_3d(values_btn: np.ndarray, valid_btn: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Central difference on [B,T,N] with validity-aware mask."""
    t_steps = int(values_btn.shape[1])
    out = np.full_like(values_btn, np.nan, dtype=np.float32)
    out_mask = np.zeros_like(valid_btn, dtype=bool)

    if t_steps < 3 or dt <= 0.0:
        return out, out_mask

    out[:, 1:-1, :] = (values_btn[:, 2:, :] - values_btn[:, :-2, :]) / float(2.0 * dt)
    out_mask[:, 1:-1, :] = valid_btn[:, 2:, :] & valid_btn[:, 1:-1, :] & valid_btn[:, :-2, :]
    return out, out_mask


def _approx_ttc_values(
    *,
    position_tn2: np.ndarray,
    velocity_tn2: np.ndarray,
    valid_tn: np.ndarray,
    radii_n: np.ndarray,
) -> np.ndarray:
    """Approximate TTC per agent/time using relative-motion collision checks.

    This is a dependency-light proxy for the WOD TTC operator used in Adv-BMT.
    """
    t_steps, n_agents, _ = position_tn2.shape
    out = np.full((t_steps, n_agents), np.nan, dtype=np.float32)
    eps = 1e-6

    for t in range(t_steps):
        idx = np.where(valid_tn[t])[0]
        if idx.size < 2:
            continue

        pos = position_tn2[t, idx]  # [M,2]
        vel = velocity_tn2[t, idx]  # [M,2]

        rel_pos = pos[None, :, :] - pos[:, None, :]  # i->j
        rel_vel = vel[None, :, :] - vel[:, None, :]

        vel_sq = np.sum(rel_vel * rel_vel, axis=-1)
        dot = np.sum(rel_pos * rel_vel, axis=-1)

        ttc = -dot / (vel_sq + eps)
        valid_pair = (vel_sq > eps) & (ttc > 0.0)

        closest = rel_pos + rel_vel * ttc[..., None]
        closest_sq = np.sum(closest * closest, axis=-1)

        thresh_sq = (radii_n[idx][:, None] + radii_n[idx][None, :]) ** 2
        valid_pair = valid_pair & (closest_sq <= thresh_sq)

        np.fill_diagonal(valid_pair, False)
        ttc = np.where(valid_pair, ttc, np.inf)

        min_ttc = np.min(ttc, axis=1)
        min_ttc[~np.isfinite(min_ttc)] = np.nan
        out[t, idx] = min_ttc.astype(np.float32)

    return out


def _reconstruct_rollout_states(
    *,
    predicted_tokens_kbtn: np.ndarray,
    action_table: np.ndarray,
    init_pos_bn2: np.ndarray,
    init_heading_bn: np.ndarray,
    init_speed_bn: np.ndarray,
    dt_chunk_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct per-mode trajectory states from sampled token rollouts.

    Integration follows the midpoint scheme used by Adv-BMT motion tokenization.
    """
    n_mode, bsz, horizon, n_agents = predicted_tokens_kbtn.shape

    pos = np.zeros((n_mode, bsz, horizon, n_agents, 2), dtype=np.float32)
    vel = np.zeros((n_mode, bsz, horizon, n_agents, 2), dtype=np.float32)
    heading = np.zeros((n_mode, bsz, horizon, n_agents), dtype=np.float32)
    speed = np.zeros((n_mode, bsz, horizon, n_agents), dtype=np.float32)

    dt_bn = dt_chunk_b[:, None].astype(np.float32)

    for k in range(n_mode):
        x = init_pos_bn2[..., 0].astype(np.float32).copy()
        y = init_pos_bn2[..., 1].astype(np.float32).copy()
        hd = init_heading_bn.astype(np.float32).copy()
        sp = init_speed_bn.astype(np.float32).copy()

        for t in range(horizon):
            tok = predicted_tokens_kbtn[k, :, t, :]
            acc = action_table[tok, 0].astype(np.float32)
            yaw = action_table[tok, 1].astype(np.float32)

            sp_next = sp + acc * dt_bn
            hd_next = hd + yaw * dt_bn

            sp_mid = 0.5 * (sp + sp_next)
            hd_mid = 0.5 * (hd + hd_next)

            x = x + sp_mid * np.cos(hd_mid) * dt_bn
            y = y + sp_mid * np.sin(hd_mid) * dt_bn

            vx = sp_next * np.cos(hd_next)
            vy = sp_next * np.sin(hd_next)

            pos[k, :, t, :, 0] = x
            pos[k, :, t, :, 1] = y
            vel[k, :, t, :, 0] = vx
            vel[k, :, t, :, 1] = vy
            heading[k, :, t, :] = hd_next
            speed[k, :, t, :] = np.abs(sp_next)

            sp = sp_next
            hd = hd_next

    return pos, vel, heading, speed


def _rollout_tokens_fixed_horizon(
    *,
    model: NNXBidirectionalMotionTransformer,
    agent_type_ids: jnp.ndarray,  # [B,N]
    agent_shape: jnp.ndarray,  # [B,N,3]
    agent_ids: jnp.ndarray,  # [B,N]
    reverse_indicator: jnp.ndarray,  # [B]
    horizon_steps: int,
    start_token_id: int,
    action_table: jnp.ndarray,  # [V,2]
    sampling_method: str,
    temperature: float,
    topp: float,
    topk: int,
    key: Any,
    scene_map_feature: jnp.ndarray | None = None,
    scene_map_valid_mask: jnp.ndarray | None = None,
    scene_map_position: jnp.ndarray | None = None,
    scene_tl_feature: jnp.ndarray | None = None,
    scene_tl_valid_mask: jnp.ndarray | None = None,
    scene_tl_position: jnp.ndarray | None = None,
    a2a_rel: jnp.ndarray | None = None,
    a2t_rel: jnp.ndarray | None = None,
    a2s_rel: jnp.ndarray | None = None,
    a2a_mask: jnp.ndarray | None = None,
    a2t_mask: jnp.ndarray | None = None,
    a2s_mask: jnp.ndarray | None = None,
    modeled_agent_delta_init: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Static-shape rollout for JIT-friendly forward-pass validation.

    This avoids dynamic sequence-length recompilation while preserving
    iterative token feedback through the `prev_token_ids` and motion channels.
    """
    bsz, n_agents = agent_type_ids.shape
    token_seq = jnp.full((bsz, horizon_steps, n_agents), int(start_token_id), dtype=jnp.int32)
    motion_seq = jnp.zeros((bsz, horizon_steps, n_agents, 2), dtype=jnp.float32)
    if modeled_agent_delta_init is None:
        modeled_delta_seq = jnp.zeros((bsz, horizon_steps, n_agents, 2), dtype=jnp.float32)
    else:
        md = jnp.asarray(modeled_agent_delta_init, dtype=jnp.float32)
        modeled_delta_seq = jnp.zeros((bsz, horizon_steps, n_agents, 2), dtype=jnp.float32)
        take = int(min(horizon_steps, md.shape[1]))
        modeled_delta_seq = modeled_delta_seq.at[:, :take, :, :].set(md[:, :take, :, :])

    model_kwargs = {
        "agent_type_ids": agent_type_ids,
        "agent_shape": agent_shape,
        "agent_ids": agent_ids,
        "reverse_indicator": reverse_indicator,
    }
    if scene_map_feature is not None:
        model_kwargs["scene_map_feature"] = scene_map_feature
    if scene_map_valid_mask is not None:
        model_kwargs["scene_map_valid_mask"] = scene_map_valid_mask
    if scene_map_position is not None:
        model_kwargs["scene_map_position"] = scene_map_position
    if scene_tl_feature is not None:
        model_kwargs["scene_tl_feature"] = scene_tl_feature
    if scene_tl_valid_mask is not None:
        model_kwargs["scene_tl_valid_mask"] = scene_tl_valid_mask
    if scene_tl_position is not None:
        model_kwargs["scene_tl_position"] = scene_tl_position
    if a2a_rel is not None:
        model_kwargs["a2a_rel"] = a2a_rel
    if a2t_rel is not None:
        model_kwargs["a2t_rel"] = a2t_rel
    if a2s_rel is not None:
        model_kwargs["a2s_rel"] = a2s_rel

    for t in range(horizon_steps):
        prefix_t = (jnp.arange(horizon_steps, dtype=jnp.int32) <= int(t))[None, :, None]
        model_kwargs["input_action_valid_mask"] = jnp.broadcast_to(prefix_t, (bsz, horizon_steps, n_agents))
        model_kwargs["modeled_agent_delta"] = modeled_delta_seq

        if a2a_mask is not None:
            prefix_a2a = model_kwargs["input_action_valid_mask"][:, :, :, None]
            model_kwargs["a2a_mask"] = jnp.logical_and(a2a_mask, prefix_a2a)
        if a2s_mask is not None:
            prefix_a2s = model_kwargs["input_action_valid_mask"][:, :, :, None]
            model_kwargs["a2s_mask"] = jnp.logical_and(a2s_mask, prefix_a2s)
        if a2t_mask is not None:
            tmask = (jnp.arange(horizon_steps, dtype=jnp.int32) <= int(t))
            causal_t = jnp.logical_and(tmask[None, None, :, None], tmask[None, None, None, :])
            model_kwargs["a2t_mask"] = jnp.logical_and(a2t_mask, causal_t)

        logits = model(
            prev_token_ids=token_seq,
            continuous_motion=motion_seq,
            **model_kwargs,
        )
        step_logits = logits[:, t, :, :]
        key, sub = jax.random.split(key)
        next_tok = sample_motion_tokens(
            step_logits,
            sub,
            sampling_method=sampling_method,
            temperature=float(temperature),
            topp=float(topp),
            topk=int(topk),
        )
        token_seq = token_seq.at[:, t, :].set(next_tok)
        next_motion = jnp.take(action_table, next_tok, axis=0)  # [B,N,2]
        motion_seq = motion_seq.at[:, t, :, :].set(next_motion)
        modeled_delta_seq = modeled_delta_seq.at[:, t, :, :].set(next_motion)

    return token_seq


def _compute_collision_and_comfort_metrics(
    *,
    pred_pos_ktn2: np.ndarray,
    pred_speed_ktn: np.ndarray,
    pred_valid_ktn: np.ndarray,
    radii_n: np.ndarray,
    sdc_index: int,
    dt_chunk_s: float,
) -> Dict[str, float]:
    """Approximate Adv-BMT collision/comfort metrics for one scenario."""
    n_mode, t_steps, n_agents = pred_valid_ktn.shape

    mode_collision_rates: List[float] = []
    mode_max_collision_speed: List[float] = []
    sdc_collision_speed_values: List[float] = []

    for k in range(n_mode):
        valid_tn = pred_valid_ktn[k]
        collided_n = np.zeros((n_agents,), dtype=bool)
        collision_speed_values: List[float] = []

        for t in range(t_steps):
            idx = np.where(valid_tn[t])[0]
            if idx.size < 2:
                continue

            pts = pred_pos_ktn2[k, t, idx]  # [M,2]
            d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
            thresh = radii_n[idx][:, None] + radii_n[idx][None, :]

            coll = d <= thresh
            np.fill_diagonal(coll, False)
            if not np.any(coll):
                continue

            coll_local = np.any(coll, axis=1)
            coll_agents = idx[coll_local]
            collided_n[coll_agents] = True

            collision_speed_values.extend(pred_speed_ktn[k, t, coll_agents].astype(np.float32).tolist())
            if int(sdc_index) in coll_agents.tolist():
                sdc_collision_speed_values.append(float(pred_speed_ktn[k, t, sdc_index]))

        valid_agents = np.any(valid_tn, axis=0)
        denom = int(valid_agents.sum())
        if denom > 0:
            mode_collision_rates.append(float(collided_n[valid_agents].mean()))

        if collision_speed_values:
            mode_max_collision_speed.append(float(np.max(collision_speed_values)))

    # SDC comfort metrics (same spirit as Adv-BMT evaluator).
    sdc_acc_avgtime: List[float] = []
    sdc_acc_maxtime: List[float] = []
    sdc_jerk_avgtime: List[float] = []
    sdc_jerk_maxtime: List[float] = []

    if 0 <= int(sdc_index) < n_agents:
        sdc_speed_kt = pred_speed_ktn[:, :, sdc_index]
        sdc_valid_kt = pred_valid_ktn[:, :, sdc_index]

        sdc_acc_kt, sdc_acc_mask_kt = _central_diff_3d(sdc_speed_kt[:, :, None], sdc_valid_kt[:, :, None], dt_chunk_s)
        sdc_acc_kt = np.abs(sdc_acc_kt[:, :, 0])
        sdc_acc_mask_kt = sdc_acc_mask_kt[:, :, 0]

        sdc_jerk_kt, sdc_jerk_mask_kt = _central_diff_3d(sdc_acc_kt[:, :, None], sdc_acc_mask_kt[:, :, None], dt_chunk_s)
        sdc_jerk_kt = np.abs(sdc_jerk_kt[:, :, 0])
        sdc_jerk_mask_kt = sdc_jerk_mask_kt[:, :, 0]

        for k in range(n_mode):
            if np.any(sdc_acc_mask_kt[k]):
                vals = sdc_acc_kt[k][sdc_acc_mask_kt[k]]
                sdc_acc_avgtime.append(float(np.mean(vals)))
                sdc_acc_maxtime.append(float(np.max(vals)))
            if np.any(sdc_jerk_mask_kt[k]):
                vals = sdc_jerk_kt[k][sdc_jerk_mask_kt[k]]
                sdc_jerk_avgtime.append(float(np.mean(vals)))
                sdc_jerk_maxtime.append(float(np.max(vals)))

    return {
        "veh_coll_avg": _safe_mean(mode_collision_rates),
        "veh_coll_min": float(np.min(mode_collision_rates)) if mode_collision_rates else float("nan"),
        "veh_coll_max": float(np.max(mode_collision_rates)) if mode_collision_rates else float("nan"),
        "coll_vel_maxagent_avg": _safe_mean(mode_max_collision_speed),
        "coll_vel_maxagent_min": float(np.min(mode_max_collision_speed)) if mode_max_collision_speed else float("nan"),
        "coll_vel_maxagent_max": float(np.max(mode_max_collision_speed)) if mode_max_collision_speed else float("nan"),
        "coll_vel_sdc_avg": _safe_mean(sdc_collision_speed_values),
        "coll_vel_sdc_min": float(np.min(sdc_collision_speed_values)) if sdc_collision_speed_values else float("nan"),
        "coll_vel_sdc_max": float(np.max(sdc_collision_speed_values)) if sdc_collision_speed_values else float("nan"),
        "sdc_acc_avgtime_avg": _safe_mean(sdc_acc_avgtime),
        "sdc_acc_avgtime_min": float(np.min(sdc_acc_avgtime)) if sdc_acc_avgtime else float("nan"),
        "sdc_acc_avgtime_max": float(np.max(sdc_acc_avgtime)) if sdc_acc_avgtime else float("nan"),
        "sdc_acc_maxtime_avg": _safe_mean(sdc_acc_maxtime),
        "sdc_acc_maxtime_min": float(np.min(sdc_acc_maxtime)) if sdc_acc_maxtime else float("nan"),
        "sdc_acc_maxtime_max": float(np.max(sdc_acc_maxtime)) if sdc_acc_maxtime else float("nan"),
        "sdc_jerk_avgtime_avg": _safe_mean(sdc_jerk_avgtime),
        "sdc_jerk_avgtime_min": float(np.min(sdc_jerk_avgtime)) if sdc_jerk_avgtime else float("nan"),
        "sdc_jerk_avgtime_max": float(np.max(sdc_jerk_avgtime)) if sdc_jerk_avgtime else float("nan"),
        "sdc_jerk_maxtime_avg": _safe_mean(sdc_jerk_maxtime),
        "sdc_jerk_maxtime_min": float(np.min(sdc_jerk_maxtime)) if sdc_jerk_maxtime else float("nan"),
        "sdc_jerk_maxtime_max": float(np.max(sdc_jerk_maxtime)) if sdc_jerk_maxtime else float("nan"),
    }


def _compute_scenario_metrics(
    *,
    pred_pos_ktn2: np.ndarray,
    pred_vel_ktn2: np.ndarray,
    pred_speed_ktn: np.ndarray,
    pred_valid_ktn: np.ndarray,
    gt_pos_tn2: np.ndarray,
    gt_vel_tn2: np.ndarray,
    gt_valid_tn: np.ndarray,
    agent_shape_n3: np.ndarray,
    dt_chunk_s: float,
    sdc_index: int,
    cfg: ForwardPassEvalConfig,
    rollout_start_valid_n: np.ndarray | None = None,  # [N] bool
    include_collision_metrics: bool = True,
) -> Dict[str, float]:
    """Compute Adv-BMT-style forward-pass metrics for one scenario."""
    n_mode, t_steps, n_agents, _ = pred_pos_ktn2.shape

    has_valid, first_idx, last_idx = _first_last_valid_indices(gt_valid_tn)
    all_valid_agents = np.where(has_valid)[0]
    if rollout_start_valid_n is not None:
        eval_mask = has_valid & np.asarray(rollout_start_valid_n, dtype=bool)
    else:
        eval_mask = has_valid
    valid_agents = np.where(eval_mask)[0]

    error_ktn = np.linalg.norm(pred_pos_ktn2 - gt_pos_tn2[None, ...], axis=-1)  # [K,T,N]

    sfde_modes: List[float] = []
    sade_modes: List[float] = []
    ssde_modes: List[float] = []

    for k in range(n_mode):
        fde_vals: List[float] = []
        sde_vals: List[float] = []
        ade_vals: List[float] = []

        for n in valid_agents.tolist():
            f = int(first_idx[n])
            l = int(last_idx[n])

            fde_vals.append(float(error_ktn[k, l, n]))
            sde_vals.append(float(error_ktn[k, f, n]))

            t_mask = gt_valid_tn[:, n]
            if np.any(t_mask):
                ade_vals.append(float(np.mean(error_ktn[k, t_mask, n])))

        sfde_modes.append(_safe_mean(fde_vals))
        ssde_modes.append(_safe_mean(sde_vals))
        sade_modes.append(_safe_mean(ade_vals))

    # Debug/diagnostic variant over all agents that have any valid transition.
    sfde_all_modes: List[float] = []
    for k in range(n_mode):
        vals: List[float] = []
        for n in all_valid_agents.tolist():
            l = int(last_idx[n])
            vals.append(float(error_ktn[k, l, n]))
        sfde_all_modes.append(_safe_mean(vals))

    # Diversity metrics from Adv-BMT evaluator (FDD/ADD/SDD).
    fdd_vals: List[float] = []
    sdd_vals: List[float] = []
    add_vals: List[float] = []

    if n_mode >= 2:
        for n in valid_agents.tolist():
            f = int(first_idx[n])
            l = int(last_idx[n])
            fdd_vals.append(_pairwise_max_dist(pred_pos_ktn2[:, l, n, :]))
            sdd_vals.append(_pairwise_max_dist(pred_pos_ktn2[:, f, n, :]))

        for t in range(t_steps):
            for n in valid_agents.tolist():
                if not gt_valid_tn[t, n]:
                    continue
                add_vals.append(_pairwise_max_dist(pred_pos_ktn2[:, t, n, :]))

    # Distribution realism histograms.
    gt_speed_tn = np.linalg.norm(gt_vel_tn2, axis=-1)
    pred_speed_flat = pred_speed_ktn[pred_valid_ktn]
    gt_speed_flat = gt_speed_tn[gt_valid_tn]

    gt_acc_tn, gt_acc_mask_tn = _central_diff_2d(gt_speed_tn, gt_valid_tn, dt_chunk_s)
    pred_acc_ktn, pred_acc_mask_ktn = _central_diff_3d(pred_speed_ktn, pred_valid_ktn, dt_chunk_s)

    gt_acc_flat = gt_acc_tn[gt_acc_mask_tn]
    pred_acc_flat = pred_acc_ktn[pred_acc_mask_ktn]

    # TTC proxy histogram.
    length = np.asarray(agent_shape_n3[:, 0], dtype=np.float32)
    width = np.asarray(agent_shape_n3[:, 1], dtype=np.float32)
    radius = 0.5 * np.sqrt(np.maximum(length, 0.0) ** 2 + np.maximum(width, 0.0) ** 2)
    radius = np.maximum(radius, float(cfg.collision_radius_floor_m)).astype(np.float32)

    gt_ttc_tn = _approx_ttc_values(
        position_tn2=gt_pos_tn2,
        velocity_tn2=gt_vel_tn2,
        valid_tn=gt_valid_tn,
        radii_n=radius,
    )
    pred_ttc_list: List[np.ndarray] = []
    for k in range(n_mode):
        pred_ttc_list.append(
            _approx_ttc_values(
                position_tn2=pred_pos_ktn2[k],
                velocity_tn2=pred_vel_ktn2[k],
                valid_tn=pred_valid_ktn[k],
                radii_n=radius,
            )
        )
    pred_ttc_ktn = np.stack(pred_ttc_list, axis=0)

    gt_ttc_flat = gt_ttc_tn[gt_valid_tn]
    pred_ttc_flat = pred_ttc_ktn[pred_valid_ktn]

    out = {
        "sfde_min": float(np.min(sfde_modes)) if sfde_modes else float("nan"),
        "sfde_avg": _safe_mean(sfde_modes),
        "sfde_all_min": float(np.min(sfde_all_modes)) if sfde_all_modes else float("nan"),
        "sfde_all_avg": _safe_mean(sfde_all_modes),
        "sade_min": float(np.min(sade_modes)) if sade_modes else float("nan"),
        "sade_avg": _safe_mean(sade_modes),
        "ssde_min": float(np.min(ssde_modes)) if ssde_modes else float("nan"),
        "ssde_avg": _safe_mean(ssde_modes),
        "fdd": _safe_mean(fdd_vals),
        "sdd": _safe_mean(sdd_vals),
        "add": _safe_mean(add_vals),
        "vel_jsd": _histogram_jsd(
            gt_speed_flat,
            pred_speed_flat,
            min_val=float(cfg.vel_hist_min),
            max_val=float(cfg.vel_hist_max),
            num_bins=int(cfg.vel_hist_bins),
        ),
        "acc_jsd": _histogram_jsd(
            gt_acc_flat,
            pred_acc_flat,
            min_val=float(cfg.acc_hist_min),
            max_val=float(cfg.acc_hist_max),
            num_bins=int(cfg.acc_hist_bins),
        ),
        "ttc_jsd": _histogram_jsd(
            gt_ttc_flat,
            pred_ttc_flat,
            min_val=float(cfg.ttc_hist_min),
            max_val=float(cfg.ttc_hist_max),
            num_bins=int(cfg.ttc_hist_bins),
        ),
        "num_eval_agents": float(valid_agents.shape[0]),
        "num_all_valid_agents": float(all_valid_agents.shape[0]),
    }

    if include_collision_metrics:
        out.update(
            _compute_collision_and_comfort_metrics(
                pred_pos_ktn2=pred_pos_ktn2,
                pred_speed_ktn=pred_speed_ktn,
                pred_valid_ktn=pred_valid_ktn,
                radii_n=radius,
                sdc_index=int(sdc_index),
                dt_chunk_s=float(dt_chunk_s),
            )
        )
    return out


def compute_forward_pass_metrics_for_batch(
    *,
    model: NNXBidirectionalMotionTransformer,
    prepared_batch: Dict[str, Any],
    tokenizer: BidirectionalMotionTokenizer,
    skip_steps: int,
    eval_cfg: ForwardPassEvalConfig,
    seed: int,
    output_dir: Path | None = None,
    global_step: int | None = None,
    max_visualizations: int = 0,
    max_artifacts: int = 0,
) -> Tuple[List[Dict[str, float]], int, int]:
    """Compute scenario-level forward-pass metrics for one validation batch.

    The rollout path is intentionally aligned with Adv-BMT evaluation intent:
    multi-mode autoregressive sampling followed by scenario-level metric
    aggregation against the same scenario's ground-truth future trajectory.
    """
    if not eval_cfg.enabled:
        return [], 0, 0

    model_inputs = prepared_batch["model_inputs"]
    raw = prepared_batch["raw_batch"]

    sample_steps = np.asarray(prepared_batch["sample_steps"], dtype=np.int32)
    gt_action_valid_btn = np.asarray(prepared_batch["target_mask"], dtype=np.float32) > 0.5

    bsz = int(raw["agent_position_xy"].shape[0])
    n_agents = int(raw["agent_position_xy"].shape[2])
    horizon_from_mask = int(gt_action_valid_btn.shape[1])
    horizon_from_steps = int(max(0, sample_steps.shape[0] - 1))
    horizon_eval = int(min(horizon_from_mask, horizon_from_steps))
    if horizon_eval <= 0:
        return [], 0, 0

    # Adv-BMT forward-pass eval samples K trajectory modes.
    n_mode = max(1, int(eval_cfg.num_modes))
    start_token_id = int(tokenizer.cfg.n_tokens)

    key = jax.random.PRNGKey(int(seed))
    pred_tokens: List[np.ndarray] = []

    action_table_jnp = jnp.asarray(tokenizer.action_table_np())
    a2a_rel_eval = None if model_inputs.get("a2a_rel") is None else model_inputs["a2a_rel"][:, :horizon_eval, ...]
    a2t_rel_eval = None if model_inputs.get("a2t_rel") is None else model_inputs["a2t_rel"][:, :, :horizon_eval, :horizon_eval, ...]
    a2s_rel_eval = None if model_inputs.get("a2s_rel") is None else model_inputs["a2s_rel"][:, :horizon_eval, ...]
    a2a_mask_eval = None if model_inputs.get("a2a_mask") is None else model_inputs["a2a_mask"][:, :horizon_eval, ...]
    a2t_mask_eval = None if model_inputs.get("a2t_mask") is None else model_inputs["a2t_mask"][:, :, :horizon_eval, :horizon_eval]
    a2s_mask_eval = None if model_inputs.get("a2s_mask") is None else model_inputs["a2s_mask"][:, :horizon_eval, ...]
    modeled_delta_eval = (
        None if model_inputs.get("modeled_agent_delta") is None else model_inputs["modeled_agent_delta"][:, :horizon_eval, ...]
    )

    for _ in range(n_mode):
        key, sub = jax.random.split(key)
        sampled_tok = _rollout_tokens_fixed_horizon(
            model=model,
            agent_type_ids=model_inputs["agent_type_ids"],
            agent_shape=model_inputs["agent_shape"],
            agent_ids=model_inputs["agent_ids"],
            reverse_indicator=jnp.zeros_like(prepared_batch["reverse_indicator"], dtype=jnp.int32),
            horizon_steps=int(horizon_eval),
            start_token_id=start_token_id,
            action_table=action_table_jnp,
            scene_map_feature=model_inputs.get("scene_map_feature"),
            scene_map_valid_mask=model_inputs.get("scene_map_valid_mask"),
            scene_map_position=model_inputs.get("scene_map_position"),
            scene_tl_feature=model_inputs.get("scene_tl_feature"),
            scene_tl_valid_mask=model_inputs.get("scene_tl_valid_mask"),
            scene_tl_position=model_inputs.get("scene_tl_position"),
            a2a_rel=a2a_rel_eval,
            a2t_rel=a2t_rel_eval,
            a2s_rel=a2s_rel_eval,
            a2a_mask=a2a_mask_eval,
            a2t_mask=a2t_mask_eval,
            a2s_mask=a2s_mask_eval,
            modeled_agent_delta_init=modeled_delta_eval,
            sampling_method=eval_cfg.sampling_method,
            temperature=float(eval_cfg.temperature),
            topp=float(eval_cfg.topp),
            topk=int(eval_cfg.topk),
            key=sub,
        )
        pred_tokens.append(np.asarray(jax.device_get(sampled_tok), dtype=np.int32))

    pred_tokens_kbtn = np.stack(pred_tokens, axis=0)  # [K,B,T,N]

    agent_pos_btn2 = np.asarray(raw["agent_position_xy"], dtype=np.float32)
    agent_vel_btn2 = np.asarray(raw["agent_velocity_xy"], dtype=np.float32)
    agent_heading_btn = np.asarray(raw["agent_heading"], dtype=np.float32)
    agent_valid_btn = np.asarray(raw["agent_valid_mask"], dtype=bool)
    agent_shape_bn3 = np.asarray(raw["agent_shape"], dtype=np.float32)

    init_t = int(sample_steps[0])
    eval_steps = sample_steps[1:1 + horizon_eval]

    init_pos_bn2 = agent_pos_btn2[:, init_t, :, :]
    init_heading_bn = agent_heading_btn[:, init_t, :]
    init_speed_bn = np.linalg.norm(agent_vel_btn2[:, init_t, :, :], axis=-1)

    dt_chunk_b = np.asarray(raw["dt_s"], dtype=np.float32) * float(skip_steps)
    action_table = tokenizer.action_table_np()

    pred_pos_kbtn2, pred_vel_kbtn2, pred_heading_kbtn, pred_speed_kbtn = _reconstruct_rollout_states(
        predicted_tokens_kbtn=pred_tokens_kbtn,
        action_table=action_table,
        init_pos_bn2=init_pos_bn2,
        init_heading_bn=init_heading_bn,
        init_speed_bn=init_speed_bn,
        dt_chunk_b=dt_chunk_b,
    )

    gt_pos_btn2 = agent_pos_btn2[:, eval_steps, :, :]
    gt_vel_btn2 = agent_vel_btn2[:, eval_steps, :, :]
    gt_heading_btn = agent_heading_btn[:, eval_steps, :]

    # Validity is action-transition validity (both endpoints valid).
    valid_sampled_btn = agent_valid_btn[:, sample_steps[: horizon_eval + 1], :]
    gt_valid_btn = valid_sampled_btn[:, 1:, :] & valid_sampled_btn[:, :-1, :]

    # Model does not currently predict valid-mask logits; use GT transition validity.
    pred_valid_kbtn = np.broadcast_to(gt_valid_btn[None, :, :, :], pred_speed_kbtn.shape)

    scenario_ids = raw.get("scenario_ids", [f"scenario_{i}" for i in range(bsz)])
    viz_saved = 0
    viz_dir: Path | None = None
    if (
        eval_cfg.save_visualizations
        and output_dir is not None
        and global_step is not None
        and max_visualizations > 0
    ):
        viz_dir = Path(output_dir) / str(eval_cfg.viz_output_subdir) / f"step_{int(global_step):07d}"

    artifact_saved = 0
    artifact_step_dir: Path | None = None
    artifact_records: List[Dict[str, Any]] = []
    if (
        bool(eval_cfg.export_artifacts)
        and output_dir is not None
        and global_step is not None
        and max_artifacts > 0
    ):
        artifact_step_dir = Path(output_dir) / str(eval_cfg.artifact_output_subdir) / f"step_{int(global_step):07d}"
        artifact_step_dir.mkdir(parents=True, exist_ok=True)

    per_scenario_metrics: List[Dict[str, float]] = []
    for b in range(bsz):
        rollout_start_valid_n = agent_valid_btn[b, init_t]
        scenario_metrics = _compute_scenario_metrics(
            pred_pos_ktn2=pred_pos_kbtn2[:, b],
            pred_vel_ktn2=pred_vel_kbtn2[:, b],
            pred_speed_ktn=pred_speed_kbtn[:, b],
            pred_valid_ktn=pred_valid_kbtn[:, b],
            gt_pos_tn2=gt_pos_btn2[b],
            gt_vel_tn2=gt_vel_btn2[b],
            gt_valid_tn=gt_valid_btn[b],
            agent_shape_n3=agent_shape_bn3[b],
            dt_chunk_s=float(max(dt_chunk_b[b], 1e-6)),
            sdc_index=0,
            cfg=eval_cfg,
            rollout_start_valid_n=rollout_start_valid_n,
            include_collision_metrics=False,
        )
        scenario_metrics = _filter_core_realism_metrics(scenario_metrics)
        per_scenario_metrics.append(scenario_metrics)

        if viz_dir is not None and viz_saved < int(max_visualizations):
            sid = str(scenario_ids[b]) if b < len(scenario_ids) else f"scenario_{b}"
            file_name = f"{_sanitize_name(sid)}.png"
            saved = _save_rollout_vs_gt_plot(
                out_file=viz_dir / file_name,
                scenario_id=sid,
                gt_pos_tn2=gt_pos_btn2[b],
                gt_valid_tn=gt_valid_btn[b],
                pred_pos_ktn2=pred_pos_kbtn2[:, b],
                max_agents=int(max(1, eval_cfg.viz_max_agents)),
            )
            if saved:
                viz_saved += 1

        if artifact_step_dir is not None and artifact_saved < int(max_artifacts):
            sid = str(scenario_ids[b]) if b < len(scenario_ids) else f"scenario_{b}"
            artifact_file = f"{_sanitize_name(sid)}.npz"
            artifact_path = artifact_step_dir / artifact_file
            metric_keys = np.asarray(list(scenario_metrics.keys()), dtype=object)
            metric_vals = np.asarray([float(scenario_metrics[k]) for k in metric_keys.tolist()], dtype=np.float32)
            np.savez_compressed(
                artifact_path,
                pred_pos_ktn2=np.asarray(pred_pos_kbtn2[:, b], dtype=np.float32),
                pred_vel_ktn2=np.asarray(pred_vel_kbtn2[:, b], dtype=np.float32),
                pred_speed_ktn=np.asarray(pred_speed_kbtn[:, b], dtype=np.float32),
                pred_valid_ktn=np.asarray(pred_valid_kbtn[:, b], dtype=bool),
                pred_heading_ktn=np.asarray(pred_heading_kbtn[:, b], dtype=np.float32),
                gt_pos_tn2=np.asarray(gt_pos_btn2[b], dtype=np.float32),
                gt_vel_tn2=np.asarray(gt_vel_btn2[b], dtype=np.float32),
                gt_valid_tn=np.asarray(gt_valid_btn[b], dtype=bool),
                gt_heading_tn=np.asarray(gt_heading_btn[b], dtype=np.float32),
                rollout_start_valid_n=np.asarray(rollout_start_valid_n, dtype=bool),
                agent_shape_n3=np.asarray(agent_shape_bn3[b], dtype=np.float32),
                dt_chunk_s=np.asarray(float(max(dt_chunk_b[b], 1e-6)), dtype=np.float32),
                sdc_index=np.asarray(0, dtype=np.int32),
                scenario_id=np.asarray(sid, dtype=object),
                forward_approx_metric_keys=metric_keys,
                forward_approx_metric_values=metric_vals,
            )
            artifact_records.append(
                {
                    "scenario_id": sid,
                    "artifact_file": artifact_file,
                    "artifact_path": str(artifact_path),
                    "forward_approx_metrics": {k: float(v) for k, v in scenario_metrics.items()},
                }
            )
            artifact_saved += 1

    if artifact_step_dir is not None and artifact_records:
        _append_step_manifest(
            manifest_path=artifact_step_dir / "manifest.json",
            seed=int(seed),
            n_mode=int(n_mode),
            skip_steps=int(skip_steps),
            cfg=eval_cfg,
            scenario_records=artifact_records,
        )

    return per_scenario_metrics, viz_saved, artifact_saved
