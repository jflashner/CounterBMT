"""Compare forward-pass metrics from exported artifacts (approx vs strict parity)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Allow running as a standalone script from repo root.
if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[2]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

from counter_bmt_v2.training.forward_metrics import (  # noqa: E402
    CORE_REALISM_METRIC_KEYS,
    ForwardPassEvalConfig,
    _compute_scenario_metrics,
    _histogram_jsd,
)

_WAYMO_TTC_MODULES: tuple[Any, Any] | None = None
_WAYMO_TTC_INIT_FAILED = False


def _safe_mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _masked_average(vals: np.ndarray, mask: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    denom = int(mask.sum())
    if denom <= 0:
        return float("nan")
    return float(vals[mask].sum() / float(denom))


def _first_valid_idx(valid_tn: np.ndarray) -> np.ndarray:
    return np.argmax(valid_tn, axis=0).astype(np.int32)


def _last_valid_idx(valid_tn: np.ndarray) -> np.ndarray:
    return (valid_tn.shape[0] - 1 - np.argmax(valid_tn[::-1, :], axis=0)).astype(np.int32)


def _central_diff(values: np.ndarray, dt: float) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float32)
    if values.shape[0] < 3 or dt <= 0.0:
        return out
    out[1:-1, ...] = (values[2:, ...] - values[:-2, ...]) / float(2.0 * dt)
    return out


def _approx_ttc_values(
    *,
    position_tn2: np.ndarray,
    velocity_tn2: np.ndarray,
    valid_tn: np.ndarray,
    radii_n: np.ndarray,
) -> np.ndarray:
    t_steps, n_agents, _ = position_tn2.shape
    out = np.full((t_steps, n_agents), np.nan, dtype=np.float32)
    eps = 1e-6
    for t in range(t_steps):
        idx = np.where(valid_tn[t])[0]
        if idx.size < 2:
            continue
        pos = position_tn2[t, idx]
        vel = velocity_tn2[t, idx]
        rel_pos = pos[None, :, :] - pos[:, None, :]
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


def _try_waymo_ttc(
    *,
    pos_tn2: np.ndarray,
    heading_tn: np.ndarray,
    valid_tn: np.ndarray,
    shape_n3: np.ndarray,
    dt_s: float,
) -> np.ndarray | None:
    global _WAYMO_TTC_MODULES, _WAYMO_TTC_INIT_FAILED
    if _WAYMO_TTC_INIT_FAILED:
        return None
    if _WAYMO_TTC_MODULES is None:
        try:
            import tensorflow as tf  # type: ignore
            from waymo_open_dataset.wdl_limited.sim_agents_metrics import interaction_features  # type: ignore
            _WAYMO_TTC_MODULES = (tf, interaction_features)
        except Exception:
            _WAYMO_TTC_INIT_FAILED = True
            return None
    tf, interaction_features = _WAYMO_TTC_MODULES

    try:
        center_x = tf.convert_to_tensor(pos_tn2[..., 0].T, dtype=tf.float32)
        center_y = tf.convert_to_tensor(pos_tn2[..., 1].T, dtype=tf.float32)
        length = tf.convert_to_tensor(np.tile(shape_n3[:, 0:1], (1, pos_tn2.shape[0])), dtype=tf.float32)
        width = tf.convert_to_tensor(np.tile(shape_n3[:, 1:2], (1, pos_tn2.shape[0])), dtype=tf.float32)
        heading = tf.convert_to_tensor(heading_tn.T, dtype=tf.float32)
        valid = tf.convert_to_tensor(valid_tn.T, dtype=tf.bool)
        eval_mask = tf.convert_to_tensor(np.ones((pos_tn2.shape[1],), dtype=bool), dtype=tf.bool)

        ttc_nt = interaction_features.compute_time_to_collision_with_object_in_front(
            center_x=center_x,
            center_y=center_y,
            length=length,
            width=width,
            heading=heading,
            valid=valid,
            evaluated_object_mask=eval_mask,
            seconds_per_step=float(dt_s),
        )
        return np.asarray(ttc_nt.numpy().T, dtype=np.float32)
    except Exception:
        return None


def _strict_metrics_from_artifact(
    data: Dict[str, np.ndarray],
    cfg: ForwardPassEvalConfig,
) -> Tuple[Dict[str, float], str]:
    pred_pos = np.asarray(data["pred_pos_ktn2"], dtype=np.float32)
    pred_vel = np.asarray(data["pred_vel_ktn2"], dtype=np.float32)
    pred_speed = np.asarray(data["pred_speed_ktn"], dtype=np.float32)
    pred_valid = np.asarray(data["pred_valid_ktn"], dtype=bool)
    pred_heading = np.asarray(data["pred_heading_ktn"], dtype=np.float32)
    gt_pos = np.asarray(data["gt_pos_tn2"], dtype=np.float32)
    gt_vel = np.asarray(data["gt_vel_tn2"], dtype=np.float32)
    gt_valid = np.asarray(data["gt_valid_tn"], dtype=bool)
    gt_heading = np.asarray(data["gt_heading_tn"], dtype=np.float32)
    shape_n3 = np.asarray(data["agent_shape_n3"], dtype=np.float32)
    dt_chunk_s = float(np.asarray(data["dt_chunk_s"]).item())
    rollout_start_valid = np.asarray(
        data.get("rollout_start_valid_n", np.any(gt_valid, axis=0)),
        dtype=bool,
    )

    # Strict path reuses the shared v2 core-realism semantics, then optionally
    # swaps TTC JSD with Waymo operator output when available in the strict env.
    base = _compute_scenario_metrics(
        pred_pos_ktn2=pred_pos,
        pred_vel_ktn2=pred_vel,
        pred_speed_ktn=pred_speed,
        pred_valid_ktn=pred_valid,
        gt_pos_tn2=gt_pos,
        gt_vel_tn2=gt_vel,
        gt_valid_tn=gt_valid,
        agent_shape_n3=shape_n3,
        dt_chunk_s=dt_chunk_s,
        sdc_index=int(np.asarray(data["sdc_index"]).item()),
        cfg=cfg,
        rollout_start_valid_n=rollout_start_valid,
        include_collision_metrics=False,
    )
    strict = {k: float(base.get(k, np.nan)) for k in CORE_REALISM_METRIC_KEYS}

    # Waymo TTC override (strict env). If unavailable, keep approximation.
    length = np.maximum(shape_n3[:, 0], 0.0)
    width = np.maximum(shape_n3[:, 1], 0.0)
    radius = np.maximum(0.5 * np.sqrt(length**2 + width**2), float(cfg.collision_radius_floor_m)).astype(np.float32)

    gt_ttc_tn = _try_waymo_ttc(
        pos_tn2=gt_pos,
        heading_tn=gt_heading,
        valid_tn=gt_valid,
        shape_n3=shape_n3,
        dt_s=dt_chunk_s,
    )
    if gt_ttc_tn is None:
        return strict, "approx_fallback"

    pred_ttc_ktn = np.full((pred_pos.shape[0], pred_pos.shape[1], pred_pos.shape[2]), np.nan, dtype=np.float32)
    for k in range(pred_pos.shape[0]):
        waymo_ttc = _try_waymo_ttc(
            pos_tn2=pred_pos[k],
            heading_tn=pred_heading[k],
            valid_tn=pred_valid[k],
            shape_n3=shape_n3,
            dt_s=dt_chunk_s,
        )
        if waymo_ttc is None:
            return strict, "approx_fallback"
        pred_ttc_ktn[k] = waymo_ttc

    gt_ttc_flat = gt_ttc_tn[gt_valid & np.isfinite(gt_ttc_tn)]
    pred_ttc_flat = pred_ttc_ktn[pred_valid & np.isfinite(pred_ttc_ktn)]
    strict["ttc_jsd"] = _histogram_jsd(
        gt_ttc_flat,
        pred_ttc_flat,
        min_val=float(cfg.ttc_hist_min),
        max_val=float(cfg.ttc_hist_max),
        num_bins=int(cfg.ttc_hist_bins),
    )
    return strict, "waymo"


def _aggregate(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in CORE_REALISM_METRIC_KEYS:
        vals = np.asarray([m.get(k, np.nan) for m in metrics], dtype=np.float32)
        out[k] = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    if np.allclose(a, a[0]) and np.allclose(b, b[0]):
        return 1.0 if np.allclose(a, b) else 0.0
    if float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _scenario_arrays(samples: List[Dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([s.get(key, np.nan) for s in samples], dtype=np.float32)


def _load_approx_metrics_from_npz(d: Dict[str, np.ndarray]) -> Dict[str, float] | None:
    if "forward_approx_metric_keys" not in d or "forward_approx_metric_values" not in d:
        return None
    keys = [str(k) for k in np.asarray(d["forward_approx_metric_keys"], dtype=object).tolist()]
    vals = np.asarray(d["forward_approx_metric_values"], dtype=np.float32).tolist()
    return {k: float(v) for k, v in zip(keys, vals)}


def _recompute_approx_metrics(d: Dict[str, np.ndarray], cfg: ForwardPassEvalConfig) -> Dict[str, float]:
    rollout_start = np.asarray(d.get("rollout_start_valid_n", np.any(np.asarray(d["gt_valid_tn"], dtype=bool), axis=0)), dtype=bool)
    metrics = _compute_scenario_metrics(
        pred_pos_ktn2=np.asarray(d["pred_pos_ktn2"], dtype=np.float32),
        pred_vel_ktn2=np.asarray(d["pred_vel_ktn2"], dtype=np.float32),
        pred_speed_ktn=np.asarray(d["pred_speed_ktn"], dtype=np.float32),
        pred_valid_ktn=np.asarray(d["pred_valid_ktn"], dtype=bool),
        gt_pos_tn2=np.asarray(d["gt_pos_tn2"], dtype=np.float32),
        gt_vel_tn2=np.asarray(d["gt_vel_tn2"], dtype=np.float32),
        gt_valid_tn=np.asarray(d["gt_valid_tn"], dtype=bool),
        agent_shape_n3=np.asarray(d["agent_shape_n3"], dtype=np.float32),
        dt_chunk_s=float(np.asarray(d["dt_chunk_s"]).item()),
        sdc_index=int(np.asarray(d["sdc_index"]).item()),
        cfg=cfg,
        rollout_start_valid_n=rollout_start,
        include_collision_metrics=False,
    )
    return {k: float(metrics[k]) for k in CORE_REALISM_METRIC_KEYS if k in metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare forward_approx and strict forward_parity metrics from artifacts")
    parser.add_argument("--artifact-dir", type=str, required=True, help="forward_eval_artifacts directory")
    parser.add_argument("--legacy-root", type=str, default="", help="optional legacy root (added to PYTHONPATH if provided)")
    parser.add_argument("--n", type=int, default=-1, help="max number of scenario artifacts to evaluate")
    parser.add_argument("--output-json", type=str, default="", help="optional JSON report path")
    parser.add_argument("--max-rel-error", type=float, default=0.01)
    parser.add_argument("--min-corr", type=float, default=0.99)
    args = parser.parse_args()

    if args.legacy_root:
        root = str(Path(args.legacy_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)

    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.exists():
        raise FileNotFoundError(f"artifact dir not found: {artifact_dir}")

    files = sorted(artifact_dir.glob("step_*/*.npz"))
    if int(args.n) > 0:
        files = files[: int(args.n)]
    if not files:
        raise ValueError(f"No artifact npz files found under {artifact_dir}")

    cfg = ForwardPassEvalConfig(metric_scope="core_realism")
    per_scenario_approx: List[Dict[str, float]] = []
    per_scenario_strict: List[Dict[str, float]] = []
    scenario_rows: List[Dict[str, Any]] = []
    ttc_backends: Dict[str, int] = {}

    for f in files:
        npz = np.load(f, allow_pickle=True)
        d: Dict[str, np.ndarray] = {k: npz[k] for k in npz.files}
        sid = str(np.asarray(d.get("scenario_id", f.stem), dtype=object).item())

        approx = _load_approx_metrics_from_npz(d)
        if approx is None:
            approx = _recompute_approx_metrics(d, cfg)
        approx = {k: float(approx.get(k, np.nan)) for k in CORE_REALISM_METRIC_KEYS}

        strict, ttc_backend = _strict_metrics_from_artifact(d, cfg)
        strict = {k: float(strict.get(k, np.nan)) for k in CORE_REALISM_METRIC_KEYS}
        ttc_backends[ttc_backend] = ttc_backends.get(ttc_backend, 0) + 1

        per_scenario_approx.append(approx)
        per_scenario_strict.append(strict)
        scenario_rows.append(
            {
                "scenario_id": sid,
                "artifact": str(f),
                "ttc_backend": ttc_backend,
                "forward_approx": approx,
                "forward_parity": strict,
            }
        )

    agg_approx = _aggregate(per_scenario_approx)
    agg_strict = _aggregate(per_scenario_strict)

    error_table: Dict[str, Dict[str, float]] = {}
    corr_table: Dict[str, float] = {}
    failures: List[str] = []
    has_nan_mismatch = False

    for k in CORE_REALISM_METRIC_KEYS:
        a = float(agg_approx.get(k, np.nan))
        s = float(agg_strict.get(k, np.nan))
        abs_err = float(abs(s - a)) if np.isfinite(a) and np.isfinite(s) else float("nan")
        rel_err = float(abs_err / max(abs(a), 1e-6)) if np.isfinite(abs_err) else float("nan")
        error_table[k] = {"abs_error": abs_err, "rel_error": rel_err}

        av = _scenario_arrays(per_scenario_approx, k)
        sv = _scenario_arrays(per_scenario_strict, k)
        finite = np.isfinite(av) & np.isfinite(sv)
        corr = _pearson(av[finite], sv[finite]) if np.any(finite) else float("nan")
        corr_table[k] = float(corr)

        if np.isfinite(a) and not np.isfinite(s):
            has_nan_mismatch = True
            failures.append(f"{k}: strict is NaN while approx is finite")
        if np.isfinite(rel_err) and rel_err > float(args.max_rel_error):
            failures.append(f"{k}: rel_error={rel_err:.6f} > {float(args.max_rel_error):.6f}")
        if np.isfinite(corr) and corr < float(args.min_corr):
            failures.append(f"{k}: corr={corr:.6f} < {float(args.min_corr):.6f}")

    payload = {
        "config": {
            "artifact_dir": str(artifact_dir),
            "legacy_root": str(args.legacy_root),
            "n": int(args.n),
            "max_rel_error": float(args.max_rel_error),
            "min_corr": float(args.min_corr),
            "metric_scope": "core_realism",
            "metric_keys": list(CORE_REALISM_METRIC_KEYS),
        },
        "summary": {
            "num_scenarios": int(len(files)),
            "ttc_backend_counts": ttc_backends,
            "has_nan_mismatch": bool(has_nan_mismatch),
            "failed_checks": int(len(failures)),
        },
        "forward_approx": {f"forward_approx/{k}": float(v) for k, v in agg_approx.items()},
        "forward_parity": {f"forward_parity/{k}": float(v) for k, v in agg_strict.items()},
        "error_table": error_table,
        "scenario_correlation": corr_table,
        "failures": failures,
        "scenarios": scenario_rows,
    }

    print(json.dumps(payload, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if failures:
        print("FAILED: forward parity comparison failed thresholds")
        return 1
    print("PASSED: forward parity comparison thresholds met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
