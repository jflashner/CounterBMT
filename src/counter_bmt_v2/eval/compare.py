"""Metric aggregation and pairwise comparison for head-to-head eval."""

from __future__ import annotations

import csv
import importlib.util
import itertools
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from counter_bmt_v2.training.forward_metrics import (
    CORE_REALISM_METRIC_KEYS,
    ForwardPassEvalConfig,
    _compute_scenario_metrics,
)


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float32)
    return float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else float("nan")


def _scenario_id_from_npz(path: Path) -> str:
    with np.load(path, allow_pickle=True) as d:
        sid = d.get("scenario_id")
        if sid is None:
            return path.stem
        if isinstance(sid, np.ndarray):
            if sid.size == 0:
                return path.stem
            sid = sid.reshape(-1)[0]
        return str(sid)


def _load_approx_metrics(d: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    keys = d.get("forward_approx_metric_keys")
    vals = d.get("forward_approx_metric_values")
    if keys is None or vals is None:
        return None
    key_list = [str(k) for k in np.asarray(keys, dtype=object).tolist()]
    val_list = np.asarray(vals, dtype=np.float32).tolist()
    return {k: float(v) for k, v in zip(key_list, val_list)}


def _recompute_approx_metrics(d: Mapping[str, Any]) -> Dict[str, float]:
    cfg = ForwardPassEvalConfig(enabled=True, metric_scope="core_realism")
    gt_valid = np.asarray(d["gt_valid_tn"], dtype=bool)
    rollout_start_valid = np.asarray(d.get("rollout_start_valid_n", np.any(gt_valid, axis=0)), dtype=bool)
    metrics = _compute_scenario_metrics(
        pred_pos_ktn2=np.asarray(d["pred_pos_ktn2"], dtype=np.float32),
        pred_vel_ktn2=np.asarray(d["pred_vel_ktn2"], dtype=np.float32),
        pred_speed_ktn=np.asarray(d["pred_speed_ktn"], dtype=np.float32),
        pred_valid_ktn=np.asarray(d["pred_valid_ktn"], dtype=bool),
        gt_pos_tn2=np.asarray(d["gt_pos_tn2"], dtype=np.float32),
        gt_vel_tn2=np.asarray(d["gt_vel_tn2"], dtype=np.float32),
        gt_valid_tn=gt_valid,
        agent_shape_n3=np.asarray(d["agent_shape_n3"], dtype=np.float32),
        dt_chunk_s=float(np.asarray(d["dt_chunk_s"]).item()),
        sdc_index=int(np.asarray(d["sdc_index"]).item()),
        cfg=cfg,
        rollout_start_valid_n=rollout_start_valid,
        include_collision_metrics=False,
    )
    return {k: float(metrics[k]) for k in CORE_REALISM_METRIC_KEYS if k in metrics}


def _try_strict_metrics(
    d: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, float]], str]:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "src" / "scripts" / "parity" / "compare_forward_metrics.py"
    if not script_path.is_file():
        return None, "strict_unavailable"
    try:
        spec = importlib.util.spec_from_file_location("_cbmt_compare_forward_metrics", str(script_path))
        if spec is None or spec.loader is None:
            return None, "strict_unavailable"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        strict_fn = getattr(module, "_strict_metrics_from_artifact", None)
        if strict_fn is None:
            return None, "strict_unavailable"
    except Exception:
        return None, "strict_unavailable"
    cfg = ForwardPassEvalConfig(enabled=True, metric_scope="core_realism")
    try:
        strict, source = strict_fn(d, cfg)
    except Exception:
        return None, "strict_failed"
    return strict, str(source)


def collect_per_scenario_metrics(
    *,
    artifact_index: Mapping[str, Mapping[str, Path]],
    metric_mode: str,
) -> List[Dict[str, Any]]:
    """Collect per-scenario metrics for each model from canonical artifacts."""
    rows: List[Dict[str, Any]] = []
    scenario_ids = sorted({sid for by_sid in artifact_index.values() for sid in by_sid.keys()})
    for sid in scenario_ids:
        for model_id, by_sid in artifact_index.items():
            path = by_sid.get(sid)
            if path is None:
                continue
            with np.load(path, allow_pickle=True) as d:
                approx = _load_approx_metrics(d)
                if approx is None:
                    approx = _recompute_approx_metrics(d)

                strict_metrics: Optional[Dict[str, float]] = None
                strict_source = "not_requested"
                if metric_mode == "strict_if_available":
                    strict_metrics, strict_source = _try_strict_metrics(d)

                row: Dict[str, Any] = {
                    "scenario_id": str(sid),
                    "model_id": str(model_id),
                    "artifact_path": str(path),
                    "strict_source": strict_source,
                }
                for k in CORE_REALISM_METRIC_KEYS:
                    row[f"approx/{k}"] = float(approx.get(k, float("nan")))
                    row[f"strict/{k}"] = float(strict_metrics.get(k, float("nan"))) if strict_metrics else float("nan")
                rows.append(row)
    return rows


def aggregate_metrics(per_scenario_rows: Sequence[Dict[str, Any]], metric_prefix: str = "approx/") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    model_ids = sorted({str(r["model_id"]) for r in per_scenario_rows})
    for mid in model_ids:
        rows = [r for r in per_scenario_rows if str(r["model_id"]) == mid]
        rec: Dict[str, Any] = {
            "model_id": mid,
            "num_scenarios": int(len(rows)),
        }
        for k in CORE_REALISM_METRIC_KEYS:
            vals = [float(r.get(f"{metric_prefix}{k}", float("nan"))) for r in rows]
            rec[k] = _safe_mean(vals)
        out.append(rec)
    return out


def _bootstrap_ci_mean(
    values: np.ndarray,
    *,
    n_boot: int = 500,
    seed: int = 0,
) -> Tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return float("nan"), float("nan")
    samples = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, vals.size, size=vals.size)
        samples.append(float(np.mean(vals[idx])))
    arr = np.asarray(samples, dtype=np.float32)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def pairwise_deltas(
    per_scenario_rows: Sequence[Dict[str, Any]],
    *,
    metric_prefix: str = "approx/",
    seed: int = 0,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    model_ids = sorted({str(r["model_id"]) for r in per_scenario_rows})
    by_model_scenario: Dict[Tuple[str, str], Dict[str, Any]] = {
        (str(r["model_id"]), str(r["scenario_id"])): r for r in per_scenario_rows
    }
    for a, b in itertools.combinations(model_ids, 2):
        sids = sorted(
            set(s for (m, s) in by_model_scenario.keys() if m == a)
            & set(s for (m, s) in by_model_scenario.keys() if m == b)
        )
        rec: Dict[str, Any] = {
            "model_a": a,
            "model_b": b,
            "num_common_scenarios": int(len(sids)),
        }
        for key in ("sfde_min", "sade_min", "fdd"):
            deltas = []
            wins_a = 0
            wins_b = 0
            for sid in sids:
                va = float(by_model_scenario[(a, sid)].get(f"{metric_prefix}{key}", float("nan")))
                vb = float(by_model_scenario[(b, sid)].get(f"{metric_prefix}{key}", float("nan")))
                if not np.isfinite(va) or not np.isfinite(vb):
                    continue
                deltas.append(va - vb)
                if va < vb:
                    wins_a += 1
                elif vb < va:
                    wins_b += 1
            darr = np.asarray(deltas, dtype=np.float32)
            rec[f"delta_{key}_mean"] = float(np.mean(darr)) if darr.size else float("nan")
            h = int(hashlib.sha256(f"{a}|{b}|{key}".encode("utf-8")).hexdigest()[:8], 16)
            lo, hi = _bootstrap_ci_mean(darr, seed=int(seed) + int(h % 100_000))
            rec[f"delta_{key}_ci_low"] = lo
            rec[f"delta_{key}_ci_high"] = hi
            denom = max(1, wins_a + wins_b)
            rec[f"win_rate_{key}_{a}"] = float(wins_a / denom)
            rec[f"win_rate_{key}_{b}"] = float(wins_b / denom)
        out.append(rec)
    return out


def rankings(
    aggregate_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = list(aggregate_rows)
    rows.sort(
        key=lambda r: (
            float(r.get("sfde_min", float("inf"))),
            float(r.get("fdd", float("inf"))),
            float(r.get("sade_min", float("inf"))),
            float(r.get("vel_jsd", float("inf"))),
            float(r.get("ttc_jsd", float("inf"))),
        )
    )
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rows, start=1):
        x = dict(r)
        x["rank"] = int(i)
        out.append(x)
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(r))


def build_artifact_index(
    model_artifact_dirs: Mapping[str, Path],
) -> Dict[str, Dict[str, Path]]:
    out: Dict[str, Dict[str, Path]] = {}
    for model_id, base in model_artifact_dirs.items():
        files = sorted(base.rglob("*.npz"))
        by_sid: Dict[str, Path] = {}
        for p in files:
            try:
                sid = _scenario_id_from_npz(p)
            except Exception:
                continue
            by_sid[str(sid)] = p
        out[str(model_id)] = by_sid
    return out


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
