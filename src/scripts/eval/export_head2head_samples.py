"""Curate and export explore/exploit samples from a head-to-head eval run.

This script:
1) Loads `metrics/per_scenario.csv` from a head-to-head run directory.
2) Ranks scenarios for:
   - exploration-prioritized examples (high diversity with a quality penalty)
   - exploitation/consensus-prioritized examples (low error + low dispersion)
3) Exports selected scenarios as replay packages per mode.
4) Renders GIFs for quick qualitative review.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from counter_bmt.scenario_export import (
    create_dataset_summary,
    create_replay_script,
    export_ground_truth_scenario,
    export_trajectory_only,
)
from counter_bmt.scenarionet_visualizer import ScenarioNetVisualizer
from counter_bmt_v2.eval.compare import build_artifact_index


def _safe_name(x: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in str(x))


def _parse_modes(s: str) -> List[int]:
    vals = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(max(0, int(tok)))
    out: List[int] = []
    seen = set()
    for v in vals:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _zscore(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    mask = np.isfinite(x)
    if not np.any(mask):
        return np.zeros_like(x, dtype=np.float32)
    mu = float(np.mean(x[mask]))
    sd = float(np.std(x[mask]) + 1e-6)
    z = np.zeros_like(x, dtype=np.float32)
    z[mask] = (x[mask] - mu) / sd
    return z


def _require_pillow():
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError("Pillow is required for GIF export. Install with `pip install pillow`.") from exc
    return Image


def _write_gif_from_frames(frame_paths: Sequence[Path], out_path: Path, *, fps: float, loop: int = 0) -> None:
    if not frame_paths:
        raise ValueError("No frame paths provided for GIF export.")
    Image = _require_pillow()
    duration_ms = max(1, int(round(1000.0 / max(0.01, float(fps)))))
    images = [Image.open(str(p)).convert("RGB") for p in frame_paths]
    try:
        images[0].save(
            str(out_path),
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=max(0, int(loop)),
            optimize=False,
        )
    finally:
        for im in images:
            try:
                im.close()
            except Exception:
                pass


def _load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _to_float(x: Any) -> float:
    try:
        y = float(x)
    except Exception:
        return float("nan")
    return y


def _load_report(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "report.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_dataset_dir(report: Mapping[str, Any]) -> str:
    direct = str(report.get("dataset_dir", "")).strip()
    if direct:
        return direct
    cfg = report.get("config")
    if isinstance(cfg, Mapping):
        nested = str(cfg.get("dataset_dir", "")).strip()
        if nested:
            return nested
    return ""


def _resolve_model_id(rows: Sequence[Mapping[str, str]], report: Mapping[str, Any], requested: str) -> str:
    if requested:
        return str(requested)
    rankings = report.get("rankings", [])
    if isinstance(rankings, list) and rankings:
        x = rankings[0]
        if isinstance(x, Mapping) and x.get("model_id"):
            return str(x["model_id"])
    ids = sorted({str(r.get("model_id", "")) for r in rows if str(r.get("model_id", "")).strip()})
    if not ids:
        raise ValueError("No model_id entries found in per_scenario.csv")
    return ids[0]


def _load_scenario_relpath_map(run_dir: Path) -> Dict[str, str]:
    p = run_dir / "scenario_subset.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing scenario subset file: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    entries = data.get("entries", []) if isinstance(data, dict) else []
    out: Dict[str, str] = {}
    for e in entries:
        if not isinstance(e, Mapping):
            continue
        sid = str(e.get("scenario_id", "")).strip()
        rel = str(e.get("relative_path", "")).strip()
        if sid and rel:
            out[sid] = rel
    return out


def _build_scores(rows: Sequence[Mapping[str, str]], *, explore_quality_penalty: float) -> List[Dict[str, Any]]:
    sids = [str(r.get("scenario_id", "")) for r in rows]
    sfde = np.asarray([_to_float(r.get("approx/sfde_min")) for r in rows], dtype=np.float32)
    sade = np.asarray([_to_float(r.get("approx/sade_min")) for r in rows], dtype=np.float32)
    fdd = np.asarray([_to_float(r.get("approx/fdd")) for r in rows], dtype=np.float32)
    add = np.asarray([_to_float(r.get("approx/add")) for r in rows], dtype=np.float32)
    sdd = np.asarray([_to_float(r.get("approx/sdd")) for r in rows], dtype=np.float32)

    z_sfde = _zscore(sfde)
    z_sade = _zscore(sade)
    z_fdd = _zscore(fdd)
    z_add = _zscore(add)
    z_sdd = _zscore(sdd)

    # Exploration: emphasize diversity terms; penalize high displacement error.
    explore_score = 0.4 * z_fdd + 0.4 * z_add + 0.2 * z_sdd - float(explore_quality_penalty) * z_sfde
    # Exploitation/consensus: low error and low spread.
    exploit_score = -0.5 * z_sfde - 0.3 * z_sade - 0.1 * z_add - 0.1 * z_sdd

    out: List[Dict[str, Any]] = []
    for i, sid in enumerate(sids):
        out.append(
            {
                "scenario_id": sid,
                "explore_score": float(explore_score[i]),
                "exploit_score": float(exploit_score[i]),
                "approx/sfde_min": float(sfde[i]),
                "approx/sade_min": float(sade[i]),
                "approx/fdd": float(fdd[i]),
                "approx/add": float(add[i]),
                "approx/sdd": float(sdd[i]),
            }
        )
    return out


def _select_regimes(
    scored: Sequence[Mapping[str, Any]],
    *,
    top_k_explore: int,
    top_k_exploit: int,
    allow_overlap: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_explore = sorted(scored, key=lambda r: float(r.get("explore_score", -1e9)), reverse=True)
    by_exploit = sorted(scored, key=lambda r: float(r.get("exploit_score", -1e9)), reverse=True)

    explore = [dict(x) for x in by_explore[: max(0, int(top_k_explore))]]
    if allow_overlap:
        exploit = [dict(x) for x in by_exploit[: max(0, int(top_k_exploit))]]
    else:
        used = {str(x.get("scenario_id")) for x in explore}
        exploit = []
        for x in by_exploit:
            sid = str(x.get("scenario_id"))
            if sid in used:
                continue
            exploit.append(dict(x))
            if len(exploit) >= max(0, int(top_k_exploit)):
                break
    return explore, exploit


@dataclass
class ExportRecord:
    regime: str
    scenario_id: str
    model_id: str
    mode: int
    replay_dir: str
    replay_pkl: str
    gif_path: str
    score: float


def _export_one_mode(
    *,
    model_id: str,
    scenario_id: str,
    mode: int,
    artifact_npz: Path,
    scenario_file: Path,
    out_dir: Path,
    gif_frames: int,
    gif_fps: float,
    include_ground_truth: bool,
) -> Tuple[Path, Path]:
    with np.load(artifact_npz, allow_pickle=True) as d:
        pred = np.asarray(d["pred_pos_ktn2"], dtype=np.float32)
        mode_i = int(np.clip(int(mode), 0, max(0, pred.shape[0] - 1)))
        ego_xy = np.asarray(pred[mode_i, :, 0, :], dtype=np.float32)

    with scenario_file.open("rb") as f:
        raw = pickle.load(f)
    map_center = raw.get("metadata", {}).get("map_center", None)

    out_dir.mkdir(parents=True, exist_ok=True)
    cf_name = f"sd_counterfactual_1.0_{_safe_name(scenario_id)}_{_safe_name(model_id)}_m{mode_i:02d}.pkl"
    cf_path = out_dir / cf_name
    saved = export_trajectory_only(
        trajectory=ego_xy,
        original_scenario=raw,
        output_path=cf_path,
        intervention_name=f"{model_id}_m{mode_i}",
        original_file_path=scenario_file,
        map_center=map_center,
    )
    if saved is None:
        raise RuntimeError(f"Failed to export trajectory replay for sid={scenario_id} mode={mode_i}")
    saved_paths = [Path(saved)]

    if include_ground_truth:
        gt_path = out_dir / f"sd_counterfactual_1.0_{_safe_name(scenario_id)}_ground_truth.pkl"
        gt_saved = export_ground_truth_scenario(original_file_path=scenario_file, output_path=gt_path)
        if gt_saved is not None:
            saved_paths.append(Path(gt_saved))

    create_dataset_summary(saved_paths, out_dir)
    create_replay_script(saved_paths, out_dir / "replay_scenarios.py")

    viz = ScenarioNetVisualizer(data_dir=str(out_dir))
    try:
        frames_dir = out_dir / "frames"
        images, _traj, sid_rendered = viz.render_scenario(
            scenario_index=0,
            num_frames=max(2, int(gif_frames)),
            output_dir=str(frames_dir),
        )
        frame_paths = [Path(p) for p, _ in images]
        gif_path = out_dir / f"{_safe_name(str(sid_rendered))}_m{mode_i:02d}.gif"
        _write_gif_from_frames(frame_paths, gif_path, fps=float(gif_fps))
    finally:
        viz.close()

    return Path(saved), gif_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Curate explore/exploit samples from a head-to-head run and export replay+GIF.")
    p.add_argument("--run-dir", type=str, required=True, help="Head-to-head run output dir (contains report.json, metrics/, artifacts/).")
    p.add_argument("--output-dir", type=str, default="", help="Output root (default: <run-dir>/curated_samples).")
    p.add_argument("--dataset-dir", type=str, default="", help="Override dataset root (default: from report.json).")
    p.add_argument("--model-id", type=str, default="", help="Target model_id (default: rank-1 model from report).")
    p.add_argument("--top-k-explore", type=int, default=3)
    p.add_argument("--top-k-exploit", type=int, default=3)
    p.add_argument("--allow-overlap", action="store_true", help="Allow same scenarios in explore and exploit sets.")
    p.add_argument("--explore-quality-penalty", type=float, default=0.3, help="Penalty coefficient on sfde_min in explore score.")
    p.add_argument("--modes-explore", type=str, default="0,1,2,3,4,5", help="Comma-separated mode indices to export for explore scenarios.")
    p.add_argument("--modes-exploit", type=str, default="0", help="Comma-separated mode indices to export for exploit scenarios.")
    p.add_argument("--gif-frames", type=int, default=12)
    p.add_argument("--gif-fps", type=float, default=4.0)
    p.add_argument("--include-ground-truth", action="store_true")
    p.add_argument("--output-json", type=str, default="", help="Optional summary JSON path.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    report = _load_report(run_dir)
    dataset_dir = Path(str(args.dataset_dir).strip() or _resolve_dataset_dir(report))
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"dataset dir not found: {dataset_dir}. Provide --dataset-dir or ensure report.json has config.dataset_dir."
        )

    per_csv = run_dir / "metrics" / "per_scenario.csv"
    if not per_csv.is_file():
        raise FileNotFoundError(f"missing per_scenario.csv: {per_csv}")
    rows = _load_rows(per_csv)
    model_id = _resolve_model_id(rows, report, str(args.model_id).strip())
    model_rows = [r for r in rows if str(r.get("model_id", "")) == model_id]
    if not model_rows:
        raise ValueError(f"No rows for model_id={model_id} in {per_csv}")

    scenario_rel = _load_scenario_relpath_map(run_dir)
    artifacts_root = run_dir / "artifacts"
    model_artifact_dirs: Dict[str, Path] = {}
    for p in sorted(artifacts_root.iterdir()):
        if p.is_dir():
            step_eval = p / "step_eval"
            if step_eval.is_dir():
                model_artifact_dirs[p.name] = step_eval
    art_index = build_artifact_index(model_artifact_dirs)
    if model_id not in art_index:
        raise ValueError(f"Artifact directory not found for model_id={model_id} under {artifacts_root}")

    scored = _build_scores(model_rows, explore_quality_penalty=float(args.explore_quality_penalty))
    explore, exploit = _select_regimes(
        scored,
        top_k_explore=int(args.top_k_explore),
        top_k_exploit=int(args.top_k_exploit),
        allow_overlap=bool(args.allow_overlap),
    )

    out_root = Path(str(args.output_dir).strip() or str(run_dir / "curated_samples"))
    out_root.mkdir(parents=True, exist_ok=True)
    explore_modes = _parse_modes(str(args.modes_explore))
    exploit_modes = _parse_modes(str(args.modes_exploit))
    if not explore_modes:
        explore_modes = [0]
    if not exploit_modes:
        exploit_modes = [0]

    exported: List[ExportRecord] = []
    for regime, picks, modes, score_key in (
        ("explore", explore, explore_modes, "explore_score"),
        ("exploit", exploit, exploit_modes, "exploit_score"),
    ):
        for rec in picks:
            sid = str(rec["scenario_id"])
            rel = scenario_rel.get(sid, "")
            ap = art_index.get(model_id, {}).get(sid)
            if not rel or ap is None:
                print(f"[skip] regime={regime} sid={sid}: missing relpath/artifact")
                continue
            scenario_file = (dataset_dir / rel).resolve()
            if not scenario_file.is_file():
                print(f"[skip] regime={regime} sid={sid}: scenario file missing: {scenario_file}")
                continue

            for mode in modes:
                mode_dir = out_root / regime / _safe_name(model_id) / _safe_name(sid) / f"mode_{int(mode):02d}"
                try:
                    replay_pkl, gif_path = _export_one_mode(
                        model_id=model_id,
                        scenario_id=sid,
                        mode=int(mode),
                        artifact_npz=ap,
                        scenario_file=scenario_file,
                        out_dir=mode_dir,
                        gif_frames=int(args.gif_frames),
                        gif_fps=float(args.gif_fps),
                        include_ground_truth=bool(args.include_ground_truth),
                    )
                    exported.append(
                        ExportRecord(
                            regime=regime,
                            scenario_id=sid,
                            model_id=model_id,
                            mode=int(mode),
                            replay_dir=str(mode_dir),
                            replay_pkl=str(replay_pkl),
                            gif_path=str(gif_path),
                            score=float(rec.get(score_key, float("nan"))),
                        )
                    )
                    print(f"[ok] regime={regime} sid={sid} mode={int(mode)} -> {gif_path}")
                except Exception as exc:
                    print(f"[error] regime={regime} sid={sid} mode={int(mode)}: {exc}")

    payload = {
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "model_id": model_id,
        "output_dir": str(out_root),
        "selection": {
            "top_k_explore": int(args.top_k_explore),
            "top_k_exploit": int(args.top_k_exploit),
            "allow_overlap": bool(args.allow_overlap),
            "explore_quality_penalty": float(args.explore_quality_penalty),
            "modes_explore": explore_modes,
            "modes_exploit": exploit_modes,
        },
        "picked": {
            "explore": explore,
            "exploit": exploit,
        },
        "exported": [r.__dict__ for r in exported],
        "counts": {
            "picked_explore": len(explore),
            "picked_exploit": len(exploit),
            "exported_total": len(exported),
        },
    }
    out_json = Path(str(args.output_json).strip()) if str(args.output_json).strip() else (out_root / "summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["counts"], indent=2))
    print(f"Saved summary: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
