"""Head-to-head multi-model evaluation orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from counter_bmt_v2.data import ScenarioNetNNXLoader

from .compare import (
    aggregate_metrics,
    build_artifact_index,
    collect_per_scenario_metrics,
    pairwise_deltas,
    rankings,
    write_csv,
    write_json,
)
from .legacy_runner import run_legacy_model
from .replay_export import export_replays_from_artifacts
from .types import (
    Head2HeadConfig,
    MetricsConfig,
    ModelRuntimeConfig,
    ModelRunResult,
    ModelSpec,
    ReplayExportConfig,
    ScenarioSubsetEntry,
    VisualizationConfig,
)
from .visualize import save_overlay_plots, select_scenarios_by_spread
from .v2_runner import run_v2_model


def _require_yaml() -> Any:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise ImportError("PyYAML is required for model registry parsing. Install `pyyaml`.") from exc
    return yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    yaml = _require_yaml()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Registry must parse to mapping, got: {type(data)}")
    return data


def _to_runtime(d: Mapping[str, Any]) -> ModelRuntimeConfig:
    return ModelRuntimeConfig(
        model_preset=(None if d.get("model_preset") in (None, "") else str(d.get("model_preset"))),
        tokenizer_mode=(None if d.get("tokenizer_mode") in (None, "") else str(d.get("tokenizer_mode"))),
        skip_steps=int(d.get("skip_steps", 5)),
        num_modes=int(d.get("num_modes", 6)),
        sampling_method=str(d.get("sampling_method", "topp")),
        topp=float(d.get("topp", 0.95)),
        temperature=float(d.get("temperature", 1.0)),
        python_bin=str(d.get("python_bin", "python")),
        legacy_root=str(d.get("legacy_root", "src/Adv-BMT")),
        config_name=str(d.get("config_name", "")),
        dag_source_mode=str(d.get("dag_source_mode", "none")),
        dag_cache_dir=str(d.get("dag_cache_dir", "")),
        dag_cache_strict=bool(d.get("dag_cache_strict", False)),
        dag_expected_schema=str(d.get("dag_expected_schema", "any")),
    )


def _to_model_spec(d: Mapping[str, Any]) -> ModelSpec:
    return ModelSpec(
        id=str(d["id"]),
        backend=str(d["backend"]),
        checkpoint=str(d["checkpoint"]),
        runtime=_to_runtime(d.get("runtime", {}) if isinstance(d.get("runtime"), dict) else {}),
        enabled=bool(d.get("enabled", True)),
    )


def _parse_registry(registry_path: Path, output_dir_override: str = "") -> Head2HeadConfig:
    raw = _load_yaml(registry_path)
    run = raw.get("run", {}) if isinstance(raw.get("run"), dict) else {}
    models_raw = raw.get("models", [])
    if not isinstance(models_raw, list):
        raise ValueError("`models` must be a list in registry YAML")
    models = [_to_model_spec(m) for m in models_raw if isinstance(m, dict)]
    if not models:
        raise ValueError("No valid models found in registry YAML")

    metrics_raw = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
    viz_raw = run.get("visualization", {}) if isinstance(run.get("visualization"), dict) else {}
    replay_raw = run.get("replay_export", {}) if isinstance(run.get("replay_export"), dict) else {}

    cfg = Head2HeadConfig(
        dataset_dir=str(run["dataset_dir"]),
        output_dir=str(output_dir_override or run.get("output_dir", "outputs/head2head_eval")),
        n_scenarios=int(run.get("n_scenarios", 100)),
        seed=int(run.get("seed", 0)),
        scenario_indices_file=str(run.get("scenario_indices_file", "")),
        metrics=MetricsConfig(mode=str(metrics_raw.get("mode", "approx"))),
        visualization=VisualizationConfig(
            max_scenarios=int(viz_raw.get("max_scenarios", 8)),
            max_agents=int(viz_raw.get("max_agents", 10)),
            selection_policy=str(viz_raw.get("selection_policy", "largest_spread_sfde_min")),
        ),
        replay_export=ReplayExportConfig(
            enabled=bool(replay_raw.get("enabled", True)),
            max_scenarios=int(replay_raw.get("max_scenarios", 8)),
            mode_index=int(replay_raw.get("mode_index", 0)),
            include_ground_truth=bool(replay_raw.get("include_ground_truth", False)),
        ),
        legacy_policy=str(run.get("legacy_policy", "required_if_available")),
        reuse_artifacts=bool(run.get("reuse_artifacts", True)),
        max_parallel_models=int(run.get("max_parallel_models", 1)),
        models=models,
    )
    return cfg


def _probe_legacy(runtime: ModelRuntimeConfig) -> Tuple[bool, str]:
    legacy_root = Path(runtime.legacy_root)
    if not legacy_root.exists():
        return False, f"legacy root not found: {legacy_root}"
    probe = (
        "import sys; "
        f"sys.path.insert(0, r'{legacy_root.resolve()}'); "
        "import bmt; import bmt.utils.utils"
    )
    proc = subprocess.run(
        [str(runtime.python_bin), "-c", probe],
        cwd=str(Path(__file__).resolve().parents[3]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return True, "ok"
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return False, (tail[-1] if tail else f"return_code={proc.returncode}")


def _subset_cache_key(
    *,
    dataset_dir: Path,
    n_scenarios: int,
    seed: int,
    scenario_indices_file: str,
) -> str:
    h = hashlib.sha256()
    h.update(str(dataset_dir.resolve()).encode("utf-8"))
    h.update(str(int(n_scenarios)).encode("utf-8"))
    h.update(str(int(seed)).encode("utf-8"))
    sif = str(scenario_indices_file or "")
    h.update(sif.encode("utf-8"))
    if sif:
        p = Path(sif)
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def _load_subset_from_file(path: Path) -> List[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [int(x) for x in data]
    if isinstance(data, dict) and isinstance(data.get("indices"), list):
        return [int(x) for x in data["indices"]]
    raise ValueError(f"Unsupported subset file format: {path}")


def _resolve_subset(
    *,
    loader: ScenarioNetNNXLoader,
    dataset_dir: Path,
    n_scenarios: int,
    seed: int,
    scenario_indices_file: str,
    cache_dir: Path,
) -> Tuple[List[ScenarioSubsetEntry], Path]:
    cache_key = _subset_cache_key(
        dataset_dir=dataset_dir,
        n_scenarios=int(n_scenarios),
        seed=int(seed),
        scenario_indices_file=str(scenario_indices_file),
    )
    cache_path = cache_dir / f"scenario_subset_{cache_key[:16]}.json"
    if cache_path.is_file():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        entries = [
            ScenarioSubsetEntry(
                rank=int(x["rank"]),
                dataset_index=int(x["dataset_index"]),
                scenario_id=str(x["scenario_id"]),
                relative_path=str(x["relative_path"]),
            )
            for x in data.get("entries", [])
        ]
        if entries:
            return entries, cache_path

    if str(scenario_indices_file).strip():
        idx = _load_subset_from_file(Path(scenario_indices_file))
    else:
        all_idx = np.arange(len(loader), dtype=np.int32)
        rng = np.random.default_rng(int(seed))
        rng.shuffle(all_idx)
        take = min(int(max(1, n_scenarios)), int(len(all_idx)))
        idx = [int(x) for x in all_idx[:take].tolist()]

    entries: List[ScenarioSubsetEntry] = []
    for rank, i in enumerate(idx):
        s = loader.load(int(i))
        rel = loader.files[int(i)].relative_to(loader.data_dir).as_posix()
        entries.append(
            ScenarioSubsetEntry(
                rank=int(rank),
                dataset_index=int(i),
                scenario_id=str(s.scenario_id),
                relative_path=str(rel),
            )
        )
    payload = {"cache_key": cache_key, "entries": [asdict(x) for x in entries]}
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return entries, cache_path


def _write_markdown_report(
    *,
    report_path: Path,
    report: Mapping[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# Head-to-Head Evaluation Report")
    lines.append("")
    lines.append(f"- Output dir: `{report.get('output_dir', '')}`")
    lines.append(f"- Dataset: `{report.get('dataset_dir', '')}`")
    lines.append(f"- Scenarios: `{report.get('num_scenarios', 0)}`")
    lines.append(f"- Metric mode: `{report.get('metric_mode', '')}`")
    lines.append("")
    lines.append("## Model Status")
    for m in report.get("model_runs", []):
        lines.append(
            f"- `{m.get('model_id')}` ({m.get('backend')}): "
            f"{'SKIPPED' if m.get('skipped') else 'OK'} "
            f"{m.get('reason', '')}"
        )
    lines.append("")
    lines.append("## Ranking")
    for r in report.get("rankings", []):
        lines.append(
            f"- #{r.get('rank')} `{r.get('model_id')}` "
            f"sfde_min={r.get('sfde_min', float('nan')):.3f} "
            f"fdd={r.get('fdd', float('nan')):.3f}"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- Per-scenario CSV: `{report.get('paths', {}).get('per_scenario_csv', '')}`")
    lines.append(f"- Aggregate CSV: `{report.get('paths', {}).get('aggregate_csv', '')}`")
    lines.append(f"- Pairwise CSV: `{report.get('paths', {}).get('pairwise_csv', '')}`")
    lines.append(f"- Ranking CSV: `{report.get('paths', {}).get('ranking_csv', '')}`")
    if report.get("paths", {}).get("strict_aggregate_csv"):
        lines.append(f"- Strict aggregate CSV: `{report.get('paths', {}).get('strict_aggregate_csv', '')}`")
    if report.get("paths", {}).get("strict_pairwise_csv"):
        lines.append(f"- Strict pairwise CSV: `{report.get('paths', {}).get('strict_pairwise_csv', '')}`")
    lines.append(f"- Overlay dir: `{report.get('paths', {}).get('viz_dir', '')}`")
    lines.append(f"- Replay dir: `{report.get('paths', {}).get('replay_dir', '')}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_head2head(
    *,
    registry_path: str,
    output_dir: str = "",
    reuse_artifacts: Optional[bool] = None,
) -> Dict[str, Any]:
    registry = Path(registry_path)
    cfg = _parse_registry(registry, output_dir_override=output_dir)
    if reuse_artifacts is not None:
        cfg.reuse_artifacts = bool(reuse_artifacts)

    t0 = time.time()
    dataset_dir = Path(cfg.dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    out_dir = Path(cfg.output_dir)
    artifacts_root = out_dir / "artifacts"
    metrics_dir = out_dir / "metrics"
    viz_dir = out_dir / "viz"
    replay_dir = out_dir / "replay"
    cache_dir = out_dir / "cache"
    logs_dir = out_dir / "logs"
    for d in (artifacts_root, metrics_dir, viz_dir, replay_dir, cache_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    loader = ScenarioNetNNXLoader(data_dir=dataset_dir)
    subset, subset_cache_path = _resolve_subset(
        loader=loader,
        dataset_dir=dataset_dir,
        n_scenarios=int(cfg.n_scenarios),
        seed=int(cfg.seed),
        scenario_indices_file=str(cfg.scenario_indices_file),
        cache_dir=cache_dir,
    )
    subset_path = out_dir / "scenario_subset.json"
    subset_path.write_text(json.dumps({"entries": [asdict(x) for x in subset]}, indent=2), encoding="utf-8")

    model_results: List[ModelRunResult] = []
    artifact_dirs: Dict[str, Path] = {}
    enabled_specs = [m for m in cfg.models if bool(m.enabled)]
    v2_specs = [m for m in enabled_specs if m.backend == "v2"]
    legacy_specs = [m for m in enabled_specs if m.backend == "legacy_adv_bmt"]

    if int(cfg.max_parallel_models) > 1 and len(v2_specs) > 1:
        max_workers = min(int(cfg.max_parallel_models), len(v2_specs))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    run_v2_model,
                    spec=spec,
                    dataset_dir=dataset_dir,
                    subset=subset,
                    out_dir=artifacts_root,
                    run_seed=int(cfg.seed),
                    reuse_artifacts=bool(cfg.reuse_artifacts),
                ): spec
                for spec in v2_specs
            }
            for fut in as_completed(futures):
                spec = futures[fut]
                result = fut.result()
                model_results.append(result)
                artifact_dirs[str(spec.id)] = Path(result.artifact_dir)
    else:
        for spec in v2_specs:
            result = run_v2_model(
                spec=spec,
                dataset_dir=dataset_dir,
                subset=subset,
                out_dir=artifacts_root,
                run_seed=int(cfg.seed),
                reuse_artifacts=bool(cfg.reuse_artifacts),
            )
            model_results.append(result)
            artifact_dirs[str(spec.id)] = Path(result.artifact_dir)

    for spec in legacy_specs:
        is_available, probe_reason = _probe_legacy(spec.runtime)
        required = str(cfg.legacy_policy) == "required"
        if str(cfg.legacy_policy) == "required_if_available":
            required = bool(is_available)
        if not is_available and str(cfg.legacy_policy) in ("required_if_available", "optional"):
            model_results.append(
                ModelRunResult(
                    model_id=str(spec.id),
                    backend="legacy_adv_bmt",
                    artifact_dir=str((artifacts_root / str(spec.id) / "step_eval")),
                    summary_path=str((artifacts_root / str(spec.id) / "runner_summary.json")),
                    skipped=True,
                    reason=f"legacy_unavailable: {probe_reason}",
                )
            )
            continue
        result = run_legacy_model(
            spec=spec,
            dataset_dir=dataset_dir,
            subset_file=subset_path,
            subset=subset,
            out_dir=artifacts_root,
            run_seed=int(cfg.seed),
            reuse_artifacts=bool(cfg.reuse_artifacts),
            required=bool(required),
        )
        model_results.append(result)
        if not result.skipped:
            artifact_dirs[str(spec.id)] = Path(result.artifact_dir)

    if len(artifact_dirs) < 1:
        raise RuntimeError("No runnable models produced artifacts")

    artifact_index = build_artifact_index(artifact_dirs)
    per_scenario_rows = collect_per_scenario_metrics(
        artifact_index=artifact_index,
        metric_mode=str(cfg.metrics.mode),
    )
    aggregate_rows = aggregate_metrics(per_scenario_rows, metric_prefix="approx/")
    pairwise_rows = pairwise_deltas(per_scenario_rows, metric_prefix="approx/", seed=int(cfg.seed))
    ranking_rows = rankings(aggregate_rows)
    strict_aggregate_rows: List[Dict[str, Any]] = []
    strict_pairwise_rows: List[Dict[str, Any]] = []
    if str(cfg.metrics.mode) == "strict_if_available":
        strict_aggregate_rows = aggregate_metrics(per_scenario_rows, metric_prefix="strict/")
        strict_pairwise_rows = pairwise_deltas(per_scenario_rows, metric_prefix="strict/", seed=int(cfg.seed) + 17)

    per_scenario_csv = metrics_dir / "per_scenario.csv"
    aggregate_csv = metrics_dir / "aggregate.csv"
    pairwise_csv = metrics_dir / "pairwise_deltas.csv"
    ranking_csv = metrics_dir / "rankings.csv"
    strict_aggregate_csv = metrics_dir / "strict_aggregate.csv"
    strict_pairwise_csv = metrics_dir / "strict_pairwise_deltas.csv"
    write_csv(per_scenario_csv, per_scenario_rows)
    write_csv(aggregate_csv, aggregate_rows)
    write_csv(pairwise_csv, pairwise_rows)
    write_csv(ranking_csv, ranking_rows)
    if strict_aggregate_rows:
        write_csv(strict_aggregate_csv, strict_aggregate_rows)
    if strict_pairwise_rows:
        write_csv(strict_pairwise_csv, strict_pairwise_rows)

    selected_for_viz = select_scenarios_by_spread(
        per_scenario_rows,
        metric_key="approx/sfde_min",
        max_scenarios=int(cfg.visualization.max_scenarios),
    )
    overlay_paths = save_overlay_plots(
        selected_scenarios=selected_for_viz,
        artifact_index=artifact_index,
        out_dir=viz_dir,
        max_agents=int(cfg.visualization.max_agents),
    )

    scenario_relpath_by_id = {str(s.scenario_id): str(s.relative_path) for s in subset}
    selected_for_replay = selected_for_viz[: max(0, int(cfg.replay_export.max_scenarios))]
    replay_exports: Dict[str, List[str]] = {}
    if bool(cfg.replay_export.enabled) and selected_for_replay:
        replay_exports = export_replays_from_artifacts(
            artifact_index=artifact_index,
            dataset_dir=dataset_dir,
            scenario_relpath_by_id=scenario_relpath_by_id,
            out_dir=replay_dir,
            selected_scenarios=selected_for_replay,
            mode_index=int(cfg.replay_export.mode_index),
            include_ground_truth=bool(cfg.replay_export.include_ground_truth),
        )

    report = {
        "suite": "head2head.v1",
        "timestamp_unix": int(time.time()),
        "registry_path": str(registry),
        "output_dir": str(out_dir),
        "dataset_dir": str(dataset_dir),
        "num_scenarios": int(len(subset)),
        "metric_mode": str(cfg.metrics.mode),
        "legacy_policy": str(cfg.legacy_policy),
        "reuse_artifacts": bool(cfg.reuse_artifacts),
        "subset_cache_path": str(subset_cache_path),
        "scenario_subset_path": str(subset_path),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "config": asdict(cfg),
        "model_runs": [asdict(x) for x in model_results],
        "aggregate": aggregate_rows,
        "pairwise": pairwise_rows,
        "strict_aggregate": strict_aggregate_rows,
        "strict_pairwise": strict_pairwise_rows,
        "rankings": ranking_rows,
        "selected_for_viz": selected_for_viz,
        "overlay_paths": overlay_paths,
        "replay_exports": replay_exports,
        "paths": {
            "per_scenario_csv": str(per_scenario_csv),
            "aggregate_csv": str(aggregate_csv),
            "pairwise_csv": str(pairwise_csv),
            "ranking_csv": str(ranking_csv),
            "strict_aggregate_csv": str(strict_aggregate_csv) if strict_aggregate_rows else "",
            "strict_pairwise_csv": str(strict_pairwise_csv) if strict_pairwise_rows else "",
            "viz_dir": str(viz_dir),
            "replay_dir": str(replay_dir),
        },
        "elapsed_seconds": float(time.time() - t0),
    }

    report_json = out_dir / "report.json"
    report_md = out_dir / "report.md"
    write_json(report_json, report)
    _write_markdown_report(report_path=report_md, report=report)
    return report
