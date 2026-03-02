"""v2 model runner for head-to-head evaluation."""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from flax import nnx

from counter_bmt_v2.data import ScenarioNetNNXLoader
from counter_bmt_v2.training.forward_metrics import ForwardPassEvalConfig, compute_forward_pass_metrics_for_batch
from counter_bmt_v2.training.supervised import (
    SupervisedTrainConfig,
    _prepare_supervised_batch,
    _resolve_model_preset,
)
from counter_bmt_v2.trajectory_jax import (
    AdvBMTParityTokenizer,
    BidirectionalMotionTokenizer,
    NNXBidirectionalMotionTransformer,
    ParityTokenizerConfig,
    midgpt_dag_latent_config,
)

from .types import ModelRunResult, ModelSpec, ScenarioSubsetEntry, model_spec_hashable_dict


def _resolve_model_cfg_for_eval(preset_name: str):
    name = str(preset_name)
    if name == "midgpt_dag_latent":
        return midgpt_dag_latent_config()
    return _resolve_model_preset(name)  # type: ignore[arg-type]


def _resolve_checkpoint_path(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / "last.pkl"
    if not p.is_file():
        raise FileNotFoundError(f"v2 checkpoint not found: {p}")
    return p


def _load_v2_checkpoint(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"Invalid v2 checkpoint payload: {path}")
    return payload


def _flatten_step_eval_dir(model_artifact_dir: Path) -> Path:
    root = model_artifact_dir / "step_eval"
    if not root.exists():
        return root
    step_dirs = [p for p in root.glob("step_*") if p.is_dir()]
    if not step_dirs:
        return root
    step_dir = sorted(step_dirs)[-1]
    for npz in step_dir.glob("*.npz"):
        dst = root / npz.name
        if not dst.exists():
            shutil.move(str(npz), str(dst))
    manifest_src = step_dir / "manifest.json"
    if manifest_src.is_file():
        shutil.copy2(str(manifest_src), str(root / "manifest.json"))
    return root


def _artifact_files(path: Path) -> List[Path]:
    return sorted(path.glob("*.npz"))


def _write_summary(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_v2_model(
    *,
    spec: ModelSpec,
    dataset_dir: Path,
    subset: Sequence[ScenarioSubsetEntry],
    out_dir: Path,
    run_seed: int,
    reuse_artifacts: bool,
) -> ModelRunResult:
    model_dir = out_dir / str(spec.id)
    artifact_dir = model_dir / "step_eval"
    summary_path = model_dir / "runner_summary.json"
    cache_meta = model_dir / "artifact_meta.json"
    model_dir.mkdir(parents=True, exist_ok=True)

    subset_rel = [str(s.relative_path) for s in subset]
    subset_ids = [str(s.scenario_id) for s in subset]
    subset_hash = hashlib.sha256(json.dumps(subset_rel, sort_keys=True).encode("utf-8")).hexdigest()
    spec_hash = json.dumps(model_spec_hashable_dict(spec), sort_keys=True)

    if reuse_artifacts and cache_meta.is_file() and artifact_dir.is_dir():
        try:
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        files = _artifact_files(artifact_dir)
        if (
            meta.get("subset_hash") == subset_hash
            and meta.get("spec_hash") == spec_hash
            and int(meta.get("num_artifacts", -1)) == len(subset)
            and len(files) == len(subset)
        ):
            _write_summary(
                summary_path,
                {
                    "model_id": str(spec.id),
                    "backend": "v2",
                    "skipped_inference": True,
                    "reason": "artifact_reuse",
                    "artifact_dir": str(artifact_dir),
                    "num_artifacts": int(len(files)),
                },
            )
            return ModelRunResult(
                model_id=str(spec.id),
                backend="v2",
                artifact_dir=str(artifact_dir),
                summary_path=str(summary_path),
                skipped=False,
            )

    ckpt_path = _resolve_checkpoint_path(spec.checkpoint)
    payload = _load_v2_checkpoint(ckpt_path)
    ckpt_train_cfg = payload.get("train_cfg", {}) if isinstance(payload.get("train_cfg"), dict) else {}
    preset_name = str(spec.runtime.model_preset or ckpt_train_cfg.get("model_preset", "paper_like_small"))
    model_cfg = _resolve_model_cfg_for_eval(preset_name)  # keeps behavior aligned with training presets

    model = NNXBidirectionalMotionTransformer(model_cfg, rngs=nnx.Rngs(run_seed))
    nnx.update(model, payload["model_state"])

    tokenizer_mode = str(spec.runtime.tokenizer_mode or ckpt_train_cfg.get("tokenizer_mode", "paper_simple"))
    skip_steps = int(spec.runtime.skip_steps or ckpt_train_cfg.get("skip_steps", 5))
    if tokenizer_mode == "adv_bmt_parity":
        tokenizer = AdvBMTParityTokenizer(ParityTokenizerConfig(num_skipped_steps=skip_steps))
    else:
        tokenizer = BidirectionalMotionTokenizer(model_cfg.token_space)

    loader = ScenarioNetNNXLoader(
        data_dir=dataset_dir,
        max_agents=int(ckpt_train_cfg.get("max_agents", 128)),
        max_map_features=int(ckpt_train_cfg.get("max_map_features", 512)),
        max_vectors_per_map_feature=int(ckpt_train_cfg.get("max_vectors_per_map_feature", 128)),
        max_traffic_lights=int(ckpt_train_cfg.get("max_traffic_lights", 64)),
        center_to_map=bool(ckpt_train_cfg.get("center_to_map", True)),
    )
    rel_to_idx = {p.relative_to(loader.data_dir).as_posix(): i for i, p in enumerate(loader.files)}

    train_cfg = SupervisedTrainConfig(
        data_dir=str(dataset_dir),
        output_dir=str(model_dir),
        model_preset=preset_name,
        seed=int(run_seed),
        mode="forward",
        reverse_probability=0.0,
        tokenizer_mode=tokenizer_mode,  # type: ignore[arg-type]
        skip_steps=int(skip_steps),
        max_time_steps=int(ckpt_train_cfg.get("max_time_steps", 91)),
        max_agents=int(ckpt_train_cfg.get("max_agents", 128)),
        max_map_features=int(ckpt_train_cfg.get("max_map_features", 512)),
        max_vectors_per_map_feature=int(ckpt_train_cfg.get("max_vectors_per_map_feature", 128)),
        max_traffic_lights=int(ckpt_train_cfg.get("max_traffic_lights", 64)),
        precision="fp32",
    )
    eval_cfg = ForwardPassEvalConfig(
        enabled=True,
        num_modes=int(spec.runtime.num_modes),
        sampling_method=str(spec.runtime.sampling_method),
        temperature=float(spec.runtime.temperature),
        topp=float(spec.runtime.topp),
        metric_scope="core_realism",
        export_artifacts=True,
        artifact_output_subdir="step_eval",
        artifact_max_scenarios_per_eval=64,
        save_visualizations=False,
    )

    rng = np.random.default_rng(int(run_seed))
    per_scenario_metrics: List[Dict[str, Any]] = []
    indexed_subset: List[tuple[int, ScenarioSubsetEntry, int]] = []
    for i, ss in enumerate(subset):
        idx = rel_to_idx.get(str(ss.relative_path))
        if idx is None:
            continue
        indexed_subset.append((int(i), ss, int(idx)))

    # Keep head2head eval memory-bounded; forward rollout can be heavy with relation features.
    batch_size_eval = 2
    for start in range(0, len(indexed_subset), batch_size_eval):
        chunk = indexed_subset[start: start + batch_size_eval]
        samples = [loader.load(int(idx)) for _, _, idx in chunk]
        prepared = _prepare_supervised_batch(
            samples,
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            tokenizer=tokenizer,
            rng=rng,
            is_training=False,
        )
        metrics_list, _, _ = compute_forward_pass_metrics_for_batch(
            model=model,
            prepared_batch=prepared,
            tokenizer=tokenizer,
            skip_steps=int(skip_steps),
            eval_cfg=eval_cfg,
            seed=int(run_seed + start),
            output_dir=model_dir,
            global_step=1,
            max_visualizations=0,
            max_artifacts=len(samples),
        )
        for j, (_orig_i, ss, _idx) in enumerate(chunk):
            rec = dict(metrics_list[j]) if j < len(metrics_list) else {}
            rec["scenario_id"] = str(ss.scenario_id)
            per_scenario_metrics.append(rec)

    flat_dir = _flatten_step_eval_dir(model_dir)
    files = _artifact_files(flat_dir)
    cache_meta.write_text(
        json.dumps(
            {
                "subset_hash": subset_hash,
                "spec_hash": spec_hash,
                "num_artifacts": int(len(files)),
                "scenario_ids": subset_ids,
                "checkpoint": str(ckpt_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_summary(
        summary_path,
        {
            "model_id": str(spec.id),
            "backend": "v2",
            "artifact_dir": str(flat_dir),
            "num_artifacts": int(len(files)),
            "subset_size": int(len(subset)),
            "checkpoint": str(ckpt_path),
            "tokenizer_mode": tokenizer_mode,
            "skip_steps": int(skip_steps),
            "per_scenario_metrics": per_scenario_metrics,
        },
    )
    return ModelRunResult(
        model_id=str(spec.id),
        backend="v2",
        artifact_dir=str(flat_dir),
        summary_path=str(summary_path),
    )
