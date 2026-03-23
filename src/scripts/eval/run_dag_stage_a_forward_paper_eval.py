"""Run paper-protocol legacy forward metrics for a Stage-A DAG checkpoint.

This is the DAG-aware sibling of `run_legacy_forward_paper_eval.py`.
It keeps the original evaluator stack and GPTmodel forward-eval protocol, but
loads checkpoints through `MotionLMDAGLatentLightning` so Stage-A additive
checkpoints can be evaluated directly.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys


def _add_import_paths(legacy_root: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    src_root = repo_root / "src"
    for p in (src_root, legacy_root):
        s = str(p.resolve())
        if s not in sys.path:
            sys.path.insert(0, s)

    # Mirror the legacy training entrypoint so local editable clones work when
    # launching from the workspace root.
    for package_root in (repo_root / "metadrive", repo_root / "scenarionet"):
        if package_root.is_dir():
            s = str(package_root.resolve())
            if s not in sys.path:
                sys.path.insert(0, s)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paper-protocol forward evaluation for Stage-A DAG checkpoints.",
    )
    p.add_argument("--legacy-root", type=str, default="src/Adv-BMT")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset-dir", type=str, required=True)
    p.add_argument("--output-prefix", type=str, required=True)
    p.add_argument("--limit-test-batches", type=int, default=500)
    p.add_argument("--num-modes", type=int, default=6)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--sampling-method", type=str, default="topp")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--topp", type=float, default=0.95)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    legacy_root = Path(args.legacy_root).resolve()
    if not legacy_root.is_dir():
        raise FileNotFoundError(f"legacy root not found: {legacy_root}")
    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset dir not found: {dataset_dir}")

    _add_import_paths(legacy_root)

    import torch  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore
    from pytorch_lightning import Trainer  # type: ignore
    from easydict import EasyDict  # type: ignore
    from omegaconf import OmegaConf  # type: ignore

    from bmt.dag_latent.lightning import MotionLMDAGLatentLightning  # type: ignore
    from bmt.dataset.dataset import InfgenDataset  # type: ignore
    from bmt.eval.evaluate_scenario_metrics import EvaluationLightningModule  # type: ignore
    from bmt.eval.scenario_evaluator import Evaluator  # type: ignore
    from bmt.utils.config import cfg_from_yaml_file, global_config  # type: ignore
    from bmt.utils.utils import checkpoint_surgery_func, load_from_checkpoint  # type: ignore

    ckpt_path = Path(args.checkpoint).expanduser()
    if not ckpt_path.is_absolute():
        if ckpt_path.exists():
            ckpt_path = ckpt_path.resolve()
        else:
            candidate = legacy_root / ckpt_path
            if not candidate.exists():
                raise FileNotFoundError(
                    f"checkpoint not found: {args.checkpoint} "
                    f"(also checked {candidate})"
                )
            ckpt_path = candidate.resolve()

    def _get_obj(container, key):
        if isinstance(container, dict):
            return container[key]
        return getattr(container, key)

    def _set_obj(container, key, value):
        if isinstance(container, dict):
            container[key] = value
        else:
            setattr(container, key, value)

    def _to_easydict(obj):
        if isinstance(obj, dict):
            return EasyDict({k: _to_easydict(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [_to_easydict(v) for v in obj]
        if OmegaConf.is_dict(obj):
            return EasyDict({k: _to_easydict(v) for k, v in obj.items()})
        if OmegaConf.is_list(obj):
            return [_to_easydict(v) for v in obj]
        return obj

    def _to_builtin(obj):
        if isinstance(obj, dict):
            return {k: _to_builtin(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_builtin(v) for v in obj]
        if OmegaConf.is_dict(obj):
            return {k: _to_builtin(v) for k, v in obj.items()}
        if OmegaConf.is_list(obj):
            return [_to_builtin(v) for v in obj]
        return obj

    def _apply_eval_overrides(cfg):
        preprocessing = _get_obj(cfg, "PREPROCESSING")
        data_cfg = _get_obj(cfg, "DATA")
        sampling_cfg = _get_obj(cfg, "SAMPLING")

        _set_obj(preprocessing, "keep_all_data", True)
        _set_obj(data_cfg, "TEST_DATA_DIR", str(dataset_dir))
        _set_obj(cfg, "eval_mode", "GPTmodel")
        _set_obj(cfg, "multi_mode", True)
        _set_obj(cfg, "BACKWARD_PREDICTION", True)
        _set_obj(cfg, "eval_backward_model", False)
        _set_obj(sampling_cfg, "SAMPLING_METHOD", str(args.sampling_method))
        _set_obj(sampling_cfg, "TEMPERATURE", float(args.temperature))
        _set_obj(sampling_cfg, "TOPP", float(args.topp))
        return cfg

    default_config_edict = cfg_from_yaml_file(
        legacy_root / "cfgs" / "motion_default.yaml",
        copy.deepcopy(global_config),
    )
    config_edict = cfg_from_yaml_file(
        legacy_root / "cfgs" / "0202_midgpt_dag_stage_a.yaml",
        copy.deepcopy(global_config),
    )
    config_edict = _apply_eval_overrides(config_edict)
    default_config = OmegaConf.create(_to_builtin(default_config_edict))
    config = OmegaConf.create(_to_builtin(config_edict))

    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        cls=MotionLMDAGLatentLightning,
        config=config,
        default_config=default_config,
        strict=True,
        checkpoint_surgery_func=checkpoint_surgery_func,
        map_location=map_location,
    ).eval()

    # Use checkpoint-resolved hparams for dataset/evaluator parity, then
    # reapply the runtime evaluation overrides.
    config = _to_easydict(model.hparams)
    config = _apply_eval_overrides(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    tokenizer = model.model.tokenizer

    evaluator = Evaluator(key_metrics_only=False, start_metrics_only=False)
    dataset = InfgenDataset(config, "test", backward_prediction=False)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda x: x[0],
        num_workers=int(args.num_workers),
    )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    module = EvaluationLightningModule(
        model=model,
        evaluator=evaluator,
        tokenizer=tokenizer,
        config=config,
        dataset=dataset,
        eval_mode="GPTmodel",
        multi_mode=True,
        num_modes=int(args.num_modes),
        backward_TF_mode="all_TF_except_adv",
        save_path=str(output_prefix),
        overwrite_all_agent=False,
        reject_sampling=False,
    )

    trainer = Trainer(
        accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
        devices=1,
        limit_test_batches=int(args.limit_test_batches),
        logger=False,
        enable_checkpointing=False,
    )
    trainer.test(module, dataloaders=dataloader)

    json_path = Path(str(output_prefix) + ".json")
    csv_path = Path(str(output_prefix) + ".csv")
    if not json_path.is_file():
        raise RuntimeError(f"Expected legacy eval output not found: {json_path}")

    metrics = json.loads(json_path.read_text(encoding="utf-8"))
    out = {
        "mode": "stage_a_dag_forward_paper_protocol",
        "legacy_root": str(legacy_root),
        "checkpoint": str(ckpt_path),
        "dataset_dir": str(dataset_dir),
        "limit_test_batches": int(args.limit_test_batches),
        "num_modes": int(args.num_modes),
        "sampling_method": str(args.sampling_method),
        "temperature": float(args.temperature),
        "topp": float(args.topp),
        "metrics_json": str(json_path),
        "metrics_csv": str(csv_path),
        "metrics": metrics,
    }
    summary_path = output_prefix.parent / (output_prefix.name + "_summary.json")
    summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
