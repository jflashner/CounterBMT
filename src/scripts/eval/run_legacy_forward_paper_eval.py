"""Run legacy Adv-BMT forward evaluation with paper-protocol metrics.

This script intentionally reuses the original legacy evaluation stack
(`EvaluationLightningModule` + `Evaluator`) in GPTmodel mode so metrics
match the legacy implementation used in the paper as closely as possible.
"""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Legacy Adv-BMT forward evaluation (paper-protocol evaluator).",
    )
    p.add_argument("--legacy-root", type=str, default="src/Adv-BMT")
    p.add_argument("--checkpoint", type=str, default="bmt/ckpt/last.ckpt")
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

    from bmt.utils import utils  # type: ignore
    from bmt.utils.config import cfg_from_yaml_file, global_config  # type: ignore
    from bmt.dataset.dataset import InfgenDataset  # type: ignore
    from bmt.eval.scenario_evaluator import Evaluator  # type: ignore
    from bmt.eval.evaluate_scenario_metrics import EvaluationLightningModule  # type: ignore

    # Resolve checkpoint path robustly:
    # 1) absolute path
    # 2) as provided relative to cwd
    # 3) relative to legacy_root
    ckpt_arg = Path(args.checkpoint)
    if ckpt_arg.is_absolute():
        ckpt_path = ckpt_arg
    elif ckpt_arg.exists():
        ckpt_path = ckpt_arg.resolve()
    else:
        candidate = legacy_root / ckpt_arg
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
        return obj

    def _apply_eval_overrides(cfg):
        # Mirror paper/evaluator assumptions for forward GPT open-loop eval.
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

    # Load the same config family used in the paper for midgpt.
    config = cfg_from_yaml_file(legacy_root / "cfgs" / "0202_midgpt.yaml", global_config)
    config = _apply_eval_overrides(config)

    model = utils.get_model(checkpoint_path=str(ckpt_path)).eval()
    # Use checkpoint-resolved hyperparameters for evaluator/dataset parity.
    # This avoids shape/config drift (e.g., traffic-light collapsing flags).
    config = _to_easydict(model.hparams)
    # `get_model` also mutates global_config defaults; re-apply runtime overrides.
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
        "mode": "legacy_paper_protocol_forward",
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
