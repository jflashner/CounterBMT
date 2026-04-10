import datetime
import ast
import json
import os
import pathlib
import subprocess
import sys
import traceback

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities.model_summary import summarize
from omegaconf import OmegaConf

import bmt.utils as utils
from bmt.dataset.datamodule import InfgenDataModule
from bmt.models.motionlm_lightning import MotionLMLightning
from bmt.utils.checkpoint_loading import load_model_from_checkpoint_forgiving
from bmt.utils import REPO_ROOT, get_time_str

torch.set_float32_matmul_precision('high')


def _wrap_tensorboard_hparams_logger(logger):
    if not isinstance(logger, TensorBoardLogger):
        return logger

    original_log_hyperparams = logger.log_hyperparams

    def _safe_log_hyperparams(*args, **kwargs):
        try:
            return original_log_hyperparams(*args, **kwargs)
        except AttributeError as exc:
            if "np.string_" not in str(exc):
                raise
            print("Skipping TensorBoard hparams logging due to NumPy/TensorBoard compatibility issue:", exc)
            return None

    logger.log_hyperparams = _safe_log_hyperparams
    return logger


def _resolve_wandb_api_key() -> str:
    api_key = str(os.environ.get("WANDB_API_KEY", "")).strip()
    if api_key:
        return api_key

    api_key_file = os.path.abspath(
        os.path.expanduser(
            str(os.environ.get("WANDB_API_KEY_FILE", "~/wandb_api_key_file.txt"))
        )
    )
    if not os.path.isfile(api_key_file):
        raise FileNotFoundError(
            "WandB logging is enabled, but no API key was found. "
            "Set WANDB_API_KEY or WANDB_API_KEY_FILE."
        )
    with open(api_key_file, "rt", encoding="utf-8") as fp:
        api_key = fp.readline().strip()
    if not api_key:
        raise RuntimeError(f"WandB API key file is empty: {api_key_file}")
    return api_key


def _serialize_metric_mapping(values):
    serialized = {}
    for key, value in dict(values).items():
        if hasattr(value, "detach"):
            tensor = value.detach().cpu()
            if tensor.numel() == 1:
                serialized[str(key)] = float(tensor.item())
            else:
                serialized[str(key)] = tensor.tolist()
        elif isinstance(value, (int, float, bool, str)):
            serialized[str(key)] = value
        elif value is None:
            serialized[str(key)] = None
        else:
            serialized[str(key)] = str(value)
    return serialized


def _resolve_configured_path(path_text: str) -> pathlib.Path:
    path = pathlib.Path(str(path_text).strip()).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _prepare_counterfactual_train_index(config) -> None:
    counterfactual_mode = str(OmegaConf.select(config, "DATA.COUNTERFACTUAL_MODE", default="") or "").strip()
    if counterfactual_mode != "sdc_semantic_only":
        return

    rollout_objective_enabled = any(
        float(OmegaConf.select(config, key, default=0.0) or 0.0) > 0.0
        for key in (
            "MODEL.LOCAL_CONTROL_SDC_ROLLOUT_GUIDE_LOSS_WEIGHT",
            "MODEL.LOCAL_CONTROL_SDC_ROLLOUT_PROGRESS_LOSS_WEIGHT",
            "MODEL.LOCAL_CONTROL_SDC_ROLLOUT_TUBE_POLICY_LOSS_WEIGHT",
        )
    )
    if not rollout_objective_enabled:
        return

    requested_alt_only = bool(OmegaConf.select(config, "DATA.COUNTERFACTUAL_ALT_ONLY_TRAIN", default=False))
    multi_gpu_rollout_rl = torch.cuda.is_available() and torch.cuda.device_count() > 1
    if not (requested_alt_only or multi_gpu_rollout_rl):
        return

    source_key = "DATA.COUNTERFACTUAL_CONTROL_INDEX_TRAIN"
    source_value = str(OmegaConf.select(config, source_key, default="") or "").strip()
    if not source_value:
        source_key = "DATA.COUNTERFACTUAL_CONTROL_INDEX"
        source_value = str(OmegaConf.select(config, source_key, default="") or "").strip()
    if not source_value:
        return

    source_path = _resolve_configured_path(source_value)
    if not source_path.is_file() or source_path.suffix != ".jsonl":
        return

    if source_path.stem.endswith("_altonly"):
        filtered_path = source_path
    else:
        filtered_path = source_path.with_name(f"{source_path.stem}_altonly.jsonl")

    total_rows = 0
    kept_rows = 0
    removed_rows = 0
    with source_path.open("rt", encoding="utf-8") as src, filtered_path.open("wt", encoding="utf-8") as dst:
        for line in src:
            text = line.strip()
            if not text:
                continue
            total_rows += 1
            row = json.loads(text)
            slot_id = str(row.get("selected_slot_id") or row.get("slot_id") or "").strip()
            if slot_id == "gt":
                removed_rows += 1
                continue
            kept_rows += 1
            dst.write(json.dumps(row, sort_keys=True))
            dst.write("\n")

    if kept_rows <= 0:
        raise RuntimeError(
            f"Alt-only counterfactual train index is empty after filtering: {source_path}"
        )

    OmegaConf.set_struct(config, False)
    config.DATA.COUNTERFACTUAL_CONTROL_INDEX_TRAIN = str(filtered_path)
    OmegaConf.set_struct(config, True)

    reason = "explicit config" if requested_alt_only else "automatic multi-GPU rollout-RL safeguard"
    print(
        "[counterfactual-train-index] using alt-only train index",
        {
            "reason": reason,
            "source": str(source_path),
            "filtered": str(filtered_path),
            "total_rows": total_rows,
            "kept_rows": kept_rows,
            "removed_gt_rows": removed_rows,
        },
    )


def _write_training_artifacts(
    *,
    artifact_root: pathlib.Path,
    config,
    trainer: pl.Trainer,
    run_name: str,
    exp_name: str,
    ckpt_path: str | None,
    pretrained_path: str | None,
    checkpoint_load_report,
    completed: bool,
    failed: bool,
    failure_reason,
):
    artifact_root.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_root / "path_control_train_summary.json"
    metrics_path = artifact_root / "path_control_train_metrics.json"

    summary = {
        "run_name": str(run_name),
        "exp_name": str(exp_name),
        "completed": bool(completed),
        "failed": bool(failed),
        "failure_reason": failure_reason,
        "global_step": int(getattr(trainer, "global_step", 0)),
        "current_epoch": int(getattr(trainer, "current_epoch", 0)),
        "ckpt_path": ckpt_path,
        "pretrained_path": pretrained_path,
        "ckpt_load_mode": str(config.get("CKPT_LOAD_MODE", "legacy_merge")),
        "checkpoint_load_report": checkpoint_load_report,
        "wandb_enabled": bool(config.wandb and not config.eval),
        "wandb_project": str(os.environ.get("WANDB_PROJECT", "infgen")).strip() or "infgen",
        "wandb_entity": str(os.environ.get("WANDB_ENTITY", "")).strip() or None,
        "wandb_group": str(os.environ.get("WANDB_GROUP", exp_name)).strip() or exp_name,
        "wandb_run_name": str(os.environ.get("WANDB_RUN_NAME", run_name)).strip() or run_name,
        "log_dir": str(artifact_root),
        "lightning_log_dir": str((artifact_root / "lightning_logs").resolve()) if artifact_root.name != "lightning_logs" else str(artifact_root.resolve()),
    }

    metrics = {
        "callback_metrics": _serialize_metric_mapping(getattr(trainer, "callback_metrics", {})),
        "logged_metrics": _serialize_metric_mapping(getattr(trainer, "logged_metrics", {})),
        "progress_bar_metrics": _serialize_metric_mapping(getattr(trainer, "progress_bar_metrics", {})),
    }

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


class SemanticRolloutGifCallback(Callback):
    def __init__(self, *, config, artifact_root: pathlib.Path):
        self.config = config
        self.artifact_root = pathlib.Path(artifact_root)
        self.enabled = bool(config.get("ROLLOUT_GIF_EVAL_ENABLED", False))
        self.every_n_validations = max(1, int(config.get("ROLLOUT_GIF_EVAL_EVERY_N_VALIDATIONS", 1)))
        self.max_scenes = max(1, int(config.get("ROLLOUT_GIF_EVAL_MAX_SCENES", 1)))
        self.num_samples = max(1, int(config.get("ROLLOUT_GIF_EVAL_NUM_SAMPLES", 6)))
        self.rollout_gif_fps = float(config.get("ROLLOUT_GIF_EVAL_FPS", 6.0))
        self.rollout_device = str(config.get("ROLLOUT_GIF_EVAL_DEVICE", "cpu")).strip() or "cpu"
        self.all_scene_slots = bool(config.get("ROLLOUT_GIF_EVAL_ALL_SCENE_SLOTS", True))
        self.scenario_ids = self._normalize_text_list(config.get("ROLLOUT_GIF_EVAL_SCENARIO_IDS", []))
        self.control_index_override = str(config.get("ROLLOUT_GIF_EVAL_CONTROL_INDEX", "")).strip()
        self.data_dir_override = str(config.get("ROLLOUT_GIF_EVAL_DATA_DIR", "")).strip()
        self.rollout_sampling_method = str(
            config.get("ROLLOUT_GIF_EVAL_ROLLOUT_SAMPLING_METHOD", "argmax")
        ).strip() or "argmax"
        self.rollout_temperature = float(config.get("ROLLOUT_GIF_EVAL_ROLLOUT_TEMPERATURE", -1.0))
        self.rollout_topp = float(config.get("ROLLOUT_GIF_EVAL_ROLLOUT_TOPP", -1.0))
        self._validation_events = 0
        self.legacy_root = pathlib.Path(REPO_ROOT).resolve()
        self.repo_root = self.legacy_root.parent.parent
        self.eval_script = self.repo_root / "scripts" / "counterfactual" / "eval_sdc_semantic_action_projections.py"

    @staticmethod
    def _normalize_text_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = ast.literal_eval(text)
                except Exception:
                    inner = text[1:-1]
                    parts = [item.strip().strip("'\"") for item in inner.split(",")]
                    return [item for item in parts if item]
                return SemanticRolloutGifCallback._normalize_text_list(parsed)
            parts = [item.strip().strip("'\"") for item in text.split(",")]
            return [item for item in parts if item]
        if isinstance(value, (list, tuple)):
            items = []
            for item in value:
                text = str(item).strip()
                if text:
                    items.append(text)
            return items
        text = str(value).strip()
        return [text] if text else []

    def _resolve_control_index(self):
        if self.control_index_override:
            return pathlib.Path(self.control_index_override).expanduser().resolve()
        for key in (
            "DATA.COUNTERFACTUAL_CONTROL_INDEX_VAL",
            "DATA.COUNTERFACTUAL_CONTROL_INDEX_TRAIN",
            "DATA.COUNTERFACTUAL_CONTROL_INDEX",
        ):
            value = str(OmegaConf.select(self.config, key, default="") or "").strip()
            if value:
                return pathlib.Path(value).expanduser().resolve()
        raise FileNotFoundError("No control index configured for rollout GIF eval.")

    def _resolve_data_dir(self):
        if self.data_dir_override:
            return pathlib.Path(self.data_dir_override).expanduser().resolve()
        for key in ("DATA.TEST_DATA_DIR", "DATA.TRAINING_DATA_DIR"):
            value = str(OmegaConf.select(self.config, key, default="") or "").strip()
            if value:
                return pathlib.Path(value).expanduser().resolve()
        raise FileNotFoundError("No data dir configured for rollout GIF eval.")

    def _select_scenario_ids(self, control_index: pathlib.Path):
        if self.scenario_ids:
            return self.scenario_ids[: self.max_scenes]
        selected = []
        seen = set()
        with control_index.open("rt", encoding="utf-8") as fp:
            for line in fp:
                text = line.strip()
                if not text:
                    continue
                scenario_id = str(json.loads(text).get("scenario_id") or "").strip()
                if not scenario_id or scenario_id in seen:
                    continue
                selected.append(scenario_id)
                seen.add(scenario_id)
                if len(selected) >= self.max_scenes:
                    break
        if not selected:
            raise RuntimeError(f"No scenario ids found in control index: {control_index}")
        return selected

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.enabled or not trainer.is_global_zero or trainer.sanity_checking:
            return

        self._validation_events += 1
        if (self._validation_events % self.every_n_validations) != 0:
            return

        try:
            control_index = self._resolve_control_index()
            data_dir = self._resolve_data_dir()
            scenario_ids = self._select_scenario_ids(control_index)
            eval_root = self.artifact_root / "rollout_gif_eval" / f"step_{int(trainer.global_step):06d}"
            eval_root.mkdir(parents=True, exist_ok=True)

            cfg_path = eval_root / "resolved_train_config.yaml"
            OmegaConf.save(config=self.config, f=str(cfg_path))

            ckpt_path = eval_root / "_snapshot.ckpt"
            trainer.save_checkpoint(str(ckpt_path))

            summary = {
                "global_step": int(trainer.global_step),
                "current_epoch": int(trainer.current_epoch),
                "scenario_ids": list(scenario_ids),
                "control_index": str(control_index),
                "data_dir": str(data_dir),
                "device": self.rollout_device,
                "results": [],
            }
            env = dict(os.environ)
            py_path = env.get("PYTHONPATH", "")
            repo_root_text = str(self.repo_root)
            env["PYTHONPATH"] = (
                f"{repo_root_text}:{py_path}" if py_path and repo_root_text not in py_path.split(":") else (py_path or repo_root_text)
            )

            for scenario_id in scenario_ids:
                scenario_outdir = eval_root / scenario_id
                scenario_outdir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    str(self.eval_script),
                    "--config",
                    str(cfg_path),
                    "--control-index",
                    str(control_index),
                    "--data-dir",
                    str(data_dir),
                    "--ckpt",
                    str(ckpt_path),
                    "--outdir",
                    str(scenario_outdir),
                    "--scenario-id",
                    str(scenario_id),
                    "--num-samples",
                    str(self.num_samples),
                    "--device",
                    str(self.rollout_device),
                    "--autoregressive-rollout",
                    "--rollout-sampling-method",
                    str(self.rollout_sampling_method),
                    "--rollout-temperature",
                    str(self.rollout_temperature),
                    "--rollout-topp",
                    str(self.rollout_topp),
                    "--save-rollout-gif",
                    "--rollout-gif-fps",
                    str(self.rollout_gif_fps),
                ]
                if self.all_scene_slots:
                    cmd.append("--all-scene-slots")
                result = subprocess.run(
                    cmd,
                    cwd=str(self.repo_root),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                log_path = scenario_outdir / "_callback_eval.log"
                log_path.write_text(result.stdout, encoding="utf-8")
                scenario_summary = {
                    "scenario_id": str(scenario_id),
                    "returncode": int(result.returncode),
                    "output_dir": str(scenario_outdir),
                    "log_path": str(log_path),
                }
                summary["results"].append(scenario_summary)
                if result.returncode != 0:
                    print(
                        f"[rollout-gif-eval] scenario={scenario_id} failed with return code {result.returncode}. "
                        f"See {log_path}"
                    )
                else:
                    print(
                        f"[rollout-gif-eval] saved autoregressive rollout GIFs for {scenario_id} to {scenario_outdir}"
                    )

            summary_path = eval_root / "callback_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            try:
                ckpt_path.unlink()
            except FileNotFoundError:
                pass
        except Exception as exc:
            print(f"[rollout-gif-eval] callback failed: {exc}")
            traceback.print_exc()


class SemanticRolloutAdvantageCallback(SemanticRolloutGifCallback):
    def __init__(self, *, config, artifact_root: pathlib.Path):
        super().__init__(config=config, artifact_root=artifact_root)
        self.enabled = bool(config.get("ROLLOUT_ADVANTAGE_EVAL_ENABLED", False))
        self.every_n_validations = max(1, int(config.get("ROLLOUT_ADVANTAGE_EVAL_EVERY_N_VALIDATIONS", 1)))
        self.max_scenes = max(1, int(config.get("ROLLOUT_ADVANTAGE_EVAL_MAX_SCENES", 1)))
        self.include_gt = bool(config.get("ROLLOUT_ADVANTAGE_EVAL_INCLUDE_GT", False))
        self.scenario_ids = self._normalize_text_list(config.get("ROLLOUT_ADVANTAGE_EVAL_SCENARIO_IDS", []))
        self.slot_ids = self._normalize_text_list(config.get("ROLLOUT_ADVANTAGE_EVAL_SLOT_IDS", []))
        self.control_index_override = str(config.get("ROLLOUT_ADVANTAGE_EVAL_CONTROL_INDEX", "")).strip()
        self.data_dir_override = str(config.get("ROLLOUT_ADVANTAGE_EVAL_DATA_DIR", "")).strip()
        self.num_rollouts = max(1, int(config.get("ROLLOUT_ADVANTAGE_EVAL_NUM_ROLLOUTS", 8)))
        self.eval_device = str(config.get("ROLLOUT_ADVANTAGE_EVAL_DEVICE", "cpu")).strip() or "cpu"
        self.sampling_method = str(config.get("ROLLOUT_ADVANTAGE_EVAL_SAMPLING_METHOD", "softmax")).strip() or "softmax"
        self.temperature = float(config.get("ROLLOUT_ADVANTAGE_EVAL_TEMPERATURE", 1.0))
        self.topp = float(config.get("ROLLOUT_ADVANTAGE_EVAL_TOPP", 0.9))
        self.tube_radius_m = float(config.get("ROLLOUT_ADVANTAGE_EVAL_TUBE_RADIUS_M", 3.0))
        self.inside_reward = float(config.get("ROLLOUT_ADVANTAGE_EVAL_INSIDE_REWARD", 1.0))
        self.outside_scale = float(config.get("ROLLOUT_ADVANTAGE_EVAL_OUTSIDE_SCALE", 1.0))
        self.discount = float(config.get("ROLLOUT_ADVANTAGE_EVAL_DISCOUNT", 1.0))
        self.grid_step_m = float(config.get("ROLLOUT_ADVANTAGE_EVAL_GRID_STEP_M", 0.35))
        self.jump_threshold_m = float(config.get("ROLLOUT_ADVANTAGE_EVAL_JUMP_THRESHOLD_M", 6.0))
        self.seed = int(config.get("ROLLOUT_ADVANTAGE_EVAL_SEED", 0))
        self.eval_script = self.repo_root / "scripts" / "counterfactual" / "analyze_sdc_semantic_group_rollout_advantages.py"

    def _select_slot_ids(self, control_index: pathlib.Path, *, scenario_id: str):
        if self.slot_ids:
            if self.include_gt:
                return self.slot_ids
            return [slot_id for slot_id in self.slot_ids if str(slot_id) != "gt"]

        matches = []
        with control_index.open("rt", encoding="utf-8") as fp:
            for line in fp:
                text = line.strip()
                if not text:
                    continue
                row = json.loads(text)
                if str(row.get("scenario_id") or "") != str(scenario_id):
                    continue
                slot_id = str(row.get("selected_slot_id") or "").strip()
                if not slot_id:
                    continue
                if not self.include_gt and slot_id == "gt":
                    continue
                matches.append(slot_id)

        def _slot_sort_key(slot_id: str):
            if slot_id == "gt":
                return (0, slot_id)
            if slot_id.startswith("alt_"):
                suffix = slot_id.split("_", 1)[-1]
                if suffix.isdigit():
                    return (1, f"{int(suffix):04d}")
            return (2, slot_id)

        ordered = []
        seen = set()
        for slot_id in sorted(matches, key=_slot_sort_key):
            if slot_id in seen:
                continue
            ordered.append(slot_id)
            seen.add(slot_id)
        return ordered

    @staticmethod
    def _sanitize_name(text):
        value = str(text).strip()
        if not value:
            return "item"
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.enabled or not trainer.is_global_zero or trainer.sanity_checking:
            return

        self._validation_events += 1
        if (self._validation_events % self.every_n_validations) != 0:
            return

        try:
            control_index = self._resolve_control_index()
            data_dir = self._resolve_data_dir()
            scenario_ids = self._select_scenario_ids(control_index)
            eval_root = self.artifact_root / "rollout_advantage_eval" / f"step_{int(trainer.global_step):06d}"
            eval_root.mkdir(parents=True, exist_ok=True)

            cfg_path = eval_root / "resolved_train_config.yaml"
            OmegaConf.save(config=self.config, f=str(cfg_path))

            ckpt_path = eval_root / "_snapshot.ckpt"
            trainer.save_checkpoint(str(ckpt_path))

            summary = {
                "global_step": int(trainer.global_step),
                "current_epoch": int(trainer.current_epoch),
                "scenario_ids": list(scenario_ids),
                "control_index": str(control_index),
                "data_dir": str(data_dir),
                "device": self.eval_device,
                "results": [],
            }
            env = dict(os.environ)
            py_path = env.get("PYTHONPATH", "")
            repo_root_text = str(self.repo_root)
            env["PYTHONPATH"] = (
                f"{repo_root_text}:{py_path}" if py_path and repo_root_text not in py_path.split(":") else (py_path or repo_root_text)
            )

            for scenario_id in scenario_ids:
                slot_ids = self._select_slot_ids(control_index, scenario_id=str(scenario_id))
                if not slot_ids:
                    print(f"[rollout-advantage-eval] no slot ids resolved for scenario={scenario_id}")
                    continue
                for slot_id in slot_ids:
                    slot_outdir = eval_root / self._sanitize_name(str(scenario_id)) / self._sanitize_name(str(slot_id))
                    slot_outdir.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        sys.executable,
                        str(self.eval_script),
                        "--config",
                        str(cfg_path),
                        "--control-index",
                        str(control_index),
                        "--data-dir",
                        str(data_dir),
                        "--ckpt",
                        str(ckpt_path),
                        "--outdir",
                        str(slot_outdir),
                        "--scenario-id",
                        str(scenario_id),
                        "--slot-id",
                        str(slot_id),
                        "--num-rollouts",
                        str(self.num_rollouts),
                        "--tube-radius-m",
                        str(self.tube_radius_m),
                        "--inside-reward",
                        str(self.inside_reward),
                        "--outside-scale",
                        str(self.outside_scale),
                        "--discount",
                        str(self.discount),
                        "--sampling-method",
                        str(self.sampling_method),
                        "--temperature",
                        str(self.temperature),
                        "--topp",
                        str(self.topp),
                        "--seed",
                        str(self.seed),
                        "--device",
                        str(self.eval_device),
                        "--grid-step-m",
                        str(self.grid_step_m),
                        "--jump-threshold-m",
                        str(self.jump_threshold_m),
                    ]
                    result = subprocess.run(
                        cmd,
                        cwd=str(self.repo_root),
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                    log_path = slot_outdir / "_callback_advantage_eval.log"
                    log_path.write_text(result.stdout, encoding="utf-8")
                    slot_summary = {
                        "scenario_id": str(scenario_id),
                        "slot_id": str(slot_id),
                        "returncode": int(result.returncode),
                        "output_dir": str(slot_outdir),
                        "log_path": str(log_path),
                    }
                    summary["results"].append(slot_summary)
                    if result.returncode != 0:
                        print(
                            f"[rollout-advantage-eval] scenario={scenario_id} slot={slot_id} failed "
                            f"with return code {result.returncode}. See {log_path}"
                        )
                    else:
                        print(
                            f"[rollout-advantage-eval] saved grouped rollout analysis for "
                            f"{scenario_id}/{slot_id} to {slot_outdir}"
                        )

            summary_path = eval_root / "callback_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            try:
                ckpt_path.unlink()
            except FileNotFoundError:
                pass
        except Exception as exc:
            print(f"[rollout-advantage-eval] callback failed: {exc}")
            traceback.print_exc()


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "cfgs"), config_name="motion_default.yaml")
def main(config):
    # Unfreeze the config to allow modification
    OmegaConf.set_struct(config, False)
    config.ROOT_DIR = REPO_ROOT
    OmegaConf.set_struct(config, True)

    from bmt.utils.config import global_config, cfg_from_yaml_file
    default_config = cfg_from_yaml_file(REPO_ROOT / "cfgs/motion_default.yaml", global_config)

    pl.seed_everything(config.seed)
    print("Everything is seeded to: ", config.seed)

    # Set up config
    # cfg_file = REPO_ROOT / config.cfg_file
    # config = cfg_from_yaml_file(cfg_file, global_config)
    exp_name = config.exp_name
    max_epochs = config.epochs  #or config.OPTIMIZATION.NUM_EPOCHS
    max_steps = int(config.get("max_steps", -1))
    val_interval = config.get("val_interval", None)
    batch_size = config.batch_size
    val_batch_size = config.val_batch_size
    num_workers = config.num_workers
    val_num_workers = config.val_num_workers
    log_dir = config.log_dir or None
    if log_dir is not None:
        log_dir = pathlib.Path(log_dir)

    _prepare_counterfactual_train_index(config)

    # Setup wandb logger
    trial_id = get_time_str(no_time=True)
    name = "{}_{}".format(exp_name, trial_id)
    if log_dir:
        save_dir = pathlib.Path(log_dir / "lightning_logs")
    else:
        save_dir = pathlib.Path(os.path.join(REPO_ROOT, "lightning_logs"))
    if config.wandb and not config.eval:
        import wandb

        api_key = _resolve_wandb_api_key()
        wandb_project = str(os.environ.get("WANDB_PROJECT", "infgen")).strip() or "infgen"
        wandb_entity = str(os.environ.get("WANDB_ENTITY", "")).strip() or None
        wandb_group = str(os.environ.get("WANDB_GROUP", exp_name)).strip() or exp_name
        wandb_run_name = str(os.environ.get("WANDB_RUN_NAME", name)).strip() or name
        wandb.login(key=api_key)
        logger = WandbLogger(
            name=wandb_run_name,
            save_dir=save_dir,
            id=wandb_run_name,
            project=wandb_project,
            entity=wandb_entity,
            log_model=False,
            group=wandb_group,
        )
    else:
        logger = TensorBoardLogger(save_dir=save_dir / "infgen", name=name)
        logger = _wrap_tensorboard_hparams_logger(logger)

    ckpt_save_dir = pathlib.Path(save_dir).absolute() / "infgen" / name

    # Set up trainer arguments
    callbacks = [
        ModelCheckpoint(
            filename=str(name) + "_{epoch}-{step}",
            monitor="monitoring_step",
            every_n_epochs=1,
            save_last=True,
            auto_insert_metric_name=True,
            mode="max",
            save_top_k=-1,
            save_on_train_epoch_end=True,
        ),
        ModelCheckpoint(
            filename=str(name) + "_{epoch}-{step}",
            train_time_interval=datetime.timedelta(minutes=30),
            auto_insert_metric_name=True,
            save_on_train_epoch_end=True,
            every_n_train_steps=None,
            every_n_epochs=None,
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    rollout_gif_callback = SemanticRolloutGifCallback(
        config=config,
        artifact_root=ckpt_save_dir,
    )
    if rollout_gif_callback.enabled:
        callbacks.append(rollout_gif_callback)
    rollout_advantage_callback = SemanticRolloutAdvantageCallback(
        config=config,
        artifact_root=ckpt_save_dir,
    )
    if rollout_advantage_callback.enabled:
        callbacks.append(rollout_advantage_callback)
    device = "auto" if torch.cuda.is_available() else "cpu"
    trainer_kwargs = dict(
        num_sanity_val_steps=config.num_sanity_val_steps,
        limit_val_batches=config.limit_val_batches if config.limit_val_batches >= 0 else None,
        limit_train_batches=config.limit_train_batches if config.limit_train_batches >= 0 else None,
        gradient_clip_val=config.OPTIMIZATION.GRAD_NORM_CLIP,
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator=device,
        devices="auto",
        log_every_n_steps=2,
        deterministic=config.deterministic,
        detect_anomaly=config.detect_anomaly,
        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 1),
        # strategy='ddp_find_unused_parameters_true'
    )
    if max_steps > 0:
        trainer_kwargs["max_steps"] = max_steps
    if val_interval is not None:
        trainer_kwargs["val_check_interval"] = val_interval
        trainer_kwargs["check_val_every_n_epoch"] = None

    # from lightning.pytorch.profilers import PyTorchProfiler
    # profiler = PyTorchProfiler(filename="profile")
    # trainer_kwargs.update(
    #     profiler=profiler,
    # )

    # if config.debug:
    #     # from lightning.pytorch.profilers import PyTorchProfiler
    #     # profiler = PyTorchProfiler(filename="profile")
    #     trainer_kwconfig.update(
    #         num_sanity_val_steps=0,
    #         # profiler=profiler,
    #         detect_anomaly=True,
    #         limit_val_batches=2,
    #         limit_train_batches=2,
    #         log_every_n_steps=1,
    #     )
    #     num_workers = 0
    #     val_num_workers = 0
    # if bf16:
    #     trainer_kwargs["precision"] = "bf16-mixed"

    datamodule = InfgenDataModule(
        config,
        train_batch_size=batch_size,
        train_num_workers=num_workers,
        train_prefetch_factor=config.prefetch_factor,
        val_batch_size=val_batch_size,
        val_num_workers=val_num_workers,
        val_prefetch_factor=config.prefetch_factor,
    )
    ddp_strategy_override = str(config.get("DDP_STRATEGY_OVERRIDE", "") or "").strip()
    if torch.cuda.device_count() > 1:
        if ddp_strategy_override:
            trainer_kwargs["strategy"] = ddp_strategy_override
        elif bool(config.MODEL.get("LOCAL_CONTROL_FORWARD_ENABLED", False)):
            trainer_kwargs["strategy"] = 'ddp_find_unused_parameters_true'
        else:
            trainer_kwargs["strategy"] = 'ddp'
    if log_dir:
        trainer_kwargs["default_root_dir"] = log_dir

    # Set up trainer
    trainer = pl.Trainer(**trainer_kwargs)

    # Set up model
    ckpt_path = config.ckpt
    load_report = None
    if ckpt_path is not None:
        ckpt_path = REPO_ROOT / pathlib.Path(ckpt_path).expanduser()
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "last.ckpt"
        ckpt_path = str(ckpt_path.resolve().absolute())
        assert os.path.isfile(ckpt_path), ckpt_path
        assert ckpt_path.endswith(".ckpt"), ckpt_path
        print("==============================")
        print("Loading checkpoint: ", ckpt_path)
        print("==============================")

    pretrained_path = config.pretrain
    if pretrained_path:
        pretrained_path = pathlib.Path(pretrained_path).expanduser()
        pretrained_path = REPO_ROOT / pretrained_path
        if pretrained_path.is_dir():
            pretrained_path = pretrained_path / "last.ckpt"
        pretrained_path = str(pretrained_path.absolute().resolve())
        assert os.path.isfile(pretrained_path), pretrained_path
        assert pretrained_path.endswith(".ckpt"), pretrained_path
        print("==============================")
        print("Loading pretrained model: ", pretrained_path)
        print("==============================")

        map_location = None
        if not torch.cuda.is_available():
            print("CUDA is not available. Loading model on CPU!")
            print("CUDA is not available. Loading model on CPU!")
            print("CUDA is not available. Loading model on CPU!")
            map_location = "cpu"

        ckpt_load_mode = str(config.get("CKPT_LOAD_MODE", "legacy_merge"))
        if ckpt_load_mode == "forgiving_state_dict":
            model, load_report = load_model_from_checkpoint_forgiving(
                config=config,
                ckpt_path=pretrained_path,
                load_mode=ckpt_load_mode,
                strict_state_dict=False,
                map_location=map_location,
                checkpoint_surgery_func=utils.checkpoint_surgery_func,
            )
            print("==============================")
            print(
                "Forgiving warm-start report:",
                {
                    "num_loaded_keys": load_report["num_loaded_keys"],
                    "num_missing_keys": load_report["num_missing_keys"],
                    "num_unexpected_keys": load_report["num_unexpected_keys"],
                    "num_shape_mismatch_keys": load_report["num_shape_mismatch_keys"],
                },
            )
            print("==============================")
        else:
            model = utils.load_from_checkpoint(
                checkpoint_path=pretrained_path,
                cls=MotionLMLightning,
                config=config,
                default_config=default_config,
                strict=True,
                checkpoint_surgery_func=utils.checkpoint_surgery_func,
                map_location=map_location
            )
        # model = MotionLMLightning.load_from_checkpoint(checkpoint_path=pretrained_path, strict=strict, **config)
    else:
        model = MotionLMLightning(config=config)
    model.exp_name = name

    assert model.config == config, "The config system is not working properly! Original:\n{}\n\nNew:\n{}".format(
        model.config, config
    )
    config_save_path = ckpt_save_dir / "config.yaml"
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    utils.rank_zero_print(summarize(model, max_depth=5))
    utils.rank_zero_print("==============================")
    utils.rank_zero_print("Root Directory: ", save_dir / "infgen")
    utils.rank_zero_print("Checkpoint Log Directory: ", ckpt_save_dir)
    utils.rank_zero_print("Config Save Path: ", config_save_path)
    utils.rank_zero_print("Exp Group: ", name)
    utils.rank_zero_print("Exp Full Name: ", name)
    utils.rank_zero_print("==============================")
    print("Rank {} is done setting up the model.".format(trainer.global_rank))
    OmegaConf.save(config, config_save_path)

    artifact_root = pathlib.Path(log_dir).absolute() if log_dir is not None else ckpt_save_dir
    run_completed = False
    run_failed = False
    failure_reason = None
    try:
        if config.eval:
            trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        else:
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        run_completed = True
    except Exception as exc:
        run_failed = True
        failure_reason = {
            "message": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-40:],
        }
        raise
    finally:
        _write_training_artifacts(
            artifact_root=artifact_root,
            config=config,
            trainer=trainer,
            run_name=name,
            exp_name=exp_name,
            ckpt_path=ckpt_path,
            pretrained_path=pretrained_path,
            checkpoint_load_report=load_report,
            completed=run_completed,
            failed=run_failed,
            failure_reason=failure_reason,
        )


if __name__ == '__main__':
    main()
