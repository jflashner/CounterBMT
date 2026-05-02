import copy
import logging
import math
import time
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from bmt.counterfactual.compile_control_code import (
    BRANCH_LABEL_ORDER,
    COMPLIANCE_LABEL_ORDER,
    COMPLIANCE_TOKEN_DIM,
    PATH_TOKEN_DIM,
    SIGNAL_CATEGORY_ORDER,
    TERMINAL_ANCHOR_DIM,
    TIMING_TOKEN_DIM,
    TIMING_LABEL_ORDER,
)
from bmt.counterfactual.sdc_path_control import (
    DEFAULT_PATH_DEADBAND_M,
    SDC_PATH_SEMANTIC_LABEL_ORDER,
    project_points_to_path_torch,
    torch_heading_to_sdc_up,
    torch_world_to_sdc_up,
)
from bmt.counterfactual.sdc_semantic_control import (
    DEFAULT_FAMILY_BACKWARD_SLACK_M,
    DEFAULT_FAMILY_GUIDE_BANDWIDTH_M,
    DEFAULT_FAMILY_HEADING_BETA_RAD,
    DEFAULT_FAMILY_HEADING_DEADBAND_RAD,
    DEFAULT_FAMILY_PATH_DEADBAND_M,
    DEFAULT_FAMILY_TEACHER_TEMPERATURE,
    compute_family_gate_torch,
    family_confidence_weights_torch,
    project_points_to_family_paths_torch,
    project_points_to_segment_tube_torch,
)
from bmt.models.motionlm import MotionLM
from bmt.models.semantic_ext_topomcpo import (
    CausalDAGContextEncoder,
    EMAGaussianNoveltyBank,
    RolloutBehaviorEncoder,
)
from bmt.tokenization import get_tokenizer
from bmt.utils import lr_schedule
from bmt.utils import utils
from bmt.utils.config import REPO_ROOT

logger = logging.getLogger(__file__)


def update_ema(target_params, source_params, rate=0.99):
    """
    PZH: From https://github.com/LTH14/mar/blob/fe470ac24afbee924668d8c5c83e9fec60af3a73/engine_mar.py#L19

    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def safe_entropy(logits, epsilon=1e-5):
    """
    Computes the entropy of the given logits safely by replacing NaN and Inf values.
    :param logits: Input logits tensor.
    :param epsilon: A small value to add to the logits to avoid log(0) which results in NaN.
    :return: Mean entropy of the logits.
    """
    # Replace NaN and Inf values in logits to avoid errors in entropy computation
    logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
    logits = torch.where(torch.isinf(logits), torch.zeros_like(logits), logits)

    # Adding a small epsilon to logits to avoid log(0)
    logits = logits + epsilon

    # Compute softmax to get probabilities
    probs = F.softmax(logits, dim=-1)

    # Compute entropy
    entropy = -(probs * torch.log(probs)).sum(-1)

    # Return the mean entropy
    return entropy.mean()


def sanitize_logits_for_loss(logits, *, clamp=None):
    if clamp is None:
        return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    logits = torch.nan_to_num(logits, nan=0.0, posinf=float(clamp), neginf=-float(clamp))
    return logits.clamp(min=-float(clamp), max=float(clamp))


def sanitize_scalar_loss(value, *, fallback=0.0):
    if torch.is_tensor(value):
        return torch.nan_to_num(value, nan=float(fallback), posinf=float(fallback), neginf=float(fallback))
    return value


def clone_nested_value(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {k: clone_nested_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_nested_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_nested_value(v) for v in value)
    return copy.deepcopy(value)


class MotionLMLightning(pl.LightningModule):
    def __init__(self, config):
        if "SEED" in config:
            pl.seed_everything(config.SEED)
            print("Everything is seeded to: ", config.SEED)
        super().__init__()
        self.config = config

        if config.MODEL.NAME in ["motionlm", "gpt"]:
            self.model = MotionLM(config=self.config)
        else:
            raise ValueError(f"Unknown model name: {config.MODEL.NAME}")

        self.save_hyperparameters(OmegaConf.to_container(self.config))

        self._tokenizer = get_tokenizer(self.config)
        self.local_control_forward_enabled = bool(self.config.MODEL.get("LOCAL_CONTROL_FORWARD_ENABLED", False))
        self.counterfactual_mode = str(self.config.DATA.get("COUNTERFACTUAL_MODE", "default")).strip() or "default"
        self.sdc_semantic_ext_enabled = bool(
            self.counterfactual_mode == "sdc_semantic_only"
            and self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_ENABLED", False)
        )
        self._init_semantic_ext_topomcpo_modules()
        if self.local_control_forward_enabled:
            d_model = int(self.config.MODEL.D_MODEL)
            self.path_head = torch.nn.Linear(d_model, len(BRANCH_LABEL_ORDER))
            self.compliance_head = torch.nn.Linear(d_model, len(COMPLIANCE_LABEL_ORDER))
            self.timing_head = torch.nn.Linear(d_model, len(TIMING_LABEL_ORDER))
            self.anchor_head = torch.nn.Linear(d_model, TERMINAL_ANCHOR_DIM)
            self.sdc_semantic_head = torch.nn.Linear(d_model, len(SDC_PATH_SEMANTIC_LABEL_ORDER))
        else:
            self.sdc_semantic_head = None
        self.policy_teacher = self._build_policy_teacher()
        self.policy_teacher_sync_report = None
        self._configure_local_control_finetune()
        # self.validation_outputs = []
        # self.validation_ground_truth = []

        self.exp_name = None
        self._last_rollout_train_debug_step = -1

    def _init_semantic_ext_topomcpo_modules(self):
        self.semantic_ext_behavior_encoder = None
        self.semantic_ext_dag_encoder = None
        self.semantic_ext_semantic_head = None
        self.semantic_ext_branch_head = None
        self.semantic_ext_compliance_head = None
        self.semantic_ext_timing_head = None
        self.semantic_ext_novelty_bank = None
        if not self.sdc_semantic_ext_enabled:
            return
        if not bool(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_BEHAVIOR_MODEL_ENABLED", False)):
            return
        hidden_dim = int(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_BEHAVIOR_HIDDEN_DIM", 64))
        embed_dim = int(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_BEHAVIOR_EMBED_DIM", 32))
        input_dim = 8
        self.semantic_ext_behavior_encoder = RolloutBehaviorEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
        )
        self.semantic_ext_dag_encoder = CausalDAGContextEncoder(
            path_dim=PATH_TOKEN_DIM,
            compliance_dim=COMPLIANCE_TOKEN_DIM,
            timing_dim=TIMING_TOKEN_DIM,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
        )
        self.semantic_ext_semantic_head = torch.nn.Linear(embed_dim, len(SDC_PATH_SEMANTIC_LABEL_ORDER))
        self.semantic_ext_branch_head = torch.nn.Linear(embed_dim, len(BRANCH_LABEL_ORDER))
        self.semantic_ext_compliance_head = torch.nn.Linear(embed_dim, len(COMPLIANCE_LABEL_ORDER))
        self.semantic_ext_timing_head = torch.nn.Linear(embed_dim, len(TIMING_LABEL_ORDER))
        self.semantic_ext_novelty_bank = EMAGaussianNoveltyBank(
            decay=float(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_NOVELTY_EMA_DECAY", 0.995)),
            min_var=float(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_NOVELTY_MIN_VAR", 1e-3)),
        )

    def _configure_local_control_finetune(self):
        if not self.local_control_forward_enabled:
            return
        if not bool(self.config.MODEL.get("LOCAL_CONTROL_FREEZE_BACKBONE", False)):
            return

        use_legacy_control = self.counterfactual_mode not in {"sdc_semantic_only", "sdc_path"}
        use_sdc_semantic_only = self.counterfactual_mode == "sdc_semantic_only"
        use_sdc_path = self.counterfactual_mode == "sdc_path"

        for parameter in self.model.parameters():
            parameter.requires_grad = False
        if use_legacy_control:
            for parameter in self.path_head.parameters():
                parameter.requires_grad = True
        if use_legacy_control and bool(self.config.MODEL.get("LOCAL_CONTROL_USE_ANCHOR", True)):
            for parameter in self.anchor_head.parameters():
                parameter.requires_grad = True

        if use_legacy_control and bool(self.config.MODEL.get("LOCAL_CONTROL_USE_COMPLIANCE", True)):
            for parameter in self.compliance_head.parameters():
                parameter.requires_grad = True
        if use_legacy_control and bool(self.config.MODEL.get("LOCAL_CONTROL_USE_TIMING", True)):
            for parameter in self.timing_head.parameters():
                parameter.requires_grad = True

        motion_decoder = self.model.motion_decoder
        if use_legacy_control and hasattr(motion_decoder, "cf_path_proj") and bool(self.config.MODEL.get("LOCAL_CONTROL_USE_PATH", True)):
            for parameter in motion_decoder.cf_path_proj.parameters():
                parameter.requires_grad = True
        if use_legacy_control and hasattr(motion_decoder, "cf_anchor_proj") and bool(self.config.MODEL.get("LOCAL_CONTROL_USE_ANCHOR", True)):
            for parameter in motion_decoder.cf_anchor_proj.parameters():
                parameter.requires_grad = True
        if use_legacy_control and hasattr(motion_decoder, "cf_compliance_proj") and bool(self.config.MODEL.get("LOCAL_CONTROL_USE_COMPLIANCE", True)):
            for parameter in motion_decoder.cf_compliance_proj.parameters():
                parameter.requires_grad = True
        if use_legacy_control and hasattr(motion_decoder, "cf_timing_proj") and bool(self.config.MODEL.get("LOCAL_CONTROL_USE_TIMING", True)):
            for parameter in motion_decoder.cf_timing_proj.parameters():
                parameter.requires_grad = True
        if use_legacy_control and hasattr(motion_decoder, "cf_local_bias"):
            for parameter in motion_decoder.cf_local_bias.parameters():
                parameter.requires_grad = True
        if use_legacy_control and hasattr(motion_decoder, "cf_local_residual_gate"):
            motion_decoder.cf_local_residual_gate.requires_grad = True
        if (use_sdc_semantic_only or use_sdc_path) and hasattr(motion_decoder, "cf_sdc_semantic_embed"):
            for parameter in motion_decoder.cf_sdc_semantic_embed.parameters():
                parameter.requires_grad = True
        if use_sdc_path and hasattr(motion_decoder, "cf_sdc_waypoint_proj"):
            for parameter in motion_decoder.cf_sdc_waypoint_proj.parameters():
                parameter.requires_grad = True
        if use_sdc_path and hasattr(motion_decoder, "cf_sdc_waypoint_summary"):
            for parameter in motion_decoder.cf_sdc_waypoint_summary.parameters():
                parameter.requires_grad = True
        if (use_sdc_semantic_only or use_sdc_path) and hasattr(motion_decoder, "cf_sdc_local_bias"):
            for parameter in motion_decoder.cf_sdc_local_bias.parameters():
                parameter.requires_grad = True
        if (use_sdc_semantic_only or use_sdc_path) and hasattr(motion_decoder, "cf_sdc_local_residual_gate"):
            motion_decoder.cf_sdc_local_residual_gate.requires_grad = True
        if (use_sdc_semantic_only or use_sdc_path) and self.sdc_semantic_head is not None:
            for parameter in self.sdc_semantic_head.parameters():
                parameter.requires_grad = True

        train_last_n = int(self.config.MODEL.get("LOCAL_CONTROL_TRAIN_LAST_N_DECODER_BLOCKS", 0))
        if train_last_n > 0 and hasattr(motion_decoder, "decoder"):
            for layer in motion_decoder.decoder.layers[-train_last_n:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

        if bool(self.config.MODEL.get("LOCAL_CONTROL_TRAIN_OUTPUT_HEAD", False)):
            for parameter in motion_decoder.prediction_head.parameters():
                parameter.requires_grad = True
            norm_module = getattr(motion_decoder, "prediction_prenorm", None)
            if norm_module is None:
                norm_module = getattr(motion_decoder, "prediction_adaln_norm", None)
            if norm_module is not None:
                for parameter in norm_module.parameters():
                    parameter.requires_grad = True

    def _build_policy_teacher(self):
        ckpt_path = str(self.config.MODEL.get("LOCAL_CONTROL_SDC_POLICY_TEACHER_CKPT", "")).strip()
        if not ckpt_path:
            return None
        resolved = Path(ckpt_path).expanduser()
        if not resolved.is_absolute():
            resolved = (REPO_ROOT / resolved).resolve()
        if not resolved.exists():
            logger.warning("Skipping policy teacher load; checkpoint not found: %s", resolved)
            return None

        teacher_config = copy.deepcopy(self.config)
        teacher_config.MODEL.LOCAL_CONTROL_FORWARD_ENABLED = False
        teacher_config.DATA.COUNTERFACTUAL_MODE = "default"
        teacher = MotionLM(config=teacher_config)
        checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_state = teacher.state_dict()
        loaded = {}
        for key, value in state_dict.items():
            key = str(key)
            if not key.startswith("model."):
                continue
            target_key = key[len("model."):]
            if target_key not in model_state:
                continue
            if tuple(model_state[target_key].shape) != tuple(value.shape):
                continue
            loaded[target_key] = value
        teacher.load_state_dict(loaded, strict=False)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        return teacher

    def sync_policy_teacher_from_student(self):
        if self.policy_teacher is None:
            report = {
                "teacher_present": False,
                "sync_source": "none",
                "num_teacher_keys": 0,
                "num_loaded_keys": 0,
                "num_missing_keys": 0,
                "num_shape_mismatch_keys": 0,
                "fully_synced": False,
            }
            self.policy_teacher_sync_report = report
            return report

        student_state = self.model.state_dict()
        teacher_state = self.policy_teacher.state_dict()
        loaded = {}
        missing_keys = []
        shape_mismatch_keys = []
        for key, teacher_value in teacher_state.items():
            student_value = student_state.get(key)
            if student_value is None:
                missing_keys.append(str(key))
                continue
            if tuple(student_value.shape) != tuple(teacher_value.shape):
                shape_mismatch_keys.append(str(key))
                continue
            loaded[str(key)] = student_value.detach().clone()

        self.policy_teacher.load_state_dict(loaded, strict=False)
        self.policy_teacher.eval()
        for parameter in self.policy_teacher.parameters():
            parameter.requires_grad = False

        report = {
            "teacher_present": True,
            "sync_source": "student_model_post_warmstart",
            "num_teacher_keys": int(len(teacher_state)),
            "num_loaded_keys": int(len(loaded)),
            "num_missing_keys": int(len(missing_keys)),
            "num_shape_mismatch_keys": int(len(shape_mismatch_keys)),
            "first_50_missing_keys": sorted(missing_keys)[:50],
            "first_50_shape_mismatch_keys": sorted(shape_mismatch_keys)[:50],
            "fully_synced": bool(len(loaded) == len(teacher_state) and not missing_keys and not shape_mismatch_keys),
        }
        self.policy_teacher_sync_report = report
        return report

    @staticmethod
    def _as_tensor(value, *, device, dtype=None):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, device=device)
        value = value.to(device=device)
        if dtype is not None:
            value = value.to(dtype=dtype)
        return value

    @staticmethod
    def _normalize_text_list(value):
        from bmt.counterfactual.sdc_rollout_training_debug import normalize_text_list

        return normalize_text_list(value)

    def _should_dump_rollout_tube_training_debug(self):
        if not self.training:
            return False
        if not bool(self.config.get("ROLLOUT_TRAIN_DEBUG_ENABLED", False)):
            return False
        step_index = int(self.global_step) + 1
        every_n_steps = max(1, int(self.config.get("ROLLOUT_TRAIN_DEBUG_EVERY_N_STEPS", 25)))
        if step_index % every_n_steps != 0:
            return False
        if self._last_rollout_train_debug_step == step_index:
            return False
        return True

    def _trace_first_step(self, stage: str, **extra):
        if not bool(self.config.get("DDP_FIRST_STEP_TRACE", False)):
            return
        step_index = int(getattr(self, "global_step", 0))
        if step_index > 0:
            return
        payload = {
            "rank": int(getattr(self, "global_rank", -1)),
            "local_rank": int(getattr(self, "local_rank", -1)),
            "stage": str(stage),
            "global_step": step_index,
            "time": round(time.time(), 3),
        }
        for key, value in extra.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    payload[key] = float(value.detach().cpu().item())
                else:
                    payload[key] = list(value.shape)
            else:
                payload[key] = value
        print(f"[ddp_first_step_trace] {payload}", flush=True)

    def _rollout_tube_training_debug_output_root(self):
        logger_obj = getattr(self, "logger", None)
        if logger_obj is not None:
            log_dir = getattr(logger_obj, "log_dir", None)
            if log_dir:
                return Path(log_dir).expanduser().resolve()
            save_dir = getattr(logger_obj, "save_dir", None)
            name = getattr(logger_obj, "name", None)
            version = getattr(logger_obj, "version", None)
            if save_dir and name is not None and version is not None:
                return Path(save_dir).expanduser().resolve() / str(name) / str(version)
        trainer_obj = getattr(self, "trainer", None)
        if trainer_obj is not None:
            default_root = getattr(trainer_obj, "default_root_dir", None)
            if default_root:
                return Path(default_root).expanduser().resolve()
        return (REPO_ROOT / "logs" / "rollout_train_debug_fallback").resolve()

    def _extract_rollout_debug_meta_list(self, data_dict, *, batch_size):
        meta_value = data_dict.get("cf/sdc_debug_meta")
        if isinstance(meta_value, list):
            meta_list = []
            for item in meta_value[:batch_size]:
                meta_list.append(dict(item) if isinstance(item, dict) else {"raw_meta": str(item)})
            while len(meta_list) < batch_size:
                meta_list.append({})
            return meta_list
        scenario_value = data_dict.get("metadata/scenario_id", data_dict.get("scenario_id"))
        scenario_list = []
        if isinstance(scenario_value, list):
            scenario_list = [str(item) for item in scenario_value]
        elif scenario_value is not None:
            scenario_list = [str(scenario_value)] * batch_size
        meta_list = []
        for idx in range(batch_size):
            meta_list.append({"scenario_id": scenario_list[idx] if idx < len(scenario_list) else ""})
        return meta_list

    def _maybe_dump_rollout_tube_training_debug(
        self,
        *,
        data_dict,
        reward_t,
        valid_mask,
        selected_log_probs,
        tube_distance,
        rtg,
        advantage_t,
        trajectories_world,
        action_token_t,
        traffic_speed_penalty_t=None,
        traffic_speed_reference_t=None,
        traffic_speed_neighbor_mean_t=None,
        traffic_speed_neighbor_median_t=None,
        traffic_speed_floor_threshold_t=None,
        traffic_speed_cap_threshold_t=None,
        traffic_speed_cap_penalty_t=None,
        traffic_speed_sdc_t=None,
        traffic_speed_neighbor_count_t=None,
    ):
        if not self._should_dump_rollout_tube_training_debug():
            return

        try:
            from bmt.counterfactual.sdc_rollout_training_debug import write_rollout_tube_training_debug

            step_index = int(self.global_step) + 1
            batch_size = int(reward_t.shape[1])
            meta_list = self._extract_rollout_debug_meta_list(data_dict, batch_size=batch_size)
            scenario_filters = set(self._normalize_text_list(self.config.get("ROLLOUT_TRAIN_DEBUG_SCENARIO_IDS", [])))
            slot_filters = set(self._normalize_text_list(self.config.get("ROLLOUT_TRAIN_DEBUG_SLOT_IDS", [])))
            include_gt = bool(self.config.get("ROLLOUT_TRAIN_DEBUG_INCLUDE_GT", False))
            max_matches = max(1, int(self.config.get("ROLLOUT_TRAIN_DEBUG_MAX_MATCHES", 4)))
            grid_step_m = float(self.config.get("ROLLOUT_TRAIN_DEBUG_GRID_STEP_M", 0.35))
            output_subdir = str(self.config.get("ROLLOUT_TRAIN_DEBUG_OUTPUT_SUBDIR", "train_rollout_debug")).strip() or "train_rollout_debug"

            printable_meta = [
                {
                    "scenario_id": str((meta or {}).get("scenario_id") or ""),
                    "selected_slot_id": str((meta or {}).get("selected_slot_id") or ""),
                    "requested_semantic_label": str((meta or {}).get("requested_semantic_label") or ""),
                }
                for meta in meta_list
            ]
            print(
                "[rollout_train_debug] "
                f"step={step_index} "
                f"scenario_filters={sorted(scenario_filters)} "
                f"slot_filters={sorted(slot_filters)} "
                f"include_gt={include_gt} "
                f"meta={printable_meta}",
                flush=True,
            )

            matches = []
            for batch_idx, meta in enumerate(meta_list):
                scenario_id = str(meta.get("scenario_id") or "").strip()
                slot_id = str(meta.get("selected_slot_id") or "").strip()
                if scenario_filters and scenario_id not in scenario_filters:
                    continue
                if slot_filters and slot_id not in slot_filters:
                    continue
                if not include_gt and slot_id == "gt":
                    continue
                matches.append((batch_idx, meta))
            if not matches:
                print(
                    f"[rollout_train_debug] no matches at step={step_index}",
                    flush=True,
                )
                return

            root = self._rollout_tube_training_debug_output_root() / output_subdir / f"step_{step_index:06d}"
            raw_path_model = self._as_tensor(
                data_dict.get("cf/sdc_selected_raw_path_model", data_dict["cf/sdc_selected_raw_path_world"]),
                device=reward_t.device,
                dtype=reward_t.dtype,
            ).detach().cpu().numpy()
            raw_path_mask = self._as_tensor(
                data_dict["cf/sdc_selected_raw_path_mask"],
                device=reward_t.device,
                dtype=reward_t.dtype,
            ).detach().cpu().numpy()
            raw_segment_mask = self._as_tensor(
                data_dict["cf/sdc_selected_raw_path_segment_mask"],
                device=reward_t.device,
                dtype=reward_t.dtype,
            ).detach().cpu().numpy()
            decision_agent_mask = self._as_tensor(
                data_dict["cf/decision_agent_mask"],
                device=reward_t.device,
                dtype=reward_t.dtype,
            )
            current_pos_world = self._as_tensor(
                data_dict["decoder/modeled_agent_position"][:, 0, :, :2],
                device=reward_t.device,
                dtype=reward_t.dtype,
            )
            current_heading_world_t = self._as_tensor(
                data_dict["decoder/modeled_agent_heading"][:, 0, :],
                device=reward_t.device,
                dtype=reward_t.dtype,
            )
            current_xy_world = (
                current_pos_world * decision_agent_mask[:, :, None]
            ).sum(dim=1).detach().cpu().numpy()
            current_heading_world = (
                current_heading_world_t * decision_agent_mask
            ).sum(dim=1).detach().cpu().numpy()
            total_return = reward_t.sum(dim=-1)
            total_return_adv = self._group_normalize_advantages(
                total_return,
                valid_mask.any(dim=-1),
            )

            for batch_idx, meta in matches[:max_matches]:
                scenario_id = str(meta.get("scenario_id") or f"batch_{batch_idx:03d}")
                slot_id = str(meta.get("selected_slot_id") or f"slot_{batch_idx:03d}")
                requested_semantic_label = str(meta.get("requested_semantic_label") or "")
                outdir = root / scenario_id / slot_id
                write_rollout_tube_training_debug(
                    outdir=outdir,
                    scenario_id=scenario_id,
                    slot_id=slot_id,
                    requested_semantic_label=requested_semantic_label,
                    global_step=step_index,
                    current_xy_world=current_xy_world[batch_idx],
                    current_heading_world=float(current_heading_world[batch_idx]),
                    path_world=raw_path_model[batch_idx],
                    point_mask=raw_path_mask[batch_idx],
                    segment_mask=raw_segment_mask[batch_idx],
                    trajectories_world=trajectories_world[:, batch_idx].detach().cpu().numpy(),
                    reward_t=reward_t[:, batch_idx].detach().cpu().numpy(),
                    return_to_go_t=rtg[:, batch_idx].detach().cpu().numpy(),
                    advantage_t=advantage_t[:, batch_idx].detach().cpu().numpy(),
                    action_token_t=action_token_t[:, batch_idx].detach().cpu().numpy(),
                    action_logprob_t=selected_log_probs[:, batch_idx].detach().cpu().numpy(),
                    tube_distance_t=tube_distance[:, batch_idx].detach().cpu().numpy(),
                    valid_mask_t=valid_mask[:, batch_idx].detach().cpu().numpy(),
                    tube_radius_m=float(self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_TUBE_RADIUS_M", 3.0)),
                    inside_reward=float(self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_TUBE_INSIDE_REWARD", 1.0)),
                    outside_scale=float(self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_TUBE_OUTSIDE_SCALE", 1.0)),
                    discount=float(self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_TUBE_DISCOUNT", 1.0)),
                    grid_step_m=grid_step_m,
                    scenario_pkl=str(meta.get("scenario_pkl") or ""),
                    current_time_index=meta.get("current_time_index"),
                    sdc_id=str(meta.get("sdc_id") or ""),
                    traffic_speed_penalty_t=(
                        None
                        if traffic_speed_penalty_t is None
                        else traffic_speed_penalty_t[:, batch_idx].detach().cpu().numpy()
                    ),
                    traffic_speed_reference_t=(
                        None
                        if traffic_speed_reference_t is None
                        else traffic_speed_reference_t[:, batch_idx].detach().cpu().numpy()
                    ),
                    traffic_speed_neighbor_mean_t=(
                        None
                        if traffic_speed_neighbor_mean_t is None
                        else traffic_speed_neighbor_mean_t[:, batch_idx].detach().cpu().numpy()
                    ),
                    traffic_speed_neighbor_median_t=(
                        None
                        if traffic_speed_neighbor_median_t is None
                        else traffic_speed_neighbor_median_t[:, batch_idx].detach().cpu().numpy()
                    ),
                    traffic_speed_floor_threshold_t=(
                        None
                        if traffic_speed_floor_threshold_t is None
                        else traffic_speed_floor_threshold_t[:, batch_idx].detach().cpu().numpy()
                    ),
                    traffic_speed_cap_threshold_t=(
                        None
                        if traffic_speed_cap_threshold_t is None
                        else traffic_speed_cap_threshold_t[:, batch_idx].detach().cpu().numpy()
                    ),
                    traffic_speed_cap_penalty_t=(
                        None
                        if traffic_speed_cap_penalty_t is None
                        else traffic_speed_cap_penalty_t[:, batch_idx].detach().cpu().numpy()
                    ),
                    traffic_speed_sdc_t=(
                        None
                        if traffic_speed_sdc_t is None
                        else traffic_speed_sdc_t[:, batch_idx].detach().cpu().numpy()
                    ),
                    traffic_speed_neighbor_count_t=(
                        None
                        if traffic_speed_neighbor_count_t is None
                        else traffic_speed_neighbor_count_t[:, batch_idx].detach().cpu().numpy()
                    ),
                    extra_summary={
                        "mean_total_return": float(total_return[:, batch_idx].mean().detach().cpu().item()),
                        "mean_scalar_group_advantage": float(total_return_adv[:, batch_idx].mean().detach().cpu().item()),
                        "requested_semantic_confidence": meta.get("requested_semantic_confidence"),
                    },
                )
                print(f"[rollout_train_debug] wrote {outdir}", flush=True)
            self._last_rollout_train_debug_step = step_index
        except Exception as exc:  # pragma: no cover - debug path should not crash training
            print(f"[rollout_train_debug] failed: {exc}", flush=True)
            logger.warning("Failed to dump rollout tube training debug: %s", exc)

    def _next_state_candidates_from_action_space(self, output_logit, data_dict):
        probs = torch.softmax(output_logit.float(), dim=-1)
        B, T, N, A = probs.shape
        current_pos = self._as_tensor(
            data_dict["decoder/modeled_agent_position"][:, :T, :, :2],
            device=output_logit.device,
            dtype=probs.dtype,
        )
        current_heading = self._as_tensor(
            data_dict["decoder/modeled_agent_heading"][:, :T],
            device=output_logit.device,
            dtype=probs.dtype,
        )
        current_vel = self._as_tensor(
            data_dict["decoder/modeled_agent_velocity"][:, :T, :, :2],
            device=output_logit.device,
            dtype=probs.dtype,
        )
        current_valid = self._as_tensor(
            data_dict["decoder/input_action_valid_mask"][:, :T],
            device=output_logit.device,
            dtype=torch.bool,
        )
        dt = float(getattr(self._tokenizer, "dt", 0.5))

        if hasattr(self._tokenizer, "all_trajs") and hasattr(self._tokenizer, "all_heading"):
            bin_centers = self._tokenizer.bin_centers.to(device=output_logit.device, dtype=probs.dtype)
            bin_centers = bin_centers.reshape(1, 1, 1, A, 2).expand(B, T, N, A, 2)
            rotate_angle = current_heading.unsqueeze(-1) - (math.pi / 2.0)
            delta_x = torch.cos(rotate_angle) * bin_centers[..., 0] - torch.sin(rotate_angle) * bin_centers[..., 1]
            delta_y = torch.cos(rotate_angle) * bin_centers[..., 1] + torch.sin(rotate_angle) * bin_centers[..., 0]
            next_pos_candidates = current_pos.unsqueeze(-2) + torch.stack([delta_x, delta_y], dim=-1)
            heading_offsets = self._tokenizer.all_heading.to(device=output_logit.device, dtype=probs.dtype)[:, -1]
            candidate_heading = torch.atan2(
                torch.sin(current_heading.unsqueeze(-1) + heading_offsets.reshape(1, 1, 1, A)),
                torch.cos(current_heading.unsqueeze(-1) + heading_offsets.reshape(1, 1, 1, A)),
            )
            next_vel_candidates = (next_pos_candidates - current_pos.unsqueeze(-2)) / max(dt, 1e-3)
        elif hasattr(self._tokenizer, "acceleration_bins") and hasattr(self._tokenizer, "steering_bins"):
            if self._tokenizer.acceleration_bins.device != output_logit.device:
                self._tokenizer.acceleration_bins = self._tokenizer.acceleration_bins.to(output_logit.device)
                self._tokenizer.steering_bins = self._tokenizer.steering_bins.to(output_logit.device)

            acc_grid = self._tokenizer.acceleration_bins.to(dtype=probs.dtype)
            steering_grid = self._tokenizer.steering_bins.to(dtype=probs.dtype)
            a_idx = torch.div(torch.arange(A, device=output_logit.device), self._tokenizer.num_bins, rounding_mode="floor")
            delta_idx = torch.remainder(torch.arange(A, device=output_logit.device), self._tokenizer.num_bins)
            candidate_acc = acc_grid[a_idx].reshape(1, 1, 1, A).expand(B, T, N, A)
            candidate_yaw_rate = steering_grid[delta_idx].reshape(1, 1, 1, A).expand(B, T, N, A)

            current_speed = torch.linalg.norm(current_vel, dim=-1, keepdim=False).unsqueeze(-1)
            next_speed = current_speed + candidate_acc * dt
            average_speed = (current_speed + next_speed) * 0.5
            delta_theta = candidate_yaw_rate * dt
            candidate_heading = torch.atan2(
                torch.sin(current_heading.unsqueeze(-1) + delta_theta),
                torch.cos(current_heading.unsqueeze(-1) + delta_theta),
            )
            average_heading = torch.atan2(
                torch.sin(current_heading.unsqueeze(-1)) + torch.sin(candidate_heading),
                torch.cos(current_heading.unsqueeze(-1)) + torch.cos(candidate_heading),
            )
            avg_vel_x = torch.cos(average_heading) * average_speed
            avg_vel_y = torch.sin(average_heading) * average_speed
            next_vel_x = torch.cos(candidate_heading) * next_speed
            next_vel_y = torch.sin(candidate_heading) * next_speed
            next_vel_candidates = torch.stack([next_vel_x, next_vel_y], dim=-1)
            next_pos_candidates = current_pos.unsqueeze(-2) + torch.stack([avg_vel_x, avg_vel_y], dim=-1) * dt
        else:
            agent_type = self._as_tensor(data_dict["decoder/agent_type"], device=output_logit.device, dtype=torch.long)
            bin_centers = self._tokenizer.get_bin_centers(agent_type)
            if bin_centers.ndim != 4:
                raise ValueError(f"Unexpected bin_centers shape for delta-delta tokenizer: {tuple(bin_centers.shape)}")
            bin_centers = bin_centers.permute(0, 2, 1, 3).unsqueeze(1).expand(B, T, N, A, 2).to(dtype=probs.dtype)
            rotate_angle = current_heading.unsqueeze(-1) - (math.pi / 2.0)
            delta_vel_x = torch.cos(rotate_angle) * bin_centers[..., 0] - torch.sin(rotate_angle) * bin_centers[..., 1]
            delta_vel_y = torch.cos(rotate_angle) * bin_centers[..., 1] + torch.sin(rotate_angle) * bin_centers[..., 0]
            delta_vel = torch.stack([delta_vel_x, delta_vel_y], dim=-1)
            next_vel_candidates = current_vel.unsqueeze(-2) + delta_vel
            next_pos_candidates = current_pos.unsqueeze(-2) + next_vel_candidates * dt
            displacement = next_pos_candidates - current_pos.unsqueeze(-2)
            candidate_heading = torch.atan2(displacement[..., 1], displacement[..., 0])
            speed = torch.linalg.norm(displacement, dim=-1)
            keep_heading = speed < float(self.config.TOKENIZATION.get("MIN_DISPLACEMENT", 0.1))
            candidate_heading = torch.where(keep_heading, current_heading.unsqueeze(-1), candidate_heading)
        return {
            "probs": probs,
            "current_pos_world": current_pos,
            "current_heading_world": current_heading,
            "current_vel_world": current_vel,
            "current_valid": current_valid,
            "next_pos_candidates_world": next_pos_candidates,
            "next_vel_candidates_world": next_vel_candidates,
            "candidate_heading_world": candidate_heading,
        }

    def _expected_next_state_from_logits(self, output_logit, data_dict):
        candidate_bundle = self._next_state_candidates_from_action_space(output_logit, data_dict)
        probs = candidate_bundle["probs"]
        current_pos = candidate_bundle["current_pos_world"]
        current_heading = candidate_bundle["current_heading_world"]
        current_vel = candidate_bundle["current_vel_world"]
        current_valid = candidate_bundle["current_valid"]
        next_pos_candidates = candidate_bundle["next_pos_candidates_world"]
        next_vel_candidates = candidate_bundle["next_vel_candidates_world"]
        candidate_heading = candidate_bundle["candidate_heading_world"]
        expected_pos = (probs.unsqueeze(-1) * next_pos_candidates).sum(dim=-2)
        expected_vel = (probs.unsqueeze(-1) * next_vel_candidates).sum(dim=-2)
        expected_heading = torch.atan2(
            (probs * torch.sin(candidate_heading)).sum(dim=-1),
            (probs * torch.cos(candidate_heading)).sum(dim=-1),
        )
        expected_pos = torch.where(current_valid[..., None], expected_pos, current_pos)
        expected_vel = torch.where(current_valid[..., None], expected_vel, current_vel)
        expected_heading = torch.where(current_valid, expected_heading, current_heading)
        return {
            "expected_pos_world": expected_pos,
            "expected_vel_world": expected_vel,
            "expected_heading_world": expected_heading,
        }

    def _adapt_rollout_output_for_semantic_guidance(self, *, base_batch, rollout_output):
        # Start from the original batch so counterfactual control tensors survive
        # the autoregressive rollout adaptation. The rollout output only carries
        # model-produced decoder tensors, while the tube/semantic objectives still
        # need the static `cf/...` metadata from the source batch.
        adapted = clone_nested_value(base_batch)
        for key, value in rollout_output.items():
            adapted[key] = clone_nested_value(value)
        rollout_logits = rollout_output["decoder/output_logit"]
        device = rollout_logits.device
        dtype = rollout_logits.dtype
        horizon = int(rollout_logits.shape[1])

        initial_pos = self._as_tensor(
            base_batch["decoder/modeled_agent_position"][:, :1],
            device=device,
            dtype=dtype,
        )
        initial_heading = self._as_tensor(
            base_batch["decoder/modeled_agent_heading"][:, :1],
            device=device,
            dtype=dtype,
        )
        initial_velocity = self._as_tensor(
            base_batch["decoder/modeled_agent_velocity"][:, :1],
            device=device,
            dtype=dtype,
        )
        rollout_next_pos = self._as_tensor(
            rollout_output["decoder/debug_ar_pos"][:, :horizon],
            device=device,
            dtype=dtype,
        )
        rollout_next_heading = self._as_tensor(
            rollout_output["decoder/debug_ar_head"][:, :horizon],
            device=device,
            dtype=dtype,
        )
        rollout_next_velocity = self._as_tensor(
            rollout_output["decoder/debug_ar_vel"][:, :horizon],
            device=device,
            dtype=dtype,
        )

        if horizon > 1:
            current_pos = torch.cat([initial_pos, rollout_next_pos[:, :-1]], dim=1)
            current_heading = torch.cat([initial_heading, rollout_next_heading[:, :-1]], dim=1)
            current_velocity = torch.cat([initial_velocity, rollout_next_velocity[:, :-1]], dim=1)
        else:
            current_pos = initial_pos.clone()
            current_heading = initial_heading.clone()
            current_velocity = initial_velocity.clone()

        adapted["decoder/modeled_agent_position"] = current_pos
        adapted["decoder/modeled_agent_heading"] = current_heading
        adapted["decoder/modeled_agent_velocity"] = current_velocity
        adapted["decoder/modeled_agent_valid_mask"] = rollout_output["decoder/input_action_valid_mask"][:, :horizon]
        adapted["decoder/modeled_agent_delta"] = rollout_next_pos[..., :2] - current_pos[..., :2]
        adapted["decoder/input_action"] = rollout_output["decoder/output_action"][:, :horizon]
        adapted["decoder/input_action_valid_mask"] = rollout_output["decoder/input_action_valid_mask"][:, :horizon]
        adapted["decoder/target_action_valid_mask"] = rollout_output["decoder/input_action_valid_mask"][:, :horizon]
        adapted["decoder/input_step"] = torch.arange(horizon, device=device, dtype=torch.long)
        adapted["decoder/rollout_next_position"] = rollout_next_pos
        adapted["decoder/rollout_next_heading"] = rollout_next_heading
        adapted["decoder/rollout_next_velocity"] = rollout_next_velocity
        return adapted

    def _prepare_batch_for_training_autoregressive_rollout(self, data_dict):
        required = (
            "decoder/agent_position",
            "decoder/agent_velocity",
            "decoder/agent_heading",
            "decoder/agent_valid_mask",
        )
        if all(key in data_dict for key in required):
            return data_dict

        rollout_batch = dict(data_dict)
        if "decoder/current_agent_position" in data_dict:
            agent_position = self._as_tensor(
                data_dict["decoder/current_agent_position"],
                device=data_dict["decoder/modeled_agent_position"].device,
                dtype=data_dict["decoder/modeled_agent_position"].dtype,
            )
        else:
            agent_position = self._as_tensor(
                data_dict["decoder/modeled_agent_position"][:, 0],
                device=data_dict["decoder/modeled_agent_position"].device,
                dtype=data_dict["decoder/modeled_agent_position"].dtype,
            )
        if "decoder/current_agent_velocity" in data_dict:
            agent_velocity = self._as_tensor(
                data_dict["decoder/current_agent_velocity"],
                device=data_dict["decoder/modeled_agent_velocity"].device,
                dtype=data_dict["decoder/modeled_agent_velocity"].dtype,
            )
        else:
            agent_velocity = self._as_tensor(
                data_dict["decoder/modeled_agent_velocity"][:, 0],
                device=data_dict["decoder/modeled_agent_velocity"].device,
                dtype=data_dict["decoder/modeled_agent_velocity"].dtype,
            )
        if "decoder/current_agent_heading" in data_dict:
            agent_heading = self._as_tensor(
                data_dict["decoder/current_agent_heading"],
                device=data_dict["decoder/modeled_agent_heading"].device,
                dtype=data_dict["decoder/modeled_agent_heading"].dtype,
            )
        else:
            agent_heading = self._as_tensor(
                data_dict["decoder/modeled_agent_heading"][:, 0],
                device=data_dict["decoder/modeled_agent_heading"].device,
                dtype=data_dict["decoder/modeled_agent_heading"].dtype,
            )
        if "decoder/current_agent_valid_mask" in data_dict:
            agent_valid_mask = self._as_tensor(
                data_dict["decoder/current_agent_valid_mask"],
                device=data_dict["decoder/modeled_agent_position"].device,
                dtype=torch.bool,
            )
        else:
            agent_valid_mask = self._as_tensor(
                data_dict["decoder/input_action_valid_mask"][:, 0],
                device=data_dict["decoder/modeled_agent_position"].device,
                dtype=torch.bool,
            )

        rollout_batch["decoder/agent_position"] = agent_position[:, None]
        rollout_batch["decoder/agent_velocity"] = agent_velocity[:, None]
        rollout_batch["decoder/agent_heading"] = agent_heading[:, None]
        rollout_batch["decoder/agent_valid_mask"] = agent_valid_mask[:, None]
        return rollout_batch

    def _get_rollout_step_dt_s(self):
        tokenizer_dt = getattr(self._tokenizer, "dt", None)
        if tokenizer_dt is not None:
            return float(tokenizer_dt)
        num_skipped_steps = int(self.config.TOKENIZATION.get("NUM_SKIPPED_STEPS", 5))
        return float(num_skipped_steps) / 10.0

    def _extract_sdc_current_speed_mps(self, *, data_dict, semantic_context, output_logit):
        if semantic_context is None:
            return output_logit.new_zeros((output_logit.shape[0],), dtype=torch.float32)
        decision_agent_mask = semantic_context["decision_agent_mask"]
        if "decoder/current_agent_velocity" in data_dict:
            current_velocity = self._as_tensor(
                data_dict["decoder/current_agent_velocity"],
                device=output_logit.device,
                dtype=output_logit.dtype,
            )
        else:
            current_velocity = self._as_tensor(
                data_dict["decoder/modeled_agent_velocity"][:, 0],
                device=output_logit.device,
                dtype=output_logit.dtype,
            )
        sdc_velocity = (
            current_velocity[:, :, :2] * decision_agent_mask[:, :, None].to(dtype=output_logit.dtype)
        ).sum(dim=1)
        return torch.linalg.norm(sdc_velocity.to(dtype=torch.float32), dim=-1)

    @staticmethod
    def _masked_mean_per_example(values, mask):
        values = values.to(dtype=torch.float32)
        mask_f = mask.to(dtype=values.dtype)
        denom = mask_f.sum(dim=-1).clamp_min(1.0)
        mean = (values * mask_f).sum(dim=-1) / denom
        return torch.where(mask_f.sum(dim=-1) > 0.0, mean, torch.zeros_like(mean))

    @staticmethod
    def _wrap_angle(angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _semantic_ext_center_curriculum_weight(self, device):
        if not self.sdc_semantic_ext_enabled:
            return torch.tensor(0.0, device=device)
        max_weight = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_CENTER_WEIGHT_MAX", 0.03))
        if max_weight <= 0.0:
            return torch.tensor(0.0, device=device)
        start_frac = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_CENTER_CURRICULUM_START_FRAC", 0.67)
        )
        trainer = getattr(self, "trainer", None)
        total_steps = 0
        if trainer is not None:
            max_steps = int(getattr(trainer, "max_steps", -1) or -1)
            if max_steps > 0:
                total_steps = max_steps
            else:
                total_steps = int(getattr(trainer, "estimated_stepping_batches", 0) or 0)
        if total_steps <= 0:
            return torch.tensor(0.0, device=device)
        start_step = int(max(0, min(total_steps, math.floor(float(start_frac) * float(total_steps)))))
        current_step = int(getattr(self, "global_step", 0))
        if current_step < start_step:
            return torch.tensor(0.0, device=device)
        ramp_denom = max(total_steps - start_step, 1)
        ramp = min(max(float(current_step - start_step) / float(ramp_denom), 0.0), 1.0)
        return torch.tensor(float(max_weight) * ramp, device=device, dtype=torch.float32)

    def _compute_semantic_ext_traffic_speed_floor_bundle(
        self,
        *,
        data_dict,
        semantic_context,
        sdc_next_pos_world,
        valid_mask,
        reference_batch=None,
        frontier_arc_t=None,
    ):
        zero_t = torch.zeros_like(valid_mask, dtype=torch.float32)
        zero = zero_t.new_tensor(0.0)
        floor_enabled = bool(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_FLOOR_ENABLED", False)
        )
        cap_enabled = bool(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_CAP_ENABLED", False)
        )

        def _zero_bundle():
            return {
                "penalty_t": zero_t,
                "floor_penalty_t": zero_t,
                "cap_penalty_t": zero_t,
                "reference_speed_t": zero_t,
                "neighbor_mean_speed_t": zero_t,
                "neighbor_median_speed_t": zero_t,
                "floor_threshold_t": zero_t,
                "cap_threshold_t": zero_t,
                "sdc_speed_t": zero_t,
                "neighbor_count_t": zero_t,
                "penalty_mean": zero,
                "floor_penalty_mean": zero,
                "cap_penalty_mean": zero,
                "reference_speed_mean": zero,
                "neighbor_mean_speed_mean": zero,
                "neighbor_median_speed_mean": zero,
                "floor_threshold_mean": zero,
                "cap_threshold_mean": zero,
                "sdc_speed_mean": zero,
                "neighbor_count_mean": zero,
            }

        if not self.sdc_semantic_ext_enabled:
            return _zero_bundle()

        if not (floor_enabled or cap_enabled):
            return _zero_bundle()

        if semantic_context is None or "decoder/rollout_next_position" not in data_dict:
            return _zero_bundle()

        device = sdc_next_pos_world.device
        dtype = sdc_next_pos_world.dtype
        horizon = int(valid_mask.shape[1])
        floor_weight = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_FLOOR_WEIGHT", 0.05)
        )
        cap_weight = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_CAP_WEIGHT", 0.05)
        )
        cap_margin_mps = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_CAP_MARGIN_MPS", 2.0)
        )
        nearby_radius_m = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_NEARBY_RADIUS_M", 30.0)
        )
        expand_factor = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_EXPAND_FACTOR", 2.0)
        )
        expand_steps = max(
            1,
            int(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_MAX_EXPANSION_STEPS", 3)),
        )
        min_moving_speed_mps = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_MIN_MOVING_SPEED_MPS", 1.0)
        )
        reference_ratio = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_REFERENCE_RATIO", 0.60)
        )
        max_reference_speed_mps = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_MAX_REFERENCE_MPS", 12.0)
        )
        min_neighbors = max(
            1,
            int(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_MIN_NEIGHBORS", 1)),
        )
        use_progress_arc_speed = bool(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_SDC_USE_PROGRESS_ARC", False)
        )
        progress_window_steps = max(
            1,
            int(
                self.config.MODEL.get(
                    "LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_SDC_PROGRESS_WINDOW_STEPS",
                    4,
                )
            ),
        )
        if (
            (floor_enabled and floor_weight <= 0.0 and (not cap_enabled or cap_weight <= 0.0))
            or (cap_enabled and cap_weight <= 0.0 and (not floor_enabled or floor_weight <= 0.0))
            or nearby_radius_m <= 0.0
            or (floor_enabled and reference_ratio <= 0.0)
            or max_reference_speed_mps <= 0.0
            or expand_factor < 1.0
            or (cap_enabled and cap_margin_mps < 0.0)
        ):
            return _zero_bundle()

        rollout_next_position = self._as_tensor(
            data_dict["decoder/rollout_next_position"][:, :horizon, :, :2],
            device=device,
            dtype=dtype,
        )
        current_position = self._as_tensor(
            data_dict.get("decoder/modeled_agent_position", torch.zeros_like(rollout_next_position))[:, :horizon, :, :2],
            device=device,
            dtype=dtype,
        )
        if "decoder/input_action_valid_mask" in data_dict:
            agent_valid_mask = self._as_tensor(
                data_dict["decoder/input_action_valid_mask"][:, :horizon],
                device=device,
                dtype=torch.bool,
            )
        else:
            agent_valid_mask = torch.ones(
                rollout_next_position.shape[:3],
                device=device,
                dtype=torch.bool,
            )

        decision_agent_mask = semantic_context["decision_agent_mask"][:, : rollout_next_position.shape[2]].to(dtype=torch.bool)
        non_sdc_mask = (~decision_agent_mask)[:, None, :]
        dt = max(self._get_rollout_step_dt_s(), 1e-3)
        realized_delta_t = rollout_next_position.to(dtype=torch.float32) - current_position.to(dtype=torch.float32)
        all_speed_t = torch.linalg.norm(realized_delta_t, dim=-1) / float(dt)
        all_speed_t = torch.nan_to_num(all_speed_t, nan=0.0, posinf=0.0, neginf=0.0)
        sdc_realized_speed_t = (
            all_speed_t
            * decision_agent_mask[:, None, :].to(dtype=torch.float32)
        ).sum(dim=-1)
        sdc_speed_t = sdc_realized_speed_t
        if use_progress_arc_speed and frontier_arc_t is not None:
            frontier_arc = self._as_tensor(
                frontier_arc_t[:, :horizon],
                device=device,
                dtype=torch.float32,
            )
            shifted_frontier = torch.zeros_like(frontier_arc)
            if progress_window_steps < int(frontier_arc.shape[1]):
                shifted_frontier[:, progress_window_steps:] = frontier_arc[:, :-progress_window_steps]
            step_idx = torch.arange(frontier_arc.shape[1], device=device, dtype=torch.float32).view(1, -1)
            effective_window = torch.clamp(step_idx + 1.0, max=float(progress_window_steps))
            sdc_speed_t = torch.relu(frontier_arc - shifted_frontier) / (
                effective_window * float(dt)
            ).clamp_min(1e-3)
            sdc_speed_t = torch.nan_to_num(sdc_speed_t, nan=0.0, posinf=0.0, neginf=0.0)
        distance_to_sdc = torch.linalg.norm(
            rollout_next_position.to(dtype=torch.float32) - sdc_next_pos_world[:, :, None, :].to(dtype=torch.float32),
            dim=-1,
        )
        moving_neighbor_mask = (
            agent_valid_mask
            & non_sdc_mask
            & torch.isfinite(all_speed_t)
            & torch.isfinite(distance_to_sdc)
            & (all_speed_t >= float(min_moving_speed_mps))
        )
        reference_mask = torch.zeros_like(moving_neighbor_mask, dtype=torch.bool)
        reference_found = torch.zeros_like(valid_mask, dtype=torch.bool)
        current_radius = float(nearby_radius_m)
        for _ in range(int(expand_steps)):
            candidate_mask = moving_neighbor_mask & (distance_to_sdc <= float(current_radius))
            candidate_found = candidate_mask.sum(dim=-1) >= int(min_neighbors)
            new_found = (~reference_found) & candidate_found
            reference_mask = reference_mask | (candidate_mask & new_found[:, :, None])
            reference_found = reference_found | candidate_found
            current_radius = float(current_radius) * float(expand_factor)

        scenewide_found = (~reference_found) & moving_neighbor_mask.any(dim=-1)
        reference_mask = reference_mask | (moving_neighbor_mask & scenewide_found[:, :, None])
        reference_found = reference_found | scenewide_found

        neighbor_count_t = reference_mask.sum(dim=-1).to(dtype=torch.float32)
        masked_speed_t = torch.where(
            reference_mask,
            all_speed_t,
            torch.full_like(all_speed_t, float("nan")),
        )
        neighbor_median_speed_t = torch.nanmedian(masked_speed_t, dim=-1).values
        neighbor_median_speed_t = torch.nan_to_num(
            neighbor_median_speed_t,
            nan=0.0,
            posinf=float(max_reference_speed_mps),
            neginf=0.0,
        ).clamp(min=0.0, max=float(max_reference_speed_mps))
        neighbor_median_speed_t = torch.where(
            reference_found,
            neighbor_median_speed_t,
            torch.zeros_like(neighbor_median_speed_t),
        )
        masked_speed_zero = torch.where(reference_mask, all_speed_t, torch.zeros_like(all_speed_t))
        neighbor_mean_speed_t = masked_speed_zero.sum(dim=-1) / neighbor_count_t.clamp_min(1.0)
        neighbor_mean_speed_t = torch.nan_to_num(
            neighbor_mean_speed_t,
            nan=0.0,
            posinf=float(max_reference_speed_mps),
            neginf=0.0,
        ).clamp(min=0.0, max=float(max_reference_speed_mps))
        neighbor_mean_speed_t = torch.where(
            reference_found,
            neighbor_mean_speed_t,
            torch.zeros_like(neighbor_mean_speed_t),
        )
        floor_reference_speed_t = neighbor_median_speed_t
        cap_reference_speed_t = neighbor_mean_speed_t
        gt_reference_speed_t = torch.zeros_like(floor_reference_speed_t)
        fallback_mask = (~reference_found) & valid_mask
        if floor_enabled and bool(fallback_mask.any()):
            source_batch = reference_batch if reference_batch is not None else data_dict
            gt_speed_mean = torch.zeros((valid_mask.shape[0],), device=device, dtype=torch.float32)
            if "decoder/agent_velocity" in source_batch and "decoder/agent_valid_mask" in source_batch:
                gt_velocity = self._as_tensor(
                    source_batch["decoder/agent_velocity"][:, :horizon, :, :2],
                    device=device,
                    dtype=dtype,
                )
                gt_valid_mask = self._as_tensor(
                    source_batch["decoder/agent_valid_mask"][:, :horizon],
                    device=device,
                    dtype=torch.bool,
                )
                gt_speed_t = (
                    torch.linalg.norm(gt_velocity.to(dtype=torch.float32), dim=-1)
                    * decision_agent_mask[:, None, :].to(dtype=torch.float32)
                ).sum(dim=-1)
                gt_sdc_valid_t = (gt_valid_mask & decision_agent_mask[:, None, :]).any(dim=-1)
                gt_speed_mean = self._masked_mean_per_example(gt_speed_t, gt_sdc_valid_t)
            elif "decoder/agent_position" in source_batch and "decoder/agent_valid_mask" in source_batch:
                gt_position = self._as_tensor(
                    source_batch["decoder/agent_position"][:, :horizon, :, :2],
                    device=device,
                    dtype=dtype,
                )
                gt_valid_mask = self._as_tensor(
                    source_batch["decoder/agent_valid_mask"][:, :horizon],
                    device=device,
                    dtype=torch.bool,
                )
                if int(gt_position.shape[1]) > 1:
                    dt = max(self._get_rollout_step_dt_s(), 1e-3)
                    gt_delta = gt_position[:, 1:] - gt_position[:, :-1]
                    gt_speed_step = torch.linalg.norm(gt_delta.to(dtype=torch.float32), dim=-1) / float(dt)
                    gt_speed_t = (
                        gt_speed_step * decision_agent_mask[:, None, :].to(dtype=torch.float32)
                    ).sum(dim=-1)
                    gt_sdc_valid_t = (gt_valid_mask[:, 1:] & gt_valid_mask[:, :-1] & decision_agent_mask[:, None, :]).any(dim=-1)
                    gt_speed_mean = self._masked_mean_per_example(gt_speed_t, gt_sdc_valid_t)
            gt_reference_speed_t = gt_speed_mean[:, None].expand_as(floor_reference_speed_t).clamp(
                min=0.0,
                max=float(max_reference_speed_mps),
            )
            floor_reference_speed_t = torch.where(
                fallback_mask,
                gt_reference_speed_t,
                floor_reference_speed_t,
            )
        floor_active = valid_mask & ((neighbor_count_t >= float(min_neighbors)) | fallback_mask) & bool(floor_enabled)
        speed_floor_t = float(reference_ratio) * floor_reference_speed_t
        floor_penalty_t = torch.where(
            floor_active,
            torch.relu(speed_floor_t - sdc_speed_t) * float(floor_weight),
            torch.zeros_like(floor_reference_speed_t),
        )
        cap_active = valid_mask & reference_found & bool(cap_enabled)
        speed_cap_t = (cap_reference_speed_t + float(cap_margin_mps)).clamp(
            min=0.0,
            max=float(max_reference_speed_mps),
        )
        cap_penalty_t = torch.where(
            cap_active,
            torch.relu(sdc_speed_t - speed_cap_t) * float(cap_weight),
            torch.zeros_like(cap_reference_speed_t),
        )
        reference_speed_t = cap_reference_speed_t if cap_enabled else floor_reference_speed_t
        penalty_t = floor_penalty_t + cap_penalty_t
        return {
            "penalty_t": penalty_t,
            "reference_speed_t": reference_speed_t,
            "neighbor_mean_speed_t": neighbor_mean_speed_t,
            "neighbor_median_speed_t": neighbor_median_speed_t,
            "floor_threshold_t": speed_floor_t,
            "cap_threshold_t": speed_cap_t,
            "floor_penalty_t": floor_penalty_t,
            "cap_penalty_t": cap_penalty_t,
            "sdc_speed_t": sdc_speed_t,
            "neighbor_count_t": neighbor_count_t,
            "penalty_mean": sanitize_scalar_loss(penalty_t[valid_mask].mean()) if bool(valid_mask.any()) else zero,
            "floor_penalty_mean": sanitize_scalar_loss(floor_penalty_t[floor_active].mean()) if bool(floor_active.any()) else zero,
            "cap_penalty_mean": sanitize_scalar_loss(cap_penalty_t[cap_active].mean()) if bool(cap_active.any()) else zero,
            "reference_speed_mean": sanitize_scalar_loss(reference_speed_t[valid_mask].mean()) if bool(valid_mask.any()) else zero,
            "neighbor_mean_speed_mean": sanitize_scalar_loss(neighbor_mean_speed_t[reference_found].mean()) if bool(reference_found.any()) else zero,
            "neighbor_median_speed_mean": sanitize_scalar_loss(neighbor_median_speed_t[reference_found].mean()) if bool(reference_found.any()) else zero,
            "floor_threshold_mean": sanitize_scalar_loss(speed_floor_t[floor_active].mean()) if bool(floor_active.any()) else zero,
            "cap_threshold_mean": sanitize_scalar_loss(speed_cap_t[cap_active].mean()) if bool(cap_active.any()) else zero,
            "sdc_speed_mean": sanitize_scalar_loss(sdc_speed_t[valid_mask].mean()) if bool(valid_mask.any()) else zero,
            "neighbor_count_mean": sanitize_scalar_loss(neighbor_count_t[valid_mask].mean()) if bool(valid_mask.any()) else zero,
        }

    def _compute_sdc_semantic_ext_policy_kl(self, *, data_dict, semantic_context, teacher_logits):
        output_logit = data_dict["decoder/output_logit"]
        zero = output_logit.new_tensor(0.0)
        if (
            not self.sdc_semantic_ext_enabled
            or semantic_context is None
            or teacher_logits is None
        ):
            return {
                "kl_gt": zero,
                "kl_alt": zero,
                "kl_total": zero,
                "gt_token_fraction": zero,
                "alt_token_fraction": zero,
            }

        decision_agent_mask = semantic_context["decision_agent_mask"][:, : output_logit.shape[2]]
        decision_mask_f = decision_agent_mask[:, None, :, None].to(dtype=output_logit.dtype)
        student_logits = sanitize_logits_for_loss(
            (output_logit * decision_mask_f).sum(dim=2)
        )
        teacher_logits = teacher_logits[:, : output_logit.shape[1], : output_logit.shape[2]]
        teacher_logits = teacher_logits.to(device=output_logit.device, dtype=output_logit.dtype)
        teacher_logits = sanitize_logits_for_loss(
            (teacher_logits * decision_mask_f).sum(dim=2)
        )

        kl_per_token = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="none",
        ).sum(dim=-1).to(dtype=torch.float32)

        sdc_valid_by_t = semantic_context["sdc_valid_by_t"]
        factual_mask = semantic_context["factual_batch_mask"][:, None]
        alt_mask = semantic_context["alternative_batch_mask"][:, None]
        gt_valid = sdc_valid_by_t & factual_mask
        alt_valid = sdc_valid_by_t & alt_mask

        kl_gt = sanitize_scalar_loss(kl_per_token[gt_valid].mean()) if bool(gt_valid.any()) else zero
        kl_alt = sanitize_scalar_loss(kl_per_token[alt_valid].mean()) if bool(alt_valid.any()) else zero
        kl_weight_gt = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_KL_WEIGHT_GT", 0.10))
        kl_weight_alt = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_KL_WEIGHT_ALT", 0.02))
        kl_total = sanitize_scalar_loss(
            float(kl_weight_gt) * kl_gt + float(kl_weight_alt) * kl_alt
        )
        return {
            "kl_gt": kl_gt,
            "kl_alt": kl_alt,
            "kl_total": kl_total,
            "gt_token_fraction": gt_valid.float().mean(),
            "alt_token_fraction": alt_valid.float().mean(),
        }

    def on_save_checkpoint(self, checkpoint):
        if self.semantic_ext_novelty_bank is not None:
            checkpoint["semantic_ext_novelty_bank"] = self.semantic_ext_novelty_bank.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.semantic_ext_novelty_bank is not None:
            self.semantic_ext_novelty_bank.load_state_dict(checkpoint.get("semantic_ext_novelty_bank"))

    @staticmethod
    def _semantic_ext_round_ids(values, *, max_id):
        values = torch.nan_to_num(values.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return values.round().to(dtype=torch.long).clamp(min=0, max=int(max_id))

    def _semantic_ext_bucket_keys(
        self,
        *,
        semantic_label_id,
        path_token,
        compliance_token,
        timing_token,
    ):
        branch_id = self._semantic_ext_round_ids(path_token[:, 0], max_id=len(BRANCH_LABEL_ORDER) - 1)
        signal_id = self._semantic_ext_round_ids(compliance_token[:, 0], max_id=len(SIGNAL_CATEGORY_ORDER) - 1)
        compliance_id = self._semantic_ext_round_ids(compliance_token[:, 1], max_id=len(COMPLIANCE_LABEL_ORDER) - 1)
        timing_id = self._semantic_ext_round_ids(timing_token[:, 2], max_id=len(TIMING_LABEL_ORDER) - 1)
        has_conflict = self._semantic_ext_round_ids(timing_token[:, 0], max_id=1)
        keys = []
        for idx in range(int(semantic_label_id.shape[0])):
            keys.append(
                (
                    int(semantic_label_id[idx].item()),
                    int(branch_id[idx].item()),
                    int(signal_id[idx].item()),
                    int(compliance_id[idx].item()),
                    int(timing_id[idx].item()),
                    int(has_conflict[idx].item()),
                )
            )
        return keys

    def _semantic_ext_target_entropy(
        self,
        *,
        semantic_label_id,
        path_token,
        compliance_token,
        timing_token,
        num_active,
    ):
        max_entropy = math.log(float(max(num_active, 1)))
        stop_label_id = int(SDC_PATH_SEMANTIC_LABEL_ORDER.index("stop"))
        stop_target = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_THERMOSTAT_TARGET_STOP", 0.10))
        if int(semantic_label_id) == stop_label_id:
            return max(0.0, min(float(stop_target), max_entropy))
        base = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_THERMOSTAT_TARGET_BASE", 0.40))
        conflict_bonus = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_THERMOSTAT_TARGET_CONFLICT_BONUS", 0.25))
        after_conflict_bonus = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_THERMOSTAT_TARGET_AFTER_CONFLICT_BONUS", 0.15)
        )
        turn_bonus = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_THERMOSTAT_TARGET_TURN_BONUS", 0.05))
        permissive_signal_bonus = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_THERMOSTAT_TARGET_PERMISSIVE_SIGNAL_BONUS", 0.10)
        )
        violation_penalty = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_THERMOSTAT_TARGET_VIOLATION_PENALTY", 0.10)
        )
        branch_id = int(self._semantic_ext_round_ids(path_token[:1, 0], max_id=len(BRANCH_LABEL_ORDER) - 1)[0].item())
        signal_id = int(self._semantic_ext_round_ids(compliance_token[:1, 0], max_id=len(SIGNAL_CATEGORY_ORDER) - 1)[0].item())
        compliance_id = int(self._semantic_ext_round_ids(compliance_token[:1, 1], max_id=len(COMPLIANCE_LABEL_ORDER) - 1)[0].item())
        timing_id = int(self._semantic_ext_round_ids(timing_token[:1, 2], max_id=len(TIMING_LABEL_ORDER) - 1)[0].item())
        has_conflict = bool(self._semantic_ext_round_ids(timing_token[:1, 0], max_id=1)[0].item() > 0)
        target = float(base)
        if branch_id in {
            int(BRANCH_LABEL_ORDER.index("left")),
            int(BRANCH_LABEL_ORDER.index("right")),
            int(BRANCH_LABEL_ORDER.index("u_turn")),
        }:
            target += float(turn_bonus)
        if has_conflict:
            target += float(conflict_bonus)
        if timing_id == int(TIMING_LABEL_ORDER.index("after_conflict")):
            target += float(after_conflict_bonus)
        if signal_id in {
            int(SIGNAL_CATEGORY_ORDER.index("go")),
            int(SIGNAL_CATEGORY_ORDER.index("caution")),
            int(SIGNAL_CATEGORY_ORDER.index("unknown")),
        }:
            target += float(permissive_signal_bonus)
        if compliance_id == int(COMPLIANCE_LABEL_ORDER.index("red_light_violation")):
            target -= float(violation_penalty)
        return max(0.0, min(float(target), max_entropy))

    def _build_semantic_ext_rollout_features(
        self,
        *,
        tube_distance,
        frontier_arc_t,
        trajectories_world,
        current_heading_world,
    ):
        dt = max(self._get_rollout_step_dt_s(), 1e-3)
        delta_pos = trajectories_world[:, :, 1:] - trajectories_world[:, :, :-1]
        speed_t = torch.linalg.norm(delta_pos.to(dtype=torch.float32), dim=-1) / float(dt)
        prev_speed = torch.cat([speed_t[:, :, :1], speed_t[:, :, :-1]], dim=-1)
        accel_t = (speed_t - prev_speed) / float(dt)
        prev_accel = torch.cat([accel_t[:, :, :1], accel_t[:, :, :-1]], dim=-1)
        jerk_t = (accel_t - prev_accel) / float(dt)
        move_heading = torch.atan2(delta_pos[..., 1], delta_pos[..., 0])
        initial_heading = current_heading_world.to(dtype=torch.float32)[None, :, None]
        prev_heading = torch.cat(
            [initial_heading.expand(move_heading.shape[0], -1, 1), move_heading[:, :, :-1]],
            dim=2,
        )
        yaw_rate_t = self._wrap_angle(move_heading - prev_heading).abs() / float(dt)
        prev_arc = torch.cat(
            [torch.zeros_like(frontier_arc_t[:, :, :1], dtype=torch.float32), frontier_arc_t[:, :, :-1].to(dtype=torch.float32)],
            dim=-1,
        )
        arc_delta_t = torch.relu(frontier_arc_t.to(dtype=torch.float32) - prev_arc)
        inside_t = (tube_distance <= float(self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_TUBE_RADIUS_M", 3.0))).to(dtype=torch.float32)
        feature_t = torch.stack(
            [
                frontier_arc_t.to(dtype=torch.float32) / 50.0,
                arc_delta_t / 5.0,
                tube_distance.to(dtype=torch.float32) / 3.0,
                inside_t,
                speed_t / 10.0,
                accel_t / 4.0,
                jerk_t / 8.0,
                yaw_rate_t / 1.0,
            ],
            dim=-1,
        )
        return torch.nan_to_num(feature_t, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_semantic_ext_behavior_aux_loss(
        self,
        *,
        behavior_embedding,
        dag_embedding,
        semantic_label_id,
        path_token,
        compliance_token,
        timing_token,
        active_mask,
    ):
        zero = behavior_embedding.new_tensor(0.0)
        if (
            self.semantic_ext_behavior_encoder is None
            or self.semantic_ext_semantic_head is None
            or not bool(active_mask.any())
        ):
            return {
                "loss": zero,
                "semantic_ce": zero,
                "branch_ce": zero,
                "compliance_ce": zero,
                "timing_ce": zero,
                "dag_align": zero,
            }
        active_idx = torch.nonzero(active_mask, as_tuple=False).reshape(-1)
        active_behavior = behavior_embedding[active_idx]
        active_dag = dag_embedding[active_idx]
        semantic_target = semantic_label_id[active_idx].to(dtype=torch.long)
        branch_target = self._semantic_ext_round_ids(path_token[active_idx, 0], max_id=len(BRANCH_LABEL_ORDER) - 1)
        compliance_target = self._semantic_ext_round_ids(compliance_token[active_idx, 1], max_id=len(COMPLIANCE_LABEL_ORDER) - 1)
        timing_target = self._semantic_ext_round_ids(timing_token[active_idx, 2], max_id=len(TIMING_LABEL_ORDER) - 1)
        semantic_ce = F.cross_entropy(self.semantic_ext_semantic_head(active_behavior), semantic_target)
        branch_ce = F.cross_entropy(self.semantic_ext_branch_head(active_behavior), branch_target)
        compliance_ce = F.cross_entropy(self.semantic_ext_compliance_head(active_behavior), compliance_target)
        timing_ce = F.cross_entropy(self.semantic_ext_timing_head(active_behavior), timing_target)
        dag_align = 1.0 - F.cosine_similarity(active_behavior, active_dag, dim=-1).mean()
        loss = (
            float(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_BEHAVIOR_SEMANTIC_CE_WEIGHT", 0.05)) * semantic_ce
            + float(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_BEHAVIOR_BRANCH_CE_WEIGHT", 0.025)) * branch_ce
            + float(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_BEHAVIOR_COMPLIANCE_CE_WEIGHT", 0.025)) * compliance_ce
            + float(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_BEHAVIOR_TIMING_CE_WEIGHT", 0.025)) * timing_ce
            + float(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_BEHAVIOR_DAG_ALIGN_WEIGHT", 0.05)) * dag_align
        )
        return {
            "loss": sanitize_scalar_loss(loss),
            "semantic_ce": sanitize_scalar_loss(semantic_ce),
            "branch_ce": sanitize_scalar_loss(branch_ce),
            "compliance_ce": sanitize_scalar_loss(compliance_ce),
            "timing_ce": sanitize_scalar_loss(timing_ce),
            "dag_align": sanitize_scalar_loss(dag_align),
        }

    def _compute_rollout_topomcpo_regularization_bundle(
        self,
        *,
        data_dict,
        reward_t,
        valid_mask,
        tube_distance,
        frontier_arc_t,
        trajectories_world,
        current_heading_world,
    ):
        zero_bonus = torch.zeros_like(reward_t, dtype=torch.float32)
        zero = reward_t.new_tensor(0.0)
        legacy = self._compute_rollout_terminal_consensus_bonus(
            reward_t=reward_t,
            valid_mask=valid_mask,
            tube_distance=tube_distance,
            frontier_arc_t=frontier_arc_t,
            trajectories_world=trajectories_world,
            current_heading_world=current_heading_world,
        )
        default_bundle = {
            "bonus_t": legacy["bonus_t"],
            "consensus_bonus_t": legacy["bonus_t"],
            "novelty_bonus_t": zero_bonus,
            "bonus_mean": legacy["bonus_mean"],
            "consensus_bonus_mean": legacy["bonus_mean"],
            "novelty_bonus_mean": zero,
            "novelty_score_mean": zero,
            "cluster_count_mean": legacy["cluster_count_mean"],
            "cluster_entropy_mean": legacy["cluster_entropy_mean"],
            "target_entropy_mean": zero,
            "thermostat_alpha_mean": zero,
            "thermostat_eta_mean": zero,
            "behavior_aux_loss": zero,
            "behavior_semantic_ce": zero,
            "behavior_branch_ce": zero,
            "behavior_compliance_ce": zero,
            "behavior_timing_ce": zero,
            "behavior_dag_align": zero,
        }
        if self.semantic_ext_behavior_encoder is None or self.semantic_ext_dag_encoder is None:
            return default_bundle

        path_token = self._as_tensor(
            data_dict.get("cf/path_token", reward_t.new_zeros((reward_t.shape[1], PATH_TOKEN_DIM))),
            device=reward_t.device,
            dtype=torch.float32,
        ).reshape(reward_t.shape[1], -1)
        compliance_token = self._as_tensor(
            data_dict.get("cf/compliance_token", reward_t.new_zeros((reward_t.shape[1], COMPLIANCE_TOKEN_DIM))),
            device=reward_t.device,
            dtype=torch.float32,
        ).reshape(reward_t.shape[1], -1)
        timing_token = self._as_tensor(
            data_dict.get("cf/timing_token", reward_t.new_zeros((reward_t.shape[1], TIMING_TOKEN_DIM))),
            device=reward_t.device,
            dtype=torch.float32,
        ).reshape(reward_t.shape[1], -1)
        semantic_label_id = self._as_tensor(
            data_dict.get("cf/sdc_semantic_label_id", reward_t.new_zeros((reward_t.shape[1],))),
            device=reward_t.device,
            dtype=torch.long,
        ).reshape(reward_t.shape[1], -1)[:, 0]

        feature_t = self._build_semantic_ext_rollout_features(
            tube_distance=tube_distance,
            frontier_arc_t=frontier_arc_t,
            trajectories_world=trajectories_world,
            current_heading_world=current_heading_world,
        )
        group_size, batch_size, horizon, feature_dim = feature_t.shape
        flat_size = int(group_size * batch_size)
        flat_feature = feature_t.reshape(flat_size, horizon, feature_dim)
        flat_valid = valid_mask.reshape(flat_size, horizon)
        flat_behavior = self.semantic_ext_behavior_encoder(flat_feature, flat_valid)
        repeated_path = path_token[None, :, :].expand(group_size, -1, -1).reshape(flat_size, -1)
        repeated_compliance = compliance_token[None, :, :].expand(group_size, -1, -1).reshape(flat_size, -1)
        repeated_timing = timing_token[None, :, :].expand(group_size, -1, -1).reshape(flat_size, -1)
        repeated_semantic = semantic_label_id[None, :].expand(group_size, -1).reshape(flat_size)
        flat_dag = self.semantic_ext_dag_encoder(
            path_token=repeated_path,
            compliance_token=repeated_compliance,
            timing_token=repeated_timing,
        )
        flat_valid_return = valid_mask.any(dim=-1).reshape(flat_size)
        behavior_aux_bundle = self._compute_semantic_ext_behavior_aux_loss(
            behavior_embedding=flat_behavior,
            dag_embedding=flat_dag,
            semantic_label_id=repeated_semantic,
            path_token=repeated_path,
            compliance_token=repeated_compliance,
            timing_token=repeated_timing,
            active_mask=flat_valid_return,
        )
        if not bool(flat_valid_return.any()):
            default_bundle.update(
                {
                    "behavior_aux_loss": behavior_aux_bundle["loss"],
                    "behavior_semantic_ce": behavior_aux_bundle["semantic_ce"],
                    "behavior_branch_ce": behavior_aux_bundle["branch_ce"],
                    "behavior_compliance_ce": behavior_aux_bundle["compliance_ce"],
                    "behavior_timing_ce": behavior_aux_bundle["timing_ce"],
                    "behavior_dag_align": behavior_aux_bundle["dag_align"],
                }
            )
            return default_bundle

        total_return = reward_t.sum(dim=-1)
        valid_return = valid_mask.any(dim=-1)
        normalized_return = self._group_normalize_advantages(total_return, valid_return).reshape(flat_size)
        flat_batch_id = torch.arange(batch_size, device=reward_t.device)[None, :].expand(group_size, -1).reshape(flat_size)
        novelty_scores_flat = reward_t.new_zeros((flat_size,), dtype=torch.float32)
        active_idx = torch.nonzero(flat_valid_return, as_tuple=False).reshape(-1)
        bucket_keys = self._semantic_ext_bucket_keys(
            semantic_label_id=repeated_semantic[active_idx],
            path_token=repeated_path[active_idx],
            compliance_token=repeated_compliance[active_idx],
            timing_token=repeated_timing[active_idx],
        )
        novelty_enabled = bool(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_NOVELTY_ENABLED", False))
        if novelty_enabled and self.semantic_ext_novelty_bank is not None:
            novelty_scores = self.semantic_ext_novelty_bank.score(
                flat_behavior[active_idx].detach(),
                bucket_keys,
                warmup_count=int(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_NOVELTY_WARMUP_COUNT", 32)),
            )
            novelty_scores_flat[active_idx] = novelty_scores.to(device=reward_t.device, dtype=torch.float32)
            if self.training:
                self.semantic_ext_novelty_bank.update(flat_behavior[active_idx].detach(), bucket_keys)

        cluster_radius = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_CONSENSUS_CLUSTER_RADIUS", 0.75))
        consensus_base = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_CONSENSUS_WEIGHT", 0.10))
        novelty_base = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_NOVELTY_BASE_WEIGHT", 0.05))
        thermostat_enabled = bool(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_THERMOSTAT_ENABLED", False))
        thermostat_k_alpha = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_THERMOSTAT_K_ALPHA", 0.10))
        thermostat_k_eta = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_THERMOSTAT_K_ETA", 0.10))
        novelty_clamp = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_NOVELTY_STD_CLAMP", 2.0))

        consensus_bonus_t = torch.zeros_like(reward_t, dtype=torch.float32)
        novelty_bonus_t = torch.zeros_like(reward_t, dtype=torch.float32)
        cluster_count_values = []
        cluster_entropy_values = []
        target_entropy_values = []
        thermostat_alpha_values = []
        thermostat_eta_values = []

        for batch_idx in range(batch_size):
            local_active = torch.nonzero(flat_valid_return & (flat_batch_id == batch_idx), as_tuple=False).reshape(-1)
            if int(local_active.numel()) <= 0:
                continue
            embeddings = flat_behavior[local_active].detach()
            norm_returns = normalized_return[local_active].to(dtype=torch.float32)
            assignments = torch.full((int(local_active.numel()),), -1, device=reward_t.device, dtype=torch.long)
            cluster_members = []
            for idx in range(int(local_active.numel())):
                if int(assignments[idx].item()) >= 0:
                    continue
                center = embeddings[idx]
                distance = torch.linalg.norm(embeddings - center[None, :], dim=-1)
                member_mask = (assignments < 0) & (distance <= float(cluster_radius))
                assignments[member_mask] = len(cluster_members)
                cluster_members.append(torch.nonzero(member_mask, as_tuple=False).reshape(-1))

            probs = [float(members.numel()) / float(max(int(local_active.numel()), 1)) for members in cluster_members]
            entropy = 0.0
            if probs:
                probs_t = torch.as_tensor(probs, device=reward_t.device, dtype=torch.float32)
                entropy = float((-(probs_t * torch.log(probs_t.clamp_min(1e-6))).sum()).item())
            target_entropy = self._semantic_ext_target_entropy(
                semantic_label_id=int(semantic_label_id[batch_idx].item()),
                path_token=path_token[batch_idx : batch_idx + 1],
                compliance_token=compliance_token[batch_idx : batch_idx + 1],
                timing_token=timing_token[batch_idx : batch_idx + 1],
                num_active=int(local_active.numel()),
            )
            alpha = float(consensus_base)
            eta = float(novelty_base if novelty_enabled else 0.0)
            if thermostat_enabled:
                alpha = max(0.0, alpha + float(thermostat_k_alpha) * float(entropy - target_entropy))
                eta = max(0.0, eta + float(thermostat_k_eta) * float(target_entropy - entropy))

            cluster_count_values.append(float(len(cluster_members)))
            cluster_entropy_values.append(float(entropy))
            target_entropy_values.append(float(target_entropy))
            thermostat_alpha_values.append(float(alpha))
            thermostat_eta_values.append(float(eta))

            for members in cluster_members:
                mass_fraction = float(members.numel()) / float(max(int(local_active.numel()), 1))
                cluster_quality = torch.clamp(norm_returns[members].mean(), min=0.0, max=1.0)
                bonus = float(alpha) * float(mass_fraction) * cluster_quality
                for local_member in members.tolist():
                    flat_idx = int(local_active[local_member].item())
                    rollout_idx = flat_idx // batch_size
                    valid_steps = torch.nonzero(valid_mask[rollout_idx, batch_idx], as_tuple=False).reshape(-1)
                    if int(valid_steps.numel()) <= 0:
                        continue
                    consensus_bonus_t[rollout_idx, batch_idx, int(valid_steps[-1].item())] += bonus

            if novelty_enabled and float(eta) > 0.0:
                local_scores = novelty_scores_flat[local_active]
                if int(local_scores.numel()) > 1:
                    standardized = (local_scores - local_scores.mean()) / local_scores.std(unbiased=False).clamp_min(1e-6)
                else:
                    standardized = torch.zeros_like(local_scores)
                novelty_signal = torch.relu(standardized).clamp(max=float(novelty_clamp))
                for local_member, bonus_value in enumerate((float(eta) * novelty_signal).tolist()):
                    if bonus_value <= 0.0:
                        continue
                    flat_idx = int(local_active[local_member].item())
                    rollout_idx = flat_idx // batch_size
                    valid_steps = torch.nonzero(valid_mask[rollout_idx, batch_idx], as_tuple=False).reshape(-1)
                    if int(valid_steps.numel()) <= 0:
                        continue
                    novelty_bonus_t[rollout_idx, batch_idx, int(valid_steps[-1].item())] += float(bonus_value)

        bonus_t = consensus_bonus_t + novelty_bonus_t
        cluster_count_mean = zero if not cluster_count_values else reward_t.new_tensor(sum(cluster_count_values) / len(cluster_count_values))
        cluster_entropy_mean = zero if not cluster_entropy_values else reward_t.new_tensor(sum(cluster_entropy_values) / len(cluster_entropy_values))
        target_entropy_mean = zero if not target_entropy_values else reward_t.new_tensor(sum(target_entropy_values) / len(target_entropy_values))
        thermostat_alpha_mean = zero if not thermostat_alpha_values else reward_t.new_tensor(sum(thermostat_alpha_values) / len(thermostat_alpha_values))
        thermostat_eta_mean = zero if not thermostat_eta_values else reward_t.new_tensor(sum(thermostat_eta_values) / len(thermostat_eta_values))
        bonus_mean = sanitize_scalar_loss(bonus_t[valid_mask].mean()) if bool(valid_mask.any()) else zero
        consensus_bonus_mean = sanitize_scalar_loss(consensus_bonus_t[valid_mask].mean()) if bool(valid_mask.any()) else zero
        novelty_bonus_mean = sanitize_scalar_loss(novelty_bonus_t[valid_mask].mean()) if bool(valid_mask.any()) else zero
        novelty_score_mean = sanitize_scalar_loss(novelty_scores_flat[flat_valid_return].mean()) if bool(flat_valid_return.any()) else zero
        return {
            "bonus_t": bonus_t,
            "consensus_bonus_t": consensus_bonus_t,
            "novelty_bonus_t": novelty_bonus_t,
            "bonus_mean": bonus_mean,
            "consensus_bonus_mean": consensus_bonus_mean,
            "novelty_bonus_mean": novelty_bonus_mean,
            "novelty_score_mean": novelty_score_mean,
            "cluster_count_mean": cluster_count_mean,
            "cluster_entropy_mean": cluster_entropy_mean,
            "target_entropy_mean": target_entropy_mean,
            "thermostat_alpha_mean": thermostat_alpha_mean,
            "thermostat_eta_mean": thermostat_eta_mean,
            "behavior_aux_loss": behavior_aux_bundle["loss"],
            "behavior_semantic_ce": behavior_aux_bundle["semantic_ce"],
            "behavior_branch_ce": behavior_aux_bundle["branch_ce"],
            "behavior_compliance_ce": behavior_aux_bundle["compliance_ce"],
            "behavior_timing_ce": behavior_aux_bundle["timing_ce"],
            "behavior_dag_align": behavior_aux_bundle["dag_align"],
        }

    def _compute_rollout_terminal_consensus_bonus(
        self,
        *,
        reward_t,
        valid_mask,
        tube_distance,
        frontier_arc_t,
        trajectories_world,
        current_heading_world,
    ):
        zero_bonus = torch.zeros_like(reward_t, dtype=torch.float32)
        zero = reward_t.new_tensor(0.0)
        if not self.sdc_semantic_ext_enabled:
            return {
                "bonus_t": zero_bonus,
                "bonus_mean": zero,
                "cluster_count_mean": zero,
                "cluster_entropy_mean": zero,
            }

        consensus_weight = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_CONSENSUS_WEIGHT", 0.10)
        )
        cluster_radius = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_CONSENSUS_CLUSTER_RADIUS", 0.75)
        )
        if consensus_weight <= 0.0 or cluster_radius <= 0.0:
            return {
                "bonus_t": zero_bonus,
                "bonus_mean": zero,
                "cluster_count_mean": zero,
                "cluster_entropy_mean": zero,
            }

        dt = max(self._get_rollout_step_dt_s(), 1e-3)
        valid_float = valid_mask.to(dtype=torch.float32)
        total_return = reward_t.sum(dim=-1)
        valid_return = valid_mask.any(dim=-1)
        normalized_return = self._group_normalize_advantages(total_return, valid_return)

        frontier_final = torch.where(
            valid_return,
            frontier_arc_t.max(dim=-1).values.to(dtype=torch.float32),
            torch.zeros_like(total_return, dtype=torch.float32),
        )
        inside_fraction = torch.where(
            valid_return,
            (valid_mask & (tube_distance <= float(self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_TUBE_RADIUS_M", 3.0)))).float().sum(dim=-1)
            / valid_float.sum(dim=-1).clamp_min(1.0),
            torch.zeros_like(total_return, dtype=torch.float32),
        )
        mean_tube_distance = self._masked_mean_per_example(tube_distance, valid_mask)

        delta_pos = trajectories_world[:, :, 1:] - trajectories_world[:, :, :-1]
        speed_t = torch.linalg.norm(delta_pos.to(dtype=torch.float32), dim=-1) / float(dt)
        mean_speed = self._masked_mean_per_example(speed_t, valid_mask)

        move_heading = torch.atan2(delta_pos[..., 1], delta_pos[..., 0])
        initial_heading = current_heading_world.to(dtype=torch.float32)[None, :, None]
        prev_heading = torch.cat([initial_heading.expand(move_heading.shape[0], -1, 1), move_heading[:, :, :-1]], dim=2)
        yaw_rate_t = self._wrap_angle(move_heading - prev_heading).abs() / float(dt)
        mean_abs_yaw_rate = self._masked_mean_per_example(yaw_rate_t, valid_mask)

        feature_stack = torch.stack(
            [
                frontier_final / 50.0,
                inside_fraction,
                mean_tube_distance / 3.0,
                mean_speed / 10.0,
                mean_abs_yaw_rate / 1.0,
            ],
            dim=-1,
        )
        feature_stack = torch.nan_to_num(feature_stack, nan=0.0, posinf=0.0, neginf=0.0)

        bonus_t = torch.zeros_like(reward_t, dtype=torch.float32)
        cluster_count_values = []
        cluster_entropy_values = []
        for batch_idx in range(int(feature_stack.shape[1])):
            active = torch.nonzero(valid_return[:, batch_idx], as_tuple=False).reshape(-1)
            if int(active.numel()) <= 0:
                continue
            embeddings = feature_stack[active, batch_idx]
            norm_returns = normalized_return[active, batch_idx].to(dtype=torch.float32)
            num_active = int(active.numel())
            assignments = torch.full((num_active,), -1, device=embeddings.device, dtype=torch.long)
            cluster_members = []
            for idx in range(num_active):
                if int(assignments[idx].item()) >= 0:
                    continue
                center = embeddings[idx]
                distance = torch.linalg.norm(embeddings - center[None, :], dim=-1)
                member_mask = (assignments < 0) & (distance <= float(cluster_radius))
                cluster_id = len(cluster_members)
                assignments[member_mask] = cluster_id
                cluster_members.append(torch.nonzero(member_mask, as_tuple=False).reshape(-1))

            cluster_count_values.append(float(len(cluster_members)))
            probs = []
            for members in cluster_members:
                probs.append(float(members.numel()) / float(max(num_active, 1)))
                mass_fraction = float(members.numel()) / float(max(num_active, 1))
                cluster_quality = torch.clamp(norm_returns[members].mean(), min=0.0, max=1.0)
                bonus = float(consensus_weight) * float(mass_fraction) * cluster_quality
                for local_idx in members.tolist():
                    rollout_idx = int(active[local_idx].item())
                    valid_steps = torch.nonzero(valid_mask[rollout_idx, batch_idx], as_tuple=False).reshape(-1)
                    if int(valid_steps.numel()) <= 0:
                        continue
                    bonus_t[rollout_idx, batch_idx, int(valid_steps[-1].item())] += bonus
            if probs:
                probs_t = torch.as_tensor(probs, device=feature_stack.device, dtype=torch.float32)
                cluster_entropy_values.append(float((-(probs_t * torch.log(probs_t.clamp_min(1e-6))).sum()).item()))

        cluster_count_mean = zero if not cluster_count_values else reward_t.new_tensor(sum(cluster_count_values) / len(cluster_count_values))
        cluster_entropy_mean = zero if not cluster_entropy_values else reward_t.new_tensor(sum(cluster_entropy_values) / len(cluster_entropy_values))
        bonus_mean = sanitize_scalar_loss(bonus_t[valid_mask].mean()) if bool(valid_mask.any()) else zero
        return {
            "bonus_t": bonus_t,
            "bonus_mean": bonus_mean,
            "cluster_count_mean": cluster_count_mean,
            "cluster_entropy_mean": cluster_entropy_mean,
        }

    def _compute_discounted_return_to_go(self, reward_t, valid_mask, *, discount):
        reward_t = reward_t.to(dtype=torch.float32)
        valid_mask = valid_mask.to(dtype=torch.bool)
        running = torch.zeros_like(reward_t[:, :, 0])
        rtg = torch.zeros_like(reward_t)
        gamma = float(discount)
        for step_idx in range(reward_t.shape[-1] - 1, -1, -1):
            reward_step = torch.where(valid_mask[:, :, step_idx], reward_t[:, :, step_idx], torch.zeros_like(running))
            running = reward_step + gamma * running
            running = torch.where(valid_mask[:, :, step_idx], running, torch.zeros_like(running))
            rtg[:, :, step_idx] = running
        return rtg

    def _group_normalize_advantages(self, values, valid_mask, *, eps=1e-6):
        values = values.to(dtype=torch.float32)
        valid_mask = valid_mask.to(dtype=torch.bool)
        valid_f = valid_mask.to(dtype=values.dtype)
        count = valid_f.sum(dim=0, keepdim=True)
        mean = (values * valid_f).sum(dim=0, keepdim=True) / count.clamp_min(1.0)
        centered = values - mean
        var = (centered.square() * valid_f).sum(dim=0, keepdim=True) / count.clamp_min(1.0)
        normalized = centered / torch.sqrt(var + float(eps))
        return torch.where(valid_mask, normalized, torch.zeros_like(normalized))

    def _extract_sdc_selected_rollout_log_probs(self, *, rollout_eval_dict, semantic_context):
        output_logit = rollout_eval_dict["decoder/output_logit"]
        decision_agent_mask = semantic_context["decision_agent_mask"][:, : output_logit.shape[2]]
        selected_action = self._as_tensor(
            rollout_eval_dict["decoder/input_action"][:, : output_logit.shape[1], : output_logit.shape[2]],
            device=output_logit.device,
            dtype=torch.long,
        )
        sdc_logits = sanitize_logits_for_loss(
            (output_logit * decision_agent_mask[:, None, :, None].to(dtype=output_logit.dtype)).sum(dim=2)
        )
        sdc_action = (selected_action * decision_agent_mask[:, None, :].to(dtype=torch.long)).sum(dim=2)
        return F.log_softmax(sdc_logits, dim=-1).gather(-1, sdc_action.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _sample_point_along_polyline_torch(points_xy: torch.Tensor, arc_m: torch.Tensor) -> torch.Tensor:
        points = points_xy.to(dtype=torch.float32)
        if points.shape[0] == 0:
            return points.new_zeros((2,), dtype=torch.float32)
        if points.shape[0] == 1:
            return points[0]
        segment_vec = points[1:] - points[:-1]
        segment_len = torch.linalg.norm(segment_vec, dim=-1)
        cumulative = torch.cat(
            [points.new_zeros((1,), dtype=torch.float32), torch.cumsum(segment_len, dim=0)],
            dim=0,
        )
        target = torch.clamp(
            arc_m.to(dtype=torch.float32),
            min=0.0,
            max=float(cumulative[-1].item()),
        )
        right_idx = int(torch.searchsorted(cumulative, target, right=True).item())
        if right_idx <= 0:
            return points[0]
        if right_idx >= points.shape[0]:
            return points[-1]
        left_idx = right_idx - 1
        left_arc = cumulative[left_idx]
        right_arc = cumulative[right_idx]
        denom = torch.clamp(right_arc - left_arc, min=1e-6)
        alpha = torch.clamp((target - left_arc) / denom, min=0.0, max=1.0)
        return (1.0 - alpha) * points[left_idx] + alpha * points[right_idx]

    @classmethod
    def _compute_sdc_progress_radius_cap_arc(
        cls,
        *,
        path_xy: torch.Tensor,
        path_mask: torch.Tensor,
        circle_center_xy: torch.Tensor,
        radius_from_divergence_m: float,
        path_total_arc: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(path_xy.shape[0])
        out = path_total_arc.to(dtype=torch.float32).clone()
        radius = max(float(radius_from_divergence_m), 1e-3)
        radius_sq = float(radius * radius)
        for batch_idx in range(batch_size):
            if not bool(path_mask[batch_idx].any()):
                out[batch_idx] = 0.0
                continue
            points = path_xy[batch_idx][path_mask[batch_idx]].to(dtype=torch.float32)
            if points.shape[0] < 2:
                out[batch_idx] = 0.0
                continue
            circle_center = circle_center_xy[batch_idx].to(dtype=torch.float32)
            if circle_center.numel() < 2 or not bool(torch.isfinite(circle_center).all()):
                out[batch_idx] = path_total_arc[batch_idx].to(dtype=torch.float32)
                continue
            segment_vec = points[1:] - points[:-1]
            segment_len = torch.linalg.norm(segment_vec, dim=-1)
            cumulative = torch.cat(
                [points.new_zeros((1,), dtype=torch.float32), torch.cumsum(segment_len, dim=0)],
                dim=0,
            )
            rel = points - circle_center[None, :]
            dist_sq = torch.sum(rel * rel, dim=-1)
            crossing_idx = None
            for idx in range(1, int(points.shape[0])):
                if float(dist_sq[idx].item()) >= radius_sq and float(dist_sq[idx - 1].item()) < radius_sq:
                    crossing_idx = idx
                    break
            if crossing_idx is None:
                out[batch_idx] = path_total_arc[batch_idx].to(dtype=torch.float32)
                continue
            seg_start = points[crossing_idx - 1] - circle_center
            seg_end = points[crossing_idx] - circle_center
            seg = seg_end - seg_start
            a = float(torch.dot(seg, seg).item())
            b = float(2.0 * torch.dot(seg_start, seg).item())
            c = float(torch.dot(seg_start, seg_start).item() - radius_sq)
            t = 1.0
            if a > 1e-8:
                disc = max(b * b - 4.0 * a * c, 0.0)
                root = math.sqrt(disc)
                candidates = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
                valid_t = [val for val in candidates if -1e-6 <= val <= 1.0 + 1e-6]
                if valid_t:
                    t = float(min(max(valid_t[0], 0.0), 1.0))
            crossing_arc = cumulative[crossing_idx - 1] + float(t) * segment_len[crossing_idx - 1]
            out[batch_idx] = torch.clamp(
                crossing_arc,
                min=0.0,
                max=path_total_arc[batch_idx].to(dtype=torch.float32),
            )
        return out.to(dtype=torch.float32)

    def _compute_sdc_rollout_tube_rewards(self, *, data_dict, semantic_context, reference_batch=None):
        output_logit = data_dict["decoder/output_logit"]
        device = output_logit.device
        dtype = output_logit.dtype
        zero = output_logit.new_tensor(0.0)
        semantic_ext_enabled = bool(self.sdc_semantic_ext_enabled)
        required = (
            "decoder/rollout_next_position",
            "cf/sdc_semantic_label_id",
            "cf/sdc_selected_raw_path_world",
            "cf/sdc_selected_raw_path_mask",
            "cf/sdc_selected_raw_path_segment_mask",
            "cf/sdc_family_divergence_onsets",
        )
        if semantic_context is None or any(key not in data_dict for key in required):
            return {
                "reward_t": output_logit.new_zeros((output_logit.shape[0], output_logit.shape[1]), dtype=torch.float32),
                "valid_mask": output_logit.new_zeros((output_logit.shape[0], output_logit.shape[1]), dtype=torch.bool),
                "tube_distance": output_logit.new_zeros((output_logit.shape[0], output_logit.shape[1]), dtype=torch.float32),
                "progress_reward_t": output_logit.new_zeros((output_logit.shape[0], output_logit.shape[1]), dtype=torch.float32),
                "frontier_arc_t": output_logit.new_zeros((output_logit.shape[0], output_logit.shape[1]), dtype=torch.float32),
                "stall_penalty_t": output_logit.new_zeros((output_logit.shape[0], output_logit.shape[1]), dtype=torch.float32),
                "center_penalty_t": output_logit.new_zeros((output_logit.shape[0], output_logit.shape[1]), dtype=torch.float32),
                "traffic_speed_penalty_t": output_logit.new_zeros((output_logit.shape[0], output_logit.shape[1]), dtype=torch.float32),
                "traffic_speed_reference_t": output_logit.new_zeros((output_logit.shape[0], output_logit.shape[1]), dtype=torch.float32),
                "traffic_speed_neighbor_mean_t": output_logit.new_zeros((output_logit.shape[0], output_logit.shape[1]), dtype=torch.float32),
                "traffic_speed_neighbor_median_t": output_logit.new_zeros((output_logit.shape[0], output_logit.shape[1]), dtype=torch.float32),
                "traffic_speed_floor_threshold_t": output_logit.new_zeros((output_logit.shape[0], output_logit.shape[1]), dtype=torch.float32),
                "traffic_speed_cap_threshold_t": output_logit.new_zeros((output_logit.shape[0], output_logit.shape[1]), dtype=torch.float32),
                "traffic_speed_cap_penalty_t": output_logit.new_zeros((output_logit.shape[0], output_logit.shape[1]), dtype=torch.float32),
                "traffic_speed_sdc_t": output_logit.new_zeros((output_logit.shape[0], output_logit.shape[1]), dtype=torch.float32),
                "traffic_speed_neighbor_count_t": output_logit.new_zeros((output_logit.shape[0], output_logit.shape[1]), dtype=torch.float32),
                "inside_fraction": zero,
                "first_step_reward_mean": zero,
                "progress_reward_mean": zero,
                "stall_penalty_mean": zero,
                "center_penalty_mean": zero,
                "traffic_speed_penalty_mean": zero,
                "traffic_speed_reference_mean": zero,
                "traffic_speed_neighbor_mean_mean": zero,
                "traffic_speed_neighbor_median_mean": zero,
                "traffic_speed_floor_penalty_mean": zero,
                "traffic_speed_cap_penalty_mean": zero,
                "traffic_speed_cap_threshold_mean": zero,
                "traffic_speed_sdc_mean": zero,
                "traffic_speed_neighbor_count_mean": zero,
                "center_curriculum_weight": zero,
                "frontier_arc_final_mean": zero,
                "progress_cap_arc_mean": zero,
                "divergence_onset_mean": zero,
                "inside_fraction_per_example": output_logit.new_zeros((output_logit.shape[0],), dtype=torch.float32),
                "tube_distance_mean_per_example": output_logit.new_zeros((output_logit.shape[0],), dtype=torch.float32),
            }

        rollout_next_position = self._as_tensor(
            data_dict["decoder/rollout_next_position"][:, : output_logit.shape[1], :, :2],
            device=device,
            dtype=dtype,
        )
        decision_agent_mask = semantic_context["decision_agent_mask"]
        sdc_next_pos_world = (
            rollout_next_position * decision_agent_mask[:, None, :, None].to(dtype=dtype)
        ).sum(dim=2)

        raw_path_model = self._as_tensor(
            data_dict.get("cf/sdc_selected_raw_path_model", data_dict["cf/sdc_selected_raw_path_world"]),
            device=device,
            dtype=dtype,
        )
        raw_path_mask = self._as_tensor(
            data_dict["cf/sdc_selected_raw_path_mask"],
            device=device,
            dtype=dtype,
        )
        raw_path_segment_mask = self._as_tensor(
            data_dict["cf/sdc_selected_raw_path_segment_mask"],
            device=device,
            dtype=dtype,
        )
        progress_path_model = self._as_tensor(
            data_dict.get("cf/sdc_selected_progress_centerline_model", data_dict.get("cf/sdc_selected_raw_path_model", data_dict["cf/sdc_selected_raw_path_world"])),
            device=device,
            dtype=dtype,
        )
        progress_path_mask = self._as_tensor(
            data_dict.get("cf/sdc_selected_progress_centerline_mask", data_dict["cf/sdc_selected_raw_path_mask"]),
            device=device,
            dtype=dtype,
        )
        progress_path_segment_mask = self._as_tensor(
            data_dict.get("cf/sdc_selected_progress_centerline_segment_mask", data_dict["cf/sdc_selected_raw_path_segment_mask"]),
            device=device,
            dtype=dtype,
        )
        tube_projection = project_points_to_segment_tube_torch(
            sdc_next_pos_world,
            path_points_world=raw_path_model,
            path_point_mask=raw_path_mask,
            path_segment_mask=raw_path_segment_mask,
        )
        tube_distance = torch.nan_to_num(
            tube_projection["nearest_distance"],
            nan=1e6,
            posinf=1e6,
            neginf=0.0,
        ).to(dtype=torch.float32)
        nearest_arc = torch.nan_to_num(
            tube_projection["nearest_arc"],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).to(dtype=torch.float32)
        path_total_arc = torch.nan_to_num(
            tube_projection["path_total_arc"],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).to(dtype=torch.float32)
        progress_projection = project_points_to_segment_tube_torch(
            sdc_next_pos_world,
            path_points_world=progress_path_model,
            path_point_mask=progress_path_mask,
            path_segment_mask=progress_path_segment_mask,
        )
        progress_nearest_arc = torch.nan_to_num(
            progress_projection["nearest_arc"],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).to(dtype=torch.float32)
        progress_path_total_arc = torch.nan_to_num(
            progress_projection["path_total_arc"],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).to(dtype=torch.float32)

        tube_radius_m = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_TUBE_RADIUS_M", 3.0))
        inside_reward = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_TUBE_INSIDE_REWARD", 1.0))
        outside_scale = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_TUBE_OUTSIDE_SCALE", 1.0))
        reward_t = torch.where(
            tube_distance <= tube_radius_m,
            torch.full_like(tube_distance, float(inside_reward)),
            -(tube_distance - float(tube_radius_m)) * float(outside_scale),
        )

        semantic_target = self._as_tensor(
            data_dict["cf/sdc_semantic_label_id"],
            device=device,
            dtype=torch.long,
        ).reshape(output_logit.shape[0], -1)[:, 0]
        stop_label_id = int(SDC_PATH_SEMANTIC_LABEL_ORDER.index("stop"))
        route_available = raw_path_mask.sum(dim=-1) > 0
        valid_mask = (
            semantic_context["sdc_valid_by_t"]
            & semantic_context["control_available"][:, None]
            & semantic_context["alternative_batch_mask"][:, None]
            & (semantic_target != stop_label_id)[:, None]
            & route_available[:, None]
        )
        progress_reward_scale_cfg = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_TUBE_PROGRESS_REWARD_SCALE", 0.0)
        )
        semantic_ext_allow_progress_reward = bool(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_ALLOW_PROGRESS_REWARD", False)
        )
        progress_reward_scale = (
            progress_reward_scale_cfg
            if ((not semantic_ext_enabled) or semantic_ext_allow_progress_reward)
            else 0.0
        )
        progress_exponent = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_TUBE_PROGRESS_EXPONENT", 2.0)
        )
        progress_unit_m = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_TUBE_PROGRESS_UNIT_M", 10.0)
        )
        progress_gate_mult = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_TUBE_PROGRESS_GATE_MULT", 1.0)
        )
        progress_radius_from_divergence_m = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_TUBE_PROGRESS_RADIUS_FROM_DIVERGENCE_M", 80.0)
        )
        family_divergence_onsets = self._as_tensor(
            data_dict["cf/sdc_family_divergence_onsets"],
            device=device,
            dtype=dtype,
        ).to(dtype=torch.float32)
        progress_reward_t = torch.zeros_like(reward_t)
        frontier_arc_t = torch.zeros_like(reward_t)
        progress_cap_arc_t = torch.zeros_like(reward_t)
        stall_penalty_t = torch.zeros_like(reward_t)
        center_penalty_t = torch.zeros_like(reward_t)
        traffic_speed_penalty_t = torch.zeros_like(reward_t)
        traffic_speed_reference_t = torch.zeros_like(reward_t)
        traffic_speed_neighbor_mean_t = torch.zeros_like(reward_t)
        traffic_speed_neighbor_median_t = torch.zeros_like(reward_t)
        traffic_speed_floor_threshold_t = torch.zeros_like(reward_t)
        traffic_speed_cap_threshold_t = torch.zeros_like(reward_t)
        traffic_speed_cap_penalty_t = torch.zeros_like(reward_t)
        traffic_speed_sdc_t = torch.zeros_like(reward_t)
        traffic_speed_neighbor_count_t = torch.zeros_like(reward_t)
        divergence_onset_m = torch.zeros((reward_t.shape[0],), device=reward_t.device, dtype=torch.float32)
        if progress_reward_scale_cfg > 0.0 or semantic_ext_enabled:
            progress_gate_radius = max(float(tube_radius_m) * float(progress_gate_mult), 0.0)
            onset_valid = torch.isfinite(family_divergence_onsets) & (family_divergence_onsets >= 0.0)
            onset_large = torch.full_like(family_divergence_onsets, 1e6)
            divergence_onset_m = torch.where(
                onset_valid,
                family_divergence_onsets,
                onset_large,
            )
            divergence_onset_m = divergence_onset_m.min(dim=-1).values
            divergence_onset_m = torch.where(
                divergence_onset_m < 1e5,
                divergence_onset_m,
                torch.zeros_like(divergence_onset_m),
            )
            divergence_point_model = torch.zeros(
                (raw_path_model.shape[0], 2),
                device=device,
                dtype=torch.float32,
            )
            for batch_idx in range(int(raw_path_model.shape[0])):
                point_mask = raw_path_mask[batch_idx] > 0.5
                if not bool(point_mask.any()):
                    continue
                raw_points = raw_path_model[batch_idx][point_mask].to(dtype=torch.float32)
                if raw_points.shape[0] == 0:
                    continue
                onset = torch.clamp(
                    divergence_onset_m[batch_idx].to(dtype=torch.float32),
                    min=0.0,
                    max=path_total_arc[batch_idx].to(dtype=torch.float32),
                )
                divergence_point_model[batch_idx] = self._sample_point_along_polyline_torch(raw_points, onset)
            progress_cap_arc_m = self._compute_sdc_progress_radius_cap_arc(
                path_xy=progress_path_model.to(dtype=torch.float32),
                path_mask=progress_path_mask > 0.5,
                circle_center_xy=divergence_point_model,
                radius_from_divergence_m=float(progress_radius_from_divergence_m),
                path_total_arc=progress_path_total_arc,
            )
            progress_cap_arc_m = progress_cap_arc_m.clamp_min(0.0)
            prefix_point_mask = torch.zeros_like(progress_path_mask, dtype=torch.bool)
            prefix_segment_mask = torch.zeros_like(progress_path_segment_mask, dtype=torch.bool)
            for batch_idx in range(int(progress_path_model.shape[0])):
                point_mask = progress_path_mask[batch_idx] > 0.5
                valid_idx = torch.nonzero(point_mask, as_tuple=False).reshape(-1)
                if int(valid_idx.numel()) < 2:
                    prefix_point_mask[batch_idx] = point_mask
                    prefix_segment_mask[batch_idx] = progress_path_segment_mask[batch_idx] > 0.5
                    continue
                valid_points = progress_path_model[batch_idx][valid_idx].to(dtype=torch.float32)
                seg_len = torch.linalg.norm(valid_points[1:] - valid_points[:-1], dim=-1)
                cumulative = torch.cat(
                    [
                        torch.zeros((1,), device=device, dtype=torch.float32),
                        torch.cumsum(seg_len, dim=0),
                    ],
                    dim=0,
                )
                cap = torch.clamp(
                    progress_cap_arc_m[batch_idx].to(dtype=torch.float32),
                    min=0.0,
                    max=cumulative[-1],
                )
                cutoff_idx = int(torch.searchsorted(cumulative, cap, right=True).item())
                keep_n = min(max(cutoff_idx + 1, 2), int(valid_idx.numel()))
                kept_idx = valid_idx[:keep_n]
                prefix_point_mask[batch_idx, kept_idx] = True
                seg_keep_n = max(keep_n - 1, 0)
                if prefix_segment_mask.shape[1] == progress_path_model.shape[1]:
                    prefix_segment_mask[batch_idx, kept_idx[:seg_keep_n]] = True
                else:
                    prefix_segment_mask[batch_idx, :seg_keep_n] = True
            progress_projection = project_points_to_segment_tube_torch(
                sdc_next_pos_world,
                path_points_world=progress_path_model,
                path_point_mask=prefix_point_mask.to(dtype=progress_path_mask.dtype),
                path_segment_mask=prefix_segment_mask.to(dtype=progress_path_segment_mask.dtype),
            )
            progress_nearest_arc = torch.nan_to_num(
                progress_projection["nearest_arc"],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).to(dtype=torch.float32)
            progress_active = (
                valid_mask
                & (progress_cap_arc_m[:, None] > 1e-3)
                & (tube_distance <= progress_gate_radius)
            )
            cap_safe = progress_cap_arc_m.clamp_min(1e-3)
            capped_arc = torch.minimum(progress_nearest_arc, cap_safe[:, None])
            scaled_arc = torch.clamp(capped_arc / max(progress_unit_m, 1e-3), min=0.0)
            progress_potential = torch.where(
                progress_active,
                scaled_arc.pow(max(progress_exponent, 1.0)),
                torch.zeros_like(scaled_arc),
            )
            frontier_potential = torch.cummax(progress_potential, dim=-1).values
            frontier_arc_t = torch.cummax(
                torch.where(progress_active, capped_arc, torch.zeros_like(capped_arc)),
                dim=-1,
            ).values
            progress_cap_arc_t = cap_safe[:, None].expand_as(reward_t)
            if progress_reward_scale > 0.0:
                prev_frontier = torch.cat(
                    [torch.zeros_like(frontier_potential[:, :1]), frontier_potential[:, :-1]],
                    dim=-1,
                )
                progress_reward_t = torch.relu(frontier_potential - prev_frontier) * float(progress_reward_scale)
                reward_t = reward_t + progress_reward_t
            if semantic_ext_enabled:
                grace_steps = max(0, int(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_STALL_GRACE_STEPS", 8)))
                window_steps = max(1, int(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_STALL_WINDOW_STEPS", 8)))
                stall_weight = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_STALL_WEIGHT", 0.10))
                stall_min_advance_m = float(
                    self.config.MODEL.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_STALL_MIN_ADVANCE_M", 0.5)
                )
                if stall_weight > 0.0:
                    shifted_frontier = torch.zeros_like(frontier_arc_t)
                    if window_steps < int(frontier_arc_t.shape[1]):
                        shifted_frontier[:, window_steps:] = frontier_arc_t[:, :-window_steps]
                    advance_over_window = frontier_arc_t - shifted_frontier
                    step_idx = torch.arange(frontier_arc_t.shape[1], device=device).view(1, -1)
                    stall_active = valid_mask & (step_idx >= int(grace_steps))
                    stall_penalty_t = torch.where(
                        stall_active,
                        torch.relu(float(stall_min_advance_m) - advance_over_window) * float(stall_weight),
                        torch.zeros_like(frontier_arc_t),
                    )
                    reward_t = reward_t - stall_penalty_t

                traffic_speed_bundle = self._compute_semantic_ext_traffic_speed_floor_bundle(
                    data_dict=data_dict,
                    semantic_context=semantic_context,
                    sdc_next_pos_world=sdc_next_pos_world.to(dtype=torch.float32),
                    valid_mask=valid_mask,
                    reference_batch=reference_batch,
                    frontier_arc_t=frontier_arc_t,
                )
                traffic_speed_penalty_t = traffic_speed_bundle["penalty_t"]
                traffic_speed_reference_t = traffic_speed_bundle["reference_speed_t"]
                traffic_speed_neighbor_mean_t = traffic_speed_bundle["neighbor_mean_speed_t"]
                traffic_speed_neighbor_median_t = traffic_speed_bundle["neighbor_median_speed_t"]
                traffic_speed_floor_threshold_t = traffic_speed_bundle["floor_threshold_t"]
                traffic_speed_cap_threshold_t = traffic_speed_bundle["cap_threshold_t"]
                traffic_speed_cap_penalty_t = traffic_speed_bundle["cap_penalty_t"]
                traffic_speed_sdc_t = traffic_speed_bundle["sdc_speed_t"]
                traffic_speed_neighbor_count_t = traffic_speed_bundle["neighbor_count_t"]
                reward_t = reward_t - traffic_speed_penalty_t

        center_curriculum_weight = self._semantic_ext_center_curriculum_weight(device=device)
        if semantic_ext_enabled and float(center_curriculum_weight.item()) > 0.0:
            center_penalty_t = F.smooth_l1_loss(
                tube_distance,
                torch.zeros_like(tube_distance),
                beta=1.0,
                reduction="none",
            ) * center_curriculum_weight.to(dtype=torch.float32)
            center_penalty_t = torch.where(valid_mask, center_penalty_t, torch.zeros_like(center_penalty_t))
            reward_t = reward_t - center_penalty_t
        reward_t = torch.where(valid_mask, reward_t, torch.zeros_like(reward_t))
        inside_mask = valid_mask & (tube_distance <= tube_radius_m)
        valid_frontier = valid_mask.any(dim=-1)
        inside_fraction_per_example = torch.where(
            valid_frontier,
            inside_mask.float().sum(dim=-1) / valid_mask.float().sum(dim=-1).clamp_min(1.0),
            torch.zeros_like(valid_frontier, dtype=torch.float32),
        )
        tube_distance_mean_per_example = self._masked_mean_per_example(tube_distance, valid_mask)
        return {
            "reward_t": reward_t,
            "valid_mask": valid_mask,
            "tube_distance": tube_distance,
            "progress_reward_t": progress_reward_t,
            "frontier_arc_t": frontier_arc_t,
            "progress_cap_arc_t": progress_cap_arc_t,
            "stall_penalty_t": stall_penalty_t,
            "center_penalty_t": center_penalty_t,
            "traffic_speed_penalty_t": traffic_speed_penalty_t,
            "traffic_speed_reference_t": traffic_speed_reference_t,
            "traffic_speed_neighbor_mean_t": traffic_speed_neighbor_mean_t,
            "traffic_speed_neighbor_median_t": traffic_speed_neighbor_median_t,
            "traffic_speed_floor_threshold_t": traffic_speed_floor_threshold_t,
            "traffic_speed_cap_threshold_t": traffic_speed_cap_threshold_t,
            "traffic_speed_cap_penalty_t": traffic_speed_cap_penalty_t,
            "traffic_speed_sdc_t": traffic_speed_sdc_t,
            "traffic_speed_neighbor_count_t": traffic_speed_neighbor_count_t,
            "divergence_onset_m": divergence_onset_m,
            "inside_fraction": sanitize_scalar_loss(inside_mask.float().mean()) if bool(valid_mask.any()) else zero,
            "first_step_reward_mean": sanitize_scalar_loss(reward_t[:, 0].mean()) if reward_t.shape[1] > 0 else zero,
            "progress_reward_mean": sanitize_scalar_loss(progress_reward_t[valid_mask].mean()) if bool(valid_mask.any()) else zero,
            "stall_penalty_mean": sanitize_scalar_loss(stall_penalty_t[valid_mask].mean()) if bool(valid_mask.any()) else zero,
            "center_penalty_mean": sanitize_scalar_loss(center_penalty_t[valid_mask].mean()) if bool(valid_mask.any()) else zero,
            "traffic_speed_penalty_mean": traffic_speed_bundle["penalty_mean"] if semantic_ext_enabled else zero,
            "traffic_speed_reference_mean": traffic_speed_bundle["reference_speed_mean"] if semantic_ext_enabled else zero,
            "traffic_speed_neighbor_mean_mean": traffic_speed_bundle["neighbor_mean_speed_mean"] if semantic_ext_enabled else zero,
            "traffic_speed_neighbor_median_mean": traffic_speed_bundle["neighbor_median_speed_mean"] if semantic_ext_enabled else zero,
            "traffic_speed_floor_penalty_mean": traffic_speed_bundle["floor_penalty_mean"] if semantic_ext_enabled else zero,
            "traffic_speed_cap_penalty_mean": traffic_speed_bundle["cap_penalty_mean"] if semantic_ext_enabled else zero,
            "traffic_speed_cap_threshold_mean": traffic_speed_bundle["cap_threshold_mean"] if semantic_ext_enabled else zero,
            "traffic_speed_sdc_mean": traffic_speed_bundle["sdc_speed_mean"] if semantic_ext_enabled else zero,
            "traffic_speed_neighbor_count_mean": traffic_speed_bundle["neighbor_count_mean"] if semantic_ext_enabled else zero,
            "center_curriculum_weight": center_curriculum_weight,
            "frontier_arc_final_mean": sanitize_scalar_loss(frontier_arc_t[:, -1][valid_frontier].mean()) if bool(valid_frontier.any()) else zero,
            "progress_cap_arc_mean": sanitize_scalar_loss(progress_cap_arc_t[valid_mask].mean()) if bool(valid_mask.any()) else zero,
            "divergence_onset_mean": sanitize_scalar_loss(divergence_onset_m[valid_frontier].mean()) if bool(valid_frontier.any()) else zero,
            "inside_fraction_per_example": inside_fraction_per_example,
            "tube_distance_mean_per_example": tube_distance_mean_per_example,
        }

    def _build_sdc_semantic_rollout_tube_policy_objective(self, *, data_dict):
        self._trace_first_step("tube_policy:start")
        output_logit = data_dict["decoder/output_logit"]
        zero = output_logit.new_tensor(0.0)
        group_size = int(self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_TUBE_GROUP_SIZE", 0))
        debug_required = self._should_dump_rollout_tube_training_debug()
        if group_size <= 0:
            self._trace_first_step("tube_policy:group_size_zero")
            return {
                "policy_loss": zero,
                "valid_fraction": zero,
                "inside_fraction": zero,
                "return_mean": zero,
                "return_std": zero,
                "advantage_abs_mean": zero,
                "tube_distance_mean": zero,
                "progress_reward_mean": zero,
                "stall_penalty_mean": zero,
                "center_penalty_mean": zero,
                "traffic_speed_penalty_mean": zero,
                "traffic_speed_reference_mean": zero,
                "traffic_speed_neighbor_mean_mean": zero,
                "traffic_speed_neighbor_median_mean": zero,
                "traffic_speed_floor_penalty_mean": zero,
                "traffic_speed_cap_penalty_mean": zero,
                "traffic_speed_cap_threshold_mean": zero,
                "traffic_speed_sdc_mean": zero,
                "traffic_speed_neighbor_count_mean": zero,
                "center_curriculum_weight": zero,
                "consensus_bonus_mean": zero,
                "novelty_bonus_mean": zero,
                "novelty_score_mean": zero,
                "consensus_cluster_count_mean": zero,
                "consensus_cluster_entropy_mean": zero,
                "thermostat_target_entropy_mean": zero,
                "thermostat_alpha_mean": zero,
                "thermostat_eta_mean": zero,
                "behavior_aux_loss": zero,
                "behavior_semantic_ce": zero,
                "behavior_branch_ce": zero,
                "behavior_compliance_ce": zero,
                "behavior_timing_ce": zero,
                "behavior_dag_align": zero,
                "frontier_arc_final_mean": zero,
                "progress_cap_arc_mean": zero,
                "divergence_onset_mean": zero,
                "group_size": 0,
            }

        sampling_method = str(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_TUBE_SAMPLING_METHOD", "softmax")
        ).strip() or "softmax"
        temperature = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_TUBE_TEMPERATURE", 1.0))
        topp = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_TUBE_TOPP", 0.9))
        discount = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_TUBE_DISCOUNT", 1.0))

        reward_list = []
        valid_list = []
        log_prob_list = []
        distance_list = []
        progress_reward_list = []
        stall_penalty_list = []
        center_penalty_list = []
        traffic_speed_penalty_list = []
        traffic_speed_reference_list = []
        traffic_speed_neighbor_mean_list = []
        traffic_speed_neighbor_median_list = []
        traffic_speed_floor_threshold_list = []
        traffic_speed_cap_threshold_list = []
        traffic_speed_cap_penalty_list = []
        traffic_speed_sdc_list = []
        traffic_speed_neighbor_count_list = []
        frontier_arc_list = []
        progress_cap_arc_list = []
        divergence_onset_list = []
        inside_fraction_list = []
        traffic_speed_penalty_mean_list = []
        traffic_speed_reference_mean_list = []
        traffic_speed_neighbor_mean_mean_list = []
        traffic_speed_neighbor_median_mean_list = []
        traffic_speed_floor_penalty_mean_list = []
        traffic_speed_cap_penalty_mean_list = []
        traffic_speed_cap_threshold_mean_list = []
        traffic_speed_sdc_mean_list = []
        traffic_speed_neighbor_count_mean_list = []
        trajectory_list = []
        action_token_list = [] if debug_required else None
        current_heading_list = []

        motion_decoder = getattr(self.model, "motion_decoder", None)
        previous_null_dropout_prob = getattr(motion_decoder, "cf_null_dropout_prob", None)
        if motion_decoder is not None and previous_null_dropout_prob is not None:
            motion_decoder.cf_null_dropout_prob = 0.0
        try:
            for rollout_idx in range(group_size):
                if rollout_idx == 0:
                    self._trace_first_step("tube_policy:first_rollout:enter", group_size=int(group_size))
                rollout_input = self._prepare_batch_for_training_autoregressive_rollout(data_dict)
                rollout_output = self.model.autoregressive_rollout(
                    rollout_input,
                    num_decode_steps=None,
                    sampling_method=sampling_method,
                    temperature=(None if temperature <= 0.0 else temperature),
                    topp=(None if topp <= 0.0 else topp),
                    autoregressive_start_step=0,
                    allow_training=True,
                )
                rollout_eval_dict = self._adapt_rollout_output_for_semantic_guidance(
                    base_batch=data_dict,
                    rollout_output=rollout_output,
                )
                rollout_semantic_context = self._extract_sdc_semantic_context(rollout_eval_dict)
                if rollout_semantic_context is None:
                    continue
                # The reward serves only as a detached REINFORCE-style baseline/advantage
                # target, so keeping it out of autograd saves a large amount of memory.
                with torch.no_grad():
                    reward_bundle = self._compute_sdc_rollout_tube_rewards(
                        data_dict=rollout_eval_dict,
                        semantic_context=rollout_semantic_context,
                        reference_batch=data_dict,
                    )
                decision_agent_mask = rollout_semantic_context["decision_agent_mask"]
                rollout_next_position = self._as_tensor(
                    rollout_eval_dict["decoder/rollout_next_position"][:, : output_logit.shape[1], :, :2],
                    device=output_logit.device,
                    dtype=output_logit.dtype,
                )
                sdc_next_pos_world = None
                if debug_required:
                    sdc_next_pos_world = (
                        rollout_next_position * decision_agent_mask[:, None, :, None].to(dtype=output_logit.dtype)
                    ).sum(dim=2).detach()
                selected_action = self._as_tensor(
                    rollout_eval_dict["decoder/input_action"][:, : output_logit.shape[1], : output_logit.shape[2]],
                    device=output_logit.device,
                    dtype=torch.long,
                )
                sdc_action_tokens = (
                    selected_action * decision_agent_mask[:, None, :].to(dtype=torch.long)
                ).sum(dim=2)
                log_prob_list.append(
                    self._extract_sdc_selected_rollout_log_probs(
                        rollout_eval_dict=rollout_eval_dict,
                        semantic_context=rollout_semantic_context,
                    )
                )
                reward_list.append(reward_bundle["reward_t"])
                valid_list.append(reward_bundle["valid_mask"])
                distance_list.append(reward_bundle["tube_distance"])
                progress_reward_list.append(reward_bundle["progress_reward_t"])
                stall_penalty_list.append(reward_bundle["stall_penalty_t"])
                center_penalty_list.append(reward_bundle["center_penalty_t"])
                traffic_speed_penalty_list.append(reward_bundle["traffic_speed_penalty_t"])
                traffic_speed_reference_list.append(reward_bundle["traffic_speed_reference_t"])
                traffic_speed_neighbor_mean_list.append(reward_bundle["traffic_speed_neighbor_mean_t"])
                traffic_speed_neighbor_median_list.append(reward_bundle["traffic_speed_neighbor_median_t"])
                traffic_speed_floor_threshold_list.append(reward_bundle["traffic_speed_floor_threshold_t"])
                traffic_speed_cap_threshold_list.append(reward_bundle["traffic_speed_cap_threshold_t"])
                traffic_speed_cap_penalty_list.append(reward_bundle["traffic_speed_cap_penalty_t"])
                traffic_speed_sdc_list.append(reward_bundle["traffic_speed_sdc_t"])
                traffic_speed_neighbor_count_list.append(reward_bundle["traffic_speed_neighbor_count_t"])
                frontier_arc_list.append(reward_bundle["frontier_arc_t"])
                progress_cap_arc_list.append(reward_bundle["progress_cap_arc_t"])
                divergence_onset_list.append(reward_bundle["divergence_onset_m"])
                inside_fraction_list.append(reward_bundle["inside_fraction"])
                traffic_speed_penalty_mean_list.append(reward_bundle["traffic_speed_penalty_mean"])
                traffic_speed_reference_mean_list.append(reward_bundle["traffic_speed_reference_mean"])
                traffic_speed_neighbor_mean_mean_list.append(reward_bundle["traffic_speed_neighbor_mean_mean"])
                traffic_speed_neighbor_median_mean_list.append(reward_bundle["traffic_speed_neighbor_median_mean"])
                traffic_speed_floor_penalty_mean_list.append(reward_bundle["traffic_speed_floor_penalty_mean"])
                traffic_speed_cap_penalty_mean_list.append(reward_bundle["traffic_speed_cap_penalty_mean"])
                traffic_speed_cap_threshold_mean_list.append(reward_bundle["traffic_speed_cap_threshold_mean"])
                traffic_speed_sdc_mean_list.append(reward_bundle["traffic_speed_sdc_mean"])
                traffic_speed_neighbor_count_mean_list.append(reward_bundle["traffic_speed_neighbor_count_mean"])
                trajectory_list.append(
                    torch.cat(
                        [
                            rollout_semantic_context["sdc_current_pos_world"][:, :1].detach(),
                            (
                                sdc_next_pos_world
                                if sdc_next_pos_world is not None
                                else (
                                    rollout_next_position
                                    * decision_agent_mask[:, None, :, None].to(dtype=output_logit.dtype)
                                ).sum(dim=2).detach()
                            ),
                        ],
                        dim=1,
                    )
                )
                current_heading_list.append(rollout_semantic_context["sdc_current_heading_world"][:, 0].detach())
                if debug_required:
                    action_token_list.append(sdc_action_tokens.detach())
                if rollout_idx == 0:
                    self._trace_first_step(
                        "tube_policy:first_rollout:exit",
                        reward_shape=list(reward_bundle["reward_t"].shape),
                        valid_fraction=reward_bundle["valid_mask"].float().mean(),
                    )
        finally:
            if motion_decoder is not None and previous_null_dropout_prob is not None:
                motion_decoder.cf_null_dropout_prob = previous_null_dropout_prob

        if not reward_list:
            self._trace_first_step("tube_policy:no_reward_list")
            return {
                "policy_loss": zero,
                "valid_fraction": zero,
                "inside_fraction": zero,
                "return_mean": zero,
                "return_std": zero,
                "advantage_abs_mean": zero,
                "tube_distance_mean": zero,
                "progress_reward_mean": zero,
                "stall_penalty_mean": zero,
                "center_penalty_mean": zero,
                "traffic_speed_penalty_mean": zero,
                "traffic_speed_reference_mean": zero,
                "traffic_speed_neighbor_mean_mean": zero,
                "traffic_speed_neighbor_median_mean": zero,
                "traffic_speed_floor_penalty_mean": zero,
                "traffic_speed_cap_penalty_mean": zero,
                "traffic_speed_cap_threshold_mean": zero,
                "traffic_speed_sdc_mean": zero,
                "traffic_speed_neighbor_count_mean": zero,
                "center_curriculum_weight": zero,
                "consensus_bonus_mean": zero,
                "novelty_bonus_mean": zero,
                "novelty_score_mean": zero,
                "consensus_cluster_count_mean": zero,
                "consensus_cluster_entropy_mean": zero,
                "thermostat_target_entropy_mean": zero,
                "thermostat_alpha_mean": zero,
                "thermostat_eta_mean": zero,
                "behavior_aux_loss": zero,
                "behavior_semantic_ce": zero,
                "behavior_branch_ce": zero,
                "behavior_compliance_ce": zero,
                "behavior_timing_ce": zero,
                "behavior_dag_align": zero,
                "frontier_arc_final_mean": zero,
                "progress_cap_arc_mean": zero,
                "divergence_onset_mean": zero,
                "group_size": 0,
            }

        reward_t = torch.stack(reward_list, dim=0)
        valid_mask = torch.stack(valid_list, dim=0)
        selected_log_probs = torch.stack(log_prob_list, dim=0)
        tube_distance = torch.stack(distance_list, dim=0)
        progress_reward_t = torch.stack(progress_reward_list, dim=0)
        stall_penalty_t = torch.stack(stall_penalty_list, dim=0)
        center_penalty_t = torch.stack(center_penalty_list, dim=0)
        traffic_speed_penalty_t = torch.stack(traffic_speed_penalty_list, dim=0)
        traffic_speed_reference_t = torch.stack(traffic_speed_reference_list, dim=0)
        traffic_speed_neighbor_mean_t = torch.stack(traffic_speed_neighbor_mean_list, dim=0)
        traffic_speed_neighbor_median_t = torch.stack(traffic_speed_neighbor_median_list, dim=0)
        traffic_speed_floor_threshold_t = torch.stack(traffic_speed_floor_threshold_list, dim=0)
        traffic_speed_cap_threshold_t = torch.stack(traffic_speed_cap_threshold_list, dim=0)
        traffic_speed_cap_penalty_t = torch.stack(traffic_speed_cap_penalty_list, dim=0)
        traffic_speed_sdc_t = torch.stack(traffic_speed_sdc_list, dim=0)
        traffic_speed_neighbor_count_t = torch.stack(traffic_speed_neighbor_count_list, dim=0)
        frontier_arc_t = torch.stack(frontier_arc_list, dim=0)
        progress_cap_arc_t = torch.stack(progress_cap_arc_list, dim=0)
        divergence_onset_m = torch.stack(divergence_onset_list, dim=0)
        trajectories_world = torch.stack(trajectory_list, dim=0)
        current_heading_world = torch.stack(current_heading_list, dim=0).mean(dim=0)
        consensus_bundle = self._compute_rollout_topomcpo_regularization_bundle(
            data_dict=data_dict,
            reward_t=reward_t,
            valid_mask=valid_mask,
            tube_distance=tube_distance,
            frontier_arc_t=frontier_arc_t,
            trajectories_world=trajectories_world,
            current_heading_world=current_heading_world,
        )
        reward_t = reward_t + consensus_bundle["bonus_t"]
        rtg = self._compute_discounted_return_to_go(reward_t, valid_mask, discount=discount)
        advantage_t = self._group_normalize_advantages(rtg, valid_mask)
        policy_loss = sanitize_scalar_loss(
            -(
                selected_log_probs[valid_mask].to(dtype=torch.float32)
                * advantage_t[valid_mask].detach()
            ).mean()
        ) if bool(valid_mask.any()) else zero
        total_return = reward_t.sum(dim=-1)
        valid_return = valid_mask.any(dim=-1)
        return_mean = sanitize_scalar_loss(total_return[valid_return].mean()) if bool(valid_return.any()) else zero
        return_std = sanitize_scalar_loss(total_return[valid_return].std(unbiased=False)) if bool(valid_return.any()) else zero
        distance_mean = sanitize_scalar_loss(tube_distance[valid_mask].mean()) if bool(valid_mask.any()) else zero
        inside_fraction = sanitize_scalar_loss(
            torch.stack(inside_fraction_list).mean()
        ) if inside_fraction_list else zero
        if debug_required:
            action_token_t = torch.stack(action_token_list, dim=0)
            self._maybe_dump_rollout_tube_training_debug(
                data_dict=data_dict,
                reward_t=reward_t,
                valid_mask=valid_mask,
                selected_log_probs=selected_log_probs,
                tube_distance=tube_distance,
                rtg=rtg,
                advantage_t=advantage_t,
                trajectories_world=trajectories_world,
                action_token_t=action_token_t,
                traffic_speed_penalty_t=traffic_speed_penalty_t,
                traffic_speed_reference_t=traffic_speed_reference_t,
                traffic_speed_neighbor_mean_t=traffic_speed_neighbor_mean_t,
                traffic_speed_neighbor_median_t=traffic_speed_neighbor_median_t,
                traffic_speed_floor_threshold_t=traffic_speed_floor_threshold_t,
                traffic_speed_cap_threshold_t=traffic_speed_cap_threshold_t,
                traffic_speed_cap_penalty_t=traffic_speed_cap_penalty_t,
                traffic_speed_sdc_t=traffic_speed_sdc_t,
                traffic_speed_neighbor_count_t=traffic_speed_neighbor_count_t,
            )
        self._trace_first_step(
            "tube_policy:end",
            policy_loss=policy_loss,
            valid_fraction=valid_mask.float().mean(),
            return_mean=return_mean,
        )
        return {
            "policy_loss": policy_loss,
            "valid_fraction": valid_mask.float().mean(),
            "inside_fraction": inside_fraction,
            "return_mean": return_mean,
            "return_std": return_std,
            "advantage_abs_mean": sanitize_scalar_loss(advantage_t[valid_mask].abs().mean()) if bool(valid_mask.any()) else zero,
            "tube_distance_mean": distance_mean,
            "progress_reward_mean": sanitize_scalar_loss(progress_reward_t[valid_mask].mean()) if bool(valid_mask.any()) else zero,
            "stall_penalty_mean": sanitize_scalar_loss(stall_penalty_t[valid_mask].mean()) if bool(valid_mask.any()) else zero,
            "center_penalty_mean": sanitize_scalar_loss(center_penalty_t[valid_mask].mean()) if bool(valid_mask.any()) else zero,
            "traffic_speed_penalty_mean": sanitize_scalar_loss(torch.stack(traffic_speed_penalty_mean_list).mean()) if traffic_speed_penalty_mean_list else zero,
            "traffic_speed_reference_mean": sanitize_scalar_loss(torch.stack(traffic_speed_reference_mean_list).mean()) if traffic_speed_reference_mean_list else zero,
            "traffic_speed_neighbor_mean_mean": sanitize_scalar_loss(torch.stack(traffic_speed_neighbor_mean_mean_list).mean()) if traffic_speed_neighbor_mean_mean_list else zero,
            "traffic_speed_neighbor_median_mean": sanitize_scalar_loss(torch.stack(traffic_speed_neighbor_median_mean_list).mean()) if traffic_speed_neighbor_median_mean_list else zero,
            "traffic_speed_floor_penalty_mean": sanitize_scalar_loss(torch.stack(traffic_speed_floor_penalty_mean_list).mean()) if traffic_speed_floor_penalty_mean_list else zero,
            "traffic_speed_cap_penalty_mean": sanitize_scalar_loss(torch.stack(traffic_speed_cap_penalty_mean_list).mean()) if traffic_speed_cap_penalty_mean_list else zero,
            "traffic_speed_cap_threshold_mean": sanitize_scalar_loss(torch.stack(traffic_speed_cap_threshold_mean_list).mean()) if traffic_speed_cap_threshold_mean_list else zero,
            "traffic_speed_sdc_mean": sanitize_scalar_loss(torch.stack(traffic_speed_sdc_mean_list).mean()) if traffic_speed_sdc_mean_list else zero,
            "traffic_speed_neighbor_count_mean": sanitize_scalar_loss(torch.stack(traffic_speed_neighbor_count_mean_list).mean()) if traffic_speed_neighbor_count_mean_list else zero,
            "center_curriculum_weight": reward_t.new_tensor(
                float(self._semantic_ext_center_curriculum_weight(device=reward_t.device).item())
            ),
            "consensus_bonus_mean": consensus_bundle["consensus_bonus_mean"],
            "novelty_bonus_mean": consensus_bundle["novelty_bonus_mean"],
            "novelty_score_mean": consensus_bundle["novelty_score_mean"],
            "consensus_cluster_count_mean": consensus_bundle["cluster_count_mean"],
            "consensus_cluster_entropy_mean": consensus_bundle["cluster_entropy_mean"],
            "thermostat_target_entropy_mean": consensus_bundle["target_entropy_mean"],
            "thermostat_alpha_mean": consensus_bundle["thermostat_alpha_mean"],
            "thermostat_eta_mean": consensus_bundle["thermostat_eta_mean"],
            "behavior_aux_loss": consensus_bundle["behavior_aux_loss"],
            "behavior_semantic_ce": consensus_bundle["behavior_semantic_ce"],
            "behavior_branch_ce": consensus_bundle["behavior_branch_ce"],
            "behavior_compliance_ce": consensus_bundle["behavior_compliance_ce"],
            "behavior_timing_ce": consensus_bundle["behavior_timing_ce"],
            "behavior_dag_align": consensus_bundle["behavior_dag_align"],
            "frontier_arc_final_mean": sanitize_scalar_loss(frontier_arc_t[:, :, -1][valid_return].mean()) if bool(valid_return.any()) else zero,
            "progress_cap_arc_mean": sanitize_scalar_loss(progress_cap_arc_t[valid_mask].mean()) if bool(valid_mask.any()) else zero,
            "divergence_onset_mean": sanitize_scalar_loss(divergence_onset_m[valid_return].mean()) if bool(valid_return.any()) else zero,
            "group_size": int(len(reward_list)),
        }

    def _compute_sdc_rollout_progress_loss(self, *, data_dict, semantic_context):
        output_logit = data_dict["decoder/output_logit"]
        device = output_logit.device
        dtype = output_logit.dtype
        zero = output_logit.new_tensor(0.0)
        required = ("decoder/rollout_next_position", "cf/sdc_semantic_label_id")
        if semantic_context is None or any(key not in data_dict for key in required):
            return {
                "progress_loss": zero,
                "progress_valid_fraction": zero,
                "realized_progress_mean": zero,
                "stall_fraction": zero,
                "rollout_family_distance": output_logit.new_zeros((output_logit.shape[0], output_logit.shape[1])),
            }

        rollout_next_position = self._as_tensor(
            data_dict["decoder/rollout_next_position"][:, : output_logit.shape[1], :, :2],
            device=device,
            dtype=dtype,
        )
        decision_agent_mask = semantic_context["decision_agent_mask"]
        sdc_next_pos_world = (
            rollout_next_position * decision_agent_mask[:, None, :, None].to(dtype=dtype)
        ).sum(dim=2)
        realized_projection = project_points_to_family_paths_torch(
            sdc_next_pos_world,
            family_path_polylines_world=semantic_context["family_paths_world"],
            family_path_mask=semantic_context["family_path_mask"],
            family_path_tangents_world=semantic_context["family_tangents_world"],
            family_path_arc_lengths=semantic_context["family_arc_lengths"],
        )

        family_weights = semantic_context["family_weights"]
        current_arc = semantic_context["current_projection"]["nearest_arc"]
        realized_arc = realized_projection["nearest_arc"]
        realized_delta_arc = realized_arc - current_arc
        weighted_progress = (realized_delta_arc * family_weights[:, None, :]).sum(dim=-1)
        weighted_distance = (realized_projection["nearest_distance"] * family_weights[:, None, :]).sum(dim=-1)
        guide_weight = (semantic_context["current_gate"] * family_weights[:, None, :]).sum(dim=-1)
        valid_mask = semantic_context["sdc_valid_by_t"] & semantic_context["control_available"][:, None] & (guide_weight > 1e-5)

        semantic_target = self._as_tensor(
            data_dict["cf/sdc_semantic_label_id"],
            device=device,
            dtype=torch.long,
        ).reshape(output_logit.shape[0], -1)[:, 0]
        stop_label_id = int(SDC_PATH_SEMANTIC_LABEL_ORDER.index("stop"))
        valid_mask = valid_mask & (semantic_target != stop_label_id)[:, None]

        progress_margin_m = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_PROGRESS_MARGIN_M", 0.25))
        stall_threshold_m = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_STALL_THRESHOLD_M", 0.05))
        progress_penalty = torch.relu(progress_margin_m - weighted_progress)

        if not bool(valid_mask.any()):
            return {
                "progress_loss": zero,
                "progress_valid_fraction": zero,
                "realized_progress_mean": zero,
                "stall_fraction": zero,
                "rollout_family_distance": torch.nan_to_num(weighted_distance, nan=0.0, posinf=0.0, neginf=0.0),
            }

        progress_loss = sanitize_scalar_loss(
            (progress_penalty[valid_mask] * guide_weight[valid_mask]).sum()
            / guide_weight[valid_mask].sum().clamp_min(1e-4)
        )
        return {
            "progress_loss": progress_loss,
            "progress_valid_fraction": valid_mask.float().mean(),
            "realized_progress_mean": sanitize_scalar_loss(weighted_progress[valid_mask].mean()),
            "stall_fraction": sanitize_scalar_loss((weighted_progress[valid_mask] < stall_threshold_m).float().mean()),
            "rollout_family_distance": torch.nan_to_num(weighted_distance, nan=0.0, posinf=0.0, neginf=0.0),
        }

    def _build_sdc_semantic_rollout_guidance(self, *, data_dict):
        output_logit = data_dict["decoder/output_logit"]
        zero = output_logit.new_tensor(0.0)
        rollout_sampling_method = str(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_SAMPLING_METHOD", "argmax")
        ).strip() or "argmax"
        rollout_temperature = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_TEMPERATURE", 1.0))
        rollout_topp = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_TOPP", 0.9))

        motion_decoder = getattr(self.model, "motion_decoder", None)
        previous_null_dropout_prob = getattr(motion_decoder, "cf_null_dropout_prob", None)
        if motion_decoder is not None and previous_null_dropout_prob is not None:
            motion_decoder.cf_null_dropout_prob = 0.0
        try:
            rollout_input = self._prepare_batch_for_training_autoregressive_rollout(data_dict)
            rollout_output = self.model.autoregressive_rollout(
                rollout_input,
                num_decode_steps=None,
                sampling_method=rollout_sampling_method,
                temperature=(None if rollout_temperature <= 0.0 else rollout_temperature),
                topp=(None if rollout_topp <= 0.0 else rollout_topp),
                autoregressive_start_step=0,
                allow_training=True,
            )
        finally:
            if motion_decoder is not None and previous_null_dropout_prob is not None:
                motion_decoder.cf_null_dropout_prob = previous_null_dropout_prob

        rollout_eval_dict = self._adapt_rollout_output_for_semantic_guidance(
            base_batch=data_dict,
            rollout_output=rollout_output,
        )
        rollout_semantic_context = self._extract_sdc_semantic_context(rollout_eval_dict)
        if rollout_semantic_context is None:
            return {
                "guide_bundle": {
                    "guide_loss": zero,
                    "guide_weight": output_logit.new_zeros((output_logit.shape[0], output_logit.shape[1])),
                    "guide_valid_fraction": zero,
                    "student_entropy": zero,
                    "expected_energy": zero,
                    "expected_position_penalty": zero,
                    "expected_heading_penalty": zero,
                    "expected_backward_penalty": zero,
                    "family_teacher_entropy": zero,
                    "projected_family_distance": output_logit.new_zeros((output_logit.shape[0], output_logit.shape[1])),
                },
                "progress_bundle": {
                    "progress_loss": zero,
                    "progress_valid_fraction": zero,
                    "realized_progress_mean": zero,
                    "stall_fraction": zero,
                    "rollout_family_distance": output_logit.new_zeros((output_logit.shape[0], output_logit.shape[1])),
                },
                "semantic_context": None,
            }

        rollout_teacher_logits = self._run_policy_teacher(rollout_eval_dict)
        guide_bundle = self._compute_sdc_semantic_family_guidance(
            data_dict=rollout_eval_dict,
            semantic_context=rollout_semantic_context,
            teacher_logits=rollout_teacher_logits,
        )
        progress_bundle = self._compute_sdc_rollout_progress_loss(
            data_dict=rollout_eval_dict,
            semantic_context=rollout_semantic_context,
        )
        return {
            "guide_bundle": guide_bundle,
            "progress_bundle": progress_bundle,
            "semantic_context": rollout_semantic_context,
        }

    def _extract_sdc_path_context(self, data_dict):
        required = (
            "cf/sdc_path_waypoints",
            "cf/sdc_path_waypoint_mask",
            "cf/sdc_path_separability",
            "cf/sdc_path_arc_lengths",
            "cf/decision_agent_mask",
            "cf/sdc_is_factual",
            "cf/sdc_control_available",
        )
        if self.counterfactual_mode != "sdc_path" or any(key not in data_dict for key in required):
            return None

        output_logit = data_dict["decoder/output_logit"]
        B, T, N, _ = output_logit.shape
        predicted = self._expected_next_state_from_logits(output_logit, data_dict)
        decision_agent_mask = self._as_tensor(
            data_dict["cf/decision_agent_mask"],
            device=output_logit.device,
            dtype=output_logit.dtype,
        )
        decision_agent_mask = decision_agent_mask[:, :N]
        if not bool((decision_agent_mask.sum(dim=-1) > 0).any()):
            return None
        sdc_valid_mask = decision_agent_mask[:, None, :] > 0
        target_action_valid_mask = self._as_tensor(
            data_dict["decoder/target_action_valid_mask"][:, :T],
            device=output_logit.device,
            dtype=torch.bool,
        )
        sdc_token_mask = sdc_valid_mask & target_action_valid_mask
        sdc_valid_by_t = sdc_token_mask.any(dim=-1)

        current_pos = self._as_tensor(
            data_dict["decoder/modeled_agent_position"][:, :T, :, :2],
            device=output_logit.device,
            dtype=output_logit.dtype,
        )
        current_heading = self._as_tensor(
            data_dict["decoder/modeled_agent_heading"][:, :T],
            device=output_logit.device,
            dtype=output_logit.dtype,
        )
        mask_f = decision_agent_mask[:, None, :, None]
        sdc_expected_pos_world = (predicted["expected_pos_world"] * mask_f).sum(dim=2)
        sdc_expected_heading_world = (predicted["expected_heading_world"] * decision_agent_mask[:, None, :]).sum(dim=2)
        sdc_origin_pos_world = (current_pos[:, 0] * decision_agent_mask[:, :, None]).sum(dim=1)
        sdc_origin_heading_world = (current_heading[:, 0] * decision_agent_mask).sum(dim=1)
        sdc_expected_pos_local = torch_world_to_sdc_up(
            sdc_expected_pos_world,
            origin_xy_world=sdc_origin_pos_world,
            origin_heading_world=sdc_origin_heading_world,
        )
        sdc_expected_heading_local = torch_heading_to_sdc_up(
            sdc_expected_heading_world,
            origin_heading_world=sdc_origin_heading_world,
        )

        path_waypoints = self._as_tensor(data_dict["cf/sdc_path_waypoints"], device=output_logit.device, dtype=output_logit.dtype)
        path_xy = path_waypoints[..., :2]
        path_heading = torch.atan2(path_waypoints[..., 2], path_waypoints[..., 3])
        path_waypoint_mask = self._as_tensor(
            data_dict["cf/sdc_path_waypoint_mask"],
            device=output_logit.device,
            dtype=output_logit.dtype,
        )
        path_arc = self._as_tensor(data_dict["cf/sdc_path_arc_lengths"], device=output_logit.device, dtype=output_logit.dtype)
        path_sep = self._as_tensor(data_dict["cf/sdc_path_separability"], device=output_logit.device, dtype=output_logit.dtype)
        projection = project_points_to_path_torch(
            sdc_expected_pos_local,
            path_waypoints_local_xy=path_xy,
            path_waypoint_mask=path_waypoint_mask,
            path_waypoint_heading=path_heading,
            path_waypoint_arc=path_arc,
            path_waypoint_separability=path_sep,
        )
        is_factual = self._as_tensor(data_dict["cf/sdc_is_factual"], device=output_logit.device, dtype=torch.bool).reshape(B, -1)[:, 0]
        control_available = self._as_tensor(data_dict["cf/sdc_control_available"], device=output_logit.device, dtype=torch.bool).reshape(B, -1)[:, 0]
        alternative_batch_mask = control_available & (~is_factual)
        factual_batch_mask = control_available & is_factual
        return {
            "decision_agent_mask": decision_agent_mask,
            "sdc_token_mask": sdc_token_mask,
            "sdc_valid_by_t": sdc_valid_by_t,
            "expected_pos_world": predicted["expected_pos_world"],
            "expected_heading_world": predicted["expected_heading_world"],
            "sdc_expected_pos_world": sdc_expected_pos_world,
            "sdc_expected_pos_local": sdc_expected_pos_local,
            "sdc_expected_heading_local": sdc_expected_heading_local,
            "path_xy": path_xy,
            "path_heading": path_heading,
            "path_arc": path_arc,
            "path_separability": path_sep,
            "projection": projection,
            "is_factual": is_factual,
            "control_available": control_available,
            "alternative_batch_mask": alternative_batch_mask,
        }

    def _extract_sdc_semantic_context(self, data_dict):
        required = (
            "cf/sdc_family_path_polylines_world",
            "cf/sdc_family_path_tangents_world",
            "cf/sdc_family_arc_lengths",
            "cf/sdc_family_divergence_onsets",
            "cf/sdc_family_path_mask",
            "cf/sdc_family_confidences",
            "cf/decision_agent_mask",
            "cf/sdc_is_factual",
            "cf/sdc_control_available",
        )
        if self.counterfactual_mode != "sdc_semantic_only" or any(key not in data_dict for key in required):
            return None

        output_logit = data_dict["decoder/output_logit"]
        B, T, N, _ = output_logit.shape
        candidate_bundle = self._next_state_candidates_from_action_space(output_logit, data_dict)
        probs = candidate_bundle["probs"]
        decision_agent_mask = self._as_tensor(
            data_dict["cf/decision_agent_mask"],
            device=output_logit.device,
            dtype=output_logit.dtype,
        )
        decision_agent_mask = decision_agent_mask[:, :N]
        if not bool((decision_agent_mask.sum(dim=-1) > 0).any()):
            return None
        sdc_valid_mask = decision_agent_mask[:, None, :] > 0
        target_action_valid_mask = self._as_tensor(
            data_dict["decoder/target_action_valid_mask"][:, :T],
            device=output_logit.device,
            dtype=torch.bool,
        )
        sdc_token_mask = sdc_valid_mask & target_action_valid_mask
        sdc_valid_by_t = sdc_token_mask.any(dim=-1)

        current_pos_world = candidate_bundle["current_pos_world"]
        current_heading_world = candidate_bundle["current_heading_world"]
        mask_f = decision_agent_mask[:, None, :, None]
        sdc_current_pos_world = (current_pos_world * mask_f).sum(dim=2)
        sdc_current_heading_world = (current_heading_world * decision_agent_mask[:, None, :]).sum(dim=2)

        next_pos_candidates_world = candidate_bundle["next_pos_candidates_world"]
        candidate_heading_world = candidate_bundle["candidate_heading_world"]
        sdc_probs = (probs * decision_agent_mask[:, None, :, None]).sum(dim=2)
        sdc_next_pos_candidates_world = (next_pos_candidates_world * mask_f.unsqueeze(-2)).sum(dim=2)
        sdc_candidate_heading_world = (candidate_heading_world * decision_agent_mask[:, None, :, None]).sum(dim=2)
        sdc_expected_pos_world = (sdc_probs.unsqueeze(-1) * sdc_next_pos_candidates_world).sum(dim=-2)
        sdc_expected_heading_world = torch.atan2(
            (sdc_probs * torch.sin(sdc_candidate_heading_world)).sum(dim=-1),
            (sdc_probs * torch.cos(sdc_candidate_heading_world)).sum(dim=-1),
        )

        family_paths_world = self._as_tensor(
            data_dict["cf/sdc_family_path_polylines_world"],
            device=output_logit.device,
            dtype=output_logit.dtype,
        )
        family_tangents_world = self._as_tensor(
            data_dict["cf/sdc_family_path_tangents_world"],
            device=output_logit.device,
            dtype=output_logit.dtype,
        )
        family_arc_lengths = self._as_tensor(
            data_dict["cf/sdc_family_arc_lengths"],
            device=output_logit.device,
            dtype=output_logit.dtype,
        )
        family_divergence_onsets = self._as_tensor(
            data_dict["cf/sdc_family_divergence_onsets"],
            device=output_logit.device,
            dtype=output_logit.dtype,
        )
        family_path_mask = self._as_tensor(
            data_dict["cf/sdc_family_path_mask"],
            device=output_logit.device,
            dtype=output_logit.dtype,
        )
        family_confidences = self._as_tensor(
            data_dict["cf/sdc_family_confidences"],
            device=output_logit.device,
            dtype=output_logit.dtype,
        )
        current_projection = project_points_to_family_paths_torch(
            sdc_current_pos_world,
            family_path_polylines_world=family_paths_world,
            family_path_mask=family_path_mask,
            family_path_tangents_world=family_tangents_world,
            family_path_arc_lengths=family_arc_lengths,
        )
        expected_projection = project_points_to_family_paths_torch(
            sdc_expected_pos_world,
            family_path_polylines_world=family_paths_world,
            family_path_mask=family_path_mask,
            family_path_tangents_world=family_tangents_world,
            family_path_arc_lengths=family_arc_lengths,
        )
        confidence_weighted = bool(self.config.MODEL.get("LOCAL_CONTROL_SDC_FAMILY_CONFIDENCE_WEIGHTED", False))
        family_weights = family_confidence_weights_torch(
            family_confidences,
            family_path_mask=family_path_mask,
            confidence_weighted=confidence_weighted,
        )
        guide_bandwidth_m = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_FAMILY_GUIDE_BANDWIDTH_M", DEFAULT_FAMILY_GUIDE_BANDWIDTH_M)
        )
        current_gate = compute_family_gate_torch(
            current_projection["nearest_arc"],
            family_divergence_onsets,
            bandwidth_m=guide_bandwidth_m,
        )
        expected_gate = compute_family_gate_torch(
            expected_projection["nearest_arc"],
            family_divergence_onsets,
            bandwidth_m=guide_bandwidth_m,
        )
        family_gate_mean = (expected_gate * family_weights[:, None, :]).sum(dim=-1)

        is_factual = self._as_tensor(data_dict["cf/sdc_is_factual"], device=output_logit.device, dtype=torch.bool).reshape(B, -1)[:, 0]
        control_available = self._as_tensor(
            data_dict["cf/sdc_control_available"],
            device=output_logit.device,
            dtype=torch.bool,
        ).reshape(B, -1)[:, 0]
        factual_batch_mask = control_available & is_factual
        alternative_batch_mask = control_available & (~is_factual)
        return {
            "decision_agent_mask": decision_agent_mask,
            "sdc_token_mask": sdc_token_mask,
            "sdc_valid_by_t": sdc_valid_by_t,
            "sdc_current_pos_world": sdc_current_pos_world,
            "sdc_current_heading_world": sdc_current_heading_world,
            "sdc_expected_pos_world": sdc_expected_pos_world,
            "sdc_expected_heading_world": sdc_expected_heading_world,
            "sdc_action_probs": sdc_probs,
            "sdc_next_pos_candidates_world": sdc_next_pos_candidates_world,
            "sdc_candidate_heading_world": sdc_candidate_heading_world,
            "family_paths_world": family_paths_world,
            "family_tangents_world": family_tangents_world,
            "family_arc_lengths": family_arc_lengths,
            "family_divergence_onsets": family_divergence_onsets,
            "family_path_mask": family_path_mask,
            "family_confidences": family_confidences,
            "family_weights": family_weights,
            "current_projection": current_projection,
            "expected_projection": expected_projection,
            "current_gate": current_gate,
            "expected_gate": expected_gate,
            "family_gate_mean": family_gate_mean,
            "is_factual": is_factual,
            "factual_batch_mask": factual_batch_mask,
            "control_available": control_available,
            "alternative_batch_mask": alternative_batch_mask,
        }

    def _run_policy_teacher(self, data_dict):
        if self.policy_teacher is None:
            return None
        teacher_input = {}
        for key, value in data_dict.items():
            if str(key).startswith("cf/"):
                continue
            if str(key).startswith("decoder/control_"):
                continue
            if str(key).startswith("decoder/debug_"):
                continue
            if str(key).startswith("decoder/rollout_"):
                continue
            if key in {"decoder/output_logit", "decoder/decoded_tokens"}:
                continue
            teacher_input[key] = clone_nested_value(value)
        with torch.no_grad():
            teacher_output = self.policy_teacher(teacher_input)
        return teacher_output.get("decoder/output_logit")

    def _compute_sdc_semantic_family_guidance(self, *, data_dict, semantic_context, teacher_logits):
        output_logit = data_dict["decoder/output_logit"]
        device = output_logit.device
        dtype = output_logit.dtype
        zero = output_logit.new_tensor(0.0)
        current_projection = semantic_context["current_projection"]
        expected_projection = semantic_context["expected_projection"]
        family_weights = semantic_context["family_weights"]
        current_gate = semantic_context["current_gate"]
        guide_weight = (current_gate * family_weights[:, None, :]).sum(dim=-1)
        valid_mask = semantic_context["sdc_valid_by_t"] & semantic_context["control_available"][:, None]
        guide_valid = valid_mask & (guide_weight > 1e-5)

        if teacher_logits is None or not bool(guide_valid.any()):
            projected_distance = (expected_projection["nearest_distance"] * family_weights[:, None, :]).sum(dim=-1)
            return {
                "guide_loss": zero,
                "guide_weight": guide_weight,
                "guide_valid_fraction": zero,
                "student_entropy": zero,
                "expected_energy": zero,
                "expected_position_penalty": zero,
                "expected_heading_penalty": zero,
                "expected_backward_penalty": zero,
                "family_teacher_entropy": zero,
                "projected_family_distance": projected_distance,
            }

        teacher_logits = sanitize_logits_for_loss(teacher_logits.to(device=device, dtype=dtype))
        decision_agent_mask = semantic_context["decision_agent_mask"]
        student_logits_sdc = sanitize_logits_for_loss((output_logit * decision_agent_mask[:, None, :, None]).sum(dim=2))
        teacher_logits_sdc = sanitize_logits_for_loss((teacher_logits * decision_agent_mask[:, None, :, None]).sum(dim=2))

        candidate_projection = project_points_to_family_paths_torch(
            semantic_context["sdc_next_pos_candidates_world"],
            family_path_polylines_world=semantic_context["family_paths_world"],
            family_path_mask=semantic_context["family_path_mask"],
            family_path_tangents_world=semantic_context["family_tangents_world"],
            family_path_arc_lengths=semantic_context["family_arc_lengths"],
        )
        position_deadband = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_FAMILY_PATH_DEADBAND_M", DEFAULT_FAMILY_PATH_DEADBAND_M)
        )
        heading_deadband = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_FAMILY_HEADING_DEADBAND_RAD", DEFAULT_FAMILY_HEADING_DEADBAND_RAD)
        )
        heading_beta = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_FAMILY_HEADING_BETA_RAD", DEFAULT_FAMILY_HEADING_BETA_RAD)
        )
        backward_slack = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_FAMILY_BACKWARD_SLACK_M", DEFAULT_FAMILY_BACKWARD_SLACK_M)
        )
        energy_temperature = float(
            self.config.MODEL.get("LOCAL_CONTROL_SDC_FAMILY_GUIDE_TEMPERATURE", DEFAULT_FAMILY_TEACHER_TEMPERATURE)
        )
        position_weight = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_FAMILY_PATH_PROX_WEIGHT", 1.0))
        heading_weight = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_FAMILY_HEADING_WEIGHT", 0.75))
        backward_weight = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_FAMILY_BACKWARD_WEIGHT", 0.5))

        distance = candidate_projection["nearest_distance"]
        nearest_arc = candidate_projection["nearest_arc"]
        current_arc = current_projection["nearest_arc"][:, :, None, :]
        candidate_heading = semantic_context["sdc_candidate_heading_world"][:, :, :, None]
        nearest_heading = candidate_projection["nearest_heading"]
        heading_delta = torch.atan2(
            torch.sin(candidate_heading - nearest_heading),
            torch.cos(candidate_heading - nearest_heading),
        ).abs()
        position_over = torch.relu(distance - position_deadband)
        position_penalty = F.smooth_l1_loss(position_over, torch.zeros_like(position_over), reduction="none")
        heading_over = torch.relu(heading_delta - heading_deadband)
        heading_penalty = F.smooth_l1_loss(
            heading_over,
            torch.zeros_like(heading_over),
            beta=max(heading_beta, 1e-3),
            reduction="none",
        )
        backward_penalty = torch.relu(current_arc - nearest_arc - backward_slack)
        energy = (
            position_weight * position_penalty
            + heading_weight * heading_penalty
            + backward_weight * backward_penalty
        ).permute(0, 1, 3, 2)
        energy = torch.nan_to_num(energy, nan=1e6, posinf=1e6, neginf=0.0)

        teacher_log_probs = F.log_softmax(teacher_logits_sdc, dim=-1)
        student_log_probs = F.log_softmax(student_logits_sdc, dim=-1)
        student_probs = torch.softmax(student_logits_sdc, dim=-1)
        family_teacher = torch.softmax(
            teacher_log_probs[:, :, None, :] - (energy / max(energy_temperature, 1e-3)),
            dim=-1,
        )
        family_teacher = torch.nan_to_num(family_teacher, nan=0.0, posinf=0.0, neginf=0.0)
        family_teacher = (family_teacher * family_weights[:, None, :, None]).sum(dim=2)
        family_teacher = family_teacher / family_teacher.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        family_teacher = torch.nan_to_num(family_teacher, nan=0.0, posinf=0.0, neginf=0.0)
        kl_per_step = torch.nan_to_num(
            (family_teacher * (torch.log(family_teacher.clamp_min(1e-6)) - student_log_probs)).sum(dim=-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        guide_loss = sanitize_scalar_loss(
            (kl_per_step[guide_valid] * guide_weight[guide_valid]).sum() / guide_weight[guide_valid].sum().clamp_min(1e-4)
        )
        guide_valid_f = guide_valid.to(dtype=dtype)
        guide_weighted_num = (kl_per_step * guide_weight * guide_valid_f).sum(dim=-1)
        guide_weighted_den = (guide_weight * guide_valid_f).sum(dim=-1)
        guide_loss_per_example = torch.where(
            guide_weighted_den > 1e-5,
            guide_weighted_num / guide_weighted_den.clamp_min(1e-4),
            torch.zeros_like(guide_weighted_num),
        )
        projected_distance = (expected_projection["nearest_distance"] * family_weights[:, None, :]).sum(dim=-1)
        family_teacher_entropy = -(family_teacher * torch.log(family_teacher.clamp_min(1e-6))).sum(dim=-1)
        student_entropy = -(student_probs * student_log_probs).sum(dim=-1)
        weighted_position_penalty = (position_penalty.permute(0, 1, 3, 2) * family_weights[:, None, :, None]).sum(dim=2)
        weighted_heading_penalty = (heading_penalty.permute(0, 1, 3, 2) * family_weights[:, None, :, None]).sum(dim=2)
        weighted_backward_penalty = (backward_penalty.permute(0, 1, 3, 2) * family_weights[:, None, :, None]).sum(dim=2)
        weighted_energy = (energy * family_weights[:, None, :, None]).sum(dim=2)
        expected_position_penalty = (weighted_position_penalty * student_probs).sum(dim=-1)
        expected_heading_penalty = (weighted_heading_penalty * student_probs).sum(dim=-1)
        expected_backward_penalty = (weighted_backward_penalty * student_probs).sum(dim=-1)
        expected_energy = (weighted_energy * student_probs).sum(dim=-1)
        return {
            "guide_loss": guide_loss,
            "guide_weight": guide_weight,
            "guide_valid_fraction": guide_valid.float().mean(),
            "guide_loss_per_example": torch.nan_to_num(guide_loss_per_example, nan=0.0, posinf=0.0, neginf=0.0),
            "guide_example_valid": guide_weighted_den > 1e-5,
            "student_entropy": sanitize_scalar_loss(student_entropy[guide_valid].mean()) if bool(guide_valid.any()) else zero,
            "expected_energy": sanitize_scalar_loss(expected_energy[guide_valid].mean()) if bool(guide_valid.any()) else zero,
            "expected_position_penalty": sanitize_scalar_loss(expected_position_penalty[guide_valid].mean()) if bool(guide_valid.any()) else zero,
            "expected_heading_penalty": sanitize_scalar_loss(expected_heading_penalty[guide_valid].mean()) if bool(guide_valid.any()) else zero,
            "expected_backward_penalty": sanitize_scalar_loss(expected_backward_penalty[guide_valid].mean()) if bool(guide_valid.any()) else zero,
            "family_teacher_entropy": sanitize_scalar_loss(family_teacher_entropy[guide_valid].mean()) if bool(guide_valid.any()) else zero,
            "projected_family_distance": torch.nan_to_num(projected_distance, nan=0.0, posinf=0.0, neginf=0.0),
        }

    def forward(self, batch_dict):
        return self.model(batch_dict)

    def get_loss(self, data_dict):

        loss_stat = {}
        loss = 0.0

        if self.config.USE_MOTION:

            # Get the decoder's output
            output_logit = data_dict["decoder/output_logit"]  # (B, T_skipped + 1, N, num_actions)

            # Get the GT actions
            target_action = data_dict["decoder/target_action"]  # (B, T_skipped, N)
            target_action_valid_mask = data_dict["decoder/target_action_valid_mask"]
            assert output_logit.shape[:3] == target_action.shape
            sdc_path_context = self._extract_sdc_path_context(data_dict)
            sdc_semantic_context = self._extract_sdc_semantic_context(data_dict)
            motion_token_weights = torch.ones_like(target_action, dtype=output_logit.dtype)
            if sdc_semantic_context is not None and self.sdc_semantic_ext_enabled:
                factual_batch_mask = sdc_semantic_context["factual_batch_mask"]
                motion_token_weights = factual_batch_mask[:, None, None].to(dtype=output_logit.dtype).expand_as(motion_token_weights)
            elif sdc_semantic_context is not None:
                stem_loss_weight = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_STEM_LOSS_WEIGHT", 1.0))
                stem_weight = (stem_loss_weight * (1.0 - sdc_semantic_context["family_gate_mean"].detach())).clamp(0.0, 1.0)
                sdc_mask = sdc_semantic_context["sdc_token_mask"]
                motion_token_weights = torch.where(
                    sdc_mask,
                    stem_weight[:, :, None].expand_as(motion_token_weights),
                    motion_token_weights,
                )
            elif sdc_path_context is not None:
                alternative_batch = sdc_path_context["alternative_batch_mask"]
                separability = sdc_path_context["projection"]["nearest_separability"].detach()
                alt_weight = (1.0 - separability).clamp(0.0, 1.0)
                alt_mask = sdc_path_context["sdc_token_mask"] & alternative_batch[:, None, None]
                motion_token_weights = torch.where(
                    alt_mask,
                    alt_weight[:, :, None].expand_as(motion_token_weights),
                    motion_token_weights,
                )

            valid_logits = sanitize_logits_for_loss(output_logit[target_action_valid_mask])
            valid_target = target_action[target_action_valid_mask]
            valid_motion_weights = motion_token_weights[target_action_valid_mask]

            # Get loss
            if self.config.OPTIMIZATION.USE_FOCAL_LOSS:
                from torchvision.ops import sigmoid_focal_loss
                # Compute Focal Loss
                alpha = 0.25
                gamma = 2
                target_onehot = F.one_hot(valid_target, valid_logits.shape[-1]).float()
                original_loss = sigmoid_focal_loss(
                    inputs=valid_logits,
                    targets=target_onehot,
                    alpha=alpha,
                    gamma=gamma,
                    reduction="none",
                ).mean(dim=-1)
            else:
                original_loss = torch.nn.functional.cross_entropy(input=valid_logits, target=valid_target, reduction="none")

            weight_denom = valid_motion_weights.sum().clamp_min(1.0)
            loss = sanitize_scalar_loss((original_loss * valid_motion_weights).sum() / weight_denom)

            with torch.no_grad():
                encodings = F.one_hot(valid_logits.argmax(-1),
                                      valid_logits.shape[-1]).float().reshape(-1, valid_logits.shape[-1])
                avg_probs = encodings.mean(0)
                perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
                cluster_use = torch.sum(avg_probs > 0)

                gt_onehot = F.one_hot(valid_target, valid_logits.shape[-1]).float()
                gt_encodings = gt_onehot.reshape(-1, valid_logits.shape[-1])
                gt_avg_probs = gt_encodings.mean(0)
                gt_perplexity = (-(gt_avg_probs * torch.log(gt_avg_probs + 1e-10)).sum()).exp()
                gt_cluster_use = torch.sum(gt_avg_probs > 0)
                debug_gt_c_use = (gt_encodings.sum(0) > 0).sum()  # .mean()

                pred_act = valid_logits.argmax(-1)
                acc = torch.sum(pred_act == valid_target) / valid_target.shape[0]
                entropy = safe_entropy(valid_logits)
                pred_act = pred_act.float()

                rate_default_pred = (pred_act == self._tokenizer.default_action).float().mean()
                rate_default_gt = (valid_target == self._tokenizer.default_action).float().mean()

                num_trained_tokens = len(valid_target)
                num_trained_tokens_sum = self.trainer.world_size * num_trained_tokens

                loss_stat.update(
                    {
                        "original_loss": (original_loss * valid_motion_weights).sum() / weight_denom,
                        "accuracy": acc,
                        "entropy": entropy.mean(),
                        "avg_action": pred_act.mean(),
                        "max_action": pred_act.max(),
                        "min_action": pred_act.min(),
                        "perplexity": perplexity,
                        "gt_perplexity": gt_perplexity,
                        "cluster_use": cluster_use,
                        "gt_cluster_use": gt_cluster_use,
                        "rate_84": rate_default_gt,
                        "rate_default_gt": rate_default_gt,
                        "rate_default_pred": rate_default_pred,
                        "num_trained_tokens": num_trained_tokens,
                        "num_trained_tokens_sum": num_trained_tokens_sum,
                        "toks": num_trained_tokens_sum,
                        "cf/sdc_motion_gt_weight_mean": valid_motion_weights.mean(),
                    }
                )
                if sdc_semantic_context is not None and self.sdc_semantic_ext_enabled:
                    loss_stat["cf/sdc_semantic_ext/gt_row_fraction"] = sdc_semantic_context["factual_batch_mask"].float().mean()
                    loss_stat["cf/sdc_semantic_ext/alt_row_fraction"] = sdc_semantic_context["alternative_batch_mask"].float().mean()
                if sdc_semantic_context is not None:
                    family_distance = sdc_semantic_context["expected_projection"]["nearest_distance"]
                    family_weights = sdc_semantic_context["family_weights"]
                    weighted_distance = (family_distance * family_weights[:, None, :]).sum(dim=-1)
                    loss_stat["cf/sdc_family_distance_mean"] = weighted_distance[
                        sdc_semantic_context["sdc_valid_by_t"]
                    ].mean() if bool(sdc_semantic_context["sdc_valid_by_t"].any()) else output_logit.new_tensor(0.0)
                    loss_stat["cf/sdc_family_gate_mean"] = sdc_semantic_context["family_gate_mean"][
                        sdc_semantic_context["sdc_valid_by_t"]
                    ].mean() if bool(sdc_semantic_context["sdc_valid_by_t"].any()) else output_logit.new_tensor(0.0)
                elif sdc_path_context is not None:
                    loss_stat["cf/sdc_nearest_path_distance_mean"] = sdc_path_context["projection"]["nearest_distance"][
                        sdc_path_context["sdc_valid_by_t"]
                    ].mean() if bool(sdc_path_context["sdc_valid_by_t"].any()) else output_logit.new_tensor(0.0)
                    loss_stat["cf/sdc_separability_mean"] = sdc_path_context["projection"]["nearest_separability"][
                        sdc_path_context["sdc_valid_by_t"]
                    ].mean() if bool(sdc_path_context["sdc_valid_by_t"].any()) else output_logit.new_tensor(0.0)

                if self.config.BACKWARD_PREDICTION:
                    in_back_mask = data_dict["in_backward_prediction"]
                    in_back_mask = in_back_mask.reshape(-1, 1, 1).expand(*target_action_valid_mask.shape)
                    in_back_mask = in_back_mask[target_action_valid_mask]
                    acc2 = (pred_act == valid_target)
                    acc_in_back = (acc2 & in_back_mask).sum() / in_back_mask.sum()
                    acc_in_forward = (acc2 & ~in_back_mask).sum() / (~in_back_mask).sum()
                    loss_in_back = original_loss[in_back_mask].mean()
                    loss_in_forward = original_loss[~in_back_mask].mean()
                    entropy_in_back = safe_entropy(valid_logits[in_back_mask]).mean()
                    entropy_in_forward = safe_entropy(valid_logits[~in_back_mask]).mean()
                    loss_stat.update(
                        {
                            "accuracy_in_backward": acc_in_back,
                            "accuracy_in_forward": acc_in_forward,
                            "loss_in_backward": loss_in_back,
                            "loss_in_forward": loss_in_forward,
                            "entropy_in_backward": entropy_in_back,
                            "entropy_in_forward": entropy_in_forward,
                            "backward_ratio": in_back_mask.float().mean(),
                        }
                    )

        if self.config.RECONSTRUCT_MAP:
            gt_map_feat = data_dict["encoder/map_feature"]
            map_feat_valid_mask = data_dict["encoder/map_valid_mask"]
            polypoint_valid_mask = data_dict["encoder/map_feature_valid_mask"]
            polypoint_valid_mask = polypoint_valid_mask[map_feat_valid_mask]  # (valid points, 128)
            map_feat = gt_map_feat[map_feat_valid_mask]  # (num_valid_map_features, 128, 27)
            polypoint = map_feat[:, :, :2]  # (valid map feat, 128, 2)
            num_points = polypoint.shape[1]
            gt_valid_mask = polypoint_valid_mask.unsqueeze(-1).expand_as(polypoint)
            gt = torch.where(gt_valid_mask, polypoint, torch.zeros_like(polypoint))
            gt_valid_mask = gt_valid_mask.reshape(-1, num_points * 2)
            gt = gt.reshape(-1, num_points * 2)
            map_token = data_dict["encoder/map_token"]
            out = self.model.map_recon_head(self.model.map_recon_head_prenorm(map_token[map_feat_valid_mask]))

            # out.shape = (num_valid_map_features, 128 * 2)
            map_recon_loss = torch.nn.functional.mse_loss(out, gt, reduction="none")
            map_recon_loss = map_recon_loss[gt_valid_mask]
            map_recon_loss = map_recon_loss.mean()

            loss += map_recon_loss
            loss_stat["map_recon_loss"] = map_recon_loss
            loss_stat["map_recon_mask_rate"] = gt_valid_mask.float().mean()

        if self.local_control_forward_enabled and "decoder/control_target_hidden" in data_dict:
            control_hidden = data_dict["decoder/control_target_hidden"]
            control_valid_mask = data_dict.get("decoder/control_target_valid_mask")
            if control_valid_mask is None:
                control_valid_mask = torch.ones(control_hidden.shape[0], device=control_hidden.device, dtype=torch.bool)
            else:
                control_valid_mask = control_valid_mask.to(device=control_hidden.device).bool()

            def _to_tensor(name, dtype=None):
                value = data_dict[name]
                if not torch.is_tensor(value):
                    value = torch.as_tensor(value, device=control_hidden.device)
                value = value.to(device=control_hidden.device)
                if dtype is not None:
                    value = value.to(dtype=dtype)
                return value

            conditioning_eligible = (
                _to_tensor("cf/conditioning_eligible", dtype=torch.bool)
                if "cf/conditioning_eligible" in data_dict
                else (_to_tensor("cf/control_available", dtype=torch.bool) if "cf/control_available" in data_dict else control_valid_mask)
            )

            if self.counterfactual_mode == "sdc_semantic_only" and self.sdc_semantic_head is not None and "cf/sdc_semantic_label_id" in data_dict:
                semantic_target = _to_tensor("cf/sdc_semantic_label_id", dtype=torch.long).reshape(control_hidden.shape[0], -1)[:, 0]
                semantic_supervision_mask = control_valid_mask & conditioning_eligible
                factual_batch_mask = (
                    sdc_semantic_context["factual_batch_mask"]
                    if sdc_semantic_context is not None
                    else torch.zeros_like(control_valid_mask, dtype=torch.bool)
                )
                alternative_batch_mask = (
                    sdc_semantic_context["alternative_batch_mask"]
                    if sdc_semantic_context is not None
                    else torch.zeros_like(control_valid_mask, dtype=torch.bool)
                )
                if self.sdc_semantic_ext_enabled:
                    semantic_supervision_mask = semantic_supervision_mask & alternative_batch_mask
                semantic_loss_weight = float(
                    self.config.MODEL.get(
                        "LOCAL_CONTROL_SDC_SEMANTIC_AUX_LOSS_WEIGHT",
                        self.config.MODEL.get("LOCAL_CONTROL_PATH_LOSS_WEIGHT", 0.2),
                    )
                )
                guide_loss_weight = float(
                    self.config.MODEL.get("LOCAL_CONTROL_SDC_FAMILY_GUIDE_LOSS_WEIGHT", 0.2)
                )
                rollout_guide_loss_weight = float(
                    self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_GUIDE_LOSS_WEIGHT", 0.0)
                )
                rollout_progress_loss_weight = float(
                    self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_PROGRESS_LOSS_WEIGHT", 0.0)
                )
                rollout_tube_policy_loss_weight = float(
                    self.config.MODEL.get("LOCAL_CONTROL_SDC_ROLLOUT_TUBE_POLICY_LOSS_WEIGHT", 0.0)
                )
                semantic_logits = sanitize_logits_for_loss(self.sdc_semantic_head(control_hidden))
                semantic_loss = control_hidden.new_tensor(0.0)
                if bool(semantic_supervision_mask.any()) and semantic_loss_weight > 0.0:
                    semantic_loss = sanitize_scalar_loss(F.cross_entropy(
                        semantic_logits[semantic_supervision_mask],
                        semantic_target[semantic_supervision_mask],
                        reduction="mean",
                    ))
                    loss = loss + semantic_loss_weight * semantic_loss
                    loss_stat["cf/sdc_semantic_acc"] = (
                        semantic_logits[semantic_supervision_mask].argmax(dim=-1) == semantic_target[semantic_supervision_mask]
                    ).float().mean()
                teacher_logits = self._run_policy_teacher(data_dict)
                guide_bundle = (
                    self._compute_sdc_semantic_family_guidance(
                        data_dict=data_dict,
                        semantic_context=sdc_semantic_context,
                        teacher_logits=teacher_logits,
                    )
                    if sdc_semantic_context is not None
                    else {
                        "guide_loss": control_hidden.new_tensor(0.0),
                        "guide_weight": control_hidden.new_zeros((control_hidden.shape[0], 1)),
                        "guide_valid_fraction": control_hidden.new_tensor(0.0),
                        "student_entropy": control_hidden.new_tensor(0.0),
                        "expected_energy": control_hidden.new_tensor(0.0),
                        "expected_position_penalty": control_hidden.new_tensor(0.0),
                        "expected_heading_penalty": control_hidden.new_tensor(0.0),
                        "expected_backward_penalty": control_hidden.new_tensor(0.0),
                        "family_teacher_entropy": control_hidden.new_tensor(0.0),
                        "projected_family_distance": control_hidden.new_zeros((control_hidden.shape[0], 1)),
                    }
                )
                rollout_guide_bundle = {
                    "guide_loss": control_hidden.new_tensor(0.0),
                    "guide_weight": control_hidden.new_zeros((control_hidden.shape[0], 1)),
                    "guide_valid_fraction": control_hidden.new_tensor(0.0),
                    "student_entropy": control_hidden.new_tensor(0.0),
                    "expected_energy": control_hidden.new_tensor(0.0),
                    "expected_position_penalty": control_hidden.new_tensor(0.0),
                    "expected_heading_penalty": control_hidden.new_tensor(0.0),
                    "expected_backward_penalty": control_hidden.new_tensor(0.0),
                    "family_teacher_entropy": control_hidden.new_tensor(0.0),
                    "projected_family_distance": control_hidden.new_zeros((control_hidden.shape[0], 1)),
                }
                rollout_progress_bundle = {
                    "progress_loss": control_hidden.new_tensor(0.0),
                    "progress_valid_fraction": control_hidden.new_tensor(0.0),
                    "realized_progress_mean": control_hidden.new_tensor(0.0),
                    "stall_fraction": control_hidden.new_tensor(0.0),
                    "rollout_family_distance": control_hidden.new_zeros((control_hidden.shape[0], 1)),
                }
                rollout_tube_policy_bundle = {
                    "policy_loss": control_hidden.new_tensor(0.0),
                    "valid_fraction": control_hidden.new_tensor(0.0),
                    "inside_fraction": control_hidden.new_tensor(0.0),
                    "return_mean": control_hidden.new_tensor(0.0),
                    "return_std": control_hidden.new_tensor(0.0),
                    "advantage_abs_mean": control_hidden.new_tensor(0.0),
                    "tube_distance_mean": control_hidden.new_tensor(0.0),
                    "progress_reward_mean": control_hidden.new_tensor(0.0),
                    "stall_penalty_mean": control_hidden.new_tensor(0.0),
                    "center_penalty_mean": control_hidden.new_tensor(0.0),
                    "traffic_speed_penalty_mean": control_hidden.new_tensor(0.0),
                    "traffic_speed_reference_mean": control_hidden.new_tensor(0.0),
                    "traffic_speed_neighbor_mean_mean": control_hidden.new_tensor(0.0),
                    "traffic_speed_neighbor_median_mean": control_hidden.new_tensor(0.0),
                    "traffic_speed_floor_penalty_mean": control_hidden.new_tensor(0.0),
                    "traffic_speed_cap_penalty_mean": control_hidden.new_tensor(0.0),
                    "traffic_speed_cap_threshold_mean": control_hidden.new_tensor(0.0),
                    "traffic_speed_sdc_mean": control_hidden.new_tensor(0.0),
                    "traffic_speed_neighbor_count_mean": control_hidden.new_tensor(0.0),
                    "center_curriculum_weight": control_hidden.new_tensor(0.0),
                    "consensus_bonus_mean": control_hidden.new_tensor(0.0),
                    "novelty_bonus_mean": control_hidden.new_tensor(0.0),
                    "novelty_score_mean": control_hidden.new_tensor(0.0),
                    "consensus_cluster_count_mean": control_hidden.new_tensor(0.0),
                    "consensus_cluster_entropy_mean": control_hidden.new_tensor(0.0),
                    "thermostat_target_entropy_mean": control_hidden.new_tensor(0.0),
                    "thermostat_alpha_mean": control_hidden.new_tensor(0.0),
                    "thermostat_eta_mean": control_hidden.new_tensor(0.0),
                    "behavior_aux_loss": control_hidden.new_tensor(0.0),
                    "behavior_semantic_ce": control_hidden.new_tensor(0.0),
                    "behavior_branch_ce": control_hidden.new_tensor(0.0),
                    "behavior_compliance_ce": control_hidden.new_tensor(0.0),
                    "behavior_timing_ce": control_hidden.new_tensor(0.0),
                    "behavior_dag_align": control_hidden.new_tensor(0.0),
                    "frontier_arc_final_mean": control_hidden.new_tensor(0.0),
                    "progress_cap_arc_mean": control_hidden.new_tensor(0.0),
                    "divergence_onset_mean": control_hidden.new_tensor(0.0),
                    "group_size": 0,
                }
                semantic_ext_policy_kl_bundle = {
                    "kl_gt": control_hidden.new_tensor(0.0),
                    "kl_alt": control_hidden.new_tensor(0.0),
                    "kl_total": control_hidden.new_tensor(0.0),
                    "gt_token_fraction": control_hidden.new_tensor(0.0),
                    "alt_token_fraction": control_hidden.new_tensor(0.0),
                }
                rollout_semantic_context = None
                if (
                    sdc_semantic_context is not None
                    and bool(sdc_semantic_context["alternative_batch_mask"].any())
                    and (
                        rollout_guide_loss_weight > 0.0
                        or rollout_progress_loss_weight > 0.0
                    )
                ):
                    rollout_bundle = self._build_sdc_semantic_rollout_guidance(data_dict=data_dict)
                    rollout_guide_bundle = rollout_bundle["guide_bundle"]
                    rollout_progress_bundle = rollout_bundle["progress_bundle"]
                    rollout_semantic_context = rollout_bundle["semantic_context"]
                if (
                    sdc_semantic_context is not None
                    and bool(sdc_semantic_context["alternative_batch_mask"].any())
                    and rollout_tube_policy_loss_weight > 0.0
                ):
                    rollout_tube_policy_bundle = self._build_sdc_semantic_rollout_tube_policy_objective(
                        data_dict=data_dict
                    )
                if self.sdc_semantic_ext_enabled:
                    semantic_ext_policy_kl_bundle = self._compute_sdc_semantic_ext_policy_kl(
                        data_dict=data_dict,
                        semantic_context=sdc_semantic_context,
                        teacher_logits=teacher_logits,
                    )
                if guide_loss_weight > 0.0 and sdc_semantic_context is not None:
                    loss = loss + guide_loss_weight * sanitize_scalar_loss(guide_bundle["guide_loss"])
                if rollout_guide_loss_weight > 0.0 and rollout_semantic_context is not None:
                    loss = loss + rollout_guide_loss_weight * sanitize_scalar_loss(rollout_guide_bundle["guide_loss"])
                if rollout_progress_loss_weight > 0.0 and rollout_semantic_context is not None:
                    loss = loss + rollout_progress_loss_weight * sanitize_scalar_loss(rollout_progress_bundle["progress_loss"])
                if rollout_tube_policy_loss_weight > 0.0:
                    loss = loss + rollout_tube_policy_loss_weight * sanitize_scalar_loss(rollout_tube_policy_bundle["policy_loss"])
                if self.sdc_semantic_ext_enabled:
                    loss = loss + sanitize_scalar_loss(rollout_tube_policy_bundle["behavior_aux_loss"])
                if self.sdc_semantic_ext_enabled:
                    loss = loss + sanitize_scalar_loss(semantic_ext_policy_kl_bundle["kl_total"])
                total_control_objective = (
                    guide_loss_weight * sanitize_scalar_loss(guide_bundle["guide_loss"])
                    + rollout_guide_loss_weight * sanitize_scalar_loss(rollout_guide_bundle["guide_loss"])
                    + rollout_progress_loss_weight * sanitize_scalar_loss(rollout_progress_bundle["progress_loss"])
                    + rollout_tube_policy_loss_weight * sanitize_scalar_loss(rollout_tube_policy_bundle["policy_loss"])
                    + sanitize_scalar_loss(rollout_tube_policy_bundle["behavior_aux_loss"])
                    + semantic_loss_weight * sanitize_scalar_loss(semantic_loss)
                    + sanitize_scalar_loss(semantic_ext_policy_kl_bundle["kl_total"])
                )
                loss_stat.update(
                    {
                        "cf/control_batch_fraction": conditioning_eligible.float().mean(),
                        "cf/conditioning_batch_fraction": conditioning_eligible.float().mean(),
                        "cf/sdc_semantic_supervision_fraction": semantic_supervision_mask.float().mean(),
                        "cf/sdc_semantic_aux_loss": semantic_loss,
                        "cf/sdc_semantic_aux_loss_weight": semantic_loss_weight,
                        "cf/sdc_family_guide_loss": guide_bundle["guide_loss"],
                        "cf/sdc_family_guide_loss_weight": guide_loss_weight,
                        "cf/sdc_family_guide_valid_fraction": guide_bundle["guide_valid_fraction"],
                        "cf/sdc_family_student_entropy": guide_bundle["student_entropy"],
                        "cf/sdc_family_teacher_entropy": guide_bundle["family_teacher_entropy"],
                        "cf/sdc_family_expected_energy": guide_bundle["expected_energy"],
                        "cf/sdc_family_expected_pos_penalty": guide_bundle["expected_position_penalty"],
                        "cf/sdc_family_expected_heading_penalty": guide_bundle["expected_heading_penalty"],
                        "cf/sdc_family_expected_backward_penalty": guide_bundle["expected_backward_penalty"],
                        "cf/sdc_family_gate_mean": guide_bundle["guide_weight"].mean(),
                        "cf/sdc_rollout_family_guide_loss": rollout_guide_bundle["guide_loss"],
                        "cf/sdc_rollout_family_guide_loss_weight": rollout_guide_loss_weight,
                        "cf/sdc_rollout_family_guide_valid_fraction": rollout_guide_bundle["guide_valid_fraction"],
                        "cf/sdc_rollout_family_student_entropy": rollout_guide_bundle["student_entropy"],
                        "cf/sdc_rollout_family_teacher_entropy": rollout_guide_bundle["family_teacher_entropy"],
                        "cf/sdc_rollout_family_expected_energy": rollout_guide_bundle["expected_energy"],
                        "cf/sdc_rollout_family_expected_pos_penalty": rollout_guide_bundle["expected_position_penalty"],
                        "cf/sdc_rollout_family_expected_heading_penalty": rollout_guide_bundle["expected_heading_penalty"],
                        "cf/sdc_rollout_family_expected_backward_penalty": rollout_guide_bundle["expected_backward_penalty"],
                        "cf/sdc_rollout_family_gate_mean": rollout_guide_bundle["guide_weight"].mean(),
                        "cf/sdc_rollout_progress_loss": rollout_progress_bundle["progress_loss"],
                        "cf/sdc_rollout_progress_loss_weight": rollout_progress_loss_weight,
                        "cf/sdc_rollout_progress_valid_fraction": rollout_progress_bundle["progress_valid_fraction"],
                        "cf/sdc_rollout_progress_mean": rollout_progress_bundle["realized_progress_mean"],
                        "cf/sdc_rollout_stall_fraction": rollout_progress_bundle["stall_fraction"],
                        "cf/sdc_rollout_tube_policy_loss": rollout_tube_policy_bundle["policy_loss"],
                        "cf/sdc_rollout_tube_policy_loss_weight": rollout_tube_policy_loss_weight,
                        "cf/sdc_rollout_tube_valid_fraction": rollout_tube_policy_bundle["valid_fraction"],
                        "cf/sdc_rollout_tube_inside_fraction": rollout_tube_policy_bundle["inside_fraction"],
                        "cf/sdc_rollout_tube_return_mean": rollout_tube_policy_bundle["return_mean"],
                        "cf/sdc_rollout_tube_return_std": rollout_tube_policy_bundle["return_std"],
                        "cf/sdc_rollout_tube_advantage_abs_mean": rollout_tube_policy_bundle["advantage_abs_mean"],
                        "cf/sdc_rollout_tube_distance_mean": rollout_tube_policy_bundle["tube_distance_mean"],
                        "cf/sdc_rollout_tube_progress_reward_mean": rollout_tube_policy_bundle["progress_reward_mean"],
                        "cf/sdc_rollout_tube_stall_penalty_mean": rollout_tube_policy_bundle["stall_penalty_mean"],
                        "cf/sdc_rollout_tube_center_penalty_mean": rollout_tube_policy_bundle["center_penalty_mean"],
                        "cf/sdc_rollout_tube_traffic_speed_penalty_mean": rollout_tube_policy_bundle["traffic_speed_penalty_mean"],
                        "cf/sdc_rollout_tube_traffic_speed_reference_mean": rollout_tube_policy_bundle["traffic_speed_reference_mean"],
                        "cf/sdc_rollout_tube_traffic_speed_neighbor_mean_mean": rollout_tube_policy_bundle["traffic_speed_neighbor_mean_mean"],
                        "cf/sdc_rollout_tube_traffic_speed_neighbor_median_mean": rollout_tube_policy_bundle["traffic_speed_neighbor_median_mean"],
                        "cf/sdc_rollout_tube_traffic_speed_floor_penalty_mean": rollout_tube_policy_bundle["traffic_speed_floor_penalty_mean"],
                        "cf/sdc_rollout_tube_traffic_speed_cap_penalty_mean": rollout_tube_policy_bundle["traffic_speed_cap_penalty_mean"],
                        "cf/sdc_rollout_tube_traffic_speed_cap_threshold_mean": rollout_tube_policy_bundle["traffic_speed_cap_threshold_mean"],
                        "cf/sdc_rollout_tube_traffic_speed_sdc_mean": rollout_tube_policy_bundle["traffic_speed_sdc_mean"],
                        "cf/sdc_rollout_tube_traffic_speed_neighbor_count_mean": rollout_tube_policy_bundle["traffic_speed_neighbor_count_mean"],
                        "cf/sdc_rollout_tube_center_curriculum_weight": rollout_tube_policy_bundle["center_curriculum_weight"],
                        "cf/sdc_rollout_tube_consensus_bonus_mean": rollout_tube_policy_bundle["consensus_bonus_mean"],
                        "cf/sdc_rollout_tube_novelty_bonus_mean": rollout_tube_policy_bundle["novelty_bonus_mean"],
                        "cf/sdc_rollout_tube_novelty_score_mean": rollout_tube_policy_bundle["novelty_score_mean"],
                        "cf/sdc_rollout_tube_consensus_cluster_count_mean": rollout_tube_policy_bundle["consensus_cluster_count_mean"],
                        "cf/sdc_rollout_tube_consensus_cluster_entropy_mean": rollout_tube_policy_bundle["consensus_cluster_entropy_mean"],
                        "cf/sdc_rollout_tube_thermostat_target_entropy_mean": rollout_tube_policy_bundle["thermostat_target_entropy_mean"],
                        "cf/sdc_rollout_tube_thermostat_alpha_mean": rollout_tube_policy_bundle["thermostat_alpha_mean"],
                        "cf/sdc_rollout_tube_thermostat_eta_mean": rollout_tube_policy_bundle["thermostat_eta_mean"],
                        "cf/sdc_rollout_tube_frontier_arc_final_mean": rollout_tube_policy_bundle["frontier_arc_final_mean"],
                        "cf/sdc_rollout_tube_progress_cap_arc_mean": rollout_tube_policy_bundle["progress_cap_arc_mean"],
                        "cf/sdc_rollout_tube_divergence_onset_mean": rollout_tube_policy_bundle["divergence_onset_mean"],
                        "cf/sdc_rollout_tube_group_size": control_hidden.new_tensor(
                            float(rollout_tube_policy_bundle["group_size"])
                        ),
                        "cf/sdc_semantic_ext/kl_gt": semantic_ext_policy_kl_bundle["kl_gt"],
                        "cf/sdc_semantic_ext/kl_alt": semantic_ext_policy_kl_bundle["kl_alt"],
                        "cf/sdc_semantic_ext/kl_total": semantic_ext_policy_kl_bundle["kl_total"],
                        "cf/sdc_semantic_ext/kl_gt_token_fraction": semantic_ext_policy_kl_bundle["gt_token_fraction"],
                        "cf/sdc_semantic_ext/kl_alt_token_fraction": semantic_ext_policy_kl_bundle["alt_token_fraction"],
                        "cf/sdc_semantic_ext/gt_row_fraction": factual_batch_mask.float().mean(),
                        "cf/sdc_semantic_ext/alt_row_fraction": alternative_batch_mask.float().mean(),
                        "cf/sdc_semantic_ext/behavior_aux_loss": rollout_tube_policy_bundle["behavior_aux_loss"],
                        "cf/sdc_semantic_ext/behavior_semantic_ce": rollout_tube_policy_bundle["behavior_semantic_ce"],
                        "cf/sdc_semantic_ext/behavior_branch_ce": rollout_tube_policy_bundle["behavior_branch_ce"],
                        "cf/sdc_semantic_ext/behavior_compliance_ce": rollout_tube_policy_bundle["behavior_compliance_ce"],
                        "cf/sdc_semantic_ext/behavior_timing_ce": rollout_tube_policy_bundle["behavior_timing_ce"],
                        "cf/sdc_semantic_ext/behavior_dag_align": rollout_tube_policy_bundle["behavior_dag_align"],
                        "cf/sdc_control_objective": total_control_objective,
                        "cf/sdc_family_distance_mean": guide_bundle["projected_family_distance"][
                            sdc_semantic_context["sdc_valid_by_t"]
                        ].mean() if sdc_semantic_context is not None and bool(sdc_semantic_context["sdc_valid_by_t"].any()) else control_hidden.new_tensor(0.0),
                        "cf/sdc_rollout_family_distance_mean": rollout_progress_bundle["rollout_family_distance"][
                            rollout_semantic_context["sdc_valid_by_t"]
                        ].mean() if rollout_semantic_context is not None and bool(rollout_semantic_context["sdc_valid_by_t"].any()) else control_hidden.new_tensor(0.0),
                    }
                )
                if "guide_loss_per_example" in guide_bundle and semantic_target.numel() == guide_bundle["guide_loss_per_example"].shape[0]:
                    guide_example_valid = guide_bundle.get("guide_example_valid", torch.zeros_like(semantic_target, dtype=torch.bool))
                    for label_id, label_name in enumerate(SDC_PATH_SEMANTIC_LABEL_ORDER):
                        label_mask = (semantic_target == int(label_id)) & guide_example_valid
                        if bool(label_mask.any()):
                            loss_stat[f"cf/sdc_family_guide_loss_by_label/{label_name}"] = sanitize_scalar_loss(
                                guide_bundle["guide_loss_per_example"][label_mask].mean()
                            )
                if "guide_loss_per_example" in rollout_guide_bundle and semantic_target.numel() == rollout_guide_bundle["guide_loss_per_example"].shape[0]:
                    rollout_guide_example_valid = rollout_guide_bundle.get(
                        "guide_example_valid",
                        torch.zeros_like(semantic_target, dtype=torch.bool),
                    )
                    for label_id, label_name in enumerate(SDC_PATH_SEMANTIC_LABEL_ORDER):
                        label_mask = (semantic_target == int(label_id)) & rollout_guide_example_valid
                        if bool(label_mask.any()):
                            loss_stat[f"cf/sdc_rollout_family_guide_loss_by_label/{label_name}"] = sanitize_scalar_loss(
                                rollout_guide_bundle["guide_loss_per_example"][label_mask].mean()
                            )
            elif self.counterfactual_mode == "sdc_path" and self.sdc_semantic_head is not None and "cf/sdc_semantic_label_id" in data_dict:
                semantic_target = _to_tensor("cf/sdc_semantic_label_id", dtype=torch.long).reshape(control_hidden.shape[0], -1)[:, 0]
                semantic_supervision_mask = control_valid_mask & conditioning_eligible
                semantic_loss_weight = float(self.config.MODEL.get("LOCAL_CONTROL_PATH_LOSS_WEIGHT", 0.2))
                prox_loss_weight = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_PATH_PROX_LOSS_WEIGHT", 0.2))
                heading_loss_weight = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_PATH_HEADING_LOSS_WEIGHT", 0.1))
                progress_loss_weight = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_PATH_PROGRESS_LOSS_WEIGHT", 0.05))
                policy_kl_weight = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_POLICY_KL_WEIGHT", 0.05))
                path_deadband_m = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_PATH_DEADBAND_M", DEFAULT_PATH_DEADBAND_M))
                heading_beta = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_PATH_HEADING_BETA_RAD", 0.35))
                progress_slack_m = float(self.config.MODEL.get("LOCAL_CONTROL_SDC_PATH_PROGRESS_BACKWARD_SLACK_M", 0.25))

                semantic_logits = sanitize_logits_for_loss(self.sdc_semantic_head(control_hidden))
                semantic_loss = control_hidden.new_tensor(0.0)
                prox_loss = control_hidden.new_tensor(0.0)
                heading_loss = control_hidden.new_tensor(0.0)
                progress_loss = control_hidden.new_tensor(0.0)
                policy_kl = control_hidden.new_tensor(0.0)

                if bool(semantic_supervision_mask.any()) and semantic_loss_weight > 0.0:
                    semantic_loss = sanitize_scalar_loss(F.cross_entropy(
                        semantic_logits[semantic_supervision_mask],
                        semantic_target[semantic_supervision_mask],
                        reduction="mean",
                    ))
                    loss = loss + semantic_loss_weight * semantic_loss
                    loss_stat["cf/sdc_semantic_acc"] = (
                        semantic_logits[semantic_supervision_mask].argmax(dim=-1) == semantic_target[semantic_supervision_mask]
                    ).float().mean()

                if sdc_path_context is not None:
                    geometry_mask = sdc_path_context["sdc_valid_by_t"] & sdc_path_context["control_available"][:, None]
                    separability = sdc_path_context["projection"]["nearest_separability"].detach()
                    distance = sdc_path_context["projection"]["nearest_distance"]
                    if bool(geometry_mask.any()) and prox_loss_weight > 0.0:
                        over = torch.relu(distance - path_deadband_m)
                        prox_penalty = F.smooth_l1_loss(over[geometry_mask], torch.zeros_like(over[geometry_mask]), reduction="none")
                        prox_weight = separability[geometry_mask]
                        prox_loss = (prox_penalty * prox_weight).sum() / prox_weight.sum().clamp_min(1e-4)
                        loss = loss + prox_loss_weight * prox_loss

                    if bool(geometry_mask.any()) and heading_loss_weight > 0.0:
                        heading_delta = torch.atan2(
                            torch.sin(sdc_path_context["sdc_expected_heading_local"] - sdc_path_context["projection"]["nearest_heading"]),
                            torch.cos(sdc_path_context["sdc_expected_heading_local"] - sdc_path_context["projection"]["nearest_heading"]),
                        ).abs()
                        heading_penalty = F.smooth_l1_loss(
                            heading_delta[geometry_mask],
                            torch.zeros_like(heading_delta[geometry_mask]),
                            beta=heading_beta,
                            reduction="none",
                        )
                        heading_weight = separability[geometry_mask]
                        heading_loss = (heading_penalty * heading_weight).sum() / heading_weight.sum().clamp_min(1e-4)
                        loss = loss + heading_loss_weight * heading_loss

                    progress_mask = geometry_mask[:, 1:] & geometry_mask[:, :-1]
                    if bool(progress_mask.any()) and progress_loss_weight > 0.0:
                        progress_delta = sdc_path_context["projection"]["nearest_arc"][:, 1:] - sdc_path_context["projection"]["nearest_arc"][:, :-1]
                        backward_penalty = torch.relu(-(progress_delta + progress_slack_m))
                        pair_weight = 0.5 * (separability[:, 1:] + separability[:, :-1])
                        progress_loss = (backward_penalty[progress_mask] * pair_weight[progress_mask]).sum() / pair_weight[progress_mask].sum().clamp_min(1e-4)
                        loss = loss + progress_loss_weight * progress_loss

                teacher_logits = self._run_policy_teacher(data_dict)
                if teacher_logits is not None and policy_kl_weight > 0.0:
                    teacher_logits = teacher_logits.to(device=control_hidden.device, dtype=data_dict["decoder/output_logit"].dtype)
                    student_logits = data_dict["decoder/output_logit"]
                    valid_mask = _to_tensor("decoder/target_action_valid_mask", dtype=torch.bool)
                    kl_per_token = F.kl_div(
                        F.log_softmax(student_logits, dim=-1),
                        F.softmax(teacher_logits, dim=-1),
                        reduction="none",
                    ).sum(dim=-1)
                    kl_weights = torch.ones_like(kl_per_token)
                    if sdc_path_context is not None:
                        alt_batch = sdc_path_context["alternative_batch_mask"]
                        alt_mask = sdc_path_context["sdc_token_mask"] & alt_batch[:, None, None]
                        relax = (1.0 - sdc_path_context["projection"]["nearest_separability"].detach()).clamp(0.0, 1.0)
                        kl_weights = torch.where(alt_mask, relax[:, :, None].expand_as(kl_weights), kl_weights)
                    kl_valid = valid_mask & (kl_weights >= 0.0)
                    if bool(kl_valid.any()):
                        policy_kl = (kl_per_token[kl_valid] * kl_weights[kl_valid]).sum() / kl_weights[kl_valid].sum().clamp_min(1e-4)
                        loss = loss + policy_kl_weight * policy_kl

                loss_stat.update(
                    {
                        "cf/control_batch_fraction": conditioning_eligible.float().mean(),
                        "cf/conditioning_batch_fraction": conditioning_eligible.float().mean(),
                        "cf/path_supervision_fraction": semantic_supervision_mask.float().mean(),
                        "cf/compliance_supervision_fraction": control_hidden.new_tensor(0.0),
                        "cf/timing_supervision_fraction": control_hidden.new_tensor(0.0),
                        "cf/path_loss": semantic_loss,
                        "cf/compliance_loss": control_hidden.new_tensor(0.0),
                        "cf/timing_loss": control_hidden.new_tensor(0.0),
                        "cf/anchor_loss": control_hidden.new_tensor(0.0),
                        "cf/path_loss_weight": semantic_loss_weight,
                        "cf/compliance_loss_weight": 0.0,
                        "cf/timing_loss_weight": 0.0,
                        "cf/anchor_loss_weight": 0.0,
                        "cf/sdc_path_prox_loss": prox_loss,
                        "cf/sdc_path_heading_loss": heading_loss,
                        "cf/sdc_path_progress_loss": progress_loss,
                        "cf/sdc_policy_kl": policy_kl,
                        "cf/sdc_path_prox_loss_weight": prox_loss_weight,
                        "cf/sdc_path_heading_loss_weight": heading_loss_weight,
                        "cf/sdc_path_progress_loss_weight": progress_loss_weight,
                        "cf/sdc_policy_kl_weight": policy_kl_weight,
                    }
                )
            else:
                path_supervision_mask = _to_tensor("cf/path_supervision_mask", dtype=torch.bool) if "cf/path_supervision_mask" in data_dict else control_valid_mask.new_zeros(control_valid_mask.shape)
                compliance_supervision_mask = _to_tensor("cf/compliance_supervision_mask", dtype=torch.bool) if "cf/compliance_supervision_mask" in data_dict else control_valid_mask.new_zeros(control_valid_mask.shape)
                timing_supervision_mask = _to_tensor("cf/timing_supervision_mask", dtype=torch.bool) if "cf/timing_supervision_mask" in data_dict else control_valid_mask.new_zeros(control_valid_mask.shape)

                path_supervision_mask = path_supervision_mask & control_valid_mask
                compliance_supervision_mask = compliance_supervision_mask & control_valid_mask
                timing_supervision_mask = timing_supervision_mask & control_valid_mask
                if self.counterfactual_mode == "path_only":
                    compliance_supervision_mask = compliance_supervision_mask & torch.zeros_like(compliance_supervision_mask)
                    timing_supervision_mask = timing_supervision_mask & torch.zeros_like(timing_supervision_mask)
                if not bool(self.config.MODEL.get("LOCAL_CONTROL_USE_PATH", True)):
                    path_supervision_mask = path_supervision_mask & torch.zeros_like(path_supervision_mask)
                if not bool(self.config.MODEL.get("LOCAL_CONTROL_USE_COMPLIANCE", True)):
                    compliance_supervision_mask = compliance_supervision_mask & torch.zeros_like(compliance_supervision_mask)
                if not bool(self.config.MODEL.get("LOCAL_CONTROL_USE_TIMING", True)):
                    timing_supervision_mask = timing_supervision_mask & torch.zeros_like(timing_supervision_mask)
                anchor_supervision_mask = path_supervision_mask if bool(self.config.MODEL.get("LOCAL_CONTROL_USE_ANCHOR", True)) else torch.zeros_like(path_supervision_mask)

                path_loss_weight = float(self.config.MODEL.get("LOCAL_CONTROL_PATH_LOSS_WEIGHT", 0.2))
                compliance_loss_weight = float(self.config.MODEL.get("LOCAL_CONTROL_COMPLIANCE_LOSS_WEIGHT", 0.1))
                timing_loss_weight = float(self.config.MODEL.get("LOCAL_CONTROL_TIMING_LOSS_WEIGHT", 0.1))
                anchor_loss_weight = float(self.config.MODEL.get("LOCAL_CONTROL_ANCHOR_LOSS_WEIGHT", 0.1))
                if self.counterfactual_mode == "path_only":
                    compliance_loss_weight = 0.0
                    timing_loss_weight = 0.0

                if "cf/path_token" in data_dict:
                    path_target = _to_tensor("cf/path_token", dtype=torch.float32)[:, 0].long()
                else:
                    path_target = torch.zeros(control_hidden.shape[0], device=control_hidden.device, dtype=torch.long)
                if "cf/compliance_token" in data_dict:
                    compliance_target = _to_tensor("cf/compliance_token", dtype=torch.float32)[:, 1].long()
                else:
                    compliance_target = torch.zeros(control_hidden.shape[0], device=control_hidden.device, dtype=torch.long)
                if "cf/timing_token" in data_dict:
                    timing_target = _to_tensor("cf/timing_token", dtype=torch.float32)[:, 2].long()
                else:
                    timing_target = torch.zeros(control_hidden.shape[0], device=control_hidden.device, dtype=torch.long)
                anchor_target = _to_tensor("cf/terminal_anchor", dtype=torch.float32) if "cf/terminal_anchor" in data_dict else torch.zeros(
                    (control_hidden.shape[0], TERMINAL_ANCHOR_DIM),
                    device=control_hidden.device,
                    dtype=torch.float32,
                )

                path_logits = self.path_head(control_hidden)
                compliance_logits = self.compliance_head(control_hidden)
                timing_logits = self.timing_head(control_hidden)
                anchor_pred = self.anchor_head(control_hidden)

                path_loss = control_hidden.new_tensor(0.0)
                compliance_loss = control_hidden.new_tensor(0.0)
                timing_loss = control_hidden.new_tensor(0.0)
                anchor_loss = control_hidden.new_tensor(0.0)

                if bool(path_supervision_mask.any()) and path_loss_weight > 0.0:
                    path_loss = F.cross_entropy(path_logits[path_supervision_mask], path_target[path_supervision_mask], reduction="mean")
                    loss = loss + path_loss_weight * path_loss
                    loss_stat["cf/path_acc"] = (path_logits[path_supervision_mask].argmax(dim=-1) == path_target[path_supervision_mask]).float().mean()
                if bool(compliance_supervision_mask.any()) and compliance_loss_weight > 0.0:
                    compliance_loss = F.cross_entropy(
                        compliance_logits[compliance_supervision_mask],
                        compliance_target[compliance_supervision_mask],
                        reduction="mean",
                    )
                    loss = loss + compliance_loss_weight * compliance_loss
                    loss_stat["cf/compliance_acc"] = (
                        compliance_logits[compliance_supervision_mask].argmax(dim=-1) == compliance_target[compliance_supervision_mask]
                    ).float().mean()
                if bool(timing_supervision_mask.any()) and timing_loss_weight > 0.0:
                    timing_loss = F.cross_entropy(
                        timing_logits[timing_supervision_mask],
                        timing_target[timing_supervision_mask],
                        reduction="mean",
                    )
                    loss = loss + timing_loss_weight * timing_loss
                    loss_stat["cf/timing_acc"] = (
                        timing_logits[timing_supervision_mask].argmax(dim=-1) == timing_target[timing_supervision_mask]
                    ).float().mean()
                if bool(anchor_supervision_mask.any()) and anchor_loss_weight > 0.0:
                    anchor_loss = F.smooth_l1_loss(anchor_pred[anchor_supervision_mask], anchor_target[anchor_supervision_mask], reduction="mean")
                    loss = loss + anchor_loss_weight * anchor_loss

                loss_stat.update(
                    {
                        "cf/control_batch_fraction": conditioning_eligible.float().mean(),
                        "cf/conditioning_batch_fraction": conditioning_eligible.float().mean(),
                        "cf/path_supervision_fraction": path_supervision_mask.float().mean(),
                        "cf/compliance_supervision_fraction": compliance_supervision_mask.float().mean(),
                        "cf/timing_supervision_fraction": timing_supervision_mask.float().mean(),
                        "cf/path_loss": path_loss,
                        "cf/compliance_loss": compliance_loss,
                        "cf/timing_loss": timing_loss,
                        "cf/anchor_loss": anchor_loss,
                        "cf/path_loss_weight": path_loss_weight,
                        "cf/compliance_loss_weight": compliance_loss_weight,
                        "cf/timing_loss_weight": timing_loss_weight,
                        "cf/anchor_loss_weight": anchor_loss_weight,
                    }
                )

        # DEBUG CODE to find unused parameters:
        # gs = torch.autograd.grad(loss.mean(), self.parameters(), allow_unused=True, retain_graph=True)
        # ns = [n for n, v in self.named_parameters()]
        # printed = False
        # for c, g in enumerate(gs):
        #     if g is None:
        #         print(ns[c])
        #         printed = True
        # if not printed:
        #     print("No unused parameters found.")


        loss_stat["total_loss"] = loss
        try:
            scheduler_configs = getattr(self.trainer, "lr_scheduler_configs", None)
            if scheduler_configs:
                loss_stat["lr"] = scheduler_configs[0].scheduler.get_last_lr()[0]
            else:
                optimizers = getattr(self.trainer, "optimizers", None)
                if optimizers:
                    loss_stat["lr"] = optimizers[0].param_groups[0]["lr"]
        except (RuntimeError, AttributeError, IndexError, KeyError):
            # Pure validation/eval runs may not construct schedulers, and some
            # debug paths do not attach a full trainer at all.
            pass
        return loss, loss_stat

    def _filter_logged_loss_stat(self, loss_stat):
        if not isinstance(loss_stat, dict):
            return loss_stat

        if self.counterfactual_mode != "sdc_semantic_only":
            return loss_stat

        model_cfg = self.config.MODEL
        semantic_aux_weight = float(model_cfg.get("LOCAL_CONTROL_SDC_SEMANTIC_AUX_LOSS_WEIGHT", 0.0))
        family_guide_weight = float(model_cfg.get("LOCAL_CONTROL_SDC_FAMILY_GUIDE_LOSS_WEIGHT", 0.0))
        rollout_guide_weight = float(model_cfg.get("LOCAL_CONTROL_SDC_ROLLOUT_GUIDE_LOSS_WEIGHT", 0.0))
        rollout_progress_weight = float(model_cfg.get("LOCAL_CONTROL_SDC_ROLLOUT_PROGRESS_LOSS_WEIGHT", 0.0))
        rollout_tube_weight = float(model_cfg.get("LOCAL_CONTROL_SDC_ROLLOUT_TUBE_POLICY_LOSS_WEIGHT", 0.0))
        progress_reward_scale = float(model_cfg.get("LOCAL_CONTROL_SDC_ROLLOUT_TUBE_PROGRESS_REWARD_SCALE", 0.0))
        semantic_ext_enabled = bool(model_cfg.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_ENABLED", False))
        semantic_ext_allow_progress_reward = bool(
            model_cfg.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_ALLOW_PROGRESS_REWARD", False)
        )
        gt_kl_weight = float(model_cfg.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_KL_WEIGHT_GT", 0.0))
        alt_kl_weight = float(model_cfg.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_KL_WEIGHT_ALT", 0.0))
        stall_weight = float(model_cfg.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_STALL_WEIGHT", 0.0))
        center_weight_max = float(model_cfg.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_CENTER_WEIGHT_MAX", 0.0))
        speed_floor_enabled = bool(
            model_cfg.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_FLOOR_ENABLED", False)
        )
        speed_cap_enabled = bool(
            model_cfg.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_CAP_ENABLED", False)
        )
        consensus_weight = float(model_cfg.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_CONSENSUS_WEIGHT", 0.0))
        novelty_enabled = bool(model_cfg.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_NOVELTY_ENABLED", False))
        novelty_base_weight = float(model_cfg.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_NOVELTY_BASE_WEIGHT", 0.0))
        thermostat_enabled = bool(model_cfg.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_THERMOSTAT_ENABLED", False))
        behavior_model_enabled = bool(
            model_cfg.get("LOCAL_CONTROL_SDC_SEMANTIC_EXT_BEHAVIOR_MODEL_ENABLED", False)
        )

        filtered = {}
        for key, value in loss_stat.items():
            keep = True
            if key.startswith("cf/path_") or key.startswith("cf/compliance_") or key.startswith("cf/timing_") or key.startswith("cf/anchor_"):
                keep = False
            elif key in ("cf/sdc_semantic_acc", "cf/sdc_semantic_aux_loss", "cf/sdc_semantic_aux_loss_weight"):
                keep = semantic_aux_weight > 0.0
            elif key.startswith("cf/sdc_family_guide_loss_by_label/") or key in (
                "cf/sdc_family_guide_loss",
                "cf/sdc_family_guide_loss_weight",
                "cf/sdc_family_guide_valid_fraction",
                "cf/sdc_family_student_entropy",
                "cf/sdc_family_teacher_entropy",
                "cf/sdc_family_expected_energy",
                "cf/sdc_family_expected_pos_penalty",
                "cf/sdc_family_expected_heading_penalty",
                "cf/sdc_family_expected_backward_penalty",
                "cf/sdc_family_gate_mean",
                "cf/sdc_family_distance_mean",
            ):
                keep = family_guide_weight > 0.0
            elif key.startswith("cf/sdc_rollout_family_guide_loss_by_label/") or key in (
                "cf/sdc_rollout_family_guide_loss",
                "cf/sdc_rollout_family_guide_loss_weight",
                "cf/sdc_rollout_family_guide_valid_fraction",
                "cf/sdc_rollout_family_student_entropy",
                "cf/sdc_rollout_family_teacher_entropy",
                "cf/sdc_rollout_family_expected_energy",
                "cf/sdc_rollout_family_expected_pos_penalty",
                "cf/sdc_rollout_family_expected_heading_penalty",
                "cf/sdc_rollout_family_expected_backward_penalty",
                "cf/sdc_rollout_family_gate_mean",
                "cf/sdc_rollout_family_distance_mean",
            ):
                keep = rollout_guide_weight > 0.0
            elif key in (
                "cf/sdc_rollout_progress_loss",
                "cf/sdc_rollout_progress_loss_weight",
                "cf/sdc_rollout_progress_valid_fraction",
                "cf/sdc_rollout_progress_mean",
                "cf/sdc_rollout_stall_fraction",
            ):
                keep = rollout_progress_weight > 0.0
            elif key.startswith("cf/sdc_rollout_tube_"):
                if rollout_tube_weight <= 0.0:
                    keep = False
                elif key == "cf/sdc_rollout_tube_progress_reward_mean":
                    keep = progress_reward_scale > 0.0 and (
                        (not semantic_ext_enabled) or semantic_ext_allow_progress_reward
                    )
                elif key == "cf/sdc_rollout_tube_stall_penalty_mean":
                    keep = semantic_ext_enabled and (stall_weight > 0.0)
                elif key in (
                    "cf/sdc_rollout_tube_center_penalty_mean",
                    "cf/sdc_rollout_tube_center_curriculum_weight",
                ):
                    keep = semantic_ext_enabled and (center_weight_max > 0.0)
                elif key in (
                    "cf/sdc_rollout_tube_traffic_speed_penalty_mean",
                    "cf/sdc_rollout_tube_traffic_speed_neighbor_count_mean",
                    "cf/sdc_rollout_tube_traffic_speed_reference_mean",
                    "cf/sdc_rollout_tube_traffic_speed_sdc_mean",
                ):
                    keep = semantic_ext_enabled and (speed_floor_enabled or speed_cap_enabled)
                elif key in (
                    "cf/sdc_rollout_tube_traffic_speed_neighbor_mean_mean",
                    "cf/sdc_rollout_tube_traffic_speed_cap_penalty_mean",
                    "cf/sdc_rollout_tube_traffic_speed_cap_threshold_mean",
                ):
                    keep = semantic_ext_enabled and speed_cap_enabled
                elif key in (
                    "cf/sdc_rollout_tube_traffic_speed_neighbor_median_mean",
                    "cf/sdc_rollout_tube_traffic_speed_floor_penalty_mean",
                ):
                    keep = semantic_ext_enabled and speed_floor_enabled
                elif key == "cf/sdc_rollout_tube_consensus_bonus_mean" or key.startswith(
                    "cf/sdc_rollout_tube_consensus_cluster_"
                ):
                    keep = semantic_ext_enabled and (consensus_weight > 0.0)
                elif key in (
                    "cf/sdc_rollout_tube_novelty_bonus_mean",
                    "cf/sdc_rollout_tube_novelty_score_mean",
                ):
                    keep = semantic_ext_enabled and novelty_enabled and (novelty_base_weight > 0.0)
                elif key.startswith("cf/sdc_rollout_tube_thermostat_"):
                    keep = semantic_ext_enabled and thermostat_enabled
            elif key.startswith("cf/sdc_semantic_ext/"):
                if not semantic_ext_enabled:
                    keep = False
                elif key in ("cf/sdc_semantic_ext/kl_gt", "cf/sdc_semantic_ext/kl_gt_token_fraction"):
                    keep = gt_kl_weight > 0.0
                elif key in ("cf/sdc_semantic_ext/kl_alt", "cf/sdc_semantic_ext/kl_alt_token_fraction"):
                    keep = alt_kl_weight > 0.0
                elif key == "cf/sdc_semantic_ext/kl_total":
                    keep = (gt_kl_weight > 0.0) or (alt_kl_weight > 0.0)
                elif key.startswith("cf/sdc_semantic_ext/behavior_"):
                    keep = behavior_model_enabled

            if keep:
                filtered[key] = value

        if (
            "cf/sdc_rollout_tube_traffic_speed_reference_mean" in loss_stat
            and semantic_ext_enabled
            and (speed_floor_enabled or speed_cap_enabled)
            and rollout_tube_weight > 0.0
        ):
            filtered["cf/sdc_rollout_tube_neighbor_speed_mean"] = loss_stat[
                "cf/sdc_rollout_tube_traffic_speed_reference_mean"
            ]
        if (
            "cf/sdc_rollout_tube_traffic_speed_sdc_mean" in loss_stat
            and semantic_ext_enabled
            and (speed_floor_enabled or speed_cap_enabled)
            and rollout_tube_weight > 0.0
        ):
            filtered["cf/sdc_rollout_tube_sdc_speed_mean"] = loss_stat[
                "cf/sdc_rollout_tube_traffic_speed_sdc_mean"
            ]

        return filtered

    def training_step(self, data_dict, batch_idx):

        # For profiling GPU usage.
        # torch.cuda.empty_cache()
        self._trace_first_step("training_step:start", batch_idx=int(batch_idx))

        if "in_evaluation" in data_dict:
            in_evaluation = data_dict["in_evaluation"]
            if not torch.is_tensor(in_evaluation):
                in_evaluation = torch.as_tensor(in_evaluation)
            data_dict["in_evaluation"] = torch.zeros_like(in_evaluation, dtype=torch.bool)

        self._trace_first_step("training_step:before_forward")
        data_dict = self(data_dict)
        self._trace_first_step("training_step:after_forward")

        loss, loss_stat = self.get_loss(data_dict)
        self._trace_first_step("training_step:after_get_loss", loss=loss)

        pbar_keys = ("total_loss", "toks", "lr")

        motion_stat = {k: v for k, v in loss_stat.items() if k.startswith("motion_stat")}
        loss_stat = {k: v for k, v in loss_stat.items() if not k.startswith("motion_stat")}
        logged_loss_stat = self._filter_logged_loss_stat(loss_stat)

        self.log_dict(
            {f"{k}": float(v)
             for k, v in logged_loss_stat.items() if k in pbar_keys},
            batch_size=data_dict["encoder/map_feature"].shape[0],
            prog_bar=True,
        )
        if motion_stat:
            self.log_dict(
                {f"{k}": float(v)
                 for k, v in motion_stat.items()},
                batch_size=data_dict["encoder/map_feature"].shape[0],
                prog_bar=False,
            )
        self.log_dict(
            {f"train/{k}": float(v)
             for k, v in logged_loss_stat.items()},
            batch_size=data_dict["encoder/map_feature"].shape[0],
            # on_epoch=True,
            prog_bar=False,
        )
        self.log('monitoring_step', float(self.global_step))
        self._trace_first_step("training_step:end")
        return loss

    def on_before_backward(self, loss):
        self._trace_first_step("before_backward", loss=loss)

    def on_after_backward(self):
        self._trace_first_step("after_backward")

    def optimizer_step(self, *args, **kwargs):
        self._trace_first_step("optimizer_step:enter")
        grad_clip_norm = float(self.config.MODEL.get("LOCAL_CONTROL_GRAD_CLIP_NORM", 1.0))
        if grad_clip_norm > 0.0:
            parameters = [p for p in self.parameters() if p.requires_grad and p.grad is not None]
            if parameters:
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=grad_clip_norm)
        super().optimizer_step(*args, **kwargs)
        self._trace_first_step("optimizer_step:exit")

    def on_validation_start(self):
        torch.cuda.empty_cache()

    def _normalize_validation_loss_stat(self, loss_stat):
        if not isinstance(loss_stat, dict):
            return loss_stat

        zero = None
        for value in loss_stat.values():
            if torch.is_tensor(value):
                zero = value.new_tensor(0.0)
                break
        if zero is None:
            device = self.device if hasattr(self, "device") else None
            zero = torch.tensor(0.0, device=device)

        optional_keys = [
            "cf/sdc_semantic_acc",
            "cf/path_acc",
            "cf/compliance_acc",
            "cf/timing_acc",
            "accuracy_in_backward",
            "accuracy_in_forward",
        ]
        for key in optional_keys:
            loss_stat.setdefault(key, zero)

        if self.counterfactual_mode == "sdc_semantic_only":
            for label_name in SDC_PATH_SEMANTIC_LABEL_ORDER:
                loss_stat.setdefault(f"cf/sdc_family_guide_loss_by_label/{label_name}", zero)
                loss_stat.setdefault(f"cf/sdc_rollout_family_guide_loss_by_label/{label_name}", zero)
            if self.sdc_semantic_ext_enabled:
                for key in (
                    "cf/sdc_semantic_ext/kl_gt",
                    "cf/sdc_semantic_ext/kl_alt",
                    "cf/sdc_semantic_ext/kl_total",
                    "cf/sdc_semantic_ext/kl_gt_token_fraction",
                    "cf/sdc_semantic_ext/kl_alt_token_fraction",
                    "cf/sdc_semantic_ext/gt_row_fraction",
                    "cf/sdc_semantic_ext/alt_row_fraction",
                    "cf/sdc_semantic_ext/behavior_aux_loss",
                    "cf/sdc_semantic_ext/behavior_semantic_ce",
                    "cf/sdc_semantic_ext/behavior_branch_ce",
                    "cf/sdc_semantic_ext/behavior_compliance_ce",
                    "cf/sdc_semantic_ext/behavior_timing_ce",
                    "cf/sdc_semantic_ext/behavior_dag_align",
                    "cf/sdc_rollout_tube_stall_penalty_mean",
                    "cf/sdc_rollout_tube_center_penalty_mean",
                    "cf/sdc_rollout_tube_traffic_speed_penalty_mean",
                    "cf/sdc_rollout_tube_traffic_speed_reference_mean",
                    "cf/sdc_rollout_tube_traffic_speed_neighbor_mean_mean",
                    "cf/sdc_rollout_tube_traffic_speed_neighbor_median_mean",
                    "cf/sdc_rollout_tube_traffic_speed_floor_penalty_mean",
                    "cf/sdc_rollout_tube_traffic_speed_cap_penalty_mean",
                    "cf/sdc_rollout_tube_traffic_speed_cap_threshold_mean",
                    "cf/sdc_rollout_tube_traffic_speed_sdc_mean",
                    "cf/sdc_rollout_tube_traffic_speed_neighbor_count_mean",
                    "cf/sdc_rollout_tube_center_curriculum_weight",
                    "cf/sdc_rollout_tube_consensus_bonus_mean",
                    "cf/sdc_rollout_tube_novelty_bonus_mean",
                    "cf/sdc_rollout_tube_novelty_score_mean",
                    "cf/sdc_rollout_tube_consensus_cluster_count_mean",
                    "cf/sdc_rollout_tube_consensus_cluster_entropy_mean",
                    "cf/sdc_rollout_tube_thermostat_target_entropy_mean",
                    "cf/sdc_rollout_tube_thermostat_alpha_mean",
                    "cf/sdc_rollout_tube_thermostat_eta_mean",
                ):
                    loss_stat.setdefault(key, zero)

        return loss_stat

    def validation_step(self, data_dict, batch_idx):

        if self.config.EVAL_MOTION and hasattr(self, "evaluator"):

            if data_dict["encoder/map_valid_mask"].shape[1] == 0:
                sid = data_dict["scenario_id"]
                print("Warning: Empty map_valid_mask found for scenario: ", sid)
                logger.error(f"Empty map_valid_mask found for scenario: {sid}")
                return

            try:
                self.evaluator.validation_step(
                    data_dict,
                    batch_idx,
                    model=self.model,
                    global_rank=self.global_rank,
                    trainer=self.trainer,
                    logger=self.logger,
                    log_func=self.log,
                    log_dict_func=self.log_dict,
                    print_func=self.print,
                    lightning_model=self,
                )
            except Exception as error:
                scenario_ids = data_dict["scenario_id"]
                rank = self.global_rank
                msg = f"Error in validation_step: {batch_idx=}, {scenario_ids=}, {rank=}, {error=}"
                print(msg)
                raise RuntimeError(msg) from error
            return None

        if "in_evaluation" in data_dict:
            in_evaluation = data_dict["in_evaluation"]
            if not torch.is_tensor(in_evaluation):
                in_evaluation = torch.as_tensor(in_evaluation)
            data_dict["in_evaluation"] = torch.zeros_like(in_evaluation, dtype=torch.bool)

        data_dict = self(data_dict)
        loss, loss_stat = self.get_loss(data_dict)
        loss_stat = self._normalize_validation_loss_stat(loss_stat)
        motion_stat = {k: v for k, v in loss_stat.items() if k.startswith("motion_stat")}
        loss_stat = {k: v for k, v in loss_stat.items() if not k.startswith("motion_stat")}
        logged_loss_stat = self._filter_logged_loss_stat(loss_stat)

        self.log_dict(
            {f"val/{k}": float(v) for k, v in logged_loss_stat.items()},
            batch_size=data_dict["encoder/map_feature"].shape[0],
            prog_bar=False,
            sync_dist=True,
        )
        if motion_stat:
            self.log_dict(
                {f"val/{k}": float(v) for k, v in motion_stat.items()},
                batch_size=data_dict["encoder/map_feature"].shape[0],
                prog_bar=False,
                sync_dist=True,
            )
        self.log("val/monitoring_step", float(self.global_step), sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        """
        This function gathers intermediate evaluation result and pass them to the Waymo
        evaluation pipeline together and log the final results.
        """
        if self.config.EVAL_MOTION and hasattr(self, "evaluator"):
            self.log("monitoring_step", float(self.global_step))
            self.evaluator.on_validation_epoch_end(
                global_rank=self.global_rank,
                trainer=self.trainer,
                logger=self.logger,
                log_func=self.log,
                log_dict_func=self.log_dict,
                print_func=self.print,
                exp_name=self.exp_name,
            )

    def configure_optimizers(self):
        """Required by Lightning."""
        opt_cfg = self.config.OPTIMIZATION

        if opt_cfg.OPTIMIZER == 'Adam':
            # optimizer = torch.optim.Adam(
            #     [each[1] for each in self.named_parameters()],
            #     lr=opt_cfg.LR,
            #     weight_decay=opt_cfg.get('WEIGHT_DECAY', 0)
            # )
            raise ValueError()
        elif opt_cfg.OPTIMIZER == 'AdamW':
            optimizer = torch.optim.AdamW(
                [parameter for parameter in self.parameters() if parameter.requires_grad],
                lr=opt_cfg.LR,
                weight_decay=opt_cfg.get('WEIGHT_DECAY', 0),
                betas=(0.9, 0.95),
                eps=1e-5
            )
        else:
            assert False

        if opt_cfg.get('SCHEDULER', None) == 'cosine':

            utils.rank_zero_print("=====================================")
            if self.trainer.train_dataloader is not None:
                num_steps_per_epoch = len(self.trainer.train_dataloader)
            elif self.trainer.datamodule is not None and self.trainer.datamodule.train_dataset is not None:
                utils.rank_zero_print(
                    "Finding num_steps_per_epoch from datamodule...", len(self.trainer.datamodule.train_dataset),
                    self.trainer.datamodule.train_batch_size, self.trainer.world_size
                )
                num_steps_per_epoch = len(self.trainer.datamodule.train_dataset
                                          ) // (self.trainer.datamodule.train_batch_size * self.trainer.world_size)
            else:
                raise ValueError("Can't find num_steps_per_epoch")

            num_epochs = self.config.epochs
            total_steps = num_steps_per_epoch * num_epochs
            utils.rank_zero_print("Configuring cosine scheduler")
            utils.rank_zero_print("Num Steps per epoch: ", num_steps_per_epoch)
            utils.rank_zero_print("Num Epochs: ", num_epochs)
            utils.rank_zero_print("Total Steps: ", total_steps)
            utils.rank_zero_print("=====================================")

            scheduler = lr_schedule.get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=opt_cfg.WARMUP_STEPS,
                num_training_steps=total_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                },
            }

        elif opt_cfg.get('SCHEDULER', None) == 'lambdaLR':
            raise ValueError()
            # def lr_lbmd(cur_epoch):
            #     cur_decay = 1
            #     for decay_step in opt_cfg.get('DECAY_STEP_LIST', [5, 10, 15, 20]):
            #         if cur_epoch >= decay_step:
            #             cur_decay = cur_decay * opt_cfg.LR_DECAY
            #     return max(cur_decay, opt_cfg.LR_CLIP / opt_cfg.LR)
            #
            # scheduler = LambdaLR(optimizer, lr_lbmd)

        elif opt_cfg.get('SCHEDULER', None) == 'linear':
            raise ValueError()
            scheduler = lr_schedule.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=opt_cfg.WARMUP_STEPS,
                num_training_steps=opt_cfg.TRAINING_STEPS,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                },
            }

        elif opt_cfg.get('SCHEDULER', None) == 'inverse_sqrt':
            scheduler = lr_schedule.get_inverse_sqrt_schedule(
                optimizer,
                num_warmup_steps=opt_cfg.WARMUP_STEPS,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                },
            }

        else:
            raise ValueError()
