import copy
import logging
import math
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from bmt.counterfactual.compile_control_code import (
    BRANCH_LABEL_ORDER,
    COMPLIANCE_LABEL_ORDER,
    TERMINAL_ANCHOR_DIM,
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
)
from bmt.models.motionlm import MotionLM
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


def sanitize_logits_for_loss(logits, *, clamp=50.0):
    logits = torch.nan_to_num(logits, nan=0.0, posinf=float(clamp), neginf=-float(clamp))
    return logits.clamp(min=-float(clamp), max=float(clamp))


def sanitize_scalar_loss(value, *, fallback=0.0):
    if torch.is_tensor(value):
        return torch.nan_to_num(value, nan=float(fallback), posinf=float(fallback), neginf=float(fallback))
    return value


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
            if key in {"decoder/output_logit", "decoder/decoded_tokens"}:
                continue
            teacher_input[key] = copy.deepcopy(value)
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
            if sdc_semantic_context is not None:
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
                semantic_loss_weight = float(
                    self.config.MODEL.get(
                        "LOCAL_CONTROL_SDC_SEMANTIC_AUX_LOSS_WEIGHT",
                        self.config.MODEL.get("LOCAL_CONTROL_PATH_LOSS_WEIGHT", 0.2),
                    )
                )
                guide_loss_weight = float(
                    self.config.MODEL.get("LOCAL_CONTROL_SDC_FAMILY_GUIDE_LOSS_WEIGHT", 0.2)
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
                if guide_loss_weight > 0.0 and sdc_semantic_context is not None:
                    loss = loss + guide_loss_weight * sanitize_scalar_loss(guide_bundle["guide_loss"])
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
                        "cf/sdc_family_distance_mean": guide_bundle["projected_family_distance"][
                            sdc_semantic_context["sdc_valid_by_t"]
                        ].mean() if sdc_semantic_context is not None and bool(sdc_semantic_context["sdc_valid_by_t"].any()) else control_hidden.new_tensor(0.0),
                    }
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

    def training_step(self, data_dict, batch_idx):

        # For profiling GPU usage.
        # torch.cuda.empty_cache()

        if "in_evaluation" in data_dict:
            in_evaluation = data_dict["in_evaluation"]
            if not torch.is_tensor(in_evaluation):
                in_evaluation = torch.as_tensor(in_evaluation)
            data_dict["in_evaluation"] = torch.zeros_like(in_evaluation, dtype=torch.bool)

        data_dict = self(data_dict)

        loss, loss_stat = self.get_loss(data_dict)

        pbar_keys = ("total_loss", "toks", "lr")

        motion_stat = {k: v for k, v in loss_stat.items() if k.startswith("motion_stat")}
        loss_stat = {k: v for k, v in loss_stat.items() if not k.startswith("motion_stat")}

        self.log_dict(
            {f"{k}": float(v)
             for k, v in loss_stat.items() if k in pbar_keys},
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
             for k, v in loss_stat.items()},
            batch_size=data_dict["encoder/map_feature"].shape[0],
            # on_epoch=True,
            prog_bar=False,
        )
        self.log('monitoring_step', float(self.global_step))
        return loss

    def optimizer_step(self, *args, **kwargs):
        grad_clip_norm = float(self.config.MODEL.get("LOCAL_CONTROL_GRAD_CLIP_NORM", 1.0))
        if grad_clip_norm > 0.0:
            parameters = [p for p in self.parameters() if p.requires_grad and p.grad is not None]
            if parameters:
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=grad_clip_norm)
        super().optimizer_step(*args, **kwargs)

    def on_validation_start(self):
        torch.cuda.empty_cache()

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
        motion_stat = {k: v for k, v in loss_stat.items() if k.startswith("motion_stat")}
        loss_stat = {k: v for k, v in loss_stat.items() if not k.startswith("motion_stat")}

        self.log_dict(
            {f"val/{k}": float(v) for k, v in loss_stat.items()},
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
