"""Adv-BMT tokenizer parity module (vendored-lite, NumPy only).

This module ports the core behavior of the legacy Adv-BMT bicycle tokenizer
into CounterBMT v2 without importing the legacy ``bmt`` package at runtime.

Key parity behaviors:
- Forward and backward tokenization over a shared 33x33 (acc, yaw-rate) grid.
- Hole-filling for rare valid-mask gaps (True -> False -> True).
- GPT-style start/end token semantics for agents appearing mid-sequence.
- Optional ``ALLOW_SKIP_STEP`` state overwrite for newly added agents.

Design constraints:
- No TensorFlow/Waymo dependencies.
- No runtime imports from ``src/Adv-BMT``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class ParityTokenizerConfig:
    """Configuration for Adv-BMT parity tokenization."""

    num_bins: int = 33
    acc_min: float = -10.0
    acc_max: float = 10.0
    yaw_min: float = -float(np.pi / 2.0)
    yaw_max: float = float(np.pi / 2.0)
    num_skipped_steps: int = 5
    gpt_style: bool = True
    allow_skip_step: bool = True
    delta_pos_is_velocity: bool = True
    steps_per_second: int = 10

    @property
    def n_tokens(self) -> int:
        return int(self.num_bins * self.num_bins)

    @property
    def dt_s(self) -> float:
        return float(self.num_skipped_steps) / float(self.steps_per_second)


@dataclass
class ParityTokenBatch:
    """Tokenized training batch produced by parity tokenizer."""

    prev_token_ids: np.ndarray  # [B,S,N] model token IDs
    targets: np.ndarray  # [B,S,N] action IDs in [0, n_tokens-1]
    target_mask: np.ndarray  # [B,S,N] float32
    continuous_motion: np.ndarray  # [B,S,N,2]
    input_mask: np.ndarray  # [B,S,N] bool (legacy decoder/input_action_valid_mask)
    modeled_agent_delta: np.ndarray  # [B,S,N,2] local-frame velocity delta
    sample_steps: np.ndarray  # [Ts]


class AdvBMTParityTokenizer:
    """NumPy implementation of Adv-BMT tokenizer parity behavior."""

    START_ACTION = 1_000_000
    END_ACTION = 7_777_777
    INVALID_ACTION = -1

    def __init__(self, cfg: ParityTokenizerConfig | None = None):
        self.parity_cfg = cfg or ParityTokenizerConfig()
        self.num_bins = int(self.parity_cfg.num_bins)
        self.num_actions = int(self.parity_cfg.n_tokens)
        self.dt = float(self.parity_cfg.dt_s)

        self.acceleration_bins = np.linspace(
            self.parity_cfg.acc_min,
            self.parity_cfg.acc_max,
            self.num_bins,
            dtype=np.float32,
        )
        self.steering_bins = np.linspace(
            self.parity_cfg.yaw_min,
            self.parity_cfg.yaw_max,
            self.num_bins,
            dtype=np.float32,
        )

        a_grid, yaw_grid = np.meshgrid(self.acceleration_bins, self.steering_bins, indexing="ij")
        self.a_grid_flat = a_grid.reshape(-1).astype(np.float32)
        self.delta_grid_flat = yaw_grid.reshape(-1).astype(np.float32)
        self._action_table = np.stack([self.a_grid_flat, self.delta_grid_flat], axis=-1).astype(np.float32)

        # Match legacy tie-breaker noise that prefers center bins slightly.
        y, x = np.ogrid[-(self.num_bins // 2):(self.num_bins + 1) // 2, -(self.num_bins // 2):(self.num_bins + 1) // 2]
        dist = np.sqrt(x**2 + y**2).astype(np.float32)
        max_dist = float(np.maximum(dist.max(), 1e-6))
        self._noise = (((dist / max_dist) - 1.0) * 1e-5).reshape(1, self.num_actions, 1).astype(np.float32)

        self.default_token_id = int(self.num_actions // 2)

        # Model-facing special IDs.
        self.START_MODEL_ID = int(self.num_actions + 0)
        self.END_MODEL_ID = int(self.num_actions + 1)
        self.PAD_MODEL_ID = int(self.num_actions + 2)
        self.MASK_MODEL_ID = int(self.num_actions + 3)  # reserved

        # Keep compatibility with call sites that read tokenizer.cfg.n_tokens.
        self.cfg = SimpleNamespace(
            n_tokens=int(self.num_actions),
            n_acc_bins=int(self.num_bins),
            n_yaw_bins=int(self.num_bins),
        )

    def action_table_np(self) -> np.ndarray:
        return self._action_table.copy()

    def build_legacy_like_inputs(self, batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Builds a legacy-style dictionary for parity debugging/export."""
        sample_steps = np.arange(
            0,
            int(np.asarray(batch["agent_position_xy"]).shape[1]),
            int(self.parity_cfg.num_skipped_steps),
            dtype=np.int32,
        )
        pos_xy = np.asarray(batch["agent_position_xy"], dtype=np.float32)
        vel_xy = np.asarray(batch["agent_velocity_xy"], dtype=np.float32)
        heading = np.asarray(batch["agent_heading"], dtype=np.float32)
        valid = np.asarray(batch["agent_valid_mask"], dtype=bool)
        shape = np.asarray(batch["agent_shape"], dtype=np.float32)
        typ = np.asarray(batch["agent_type_ids"], dtype=np.int32)

        pos_xyz = np.concatenate(
            [pos_xy[:, :, :, :2], np.zeros(pos_xy.shape[:3] + (1,), dtype=np.float32)],
            axis=-1,
        )
        vel = np.concatenate(
            [vel_xy[:, :, :, :2], np.zeros(vel_xy.shape[:3] + (1,), dtype=np.float32)],
            axis=-1,
        )

        return {
            "sample_steps": sample_steps,
            "decoder/agent_position": pos_xyz,
            "decoder/agent_heading": heading,
            "decoder/agent_valid_mask": valid,
            "decoder/agent_velocity": vel[:, :, :, :2],
            "decoder/current_agent_shape": shape,
            "decoder/agent_type": typ,
        }

    def tokenize_batch(
        self,
        batch: Dict[str, Any],
        *,
        backward_prediction: bool,
    ) -> ParityTokenBatch:
        """Tokenize a collated v2 batch in forward or backward direction."""
        (
            pos_macro,
            heading_macro,
            vel_macro,
            valid_macro,
            agent_shape,
            sample_steps,
        ) = self._prepare_macro_tensors(batch)

        if backward_prediction:
            input_action, input_mask, target_action, target_mask, modeled_agent_delta = self._tokenize_backward_macro(
                pos_macro,
                heading_macro,
                vel_macro,
                valid_macro,
                agent_shape,
            )
        else:
            input_action, input_mask, target_action, target_mask, modeled_agent_delta = self._tokenize_forward_macro(
                pos_macro,
                heading_macro,
                vel_macro,
                valid_macro,
                agent_shape,
            )

        prev_token_ids = self._map_input_actions_to_model_ids(input_action)
        targets, target_mask_f = self._targets_and_mask_from_actions(target_action, target_mask)
        continuous_motion = self._continuous_motion_from_prev(prev_token_ids)

        return ParityTokenBatch(
            prev_token_ids=prev_token_ids,
            targets=targets,
            target_mask=target_mask_f,
            continuous_motion=continuous_motion,
            input_mask=input_mask.astype(bool),
            modeled_agent_delta=modeled_agent_delta.astype(np.float32),
            sample_steps=sample_steps,
        )

    def _prepare_macro_tensors(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos_xy = np.asarray(batch["agent_position_xy"], dtype=np.float32)
        heading = np.asarray(batch["agent_heading"], dtype=np.float32)
        vel_xy = np.asarray(batch["agent_velocity_xy"], dtype=np.float32)
        valid = np.asarray(batch["agent_valid_mask"], dtype=bool)
        agent_shape = np.asarray(batch["agent_shape"], dtype=np.float32)

        sample_steps = np.arange(0, int(pos_xy.shape[1]), int(self.parity_cfg.num_skipped_steps), dtype=np.int32)
        if sample_steps.shape[0] < 2:
            raise ValueError(
                "Not enough sampled steps for parity tokenization: "
                f"T={pos_xy.shape[1]}, skip={self.parity_cfg.num_skipped_steps}, sampled={sample_steps.shape[0]}"
            )

        pos_macro = pos_xy[:, sample_steps, :, :2].copy()
        heading_macro = heading[:, sample_steps, :].copy()
        vel_macro = vel_xy[:, sample_steps, :, :2].copy()
        valid_macro = valid[:, sample_steps, :].copy()

        self._hole_fill_macro(pos_macro, heading_macro, vel_macro, valid_macro)
        return pos_macro, heading_macro, vel_macro, valid_macro, agent_shape, sample_steps

    def _tokenize_forward_macro(
        self,
        pos_macro: np.ndarray,  # [B,T,N,2]
        heading_macro: np.ndarray,  # [B,T,N]
        vel_macro: np.ndarray,  # [B,T,N,2]
        valid_macro: np.ndarray,  # [B,T,N]
        agent_shape: np.ndarray,  # [B,N,3]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        bsz, t_chunks, n_agents, _ = pos_macro.shape

        current_pos = pos_macro[:, 0:1].copy()
        current_heading = heading_macro[:, 0:1].copy()
        current_vel = vel_macro[:, 0:1].copy()
        current_valid = valid_macro[:, 0:1].copy()
        init_valid = current_valid.copy()
        init_delta = self._get_relative_velocity(current_vel, current_heading).astype(np.float32)
        init_delta = np.where(init_valid[..., None], init_delta, 0.0).astype(np.float32)

        target_actions = []
        target_masks = []
        delta_list = []

        for next_step in range(1, t_chunks):
            res = self._tokenize_step_forward(
                current_pos=current_pos,
                current_heading=current_heading,
                current_valid=current_valid,
                current_vel=current_vel,
                next_pos=pos_macro[:, next_step:next_step + 1],
                next_heading=heading_macro[:, next_step:next_step + 1],
                next_valid=valid_macro[:, next_step:next_step + 1],
                next_vel=vel_macro[:, next_step:next_step + 1],
                agent_shape=agent_shape,
            )

            best_action = res["action"].copy()
            recon_pos = res["pos"].copy()
            recon_heading = res["heading"].copy()
            recon_vel = res["vel"].copy()
            recon_valid = res["mask"].copy()
            recon_delta = res["delta_pos"].copy()
            # Legacy parity: target valid-mask is recorded before ALLOW_SKIP_STEP
            # overwrite for newly-added/reappearing agents.
            target_valid_step = recon_valid.copy()

            if self.parity_cfg.allow_skip_step:
                next_valid = valid_macro[:, next_step:next_step + 1]
                newly_added = np.logical_and(~recon_valid, next_valid)
                if np.any(newly_added):
                    gt_pos = pos_macro[:, next_step:next_step + 1]
                    gt_heading = heading_macro[:, next_step:next_step + 1]
                    gt_vel = vel_macro[:, next_step:next_step + 1]
                    recon_pos[newly_added] = gt_pos[newly_added]
                    recon_heading[newly_added] = gt_heading[newly_added]
                    recon_vel[newly_added] = gt_vel[newly_added]
                    recon_delta[newly_added] = self._get_relative_velocity(gt_vel, gt_heading)[newly_added]
                    recon_valid[newly_added] = next_valid[newly_added]

            target_actions.append(best_action)
            target_masks.append(target_valid_step)
            delta_list.append(recon_delta)

            current_pos = recon_pos
            current_heading = recon_heading
            current_vel = recon_vel
            current_valid = recon_valid

        if target_actions:
            target_action = np.concatenate(target_actions, axis=1).astype(np.int32)
            target_mask = np.concatenate(target_masks, axis=1).astype(bool)
            modeled_delta = np.concatenate([init_delta] + delta_list, axis=1).astype(np.float32)
        else:
            target_action = np.zeros((bsz, 0, n_agents), dtype=np.int32)
            target_mask = np.zeros((bsz, 0, n_agents), dtype=bool)
            modeled_delta = init_delta.astype(np.float32)

        start_action = np.full((bsz, 1, n_agents), self.INVALID_ACTION, dtype=np.int32)
        start_action[init_valid] = self.START_ACTION

        input_action = np.concatenate([start_action, target_action], axis=1)
        input_mask = np.concatenate([init_valid, target_mask], axis=1)

        already_tokenized = init_valid.copy()
        for next_step in range(1, t_chunks):
            next_valid = valid_macro[:, next_step:next_step + 1]
            is_new = np.logical_and(~already_tokenized, next_valid)
            if np.any(is_new):
                input_action[:, next_step:next_step + 1][is_new] = self.START_ACTION
                input_mask[:, next_step:next_step + 1][is_new] = next_valid[is_new]
            already_tokenized = np.logical_or(already_tokenized, is_new)

        target_action = np.concatenate(
            [target_action, np.full((bsz, 1, n_agents), self.INVALID_ACTION, dtype=np.int32)],
            axis=1,
        )
        target_mask = np.concatenate([target_mask, np.zeros((bsz, 1, n_agents), dtype=bool)], axis=1)
        return input_action, input_mask, target_action, target_mask, modeled_delta

    def _tokenize_backward_macro(
        self,
        pos_macro: np.ndarray,  # [B,T,N,2]
        heading_macro: np.ndarray,  # [B,T,N]
        vel_macro: np.ndarray,  # [B,T,N,2]
        valid_macro: np.ndarray,  # [B,T,N]
        agent_shape: np.ndarray,  # [B,N,3]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        bsz, t_chunks, n_agents, _ = pos_macro.shape

        current_pos = pos_macro[:, -1:].copy()
        current_heading = heading_macro[:, -1:].copy()
        current_vel = vel_macro[:, -1:].copy()
        current_valid = valid_macro[:, -1:].copy()
        init_valid = current_valid.copy()
        init_delta = self._get_relative_velocity(current_vel, current_heading).astype(np.float32)
        init_delta = np.where(init_valid[..., None], init_delta, 0.0).astype(np.float32)

        target_actions = []
        target_masks = []
        delta_list = []

        for backward_next_step in range(1, t_chunks):
            forward_next_step = t_chunks - backward_next_step - 1
            res = self._tokenize_step_backward(
                future_pos=current_pos,
                future_heading=current_heading,
                future_valid=current_valid,
                future_vel=current_vel,
                past_pos=pos_macro[:, forward_next_step:forward_next_step + 1],
                past_heading=heading_macro[:, forward_next_step:forward_next_step + 1],
                past_valid=valid_macro[:, forward_next_step:forward_next_step + 1],
                past_vel=vel_macro[:, forward_next_step:forward_next_step + 1],
                agent_shape=agent_shape,
            )

            best_action = res["action"].copy()
            recon_pos = res["pos"].copy()
            recon_heading = res["heading"].copy()
            recon_vel = res["vel"].copy()
            recon_valid = res["mask"].copy()
            recon_delta = res["delta_pos"].copy()
            # Legacy parity: target valid-mask is recorded before ALLOW_SKIP_STEP
            # overwrite for newly-added/reappearing agents.
            target_valid_step = recon_valid.copy()

            if self.parity_cfg.allow_skip_step:
                next_valid = valid_macro[:, forward_next_step:forward_next_step + 1]
                newly_added = np.logical_and(~recon_valid, next_valid)
                if np.any(newly_added):
                    gt_pos = pos_macro[:, forward_next_step:forward_next_step + 1]
                    gt_heading = heading_macro[:, forward_next_step:forward_next_step + 1]
                    gt_vel = vel_macro[:, forward_next_step:forward_next_step + 1]
                    recon_pos[newly_added] = gt_pos[newly_added]
                    recon_heading[newly_added] = gt_heading[newly_added]
                    recon_vel[newly_added] = gt_vel[newly_added]
                    recon_delta[newly_added] = self._get_relative_velocity(gt_vel, gt_heading)[newly_added]
                    recon_valid[newly_added] = next_valid[newly_added]

            target_actions.append(best_action)
            target_masks.append(target_valid_step)
            delta_list.append(recon_delta)

            current_pos = recon_pos
            current_heading = recon_heading
            current_vel = recon_vel
            current_valid = recon_valid

        if target_actions:
            target_action = np.concatenate(target_actions, axis=1).astype(np.int32)
            target_mask = np.concatenate(target_masks, axis=1).astype(bool)
            modeled_delta = np.concatenate([init_delta] + delta_list, axis=1).astype(np.float32)
        else:
            target_action = np.zeros((bsz, 0, n_agents), dtype=np.int32)
            target_mask = np.zeros((bsz, 0, n_agents), dtype=bool)
            modeled_delta = init_delta.astype(np.float32)

        start_action = np.full((bsz, 1, n_agents), self.INVALID_ACTION, dtype=np.int32)
        start_action[init_valid] = self.END_ACTION

        input_action = np.concatenate([start_action, target_action], axis=1)
        input_mask = np.concatenate([init_valid, target_mask], axis=1)

        already_tokenized = init_valid.copy()
        for backward_next_step in range(1, t_chunks):
            forward_next_step = t_chunks - backward_next_step - 1
            next_valid = valid_macro[:, forward_next_step:forward_next_step + 1]
            is_new = np.logical_and(~already_tokenized, next_valid)
            if np.any(is_new):
                input_action[:, backward_next_step:backward_next_step + 1][is_new] = self.END_ACTION
                input_mask[:, backward_next_step:backward_next_step + 1][is_new] = next_valid[is_new]
            already_tokenized = np.logical_or(already_tokenized, is_new)

        target_action = np.concatenate(
            [target_action, np.full((bsz, 1, n_agents), self.INVALID_ACTION, dtype=np.int32)],
            axis=1,
        )
        target_mask = np.concatenate([target_mask, np.zeros((bsz, 1, n_agents), dtype=bool)], axis=1)
        return input_action, input_mask, target_action, target_mask, modeled_delta

    def _tokenize_step_forward(
        self,
        *,
        current_pos: np.ndarray,  # [B,1,N,2]
        current_heading: np.ndarray,  # [B,1,N]
        current_valid: np.ndarray,  # [B,1,N]
        current_vel: np.ndarray,  # [B,1,N,2]
        next_pos: np.ndarray,  # [B,1,N,2]
        next_heading: np.ndarray,  # [B,1,N]
        next_valid: np.ndarray,  # [B,1,N]
        next_vel: np.ndarray,  # [B,1,N,2]
        agent_shape: np.ndarray,  # [B,N,3]
    ) -> Dict[str, np.ndarray]:
        del next_vel  # kept for parity signature consistency
        bsz, _, n_agents, _ = current_pos.shape
        a = self.num_actions

        a_grid = self.a_grid_flat.reshape(1, a, 1)
        yaw_grid = self.delta_grid_flat.reshape(1, a, 1)

        current_pos_exp = np.broadcast_to(current_pos, (bsz, a, n_agents, 2))
        current_heading_exp = np.broadcast_to(current_heading, (bsz, a, n_agents))

        current_speed = np.linalg.norm(current_vel, axis=-1)  # [B,1,N]
        current_speed = np.broadcast_to(current_speed, (bsz, a, n_agents))
        next_speed_candidate = current_speed + a_grid * self.dt
        average_speed = (current_speed + next_speed_candidate) * 0.5

        next_heading_candidate = self._wrap_to_pi(current_heading_exp + yaw_grid * self.dt)
        average_heading = self._average_heading(next_heading_candidate, current_heading_exp)

        next_velocity_candidate = self._rotate(next_speed_candidate, np.zeros_like(next_speed_candidate), next_heading_candidate)
        avg_vel = self._rotate(average_speed, np.zeros_like(average_speed), average_heading)
        next_pos_candidate = current_pos_exp + avg_vel * self.dt

        width = agent_shape[..., 1].reshape(bsz, 1, n_agents)
        length = agent_shape[..., 0].reshape(bsz, 1, n_agents)
        contour = self._cal_polygon_contour(
            next_pos_candidate[..., 0],
            next_pos_candidate[..., 1],
            next_heading_candidate,
            width,
            length,
        )
        gt_contour = self._cal_polygon_contour(
            next_pos[..., 0],
            next_pos[..., 1],
            next_heading,
            width,
            length,
        )

        error = np.linalg.norm(contour - gt_contour, axis=-1).mean(axis=-1)
        error = error + self._noise
        best_action = np.argmin(error, axis=1).astype(np.int32)  # [B,N]

        valid_mask = np.logical_and(current_valid[:, 0, :], next_valid[:, 0, :])
        best_action[~valid_mask] = self.INVALID_ACTION

        b_idx = np.arange(bsz, dtype=np.int32)[:, None]
        n_idx = np.arange(n_agents, dtype=np.int32)[None, :]
        gather_idx = np.where(best_action < 0, self.default_token_id, best_action).astype(np.int32)

        recon_pos = next_pos_candidate[b_idx, gather_idx, n_idx][:, None, :, :]
        recon_vel = next_velocity_candidate[b_idx, gather_idx, n_idx][:, None, :, :]
        recon_heading = next_heading_candidate[b_idx, gather_idx, n_idx][:, None, :]

        valid_expand = valid_mask[:, None, :, None]
        recon_pos = np.where(valid_expand, recon_pos, 0.0).astype(np.float32)
        recon_vel = np.where(valid_expand, recon_vel, 0.0).astype(np.float32)
        recon_heading = np.where(valid_mask[:, None, :], recon_heading, 0.0).astype(np.float32)

        rel_delta = self._get_relative_velocity(recon_vel, recon_heading)
        rel_delta = np.where(valid_expand, rel_delta, 0.0).astype(np.float32)

        return {
            "action": best_action[:, None, :].astype(np.int32),
            "pos": recon_pos,
            "heading": recon_heading,
            "vel": recon_vel,
            "mask": valid_mask[:, None, :],
            "delta_pos": rel_delta,
        }

    def _tokenize_step_backward(
        self,
        *,
        future_pos: np.ndarray,  # [B,1,N,2]
        future_heading: np.ndarray,  # [B,1,N]
        future_valid: np.ndarray,  # [B,1,N]
        future_vel: np.ndarray,  # [B,1,N,2]
        past_pos: np.ndarray,  # [B,1,N,2]
        past_heading: np.ndarray,  # [B,1,N]
        past_valid: np.ndarray,  # [B,1,N]
        past_vel: np.ndarray,  # [B,1,N,2]
        agent_shape: np.ndarray,  # [B,N,3]
    ) -> Dict[str, np.ndarray]:
        del past_vel  # kept for parity signature consistency
        bsz, _, n_agents, _ = future_pos.shape
        a = self.num_actions

        a_grid = self.a_grid_flat.reshape(1, a, 1)
        yaw_grid = self.delta_grid_flat.reshape(1, a, 1)

        future_pos_exp = np.broadcast_to(future_pos, (bsz, a, n_agents, 2))
        future_heading_exp = np.broadcast_to(future_heading, (bsz, a, n_agents))
        past_heading_exp = np.broadcast_to(past_heading, (bsz, a, n_agents))

        future_speed = np.linalg.norm(future_vel, axis=-1)
        future_speed = np.broadcast_to(future_speed, (bsz, a, n_agents))
        past_speed_candidate = future_speed - a_grid * self.dt
        average_speed = (past_speed_candidate + future_speed) * 0.5

        past_heading_candidate = self._wrap_to_pi(future_heading_exp - yaw_grid * self.dt)
        average_heading = self._average_heading(past_heading_candidate, future_heading_exp)

        average_vel = self._rotate(average_speed, np.zeros_like(average_speed), average_heading)
        past_vel_candidate = self._rotate(past_speed_candidate, np.zeros_like(past_speed_candidate), past_heading_candidate)
        past_pos_candidate = future_pos_exp - average_vel * self.dt

        width = agent_shape[..., 1].reshape(bsz, 1, n_agents)
        length = agent_shape[..., 0].reshape(bsz, 1, n_agents)
        contour = self._cal_polygon_contour(
            past_pos_candidate[..., 0],
            past_pos_candidate[..., 1],
            past_heading_candidate,
            width,
            length,
        )
        gt_contour = self._cal_polygon_contour(
            past_pos[..., 0],
            past_pos[..., 1],
            past_heading_exp,
            width,
            length,
        )

        error = np.linalg.norm(contour - gt_contour, axis=-1).mean(axis=-1)
        error = error + self._noise
        best_action = np.argmin(error, axis=1).astype(np.int32)

        valid_mask = np.logical_and(future_valid[:, 0, :], past_valid[:, 0, :])
        best_action[~valid_mask] = self.INVALID_ACTION

        b_idx = np.arange(bsz, dtype=np.int32)[:, None]
        n_idx = np.arange(n_agents, dtype=np.int32)[None, :]
        gather_idx = np.where(best_action < 0, self.default_token_id, best_action).astype(np.int32)

        recon_pos = past_pos_candidate[b_idx, gather_idx, n_idx][:, None, :, :]
        recon_vel = past_vel_candidate[b_idx, gather_idx, n_idx][:, None, :, :]
        recon_heading = past_heading_candidate[b_idx, gather_idx, n_idx][:, None, :]

        valid_expand = valid_mask[:, None, :, None]
        recon_pos = np.where(valid_expand, recon_pos, 0.0).astype(np.float32)
        recon_vel = np.where(valid_expand, recon_vel, 0.0).astype(np.float32)
        recon_heading = np.where(valid_mask[:, None, :], recon_heading, 0.0).astype(np.float32)

        rel_delta = self._get_relative_velocity(recon_vel, recon_heading)
        rel_delta = np.where(valid_expand, rel_delta, 0.0).astype(np.float32)

        return {
            "action": best_action[:, None, :].astype(np.int32),
            "pos": recon_pos,
            "heading": recon_heading,
            "vel": recon_vel,
            "mask": valid_mask[:, None, :],
            "delta_pos": rel_delta,
        }

    def _hole_fill_macro(
        self,
        pos_macro: np.ndarray,
        heading_macro: np.ndarray,
        vel_macro: np.ndarray,
        valid_macro: np.ndarray,
    ) -> None:
        """In-place parity hole filling for rare True/False/True validity gaps."""
        t_chunks = int(pos_macro.shape[1])
        for i in range(1, t_chunks - 1):
            step0 = valid_macro[:, i - 1:i]
            step1 = valid_macro[:, i:i + 1]
            step2 = valid_macro[:, i + 1:i + 2]
            is_rare = np.logical_and(np.logical_and(step2, step0), ~step1)
            if not np.any(is_rare):
                continue

            int_pos = (pos_macro[:, i - 1:i] + pos_macro[:, i + 1:i + 2]) * 0.5
            int_vel = (vel_macro[:, i - 1:i] + vel_macro[:, i + 1:i + 2]) * 0.5

            head_s = heading_macro[:, i - 1:i]
            head_e = heading_macro[:, i + 1:i + 2]
            int_heading = np.arctan2(np.sin(head_s) + np.sin(head_e), np.cos(head_s) + np.cos(head_e))
            int_heading = self._wrap_to_pi(int_heading)

            pos_macro[:, i:i + 1] = np.where(is_rare[..., None], int_pos, pos_macro[:, i:i + 1])
            heading_macro[:, i:i + 1] = np.where(is_rare, int_heading, heading_macro[:, i:i + 1])
            vel_macro[:, i:i + 1] = np.where(is_rare[..., None], int_vel, vel_macro[:, i:i + 1])
            valid_macro[:, i:i + 1] = np.logical_or(valid_macro[:, i:i + 1], is_rare)

    def _map_input_actions_to_model_ids(self, input_action: np.ndarray) -> np.ndarray:
        out = np.full(input_action.shape, self.PAD_MODEL_ID, dtype=np.int32)
        is_action = input_action >= 0
        out[is_action] = input_action[is_action].astype(np.int32)
        out[input_action == self.START_ACTION] = self.START_MODEL_ID
        out[input_action == self.END_ACTION] = self.END_MODEL_ID
        out[input_action == self.INVALID_ACTION] = self.PAD_MODEL_ID
        return out

    def _targets_and_mask_from_actions(
        self,
        target_action: np.ndarray,
        target_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        targets = np.full(target_action.shape, self.default_token_id, dtype=np.int32)
        real = target_action >= 0
        targets[real] = target_action[real].astype(np.int32)
        return targets, target_mask.astype(np.float32)

    def _continuous_motion_from_prev(self, prev_token_ids: np.ndarray) -> np.ndarray:
        out = np.zeros(prev_token_ids.shape + (2,), dtype=np.float32)
        is_action = prev_token_ids < self.num_actions
        if np.any(is_action):
            out[is_action] = self._action_table[prev_token_ids[is_action]]
        return out

    @staticmethod
    def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
        # Match legacy wrap_to_pi behavior used by Adv-BMT utils:
        #   wrapped = mod(x, 2*pi); wrapped[wrapped > pi] -= 2*pi
        wrapped = np.mod(x, 2.0 * np.pi)
        wrapped = np.where(wrapped > np.pi, wrapped - (2.0 * np.pi), wrapped)
        return wrapped.astype(np.float32)

    @staticmethod
    def _average_heading(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
        return np.arctan2(np.sin(h1) + np.sin(h2), np.cos(h1) + np.cos(h2))

    @staticmethod
    def _rotate(x: np.ndarray, y: np.ndarray, angle: np.ndarray) -> np.ndarray:
        out_x = np.cos(angle) * x - np.sin(angle) * y
        out_y = np.cos(angle) * y + np.sin(angle) * x
        return np.stack([out_x, out_y], axis=-1).astype(np.float32)

    @staticmethod
    def _get_relative_velocity(vel: np.ndarray, heading: np.ndarray) -> np.ndarray:
        x = vel[..., 0]
        y = vel[..., 1]
        out_x = np.cos(-heading) * x - np.sin(-heading) * y
        out_y = np.cos(-heading) * y + np.sin(-heading) * x
        return np.stack([out_x, out_y], axis=-1).astype(np.float32)

    @staticmethod
    def _cal_polygon_contour(
        x: np.ndarray,  # [...,]
        y: np.ndarray,  # [...,]
        theta: np.ndarray,  # [...,]
        width: np.ndarray,  # [...,]
        length: np.ndarray,  # [...,]
    ) -> np.ndarray:
        lf_x = x + 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
        lf_y = y + 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)

        rf_x = x + 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
        rf_y = y + 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)

        rb_x = x - 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
        rb_y = y - 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)

        lb_x = x - 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
        lb_y = y - 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)

        return np.stack(
            [
                np.stack([lf_x, lf_y], axis=-1),
                np.stack([rf_x, rf_y], axis=-1),
                np.stack([rb_x, rb_y], axis=-1),
                np.stack([lb_x, lb_y], axis=-1),
            ],
            axis=-2,
        ).astype(np.float32)
