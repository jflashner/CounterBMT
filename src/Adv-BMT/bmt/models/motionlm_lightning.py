import logging

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
from bmt.models.motionlm import MotionLM
from bmt.tokenization import get_tokenizer
from bmt.utils import lr_schedule
from bmt.utils import utils

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
        self._configure_local_control_finetune()
        # self.validation_outputs = []
        # self.validation_ground_truth = []

        self.exp_name = None

    def _configure_local_control_finetune(self):
        if not self.local_control_forward_enabled:
            return
        if not bool(self.config.MODEL.get("LOCAL_CONTROL_FREEZE_BACKBONE", False)):
            return

        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for parameter in self.path_head.parameters():
            parameter.requires_grad = True
        for parameter in self.anchor_head.parameters():
            parameter.requires_grad = True

        if bool(self.config.MODEL.get("LOCAL_CONTROL_USE_COMPLIANCE", True)):
            for parameter in self.compliance_head.parameters():
                parameter.requires_grad = True
        if bool(self.config.MODEL.get("LOCAL_CONTROL_USE_TIMING", True)):
            for parameter in self.timing_head.parameters():
                parameter.requires_grad = True

        motion_decoder = self.model.motion_decoder
        if hasattr(motion_decoder, "cf_path_proj") and bool(self.config.MODEL.get("LOCAL_CONTROL_USE_PATH", True)):
            for parameter in motion_decoder.cf_path_proj.parameters():
                parameter.requires_grad = True
        if hasattr(motion_decoder, "cf_anchor_proj") and bool(self.config.MODEL.get("LOCAL_CONTROL_USE_ANCHOR", True)):
            for parameter in motion_decoder.cf_anchor_proj.parameters():
                parameter.requires_grad = True
        if hasattr(motion_decoder, "cf_compliance_proj") and bool(self.config.MODEL.get("LOCAL_CONTROL_USE_COMPLIANCE", True)):
            for parameter in motion_decoder.cf_compliance_proj.parameters():
                parameter.requires_grad = True
        if hasattr(motion_decoder, "cf_timing_proj") and bool(self.config.MODEL.get("LOCAL_CONTROL_USE_TIMING", True)):
            for parameter in motion_decoder.cf_timing_proj.parameters():
                parameter.requires_grad = True
        if hasattr(motion_decoder, "cf_local_bias"):
            for parameter in motion_decoder.cf_local_bias.parameters():
                parameter.requires_grad = True
        if hasattr(motion_decoder, "cf_local_residual_gate"):
            motion_decoder.cf_local_residual_gate.requires_grad = True

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

            output_logit = output_logit[target_action_valid_mask]
            target_action = target_action[target_action_valid_mask]

            # Get loss
            if self.config.OPTIMIZATION.USE_FOCAL_LOSS:
                from torchvision.ops import sigmoid_focal_loss
                # Compute Focal Loss
                alpha = 0.25
                gamma = 2
                target_onehot = F.one_hot(target_action, output_logit.shape[-1]).float()
                loss = sigmoid_focal_loss(
                    inputs=output_logit, targets=target_onehot, alpha=alpha, gamma=gamma, reduction="none"
                )
            else:
                loss = torch.nn.functional.cross_entropy(input=output_logit, target=target_action, reduction="none")

            original_loss = loss
            loss = loss.mean()

            assert not np.isnan(loss.item())
            assert not np.isinf(loss.item())

            with torch.no_grad():
                encodings = F.one_hot(output_logit.argmax(-1),
                                      output_logit.shape[-1]).float().reshape(-1, output_logit.shape[-1])
                avg_probs = encodings.mean(0)
                perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
                cluster_use = torch.sum(avg_probs > 0)

                gt_onehot = F.one_hot(target_action, output_logit.shape[-1]).float()
                gt_encodings = gt_onehot.reshape(-1, output_logit.shape[-1])
                gt_avg_probs = gt_encodings.mean(0)
                gt_perplexity = (-(gt_avg_probs * torch.log(gt_avg_probs + 1e-10)).sum()).exp()
                gt_cluster_use = torch.sum(gt_avg_probs > 0)
                debug_gt_c_use = (gt_encodings.sum(0) > 0).sum()  # .mean()

                pred_act = output_logit.argmax(-1)
                acc = torch.sum(pred_act == target_action) / target_action.shape[0]
                entropy = safe_entropy(output_logit)
                pred_act = pred_act.float()

                rate_default_pred = (pred_act == self._tokenizer.default_action).float().mean()
                rate_default_gt = (target_action == self._tokenizer.default_action).float().mean()

                num_trained_tokens = len(target_action)
                num_trained_tokens_sum = self.trainer.world_size * num_trained_tokens

                loss_stat.update(
                    {
                        "original_loss": original_loss.mean(),
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
                    }
                )

                if self.config.BACKWARD_PREDICTION:
                    in_back_mask = data_dict["in_backward_prediction"]
                    in_back_mask = in_back_mask.reshape(-1, 1, 1).expand(*target_action_valid_mask.shape)
                    in_back_mask = in_back_mask[target_action_valid_mask]
                    acc2 = (pred_act == target_action)
                    acc_in_back = (acc2 & in_back_mask).sum() / in_back_mask.sum()
                    acc_in_forward = (acc2 & ~in_back_mask).sum() / (~in_back_mask).sum()
                    loss_in_back = original_loss[in_back_mask].mean()
                    loss_in_forward = original_loss[~in_back_mask].mean()
                    entropy_in_back = safe_entropy(output_logit[in_back_mask]).mean()
                    entropy_in_forward = safe_entropy(output_logit[~in_back_mask]).mean()
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
        super().optimizer_step(*args, **kwargs)

    def on_validation_start(self):
        torch.cuda.empty_cache()

    def validation_step(self, data_dict, batch_idx):

        if self.config.EVAL_MOTION:

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

    def on_validation_epoch_end(self):
        """
        This function gathers intermediate evaluation result and pass them to the Waymo
        evaluation pipeline together and log the final results.
        """
        if self.config.EVAL_MOTION:
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
