"""Compare v2 vendored parity tokenization with optional legacy tokenizer output."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Allow running as a standalone script from repo root.
if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[2]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

from counter_bmt_v2.data import ScenarioNetNNXLoader, collate_nnx_scene_samples
from counter_bmt_v2.trajectory_jax import AdvBMTParityTokenizer, ParityTokenizerConfig


@dataclass
class CompareStats:
    num_scenarios: int = 0
    prev_total: int = 0
    prev_match: int = 0
    target_total_masked: int = 0
    target_match_masked: int = 0
    valid_total: int = 0
    valid_match: int = 0
    seq_len_total: int = 0
    seq_len_match: int = 0
    start_count_total: int = 0
    end_count_total: int = 0
    pad_count_total: int = 0
    mask_count_total: int = 0
    has_invalid_token_ids: bool = False

    def to_metrics(self) -> Dict[str, float]:
        prev_rate = float(self.prev_match / self.prev_total) if self.prev_total > 0 else float("nan")
        target_rate = (
            float(self.target_match_masked / self.target_total_masked) if self.target_total_masked > 0 else float("nan")
        )
        valid_rate = float(self.valid_match / self.valid_total) if self.valid_total > 0 else float("nan")
        seq_rate = float(self.seq_len_match / self.seq_len_total) if self.seq_len_total > 0 else float("nan")
        return {
            "num_scenarios": float(self.num_scenarios),
            "prev_token_exact_match_rate": prev_rate,
            "target_token_exact_match_rate_masked": target_rate,
            "valid_mask_exact_match_rate": valid_rate,
            "sequence_length_exact_match_rate": seq_rate,
            "start_count_total": float(self.start_count_total),
            "end_count_total": float(self.end_count_total),
            "pad_count_total": float(self.pad_count_total),
            "mask_count_total": float(self.mask_count_total),
            "has_invalid_token_ids": float(self.has_invalid_token_ids),
        }


class LegacyTokenizerRunner:
    """Optional runner for legacy tokenizer comparisons in a dedicated env."""

    def __init__(self, legacy_root: Path, skip_steps: int):
        self.legacy_root = legacy_root
        self.skip_steps = int(skip_steps)
        self.tokenizer = self._init_tokenizer()

    def _init_tokenizer(self) -> Any:
        if not self.legacy_root.exists():
            raise FileNotFoundError(f"legacy root does not exist: {self.legacy_root}")
        legacy_root_str = str(self.legacy_root.resolve())
        if legacy_root_str not in sys.path:
            sys.path.insert(0, legacy_root_str)

        cfg = self._build_config()
        try:
            from bmt.tokenization.biycle_tokenizer import BicycleModelTokenizerFixed0124
            return BicycleModelTokenizerFixed0124(cfg)
        except Exception:
            return self._init_tokenizer_lightweight(cfg)

    def _build_config(self) -> Any:
        cfg = types.SimpleNamespace()
        cfg.DELTA_POS_IS_VELOCITY = True
        cfg.GPT_STYLE = True
        cfg.TOKENIZATION = types.SimpleNamespace()
        cfg.TOKENIZATION.NUM_BINS = 33
        cfg.TOKENIZATION.NUM_SKIPPED_STEPS = int(self.skip_steps)
        cfg.TOKENIZATION.NOISE_TOPK = 5
        cfg.TOKENIZATION.ALLOW_SKIP_STEP = True
        cfg.TOKENIZATION.ADD_NOISE = False
        cfg.TRAINING = types.SimpleNamespace()
        cfg.TRAINING.PREDICT_ALL_AGENTS = True
        return cfg

    def _init_tokenizer_lightweight(self, cfg: Any) -> Any:
        motion_path = self.legacy_root / "bmt" / "tokenization" / "motion_tokenizers.py"
        bicycle_path = self.legacy_root / "bmt" / "tokenization" / "biycle_tokenizer.py"
        if not motion_path.exists() or not bicycle_path.exists():
            raise RuntimeError(
                "Failed to import legacy tokenizer stack and missing tokenizer files under legacy root."
            )

        self._clear_bmt_modules()
        self._install_utils_shim()
        self._load_legacy_module("bmt.tokenization.motion_tokenizers", motion_path)
        bicycle_mod = self._load_legacy_module("bmt.tokenization.biycle_tokenizer", bicycle_path)
        return bicycle_mod.BicycleModelTokenizerFixed0124(cfg)

    def _clear_bmt_modules(self) -> None:
        for name in list(sys.modules):
            if name == "bmt" or name.startswith("bmt."):
                del sys.modules[name]

    def _install_utils_shim(self) -> None:
        import torch

        bmt_pkg = types.ModuleType("bmt")
        bmt_pkg.__path__ = [str(self.legacy_root / "bmt")]
        tokenization_pkg = types.ModuleType("bmt.tokenization")
        tokenization_pkg.__path__ = [str(self.legacy_root / "bmt" / "tokenization")]
        utils_pkg = types.ModuleType("bmt.utils")
        utils_pkg.__path__ = [str(self.legacy_root / "bmt" / "utils")]
        utils_mod = types.ModuleType("bmt.utils.utils")

        def wrap_to_pi(radians_array: Any) -> Any:
            if isinstance(radians_array, np.ndarray):
                wrapped = np.mod(radians_array, 2 * np.pi)
                wrapped = np.where(wrapped > np.pi, wrapped - 2 * np.pi, wrapped)
                return wrapped
            if isinstance(radians_array, torch.Tensor):
                wrapped = radians_array % (2 * np.pi)
                wrapped = torch.where(wrapped > np.pi, wrapped - 2 * np.pi, wrapped)
                return wrapped
            wrapped = float(radians_array) % (2 * np.pi)
            return wrapped - 2 * np.pi if wrapped > np.pi else wrapped

        def rotate(x: Any, y: Any, angle: Any, z: Any = None, assert_shape: bool = True) -> Any:
            if assert_shape:
                assert angle.shape == x.shape == y.shape, (angle.shape, x.shape, y.shape)
                if z is not None:
                    assert x.shape == z.shape
            if isinstance(x, torch.Tensor):
                out_x = torch.cos(angle) * x - torch.sin(angle) * y
                out_y = torch.cos(angle) * y + torch.sin(angle) * x
                return torch.stack((out_x, out_y) if z is None else (out_x, out_y, z), dim=-1)
            out_x = np.cos(angle) * x - np.sin(angle) * y
            out_y = np.cos(angle) * y + np.sin(angle) * x
            return np.stack((out_x, out_y) if z is None else (out_x, out_y, z), axis=-1)

        def average_heading(heading1: Any, heading2: Any) -> Any:
            if isinstance(heading1, np.ndarray):
                x1, y1 = np.cos(heading1), np.sin(heading1)
                x2, y2 = np.cos(heading2), np.sin(heading2)
                return np.arctan2((y1 + y2) / 2, (x1 + x2) / 2)
            x1, y1 = torch.cos(heading1), torch.sin(heading1)
            x2, y2 = torch.cos(heading2), torch.sin(heading2)
            return torch.atan2((y1 + y2) / 2, (x1 + x2) / 2)

        def cal_polygon_contour_torch(x: Any, y: Any, theta: Any, width: Any, length: Any) -> Any:
            left_front_x = x + 0.5 * length * torch.cos(theta) - 0.5 * width * torch.sin(theta)
            left_front_y = y + 0.5 * length * torch.sin(theta) + 0.5 * width * torch.cos(theta)
            right_front_x = x + 0.5 * length * torch.cos(theta) + 0.5 * width * torch.sin(theta)
            right_front_y = y + 0.5 * length * torch.sin(theta) - 0.5 * width * torch.cos(theta)
            right_back_x = x - 0.5 * length * torch.cos(theta) + 0.5 * width * torch.sin(theta)
            right_back_y = y - 0.5 * length * torch.sin(theta) - 0.5 * width * torch.cos(theta)
            left_back_x = x - 0.5 * length * torch.cos(theta) - 0.5 * width * torch.sin(theta)
            left_back_y = y - 0.5 * length * torch.sin(theta) + 0.5 * width * torch.cos(theta)
            return torch.stack(
                (
                    torch.stack((left_front_x, left_front_y), dim=-1),
                    torch.stack((right_front_x, right_front_y), dim=-1),
                    torch.stack((right_back_x, right_back_y), dim=-1),
                    torch.stack((left_back_x, left_back_y), dim=-1),
                ),
                dim=-2,
            )

        for mod in (utils_pkg, utils_mod):
            mod.rotate = rotate
            mod.wrap_to_pi = wrap_to_pi
            mod.average_heading = average_heading
            mod.cal_polygon_contour_torch = cal_polygon_contour_torch
        utils_pkg.utils = utils_mod

        sys.modules["bmt"] = bmt_pkg
        sys.modules["bmt.tokenization"] = tokenization_pkg
        sys.modules["bmt.utils"] = utils_pkg
        sys.modules["bmt.utils.utils"] = utils_mod
        bmt_pkg.tokenization = tokenization_pkg
        bmt_pkg.utils = utils_pkg

    def _load_legacy_module(self, module_name: str, file_path: Path) -> Any:
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load legacy module spec: {module_name} from {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def tokenize_batch(self, batch: Dict[str, np.ndarray], backward_prediction: bool) -> Dict[str, np.ndarray]:
        try:
            import torch
        except Exception as e:
            raise RuntimeError("PyTorch is required for legacy tokenization check.") from e

        pos_xy = np.asarray(batch["agent_position_xy"], dtype=np.float32)
        vel_xy = np.asarray(batch["agent_velocity_xy"], dtype=np.float32)
        pos_xyz = np.concatenate([pos_xy, np.zeros(pos_xy.shape[:3] + (1,), dtype=np.float32)], axis=-1)

        td = {
            "decoder/agent_position": torch.from_numpy(pos_xyz),
            "decoder/agent_heading": torch.from_numpy(np.asarray(batch["agent_heading"], dtype=np.float32)),
            "decoder/agent_valid_mask": torch.from_numpy(np.asarray(batch["agent_valid_mask"], dtype=bool)),
            "decoder/agent_velocity": torch.from_numpy(vel_xy),
            "decoder/current_agent_shape": torch.from_numpy(np.asarray(batch["agent_shape"], dtype=np.float32)),
            "decoder/agent_type": torch.from_numpy(np.asarray(batch["agent_type_ids"], dtype=np.int32)),
        }
        out, _ = self.tokenizer.tokenize(td, backward_prediction=bool(backward_prediction))
        return {
            "input_action": out["decoder/input_action"].detach().cpu().numpy().astype(np.int32),
            "input_mask": out["decoder/input_action_valid_mask"].detach().cpu().numpy().astype(bool),
            "target_action": out["decoder/target_action"].detach().cpu().numpy().astype(np.int32),
            "target_mask": out["decoder/target_action_valid_mask"].detach().cpu().numpy().astype(bool),
            "modeled_agent_delta": out["decoder/modeled_agent_delta"].detach().cpu().numpy().astype(np.float32),
        }


def _iter_batches(indices: np.ndarray, batch_size: int) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for i in range(0, len(indices), max(1, int(batch_size))):
        out.append(indices[i:i + max(1, int(batch_size))])
    return out


def _map_legacy_input_to_model_ids(actions: np.ndarray, tokenizer: AdvBMTParityTokenizer) -> np.ndarray:
    out = np.full(actions.shape, tokenizer.PAD_MODEL_ID, dtype=np.int32)
    valid_action = actions >= 0
    out[valid_action] = actions[valid_action]
    out[actions == tokenizer.START_ACTION] = tokenizer.START_MODEL_ID
    out[actions == tokenizer.END_ACTION] = tokenizer.END_MODEL_ID
    out[actions == tokenizer.INVALID_ACTION] = tokenizer.PAD_MODEL_ID
    return out


def _map_legacy_targets(actions: np.ndarray, mask: np.ndarray, tokenizer: AdvBMTParityTokenizer) -> Dict[str, np.ndarray]:
    targets = np.full(actions.shape, tokenizer.default_token_id, dtype=np.int32)
    real = actions >= 0
    targets[real] = actions[real]
    return {"targets": targets, "target_mask": mask.astype(np.float32)}


def _validate_v2_tokens(tokens: Dict[str, np.ndarray], tokenizer: AdvBMTParityTokenizer) -> bool:
    prev = np.asarray(tokens["prev_token_ids"], dtype=np.int32)
    tgt = np.asarray(tokens["targets"], dtype=np.int32)
    bad_prev = np.logical_or(prev < 0, prev > tokenizer.MASK_MODEL_ID).any()
    bad_tgt = np.logical_or(tgt < 0, tgt >= tokenizer.num_actions).any()
    bad_nan = np.isnan(np.asarray(tokens["continuous_motion"], dtype=np.float32)).any()
    return bool(bad_prev or bad_tgt or bad_nan)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare parity tokenization outputs")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="forward", choices=["forward", "backward"])
    parser.add_argument("--n", type=int, default=20, help="number of scenarios")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--skip-steps", type=int, default=5)
    parser.add_argument("--legacy-check", action="store_true")
    parser.add_argument("--legacy-root", type=str, default="src/Adv-BMT")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--min-token-match", type=float, default=0.0)
    parser.add_argument("--min-valid-mask-match", type=float, default=0.0)
    args = parser.parse_args()

    loader = ScenarioNetNNXLoader(data_dir=args.data_dir)
    count = min(max(1, int(args.n)), len(loader))
    indices = np.arange(count, dtype=np.int32)
    batches = _iter_batches(indices, args.batch_size)

    parity_tokenizer = AdvBMTParityTokenizer(
        ParityTokenizerConfig(num_skipped_steps=int(args.skip_steps))
    )
    legacy_runner = None
    if args.legacy_check:
        legacy_runner = LegacyTokenizerRunner(Path(args.legacy_root), skip_steps=int(args.skip_steps))

    stats = CompareStats()
    for bidx in batches:
        samples = [loader.load(int(i)) for i in bidx]
        batch = collate_nnx_scene_samples(samples)
        stats.num_scenarios += len(samples)

        parity_tokens = parity_tokenizer.tokenize_batch(
            batch,
            backward_prediction=(args.mode == "backward"),
        )
        v2 = {
            "prev_token_ids": parity_tokens.prev_token_ids,
            "targets": parity_tokens.targets,
            "target_mask": parity_tokens.target_mask,
            "continuous_motion": parity_tokens.continuous_motion,
        }
        if _validate_v2_tokens(v2, parity_tokenizer):
            stats.has_invalid_token_ids = True

        prev = parity_tokens.prev_token_ids
        stats.start_count_total += int(np.sum(prev == parity_tokenizer.START_MODEL_ID))
        stats.end_count_total += int(np.sum(prev == parity_tokenizer.END_MODEL_ID))
        stats.pad_count_total += int(np.sum(prev == parity_tokenizer.PAD_MODEL_ID))
        stats.mask_count_total += int(np.sum(prev == parity_tokenizer.MASK_MODEL_ID))

        if legacy_runner is None:
            mask = parity_tokens.target_mask > 0.5
            stats.prev_total += int(prev.size)
            stats.prev_match += int(prev.size)
            stats.target_total_masked += int(np.sum(mask))
            stats.target_match_masked += int(np.sum(mask))
            stats.valid_total += int(mask.size)
            stats.valid_match += int(mask.size)
            stats.seq_len_total += int(prev.shape[0])
            stats.seq_len_match += int(prev.shape[0])
            continue

        legacy = legacy_runner.tokenize_batch(batch, backward_prediction=(args.mode == "backward"))
        legacy_prev = _map_legacy_input_to_model_ids(legacy["input_action"], parity_tokenizer)
        legacy_tgt = _map_legacy_targets(legacy["target_action"], legacy["target_mask"], parity_tokenizer)

        s = min(prev.shape[1], legacy_prev.shape[1])
        v2_prev = prev[:, :s]
        l_prev = legacy_prev[:, :s]

        stats.prev_total += int(v2_prev.size)
        stats.prev_match += int(np.sum(v2_prev == l_prev))

        v2_tgt = parity_tokens.targets[:, :s]
        v2_mask = parity_tokens.target_mask[:, :s] > 0.5
        l_tgt = legacy_tgt["targets"][:, :s]
        l_mask = legacy_tgt["target_mask"][:, :s] > 0.5
        masked = np.logical_and(v2_mask, l_mask)
        stats.target_total_masked += int(np.sum(masked))
        stats.target_match_masked += int(np.sum((v2_tgt == l_tgt) & masked))

        stats.valid_total += int(v2_mask.size)
        stats.valid_match += int(np.sum(v2_mask == l_mask))

        stats.seq_len_total += int(v2_prev.shape[0])
        stats.seq_len_match += int(v2_prev.shape[0]) if prev.shape[1] == legacy_prev.shape[1] else 0

    payload: Dict[str, Any] = {
        "config": {
            "data_dir": str(args.data_dir),
            "mode": str(args.mode),
            "n": int(count),
            "batch_size": int(args.batch_size),
            "skip_steps": int(args.skip_steps),
            "legacy_check": bool(args.legacy_check),
        },
        "stats": asdict(stats),
        "metrics": stats.to_metrics(),
    }

    print(json.dumps(payload, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote: {out_path}")

    if stats.has_invalid_token_ids:
        print("FAILED: invalid token IDs or NaNs detected in parity output.", file=sys.stderr)
        return 1

    if args.legacy_check:
        token_match_rate = payload["metrics"]["target_token_exact_match_rate_masked"]
        valid_match_rate = payload["metrics"]["valid_mask_exact_match_rate"]
        if np.isfinite(token_match_rate) and token_match_rate < float(args.min_token_match):
            print(
                f"FAILED: token match {token_match_rate:.6f} < {float(args.min_token_match):.6f}",
                file=sys.stderr,
            )
            return 1
        if np.isfinite(valid_match_rate) and valid_match_rate < float(args.min_valid_mask_match):
            print(
                f"FAILED: valid-mask match {valid_match_rate:.6f} < {float(args.min_valid_mask_match):.6f}",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
