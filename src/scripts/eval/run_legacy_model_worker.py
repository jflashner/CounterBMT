"""Legacy Adv-BMT worker: produce canonical forward-eval artifacts for subset."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return getattr(cfg, key)
    except Exception:
        return default


def _cfg_nested_get(cfg: Any, key: str, subkey: str, default: Any) -> Any:
    parent = _cfg_get(cfg, key, None)
    if parent is None:
        return default
    if isinstance(parent, dict):
        return parent.get(subkey, default)
    try:
        return getattr(parent, subkey)
    except Exception:
        return default


def _add_paths(legacy_root: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    src_root = repo_root / "src"
    for p in (src_root, legacy_root):
        s = str(p.resolve())
        if s not in sys.path:
            sys.path.insert(0, s)


def _to_np(x: Any) -> np.ndarray:
    try:
        import torch  # type: ignore
    except Exception:
        torch = None
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _extract_dt_s(raw_scenario: Dict[str, Any]) -> float:
    ts = raw_scenario.get("metadata", {}).get("ts")
    if ts is None:
        return 0.1
    arr = np.asarray(ts, dtype=np.float32)
    if arr.shape[0] < 2:
        return 0.1
    dt = float(np.median(np.diff(arr)))
    if not np.isfinite(dt) or dt <= 0:
        return 0.1
    return dt


def _shape_n3(agent_shape: np.ndarray) -> np.ndarray:
    arr = np.asarray(agent_shape, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        t_idx = min(10, arr.shape[0] - 1)
        return arr[t_idx]
    raise ValueError(f"Unsupported agent_shape rank: {arr.shape}")


def _safe_scenario_id(raw: Dict[str, Any], fallback: str) -> str:
    sid = raw.get("id") or raw.get("metadata", {}).get("scenario_id")
    if sid is None:
        return fallback
    return str(sid)


def _ensure_k_t_n_2(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr)
    while a.ndim > 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 3:  # [T,N,2] -> [1,T,N,2]
        a = a[None, ...]
    if a.ndim != 4:
        raise ValueError(f"{name} expected rank 4, got {a.shape}")
    return np.asarray(a, dtype=np.float32)


def _ensure_k_t_n(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr)
    while a.ndim > 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 2:  # [T,N] -> [1,T,N]
        a = a[None, ...]
    if a.ndim != 3:
        raise ValueError(f"{name} expected rank 3, got {a.shape}")
    return np.asarray(a)


def _double_keys(input_dict: Dict[str, Any]) -> None:
    keys = [
        "decoder/agent_position",
        "decoder/agent_heading",
        "decoder/agent_velocity",
        "decoder/reconstructed_position",
        "decoder/reconstructed_heading",
        "decoder/reconstructed_velocity",
        "decoder/agent_shape",
        "decoder/current_agent_shape",
        "decoder/current_agent_position",
        "encoder/current_agent_position",
        "encoder/current_agent_velocity",
    ]
    try:
        import torch  # type: ignore
    except Exception:
        return
    for k in keys:
        v = input_dict.get(k)
        if isinstance(v, torch.Tensor) and v.dtype in (torch.float16, torch.float32, torch.float64):
            # Keep model inputs in fp32 for broad checkpoint compatibility.
            input_dict[k] = v.float()


def _preprocess_motionlm_compat(
    *,
    preprocess_fn: Any,
    scenario: Dict[str, Any],
    config: Any,
    tokenizer: Any,
) -> Dict[str, Any]:
    """Call legacy preprocessor across API variants.

    Older Adv-BMT versions may not accept `cache=...` and/or `tokenizer=...`.
    """
    base_kwargs = dict(
        scenario=scenario,
        config=config,
        in_evaluation=True,
        keep_all_data=True,
    )
    candidates = [
        dict(base_kwargs, tokenizer=tokenizer, cache=None),
        dict(base_kwargs, tokenizer=tokenizer),
        dict(base_kwargs, cache=None),
        dict(base_kwargs),
    ]
    last_exc: Exception | None = None
    for kw in candidates:
        try:
            return preprocess_fn(**kw)
        except TypeError as exc:
            last_exc = exc
            msg = str(exc)
            if ("unexpected keyword argument" in msg) or ("required positional argument" in msg):
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Failed to call preprocess_scenario_description_for_motionlm with compatible signature")


def _ensure_backward_prediction_tensor(input_dict: Dict[str, Any], device: Any) -> None:
    """Normalize in_backward_prediction to a tensor shaped [B_modes]."""
    try:
        import torch  # type: ignore
    except Exception:
        return

    b_modes = None
    ref = input_dict.get("decoder/input_action")
    if isinstance(ref, torch.Tensor) and ref.ndim >= 1:
        b_modes = int(ref.shape[0])
    if b_modes is None:
        for v in input_dict.values():
            if isinstance(v, torch.Tensor) and v.ndim >= 1:
                b_modes = int(v.shape[0])
                break
    if b_modes is None:
        b_modes = 1

    flag = input_dict.get("in_backward_prediction", False)
    if isinstance(flag, torch.Tensor):
        t = flag.to(device=device, dtype=torch.bool)
        if t.ndim == 0:
            t = t.reshape(1)
        if int(t.numel()) == 1 and b_modes > 1:
            t = t.expand(b_modes)
        elif int(t.numel()) != b_modes:
            t = t.reshape(-1)
            if int(t.numel()) < b_modes:
                pad = torch.zeros((b_modes - int(t.numel()),), dtype=torch.bool, device=device)
                t = torch.cat([t, pad], dim=0)
            else:
                t = t[:b_modes]
        input_dict["in_backward_prediction"] = t
        return

    value = bool(flag)
    input_dict["in_backward_prediction"] = torch.full(
        (b_modes,),
        fill_value=value,
        dtype=torch.bool,
        device=device,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Legacy Adv-BMT worker for head-to-head artifacts")
    p.add_argument("--legacy-root", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--dataset-dir", type=str, required=True)
    p.add_argument("--scenario-subset-file", "--scenario-indices-file", dest="scenario_subset_file", type=str, required=True)
    p.add_argument("--output-artifact-dir", type=str, required=True)
    p.add_argument("--num-modes", type=int, default=6)
    p.add_argument("--sampling-method", type=str, default="topp")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--topp", type=float, default=0.95)
    p.add_argument("--skip-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=str, default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    legacy_root = Path(args.legacy_root)
    dataset_dir = Path(args.dataset_dir)
    subset_path = Path(args.scenario_subset_file)
    out_dir = Path(args.output_artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _add_paths(legacy_root)

    import torch  # type: ignore
    from bmt.dataset.preprocessor import preprocess_scenario_description_for_motionlm  # type: ignore
    from bmt.tokenization import get_tokenizer  # type: ignore
    from bmt.utils import utils as legacy_utils  # type: ignore
    from bmt.utils.utils import numpy_to_torch  # type: ignore

    subset_raw = json.loads(subset_path.read_text(encoding="utf-8"))
    entries = subset_raw.get("entries", []) if isinstance(subset_raw, dict) else []
    if not isinstance(entries, list):
        raise ValueError(f"Invalid subset entries in {subset_path}")

    pl_model = legacy_utils.get_model(checkpoint_path=str(args.ckpt)).eval()
    device = pl_model.device
    config = getattr(pl_model, "config", None)
    if config is None:
        hp = getattr(pl_model, "hparams", None)
        if isinstance(hp, dict):
            config = hp.get("config")
        elif hp is not None:
            config = getattr(hp, "config", None)
    if config is None:
        config = pl_model.model.config
    tokenizer = getattr(pl_model.model, "tokenizer", None)
    if tokenizer is None:
        tokenizer = get_tokenizer(config)
        try:
            pl_model.model.tokenizer = tokenizer
        except Exception:
            pass

    n_ok = 0
    n_fail = 0
    errors: List[Dict[str, Any]] = []
    saved: List[str] = []

    for rank, rec in enumerate(entries):
        rel_path = str(rec.get("relative_path", ""))
        sid_hint = str(rec.get("scenario_id", f"scenario_{rank}"))
        sc_path = dataset_dir / rel_path
        if not sc_path.is_file():
            n_fail += 1
            errors.append({"scenario_id": sid_hint, "error": f"missing file: {sc_path}"})
            continue
        try:
            with sc_path.open("rb") as f:
                raw = pickle.load(f)
            scenario_id = _safe_scenario_id(raw, sid_hint)
            pre = _preprocess_motionlm_compat(
                preprocess_fn=preprocess_scenario_description_for_motionlm,
                scenario=raw,
                config=config,
                tokenizer=tokenizer,
            )
            pre["metadata/scenario_id"] = scenario_id

            input_data = numpy_to_torch(pre, device=device)
            _double_keys(input_data)
            input_data = {
                k: legacy_utils.expand_for_modes(v.unsqueeze(0), num_modes=int(args.num_modes))
                if isinstance(v, torch.Tensor)
                else v
                for k, v in input_data.items()
            }
            input_data["in_evaluation"] = torch.tensor([True], dtype=torch.bool).to(device)
            if bool(_cfg_get(config, "BACKWARD_PREDICTION", False)):
                input_data["in_backward_prediction"] = torch.tensor(
                    [False] * int(args.num_modes),
                    dtype=torch.bool,
                ).to(device)
            _ensure_backward_prediction_tensor(input_data, device)

            tok_data, _ = tokenizer.tokenize(input_data, backward_prediction=False)
            input_data.update(tok_data)
            _ensure_backward_prediction_tensor(input_data, device)

            with torch.no_grad():
                output = pl_model.model.autoregressive_rollout(
                    input_data,
                    num_decode_steps=None,
                    sampling_method=str(args.sampling_method),
                    temperature=float(args.temperature),
                    topp=float(args.topp),
                )

            output = tokenizer.detokenize(
                output,
                detokenizing_gt=False,
                backward_prediction=False,
                flip_wrong_heading=bool(_cfg_nested_get(config, "TOKENIZATION", "FLIP_WRONG_HEADING", True)),
            )

            pred_pos_all = _ensure_k_t_n_2(_to_np(output["decoder/reconstructed_position"]), "pred_pos")
            if "decoder/reconstructed_velocity" in output:
                pred_vel_all = _ensure_k_t_n_2(_to_np(output["decoder/reconstructed_velocity"]), "pred_vel")
            else:
                pred_vel_all = np.zeros_like(pred_pos_all, dtype=np.float32)
            if "decoder/reconstructed_heading" in output:
                pred_heading_all = _ensure_k_t_n(_to_np(output["decoder/reconstructed_heading"]), "pred_heading")
            else:
                pred_heading_all = np.zeros(pred_pos_all.shape[:3], dtype=np.float32)
            if "decoder/reconstructed_valid_mask" in output:
                pred_valid_all = _ensure_k_t_n(_to_np(output["decoder/reconstructed_valid_mask"]), "pred_valid").astype(bool)
            else:
                pred_valid_all = np.ones(pred_pos_all.shape[:3], dtype=bool)

            gt_pos_all = np.asarray(pre["decoder/agent_position"], dtype=np.float32)[..., :2]
            gt_vel_all = np.asarray(pre["decoder/agent_velocity"], dtype=np.float32)
            gt_heading_all = np.asarray(pre["decoder/agent_heading"], dtype=np.float32)
            gt_valid_all = np.asarray(pre["decoder/agent_valid_mask"], dtype=bool)
            shape_n3 = _shape_n3(np.asarray(pre["decoder/agent_shape"], dtype=np.float32))
            sdc_index = int(np.asarray(pre.get("decoder/sdc_index", 0)).reshape(-1)[0])

            t_gt = int(gt_pos_all.shape[0])
            sample_steps = np.arange(0, min(t_gt, 91), int(args.skip_steps), dtype=np.int32)
            if sample_steps.shape[0] < 2:
                raise ValueError(f"too few sampled steps for scenario {scenario_id}")
            eval_steps = sample_steps[1:]

            pred_t = int(pred_pos_all.shape[1])
            if pred_t > int(np.max(eval_steps)):
                pred_idx = eval_steps
            else:
                pred_idx = np.arange(min(pred_t, eval_steps.shape[0]), dtype=np.int32)
                eval_steps = eval_steps[: pred_idx.shape[0]]
            horizon = int(min(pred_idx.shape[0], eval_steps.shape[0]))
            if horizon <= 0:
                raise ValueError(f"zero rollout horizon for scenario {scenario_id}")
            pred_idx = pred_idx[:horizon]
            eval_steps = eval_steps[:horizon]

            pred_pos = np.asarray(pred_pos_all[:, pred_idx, :, :2], dtype=np.float32)
            pred_vel = np.asarray(pred_vel_all[:, pred_idx, :, :2], dtype=np.float32)
            pred_heading = np.asarray(pred_heading_all[:, pred_idx, :], dtype=np.float32)
            pred_speed = np.linalg.norm(pred_vel, axis=-1).astype(np.float32)

            gt_pos = np.asarray(gt_pos_all[eval_steps, :, :2], dtype=np.float32)
            gt_vel = np.asarray(gt_vel_all[eval_steps, :, :2], dtype=np.float32)
            gt_heading = np.asarray(gt_heading_all[eval_steps, :], dtype=np.float32)

            valid_steps_for_transition = sample_steps[: horizon + 1]
            gt_valid_sampled = np.asarray(gt_valid_all[valid_steps_for_transition, :], dtype=bool)
            gt_valid = np.asarray(gt_valid_sampled[1:, :] & gt_valid_sampled[:-1, :], dtype=bool)
            pred_valid = np.broadcast_to(gt_valid[None, :, :], pred_speed.shape).copy()
            pred_valid &= np.asarray(pred_valid_all[:, pred_idx, :], dtype=bool)

            dt_raw = _extract_dt_s(raw)
            dt_chunk = float(max(1e-6, dt_raw * float(args.skip_steps)))

            out_path = out_dir / f"{''.join(ch if ch.isalnum() or ch in ('-','_','.') else '_' for ch in scenario_id)}.npz"
            np.savez_compressed(
                out_path,
                pred_pos_ktn2=pred_pos,
                pred_vel_ktn2=pred_vel,
                pred_speed_ktn=pred_speed,
                pred_valid_ktn=pred_valid,
                pred_heading_ktn=pred_heading,
                gt_pos_tn2=gt_pos,
                gt_vel_tn2=gt_vel,
                gt_valid_tn=gt_valid,
                gt_heading_tn=gt_heading,
                agent_shape_n3=shape_n3,
                dt_chunk_s=np.asarray(dt_chunk, dtype=np.float32),
                sdc_index=np.asarray(int(sdc_index), dtype=np.int32),
                scenario_id=np.asarray(str(scenario_id), dtype=object),
            )
            saved.append(str(out_path))
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            errors.append({"scenario_id": sid_hint, "error": str(exc)})

    summary = {
        "backend": "legacy_adv_bmt",
        "checkpoint": str(args.ckpt),
        "dataset_dir": str(dataset_dir),
        "num_requested": int(len(entries)),
        "num_saved": int(n_ok),
        "num_failed": int(n_fail),
        "artifact_dir": str(out_dir),
        "saved_artifacts": saved,
        "errors": errors[:50],
    }
    if str(args.output_json).strip():
        Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
