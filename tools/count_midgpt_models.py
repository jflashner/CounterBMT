#!/usr/bin/env python3
"""Count released Adv-BMT MidGPT params and compare against the v2 parity model.

This utility is intentionally lightweight:
- it instantiates the real legacy `MotionLM` from `src/Adv-BMT`
- it avoids pulling in the whole legacy training stack by installing tiny
  import shims for non-counting dependencies (`metadrive`, `torch_geometric`,
  and the broad legacy `bmt.utils` package)
- it reports both total parameter count and top-level parameter buckets so we
  can explain any remaining discrepancy instead of only printing one number

The goal is exact model-construction parity for the released `0202_midgpt.yaml`
architecture, not forward execution of the legacy stack.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import types
from typing import Any, Dict, Iterable, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = REPO_ROOT / "src" / "Adv-BMT"
LEGACY_CFG_PATH = LEGACY_ROOT / "cfgs" / "0202_midgpt.yaml"


class AttrDict(dict):
    """Small recursive dict with attribute access and `.get()` compatibility."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive only
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    @classmethod
    def wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls({k: cls.wrap(v) for k, v in value.items()})
        if isinstance(value, list):
            return [cls.wrap(v) for v in value]
        return value


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key == "defaults":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_legacy_yaml(cfg_path: Path) -> Dict[str, Any]:
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    merged: Dict[str, Any] = {}
    for entry in raw.get("defaults", []) or []:
        if entry == "_self_":
            continue
        if not isinstance(entry, str):
            raise ValueError(f"Unsupported Hydra defaults entry in {cfg_path}: {entry!r}")
        merged = _deep_merge(merged, _load_legacy_yaml(cfg_path.parent / f"{entry}.yaml"))
    merged = _deep_merge(merged, raw)
    return merged


def _install_metadrive_stub() -> None:
    if "metadrive.scenario.scenario_description" in sys.modules:
        return

    metadrive_mod = types.ModuleType("metadrive")
    scenario_mod = types.ModuleType("metadrive.scenario")
    scenario_desc_mod = types.ModuleType("metadrive.scenario.scenario_description")

    class _MetaDriveType:
        UNSET = "UNSET"
        VEHICLE = "VEHICLE"
        PEDESTRIAN = "PEDESTRIAN"
        CYCLIST = "CYCLIST"
        OTHER = "OTHER"
        TRAFFIC_LIGHT = "TRAFFIC_LIGHT"

        @staticmethod
        def is_traffic_light_in_green(state: Any) -> bool:
            return str(state).lower() == "green"

        @staticmethod
        def is_traffic_light_in_yellow(state: Any) -> bool:
            return str(state).lower() == "yellow"

        @staticmethod
        def is_traffic_light_in_red(state: Any) -> bool:
            return str(state).lower() == "red"

        @staticmethod
        def is_traffic_light_unknown(state: Any) -> bool:
            return str(state).lower() not in {"green", "yellow", "red"}

    scenario_desc_mod.MetaDriveType = _MetaDriveType
    sys.modules["metadrive"] = metadrive_mod
    sys.modules["metadrive.scenario"] = scenario_mod
    sys.modules["metadrive.scenario.scenario_description"] = scenario_desc_mod


def _install_torch_geometric_stub() -> None:
    if "torch_geometric.utils" in sys.modules and "torch_geometric.nn" in sys.modules:
        return

    import torch
    import torch.nn as nn

    tg_mod = types.ModuleType("torch_geometric")
    tg_utils_mod = types.ModuleType("torch_geometric.utils")
    tg_nn_mod = types.ModuleType("torch_geometric.nn")

    def dense_to_sparse(adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        adj = adj.contiguous()
        if adj.ndim == 2:
            nz = adj.nonzero(as_tuple=False)
            edge_index = nz.t().contiguous()
            edge_attr = adj[nz[:, 0], nz[:, 1]]
            return edge_index, edge_attr
        if adj.ndim == 3:
            bsz, n_src, n_dst = adj.shape
            nz = adj.nonzero(as_tuple=False)
            src = nz[:, 0] * n_src + nz[:, 1]
            dst = nz[:, 0] * n_dst + nz[:, 2]
            edge_index = torch.stack([src, dst], dim=0).contiguous()
            edge_attr = adj[nz[:, 0], nz[:, 1], nz[:, 2]]
            return edge_index, edge_attr
        raise ValueError(f"dense_to_sparse stub expects 2D or 3D tensor, got {adj.shape}")

    def softmax(src: torch.Tensor, index: torch.Tensor, ptr: Any = None) -> torch.Tensor:
        # Good enough for local forward-debug use. The count utility never executes
        # message passing, but keeping this numerically sane makes the shim safer.
        out = torch.zeros_like(src)
        unique = torch.unique(index)
        for idx in unique:
            mask = index == idx
            vals = src[mask]
            max_vals = vals.max(dim=0, keepdim=True).values
            exp_vals = torch.exp(vals - max_vals)
            out[mask] = exp_vals / exp_vals.sum(dim=0, keepdim=True)
        return out

    class MessagePassing(nn.Module):
        def __init__(self, aggr: str = "add", node_dim: int = 0):
            super().__init__()
            self.aggr = aggr
            self.node_dim = node_dim

        def propagate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - count utility never calls it
            raise RuntimeError(
                "torch_geometric shim is for model construction/counting only; "
                "forward message passing requires the real torch_geometric package."
            )

    tg_utils_mod.dense_to_sparse = dense_to_sparse
    tg_utils_mod.softmax = softmax
    tg_nn_mod.MessagePassing = MessagePassing
    sys.modules["torch_geometric"] = tg_mod
    sys.modules["torch_geometric.utils"] = tg_utils_mod
    sys.modules["torch_geometric.nn"] = tg_nn_mod


def _install_bmt_utils_stub() -> None:
    if "bmt.utils" in sys.modules and "bmt.utils.utils" in sys.modules:
        return

    import numpy as np
    import torch

    utils_mod = types.ModuleType("bmt.utils")
    utils_utils_mod = types.ModuleType("bmt.utils.utils")

    def rotate(x: Any, y: Any, angle: Any) -> Any:
        return (
            x * torch.cos(angle) - y * torch.sin(angle),
            x * torch.sin(angle) + y * torch.cos(angle),
        )

    def unwrap(flatten_array: torch.Tensor, valid_mask: torch.Tensor, existing: Any = None, fill: Any = None) -> torch.Tensor:
        out = existing if existing is not None else flatten_array.new_zeros(valid_mask.shape + (flatten_array.shape[-1],))
        if fill is not None:
            out.fill_(fill)
        out[valid_mask] = flatten_array
        return out

    def wrap_to_pi(values: Any) -> Any:
        if isinstance(values, np.ndarray):
            out = np.mod(values, 2 * np.pi)
            out[out > np.pi] -= 2 * np.pi
            return out
        out = values % (2 * torch.tensor(np.pi, device=values.device, dtype=values.dtype))
        out[out > torch.tensor(np.pi, device=values.device, dtype=values.dtype)] -= 2 * torch.tensor(
            np.pi, device=values.device, dtype=values.dtype
        )
        return out

    def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale) + shift

    def _unsupported(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - defensive only
        raise RuntimeError("This lightweight bmt.utils shim only supports model construction/counting.")

    for mod in (utils_mod, utils_utils_mod):
        mod.REPO_ROOT = LEGACY_ROOT
        mod.rotate = rotate
        mod.unwrap = unwrap
        mod.wrap_to_pi = wrap_to_pi
        mod.modulate = modulate
        mod.calculate_trajectory_probabilities = _unsupported
        mod.cal_polygon_contour_torch = _unsupported
        mod.average_heading = _unsupported
        mod.utils = mod

    sys.modules["bmt.utils"] = utils_mod
    sys.modules["bmt.utils.utils"] = utils_utils_mod


def _install_legacy_import_shims() -> None:
    _install_metadrive_stub()
    _install_torch_geometric_stub()
    _install_bmt_utils_stub()


def _import_legacy_motionlm(legacy_root: Path):
    legacy_root_str = str(legacy_root)
    if legacy_root_str not in sys.path:
        sys.path.insert(0, legacy_root_str)

    _install_legacy_import_shims()
    import bmt  # type: ignore

    # Make the shim visible as `bmt.utils` after the real package object exists.
    bmt.utils = sys.modules["bmt.utils"]  # type: ignore[attr-defined]

    from bmt.models.motionlm import MotionLM  # type: ignore

    return MotionLM


def _count_torch_model(model: Any) -> Tuple[int, Dict[str, int]]:
    total = 0
    buckets: Dict[str, int] = defaultdict(int)
    for name, param in model.named_parameters():
        n = int(param.numel())
        total += n
        top = name.split(".", 1)[0]
        buckets[top] += n
    return total, dict(sorted(buckets.items(), key=lambda kv: kv[1], reverse=True))


def _count_nnx_midgpt() -> Tuple[int, Dict[str, int]]:
    from flax import nnx
    from counter_bmt_v2.trajectory_jax.nnx_bmt import NNXBidirectionalMotionTransformer
    from counter_bmt_v2.trajectory_jax.presets import midgpt_parity_config

    model = NNXBidirectionalMotionTransformer(midgpt_parity_config(), rngs=nnx.Rngs(0))
    flat = nnx.to_flat_state(nnx.state(model))
    total = 0
    buckets: Dict[str, int] = defaultdict(int)
    for path, value in flat:
        arr = value.value if hasattr(value, "value") else value
        n = int(getattr(arr, "size", 0))
        total += n
        if isinstance(path, tuple) and path:
            top = str(path[0])
        else:
            top = str(path)
        buckets[top] += n
    return total, dict(sorted(buckets.items(), key=lambda kv: kv[1], reverse=True))


def _build_report(legacy_total: int, legacy_buckets: Dict[str, int], v2_total: int, v2_buckets: Dict[str, int]) -> Dict[str, Any]:
    delta = int(v2_total - legacy_total)
    denom = float(legacy_total) if legacy_total else 1.0
    return {
        "legacy": {
            "config": str(LEGACY_CFG_PATH),
            "total_params": int(legacy_total),
            "top_level_buckets": legacy_buckets,
        },
        "v2_midgpt_parity": {
            "preset": "midgpt_parity",
            "total_params": int(v2_total),
            "top_level_buckets": v2_buckets,
        },
        "comparison": {
            "delta_params": delta,
            "delta_fraction": float(delta / denom),
            "absolute_delta_fraction": float(abs(delta) / denom),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=str, default=str(LEGACY_ROOT), help="Path to the legacy Adv-BMT root")
    parser.add_argument("--legacy-config", type=str, default=str(LEGACY_CFG_PATH), help="Legacy MidGPT yaml config")
    parser.add_argument("--json-out", type=str, default="", help="Optional JSON output path")
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - local env dependent
        raise SystemExit(
            "PyTorch is required for exact legacy model counting. "
            "Install it into the active environment first. "
            f"Original import error: {exc}"
        )

    legacy_root = Path(args.legacy_root).resolve()
    legacy_cfg_path = Path(args.legacy_config).resolve()

    MotionLM = _import_legacy_motionlm(legacy_root)
    legacy_cfg = AttrDict.wrap(_load_legacy_yaml(legacy_cfg_path))
    legacy_cfg.ROOT_DIR = legacy_root
    legacy_cfg.LOCAL_RANK = 0

    legacy_model = MotionLM(config=legacy_cfg)
    legacy_total, legacy_buckets = _count_torch_model(legacy_model)
    v2_total, v2_buckets = _count_nnx_midgpt()
    report = _build_report(legacy_total, legacy_buckets, v2_total, v2_buckets)

    print(json.dumps(report, indent=2))
    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote JSON report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
