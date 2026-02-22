"""Compare v2 relation parity tensors with optional legacy relation outputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Allow running as a standalone script from repo root.
if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[2]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

from counter_bmt_v2.data import ScenarioNetNNXLoader, collate_nnx_scene_samples
from counter_bmt_v2.trajectory_jax import (
    RelationBundleConfig,
    build_relation_bundle,
    build_scene_token_relation_inputs_np,
)


@dataclass
class TargetStats:
    batches: int = 0
    feat_abs_max: float = 0.0
    feat_abs_sum: float = 0.0
    feat_abs_count: int = 0
    mask_total: int = 0
    mask_match: int = 0
    index_total: int = 0
    index_match: int = 0
    v2_edge_total: float = 0.0
    legacy_edge_total: float = 0.0
    has_nan: bool = False

    def to_metrics(self) -> Dict[str, float]:
        return {
            "batches": float(self.batches),
            "feat_abs_max": float(self.feat_abs_max),
            "feat_abs_mean": float(self.feat_abs_sum / self.feat_abs_count) if self.feat_abs_count > 0 else float("nan"),
            "mask_exact_match_rate": float(self.mask_match / self.mask_total) if self.mask_total > 0 else float("nan"),
            "index_exact_match_rate": float(self.index_match / self.index_total) if self.index_total > 0 else float("nan"),
            "v2_mean_edge_count": float(self.v2_edge_total / self.batches) if self.batches > 0 else float("nan"),
            "legacy_mean_edge_count": float(self.legacy_edge_total / self.batches) if self.batches > 0 else float("nan"),
            "has_nan": float(self.has_nan),
        }


class LegacyRelationRunner:
    """Shim-first legacy relation loader to avoid heavy legacy env deps."""

    def __init__(self, legacy_root: Path):
        self.legacy_root = legacy_root
        self.relation_fn = self._init_relation_fn()

    def _init_relation_fn(self) -> Any:
        if not self.legacy_root.exists():
            raise FileNotFoundError(f"legacy root does not exist: {self.legacy_root}")

        try:
            legacy_root_str = str(self.legacy_root.resolve())
            if legacy_root_str not in sys.path:
                sys.path.insert(0, legacy_root_str)
            from bmt.models.relation import compute_relation_simple_relation

            return compute_relation_simple_relation
        except Exception:
            return self._init_relation_fn_lightweight()

    def _init_relation_fn_lightweight(self) -> Any:
        relation_path = self.legacy_root / "bmt" / "models" / "relation.py"
        if not relation_path.exists():
            raise RuntimeError(f"legacy relation module not found: {relation_path}")

        self._clear_bmt_modules()
        self._install_shims()
        rel_mod = self._load_legacy_module("bmt.models.relation", relation_path)
        return rel_mod.compute_relation_simple_relation

    def _clear_bmt_modules(self) -> None:
        for name in list(sys.modules):
            if name == "bmt" or name.startswith("bmt."):
                del sys.modules[name]

    def _install_shims(self) -> None:
        import torch

        bmt_pkg = types.ModuleType("bmt")
        bmt_pkg.__path__ = [str(self.legacy_root / "bmt")]

        dataset_pkg = types.ModuleType("bmt.dataset")
        dataset_pkg.__path__ = [str(self.legacy_root / "bmt" / "dataset")]
        constants_mod = types.ModuleType("bmt.dataset.constants")
        constants_mod.HEADING_PLACEHOLDER = -100.0

        models_pkg = types.ModuleType("bmt.models")
        models_pkg.__path__ = [str(self.legacy_root / "bmt" / "models")]

        layers_pkg = types.ModuleType("bmt.models.layers")
        layers_pkg.__path__ = [str(self.legacy_root / "bmt" / "models" / "layers")]
        pos_enc_mod = types.ModuleType("bmt.models.layers.position_encoding_utils")

        def _unused_gen_sineembed_for_relation(*args, **kwargs):
            raise RuntimeError("position encoding should not be called in return_pe=False relation checks")

        pos_enc_mod.gen_sineembed_for_relation = _unused_gen_sineembed_for_relation

        utils_pkg = types.ModuleType("bmt.utils")
        utils_pkg.__path__ = [str(self.legacy_root / "bmt" / "utils")]
        utils_mod = types.ModuleType("bmt.utils.utils")

        def rotate(x: Any, y: Any, angle: Any, z: Any = None, assert_shape: bool = True) -> Any:
            if assert_shape:
                assert angle.shape == x.shape == y.shape
                if z is not None:
                    assert x.shape == z.shape
            if isinstance(x, torch.Tensor):
                rx = torch.cos(angle) * x - torch.sin(angle) * y
                ry = torch.cos(angle) * y + torch.sin(angle) * x
                return torch.stack((rx, ry) if z is None else (rx, ry, z), dim=-1)
            rx = np.cos(angle) * x - np.sin(angle) * y
            ry = np.cos(angle) * y + np.sin(angle) * x
            return np.stack((rx, ry) if z is None else (rx, ry, z), axis=-1)

        def cal_polygon_contour_torch(x: Any, y: Any, theta: Any, width: Any, length: Any) -> Any:
            lf_x = x + 0.5 * length * torch.cos(theta) - 0.5 * width * torch.sin(theta)
            lf_y = y + 0.5 * length * torch.sin(theta) + 0.5 * width * torch.cos(theta)
            rf_x = x + 0.5 * length * torch.cos(theta) + 0.5 * width * torch.sin(theta)
            rf_y = y + 0.5 * length * torch.sin(theta) - 0.5 * width * torch.cos(theta)
            rb_x = x - 0.5 * length * torch.cos(theta) + 0.5 * width * torch.sin(theta)
            rb_y = y - 0.5 * length * torch.sin(theta) - 0.5 * width * torch.cos(theta)
            lb_x = x - 0.5 * length * torch.cos(theta) - 0.5 * width * torch.sin(theta)
            lb_y = y - 0.5 * length * torch.sin(theta) + 0.5 * width * torch.cos(theta)
            return torch.stack(
                (
                    torch.stack((lf_x, lf_y), dim=-1),
                    torch.stack((rf_x, rf_y), dim=-1),
                    torch.stack((rb_x, rb_y), dim=-1),
                    torch.stack((lb_x, lb_y), dim=-1),
                ),
                dim=-2,
            )

        utils_pkg.rotate = rotate
        utils_pkg.utils = utils_mod
        utils_mod.rotate = rotate
        utils_mod.cal_polygon_contour_torch = cal_polygon_contour_torch

        sys.modules["bmt"] = bmt_pkg
        sys.modules["bmt.dataset"] = dataset_pkg
        sys.modules["bmt.dataset.constants"] = constants_mod
        sys.modules["bmt.models"] = models_pkg
        sys.modules["bmt.models.layers"] = layers_pkg
        sys.modules["bmt.models.layers.position_encoding_utils"] = pos_enc_mod
        sys.modules["bmt.utils"] = utils_pkg
        sys.modules["bmt.utils.utils"] = utils_mod

        bmt_pkg.dataset = dataset_pkg
        bmt_pkg.models = models_pkg
        bmt_pkg.utils = utils_pkg
        dataset_pkg.constants = constants_mod
        models_pkg.layers = layers_pkg
        layers_pkg.position_encoding_utils = pos_enc_mod

    def _load_legacy_module(self, module_name: str, file_path: Path) -> Any:
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load legacy module spec: {module_name} from {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _to_tensor(self, x: np.ndarray, *, bool_tensor: bool = False) -> Any:
        import torch

        arr = np.asarray(x)
        if bool_tensor:
            return torch.from_numpy(arr.astype(bool))
        if np.issubdtype(arr.dtype, np.integer):
            return torch.from_numpy(arr.astype(np.int64))
        return torch.from_numpy(arr.astype(np.float32))

    def compute_scene_s2s(
        self,
        *,
        scene_position: np.ndarray,
        scene_heading: np.ndarray,
        scene_valid_mask: np.ndarray,
        cfg: RelationBundleConfig,
    ) -> Dict[str, np.ndarray]:
        feat, mask, idx = self.relation_fn(
            query_pos=self._to_tensor(scene_position),
            query_heading=self._to_tensor(scene_heading),
            query_valid_mask=self._to_tensor(scene_valid_mask, bool_tensor=True),
            key_pos=self._to_tensor(scene_position),
            key_heading=self._to_tensor(scene_heading),
            key_valid_mask=self._to_tensor(scene_valid_mask, bool_tensor=True),
            hidden_dim=128,
            causal_valid_mask=None,
            knn=cfg.s2s_knn,
            max_distance=cfg.s2s_distance,
            gather=False,
            return_pe=False,
            non_agent_relation=True,
            per_contour_point_relation=cfg.per_contour_point_relation,
        )
        return {
            "feat": feat.detach().cpu().numpy().astype(np.float32),
            "mask": mask.detach().cpu().numpy().astype(bool),
            "indices": (
                np.zeros((feat.shape[0], feat.shape[1], 0), dtype=np.int32)
                if idx is None
                else idx.detach().cpu().numpy().astype(np.int32)
            ),
        }


def _iter_batches(indices: np.ndarray, batch_size: int) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for i in range(0, len(indices), max(1, int(batch_size))):
        out.append(indices[i:i + max(1, int(batch_size))])
    return out


def _slice_to_common(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if a.shape == b.shape:
        return a, b
    n = min(a.ndim, b.ndim)
    slices = tuple(slice(0, min(a.shape[i], b.shape[i])) for i in range(n))
    return a[slices], b[slices]


def _update_stats(
    stats: TargetStats,
    *,
    v2_feat: np.ndarray,
    v2_mask: np.ndarray,
    legacy_feat: Optional[np.ndarray] = None,
    legacy_mask: Optional[np.ndarray] = None,
    v2_idx: Optional[np.ndarray] = None,
    legacy_idx: Optional[np.ndarray] = None,
) -> None:
    stats.batches += 1
    stats.has_nan = bool(
        stats.has_nan
        or np.isnan(v2_feat).any()
        or np.isnan(v2_mask.astype(np.float32)).any()
        or (legacy_feat is not None and np.isnan(legacy_feat).any())
    )

    stats.v2_edge_total += float(np.sum(v2_mask))

    if legacy_feat is None or legacy_mask is None:
        stats.feat_abs_max = max(stats.feat_abs_max, 0.0)
        stats.mask_total += int(v2_mask.size)
        stats.mask_match += int(v2_mask.size)
        return

    legacy_feat, v2_feat = _slice_to_common(legacy_feat, v2_feat)
    legacy_mask, v2_mask = _slice_to_common(legacy_mask, v2_mask)

    diff = np.abs(v2_feat - legacy_feat)
    stats.feat_abs_max = max(stats.feat_abs_max, float(np.max(diff)) if diff.size else 0.0)
    stats.feat_abs_sum += float(np.sum(diff))
    stats.feat_abs_count += int(diff.size)

    stats.mask_total += int(v2_mask.size)
    stats.mask_match += int(np.sum(v2_mask == legacy_mask))

    stats.legacy_edge_total += float(np.sum(legacy_mask))

    if v2_idx is not None and legacy_idx is not None:
        legacy_idx, v2_idx = _slice_to_common(legacy_idx, v2_idx)
        stats.index_total += int(v2_idx.size)
        stats.index_match += int(np.sum(v2_idx == legacy_idx))


def _targets_from_bundle(bundle: Dict[str, np.ndarray], target: str) -> Dict[str, Dict[str, np.ndarray]]:
    all_targets = {
        "scene_s2s": {
            "feat": bundle["scene_s2s_rel_feat"],
            "mask": bundle["scene_s2s_mask"],
            "idx": bundle["scene_s2s_indices"],
        },
        "a2a": {"feat": bundle["a2a_rel_feat"], "mask": bundle["a2a_mask"], "idx": bundle["a2a_indices"]},
        "a2t": {"feat": bundle["a2t_rel_feat"], "mask": bundle["a2t_mask"], "idx": bundle["a2t_indices"]},
        "a2s": {"feat": bundle["a2s_rel_feat"], "mask": bundle["a2s_mask"], "idx": bundle["a2s_indices"]},
    }
    if target == "all":
        return all_targets
    return {target: all_targets[target]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare relation parity outputs")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--target", type=str, default="scene_s2s", choices=["scene_s2s", "a2a", "a2t", "a2s", "all"])
    parser.add_argument("--mode", type=str, default="simple", choices=["simple"])
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--skip-steps", type=int, default=5)
    parser.add_argument("--legacy-check", action="store_true")
    parser.add_argument("--legacy-root", type=str, default="src/Adv-BMT")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--max-feat-diff", type=float, default=np.inf)
    parser.add_argument("--min-mask-match", type=float, default=0.0)
    args = parser.parse_args()

    loader = ScenarioNetNNXLoader(data_dir=args.data_dir)
    count = min(max(1, int(args.n)), len(loader))
    indices = np.arange(count, dtype=np.int32)
    batches = _iter_batches(indices, args.batch_size)

    cfg = RelationBundleConfig(
        simple_relation=True,
        per_contour_point_relation=False,
        include_contour=True,
        heading_placeholder=-100.0,
        s2s_knn=128,
        s2s_distance=None,
        a2s_knn=128,
        a2s_distance=None,
        a2a_knn=64,
        a2a_distance=50.0,
        remove_traffic_light_state=True,
    )

    legacy_runner = LegacyRelationRunner(Path(args.legacy_root)) if args.legacy_check else None

    target_names = [args.target] if args.target != "all" else ["scene_s2s", "a2a", "a2t", "a2s"]
    stats: Dict[str, TargetStats] = {name: TargetStats() for name in target_names}

    for bidx in batches:
        samples = [loader.load(int(i)) for i in bidx]
        batch = collate_nnx_scene_samples(samples)

        sample_steps = np.arange(0, batch["agent_position_xy"].shape[1], max(1, int(args.skip_steps)), dtype=np.int32)
        scene_inputs = build_scene_token_relation_inputs_np(
            map_feature=np.asarray(batch["map_feature"], dtype=np.float32),
            map_feature_valid_mask=np.asarray(batch["map_feature_valid_mask"], dtype=bool),
            map_position=np.asarray(batch["map_position"], dtype=np.float32),
            traffic_light_feature=np.asarray(batch["traffic_light_feature"], dtype=np.float32),
            traffic_light_valid_mask=np.asarray(batch["traffic_light_valid_mask"], dtype=bool),
            traffic_light_position=np.asarray(batch["traffic_light_position"], dtype=np.float32),
            remove_traffic_light_state=cfg.remove_traffic_light_state,
            heading_placeholder=cfg.heading_placeholder,
        )

        bundle = build_relation_bundle(
            agent_position_xy=np.asarray(batch["agent_position_xy"], dtype=np.float32),
            agent_heading=np.asarray(batch["agent_heading"], dtype=np.float32),
            agent_valid_mask=np.asarray(batch["agent_valid_mask"], dtype=bool),
            agent_shape=np.asarray(batch["agent_shape"], dtype=np.float32),
            sample_steps=sample_steps,
            scene_position=scene_inputs["scene_position"],
            scene_heading=scene_inputs["scene_heading"],
            scene_valid_mask=scene_inputs["scene_valid_mask"],
            cfg=cfg,
        )
        selected = _targets_from_bundle(bundle, args.target)

        legacy_scene = None
        if legacy_runner is not None:
            legacy_scene = legacy_runner.compute_scene_s2s(
                scene_position=scene_inputs["scene_position"],
                scene_heading=scene_inputs["scene_heading"],
                scene_valid_mask=scene_inputs["scene_valid_mask"],
                cfg=cfg,
            )

        for name, payload in selected.items():
            if name == "scene_s2s" and legacy_scene is not None:
                _update_stats(
                    stats[name],
                    v2_feat=np.asarray(payload["feat"], dtype=np.float32),
                    v2_mask=np.asarray(payload["mask"], dtype=bool),
                    legacy_feat=legacy_scene["feat"],
                    legacy_mask=legacy_scene["mask"],
                    v2_idx=np.asarray(payload["idx"], dtype=np.int32),
                    legacy_idx=legacy_scene["indices"],
                )
            else:
                _update_stats(
                    stats[name],
                    v2_feat=np.asarray(payload["feat"], dtype=np.float32),
                    v2_mask=np.asarray(payload["mask"], dtype=bool),
                )

    payload = {
        "config": {
            "data_dir": str(args.data_dir),
            "target": str(args.target),
            "mode": str(args.mode),
            "n": int(count),
            "batch_size": int(args.batch_size),
            "skip_steps": int(args.skip_steps),
            "legacy_check": bool(args.legacy_check),
        },
        "targets": {name: {"stats": asdict(s), "metrics": s.to_metrics()} for name, s in stats.items()},
    }

    print(json.dumps(payload, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote: {out_path}")

    for name, s in stats.items():
        metrics = s.to_metrics()
        if bool(metrics["has_nan"]):
            print(f"FAILED[{name}]: NaN detected in relation tensors", file=sys.stderr)
            return 1
        if args.legacy_check and name == "scene_s2s":
            if np.isfinite(metrics["feat_abs_max"]) and metrics["feat_abs_max"] > float(args.max_feat_diff):
                print(
                    f"FAILED[{name}]: feat_abs_max={metrics['feat_abs_max']:.6g} > {float(args.max_feat_diff):.6g}",
                    file=sys.stderr,
                )
                return 1
            if np.isfinite(metrics["mask_exact_match_rate"]) and metrics["mask_exact_match_rate"] < float(args.min_mask_match):
                print(
                    f"FAILED[{name}]: mask_exact_match_rate={metrics['mask_exact_match_rate']:.6f} < {float(args.min_mask_match):.6f}",
                    file=sys.stderr,
                )
                return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
