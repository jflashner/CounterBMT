"""Compare v2 dataset discovery/index order against legacy summary-list semantics."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# Allow running as a standalone script from repo root.
if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[2]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

from counter_bmt_v2.data import ScenarioNetNNXLoader


def _load_legacy_reader() -> Any:
    """Return a read_dataset_summary function using legacy-compatible import order."""
    try:
        from scenarionet import read_dataset_summary

        return read_dataset_summary
    except Exception:
        try:
            from metadrive.scenario.utils import read_dataset_summary

            return read_dataset_summary
        except Exception as exc:
            raise RuntimeError(
                "Unable to import read_dataset_summary from scenarionet or metadrive. "
                "Install scenarionet/metadrive in this environment."
            ) from exc


def _read_legacy_summary_pickles(data_dir: Path) -> Tuple[List[str], Dict[str, Any]]:
    summary_path = data_dir / "dataset_summary.pkl"
    if not summary_path.exists():
        raise FileNotFoundError(f"dataset_summary.pkl not found under {data_dir}")

    with summary_path.open("rb") as f:
        summary_obj = pickle.load(f)
    if not isinstance(summary_obj, dict):
        raise TypeError(f"dataset_summary.pkl must contain dict, got: {type(summary_obj)}")
    summary_list = [str(k) for k in summary_obj.keys()]

    mapping_path = data_dir / "dataset_mapping.pkl"
    mapping_obj: Dict[str, Any] = {}
    if mapping_path.exists():
        with mapping_path.open("rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict):
            mapping_obj = {str(k): v for k, v in loaded.items()}
    return summary_list, mapping_obj


def _apply_interval(items: Sequence[str], interval: int) -> List[str]:
    interval = int(interval)
    if interval < 1:
        raise ValueError(f"interval must be >= 1, got {interval}")
    return list(items)[::interval]


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return path.as_posix()


def _resolve_legacy_rel_path(
    *,
    data_dir: Path,
    scenario_file_name: str,
    mapping_value: Any,
) -> str:
    """Resolve legacy summary key + mapping into a relative path for deterministic comparison."""
    root = data_dir.resolve()
    key = str(scenario_file_name)
    key_path = Path(key)

    mapping_dir: Path | None = None
    if mapping_value is not None and str(mapping_value) != "":
        mv = Path(str(mapping_value))
        mapping_dir = mv if mv.is_absolute() else (root / mv)

    candidates: List[Path] = []
    if key_path.is_absolute():
        candidates.append(key_path)
    else:
        candidates.append(root / key_path)
        if mapping_dir is not None:
            # Legacy mapping typically points to the shard folder and key is basename.
            candidates.append(mapping_dir / key_path.name)
            candidates.append(mapping_dir / key_path)

    for cand in candidates:
        if cand.exists():
            return _safe_rel(cand, root)

    # Fallback path reconstruction for non-local checks.
    if mapping_dir is not None:
        return _safe_rel(mapping_dir / key_path.name, root)
    return key_path.as_posix()


def _first_mismatch(a: Sequence[str], b: Sequence[str]) -> Dict[str, Any] | None:
    upto = min(len(a), len(b))
    for i in range(upto):
        if a[i] != b[i]:
            return {"index": i, "v2": a[i], "legacy": b[i]}
    if len(a) != len(b):
        return {"index": upto, "v2": "<end>" if upto >= len(a) else a[upto], "legacy": "<end>" if upto >= len(b) else b[upto]}
    return None


def _compare_lists(
    *,
    split_name: str,
    v2_items: Sequence[str],
    legacy_items: Sequence[str],
    interval: int,
    check_order: bool,
) -> Tuple[Dict[str, Any], bool]:
    v2_after = _apply_interval(v2_items, interval)
    legacy_after = _apply_interval(legacy_items, interval)

    count_match = len(v2_after) == len(legacy_after)
    set_match = set(v2_after) == set(legacy_after)
    order_match = (v2_after == legacy_after) if check_order else None
    mismatch = _first_mismatch(v2_after, legacy_after) if check_order else None

    result = {
        "split": split_name,
        "interval": int(interval),
        "v2_total_pre_interval": int(len(v2_items)),
        "legacy_total_pre_interval": int(len(legacy_items)),
        "v2_total_post_interval": int(len(v2_after)),
        "legacy_total_post_interval": int(len(legacy_after)),
        "count_match": bool(count_match),
        "set_match": bool(set_match),
        "order_match": (None if order_match is None else bool(order_match)),
        "first_mismatch": mismatch,
    }
    failed = (not count_match) or (check_order and not bool(order_match))
    return result, failed


def _collect_v2_rel_paths(data_dir: Path) -> List[str]:
    loader = ScenarioNetNNXLoader(data_dir=data_dir)
    root = data_dir.resolve()
    rel_paths: List[str] = []
    for p in loader.files:
        pp = Path(p)
        if not pp.is_absolute():
            pp = pp.resolve()
        rel_paths.append(_safe_rel(pp, root))
    return rel_paths


def _collect_legacy_rel_paths(data_dir: Path) -> List[str]:
    try:
        summary_list, mapping = _read_legacy_summary_pickles(data_dir)
    except Exception:
        read_dataset_summary = _load_legacy_reader()
        _summary_dict, summary_list, mapping = read_dataset_summary(str(data_dir))
    root = data_dir.resolve()
    out: List[str] = []
    for k in summary_list:
        key = str(k)
        out.append(
            _resolve_legacy_rel_path(
                data_dir=root,
                scenario_file_name=key,
                mapping_value=mapping.get(key) if isinstance(mapping, dict) else None,
            )
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare v2 dataset index ordering with legacy summary-list semantics")
    parser.add_argument("--train", type=str, required=True, help="train ScenarioNet directory")
    parser.add_argument("--val", type=str, required=True, help="val ScenarioNet directory")
    parser.add_argument("--sample-interval-training", type=int, default=1)
    parser.add_argument("--sample-interval-test", type=int, default=1)
    parser.add_argument("--check-order", action="store_true", help="require exact order match after interval slicing")
    parser.add_argument("--json", type=str, default="", help="optional output JSON path")
    args = parser.parse_args()

    train_dir = Path(args.train)
    val_dir = Path(args.val)
    if not train_dir.exists():
        raise FileNotFoundError(f"--train does not exist: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"--val does not exist: {val_dir}")
    if int(args.sample_interval_training) < 1:
        raise ValueError(f"--sample-interval-training must be >= 1, got {args.sample_interval_training}")
    if int(args.sample_interval_test) < 1:
        raise ValueError(f"--sample-interval-test must be >= 1, got {args.sample_interval_test}")

    v2_train = _collect_v2_rel_paths(train_dir)
    legacy_train = _collect_legacy_rel_paths(train_dir)
    v2_val = _collect_v2_rel_paths(val_dir)
    legacy_val = _collect_legacy_rel_paths(val_dir)

    train_result, train_failed = _compare_lists(
        split_name="train",
        v2_items=v2_train,
        legacy_items=legacy_train,
        interval=int(args.sample_interval_training),
        check_order=bool(args.check_order),
    )
    val_result, val_failed = _compare_lists(
        split_name="val",
        v2_items=v2_val,
        legacy_items=legacy_val,
        interval=int(args.sample_interval_test),
        check_order=bool(args.check_order),
    )

    report = {
        "config": {
            "train": str(train_dir),
            "val": str(val_dir),
            "sample_interval_training": int(args.sample_interval_training),
            "sample_interval_test": int(args.sample_interval_test),
            "check_order": bool(args.check_order),
        },
        "results": {
            "train": train_result,
            "val": val_result,
        },
        "pass": bool(not (train_failed or val_failed)),
    }

    print(json.dumps(report, indent=2))
    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if train_failed or val_failed:
        print("FAILED: dataset index parity mismatch detected")
        return 1
    print("PASSED: dataset index parity checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
