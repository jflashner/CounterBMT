"""Create GIFs from ScenarioNet scenarios or existing frame folders.

Modes:
1) Frames mode:
   - Convert an existing directory of PNG/JPG frames into one GIF.
2) Scenario render mode:
   - Render one or more scenarios via ScenarioNetVisualizer and emit GIFs.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[2]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

from counter_bmt.scenarionet_visualizer import ScenarioNetVisualizer


def _require_pillow():
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover - env dependent
        raise RuntimeError("Pillow is required for GIF export. Install with `pip install pillow`.") from exc
    return Image


def _safe_name(value: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in str(value))


def _numeric_sort_key(path: Path) -> tuple[int, float, str]:
    nums = re.findall(r"[-+]?\d*\.?\d+", path.stem)
    if nums:
        try:
            return (0, float(nums[-1]), path.name)
        except Exception:
            pass
    return (1, 0.0, path.name)


def _collect_frames(frames_dir: Path, glob_pattern: str) -> List[Path]:
    frames = [p for p in frames_dir.glob(glob_pattern) if p.is_file()]
    return sorted(frames, key=_numeric_sort_key)


def _write_gif_from_frames(
    frame_paths: Sequence[Path],
    out_path: Path,
    *,
    fps: float,
    loop: int,
) -> None:
    if not frame_paths:
        raise ValueError("No frames provided for GIF export.")
    Image = _require_pillow()
    duration_ms = max(1, int(round(1000.0 / max(0.01, float(fps)))))
    images = [Image.open(str(p)).convert("RGB") for p in frame_paths]
    try:
        images[0].save(
            str(out_path),
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=max(0, int(loop)),
            optimize=False,
        )
    finally:
        for img in images:
            try:
                img.close()
            except Exception:
                pass


def _parse_indices_file(path: Path) -> List[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [int(x) for x in payload]
    if isinstance(payload, dict):
        if isinstance(payload.get("indices"), list):
            return [int(x) for x in payload["indices"]]
        if isinstance(payload.get("entries"), list):
            return [int(x["dataset_index"]) for x in payload["entries"] if isinstance(x, dict) and "dataset_index" in x]
    raise ValueError(f"Unsupported indices file format: {path}")


def _parse_ids_file(path: Path) -> List[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(x) for x in payload if str(x).strip()]
    if isinstance(payload, dict):
        if isinstance(payload.get("scenario_ids"), list):
            return [str(x) for x in payload["scenario_ids"] if str(x).strip()]
        if isinstance(payload.get("entries"), list):
            return [
                str(x["scenario_id"])
                for x in payload["entries"]
                if isinstance(x, dict) and x.get("scenario_id")
            ]
    raise ValueError(f"Unsupported scenario-ids file format: {path}")


def _parse_indices(args: argparse.Namespace) -> List[int]:
    if str(args.scenario_indices_file).strip():
        return _parse_indices_file(Path(args.scenario_indices_file))
    if str(args.scenario_indexes).strip():
        return [int(x.strip()) for x in str(args.scenario_indexes).split(",") if x.strip()]
    if args.start_index is not None and args.end_index is not None:
        return list(range(int(args.start_index), int(args.end_index)))
    if args.scenario_index is not None:
        return [int(args.scenario_index)]
    if args.num_scenarios is not None:
        return list(range(max(0, int(args.num_scenarios))))
    raise ValueError(
        "Scenario mode requires one of: --scenario-index, --scenario-indexes, "
        "--scenario-indices-file, --start-index/--end-index, or --num-scenarios"
    )


def _parse_scenario_ids(args: argparse.Namespace) -> List[str]:
    if str(args.scenario_ids_file).strip():
        return _parse_ids_file(Path(args.scenario_ids_file))
    if str(args.scenario_ids).strip():
        return [x.strip() for x in str(args.scenario_ids).split(",") if x.strip()]
    if str(args.scenario_id).strip():
        return [str(args.scenario_id).strip()]
    return []


def _build_index_from_dataset_mapping(data_dir: Path) -> Dict[str, int]:
    """Build scenario_id -> scenario_index using ScenarioNet dataset_mapping order.

    This order matches ScenarioEnv reset(seed=index) semantics.
    """
    mapping_path = data_dir / "dataset_mapping.pkl"
    if not mapping_path.is_file():
        return {}
    raw = pickle.loads(mapping_path.read_bytes())
    if not isinstance(raw, dict):
        return {}
    index_by_id: Dict[str, int] = {}
    # Keep insertion order from mapping, which defines scenario index order.
    for idx, key in enumerate(raw.keys()):
        name = str(key)
        if name.endswith(".pkl"):
            name = name[:-4]
        sid = name.split("_")[-1].strip().lower()
        if sid and sid not in index_by_id:
            index_by_id[sid] = int(idx)
    return index_by_id


def _resolve_indices_from_ids(
    visualizer: ScenarioNetVisualizer,
    scenario_ids: Sequence[str],
    *,
    data_dir: Path,
) -> List[int]:
    # Prefer dataset_mapping order (true simulator index order); fallback to filename order.
    index_by_id = _build_index_from_dataset_mapping(data_dir)
    if not index_by_id:
        index_by_id = {str(sid).strip().lower(): int(i) for i, sid in enumerate(visualizer.db.scenario_ids)}
    resolved: List[int] = []
    missing: List[str] = []
    for sid in scenario_ids:
        key = str(sid).strip().lower()
        idx = index_by_id.get(key)
        if idx is None:
            missing.append(str(sid))
            continue
        resolved.append(int(idx))
    if missing:
        sample = sorted(index_by_id.keys())[:5]
        raise ValueError(
            "Scenario ID(s) not found in dataset: "
            + ", ".join(missing)
            + f". Sample IDs: {sample}"
        )
    # Preserve user order, remove accidental duplicates.
    uniq: List[int] = []
    seen = set()
    for idx in resolved:
        if idx in seen:
            continue
        seen.add(idx)
        uniq.append(idx)
    return uniq


def _run_frames_mode(args: argparse.Namespace) -> Dict[str, Any]:
    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"frames dir not found: {frames_dir}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = _collect_frames(frames_dir, str(args.frames_glob))
    out_name = str(args.output_name).strip() or f"{frames_dir.name}.gif"
    out_path = output_dir / out_name
    _write_gif_from_frames(frame_paths, out_path, fps=float(args.fps), loop=int(args.loop))
    return {
        "mode": "frames",
        "frames_dir": str(frames_dir),
        "num_frames": int(len(frame_paths)),
        "gif_path": str(out_path),
    }


def _run_scenario_mode(args: argparse.Namespace) -> Dict[str, Any]:
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data dir not found: {data_dir}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = ScenarioNetVisualizer(
        data_dir=str(data_dir),
        film_size=(int(args.film_size), int(args.film_size)),
        screen_size=(int(args.screen_size), int(args.screen_size)),
    )
    scenario_ids = _parse_scenario_ids(args)
    if scenario_ids:
        indices = _resolve_indices_from_ids(visualizer, scenario_ids, data_dir=data_dir)
    else:
        indices = _parse_indices(args)
    results: List[Dict[str, Any]] = []
    try:
        for idx in indices:
            try:
                scenario_dir = output_dir / f"scenario_{int(idx):05d}"
                frames_dir = scenario_dir / "frames"
                scenario_dir.mkdir(parents=True, exist_ok=True)
                saved_images, _trajectory, scenario_id = visualizer.render_scenario(
                    scenario_index=int(idx),
                    num_frames=int(args.num_frames),
                    output_dir=str(frames_dir),
                )
                frame_paths = [Path(p) for p, _ in saved_images]
                gif_name = f"scenario_{int(idx):05d}_{_safe_name(str(scenario_id))}.gif"
                gif_path = output_dir / gif_name
                _write_gif_from_frames(frame_paths, gif_path, fps=float(args.fps), loop=int(args.loop))
                results.append(
                    {
                        "scenario_index": int(idx),
                        "scenario_id": str(scenario_id),
                        "num_frames": int(len(frame_paths)),
                        "frames_dir": str(frames_dir),
                        "gif_path": str(gif_path),
                        "status": "ok",
                    }
                )
                print(f"[gif] ok idx={int(idx)} sid={scenario_id} frames={len(frame_paths)} -> {gif_path}")
            except Exception as exc:
                results.append(
                    {
                        "scenario_index": int(idx),
                        "status": "error",
                        "error": str(exc),
                    }
                )
                print(f"[gif] error idx={int(idx)}: {exc}")
                if not bool(args.continue_on_error):
                    raise
    finally:
        visualizer.close()

    ok = sum(1 for r in results if r.get("status") == "ok")
    return {
        "mode": "scenario",
        "data_dir": str(data_dir),
        "num_requested": int(len(indices)),
        "num_success": int(ok),
        "num_failed": int(len(indices) - ok),
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create GIFs from ScenarioNet renders or existing frame folders.")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory for GIFs and optional summary JSON.")
    p.add_argument("--fps", type=float, default=4.0, help="GIF playback FPS.")
    p.add_argument("--loop", type=int, default=0, help="GIF loop count (0=infinite).")
    p.add_argument("--output-json", type=str, default="", help="Optional JSON summary path.")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--frames-dir", type=str, default="", help="Existing directory of image frames to convert.")
    mode.add_argument("--data-dir", type=str, default="", help="ScenarioNet dataset/replay directory for render mode.")

    # Frames mode
    p.add_argument("--frames-glob", type=str, default="*.png", help="Glob for frame files in --frames-dir.")
    p.add_argument("--output-name", type=str, default="", help="Output gif filename in frames mode.")

    # Scenario mode
    p.add_argument("--scenario-index", type=int, default=None)
    p.add_argument("--scenario-indexes", type=str, default="", help="Comma-separated scenario indices, e.g. 0,4,7")
    p.add_argument("--scenario-indices-file", type=str, default="", help="JSON file with indices list or entries[].dataset_index")
    p.add_argument("--scenario-id", type=str, default="", help="Single scenario ID to render, e.g. 10af3d70d93ef629")
    p.add_argument("--scenario-ids", type=str, default="", help="Comma-separated scenario IDs.")
    p.add_argument("--scenario-ids-file", type=str, default="", help="JSON list or object with scenario_ids / entries[].scenario_id")
    p.add_argument("--start-index", type=int, default=None)
    p.add_argument("--end-index", type=int, default=None)
    p.add_argument("--num-scenarios", type=int, default=None, help="Use first N scenario indices [0..N-1].")
    p.add_argument("--num-frames", type=int, default=16, help="Number of rendered frames per scenario.")
    p.add_argument("--film-size", type=int, default=1200, help="MetaDrive top-down render film_size (square).")
    p.add_argument("--screen-size", type=int, default=800, help="Saved image size (square).")
    p.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue if one scenario render fails.",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()
    if str(args.frames_dir).strip():
        payload = _run_frames_mode(args)
    else:
        payload = _run_scenario_mode(args)
    if str(args.output_json).strip():
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
