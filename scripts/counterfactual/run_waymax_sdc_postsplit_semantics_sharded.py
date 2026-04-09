from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    for path in (repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from scripts.counterfactual.label_waymax_sdc_path_semantics import write_json, write_jsonl  # type: ignore[attr-defined]

DEFAULT_WOD_131_TRAIN_PATH = "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/training/training_tfexample.tfrecord-00000-of-01000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and merge sharded postsplit SDC semantic rendering jobs.")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--shard-root", type=str, default="")
    parser.add_argument("--path", type=str, default=DEFAULT_WOD_131_TRAIN_PATH)
    parser.add_argument("--config-name", type=str, default="WOD_1_3_1_TRAINING")
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--candidate-scenes", type=int, required=True)
    parser.add_argument("--shard-size", type=int, required=True)
    parser.add_argument("--num-selected-scenes", type=int, required=True)
    parser.add_argument("--local-num-selected-scenes", type=int, default=0)
    parser.add_argument("--max-parallel-shards", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--current-time-index", type=int, default=-1)
    parser.add_argument("--num-paths", type=int, default=45)
    parser.add_argument("--num-points-per-path", type=int, default=800)
    parser.add_argument("--min-route-length-m", type=float, default=15.0)
    parser.add_argument("--min-gt-length-m", type=float, default=10.0)
    parser.add_argument("--gt-relative-threshold-m", type=float, default=10.0)
    parser.add_argument("--alt-diversity-weight", type=float, default=1.0)
    parser.add_argument("--include-off-route-paths", action="store_true")
    parser.add_argument("--diversity-top-k", type=int, default=0)
    # Match the current postsplit gold-standard bundle defaults unless explicitly overridden.
    parser.add_argument("--gradient-display-reference", type=float, default=0.75)
    parser.add_argument("--gradient-display-gamma", type=float, default=1.10)
    parser.add_argument("--show-traffic-lights", action="store_true")
    parser.add_argument("--save-scene-grid", action="store_true")
    parser.add_argument("--scene-grid-columns", type=int, default=4)
    parser.add_argument("--scene-grid-padding-m", type=float, default=18.0)
    parser.add_argument("--resample-spacing-m", type=float, default=2.0)
    parser.add_argument("--separability-scale-m", type=float, default=6.0)
    parser.add_argument("--separability-heading-weight-m", type=float, default=2.0)
    parser.add_argument("--stitch-discontinuities", action="store_true")
    parser.add_argument("--stitch-radius-m", type=float, default=2.0)
    parser.add_argument("--stitch-jump-threshold-m", type=float, default=6.0)
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--image-detail", type=str, default="original", choices=("low", "high", "original", "auto"))
    parser.add_argument("--save-pkls", action="store_true")
    parser.add_argument("--render-workers", type=int, default=1)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    write_json(path, dict(payload))
    return path


def _build_shard_specs(*, scene_offset: int, candidate_scenes: int, shard_size: int) -> List[Dict[str, int]]:
    specs: List[Dict[str, int]] = []
    start = int(scene_offset)
    stop = int(scene_offset) + int(candidate_scenes)
    shard_idx = 0
    while start < stop:
        shard_candidate_scenes = min(int(shard_size), stop - start)
        specs.append(
            {
                "shard_index": int(shard_idx),
                "scene_offset": int(start),
                "candidate_scenes": int(shard_candidate_scenes),
            }
        )
        shard_idx += 1
        start += shard_candidate_scenes
    return specs


def _render_script_path() -> Path:
    return Path(__file__).resolve().parent / "render_waymax_sdc_postsplit_semantics.py"


def _build_render_command(
    *,
    args: argparse.Namespace,
    shard_outdir: Path,
    shard_spec: Mapping[str, int],
    local_num_selected_scenes: int,
) -> List[str]:
    cmd = [
        sys.executable,
        str(_render_script_path()),
        "--outdir",
        str(shard_outdir),
        "--path",
        str(args.path),
        "--config-name",
        str(args.config_name),
        "--scene-offset",
        str(int(shard_spec["scene_offset"])),
        "--candidate-scenes",
        str(int(shard_spec["candidate_scenes"])),
        "--num-selected-scenes",
        str(int(local_num_selected_scenes)),
        "--current-time-index",
        str(int(args.current_time_index)),
        "--num-paths",
        str(int(args.num_paths)),
        "--num-points-per-path",
        str(int(args.num_points_per_path)),
        "--min-route-length-m",
        str(float(args.min_route_length_m)),
        "--min-gt-length-m",
        str(float(args.min_gt_length_m)),
        "--gt-relative-threshold-m",
        str(float(args.gt_relative_threshold_m)),
        "--alt-diversity-weight",
        str(float(args.alt_diversity_weight)),
        "--diversity-top-k",
        str(int(args.diversity_top_k)),
        "--gradient-display-reference",
        str(float(args.gradient_display_reference)),
        "--gradient-display-gamma",
        str(float(args.gradient_display_gamma)),
        "--scene-grid-columns",
        str(int(args.scene_grid_columns)),
        "--scene-grid-padding-m",
        str(float(args.scene_grid_padding_m)),
        "--resample-spacing-m",
        str(float(args.resample_spacing_m)),
        "--separability-scale-m",
        str(float(args.separability_scale_m)),
        "--separability-heading-weight-m",
        str(float(args.separability_heading_weight_m)),
        "--stitch-radius-m",
        str(float(args.stitch_radius_m)),
        "--stitch-jump-threshold-m",
        str(float(args.stitch_jump_threshold_m)),
        "--model",
        str(args.model),
        "--image-detail",
        str(args.image_detail),
        "--render-workers",
        str(int(args.render_workers)),
    ]
    if bool(args.include_off_route_paths):
        cmd.append("--include-off-route-paths")
    if bool(args.stitch_discontinuities):
        cmd.append("--stitch-discontinuities")
    if bool(args.show_traffic_lights):
        cmd.append("--show-traffic-lights")
    if bool(args.save_scene_grid):
        cmd.append("--save-scene-grid")
    if bool(args.save_pkls):
        cmd.append("--save-pkls")
    return cmd


def _run_shard_command(*, cmd: Sequence[str], log_path: Path) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wt", encoding="utf-8") as f:
        process = subprocess.run(list(cmd), stdout=f, stderr=subprocess.STDOUT, check=False)
    return {
        "returncode": int(process.returncode),
        "log_path": str(log_path.resolve()),
        "command": list(cmd),
    }


def _ranking_sort_key(row: Mapping[str, Any]) -> Tuple[float, int, str, str]:
    return (
        -float(row.get("selection_score") or 0.0),
        int(row.get("scene_index") or 0),
        str(row.get("scenario_id") or ""),
        str(row.get("sdc_id") or ""),
    )


def _example_id_from_row(row: Mapping[str, Any]) -> str:
    return f"{str(row.get('scenario_id') or '')}__sdc_{str(row.get('sdc_id') or '')}__t_{int(row.get('current_time_index') or 0):03d}"


def _rewrite_example_dir(example_dir: Path) -> Dict[str, Any]:
    render_metadata_path = example_dir / "render_metadata.json"
    payload = _read_json(render_metadata_path)
    images = dict(payload.get("images") or {})
    rewritten_images: Dict[str, str] = {}
    for slot_id, old_path in images.items():
        image_name = Path(str(old_path)).name
        rewritten_images[str(slot_id)] = str((example_dir / "images" / image_name).resolve())
    payload["images"] = rewritten_images

    grid_path = payload.get("all_sdc_paths_grid_png")
    if str(grid_path or "").strip():
        payload["all_sdc_paths_grid_png"] = str((example_dir / Path(str(grid_path)).name).resolve())

    for request_json_path in sorted(example_dir.glob("request_*.json")):
        request_payload = _read_json(request_json_path)
        new_image_paths = [
            str((example_dir / "images" / Path(str(path)).name).resolve())
            for path in list(request_payload.get("image_paths") or [])
        ]
        request_payload["image_paths"] = new_image_paths
        _write_json(request_json_path, request_payload)

    _write_json(render_metadata_path, payload)
    return payload


def _prompt_manifest_rows_from_example(payload: Mapping[str, Any], *, example_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for slot_row in list(payload.get("slot_metadata") or []):
        slot_id = str(slot_row.get("slot_id") or "")
        request_json_path = example_dir / f"request_{slot_id}.json"
        if not request_json_path.is_file():
            continue
        request_payload = _read_json(request_json_path)
        rows.append(
            {
                "example_id": str(payload.get("example_id") or ""),
                "scenario_id": str(payload.get("scenario_id") or ""),
                "sdc_id": str(payload.get("sdc_id") or ""),
                "slot_id": slot_id,
                "prompt_path": str((example_dir / f"prompt_{slot_id}.txt").resolve()),
                "request_json": str(request_json_path.resolve()),
                "image_paths": [str(path) for path in list(request_payload.get("image_paths") or [])],
            }
        )
    return rows


def _aggregate_row_from_payload(payload: Mapping[str, Any], ranking_row: Mapping[str, Any], *, example_dir: Path, selection_rank: int) -> Dict[str, Any]:
    return {
        "example_id": str(payload.get("example_id") or ""),
        "scenario_id": str(payload.get("scenario_id") or ""),
        "sdc_id": str(payload.get("sdc_id") or ""),
        "scene_index": int(ranking_row.get("scene_index") or 0),
        "current_time_index": int(payload.get("current_time_index") or 0),
        "selection_rank": int(selection_rank),
        "selection_score": float(ranking_row.get("selection_score") or 0.0),
        "selection_score_kind": str(ranking_row.get("selection_score_kind") or "unknown"),
        "selection_gt_component": float(ranking_row.get("selection_gt_component") or 0.0),
        "selection_alt_diversity_component": float(ranking_row.get("selection_alt_diversity_component") or 0.0),
        "gt_length_m": float(ranking_row.get("gt_length_m") or 0.0),
        "selected_alt_path_ids": list(ranking_row.get("selected_alt_path_ids") or []),
        "slot_metadata": payload.get("slot_metadata"),
        "images": payload.get("images"),
        "all_sdc_paths_grid_png": payload.get("all_sdc_paths_grid_png"),
        "all_sdc_paths_grid_summary": payload.get("all_sdc_paths_grid_summary"),
        "prompt_paths": {
            slot_id: str((example_dir / f"prompt_{slot_id}.txt").resolve())
            for slot_id in ["gt", "alt_1", "alt_2", "alt_3"]
            if (example_dir / f"prompt_{slot_id}.txt").is_file()
        },
        "request_jsons": {
            slot_id: str((example_dir / f"request_{slot_id}.json").resolve())
            for slot_id in ["gt", "alt_1", "alt_2", "alt_3"]
            if (example_dir / f"request_{slot_id}.json").is_file()
        },
    }


def _copy_example_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _progress_bar(*, completed: int, total: int, width: int = 24) -> str:
    total = max(1, int(total))
    completed = min(max(0, int(completed)), total)
    filled = int(round((completed / total) * width))
    filled = min(max(filled, 0), width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _print_progress(*, label: str, completed: int, total: int, start_time: float) -> None:
    elapsed = max(0.0, time.time() - float(start_time))
    rate = (elapsed / completed) if completed > 0 else 0.0
    remaining = max(0, total - completed)
    eta = (rate * remaining) if completed > 0 else 0.0
    print(
        f"{label} {_progress_bar(completed=completed, total=total)} "
        f"{completed}/{total} | elapsed {_format_duration(elapsed)} | eta {_format_duration(eta)}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    shard_root = Path(str(args.shard_root).strip()).expanduser().resolve() if str(args.shard_root).strip() else outdir.parent / f"{outdir.name}__shards"
    shard_specs = _build_shard_specs(
        scene_offset=int(args.scene_offset),
        candidate_scenes=int(args.candidate_scenes),
        shard_size=int(args.shard_size),
    )
    local_num_selected_scenes = int(args.local_num_selected_scenes) if int(args.local_num_selected_scenes) > 0 else int(args.shard_size)

    shard_jobs: List[Dict[str, Any]] = []
    for shard_spec in shard_specs:
        shard_dir = shard_root / f"shard_{int(shard_spec['shard_index']):04d}"
        cmd = _build_render_command(
            args=args,
            shard_outdir=shard_dir,
            shard_spec=shard_spec,
            local_num_selected_scenes=local_num_selected_scenes,
        )
        shard_jobs.append(
            {
                "shard_index": int(shard_spec["shard_index"]),
                "scene_offset": int(shard_spec["scene_offset"]),
                "candidate_scenes": int(shard_spec["candidate_scenes"]),
                "shard_dir": str(shard_dir.resolve()),
                "log_path": str((shard_root / "logs" / f"shard_{int(shard_spec['shard_index']):04d}.log").resolve()),
                "command": cmd,
            }
        )

    if bool(args.dry_run):
        print(json.dumps({"outdir": str(outdir), "shard_root": str(shard_root), "shards": shard_jobs}, indent=2, sort_keys=True))
        return 0

    if not bool(args.merge_only):
        shard_start_time = time.time()
        if bool(args.progress):
            print(
                f"Starting shard rendering for {len(shard_jobs)} shards "
                f"(max_parallel={max(1, int(args.max_parallel_shards))})",
                flush=True,
            )
        max_parallel = max(1, int(args.max_parallel_shards))
        if max_parallel <= 1 or len(shard_jobs) <= 1:
            run_results = []
            for completed_count, job in enumerate(shard_jobs, start=1):
                result = _run_shard_command(
                    cmd=job["command"],
                    log_path=Path(job["log_path"]),
                )
                result["shard_index"] = int(job["shard_index"])
                run_results.append(result)
                if bool(args.progress):
                    _print_progress(
                        label="Shards",
                        completed=completed_count,
                        total=len(shard_jobs),
                        start_time=shard_start_time,
                    )
        else:
            run_results = []
            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                future_map = {
                    executor.submit(
                        _run_shard_command,
                        cmd=job["command"],
                        log_path=Path(job["log_path"]),
                    ): dict(job)
                    for job in shard_jobs
                }
                completed_count = 0
                for future in as_completed(future_map):
                    job = future_map[future]
                    result = future.result()
                    result["shard_index"] = int(job["shard_index"])
                    run_results.append(result)
                    completed_count += 1
                    if bool(args.progress):
                        _print_progress(
                            label="Shards",
                            completed=completed_count,
                            total=len(shard_jobs),
                            start_time=shard_start_time,
                        )
        nonzero = [result for result in run_results if int(result.get("returncode") or 0) != 0]
        if nonzero:
            raise SystemExit(f"Shard render failed; see logs: {[result['log_path'] for result in nonzero]}")

    if bool(args.progress):
        print("Collecting shard rankings...", flush=True)
    ranking_rows: List[Dict[str, Any]] = []
    shard_lookup: Dict[str, Dict[str, Any]] = {}
    for job in shard_jobs:
        shard_dir = Path(str(job["shard_dir"]))
        selection_summary_path = shard_dir / "postsplit_scene_selection.json"
        if not selection_summary_path.is_file():
            raise SystemExit(f"Missing shard selection summary: {selection_summary_path}")
        selection_summary = _read_json(selection_summary_path)
        shard_lookup[str(shard_dir)] = selection_summary
        for row in list(selection_summary.get("rows") or []):
            row_dict = dict(row)
            row_dict["source_shard_dir"] = str(shard_dir.resolve())
            row_dict["source_shard_index"] = int(job["shard_index"])
            ranking_rows.append(row_dict)

    ranking_rows = sorted(ranking_rows, key=_ranking_sort_key)
    selected_ranking_rows = ranking_rows[: int(args.num_selected_scenes)]

    if outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir.mkdir(parents=True, exist_ok=True)
    examples_root = outdir / "examples"
    examples_root.mkdir(parents=True, exist_ok=True)

    render_rows: List[Dict[str, Any]] = []
    prompt_manifest_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []
    merge_start_time = time.time()
    if bool(args.progress):
        print(
            f"Merging top {len(selected_ranking_rows)} selected scenes from {len(ranking_rows)} ranked candidates",
            flush=True,
        )

    for selection_rank, ranking_row in enumerate(selected_ranking_rows):
        example_id = _example_id_from_row(ranking_row)
        shard_dir = Path(str(ranking_row["source_shard_dir"]))
        src_example_dir = shard_dir / "examples" / example_id
        if not src_example_dir.is_dir():
            raise SystemExit(
                f"Missing rendered example for {example_id} in {src_example_dir}. "
                f"Increase --local-num-selected-scenes to preserve all ranked shard candidates."
            )
        dst_example_dir = examples_root / example_id
        _copy_example_dir(src_example_dir, dst_example_dir)
        payload = _rewrite_example_dir(dst_example_dir)
        payload["selection_rank"] = int(selection_rank)
        payload["selection_score"] = float(ranking_row.get("selection_score") or 0.0)
        _write_json(dst_example_dir / "render_metadata.json", payload)
        render_rows.append(payload)
        prompt_manifest_rows.extend(_prompt_manifest_rows_from_example(payload, example_dir=dst_example_dir))
        aggregate_rows.append(
            _aggregate_row_from_payload(
                payload=payload,
                ranking_row=ranking_row,
                example_dir=dst_example_dir,
                selection_rank=selection_rank,
            )
        )
        if bool(args.progress):
            _print_progress(
                label="Merge ",
                completed=selection_rank + 1,
                total=len(selected_ranking_rows),
                start_time=merge_start_time,
            )

    render_manifest_path = outdir / "postsplit_render_manifest.json"
    _write_json(
        render_manifest_path,
        {
            "path": str(args.path),
            "candidate_scenes_scanned": int(args.candidate_scenes),
            "scene_offset": int(args.scene_offset),
            "num_selected_scenes": int(len(render_rows)),
            "rows": render_rows,
        },
    )
    request_manifest_path = outdir / "postsplit_request_manifest.jsonl"
    write_jsonl(request_manifest_path, prompt_manifest_rows)
    selection_summary_path = outdir / "postsplit_scene_selection.json"
    _write_json(
        selection_summary_path,
        {
            "path": str(args.path),
            "candidate_scenes_requested": int(args.candidate_scenes),
            "num_candidate_scenes_kept": int(len(ranking_rows)),
            "num_selected_scenes": int(len(selected_ranking_rows)),
            "selection_metric": "choose best 3-alt set by sum(base GT disagreement scores) + alt_diversity_weight * sum(pairwise alt-alt diversity); keep gradients relative to GT",
            "min_gt_length_m": float(args.min_gt_length_m),
            "gt_relative_threshold_m": float(args.gt_relative_threshold_m),
            "alt_diversity_weight": float(args.alt_diversity_weight),
            "diversity_top_k": int(args.diversity_top_k),
            "include_off_route_paths": bool(args.include_off_route_paths),
            "gradient_display_reference": float(args.gradient_display_reference),
            "gradient_display_gamma": float(args.gradient_display_gamma),
            "rows": ranking_rows,
        },
    )
    aggregate_index_path = outdir / "postsplit_selected_scene_index.jsonl"
    write_jsonl(aggregate_index_path, aggregate_rows)

    summary = {
        "path": str(args.path),
        "candidate_scenes_requested": int(args.candidate_scenes),
        "num_shards": int(len(shard_jobs)),
        "shard_root": str(shard_root.resolve()),
        "local_num_selected_scenes": int(local_num_selected_scenes),
        "num_candidate_scenes_kept": int(len(ranking_rows)),
        "num_selected_scenes": int(len(selected_ranking_rows)),
        "model": str(args.model),
        "image_detail": str(args.image_detail),
        "min_gt_length_m": float(args.min_gt_length_m),
        "gt_relative_threshold_m": float(args.gt_relative_threshold_m),
        "alt_diversity_weight": float(args.alt_diversity_weight),
        "diversity_top_k": int(args.diversity_top_k),
        "include_off_route_paths": bool(args.include_off_route_paths),
        "gradient_display_reference": float(args.gradient_display_reference),
        "gradient_display_gamma": float(args.gradient_display_gamma),
        "render_manifest_json": str(render_manifest_path.resolve()),
        "request_manifest_jsonl": str(request_manifest_path.resolve()),
        "selection_summary_json": str(selection_summary_path.resolve()),
        "selected_scene_index_jsonl": str(aggregate_index_path.resolve()),
    }
    _write_json(outdir / "postsplit_render_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
