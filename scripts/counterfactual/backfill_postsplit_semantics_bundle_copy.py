from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an existing postsplit semantics bundle, rerender the copy in place, "
            "and relabel the copy with explicit VLM settings."
        )
    )
    parser.add_argument("--bundle-root", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="")
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--image-detail", type=str, default="original", choices=("low", "high", "original", "auto"))
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-completion-tokens", type=int, default=1000)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-sleep-s", type=float, default=3.0)
    parser.add_argument("--batch-mode", action="store_true")
    parser.add_argument("--submit-batch", action="store_true")
    parser.add_argument("--wait-for-batch", action="store_true")
    parser.add_argument("--completion-window", type=str, default="24h")
    parser.add_argument("--batch-max-bytes", type=int, default=180_000_000)
    parser.add_argument("--batch-max-requests", type=int, default=50_000)
    parser.add_argument("--batch-shard-indices", type=str, default="")
    parser.add_argument("--copy-only", action="store_true")
    parser.add_argument("--skip-rerender", action="store_true")
    parser.add_argument("--skip-relabel", action="store_true")
    return parser.parse_args()


def _default_outdir(bundle_root: Path) -> Path:
    return bundle_root.parent / f"{bundle_root.name}_vlmrisk_gpt54_original_copy"


def _run_checked(cmd: List[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    bundle_root = Path(args.bundle_root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if str(args.outdir).strip() else _default_outdir(bundle_root)

    if not bundle_root.is_dir():
        raise SystemExit(f"--bundle-root does not exist: {bundle_root}")
    if outdir.exists():
        raise SystemExit(f"--outdir already exists; refusing to overwrite copy target: {outdir}")

    shutil.copytree(bundle_root, outdir)

    rerender_cmd = [
        sys.executable,
        "scripts/counterfactual/rerender_postsplit_bundle_from_selection.py",
        "--bundle-root",
        str(outdir),
        "--outdir",
        str(outdir),
        "--model",
        str(args.model),
        "--image-detail",
        str(args.image_detail),
        "--progress-every",
        str(int(args.progress_every)),
        "--show-traffic-lights",
    ]

    relabel_cmd = [
        sys.executable,
        "scripts/counterfactual/label_existing_postsplit_semantics_bundle.py",
        "--bundle-root",
        str(outdir),
        "--model",
        str(args.model),
        "--image-detail",
        str(args.image_detail),
        "--max-examples",
        str(int(args.max_examples)),
        "--num-workers",
        str(int(args.num_workers)),
        "--max-completion-tokens",
        str(int(args.max_completion_tokens)),
        "--max-retries",
        str(int(args.max_retries)),
        "--retry-sleep-s",
        str(float(args.retry_sleep_s)),
    ]
    if bool(args.batch_mode):
        relabel_cmd.append("--batch-mode")
    if bool(args.submit_batch):
        relabel_cmd.append("--submit-batch")
    if bool(args.wait_for_batch):
        relabel_cmd.append("--wait-for-batch")
    if str(args.completion_window).strip():
        relabel_cmd.extend(["--completion-window", str(args.completion_window)])
    if int(args.batch_max_bytes) > 0:
        relabel_cmd.extend(["--batch-max-bytes", str(int(args.batch_max_bytes))])
    if int(args.batch_max_requests) > 0:
        relabel_cmd.extend(["--batch-max-requests", str(int(args.batch_max_requests))])
    if str(args.batch_shard_indices).strip():
        relabel_cmd.extend(["--batch-shard-indices", str(args.batch_shard_indices)])

    manifest = {
        "source_bundle_root": str(bundle_root),
        "copied_bundle_root": str(outdir),
        "model": str(args.model),
        "image_detail": str(args.image_detail),
        "rerender_cmd": rerender_cmd,
        "relabel_cmd": relabel_cmd,
        "show_traffic_lights": True,
    }
    _write_manifest(outdir / "vlm_risk_backfill_copy_plan.json", manifest)

    if bool(args.copy_only):
        return 0
    if not bool(args.skip_rerender):
        _run_checked(rerender_cmd, cwd=repo_root)
    if not bool(args.skip_relabel):
        _run_checked(relabel_cmd, cwd=repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
