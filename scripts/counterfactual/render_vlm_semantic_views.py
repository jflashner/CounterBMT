from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    for path in (repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual.path_eval_bundle import write_json
from bmt.counterfactual.vlm_semantics.audit import load_bundle_selected_examples, load_materialized_manifest_examples
from bmt.counterfactual.vlm_semantics.render import render_vlm_semantic_views


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render VLM-ready semantic views from the local audit bundle or a materialized manifest.")
    parser.add_argument("--bundle-root", type=str, default="")
    parser.add_argument("--selected-manifest", type=str, default="")
    parser.add_argument("--materialized-manifest", type=str, default="")
    parser.add_argument("--path-index", type=str, default="")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--max-examples", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.bundle_root:
        loaded = load_bundle_selected_examples(
            bundle_root=args.bundle_root,
            selected_manifest_path=args.selected_manifest or None,
            max_examples=int(args.max_examples),
        )
    else:
        if not args.materialized_manifest or not args.path_index:
            raise ValueError("Either --bundle-root or both --materialized-manifest and --path-index are required")
        loaded = load_materialized_manifest_examples(
            materialized_manifest_path=args.materialized_manifest,
            path_index_path=args.path_index,
            max_examples=int(args.max_examples),
        )

    render_manifest = render_vlm_semantic_views(records=loaded["selected_examples"], outdir=outdir)
    summary = {
        "mode": "bundle" if args.bundle_root else "materialized_manifest",
        "bundle_root": (str(Path(args.bundle_root).expanduser()) if args.bundle_root else None),
        "selected_manifest_path": loaded.get("selected_manifest_path"),
        "num_examples": int(render_manifest["num_examples"]),
        "num_plot_failures": int(render_manifest["num_plot_failures"]),
        "vlm_render_manifest_json": str((outdir / "vlm_render_manifest.json").resolve()),
        "plot_failures_json": str((outdir / "plot_failures.json").resolve()),
        "corrected_visuals_vlm_dir": str((outdir / "corrected_visuals_vlm").resolve()),
    }
    if loaded.get("bundle_inventory"):
        write_json(outdir / "bundle_inventory.json", loaded["bundle_inventory"])
    write_json(outdir / "render_summary.json", summary)
    print(summary["vlm_render_manifest_json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
