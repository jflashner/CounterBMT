from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    for path in (repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual import load_and_normalize_scenario, write_inspection_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize and inspect a ScenarioNet/MetaDrive scenario pickle.")
    parser.add_argument("--scenario-pkl", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    canonical = load_and_normalize_scenario(args.scenario_pkl)
    paths = write_inspection_artifacts(canonical, args.outdir)
    print(
        json.dumps(
            {
                "scenario_id": canonical.scenario_id,
                "outdir": str(Path(args.outdir).expanduser()),
                "artifacts": {key: str(value) for key, value in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
