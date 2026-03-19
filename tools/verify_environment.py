#!/usr/bin/env python3
"""Verify CounterBMT environments against pinned requirements files.

This script checks:
1. Python version
2. Editable paths in requirements files
3. Exact package version matches for pinned entries (name==version)
4. Critical runtime imports for the selected profile

Profiles:
- v2: requirements.txt
- mac-cpu: requirements-mac-cpu.txt
- mac-parity: requirements-mac-parity-tools.txt
- legacy: requirements-legacy.txt
- legacy-train-linux-cu121: requirements-legacy-train-linux-cu121.txt
- legacy-train-cpu: requirements-legacy-train-cpu.txt
- legacy-waymo-eval: requirements-legacy-waymo-eval.txt
- full: requirements-installed-freeze.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib import import_module, metadata, util
from pathlib import Path
from typing import Dict, List, Set, Tuple

REQ_FILES = {
    "v2": "requirements.txt",
    "mac-cpu": "requirements-mac-cpu.txt",
    "mac-parity": "requirements-mac-parity-tools.txt",
    "legacy": "requirements-legacy.txt",
    "legacy-train-linux-cu121": "requirements-legacy-train-linux-cu121.txt",
    "legacy-train-cpu": "requirements-legacy-train-cpu.txt",
    "legacy-waymo-eval": "requirements-legacy-waymo-eval.txt",
    "full": "requirements-installed-freeze.txt",
}

CRITICAL_IMPORTS = {
    "v2": [
        "counter_bmt_v2",
        "counter_bmt_v2.data",
        "numpy",
        "jax",
        "flax",
        "optax",
        "openai",
        "matplotlib",
        "metadrive",
        "scenarionet",
    ],
    "mac-cpu": [
        "counter_bmt_v2",
        "counter_bmt_v2.data",
        "numpy",
        "jax",
        "flax",
        "optax",
        "openai",
        "matplotlib",
        "tensorboard",
        "pytest",
    ],
    "mac-parity": [
        "counter_bmt_v2",
        "counter_bmt_v2.data",
        "numpy",
        "jax",
        "flax",
        "optax",
        "openai",
        "matplotlib",
        "tensorboard",
        "pytest",
        "torch",
        "easydict",
        "shapely",
    ],
    "legacy": ["torch", "torchvision", "torch_geometric", "hydra", "lightning"],
    "legacy-train-linux-cu121": [
        "bmt",
        "torch",
        "torchvision",
        "torch_geometric",
        "hydra",
        "lightning",
        "wandb",
        "shapely",
        "metadrive",
        "scenarionet",
    ],
    "legacy-train-cpu": [
        "bmt",
        "torch",
        "torchvision",
        "torch_geometric",
        "hydra",
        "lightning",
        "wandb",
        "shapely",
        "metadrive",
        "scenarionet",
    ],
    "legacy-waymo-eval": [
        "tensorflow",
        "tensorflow_addons",
        "tensorflow_datasets",
        "tensorflow_probability",
        "waymo_open_dataset",
    ],
    "full": ["numpy", "jax", "torch", "tensorflow"],
}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def parse_requirements(path: Path, seen: Set[Path] | None = None) -> Tuple[Dict[str, str], List[str]]:
    if seen is None:
        seen = set()
    path = path.resolve()
    if path in seen:
        return {}, []
    seen.add(path)

    pinned: Dict[str, str] = {}
    editable_paths: List[str] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("-r ") or line.startswith("--requirement "):
            _, include = line.split(None, 1)
            include_path = (path.parent / include.strip()).resolve()
            child_pinned, child_editables = parse_requirements(include_path, seen=seen)
            pinned.update(child_pinned)
            editable_paths.extend(child_editables)
            continue

        if line.startswith("-e "):
            editable_paths.append(line[3:].strip())
            continue

        if "==" in line and not line.startswith("--"):
            name, version = line.split("==", 1)
            name = _normalize_name(name)
            version = version.strip()
            if name and version:
                pinned[name] = version

    return pinned, editable_paths


def check_editables(repo_root: Path, editable_paths: List[str]) -> List[str]:
    problems: List[str] = []
    for p in editable_paths:
        # Skip URL editables; only validate local paths.
        if re.match(r"^[a-zA-Z]+://", p):
            continue
        full = (repo_root / p).resolve()
        if not full.exists():
            problems.append(f"Missing editable path: {p} -> {full}")
    return problems


def check_versions(pinned: Dict[str, str]) -> Tuple[List[str], List[str]]:
    missing: List[str] = []
    mismatched: List[str] = []

    for name, expected in sorted(pinned.items()):
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(f"{name}=={expected}")
            continue

        if installed != expected:
            mismatched.append(f"{name}: installed={installed}, expected={expected}")

    return missing, mismatched


def check_imports(modules: List[str], mode: str = "spec") -> List[str]:
    failures: List[str] = []
    for mod in modules:
        try:
            if mode == "import":
                import_module(mod)
            else:
                if util.find_spec(mod) is None:
                    raise ModuleNotFoundError(f"No module named '{mod}'")
        except Exception as exc:  # pragma: no cover
            failures.append(f"{mod}: {exc}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify pinned CounterBMT environment")
    parser.add_argument(
        "--profile",
        choices=[
            "v2",
            "mac-cpu",
            "mac-parity",
            "legacy",
            "legacy-train-linux-cu121",
            "legacy-train-cpu",
            "legacy-waymo-eval",
            "full",
        ],
        default="v2",
    )
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--min-python", type=str, default="3.10")
    parser.add_argument("--import-mode", choices=["spec", "import"], default="spec")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    req_path = repo_root / REQ_FILES[args.profile]

    if not req_path.exists():
        print(f"ERROR: requirements file not found: {req_path}")
        return 2

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    min_major, min_minor = [int(v) for v in args.min_python.split(".", 1)]
    if (sys.version_info.major, sys.version_info.minor) < (min_major, min_minor):
        print(f"ERROR: Python {py_ver} is below minimum {args.min_python}")
        return 2

    pinned, editables = parse_requirements(req_path)
    edit_problems = check_editables(repo_root, editables)
    missing, mismatched = check_versions(pinned)
    import_failures = check_imports(CRITICAL_IMPORTS[args.profile], mode=args.import_mode)

    print(f"Profile: {args.profile}")
    print(f"Requirements: {req_path}")
    print(f"Python: {py_ver}")
    print(f"Pinned packages checked: {len(pinned)}")

    ok = True

    if edit_problems:
        ok = False
        print("\nEditable path issues:")
        for p in edit_problems:
            print(f"  - {p}")

    if missing:
        ok = False
        print("\nMissing packages:")
        for m in missing:
            print(f"  - {m}")

    if mismatched:
        ok = False
        print("\nVersion mismatches:")
        for m in mismatched:
            print(f"  - {m}")

    if import_failures:
        ok = False
        print("\nCritical import failures:")
        for f in import_failures:
            print(f"  - {f}")

    if ok:
        print("\nEnvironment verification PASSED.")
        return 0

    print("\nEnvironment verification FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
