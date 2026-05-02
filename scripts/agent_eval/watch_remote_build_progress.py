from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List


REMOTE_PY = r"""
import json
import os
import pathlib
import re
import subprocess

label = os.environ["LABEL"]
screen_name = os.environ["SCREEN_NAME"]
root_text = os.environ["ROOT_DIR"]
target_text = os.environ["TARGET_SCENES"]
root = pathlib.Path(root_text)
target = int(target_text)

screen_proc = subprocess.run(["screen", "-ls"], capture_output=True, text=True)
screen_stdout = screen_proc.stdout or ""
screen_live = screen_name in screen_stdout

ps_proc = subprocess.run("ps -u $USER -o cmd=", shell=True, capture_output=True, text=True)
proc_lines = [line.strip() for line in (ps_proc.stdout or "").splitlines()]

scene_analysis = root / "scene_analysis"
natural_dir = root / "natural_scenarios"
adversarial_dir = root / "adversarial_scenarios"

builder_summary_path = root / "builder_summary.json"
builder_summary = None
if builder_summary_path.exists():
    try:
        builder_summary = json.loads(builder_summary_path.read_text())
    except Exception:
        builder_summary = None

train_log = root / "train.log"
checkpoint_files = sorted(root.glob("seed_*_steps.zip"))
is_td3 = train_log.exists() or bool(checkpoint_files) or (root / "evaluations.npz").exists()

if is_td3:
    proc_count = sum(
        1
        for line in proc_lines
        if "train_td3.py" in line
        and root_text in line
        and "SCREEN -S" not in line
        and "bash -lc" not in line
    )
    log_tail = ""
    if train_log.exists():
        tail_proc = subprocess.run(["tail", "-n", "250", str(train_log)], capture_output=True, text=True)
        log_tail = tail_proc.stdout or ""

    def _find_last(pattern: str):
        matches = re.findall(pattern, log_tail)
        return matches[-1] if matches else None

    latest_step_text = _find_last(r"\|\s+total_timesteps\s+\|\s+([0-9]+)\s+\|")
    latest_reward_text = _find_last(r"\|\s+ep_rew_mean\s+\|\s+([-+0-9.eE]+)\s+\|")
    latest_length_text = _find_last(r"\|\s+ep_len_mean\s+\|\s+([-+0-9.eE]+)\s+\|")
    latest_fps_text = _find_last(r"\|\s+fps\s+\|\s+([-+0-9.eE]+)\s+\|")
    latest_elapsed_text = _find_last(r"\|\s+time_elapsed\s+\|\s+([0-9]+)\s+\|")

    checkpoint_steps = []
    for path in checkpoint_files:
        match = re.search(r"seed_\d+_(\d+)_steps\.zip$", path.name)
        if match:
            checkpoint_steps.append(int(match.group(1)))

    latest_step = int(latest_step_text) if latest_step_text else 0
    latest_checkpoint = max(checkpoint_steps) if checkpoint_steps else 0
    status = "running" if (screen_live or proc_count > 0) else "stopped"
    if target > 0 and latest_step >= target and proc_count == 0:
        status = "finished"

    payload = {
        "label": label,
        "screen": screen_name,
        "root": root_text,
        "target": target,
        "screen_live": bool(screen_live),
        "proc_count": int(proc_count),
        "status": status,
        "mode": "td3",
        "latest_step": int(latest_step),
        "latest_checkpoint": int(latest_checkpoint),
        "checkpoint_count": int(len(checkpoint_steps)),
        "best_model": bool((root / "best_model.zip").exists()),
        "has_evaluations": bool((root / "evaluations.npz").exists()),
        "log_exists": bool(train_log.exists()),
        "ep_rew_mean": float(latest_reward_text) if latest_reward_text else None,
        "ep_len_mean": float(latest_length_text) if latest_length_text else None,
        "fps": float(latest_fps_text) if latest_fps_text else None,
        "time_elapsed": int(latest_elapsed_text) if latest_elapsed_text else None,
    }
else:
    proc_count = sum(
        1
        for line in proc_lines
        if "build_advbmt_table4_dataset.py" in line
        and root_text in line
        and "SCREEN -S" not in line
        and "bash -lc" not in line
    )
    summary_count = sum(1 for _ in scene_analysis.rglob("scene_summary.json")) if scene_analysis.exists() else 0
    skip_count = sum(1 for _ in scene_analysis.rglob("scene_skip.json")) if scene_analysis.exists() else 0
    natural_count = sum(1 for _ in natural_dir.glob("sd_*.pkl")) if natural_dir.exists() else 0
    adversarial_count = sum(1 for _ in adversarial_dir.glob("sd_*.pkl")) if adversarial_dir.exists() else 0

    status = "running" if (screen_live or proc_count > 0) else "stopped"
    if builder_summary is not None and proc_count == 0:
        status = "finished"

    payload = {
        "label": label,
        "screen": screen_name,
        "root": root_text,
        "target": target,
        "screen_live": bool(screen_live),
        "proc_count": int(proc_count),
        "completed": int(summary_count),
        "skipped": int(skip_count),
        "natural_count": int(natural_count),
        "adversarial_count": int(adversarial_count),
        "status": status,
        "builder_summary": builder_summary,
        "mode": "build",
    }

print(json.dumps(payload))
"""


@dataclass(frozen=True)
class RunSpec:
    label: str
    host: str
    screen: str
    root: str
    target: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch remote Adv-BMT/CounterBMT bank builds or TD3 runs live from the local terminal."
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help=(
            "Run spec in the form "
            "'label|host|screen_name|remote_root|target_total'. "
            "May be provided multiple times."
        ),
    )
    parser.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Print a single snapshot and exit.")
    return parser.parse_args()


def default_runs() -> List[RunSpec]:
    return [
        RunSpec(
            label="advbmt-td3",
            host="zhoulab-1.cs.vt.edu",
            screen="td3_advbmt_paperfaithful_seed0",
            root="/data/home/grads/jflashner/CounterBMT_run/logs/td3_table4_runs/td3_table4_advbmt_paperfaithful_train476_eval_natural_seed0",
            target=1_000_000,
        ),
    ]


def parse_run_spec(raw: str) -> RunSpec:
    parts = raw.split("|")
    if len(parts) != 5:
        raise ValueError(
            f"Invalid --run spec {raw!r}. Expected "
            "'label|host|screen_name|remote_root|target_total'."
        )
    label, host, screen, root, target = parts
    return RunSpec(
        label=label.strip(),
        host=host.strip(),
        screen=screen.strip(),
        root=root.strip(),
        target=int(target),
    )


def fetch_status(spec: RunSpec) -> Dict[str, Any]:
    cmd = [
        "ssh",
        spec.host,
        "env",
        f"LABEL={spec.label}",
        f"SCREEN_NAME={spec.screen}",
        f"ROOT_DIR={spec.root}",
        f"TARGET_SCENES={spec.target}",
        "python3",
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=REMOTE_PY,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "label": spec.label,
            "host": spec.host,
            "status": "ssh-timeout",
            "error": "SSH timed out",
            "target": spec.target,
        }
    if proc.returncode != 0:
        return {
            "label": spec.label,
            "host": spec.host,
            "status": "ssh-error",
            "error": (proc.stderr or proc.stdout or f"ssh exited {proc.returncode}").strip(),
            "target": spec.target,
        }
    try:
        payload = json.loads(proc.stdout.strip())
    except Exception as exc:
        return {
            "label": spec.label,
            "host": spec.host,
            "status": "parse-error",
            "error": f"Could not parse remote payload: {exc}",
            "target": spec.target,
        }
    payload["host"] = spec.host
    return payload


def progress_bar(current: int, target: int, *, width: int = 28) -> str:
    if target <= 0:
        return "[" + ("?" * width) + "]"
    bounded = max(0, min(int(current), int(target)))
    filled = int(round(width * (bounded / float(target))))
    filled = max(0, min(width, filled))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def format_row(status: Dict[str, Any]) -> str:
    label = str(status.get("label", "unknown"))
    state = str(status.get("status", "unknown"))
    target = int(status.get("target", 0) or 0)
    proc_count = int(status.get("proc_count", 0) or 0)
    screen_live = "yes" if status.get("screen_live") else "no"
    mode = str(status.get("mode", "build"))

    if mode == "td3":
        latest_step = int(status.get("latest_step", 0) or 0)
        latest_checkpoint = int(status.get("latest_checkpoint", 0) or 0)
        checkpoint_count = int(status.get("checkpoint_count", 0) or 0)
        percent = 0.0 if target <= 0 else (100.0 * latest_step / float(target))
        ep_rew_mean = status.get("ep_rew_mean")
        ep_len_mean = status.get("ep_len_mean")
        fps = status.get("fps")
        rew_text = "n/a" if ep_rew_mean is None else f"{float(ep_rew_mean):.1f}"
        len_text = "n/a" if ep_len_mean is None else f"{float(ep_len_mean):.1f}"
        fps_text = "n/a" if fps is None else f"{float(fps):.0f}"
        return (
            f"{label:<14}  {state:<10}  td3    "
            f"{progress_bar(latest_step, target)}  "
            f"{latest_step:>7}/{target:<7} ({percent:>5.1f}%)  "
            f"rew={rew_text:<6}  len={len_text:<6}  fps={fps_text:<4}  "
            f"ckpt={checkpoint_count:<2}  last={latest_checkpoint:<7}  "
            f"screen={screen_live:<3}  proc={proc_count:<2}"
        )

    completed = int(status.get("completed", 0) or 0)
    skipped = int(status.get("skipped", 0) or 0)
    natural_count = int(status.get("natural_count", 0) or 0)
    adversarial_count = int(status.get("adversarial_count", 0) or 0)
    percent = 0.0 if target <= 0 else (100.0 * completed / float(target))
    return (
        f"{label:<14}  {state:<10}  build  "
        f"{progress_bar(completed, target)}  "
        f"{completed:>4}/{target:<4} ({percent:>5.1f}%)  "
        f"skip={skipped:<3}  nat={natural_count:<4}  adv={adversarial_count:<4}  "
        f"screen={screen_live:<3}  proc={proc_count:<2}"
    )


def render(statuses: List[Dict[str, Any]], *, interval: float) -> str:
    width = shutil.get_terminal_size((120, 30)).columns
    lines = []
    lines.append("Remote Run Watch")
    lines.append("=" * min(width, 120))
    lines.append(f"Refresh every {interval:.1f}s")
    lines.append("")
    for status in statuses:
        lines.append(format_row(status))
        error = status.get("error")
        if error:
            lines.append(f"  error: {error}")
    lines.append("")
    lines.append(
        "Build columns: completed/target, skips, natural .pkl count, adversarial .pkl count, screen live, python proc count"
    )
    lines.append(
        "TD3 columns: step/target, episode reward mean, episode length mean, fps, checkpoint count, latest checkpoint step, screen live, python proc count"
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    specs = [parse_run_spec(raw) for raw in args.run] if args.run else default_runs()
    while True:
        statuses = [fetch_status(spec) for spec in specs]
        output = render(statuses, interval=float(args.interval))
        if sys.stdout.isatty():
            sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(output + "\n")
        sys.stdout.flush()
        if args.once:
            return 0
        time.sleep(max(float(args.interval), 0.5))


if __name__ == "__main__":
    raise SystemExit(main())
