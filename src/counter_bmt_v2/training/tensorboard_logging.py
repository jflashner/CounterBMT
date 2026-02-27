"""TensorBoard helpers for CounterBMT v2 training loops."""

from __future__ import annotations

import math
import os
import sys
import types
from pathlib import Path
from typing import Dict


def _require_summary_writer():
    try:
        # Avoid TensorBoard pulling TensorFlow into the training environment.
        # This keeps JAX/PyTorch-only runs independent from TF ABI constraints.
        os.environ.setdefault("TENSORBOARD_NO_TF", "1")
        if os.environ.get("TENSORBOARD_NO_TF", "0") == "1":
            # Some TensorBoard distributions expect this module to exist when
            # running in "no TensorFlow" mode; create a shim if absent.
            sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "TensorBoard logging is enabled but `torch.utils.tensorboard` is unavailable. "
            "Install TensorBoard support (e.g. `pip install tensorboard`) or run with --no-tensorboard."
        ) from exc
    return SummaryWriter


def create_tb_writer(
    output_dir: Path,
    subdir: str = "tensorboard",
    enabled: bool = True,
    flush_secs: int = 30,
):
    """Create and return a SummaryWriter (or None when disabled)."""
    if not enabled:
        return None
    SummaryWriter = _require_summary_writer()
    log_dir = Path(output_dir) / str(subdir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir), flush_secs=max(1, int(flush_secs)))


def _is_finite_number(value: object) -> bool:
    try:
        v = float(value)
    except Exception:
        return False
    return math.isfinite(v)


def tb_write_scalar(writer, tag: str, value: object, step: int) -> None:
    """Write a scalar if finite; silently skip invalid values."""
    if writer is None:
        return
    if not _is_finite_number(value):
        return
    writer.add_scalar(str(tag), float(value), int(step))


def tb_write_scalars(writer, prefix: str, metrics: Dict[str, float], step: int) -> None:
    """Write a dict of scalar metrics under a prefix."""
    if writer is None:
        return
    pre = str(prefix).strip("/")
    for k, v in metrics.items():
        tag = f"{pre}/{k}" if pre else str(k)
        tb_write_scalar(writer, tag, v, step)


def tb_write_text(writer, tag: str, text: str, step: int = 0) -> None:
    """Write text entry to TensorBoard."""
    if writer is None:
        return
    writer.add_text(str(tag), str(text), int(step))


def tb_close(writer) -> None:
    if writer is None:
        return
    try:
        writer.flush()
    except Exception:
        pass
    try:
        writer.close()
    except Exception:
        pass
