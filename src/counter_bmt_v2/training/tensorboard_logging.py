"""TensorBoard helpers for CounterBMT v2 training loops."""

from __future__ import annotations

import inspect
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Dict


def _require_summary_writer():
    # Avoid TensorBoard pulling TensorFlow into the training environment.
    # This keeps JAX-only runs independent from TF ABI constraints.
    os.environ.setdefault("TENSORBOARD_NO_TF", "1")
    if os.environ.get("TENSORBOARD_NO_TF", "0") == "1":
        sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))

    # Preferred path: torch SummaryWriter.
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore

        return SummaryWriter
    except Exception:
        pass

    # Fallback path: native TensorBoard event writer (no torch dependency).
    try:
        from tensorboard.compat.proto.event_pb2 import Event  # type: ignore
        from tensorboard.compat.proto.summary_pb2 import Summary  # type: ignore
        from tensorboard.summary.writer.event_file_writer import EventFileWriter  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "TensorBoard logging is enabled but no writer backend is available. "
            "Install either `torch` (for torch.utils.tensorboard) or `tensorboard`, "
            "or run with --no-tensorboard."
        ) from exc

    class _NativeSummaryWriter:
        def __init__(self, log_dir: str, flush_secs: int = 30, **_: object) -> None:
            flush_s = max(1, int(flush_secs))
            params = {}
            try:
                params = dict(inspect.signature(EventFileWriter.__init__).parameters)
            except Exception:
                params = {}

            kwargs = {}
            if "logdir" in params:
                kwargs["logdir"] = str(log_dir)
            elif "log_dir" in params:
                kwargs["log_dir"] = str(log_dir)
            if "max_queue" in params:
                kwargs["max_queue"] = 1000
            elif "max_queue_size" in params:
                kwargs["max_queue_size"] = 1000
            if "flush_secs" in params:
                kwargs["flush_secs"] = flush_s
            if "filename_suffix" in params:
                kwargs["filename_suffix"] = ""

            # Try signature-aware kwargs first, then degrade to positional variants
            # for older TensorBoard releases.
            try:
                self._writer = EventFileWriter(**kwargs)
            except TypeError:
                try:
                    self._writer = EventFileWriter(str(log_dir), 1000, flush_s, "")
                except TypeError:
                    try:
                        self._writer = EventFileWriter(str(log_dir), 1000, flush_s)
                    except TypeError:
                        self._writer = EventFileWriter(str(log_dir))

        def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
            summary = Summary(value=[Summary.Value(tag=str(tag), simple_value=float(scalar_value))])
            event = Event(wall_time=float(time.time()), step=int(global_step), summary=summary)
            self._writer.add_event(event)

        def add_text(self, tag: str, text_string: str, global_step: int) -> None:
            # Keep fallback lightweight; write a scalar marker for text payload size.
            text_len = float(len(str(text_string)))
            self.add_scalar(f"{str(tag).rstrip('/')}/_text_len", text_len, int(global_step))

        def flush(self) -> None:
            self._writer.flush()

        def close(self) -> None:
            self._writer.close()

    return _NativeSummaryWriter


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
