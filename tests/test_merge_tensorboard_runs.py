from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from counter_bmt_v2.training.merge_tensorboard import (
    dedupe_records,
    load_metric_records,
    merge_metric_runs,
    sort_records,
    write_record,
)


class _DummyWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.text: list[tuple[str, str, int]] = []

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self.scalars.append((str(tag), float(scalar_value), int(global_step)))

    def add_text(self, tag: str, text_string: str, global_step: int) -> None:
        self.text.append((str(tag), str(text_string), int(global_step)))

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class MergeTensorBoardRunsTests(unittest.TestCase):
    def test_load_metric_records_keeps_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"phase": "train", "step": 1, "lr": 1e-4, "metrics": {"total_loss": 5.0}}),
                        json.dumps({"phase": "eval", "step": 2, "metrics": {"total_loss": 4.0}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows = load_metric_records(run_dir)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["_line_index"], 0)
            self.assertEqual(rows[1]["_line_index"], 1)
            self.assertEqual(rows[0]["_run_dir"], str(run_dir))

    def test_dedupe_prefers_later_record(self) -> None:
        rows = [
            {"phase": "train", "step": 10, "metrics": {"total_loss": 5.0}, "_source_index": 0},
            {"phase": "train", "step": 10, "metrics": {"total_loss": 4.0}, "_source_index": 1},
            {"phase": "eval", "step": 10, "metrics": {"total_loss": 3.0}, "_source_index": 1},
        ]
        deduped = dedupe_records(rows)
        train_rows = [r for r in deduped if r["phase"] == "train"]
        self.assertEqual(len(train_rows), 1)
        self.assertEqual(train_rows[0]["metrics"]["total_loss"], 4.0)

    def test_sort_records_orders_by_step_then_phase(self) -> None:
        rows = [
            {"phase": "eval", "step": 10, "_source_index": 0, "_line_index": 1},
            {"phase": "train", "step": 9, "_source_index": 0, "_line_index": 0},
            {"phase": "train", "step": 10, "_source_index": 0, "_line_index": 0},
            {"phase": "final_eval", "step": 10, "_source_index": 0, "_line_index": 2},
        ]
        sorted_rows = sort_records(rows)
        self.assertEqual(
            [(r["step"], r["phase"]) for r in sorted_rows],
            [(9, "train"), (10, "train"), (10, "eval"), (10, "final_eval")],
        )

    def test_write_record_emits_train_lr_and_scalars(self) -> None:
        writer = _DummyWriter()
        write_record(
            writer,
            {
                "phase": "train",
                "step": 12,
                "lr": 3e-4,
                "metrics": {"total_loss": 1.5, "accuracy": 0.7},
            },
        )
        self.assertIn(("train/lr", 3e-4, 12), writer.scalars)
        self.assertIn(("train/total_loss", 1.5, 12), writer.scalars)
        self.assertIn(("train/accuracy", 0.7, 12), writer.scalars)

    def test_merge_metric_runs_writes_manifest_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "run_a"
            run_b = root / "run_b"
            out_dir = root / "merged"
            run_a.mkdir()
            run_b.mkdir()
            (run_a / "metrics.jsonl").write_text(
                json.dumps({"phase": "train", "step": 1, "lr": 1e-4, "metrics": {"total_loss": 5.0}}) + "\n",
                encoding="utf-8",
            )
            (run_b / "metrics.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"phase": "train", "step": 1, "lr": 2e-4, "metrics": {"total_loss": 4.0}}),
                        json.dumps({"phase": "train", "step": 2, "lr": 2e-4, "metrics": {"total_loss": 3.0}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            def _dummy_create_tb_writer(output_dir: Path, subdir: str = "tensorboard", **_: object) -> _DummyWriter:
                (Path(output_dir) / str(subdir)).mkdir(parents=True, exist_ok=True)
                return _DummyWriter()

            with mock.patch(
                "counter_bmt_v2.training.merge_tensorboard.create_tb_writer",
                side_effect=_dummy_create_tb_writer,
            ):
                manifest = merge_metric_runs([run_a, run_b], out_dir, overwrite=True)
            self.assertEqual(manifest["raw_record_count"], 3)
            self.assertEqual(manifest["merged_record_count"], 2)
            merged_rows = [json.loads(line) for line in (out_dir / "merged_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["step"] for row in merged_rows], [1, 2])
            self.assertEqual(merged_rows[0]["metrics"]["total_loss"], 4.0)
            self.assertTrue((out_dir / "merge_manifest.json").exists())
            self.assertTrue((out_dir / "tensorboard").exists())


if __name__ == "__main__":
    unittest.main()
