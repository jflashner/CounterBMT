from __future__ import annotations

import unittest
from types import SimpleNamespace

from counter_bmt_v2.cli.train_nnx_bmt import _resolve_runtime_defaults as resolve_supervised_runtime
from counter_bmt_v2.cli.train_nnx_bmt_dag_latent import _resolve_runtime_defaults as resolve_dag_runtime
from counter_bmt_v2.trajectory_jax import get_runtime_preset


def _base_runtime_args(runtime_preset: str = "legacy_midgpt_recipe") -> SimpleNamespace:
    return SimpleNamespace(
        runtime_preset=runtime_preset,
        model_preset=None,
        tokenizer_mode=None,
        lr=None,
        warmup_steps=None,
        weight_decay=None,
        grad_clip=None,
        skip_steps=None,
        lr_schedule_mode=None,
        epochs=3,
        mode="mixed",
        reverse_prob=0.5,
        collate_padding_mode=None,
    )


class RuntimePresetTests(unittest.TestCase):
    def test_legacy_midgpt_recipe_matches_expected_values(self) -> None:
        preset = get_runtime_preset("legacy_midgpt_recipe")

        self.assertEqual(preset["model_preset"], "midgpt_parity")
        self.assertEqual(preset["tokenizer_mode"], "adv_bmt_parity")
        self.assertEqual(preset["warmup_steps"], 2000)
        self.assertEqual(preset["lr_schedule_mode"], "legacy_cosine_zero")
        self.assertEqual(preset["num_epochs"], 30)
        self.assertEqual(preset["mode"], "forward")
        self.assertEqual(preset["reverse_probability"], 0.0)
        self.assertEqual(preset["collate_padding_mode"], "batch_local")

    def test_supervised_cli_applies_legacy_midgpt_recipe_defaults(self) -> None:
        resolved = resolve_supervised_runtime(_base_runtime_args(), provided_flags=set())

        self.assertEqual(resolved["num_epochs"], 30)
        self.assertEqual(resolved["mode"], "forward")
        self.assertEqual(resolved["reverse_probability"], 0.0)
        self.assertEqual(resolved["collate_padding_mode"], "batch_local")

    def test_supervised_cli_explicit_flags_override_recipe(self) -> None:
        args = _base_runtime_args()
        args.epochs = 12
        args.mode = "mixed"
        args.reverse_prob = 0.25
        args.collate_padding_mode = "fixed"
        resolved = resolve_supervised_runtime(
            args,
            provided_flags={"--epochs", "--mode", "--reverse-prob", "--collate-padding-mode"},
        )

        self.assertEqual(resolved["num_epochs"], 12)
        self.assertEqual(resolved["mode"], "mixed")
        self.assertEqual(resolved["reverse_probability"], 0.25)
        self.assertEqual(resolved["collate_padding_mode"], "fixed")

    def test_dag_cli_receives_same_recipe_direction_defaults(self) -> None:
        resolved = resolve_dag_runtime(_base_runtime_args(), provided_flags=set())

        self.assertEqual(resolved["num_epochs"], 30)
        self.assertEqual(resolved["mode"], "forward")
        self.assertEqual(resolved["reverse_probability"], 0.0)
        self.assertEqual(resolved["collate_padding_mode"], "batch_local")

    def test_speed_recipe_uses_bucketed_padding(self) -> None:
        preset = get_runtime_preset("legacy_midgpt_speed_recipe")
        self.assertEqual(preset["model_preset"], "midgpt_parity")
        self.assertEqual(preset["tokenizer_mode"], "adv_bmt_parity")
        self.assertEqual(preset["collate_padding_mode"], "bucketed")

        resolved = resolve_supervised_runtime(
            _base_runtime_args(runtime_preset="legacy_midgpt_speed_recipe"),
            provided_flags=set(),
        )
        self.assertEqual(resolved["collate_padding_mode"], "bucketed")


if __name__ == "__main__":
    unittest.main()
