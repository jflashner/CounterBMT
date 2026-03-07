from __future__ import annotations

import unittest

import numpy as np

from counter_bmt_v2.training.supervised_dag_latent import (
    _clone_with_null_dag_inputs,
    _clone_with_shuffled_dag_inputs,
    _compute_dag_alignment_metrics,
)
from counter_bmt_v2.trajectory_jax import NNXBMTConfig, NNXBidirectionalMotionTransformer
from counter_bmt_v2.trajectory_jax.dag_gnn_nnx import HAS_NNX, NNXDAGEncoderConfig, NNXDAGGraphEncoder


def _sample_dag_inputs() -> dict:
    return {
        "prev_token_ids": np.zeros((2, 3, 1), dtype=np.int32),
        "dag_node_feat": np.arange(2 * 4 * 24, dtype=np.float32).reshape(2, 4, 24),
        "dag_node_mask": np.asarray(
            [
                [True, True, False, False],
                [True, False, False, False],
            ],
            dtype=bool,
        ),
        "dag_edge_src": np.asarray([[0, 1, 0], [0, 0, 0]], dtype=np.int32),
        "dag_edge_dst": np.asarray([[1, 0, 0], [0, 0, 0]], dtype=np.int32),
        "dag_edge_feat": np.arange(2 * 3 * 8, dtype=np.float32).reshape(2, 3, 8),
        "dag_edge_mask": np.asarray([[True, True, False], [False, False, False]], dtype=bool),
        "dag_global_feat": np.arange(2 * 4, dtype=np.float32).reshape(2, 4),
    }


class DAGLatentAlignmentTests(unittest.TestCase):
    def test_null_dag_inputs_zero_only_dag_tensors(self) -> None:
        inputs = _sample_dag_inputs()
        out = _clone_with_null_dag_inputs(inputs)

        self.assertTrue(np.array_equal(out["prev_token_ids"], inputs["prev_token_ids"]))
        self.assertFalse(np.any(np.asarray(out["dag_node_mask"], dtype=bool)))
        self.assertFalse(np.any(np.asarray(out["dag_edge_mask"], dtype=bool)))
        self.assertTrue(np.allclose(np.asarray(out["dag_node_feat"]), 0.0))
        self.assertTrue(np.allclose(np.asarray(out["dag_edge_feat"]), 0.0))
        self.assertTrue(np.allclose(np.asarray(out["dag_global_feat"]), 0.0))

    def test_shuffle_dag_inputs_rotates_batch_consistently(self) -> None:
        inputs = _sample_dag_inputs()
        out, available = _clone_with_shuffled_dag_inputs(inputs)

        self.assertTrue(available)
        self.assertTrue(np.array_equal(np.asarray(out["dag_node_feat"])[0], np.asarray(inputs["dag_node_feat"])[1]))
        self.assertTrue(np.array_equal(np.asarray(out["dag_node_feat"])[1], np.asarray(inputs["dag_node_feat"])[0]))
        self.assertTrue(np.array_equal(np.asarray(out["dag_edge_feat"])[0], np.asarray(inputs["dag_edge_feat"])[1]))
        self.assertTrue(np.array_equal(np.asarray(out["dag_global_feat"])[1], np.asarray(inputs["dag_global_feat"])[0]))

    def test_alignment_metrics_report_positive_gain_when_dag_helps(self) -> None:
        metrics = _compute_dag_alignment_metrics(
            with_dag={"total_loss": 1.0, "accuracy": 0.70},
            without_dag={"total_loss": 1.4, "accuracy": 0.60},
            shuffled_dag={"total_loss": 1.2, "accuracy": 0.65},
            dag_present_rate=1.0,
            shuffle_available=True,
        )
        self.assertGreater(metrics["dag_alignment/loss_gain_vs_without_dag"], 0.0)
        self.assertGreater(metrics["dag_alignment/loss_gain_vs_shuffled_dag"], 0.0)
        self.assertGreater(metrics["dag_alignment/accuracy_gain_vs_without_dag"], 0.0)
        self.assertGreater(metrics["dag_alignment/accuracy_gain_vs_shuffled_dag"], 0.0)

    @unittest.skipUnless(HAS_NNX, "flax.nnx unavailable")
    def test_nnx_dag_encoder_forward_shapes(self) -> None:
        import jax.numpy as jnp
        from flax import nnx

        cfg = NNXDAGEncoderConfig(
            enabled=True,
            d_node_in=24,
            d_edge_in=8,
            d_hidden=16,
            n_layers=2,
            max_nodes=4,
            max_edges=3,
        )
        enc = NNXDAGGraphEncoder(cfg, rngs=nnx.Rngs(0))
        inputs = _sample_dag_inputs()
        node_latent, z_dag = enc(
            dag_node_feat=jnp.asarray(inputs["dag_node_feat"]),
            dag_node_mask=jnp.asarray(inputs["dag_node_mask"]),
            dag_edge_src=jnp.asarray(inputs["dag_edge_src"]),
            dag_edge_dst=jnp.asarray(inputs["dag_edge_dst"]),
            dag_edge_feat=jnp.asarray(inputs["dag_edge_feat"]),
            dag_edge_mask=jnp.asarray(inputs["dag_edge_mask"]),
            dag_global_feat=jnp.asarray(inputs["dag_global_feat"]),
        )

        self.assertEqual(tuple(node_latent.shape), (2, 4, 16))
        self.assertEqual(tuple(z_dag.shape), (2, 32))
        self.assertTrue(np.isfinite(np.asarray(node_latent)).all())
        self.assertTrue(np.isfinite(np.asarray(z_dag)).all())

    @unittest.skipUnless(HAS_NNX, "flax.nnx unavailable")
    def test_full_dag_dropout_matches_no_conditioning_baseline(self) -> None:
        import jax.numpy as jnp
        from flax import nnx

        cfg = NNXBMTConfig(d_model=16, n_layers=1, n_heads=4, ff_mult=2)
        cfg.dag_encoder = NNXDAGEncoderConfig(
            enabled=True,
            d_node_in=24,
            d_edge_in=8,
            d_hidden=16,
            n_layers=2,
            max_nodes=4,
            max_edges=3,
        )
        cfg.dag_conditioning.enabled = True
        cfg.dag_conditioning.dag_dropout_prob = 1.0
        model = NNXBidirectionalMotionTransformer(cfg, rngs=nnx.Rngs(0))

        h = jnp.arange(2 * 3 * 1 * 16, dtype=jnp.float32).reshape(2, 3, 1, 16)
        inputs = _sample_dag_inputs()
        out, meta = model._apply_dag_conditioning(
            h,
            dag_node_feat=jnp.asarray(inputs["dag_node_feat"]),
            dag_node_mask=jnp.asarray(inputs["dag_node_mask"]),
            dag_edge_src=jnp.asarray(inputs["dag_edge_src"]),
            dag_edge_dst=jnp.asarray(inputs["dag_edge_dst"]),
            dag_edge_feat=jnp.asarray(inputs["dag_edge_feat"]),
            dag_edge_mask=jnp.asarray(inputs["dag_edge_mask"]),
            dag_global_feat=jnp.asarray(inputs["dag_global_feat"]),
        )

        self.assertTrue(np.allclose(np.asarray(out), np.asarray(h)))
        self.assertTrue(np.allclose(np.asarray(meta["dag_latent_norm"]), 0.0))
        self.assertTrue(np.allclose(np.asarray(meta["dag_gate_mean"]), 0.0))


if __name__ == "__main__":
    unittest.main()
