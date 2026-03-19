from __future__ import annotations

import unittest

import numpy as np

try:
    import jax.numpy as jnp
    from flax import nnx

    HAS_NNX = True
except Exception:  # pragma: no cover
    jnp = None
    nnx = None
    HAS_NNX = False

from counter_bmt_v2.trajectory_jax.nnx_bmt import NNXBidirectionalMotionTransformer
from counter_bmt_v2.trajectory_jax.presets import midgpt_dag_latent_config, midgpt_parity_config


@unittest.skipUnless(HAS_NNX, "jax/flax nnx required")
class SceneEncoderParityTests(unittest.TestCase):
    def _dummy_scene_inputs(self) -> dict[str, np.ndarray]:
        batch = 1
        n_map = 3
        n_vec = 5
        n_tl = 2
        t_hist = 11

        map_feature = np.zeros((batch, n_map, n_vec, 27), dtype=np.float32)
        map_feature_valid_mask = np.zeros((batch, n_map, n_vec), dtype=bool)
        map_feature_valid_mask[0, 0, :3] = True
        map_feature_valid_mask[0, 1, :2] = True

        # Populate valid vectors with deterministic values and a heading channel.
        for poly_idx in range(n_map):
            for vec_idx in range(n_vec):
                map_feature[0, poly_idx, vec_idx, :3] = np.array(
                    [poly_idx + 0.1 * vec_idx, vec_idx * 0.2, 1.0],
                    dtype=np.float32,
                )
                map_feature[0, poly_idx, vec_idx, 9] = 0.05 * (poly_idx + vec_idx)

        map_position = np.array(
            [[[0.0, 0.0, 0.0], [8.0, 0.5, 0.0], [20.0, -1.0, 0.0]]],
            dtype=np.float32,
        )

        traffic_light_feature = np.zeros((batch, t_hist, n_tl, 7), dtype=np.float32)
        traffic_light_valid_mask = np.zeros((batch, t_hist, n_tl), dtype=bool)
        traffic_light_valid_mask[0, :, 0] = True
        traffic_light_feature[0, :, 0, :3] = np.array([3.0, 1.0, 0.0], dtype=np.float32)
        traffic_light_feature[0, :, 0, 3] = 1.0
        traffic_light_position = np.array([[[3.0, 1.0, 0.0], [12.0, -2.0, 0.0]]], dtype=np.float32)

        return {
            "scene_map_feature": map_feature,
            "scene_map_valid_mask": map_feature_valid_mask,
            "scene_map_position": map_position,
            "scene_tl_feature": traffic_light_feature,
            "scene_tl_valid_mask": traffic_light_valid_mask,
            "scene_tl_position": traffic_light_position,
        }

    def test_midgpt_parity_uses_legacy_pointnet_scene_encoder(self) -> None:
        cfg = midgpt_parity_config()
        self.assertEqual(cfg.scene_encoder.map_encoder_style, "legacy_pointnet")
        self.assertEqual(cfg.scene_encoder.legacy_polyline_hidden_dim, 64)
        self.assertEqual(cfg.scene_encoder.legacy_polyline_num_layers, 2)
        self.assertEqual(cfg.scene_encoder.legacy_polyline_num_pre_layers, 1)
        self.assertEqual(cfg.scene_encoder.norm_style, "layernorm")
        self.assertTrue(cfg.scene_encoder.use_post_proj_head)
        self.assertFalse(cfg.decoder.dense_masked_relation_attn)

    def test_midgpt_parity_uses_decoder_relation_embedders_and_no_step_embed(self) -> None:
        cfg = midgpt_parity_config()
        model = NNXBidirectionalMotionTransformer(cfg, rngs=nnx.Rngs(0))

        self.assertFalse(hasattr(model, "step_embed"))
        self.assertIsNotNone(model.relation_embed_a2a)
        self.assertIsNotNone(model.relation_embed_a2t)
        self.assertIsNotNone(model.relation_embed_a2s)
        self.assertEqual(model.relation_embed_a2a.input_dim, cfg.a2a_rel_dim)
        self.assertEqual(model.relation_embed_a2t.input_dim, cfg.a2t_rel_dim)
        self.assertEqual(model.relation_embed_a2s.input_dim, cfg.a2s_rel_dim)
        self.assertEqual(model.relation_embed_a2a.hidden_dim, cfg.d_model)
        self.assertEqual(model.decoder_relation_hidden_dim, cfg.d_model)

    def test_invalid_map_points_do_not_affect_legacy_scene_tokens(self) -> None:
        cfg = midgpt_parity_config()
        model = NNXBidirectionalMotionTransformer(cfg, rngs=nnx.Rngs(0))
        base = self._dummy_scene_inputs()
        perturbed = {k: np.array(v, copy=True) for k, v in base.items()}

        invalid_mask = ~perturbed["scene_map_valid_mask"]
        perturbed["scene_map_feature"][invalid_mask[..., None].repeat(27, axis=-1)] = 999.0
        perturbed["scene_map_feature"][..., 9][invalid_mask] = -123.0

        scene_1, mask_1, pos_1 = model.encode_scene_tokens(
            map_feature=jnp.asarray(base["scene_map_feature"]),
            map_feature_valid_mask=jnp.asarray(base["scene_map_valid_mask"]),
            map_position=jnp.asarray(base["scene_map_position"]),
            traffic_light_feature=jnp.asarray(base["scene_tl_feature"]),
            traffic_light_valid_mask=jnp.asarray(base["scene_tl_valid_mask"]),
            traffic_light_position=jnp.asarray(base["scene_tl_position"]),
        )
        scene_2, mask_2, pos_2 = model.encode_scene_tokens(
            map_feature=jnp.asarray(perturbed["scene_map_feature"]),
            map_feature_valid_mask=jnp.asarray(perturbed["scene_map_valid_mask"]),
            map_position=jnp.asarray(perturbed["scene_map_position"]),
            traffic_light_feature=jnp.asarray(perturbed["scene_tl_feature"]),
            traffic_light_valid_mask=jnp.asarray(perturbed["scene_tl_valid_mask"]),
            traffic_light_position=jnp.asarray(perturbed["scene_tl_position"]),
        )

        np.testing.assert_allclose(np.asarray(scene_1), np.asarray(scene_2), atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(np.asarray(mask_1), np.asarray(mask_2))
        np.testing.assert_allclose(np.asarray(pos_1), np.asarray(pos_2), atol=0.0, rtol=0.0)

    def test_decoder_relation_embedding_zeros_masked_edges_and_expands_width(self) -> None:
        cfg = midgpt_parity_config()
        model = NNXBidirectionalMotionTransformer(cfg, rngs=nnx.Rngs(0))

        raw_rel = jnp.asarray(
            [[[[[1.0] * cfg.a2a_rel_dim, [2.0] * cfg.a2a_rel_dim]]]],
            dtype=jnp.float32,
        )
        rel_mask = jnp.asarray([[[[True, False]]]])

        embedded = model._embed_decoder_relation_tensor(
            raw_rel,
            rel_mask,
            embedder=model.relation_embed_a2a,
            expected_raw_dim=cfg.a2a_rel_dim,
            expected_emb_dim=cfg.d_model,
            label="a2a_rel",
        )

        self.assertEqual(tuple(embedded.shape), (1, 1, 1, 2, cfg.d_model))
        np.testing.assert_allclose(np.asarray(embedded[0, 0, 0, 1]), 0.0, atol=1e-6, rtol=1e-6)
        self.assertGreater(float(np.linalg.norm(np.asarray(embedded[0, 0, 0, 0]))), 0.0)

    def test_collapsed_traffic_light_tokens_do_not_add_extra_position_embedding(self) -> None:
        cfg = midgpt_parity_config()
        model = NNXBidirectionalMotionTransformer(cfg, rngs=nnx.Rngs(0))

        light_feat = jnp.asarray([[[3.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]]], dtype=jnp.float32)
        light_mask = jnp.asarray([[True]])
        pos_a = jnp.asarray([[[3.0, 1.0, 0.0]]], dtype=jnp.float32)
        pos_b = jnp.asarray([[[30.0, -7.0, 0.0]]], dtype=jnp.float32)

        tok_a, mask_a, _, _ = model.scene_encoder._encode_traffic_lights(
            traffic_light_feature=light_feat,
            traffic_light_valid_mask=light_mask,
            traffic_light_position=pos_a,
        )
        tok_b, mask_b, _, _ = model.scene_encoder._encode_traffic_lights(
            traffic_light_feature=light_feat,
            traffic_light_valid_mask=light_mask,
            traffic_light_position=pos_b,
        )

        np.testing.assert_allclose(np.asarray(tok_a), np.asarray(tok_b), atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(np.asarray(mask_a), np.asarray(mask_b))

    def test_time_major_traffic_light_collapse_uses_feature_xyz_not_external_position(self) -> None:
        cfg = midgpt_parity_config()
        model = NNXBidirectionalMotionTransformer(cfg, rngs=nnx.Rngs(0))

        light_feat = np.zeros((1, 11, 1, 7), dtype=np.float32)
        light_feat[:, :, 0, :3] = np.array([3.0, 1.0, 0.0], dtype=np.float32)
        light_feat[:, :, 0, 3] = 1.0
        light_mask = np.ones((1, 11, 1), dtype=bool)
        pos_a = np.array([[[3.0, 1.0, 0.0]]], dtype=np.float32)
        pos_b = np.array([[[30.0, -7.0, 0.0]]], dtype=np.float32)

        tok_a, _, _, _ = model.scene_encoder._encode_traffic_lights(
            traffic_light_feature=jnp.asarray(light_feat),
            traffic_light_valid_mask=jnp.asarray(light_mask),
            traffic_light_position=jnp.asarray(pos_a),
        )
        tok_b, _, _, _ = model.scene_encoder._encode_traffic_lights(
            traffic_light_feature=jnp.asarray(light_feat),
            traffic_light_valid_mask=jnp.asarray(light_mask),
            traffic_light_position=jnp.asarray(pos_b),
        )

        np.testing.assert_allclose(np.asarray(tok_a), np.asarray(tok_b), atol=1e-6, rtol=1e-6)

    def test_midgpt_dag_latent_forward_remains_compatible(self) -> None:
        cfg = midgpt_dag_latent_config()
        self.assertEqual(cfg.scene_encoder.map_encoder_style, "legacy_pointnet")
        self.assertTrue(cfg.dag_encoder.enabled)
        self.assertTrue(cfg.dag_conditioning.enabled)

        model = NNXBidirectionalMotionTransformer(cfg, rngs=nnx.Rngs(0))
        scene = self._dummy_scene_inputs()

        batch = 1
        t_steps = 4
        n_agents = 2
        logits, meta = model(
            prev_token_ids=jnp.zeros((batch, t_steps, n_agents), dtype=jnp.int32),
            agent_type_ids=jnp.ones((batch, n_agents), dtype=jnp.int32),
            agent_shape=jnp.ones((batch, n_agents, 3), dtype=jnp.float32),
            agent_ids=jnp.asarray([[0, 1]], dtype=jnp.int32),
            continuous_motion=jnp.zeros((batch, t_steps, n_agents, 2), dtype=jnp.float32),
            reverse_indicator=jnp.zeros((batch,), dtype=jnp.int32),
            input_action_valid_mask=jnp.ones((batch, t_steps, n_agents), dtype=bool),
            modeled_agent_delta=jnp.zeros((batch, t_steps, n_agents, 2), dtype=jnp.float32),
            scene_map_feature=jnp.asarray(scene["scene_map_feature"]),
            scene_map_valid_mask=jnp.asarray(scene["scene_map_valid_mask"]),
            scene_map_position=jnp.asarray(scene["scene_map_position"]),
            scene_tl_feature=jnp.asarray(scene["scene_tl_feature"]),
            scene_tl_valid_mask=jnp.asarray(scene["scene_tl_valid_mask"]),
            scene_tl_position=jnp.asarray(scene["scene_tl_position"]),
            dag_node_feat=jnp.ones((batch, 2, cfg.dag_encoder.d_node_in), dtype=jnp.float32),
            dag_node_mask=jnp.asarray([[True, True]]),
            dag_edge_src=jnp.asarray([[0, 1]], dtype=jnp.int32),
            dag_edge_dst=jnp.asarray([[1, 1]], dtype=jnp.int32),
            dag_edge_feat=jnp.zeros((batch, 2, cfg.dag_encoder.d_edge_in), dtype=jnp.float32),
            dag_edge_mask=jnp.asarray([[True, False]]),
            dag_global_feat=jnp.zeros((batch, 4), dtype=jnp.float32),
            return_metadata=True,
        )

        self.assertEqual(tuple(logits.shape), (batch, t_steps, n_agents, cfg.token_space.n_tokens))
        self.assertIn("scene", meta)

    def test_sparse_relation_indices_ignore_non_neighbor_entries(self) -> None:
        cfg = midgpt_parity_config()
        cfg.n_layers = 1
        model = NNXBidirectionalMotionTransformer(cfg, rngs=nnx.Rngs(0))

        batch = 1
        t_steps = 2
        n_agents = 2
        scene_tokens = jnp.zeros((batch, 1, cfg.d_model), dtype=jnp.float32)
        scene_mask = jnp.asarray([[True]])

        base_rel = np.zeros((batch, t_steps, n_agents, n_agents, cfg.a2a_rel_dim), dtype=np.float32)
        perturbed_rel = np.array(base_rel, copy=True)
        perturb_vec = np.linspace(-3.0, 5.0, cfg.a2a_rel_dim, dtype=np.float32) * 40.0
        perturbed_rel[:, :, 0, 1, :] = perturb_vec
        perturbed_rel[:, :, 1, 0, :] = -perturb_vec

        common_inputs = dict(
            prev_token_ids=jnp.zeros((batch, t_steps, n_agents), dtype=jnp.int32),
            agent_type_ids=jnp.zeros((batch, n_agents), dtype=jnp.int32),
            agent_shape=jnp.ones((batch, n_agents, 3), dtype=jnp.float32),
            agent_ids=jnp.asarray([[0, 1]], dtype=jnp.int32),
            continuous_motion=jnp.zeros((batch, t_steps, n_agents, 2), dtype=jnp.float32),
            reverse_indicator=jnp.zeros((batch,), dtype=jnp.int32),
            input_action_valid_mask=jnp.ones((batch, t_steps, n_agents), dtype=bool),
            modeled_agent_delta=jnp.zeros((batch, t_steps, n_agents, 2), dtype=jnp.float32),
            scene_tokens=scene_tokens,
            scene_token_mask=scene_mask,
            a2a_mask=jnp.ones((batch, t_steps, n_agents, n_agents), dtype=bool),
            a2t_rel=jnp.zeros((batch, n_agents, t_steps, t_steps, cfg.a2t_rel_dim), dtype=jnp.float32),
            a2t_mask=jnp.tril(jnp.ones((batch, n_agents, t_steps, t_steps), dtype=bool)),
            a2t_indices=jnp.zeros((batch, n_agents, t_steps, 0), dtype=jnp.int32),
            a2s_rel=jnp.zeros((batch, t_steps, n_agents, 1, cfg.a2s_rel_dim), dtype=jnp.float32),
            a2s_mask=jnp.ones((batch, t_steps, n_agents, 1), dtype=bool),
            a2s_indices=jnp.zeros((batch, t_steps, n_agents, 1), dtype=jnp.int32),
            a2a_indices=jnp.asarray([[[[0], [1]], [[0], [1]]]], dtype=jnp.int32),
        )

        logits_base = model(a2a_rel=jnp.asarray(base_rel), **common_inputs)
        logits_perturbed = model(a2a_rel=jnp.asarray(perturbed_rel), **common_inputs)

        np.testing.assert_allclose(
            np.asarray(logits_base),
            np.asarray(logits_perturbed),
            atol=1e-6,
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
