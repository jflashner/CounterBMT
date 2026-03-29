"""Helpers for DAG-latent config on top of the legacy Hydra tree."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from .encoder import DAGLatentConfig


DEFAULT_DAG_LATENT_BLOCK: Dict[str, Any] = {
    "ENABLED": False,
    "STAGE": "",
    "SOURCE_MODE": "",
    "CACHE_DIR": "",
    "CACHE_STRICT": False,
    "EXPECTED_SCHEMA": "any",
    "ONLY_CACHE_IDS": False,
    "TRAIN_CACHE_DIR": "",
    "VAL_CACHE_DIR": "",
    "TRAIN_CACHE_STRICT": None,
    "VAL_CACHE_STRICT": None,
    "TRAIN_EXPECTED_SCHEMA": "",
    "VAL_EXPECTED_SCHEMA": "",
    "ONLY_CACHE_IDS_TRAIN": None,
    "ONLY_CACHE_IDS_VAL": None,
    "USE_GRAPH_ENCODER": True,
    "D_NODE_IN": 24,
    "D_EDGE_IN": 8,
    "D_HIDDEN": 128,
    "N_LAYERS": 3,
    "DROPOUT": 0.0,
    "MAX_NODES": 64,
    "MAX_EDGES": 256,
    "LATENT_DIM": None,
    "INJECTION_MODE": "global_gated_residual",
    "DAG_DROPOUT_PROB": 0.0,
    "USE_NULL_LATENT": False,
    "NULL_LATENT_INIT_STD": 0.02,
    "USE_TIME_GUIDANCE": False,
    "TIME_GUIDANCE_FEATURE_DIM": 18,
    "TIME_GUIDANCE_MODE": "gated",
    "TIME_GUIDANCE_USE_GLOBAL": False,
    "TIME_GUIDANCE_INIT_GATE_BIAS": -2.0,
    "TIME_GUIDANCE_STEP_DT": 0.0,
    "TIME_GUIDANCE_MIN_STEPS": 19,
    "TIME_GUIDANCE_ACTIVE_AGG": "sum",
    "USE_MANEUVER_TOKENS": False,
    "MANEUVER_TOKEN_FEATURE_DIM": 20,
    "MANEUVER_TOKEN_USE_GLOBAL": False,
    "MANEUVER_TOKEN_INIT_GATE_BIAS": -2.0,
    "EVAL_ALIGNMENT": False,
    "TRAIN_ALIGNMENT_EVERY_N_STEPS": 0,
    "STAGE_B_FREEZE_NON_DAG": True,
    "STAGE_C_BACKBONE_LR_SCALE": 0.02,
    "STAGE_C_DECODER_LR_SCALE": 0.1,
    "STAGE_C_DAG_LR_SCALE": 1.0,
}


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return int(value)


def get_dag_latent_block(config: Any) -> Dict[str, Any]:
    raw = config.get("DAG_LATENT", {}) if hasattr(config, "get") else getattr(config, "DAG_LATENT", {})
    merged = dict(DEFAULT_DAG_LATENT_BLOCK)
    if raw:
        for key, value in raw.items():
            merged[str(key)] = value
    return merged


def build_dag_latent_config(config: Any) -> DAGLatentConfig:
    block = get_dag_latent_block(config)
    return DAGLatentConfig(
        enabled=bool(block.get("ENABLED", False)),
        use_graph_encoder=bool(block.get("USE_GRAPH_ENCODER", True)),
        d_node_in=int(block.get("D_NODE_IN", 24)),
        d_edge_in=int(block.get("D_EDGE_IN", 8)),
        d_hidden=int(block.get("D_HIDDEN", 128)),
        n_layers=int(block.get("N_LAYERS", 3)),
        dropout=float(block.get("DROPOUT", 0.0)),
        max_nodes=int(block.get("MAX_NODES", 64)),
        max_edges=int(block.get("MAX_EDGES", 256)),
        latent_dim=_maybe_int(block.get("LATENT_DIM")),
        injection_mode=str(block.get("INJECTION_MODE", "global_gated_residual")),
        dag_dropout_prob=float(block.get("DAG_DROPOUT_PROB", 0.0)),
        use_null_latent=bool(block.get("USE_NULL_LATENT", False)),
        null_latent_init_std=float(block.get("NULL_LATENT_INIT_STD", 0.02)),
        use_time_guidance=bool(block.get("USE_TIME_GUIDANCE", False)),
        time_guidance_feature_dim=int(block.get("TIME_GUIDANCE_FEATURE_DIM", 18)),
        time_guidance_mode=str(block.get("TIME_GUIDANCE_MODE", "gated")),
        time_guidance_use_global=bool(block.get("TIME_GUIDANCE_USE_GLOBAL", False)),
        time_guidance_init_gate_bias=float(block.get("TIME_GUIDANCE_INIT_GATE_BIAS", -2.0)),
        use_maneuver_tokens=bool(block.get("USE_MANEUVER_TOKENS", False)),
        maneuver_token_feature_dim=int(block.get("MANEUVER_TOKEN_FEATURE_DIM", 20)),
        maneuver_token_use_global=bool(block.get("MANEUVER_TOKEN_USE_GLOBAL", False)),
        maneuver_token_init_gate_bias=float(block.get("MANEUVER_TOKEN_INIT_GATE_BIAS", -2.0)),
    )


def dag_latent_config_as_dict(config: Any) -> Dict[str, Any]:
    return asdict(build_dag_latent_config(config))
