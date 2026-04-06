from .client import OpenAIVLMSemanticClient, api_key_available
from .contract import (
    VLM_SEMANTIC_CONTRACT_SCHEMA_VERSION,
    make_empty_contract,
    normalize_contract,
    semantic_contract_json_schema,
    should_escalate_contract,
)
from .fuse import fuse_geometry_and_vlm_contracts, merge_pass_contracts

try:
    from .audit import load_bundle_selected_examples, load_materialized_manifest_examples
except Exception:  # pragma: no cover - optional heavy dependency path
    load_bundle_selected_examples = None
    load_materialized_manifest_examples = None

try:
    from .render import render_vlm_semantic_views
except Exception:  # pragma: no cover - optional heavy dependency path
    render_vlm_semantic_views = None

__all__ = [
    "OpenAIVLMSemanticClient",
    "VLM_SEMANTIC_CONTRACT_SCHEMA_VERSION",
    "api_key_available",
    "fuse_geometry_and_vlm_contracts",
    "load_bundle_selected_examples",
    "load_materialized_manifest_examples",
    "make_empty_contract",
    "merge_pass_contracts",
    "normalize_contract",
    "render_vlm_semantic_views",
    "semantic_contract_json_schema",
    "should_escalate_contract",
]
