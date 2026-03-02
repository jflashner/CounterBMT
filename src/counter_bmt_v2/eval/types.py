"""Types for head-to-head model evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


BackendType = Literal["v2", "legacy_adv_bmt"]
MetricModeType = Literal["approx", "strict_if_available"]
LegacyPolicyType = Literal["required_if_available", "required", "optional"]
SelectionPolicyType = Literal["largest_spread_sfde_min"]


@dataclass
class ModelRuntimeConfig:
    model_preset: Optional[str] = None
    tokenizer_mode: Optional[str] = None
    skip_steps: int = 5
    num_modes: int = 6
    sampling_method: str = "topp"
    topp: float = 0.95
    temperature: float = 1.0
    python_bin: str = "python"
    legacy_root: str = "src/Adv-BMT"
    config_name: str = ""
    dag_source_mode: str = "none"  # none|dual|cache|scene_derived
    dag_cache_dir: str = ""
    dag_cache_strict: bool = False
    dag_expected_schema: str = "any"  # any|v2_compact10|v3_maneuver_outcome


@dataclass
class ModelSpec:
    id: str
    backend: BackendType
    checkpoint: str
    runtime: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    enabled: bool = True


@dataclass
class MetricsConfig:
    mode: MetricModeType = "approx"


@dataclass
class VisualizationConfig:
    max_scenarios: int = 8
    max_agents: int = 10
    selection_policy: SelectionPolicyType = "largest_spread_sfde_min"


@dataclass
class ReplayExportConfig:
    enabled: bool = True
    max_scenarios: int = 8
    mode_index: int = 0
    include_ground_truth: bool = False


@dataclass
class Head2HeadConfig:
    dataset_dir: str
    output_dir: str
    n_scenarios: int = 100
    seed: int = 0
    scenario_indices_file: str = ""
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    replay_export: ReplayExportConfig = field(default_factory=ReplayExportConfig)
    legacy_policy: LegacyPolicyType = "required_if_available"
    reuse_artifacts: bool = True
    max_parallel_models: int = 1
    models: List[ModelSpec] = field(default_factory=list)


@dataclass
class ScenarioSubsetEntry:
    rank: int
    dataset_index: int
    scenario_id: str
    relative_path: str


@dataclass
class ModelRunResult:
    model_id: str
    backend: BackendType
    artifact_dir: str
    summary_path: str
    skipped: bool = False
    reason: str = ""
    log_path: str = ""
    stderr_path: str = ""


def model_spec_hashable_dict(spec: ModelSpec) -> Dict[str, Any]:
    """Stable dict for hash-based cache keys."""
    return {
        "id": str(spec.id),
        "backend": str(spec.backend),
        "checkpoint": str(Path(spec.checkpoint)),
        "enabled": bool(spec.enabled),
        "runtime": {
            "model_preset": spec.runtime.model_preset or "",
            "tokenizer_mode": spec.runtime.tokenizer_mode or "",
            "skip_steps": int(spec.runtime.skip_steps),
            "num_modes": int(spec.runtime.num_modes),
            "sampling_method": str(spec.runtime.sampling_method),
            "topp": float(spec.runtime.topp),
            "temperature": float(spec.runtime.temperature),
            "python_bin": str(spec.runtime.python_bin),
            "legacy_root": str(spec.runtime.legacy_root),
            "config_name": str(spec.runtime.config_name),
            "dag_source_mode": str(spec.runtime.dag_source_mode),
            "dag_cache_dir": str(spec.runtime.dag_cache_dir),
            "dag_cache_strict": bool(spec.runtime.dag_cache_strict),
            "dag_expected_schema": str(spec.runtime.dag_expected_schema),
        },
    }
