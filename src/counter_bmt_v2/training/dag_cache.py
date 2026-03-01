"""DAG cache reader for supervised DAG-latent training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from counter_bmt_v2.training.dag_cache_schema import (
    SUPPORTED_SCHEMA_VERSIONS,
    detect_schema_version,
    validate_cache_payload,
)


@dataclass
class DAGCacheReader:
    cache_dir: str
    schema_version: str = ""
    allowed_schema_versions: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        self.root = Path(self.cache_dir)
        if self.allowed_schema_versions:
            allowed = {str(v).strip() for v in self.allowed_schema_versions if str(v).strip()}
        elif str(self.schema_version).strip():
            allowed = {str(self.schema_version).strip()}
        else:
            # Dual-read default during schema migration.
            allowed = set(SUPPORTED_SCHEMA_VERSIONS)
        # Keep deterministic ordering for debug messages.
        self._allowed_schema_versions = tuple(sorted(allowed))

    def _path(self, scenario_id: str) -> Path:
        safe = str(scenario_id).strip()
        return self.root / f"{safe}.json"

    def get(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(scenario_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        schema_version = detect_schema_version(payload)
        if schema_version not in self._allowed_schema_versions:
            return None
        if str(payload.get("scenario_id", "")) != str(scenario_id):
            return None
        if not validate_cache_payload(payload, allowed_schema_versions=self._allowed_schema_versions):
            return None
        return payload
