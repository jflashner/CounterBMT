"""DAG cache reader for supervised DAG-latent training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class DAGCacheReader:
    cache_dir: str
    schema_version: str = "counter_bmt_v2_dag_cache_v1"

    def __post_init__(self) -> None:
        self.root = Path(self.cache_dir)

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
        if str(payload.get("schema_version", "")) != str(self.schema_version):
            return None
        if str(payload.get("scenario_id", "")) != str(scenario_id):
            return None
        if "nodes" not in payload or "edges" not in payload:
            return None
        return payload

