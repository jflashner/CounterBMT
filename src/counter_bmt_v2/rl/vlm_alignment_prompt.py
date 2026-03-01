"""Prompt + response parsing for VLM DAG-conformance alignment."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ParsedVLMAlignment:
    score: float
    confidence: float
    matched_factors: List[str]
    violations: List[str]
    reason: str
    raw: Dict[str, Any]


def _extract_json_object(text: str) -> Dict[str, Any]:
    src = str(text or "").strip()
    if not src:
        return {}
    try:
        return json.loads(src)
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", src, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    start = src.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(src)):
        ch = src[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = src[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    break
    return {}


def build_alignment_prompt(
    *,
    scenario_id: str,
    intervention_text: str,
    dag_text: str,
    prompt_version: str,
    num_frames: int,
) -> str:
    return f"""
You are verifying whether a predicted ego trajectory conforms to a sampled causal DAG.

Prompt version: {prompt_version}
Scenario id: {scenario_id}
You are given top-down time-ordered images. For each timestep, one base scene frame
and one trajectory-overlay frame may be provided.
Frame count sent: {int(num_frames)}

Intervention:
{intervention_text}

Compact DAG:
{dag_text}

Task:
Score how well the predicted trajectory follows the causal intent represented by the intervention + DAG.
Focus on maneuver/decision consistency and whether the rollout violates implied risk/outcome constraints.

Output JSON only with schema:
{{
  "conformance_score": <float 0..1>,
  "confidence": <float 0..1>,
  "matched_factors": ["..."],
  "violations": ["..."],
  "reason": "short explanation"
}}

Rules:
- Be conservative and avoid hallucinations.
- If evidence is weak, lower confidence and use a middle/low score.
- Keep explanation short and factual.
""".strip()


def parse_alignment_response(text: str) -> Optional[ParsedVLMAlignment]:
    data = _extract_json_object(text)
    if not data:
        return None

    def _to_list_str(v: Any) -> List[str]:
        if isinstance(v, list):
            return [str(x) for x in v]
        if v is None:
            return []
        return [str(v)]

    try:
        score = float(data.get("conformance_score"))
    except Exception:
        return None

    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    score = max(0.0, min(1.0, score))
    confidence = max(0.0, min(1.0, confidence))
    return ParsedVLMAlignment(
        score=score,
        confidence=confidence,
        matched_factors=_to_list_str(data.get("matched_factors", [])),
        violations=_to_list_str(data.get("violations", [])),
        reason=str(data.get("reason", "")),
        raw=data,
    )

