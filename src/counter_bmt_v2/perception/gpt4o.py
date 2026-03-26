"""OpenAI perception adapter.

Implements a structured extraction step for maneuvers and outcome labels from
scene frames, producing v2 contracts.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from counter_bmt_v2.contracts import (
    ManeuverSegment,
    ManeuverType,
    ScenarioInput,
    VLMFeatures,
)
from counter_bmt_v2.llm import OpenAIChatClient
from counter_bmt_v2.perception.base import PerceptionModel
from counter_bmt_v2.perception.mock import MockPerceptionModel
from counter_bmt_v2.runtime_guards import coalesce_debug_fallbacks

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except Exception:
        pass

    # Fenced block.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # First balanced object.
    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    break
    return {}


def _response_excerpt(text: str, limit: int = 800) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "...<truncated>"


def _maneuver_type_from_str(s: str) -> ManeuverType:
    t = s.lower().strip().replace("-", "_").replace(" ", "_")
    for m in ManeuverType:
        if t == m.value:
            return m

    if "lane" in t and "left" in t:
        return ManeuverType.LANE_CHANGE_LEFT
    if "lane" in t and "right" in t:
        return ManeuverType.LANE_CHANGE_RIGHT
    if "left" in t and "turn" in t:
        return ManeuverType.LEFT_TURN
    if "right" in t and "turn" in t:
        return ManeuverType.RIGHT_TURN
    if "accel" in t or "speed_up" in t:
        return ManeuverType.ACCELERATE
    if "decel" in t or "brake" in t or "slow" in t:
        return ManeuverType.DECELERATE
    if "stop" in t:
        return ManeuverType.STOP
    if "straight" in t:
        return ManeuverType.STRAIGHT
    return ManeuverType.UNKNOWN


def _normalize_outcome_label(name: str, value: Any) -> str:
    key = str(name).strip()
    text = str(value).lower().strip().replace("-", "_").replace(" ", "_")
    if key == "collision_outcome":
        if "possible" in text or "collision" in text or "crash" in text or "unsafe" in text:
            return "collision_possible"
        return "collision_avoided"
    if key == "progress_outcome":
        if "limited" in text or "blocked" in text or "stuck" in text or "slow" in text:
            return "progress_limited"
        return "progress_good"
    if key == "compliance_outcome":
        if "violation" in text or "illegal" in text or "non" in text:
            return "violation_possible"
        return "compliant"
    return str(value)


@dataclass
class OpenAIPerceptionModel(PerceptionModel):
    model: str = "gpt-5-mini"
    api_key: Optional[str] = None
    max_frames: int = 10
    allow_debug_fallbacks: bool = False
    use_mock_fallback: bool | None = None

    def __post_init__(self) -> None:
        self._fallback = MockPerceptionModel()
        self._allow_debug_fallbacks = coalesce_debug_fallbacks(
            allow_debug_fallbacks=bool(self.allow_debug_fallbacks),
            legacy_fallback_flag=self.use_mock_fallback,
        )
        self._client: Optional[OpenAIChatClient] = None
        try:
            self._client = OpenAIChatClient(model=self.model, api_key=self.api_key)
        except Exception as exc:
            if not self._allow_debug_fallbacks:
                raise RuntimeError(
                    f"OpenAI perception initialization failed for model={self.model!r}"
                ) from exc
            logger.warning("OpenAI perception fallback to mock: %s", exc)

    def _encode_images(self, scene: ScenarioInput) -> List[str]:
        encoded: List[str] = []
        for frame in scene.frames[: self.max_frames]:
            p = Path(frame.path)
            if not p.exists():
                continue
            encoded.append(base64.b64encode(p.read_bytes()).decode("utf-8"))
        return encoded

    def _build_prompt(self, scene: ScenarioInput) -> str:
        frames = scene.frames[: self.max_frames]
        ts = ", ".join(f"{f.timestamp_s:.2f}s" for f in frames) or "none"
        max_ts = float(max((f.timestamp_s for f in frames), default=0.0))
        meta = scene.metadata if isinstance(scene.metadata, dict) else {}
        ego_color_hint = str(meta.get("ego_color_hint", "green"))
        dual_view_enabled = bool(meta.get("dual_view_enabled", False))
        dual_view_mode = str(meta.get("dual_view_mode", ""))
        add_ego_inset = bool(meta.get("add_ego_inset", False))
        context_text = str(meta.get("vlm_context_text", "")).strip()
        frame_lines = []
        for i, f in enumerate(frames):
            frame_lines.append(f"- seq={i:02d} t={float(f.timestamp_s):.2f}s path={Path(f.path).name}")
        frame_block = "\n".join(frame_lines) if frame_lines else "- (no frames)"

        semantics = [
            "You are analyzing a top-down traffic scenario.",
            f"The ego vehicle is highlighted in {ego_color_hint.upper()}.",
            "Frames are a time-ordered sequence. Use timestamp ordering only; do not reverse time.",
            "Track the same ego vehicle consistently across the full sequence.",
            "Inspect the full horizon including the final 25% of frames for late maneuvers.",
            "Infer ego maneuvers from visible motion over time, not single-frame snapshots.",
            "When the ego moves laterally into an adjacent lane while continuing forward motion, label it as lane_change_left or lane_change_right rather than straight.",
            "Reserve left_turn and right_turn for true turning maneuvers at intersections or curved turn geometry, not simple lane shifts.",
        ]
        if dual_view_enabled:
            semantics.append(
                "Each timestep may include two images in order: global scene first, then ego-focused companion view."
            )
            if dual_view_mode:
                semantics.append(f"Dual-view mode: {dual_view_mode}.")

        context_block = ""
        if context_text:
            context_block = f"\nKnown context (trusted side-channel):\n{context_text}\n"

        return f"""
You are extracting driving behavior from top-down traffic frame sequences.

Scenario id: {scene.scenario_id}
Frame timestamps: {ts}
Frame ordering:
{frame_block}

Semantics:
{chr(10).join(f"- {s}" for s in semantics)}
{context_block}

Return JSON only with schema:
{{
  "maneuvers": [
    {{
      "type": "straight|left_turn|right_turn|lane_change_left|lane_change_right|accelerate|decelerate|stop|unknown",
      "start_s": <float>,
      "end_s": <float>,
      "aggressiveness": "passive|normal|aggressive",
      "confidence": <float 0..1>,
      "reasoning": "..."
    }}
  ],
  "outcomes": {{
    "collision_outcome": {{
      "value": "collision_avoided|collision_possible",
      "confidence": <float 0..1>,
      "reasoning": "..."
    }},
    "progress_outcome": {{
      "value": "progress_good|progress_limited",
      "confidence": <float 0..1>,
      "reasoning": "..."
    }},
    "compliance_outcome": {{
      "value": "compliant|violation_possible",
      "confidence": <float 0..1>,
      "reasoning": "..."
    }}
  }},
  "summary": "..."
}}

Rules:
- Keep maneuvers time-ordered and non-overlapping when possible.
- Always include at least one maneuver segment that covers the end of the horizon (end_s near {max_ts:.2f}s).
- If the ego performs a late turn (left/right) near the end, include that maneuver explicitly with timestamps.
- If the ego changes lanes, explicitly use lane_change_left or lane_change_right with timestamps covering the visible lateral transition.
- Choose outcome labels from the full sequence, not just the initial frames.
- Base conclusions only on visible evidence and supplied context text.
""".strip()

    def _fallback_extract(self, scene: ScenarioInput, *, reason: str, exc: Exception | None = None) -> VLMFeatures:
        if not self._allow_debug_fallbacks:
            detail = f" scene={scene.scenario_id} model={self.model!r} reason={reason}"
            if exc is not None:
                detail += f" error={exc!r}"
                raw_excerpt = getattr(exc, "raw_excerpt", "")
                if raw_excerpt:
                    detail += f" raw_excerpt={raw_excerpt!r}"
            wrapped = RuntimeError(f"OpenAI perception failed:{detail}")
            for attr in ("raw_response", "raw_excerpt"):
                value = getattr(exc, attr, None) if exc is not None else None
                if value:
                    setattr(wrapped, attr, value)
            raise wrapped from exc
        logger.warning("OpenAI perception fallback to mock (%s): %s", reason, exc or "no exception")
        features = self._fallback.extract(scene)
        features.raw = dict(features.raw)
        features.raw["fallback_reason"] = str(reason)
        features.raw["backend"] = "mock"
        return features

    def extract(self, scene: ScenarioInput) -> VLMFeatures:
        if self._client is None:
            return self._fallback_extract(scene, reason="client_unavailable")

        meta = scene.metadata if isinstance(scene.metadata, dict) else {}
        ego_color_hint = str(meta.get("ego_color_hint", "green"))
        dual_view_enabled = bool(meta.get("dual_view_enabled", False))
        add_ego_inset = bool(meta.get("add_ego_inset", False))
        prompt = self._build_prompt(scene)
        images = self._encode_images(scene)

        try:
            raw = self._client.complete(
                prompt=prompt,
                images_base64=images,
                temperature=0.1,
                max_tokens=3200,
                response_format={"type": "json_object"},
                reasoning_effort="low",
            )
        except Exception as exc:
            return self._fallback_extract(scene, reason="api_call_failed", exc=exc)

        data = _extract_json(raw)
        if not data:
            parse_exc = ValueError(
                "OpenAI perception returned non-JSON or unparsable JSON. "
                f"raw_excerpt={_response_excerpt(raw)!r}"
            )
            setattr(parse_exc, "raw_response", raw)
            setattr(parse_exc, "raw_excerpt", _response_excerpt(raw, limit=1600))
            return self._fallback_extract(scene, reason="json_parse_failed", exc=parse_exc)

        maneuvers: List[ManeuverSegment] = []
        for m in data.get("maneuvers", []):
            try:
                maneuvers.append(
                    ManeuverSegment(
                        maneuver_type=_maneuver_type_from_str(str(m.get("type", "unknown"))),
                        start_s=float(m.get("start_s", 0.0)),
                        end_s=float(m.get("end_s", m.get("start_s", 0.0))),
                        aggressiveness=str(m.get("aggressiveness", "normal")),
                        confidence=float(m.get("confidence", 0.5)),
                        reasoning=str(m.get("reasoning", "")),
                    )
                )
            except Exception:
                continue

        maneuvers.sort(key=lambda x: (x.start_s, x.end_s))

        outcomes_raw = data.get("outcomes", {})
        if not isinstance(outcomes_raw, dict):
            outcomes_raw = {}
        normalized_outcomes: Dict[str, Dict[str, Any]] = {}
        for name in ("collision_outcome", "progress_outcome", "compliance_outcome"):
            item = outcomes_raw.get(name, {})
            if not isinstance(item, dict):
                item = {"value": item}
            normalized_outcomes[name] = {
                "value": _normalize_outcome_label(name, item.get("value", "")),
                "confidence": float(item.get("confidence", 0.5)),
                "reasoning": str(item.get("reasoning", "")),
            }

        return VLMFeatures(
            scenario_id=scene.scenario_id,
            maneuvers=maneuvers,
            decisions=[],
            raw={
                "backend": "openai",
                "summary": data.get("summary", ""),
                "outcomes": normalized_outcomes,
                "raw_response": raw,
                "n_images_sent": len(images),
                "prompt_version": "v3_maneuver_outcome_no_decisions",
                "ego_color_hint": ego_color_hint,
                "dual_view_enabled": dual_view_enabled,
                "add_ego_inset": add_ego_inset,
                "frame_count_sent": len(images),
            },
        )


GPT4oPerceptionModel = OpenAIPerceptionModel
