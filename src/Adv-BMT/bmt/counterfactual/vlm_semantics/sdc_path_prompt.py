from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .sdc_path_contract import SDC_PATH_SEMANTIC_CONTRACT_SCHEMA_VERSION


def _json_block(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), indent=2, sort_keys=True)


def _slot_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "slot_id": str(row.get("slot_id") or ""),
                "source_kind": str(row.get("source_kind") or ""),
                "path_id": row.get("path_id"),
                "on_route": bool(row.get("on_route")),
                "route_length_m": row.get("route_length_m"),
            }
        )
    return out


def build_single_sdc_path_semantic_prompt(record: Mapping[str, Any], *, slot_row: Mapping[str, Any]) -> str:
    slot_payload = _slot_rows([slot_row])[0] if slot_row else {}
    slot_id = str(slot_payload.get("slot_id") or "")
    source_kind = str(slot_payload.get("source_kind") or "")
    return f"""
You are labeling the semantics of one highlighted self-driving car route from one road scene.

You will receive exactly 1 image from the scene.

In the image:
- the current SDC position is marked clearly with a large arrow showing heading
- tiny L / R markers near the arrow indicate vehicle-relative left and right
- the image is rotated so the current SDC heading points straight up
- roads / lanes / boundaries are the map context
- the highlighted route is the thick bright path to classify
- nearby agents and traffic lights are context only
- the starting lane is shaded to show the current lane more clearly
- the final lane reached by the highlighted path may be shaded differently
- a dashed guide may show the \"stay in current lane\" continuation

Visual interpretation tips:
- The opaque dark lane boundaries / road-edge lines are important. Pay close attention to whether the highlighted route crosses those boundaries.
- A lane switch should usually be identified when the highlighted route crosses an opaque lane boundary and then continues in the adjacent lane.
- Compare the highlighted route against the dashed stay-in-lane guide to judge lateral deviation.
- Use the shaded starting lane to understand what the current lane is before the highlighted route departs from it.
- If a differently shaded final lane is shown, use it as supporting evidence for where the route ends up, but still rely on the actual route geometry.
- The lighter lane-centerline geometry is supporting context only; rely more on the opaque lane boundaries to judge lane changes.
- Some highlighted routes may contain discontinuous segments. Follow the continuous route for as long as it remains plausible.
- If a route segment stops and the only continuing segment is clearly in an adjacent left or right lane, classify it as left_lane_change or right_lane_change.
- Do not invent a smooth zig-zag connection across unrelated discontinuous segments.
- If the discontinuity makes the maneuver unclear, choose the closest maneuver label and lower confidence.

Classify the highlighted route with exactly one label:
- left
- right
- left_lane_change
- right_lane_change
- straight
- stop

Interpretation rules:
- left / right = intersection turn maneuver
- left_lane_change / right_lane_change = lateral lane transition while overall travel direction stays roughly forward
- straight = continue forward through the visible road geometry without a major turn or lane change
- stop = route clearly stops at the stop line / before entering the maneuver

Important:
- Use the SDC heading arrow as the reference direction.
- Interpret left and right from the perspective of the vehicle. The large SDC arrow shows forward heading, and the small L / R markers are vehicle-relative.
- Do not hallucinate hidden map structure.
- Keep rationales short.
- If the scene is ambiguous, still choose the closest maneuver label but mark use_for_training=false and lower confidence.
- Output JSON only matching schema_version={SDC_PATH_SEMANTIC_CONTRACT_SCHEMA_VERSION}.

Scene metadata:
{_json_block({
    "example_id": record.get("example_id"),
    "scenario_id": record.get("scenario_id"),
    "sdc_id": record.get("sdc_id"),
    "current_time_index": record.get("current_time_index"),
    "slot_metadata": [slot_payload],
})}

Required output behavior:
- Produce exactly one highlighted_paths entry for slot_id={slot_id}.
- source_kind must be {source_kind}.
- path_id must match the metadata above. If the path is GT, path_id must be null.
- Use confidence in [0, 1].
- Set is_valid_target=false if the highlighted path is too weak / degenerate / not a meaningful maneuver target.
""".strip()


def build_single_sdc_path_postsplit_prompt(record: Mapping[str, Any], *, slot_row: Mapping[str, Any]) -> str:
    slot_payload = _slot_rows([slot_row])[0] if slot_row else {}
    slot_id = str(slot_payload.get("slot_id") or "")
    source_kind = str(slot_payload.get("source_kind") or "")
    return f"""
You are labeling the first maneuver taken by one highlighted self-driving car route after it meaningfully separates from competing routes.

You will receive exactly 1 image from the scene.

In the image:
- the current SDC position is marked clearly with a large arrow showing heading
- the image is rotated so the current SDC heading points straight up
- roads / lanes / boundaries are the map context
- the highlighted route is the thick bright path to classify
- the highlighted route is colored with a gradient tied to route separability
- nearby agents and traffic lights are context only

Gradient meaning:
- dark / cool colors = this part of the highlighted route is still shared with or very similar to competing routes
- bright / warm colors = this part of the highlighted route has become distinct from competing routes
- use the higher-separability colors to focus on what maneuver happens after the routes separate

Task:
- identify the first maneuver that happens after the highlighted route becomes distinct
- if the route stays shared for almost the whole image, choose the closest maneuver that best matches the earliest clearly distinct segment

Classify the highlighted route with exactly one label:
- left
- right
- left_lane_change
- right_lane_change
- straight
- stop

Interpretation rules:
- left / right = intersection turn maneuver
- left_lane_change / right_lane_change = lateral lane transition while overall travel direction stays roughly forward
- straight = continue forward through the visible road geometry without a major turn or lane change
- stop = route clearly stops before entering the maneuver

Important:
- Use the SDC heading arrow as the reference direction.
- Interpret left and right from the perspective of the vehicle.
- Use the path gradient intentionally: focus on the first clearly distinct part of the route, not just the shared prefix.
- Do not hallucinate hidden map structure.
- Keep rationales short.
- If the scene is ambiguous, still choose the closest maneuver label but mark use_for_training=false and lower confidence.
- Output JSON only matching schema_version={SDC_PATH_SEMANTIC_CONTRACT_SCHEMA_VERSION}.

Scene metadata:
{_json_block({
    "example_id": record.get("example_id"),
    "scenario_id": record.get("scenario_id"),
    "sdc_id": record.get("sdc_id"),
    "current_time_index": record.get("current_time_index"),
    "slot_metadata": [slot_payload],
})}

Required output behavior:
- Produce exactly one highlighted_paths entry for slot_id={slot_id}.
- source_kind must be {source_kind}.
- path_id must match the metadata above. If the path is GT, path_id must be null.
- Use confidence in [0, 1].
- Set is_valid_target=false if the highlighted path is too weak / degenerate / not a meaningful maneuver target.
""".strip()


def build_sdc_path_semantic_prompt(record: Mapping[str, Any]) -> str:
    return f"""
You are labeling the semantics of highlighted self-driving car routes from one road scene.

You will receive exactly 4 images from the same scene:
- image 1: GT future trajectory of the SDC
- image 2: alternate candidate path alt_1
- image 3: alternate candidate path alt_2
- image 4: alternate candidate path alt_3

In every image:
- the current SDC position is marked clearly with a large arrow showing heading
- tiny L / R markers near the arrow indicate vehicle-relative left and right
- the image is rotated so the current SDC heading points straight up
- roads / lanes / boundaries are the map context
- the highlighted route is the thick bright path to classify
- nearby agents and traffic lights are context only
- the starting lane is shaded to show the current lane more clearly
- the final lane reached by the highlighted path may be shaded differently
- a dashed guide may show the \"stay in current lane\" continuation

Visual interpretation tips:
- The opaque dark lane boundaries / road-edge lines are important. Pay close attention to whether the highlighted route crosses those boundaries.
- A lane switch should usually be identified when the highlighted route crosses an opaque lane boundary and then continues in the adjacent lane.
- Compare the highlighted route against the dashed stay-in-lane guide to judge lateral deviation.
- Use the shaded starting lane to understand what the current lane is before the highlighted route departs from it.
- If a differently shaded final lane is shown, use it as supporting evidence for where the route ends up, but still rely on the actual route geometry.
- The lighter lane-centerline geometry is supporting context only; rely more on the opaque lane boundaries to judge lane changes.
- Some highlighted routes may contain discontinuous segments. Follow the continuous route for as long as it remains plausible.
- If a route segment stops and the only continuing segment is clearly in an adjacent left or right lane, classify it as left_lane_change or right_lane_change.
- Do not invent a smooth zig-zag connection across unrelated discontinuous segments.
- If the discontinuity makes the maneuver unclear, choose the closest maneuver label and lower confidence.

Classify each highlighted route with exactly one label:
- left
- right
- left_lane_change
- right_lane_change
- straight
- stop

Interpretation rules:
- left / right = intersection turn maneuver
- left_lane_change / right_lane_change = lateral lane transition while overall travel direction stays roughly forward
- straight = continue forward through the visible road geometry without a major turn or lane change
- stop = route clearly stops at the stop line / before entering the maneuver

Important:
- Use the SDC heading arrow as the reference direction.
- Interpret left and right from the perspective of the vehicle. The large SDC arrow shows forward heading, and the small L / R markers are vehicle-relative.
- Do not hallucinate hidden map structure.
- Keep rationales short.
- If the scene is ambiguous, still choose the closest maneuver label but mark use_for_training=false and lower confidence.
- Output JSON only matching schema_version={SDC_PATH_SEMANTIC_CONTRACT_SCHEMA_VERSION}.

Scene metadata:
{_json_block({
    "example_id": record.get("example_id"),
    "scenario_id": record.get("scenario_id"),
    "sdc_id": record.get("sdc_id"),
    "current_time_index": record.get("current_time_index"),
    "slot_metadata": _slot_rows(record.get("slot_metadata") or []),
})}

Required output behavior:
- Produce one highlighted_paths entry for each slot_id: gt, alt_1, alt_2, alt_3.
- For gt, source_kind must be ground_truth and path_id should be null.
- For alt_1/alt_2/alt_3, source_kind must be sdc_path and path_id must match the metadata above.
- Use confidence in [0, 1].
- Set is_valid_target=false if a highlighted path is too weak / degenerate / not a meaningful maneuver target.
""".strip()
