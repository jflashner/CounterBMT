from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


UNKNOWN_SIGNAL_STATES = {None, "", "LANE_STATE_UNKNOWN"}


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    return value


@dataclass
class SignalQCResult:
    fraction_unknown: float
    short_oscillation_flag: bool
    missing_stop_point_flag: bool
    confidence_score: float
    state_transition_count: int
    first_known_state: Optional[str]
    dominant_state: Optional[str]
    state_at_reference_time: Optional[str]
    ambiguous_light_state: bool
    known_state_counts: Dict[str, int] = field(default_factory=dict)
    run_lengths: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


def is_unknown_signal_state(state: Optional[str]) -> bool:
    return state in UNKNOWN_SIGNAL_STATES


def is_stop_like_signal_state(state: Optional[str]) -> bool:
    text = "" if state is None else str(state).upper()
    return "STOP" in text or "RED" in text


def is_go_like_signal_state(state: Optional[str]) -> bool:
    text = "" if state is None else str(state).upper()
    return "GO" in text or "GREEN" in text


def is_caution_signal_state(state: Optional[str]) -> bool:
    text = "" if state is None else str(state).upper()
    return "CAUTION" in text or "YELLOW" in text


def evaluate_signal_qc(
    states: Sequence[Optional[str]],
    *,
    stop_point_present: bool,
    reference_time_index: Optional[int] = None,
    ambiguity_threshold: float = 0.45,
) -> SignalQCResult:
    clean_states: List[Optional[str]] = []
    for state in states:
        if state is None:
            clean_states.append(None)
        elif isinstance(state, str):
            clean_states.append(state)
        else:
            clean_states.append(str(state))

    length = max(len(clean_states), 1)
    unknown_count = sum(1 for state in clean_states if is_unknown_signal_state(state))
    fraction_unknown = float(unknown_count / float(length))

    known_states = [state for state in clean_states if not is_unknown_signal_state(state)]
    known_state_counts: Dict[str, int] = {}
    for state in known_states:
        known_state_counts[str(state)] = int(known_state_counts.get(str(state), 0) + 1)

    first_known_state = known_states[0] if known_states else None
    dominant_state = None
    if known_state_counts:
        dominant_state = sorted(known_state_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    state_transition_count = 0
    prev_state: Optional[str] = None
    for state in known_states:
        if prev_state is not None and state != prev_state:
            state_transition_count += 1
        prev_state = state

    run_lengths = _state_runs(clean_states)
    short_oscillation_flag = bool(
        state_transition_count >= 2
        and any((run["state"] is not None and run["length"] <= 2) for run in run_lengths)
    )

    reference_state: Optional[str] = None
    if reference_time_index is not None and clean_states:
        idx = int(np.clip(int(reference_time_index), 0, len(clean_states) - 1))
        reference_state = clean_states[idx]

    confidence = 1.0
    confidence *= max(0.0, 1.0 - fraction_unknown)
    if short_oscillation_flag:
        confidence *= 0.55
    if not stop_point_present:
        confidence *= 0.5
    if is_unknown_signal_state(reference_state):
        confidence *= 0.45
    confidence = float(np.clip(confidence, 0.0, 1.0))

    ambiguous = bool(is_unknown_signal_state(reference_state) or confidence < float(ambiguity_threshold))
    return SignalQCResult(
        fraction_unknown=fraction_unknown,
        short_oscillation_flag=short_oscillation_flag,
        missing_stop_point_flag=not bool(stop_point_present),
        confidence_score=confidence,
        state_transition_count=int(state_transition_count),
        first_known_state=first_known_state,
        dominant_state=dominant_state,
        state_at_reference_time=reference_state,
        ambiguous_light_state=ambiguous,
        known_state_counts=known_state_counts,
        run_lengths=run_lengths,
    )


def _state_runs(states: Sequence[Optional[str]]) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    if not states:
        return runs
    current = states[0]
    start = 0
    for idx, state in enumerate(states[1:], start=1):
        if state != current:
            runs.append({"state": current, "start_idx": int(start), "end_idx": int(idx - 1), "length": int(idx - start)})
            current = state
            start = idx
    runs.append({"state": current, "start_idx": int(start), "end_idx": int(len(states) - 1), "length": int(len(states) - start)})
    return runs
