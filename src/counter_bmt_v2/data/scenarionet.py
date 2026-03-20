"""Minimal ScenarioNet loader for NNX Adv-BMT style training.

Paper alignment notes:
- Adv-BMT models three scene channels in the encoder path: agent history,
  map vectors, and traffic-light states. We expose those channels directly.
- We keep map vectors in the same 27-d layout used in the Adv-BMT preprocessor
  (`bmt/dataset/preprocessor.py`) so scene-token learning remains comparable.
- This module avoids legacy trainer/dataset coupling and only emits fields the
  new NNX pipeline consumes.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# Map the dataset's object type strings into compact IDs for embedding lookup.
_AGENT_TYPE_TO_ID = {
    "VEHICLE": 1,
    "PEDESTRIAN": 2,
    "CYCLIST": 3,
    "OTHER": 4,
    "UNSET": 0,
}


@dataclass
class NNXBMTSceneSample:
    """Compact scene tensors for Adv-BMT style NNX training.

    Shapes (unbatched):
    - agent_position_xy: [T, N, 2]
    - agent_heading: [T, N]
    - agent_velocity_xy: [T, N, 2]
    - agent_valid_mask: [T, N]
    - map_feature: [M, V, 27]
    - map_feature_valid_mask: [M, V]
    - map_position: [M, 3]
    - traffic_light_feature: [T, L, 7]
    - traffic_light_valid_mask: [T, L]
    - traffic_light_position: [L, 3]
    """

    scenario_id: str
    current_time_index: int
    dt_s: float
    map_center_xyz: np.ndarray

    agent_ids: np.ndarray
    agent_type_ids: np.ndarray
    agent_shape: np.ndarray
    agent_position_xy: np.ndarray
    agent_heading: np.ndarray
    agent_velocity_xy: np.ndarray
    agent_valid_mask: np.ndarray

    map_feature: np.ndarray
    map_feature_valid_mask: np.ndarray
    map_position: np.ndarray

    traffic_light_feature: np.ndarray
    traffic_light_valid_mask: np.ndarray
    traffic_light_position: np.ndarray


class ScenarioNetNNXLoader:
    """Load ScenarioNet pickles and emit only NNX-relevant tensors.

    This is a clean replacement for the legacy all-in-one dataset path. It
    favors explicit tensor outputs and shape stability over implicit side data.
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        max_agents: int = 128,
        max_map_features: int = 512,
        max_vectors_per_map_feature: int = 128,
        max_traffic_lights: int = 64,
        center_to_map: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_agents = int(max_agents)
        self.max_map_features = int(max_map_features)
        self.max_vectors_per_map_feature = int(max_vectors_per_map_feature)
        self.max_traffic_lights = int(max_traffic_lights)
        self.center_to_map = bool(center_to_map)

        self._files = self._discover_scenarios(self.data_dir)
        if not self._files:
            raise ValueError(f"No scenario files found under: {self.data_dir}")

    @staticmethod
    def _discover_scenarios(data_dir: Path) -> List[Path]:
        files = list(data_dir.glob("sd_*.pkl"))
        files += list(data_dir.glob("_*/*.pkl"))
        files = [p for p in files if p.name.startswith("sd_") and p.suffix == ".pkl"]
        # Deterministic ordering for reproducible train/val splits/manifests.
        # Sort by stable path relative to the dataset root, not basename only.
        unique = sorted(set(files), key=lambda p: p.as_posix())
        return sorted(unique, key=lambda p: p.relative_to(data_dir).as_posix())

    def __len__(self) -> int:
        return len(self._files)

    @property
    def files(self) -> Sequence[Path]:
        return self._files

    def load(self, index: int) -> NNXBMTSceneSample:
        if index < 0 or index >= len(self._files):
            raise IndexError(f"Scenario index out of range: {index}")

        with self._files[index].open("rb") as f:
            raw = pickle.load(f)

        return self._convert_raw_scenario(raw)

    def iter_samples(self, indices: Iterable[int] | None = None) -> Iterable[NNXBMTSceneSample]:
        if indices is None:
            indices = range(len(self))
        for i in indices:
            yield self.load(int(i))

    def _convert_raw_scenario(self, raw: Dict[str, Any]) -> NNXBMTSceneSample:
        scenario_id = str(raw.get("id") or raw.get("metadata", {}).get("scenario_id") or "unknown")
        metadata = raw.get("metadata", {})
        tracks = raw.get("tracks", {})

        map_center = self._compute_map_center(raw.get("map_features", {}))
        dt_s = self._infer_dt_s(metadata)
        current_time_index = int(metadata.get("current_time_index", 10))

        (
            agent_ids,
            agent_type_ids,
            agent_shape,
            agent_pos,
            agent_heading,
            agent_vel,
            agent_valid,
            sdc_current_xy,
        ) = self._extract_agents(
            tracks=tracks,
            metadata=metadata,
            map_center=map_center,
            current_time_index=current_time_index,
        )

        map_feature, map_feature_mask, map_pos = self._extract_map_features(
            raw.get("map_features", {}),
            map_center=map_center,
            sdc_current_xy=sdc_current_xy,
        )

        tl_feat, tl_valid, tl_pos = self._extract_traffic_lights(
            raw.get("dynamic_map_states", {}),
            map_center=map_center,
            horizon_steps=agent_pos.shape[0],
        )

        return NNXBMTSceneSample(
            scenario_id=scenario_id,
            current_time_index=current_time_index,
            dt_s=dt_s,
            map_center_xyz=map_center.astype(np.float32),
            agent_ids=agent_ids,
            agent_type_ids=agent_type_ids,
            agent_shape=agent_shape,
            agent_position_xy=agent_pos,
            agent_heading=agent_heading,
            agent_velocity_xy=agent_vel,
            agent_valid_mask=agent_valid,
            map_feature=map_feature,
            map_feature_valid_mask=map_feature_mask,
            map_position=map_pos,
            traffic_light_feature=tl_feat,
            traffic_light_valid_mask=tl_valid,
            traffic_light_position=tl_pos,
        )

    @staticmethod
    def _infer_dt_s(metadata: Dict[str, Any]) -> float:
        ts = metadata.get("ts")
        if ts is None:
            return 0.1
        ts = np.asarray(ts, dtype=np.float32)
        if ts.shape[0] < 2:
            return 0.1
        delta = float(np.median(np.diff(ts)))
        if not np.isfinite(delta) or delta <= 0:
            return 0.1
        return delta

    @staticmethod
    def _safe_track_lookup(tracks: Dict[Any, Any], key: Any) -> Any:
        if key in tracks:
            return tracks[key]
        key_str = str(key)
        key_int = None
        try:
            key_int = int(key)
        except Exception:
            key_int = None

        if key_str in tracks:
            return tracks[key_str]
        if key_int is not None and key_int in tracks:
            return tracks[key_int]

        for k in tracks.keys():
            if str(k) == key_str:
                return tracks[k]
        return None

    def _extract_agents(
        self,
        *,
        tracks: Dict[Any, Any],
        metadata: Dict[str, Any],
        map_center: np.ndarray,
        current_time_index: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Keep agents with any valid state, sort by availability, force SDC first.
        scored: List[Tuple[int, Any, Dict[str, Any]]] = []
        for tid, track in tracks.items():
            state = track.get("state", {})
            valid = np.asarray(state.get("valid", []), dtype=bool)
            score = int(valid.sum())
            if score > 0:
                scored.append((score, tid, track))

        scored.sort(key=lambda x: x[0], reverse=True)

        sdc_id = metadata.get("sdc_id")
        sdc_track = self._safe_track_lookup(tracks, sdc_id) if sdc_id is not None else None

        ordered: List[Tuple[Any, Dict[str, Any]]] = []
        if sdc_track is not None:
            ordered.append((sdc_id, sdc_track))

        for _, tid, track in scored:
            if sdc_track is not None and track is sdc_track:
                continue
            ordered.append((tid, track))

        ordered = ordered[: self.max_agents]

        if not ordered:
            # Return shape-stable empty tensors.
            empty_t = 0
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((empty_t, 0, 2), dtype=np.float32),
                np.zeros((empty_t, 0), dtype=np.float32),
                np.zeros((empty_t, 0, 2), dtype=np.float32),
                np.zeros((empty_t, 0), dtype=bool),
                np.zeros((2,), dtype=np.float32),
            )

        horizon = max(
            int(np.asarray(track.get("state", {}).get("position", np.zeros((0, 3)))).shape[0])
            for _, track in ordered
        )

        n_agents = len(ordered)
        agent_ids = np.zeros((n_agents,), dtype=np.int32)
        agent_type_ids = np.zeros((n_agents,), dtype=np.int32)
        agent_shape = np.zeros((n_agents, 3), dtype=np.float32)

        agent_pos = np.zeros((horizon, n_agents, 2), dtype=np.float32)
        agent_heading = np.zeros((horizon, n_agents), dtype=np.float32)
        agent_vel = np.zeros((horizon, n_agents, 2), dtype=np.float32)
        agent_valid = np.zeros((horizon, n_agents), dtype=bool)

        for j, (tid, track) in enumerate(ordered):
            state = track.get("state", {})

            # Use object_id when available to preserve stable identifiers.
            object_id = track.get("metadata", {}).get("object_id", tid)
            try:
                agent_ids[j] = int(object_id)
            except Exception:
                agent_ids[j] = int(j)

            raw_type = str(track.get("type") or track.get("metadata", {}).get("type") or "UNSET").upper()
            agent_type_ids[j] = _AGENT_TYPE_TO_ID.get(raw_type, _AGENT_TYPE_TO_ID["OTHER"])

            pos = np.asarray(state.get("position", np.zeros((0, 3))), dtype=np.float32)
            vel = np.asarray(state.get("velocity", np.zeros((0, 2))), dtype=np.float32)
            heading = np.asarray(state.get("heading", np.zeros((0,))), dtype=np.float32)
            valid = np.asarray(state.get("valid", np.zeros((0,), dtype=bool)), dtype=bool)

            t = min(horizon, pos.shape[0], vel.shape[0], heading.shape[0], valid.shape[0])
            if t > 0:
                pos_xy = pos[:t, :2].copy()
                if self.center_to_map:
                    pos_xy = pos_xy - map_center[None, :2]
                vel_xy = vel[:t, :2].copy()
                heading_t = heading[:t].copy()
                valid_t = valid[:t]

                # ScenarioNet can carry placeholder/sentinel values in invalid
                # states. Zeroing invalid channels avoids contaminating rollout
                # initialization and metric reconstruction.
                pos_xy[~valid_t] = 0.0
                vel_xy[~valid_t] = 0.0
                heading_t[~valid_t] = 0.0

                agent_pos[:t, j] = pos_xy
                agent_vel[:t, j] = vel_xy
                agent_heading[:t, j] = heading_t
                agent_valid[:t, j] = valid_t

            # Shape is static per object, but stored as per-step arrays.
            length = np.asarray(state.get("length", np.zeros((0,))), dtype=np.float32)
            width = np.asarray(state.get("width", np.zeros((0,))), dtype=np.float32)
            height = np.asarray(state.get("height", np.zeros((0,))), dtype=np.float32)
            idx = min(max(current_time_index, 0), max(0, length.shape[0] - 1))

            def _pick(arr: np.ndarray) -> float:
                if arr.size == 0:
                    return 0.0
                val = float(arr[idx])
                if val > 0:
                    return val
                positive = arr[arr > 0]
                if positive.size > 0:
                    return float(positive.max())
                return 0.0

            agent_shape[j, 0] = _pick(length)
            agent_shape[j, 1] = _pick(width)
            agent_shape[j, 2] = _pick(height)

        # Scene-centered SDC anchor for map-token truncation by distance.
        sdc_current_xy = np.zeros((2,), dtype=np.float32)
        if n_agents > 0 and horizon > 0:
            sdc_t = min(max(current_time_index, 0), horizon - 1)
            sdc_current_xy = agent_pos[sdc_t, 0]

        return agent_ids, agent_type_ids, agent_shape, agent_pos, agent_heading, agent_vel, agent_valid, sdc_current_xy

    @staticmethod
    def _polyline_from_map_feature(feat: Dict[str, Any]) -> np.ndarray | None:
        points = None
        if "polyline" in feat:
            points = np.asarray(feat["polyline"], dtype=np.float32)
        elif "polygon" in feat:
            points = np.asarray(feat["polygon"], dtype=np.float32)
            if points.shape[0] > 0:
                points = np.concatenate([points, points[:1]], axis=0)
        elif "position" in feat:
            points = np.asarray(feat["position"], dtype=np.float32)

        if points is None or points.ndim != 2 or points.shape[0] == 0:
            return None
        if points.shape[1] == 2:
            points = np.concatenate([points, np.zeros((points.shape[0], 1), dtype=np.float32)], axis=-1)
        if points.shape[0] == 1:
            points = np.concatenate([points, points], axis=0)
        return points

    def _compute_map_center(self, map_features: Dict[Any, Any]) -> np.ndarray:
        min_xyz = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        max_xyz = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)

        any_point = False
        for feat in map_features.values():
            points = self._polyline_from_map_feature(feat)
            if points is None:
                continue
            any_point = True
            min_xyz = np.minimum(min_xyz, points.min(axis=0))
            max_xyz = np.maximum(max_xyz, points.max(axis=0))

        if not any_point:
            return np.zeros((3,), dtype=np.float32)
        return ((min_xyz + max_xyz) * 0.5).astype(np.float32)

    @staticmethod
    def _encode_map_type(raw_type: Any) -> np.ndarray:
        # Match the Adv-BMT 27-d map vector layout slots [13..24].
        t = str(raw_type or "").upper()
        flags = np.zeros((12,), dtype=np.float32)

        if "LANE" in t:
            flags[0] = 1.0
        if "SIDEWALK" in t:
            flags[1] = 1.0
        if "ROAD_EDGE" in t or "BOUNDARY" in t:
            flags[2] = 1.0
        if "ROAD_LINE" in t or "LINE" in t:
            flags[3] = 1.0
        if "BROKEN" in t:
            flags[4] = 1.0
        if "SOLID" in t:
            flags[5] = 1.0
        if "YELLOW" in t:
            flags[6] = 1.0
        if "WHITE" in t:
            flags[7] = 1.0
        if "DRIVEWAY" in t:
            flags[8] = 1.0
        if "CROSSWALK" in t:
            flags[9] = 1.0
        if "SPEED_BUMP" in t:
            flags[10] = 1.0
        if "STOP_SIGN" in t:
            flags[11] = 1.0
        return flags

    def _extract_map_features(
        self,
        map_features: Dict[Any, Any],
        *,
        map_center: np.ndarray,
        sdc_current_xy: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        packed_features: List[np.ndarray] = []
        packed_masks: List[np.ndarray] = []
        packed_positions: List[np.ndarray] = []

        for feat in map_features.values():
            points = self._polyline_from_map_feature(feat)
            if points is None:
                continue

            if self.center_to_map:
                points = points.copy()
                points[:, :2] = points[:, :2] - map_center[None, :2]
                points[:, 2] = points[:, 2] - map_center[2]

            starts = points[:-1]
            ends = points[1:]
            direction = ends - starts
            heading = np.arctan2(direction[:, 1], direction[:, 0]).astype(np.float32)
            seg_len = np.linalg.norm(direction[:, :2], axis=-1).astype(np.float32)
            cumulative = np.cumsum(seg_len)
            type_flags = self._encode_map_type(feat.get("type"))

            n_vec = starts.shape[0]
            cursor = 0
            while cursor < n_vec:
                end_idx = min(cursor + self.max_vectors_per_map_feature, n_vec)
                span = end_idx - cursor

                arr = np.zeros((self.max_vectors_per_map_feature, 27), dtype=np.float32)
                mask = np.zeros((self.max_vectors_per_map_feature,), dtype=bool)

                arr[:span, 0:3] = starts[cursor:end_idx]
                arr[:span, 3:6] = ends[cursor:end_idx]
                arr[:span, 6:9] = direction[cursor:end_idx]
                arr[:span, 9] = heading[cursor:end_idx]
                arr[:span, 10] = np.sin(heading[cursor:end_idx])
                arr[:span, 11] = np.cos(heading[cursor:end_idx])
                arr[:span, 12] = seg_len[cursor:end_idx]
                arr[:span, 13:25] = type_flags[None, :]
                arr[:span, 25] = cumulative[cursor:end_idx]
                arr[:span, 26] = 1.0
                mask[:span] = True

                midpoint = (arr[:span, 0:3] + arr[:span, 3:6]) * 0.5
                token_pos = midpoint.mean(axis=0) if span > 0 else np.zeros((3,), dtype=np.float32)

                packed_features.append(arr)
                packed_masks.append(mask)
                packed_positions.append(token_pos.astype(np.float32))
                cursor = end_idx

        if not packed_features:
            return (
                np.zeros((0, self.max_vectors_per_map_feature, 27), dtype=np.float32),
                np.zeros((0, self.max_vectors_per_map_feature), dtype=bool),
                np.zeros((0, 3), dtype=np.float32),
            )

        map_feature = np.stack(packed_features, axis=0)
        map_mask = np.stack(packed_masks, axis=0)
        map_pos = np.stack(packed_positions, axis=0)

        # Trim to configured token budget using nearest tokens to current SDC position.
        if map_feature.shape[0] > self.max_map_features:
            d = np.linalg.norm(map_pos[:, :2] - sdc_current_xy[None, :2], axis=-1)
            keep = np.argsort(d)[: self.max_map_features]
            map_feature = map_feature[keep]
            map_mask = map_mask[keep]
            map_pos = map_pos[keep]

        return map_feature.astype(np.float32), map_mask.astype(bool), map_pos.astype(np.float32)

    @staticmethod
    def _traffic_light_state_flags(state: Any) -> Tuple[float, float, float, float]:
        s = str(state or "").upper()

        is_green = float("GO" in s or "GREEN" in s)
        is_yellow = float("CAUTION" in s or "YELLOW" in s)
        is_red = float("STOP" in s or "RED" in s)
        is_unknown = float((is_green + is_yellow + is_red) == 0.0)
        return is_green, is_yellow, is_red, is_unknown

    def _extract_traffic_lights(
        self,
        dynamic_map_states: Dict[Any, Any],
        *,
        map_center: np.ndarray,
        horizon_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not dynamic_map_states or self.max_traffic_lights <= 0:
            return (
                np.zeros((horizon_steps, 0, 7), dtype=np.float32),
                np.zeros((horizon_steps, 0), dtype=bool),
                np.zeros((0, 3), dtype=np.float32),
            )

        keys = sorted(dynamic_map_states.keys(), key=lambda x: str(x))[: self.max_traffic_lights]
        n_lights = len(keys)

        feat = np.zeros((horizon_steps, n_lights, 7), dtype=np.float32)
        valid = np.zeros((horizon_steps, n_lights), dtype=bool)
        pos = np.zeros((n_lights, 3), dtype=np.float32)

        for j, k in enumerate(keys):
            tl = dynamic_map_states[k]
            stop = np.asarray(tl.get("stop_point", np.zeros((3,))), dtype=np.float32)
            if stop.shape[0] == 2:
                stop = np.concatenate([stop, np.zeros((1,), dtype=np.float32)], axis=0)
            if self.center_to_map:
                stop = stop - map_center
            pos[j] = stop[:3]

            state_seq = tl.get("state", {}).get("object_state", [])
            for t in range(horizon_steps):
                st = state_seq[t] if t < len(state_seq) else None
                g, y, r, u = self._traffic_light_state_flags(st)
                feat[t, j, :3] = pos[j]
                feat[t, j, 3] = g
                feat[t, j, 4] = y
                feat[t, j, 5] = r
                feat[t, j, 6] = u
                # Adv-BMT traffic-light tokens are present for the light once it exists.
                valid[t, j] = True

        return feat, valid, pos


def collate_nnx_scene_samples(
    samples: Sequence[NNXBMTSceneSample],
    *,
    max_time_steps: int | None = None,
    max_agents: int | None = None,
    max_map_features: int | None = None,
    max_vectors_per_map_feature: int | None = None,
    max_traffic_lights: int | None = None,
) -> Dict[str, Any]:
    """Pad scene samples to a batch for NNX scene/motion modules.

    This collate is intentionally explicit and framework-agnostic. Conversion to
    JAX arrays happens in the training step, not in data loading.

    If a sample has more timesteps than ``max_time_steps``, we truncate to the
    leading ``max_time_steps`` frames. This keeps full-length WOMD (20s) scenes
    compatible with Adv-BMT-style 91-step training runs.

    Passing ``None`` for a count-like maximum (agents / map features / traffic
    lights) means "pad to the batch-local maximum". This matches legacy
    Adv-BMT's `PADDING_TO_MAX=false` behavior as long as the individual samples
    have already been truncated to the configured ceilings at load time.
    """

    if not samples:
        raise ValueError("collate_nnx_scene_samples expects a non-empty sample list")

    bsz = len(samples)
    inferred_t = max(s.agent_position_xy.shape[0] for s in samples)
    inferred_n = max(s.agent_position_xy.shape[1] for s in samples)
    inferred_m = max(s.map_feature.shape[0] for s in samples)
    inferred_v = max(s.map_feature.shape[1] if s.map_feature.ndim == 3 else 0 for s in samples)
    inferred_l = max(s.traffic_light_feature.shape[1] for s in samples)

    t_max = int(max_time_steps) if max_time_steps is not None else inferred_t
    n_max = int(max_agents) if max_agents is not None else inferred_n
    m_max = int(max_map_features) if max_map_features is not None else inferred_m
    v_max = (
        int(max_vectors_per_map_feature)
        if max_vectors_per_map_feature is not None
        else inferred_v
    )
    l_max = int(max_traffic_lights) if max_traffic_lights is not None else inferred_l

    agent_ids = np.zeros((bsz, n_max), dtype=np.int32)
    agent_type_ids = np.zeros((bsz, n_max), dtype=np.int32)
    agent_shape = np.zeros((bsz, n_max, 3), dtype=np.float32)

    agent_pos = np.zeros((bsz, t_max, n_max, 2), dtype=np.float32)
    agent_heading = np.zeros((bsz, t_max, n_max), dtype=np.float32)
    agent_vel = np.zeros((bsz, t_max, n_max, 2), dtype=np.float32)
    agent_valid = np.zeros((bsz, t_max, n_max), dtype=bool)

    map_feature = np.zeros((bsz, m_max, v_max, 27), dtype=np.float32)
    map_feature_valid = np.zeros((bsz, m_max, v_max), dtype=bool)
    map_pos = np.zeros((bsz, m_max, 3), dtype=np.float32)

    tl_feature = np.zeros((bsz, t_max, l_max, 7), dtype=np.float32)
    tl_valid = np.zeros((bsz, t_max, l_max), dtype=bool)
    tl_pos = np.zeros((bsz, l_max, 3), dtype=np.float32)

    scenario_ids: List[str] = []
    current_time_index = np.zeros((bsz,), dtype=np.int32)
    dt_s = np.zeros((bsz,), dtype=np.float32)

    for b, s in enumerate(samples):
        scenario_ids.append(s.scenario_id)
        current_time_index[b] = int(s.current_time_index)
        dt_s[b] = float(s.dt_s)

        n = s.agent_position_xy.shape[1]
        t_raw = s.agent_position_xy.shape[0]
        m = s.map_feature.shape[0]
        v = s.map_feature.shape[1] if s.map_feature.ndim == 3 else 0
        l = s.traffic_light_feature.shape[1]

        if n > n_max or m > m_max or v > v_max or l > l_max:
            raise ValueError(
                "Sample exceeds requested collate shape: "
                f"sample(T={t_raw},N={n},M={m},V={v},L={l}) vs "
                f"collate(T={t_max},N={n_max},M={m_max},V={v_max},L={l_max})"
            )

        # Truncate long time horizons (e.g., 198-199 steps from full WOMD 20s
        # conversion) to keep fixed-shape batches for Adv-BMT-style training.
        t = min(t_raw, t_max)
        if t == 0:
            current_time_index[b] = 0
        else:
            current_time_index[b] = min(int(s.current_time_index), t - 1)

        agent_ids[b, :n] = s.agent_ids
        agent_type_ids[b, :n] = s.agent_type_ids
        agent_shape[b, :n] = s.agent_shape

        agent_pos[b, :t, :n] = s.agent_position_xy[:t]
        agent_heading[b, :t, :n] = s.agent_heading[:t]
        agent_vel[b, :t, :n] = s.agent_velocity_xy[:t]
        agent_valid[b, :t, :n] = s.agent_valid_mask[:t]

        if m > 0 and v > 0:
            map_feature[b, :m, :v] = s.map_feature
            map_feature_valid[b, :m, :v] = s.map_feature_valid_mask
            map_pos[b, :m] = s.map_position

        if l > 0:
            tl_feature[b, :t, :l] = s.traffic_light_feature[:t]
            tl_valid[b, :t, :l] = s.traffic_light_valid_mask[:t]
            tl_pos[b, :l] = s.traffic_light_position

    return {
        "scenario_ids": scenario_ids,
        "collate_shape": {
            "batch_size": bsz,
            "time_steps": t_max,
            "agents": n_max,
            "map_features": m_max,
            "vectors_per_map_feature": v_max,
            "traffic_lights": l_max,
        },
        "current_time_index": current_time_index,
        "dt_s": dt_s,
        "agent_ids": agent_ids,
        "agent_type_ids": agent_type_ids,
        "agent_shape": agent_shape,
        "agent_position_xy": agent_pos,
        "agent_heading": agent_heading,
        "agent_velocity_xy": agent_vel,
        "agent_valid_mask": agent_valid,
        "map_feature": map_feature,
        "map_feature_valid_mask": map_feature_valid,
        "map_position": map_pos,
        "traffic_light_feature": tl_feature,
        "traffic_light_valid_mask": tl_valid,
        "traffic_light_position": tl_pos,
    }
