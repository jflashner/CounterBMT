"""Evidence builders for VLM-based DAG-conformance alignment scoring."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from counter_bmt_v2.contracts import BayesianDAG, Intervention, ScenarioInput, TimestampedFrame, TrajectoryRollout
from counter_bmt_v2.data.frame_render import render_scenario_frames
from counter_bmt_v2.data.scenarionet import NNXBMTSceneSample


@dataclass
class AlignmentEvidenceBundle:
    base_frames: List[TimestampedFrame]
    overlay_frames: List[TimestampedFrame]
    frames_for_vlm: List[TimestampedFrame]
    dag_text: str
    intervention_text: str


def _time_indices(total: int, count: int) -> np.ndarray:
    if total <= 0:
        return np.zeros((0,), dtype=np.int32)
    k = max(1, int(count))
    if total == 1:
        return np.zeros((k,), dtype=np.int32)
    idx = np.linspace(0, total - 1, num=k, dtype=np.int32)
    return np.clip(idx, 0, total - 1)


def _copy_existing_frames(scene: ScenarioInput, out_dir: Path, num_frames: int) -> List[TimestampedFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    kept: List[TimestampedFrame] = []
    src_frames = [f for f in scene.frames if Path(f.path).is_file()]
    if not src_frames:
        return kept
    idx = _time_indices(len(src_frames), num_frames)
    for i, j in enumerate(idx.tolist()):
        src = Path(src_frames[j].path)
        dst = out_dir / f"base_{i:03d}{src.suffix or '.png'}"
        shutil.copy2(str(src), str(dst))
        kept.append(TimestampedFrame(path=str(dst), timestamp_s=float(src_frames[j].timestamp_s)))
    return kept


def _render_from_sample(scene: ScenarioInput, out_dir: Path, num_frames: int, max_agents_render: int) -> List[TimestampedFrame]:
    sample = scene.metadata.get("nnx_sample") if isinstance(scene.metadata, dict) else None
    if not isinstance(sample, NNXBMTSceneSample):
        return []
    return render_scenario_frames(sample, out_dir, num_frames=num_frames, max_agents=max_agents_render)


def _render_fallback_ego(scene: ScenarioInput, out_dir: Path, num_frames: int) -> List[TimestampedFrame]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    traj = np.asarray(scene.ego_trajectory_xy, dtype=np.float32) if scene.ego_trajectory_xy is not None else np.zeros((0, 2), dtype=np.float32)
    if traj.ndim != 2 or traj.shape[1] < 2 or traj.shape[0] <= 0:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    idx = _time_indices(traj.shape[0], num_frames)
    frames: List[TimestampedFrame] = []
    for i, t in enumerate(idx.tolist()):
        fig, ax = plt.subplots(figsize=(7, 7))
        obs = traj[: t + 1, :2]
        ax.plot(obs[:, 0], obs[:, 1], color="#2563EB", linewidth=2.0, alpha=0.9, label="ego_observed")
        ax.scatter([obs[-1, 0]], [obs[-1, 1]], c="#10B981", s=45, label="ego_now")
        ax.set_title(f"{scene.scenario_id} | t={t}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        path = out_dir / f"base_{i:03d}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        frames.append(TimestampedFrame(path=str(path), timestamp_s=float(t)))
    return frames


def build_alignment_frame_pack(
    *,
    scene: ScenarioInput,
    out_dir: str | Path,
    num_frames: int,
    max_agents_render: int,
) -> List[TimestampedFrame]:
    """Create base evidence frames for one scene."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    frames = _copy_existing_frames(scene, root, num_frames=int(num_frames))
    if frames:
        return frames

    frames = _render_from_sample(scene, root, num_frames=int(num_frames), max_agents_render=int(max_agents_render))
    if frames:
        return frames

    return _render_fallback_ego(scene, root, num_frames=int(num_frames))


def render_rollout_overlay(
    *,
    base_frames: Sequence[TimestampedFrame],
    scene: ScenarioInput,
    rollout: TrajectoryRollout,
    out_dir: str | Path,
) -> List[TimestampedFrame]:
    """Overlay predicted rollout onto base frames."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pred = np.asarray(rollout.trajectory_xy, dtype=np.float32)
    obs = np.asarray(scene.ego_trajectory_xy, dtype=np.float32) if scene.ego_trajectory_xy is not None else np.zeros((0, 2), dtype=np.float32)
    overlays: List[TimestampedFrame] = []

    for i, base in enumerate(base_frames):
        img = plt.imread(str(base.path))
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.imshow(img)
        ax.axis("off")
        # Overlay legend only; trajectory is drawn in an inset in metric space to avoid projection mismatch.
        inset = fig.add_axes([0.62, 0.06, 0.34, 0.34])
        if obs.ndim == 2 and obs.shape[0] > 1:
            inset.plot(obs[:, 0], obs[:, 1], color="#2563EB", linewidth=1.5, alpha=0.9, label="obs_ego")
        if pred.ndim == 2 and pred.shape[0] > 1:
            inset.plot(pred[:, 0], pred[:, 1], color="#DC2626", linestyle="--", linewidth=1.6, alpha=0.9, label="pred_rollout")
            inset.scatter([pred[-1, 0]], [pred[-1, 1]], c="#DC2626", s=12)
        inset.set_title("Trajectory Overlay", fontsize=8)
        inset.grid(True, alpha=0.2)
        inset.tick_params(labelsize=6)
        if (obs.shape[0] > 1) or (pred.shape[0] > 1):
            inset.legend(loc="best", fontsize=6)
        fig.text(
            0.01,
            0.99,
            f"VLM alignment overlay | step frame {i + 1}/{len(base_frames)}",
            ha="left",
            va="top",
            fontsize=8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 3},
        )
        out_file = out_path / f"overlay_{i:03d}.png"
        fig.savefig(out_file, dpi=130)
        plt.close(fig)
        overlays.append(TimestampedFrame(path=str(out_file), timestamp_s=float(base.timestamp_s)))

    return overlays


def build_compact_dag_text(dag: BayesianDAG, intervention: Intervention) -> str:
    nodes = list(dag.nodes.values())
    edges = list(dag.edges)
    if intervention.assignments:
        intervention_lines = ["Sampled DAG assignment:"]
        for node_id in intervention.assignment_order or sorted(intervention.assignments.keys()):
            if node_id in intervention.assignments:
                intervention_lines.append(f"- {node_id}={intervention.assignments[node_id]}")
    else:
        intervention_lines = [
            f"Intervention variable: {intervention.variable}",
            f"Intervention value: {intervention.value}",
        ]
    lines = [
        f"Scenario: {dag.scenario_id}",
        *intervention_lines,
        f"Nodes ({len(nodes)}):",
    ]
    for n in nodes[:20]:
        val = str(n.value)
        if len(val) > 64:
            val = val[:61] + "..."
        lines.append(f"- {n.node_id} [{n.node_type}] value={val}")
    lines.append(f"Edges ({len(edges)}):")
    for e in edges[:40]:
        lines.append(f"- {e.parent_id} -> {e.child_id} (conf={float(e.confidence):.2f}, mech={e.mechanism})")
    if dag.cpts:
        lines.append(f"CPT nodes: {', '.join(sorted(dag.cpts.keys())[:20])}")
    return "\n".join(lines)


def build_alignment_evidence_bundle(
    *,
    scene: ScenarioInput,
    dag: BayesianDAG,
    intervention: Intervention,
    rollout: TrajectoryRollout,
    out_dir: str | Path,
    num_frames: int,
    max_agents_render: int,
) -> AlignmentEvidenceBundle:
    root = Path(out_dir)
    frames_base_dir = root / "frames_base"
    frames_overlay_dir = root / "frames_overlay"
    base = build_alignment_frame_pack(
        scene=scene,
        out_dir=frames_base_dir,
        num_frames=int(num_frames),
        max_agents_render=int(max_agents_render),
    )
    overlay = render_rollout_overlay(
        base_frames=base,
        scene=scene,
        rollout=rollout,
        out_dir=frames_overlay_dir,
    )
    frames_for_vlm: List[TimestampedFrame] = []
    for b, o in zip(base, overlay):
        frames_for_vlm.append(b)
        frames_for_vlm.append(o)
    return AlignmentEvidenceBundle(
        base_frames=list(base),
        overlay_frames=list(overlay),
        frames_for_vlm=frames_for_vlm,
        dag_text=build_compact_dag_text(dag, intervention),
        intervention_text=(
            ", ".join(
                f"{k}={intervention.assignments[k]}"
                for k in (intervention.assignment_order or sorted(intervention.assignments.keys()))
                if k in intervention.assignments
            )
            if intervention.assignments
            else f"{intervention.variable}={intervention.value}"
        ),
    )
