from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
LEGACY_SRC = ROOT / "src" / "Adv-BMT"
VENDORED_SCENARIONET = ROOT / "scenarionet"
VENDORED_METADRIVE = ROOT / "metadrive"
for path in (VENDORED_SCENARIONET, VENDORED_METADRIVE, LEGACY_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from bmt.counterfactual.sdc_semantic_control import (
    build_sdc_semantic_dataset_fields,
    compute_family_gate_torch,
    first_divergence_onset_m,
    project_points_to_family_paths_torch,
)
from bmt.models.motionlm_lightning import MotionLMLightning


def test_first_divergence_onset_detects_first_sustained_rise():
    arc = np.asarray([0.0, 2.0, 4.0, 6.0, 8.0], dtype=np.float32)
    sep = np.asarray([0.05, 0.08, 0.28, 0.31, 0.9], dtype=np.float32)
    onset = first_divergence_onset_m(arc, sep, threshold=0.25, min_run=2)
    assert np.isfinite(onset)
    assert abs(float(onset) - 4.0) < 1e-5


def test_build_sdc_semantic_dataset_fields_emits_expected_shapes():
    row = {
        "schema_version": "sdc_semantic_control_v1",
        "sdc_id": "42",
        "requested_semantic_label": "left",
        "requested_semantic_confidence": 0.93,
        "use_for_training": True,
        "source_kind": "alternative_sdc_path",
        "candidate_family_path_ids": ["sdc_path_3", "sdc_path_11"],
        "candidate_family_confidences": [0.93, 0.81],
        "candidate_family_resampled_paths_world": [
            [[0.0, 0.0], [0.0, 2.0], [1.0, 4.0]],
            [[0.0, 0.0], [0.2, 2.0], [1.5, 4.0]],
        ],
        "candidate_family_resampled_path_tangents_world": [
            [[0.0, 1.0], [0.0, 1.0], [0.4, 0.9]],
            [[0.0, 1.0], [0.1, 0.99], [0.6, 0.8]],
        ],
        "candidate_family_arc_lengths_m": [
            [0.0, 2.0, 4.2],
            [0.0, 2.0, 4.4],
        ],
        "candidate_family_divergence_onsets_m": [2.0, 2.5],
    }
    fields = build_sdc_semantic_dataset_fields(
        scenario_id="scene_a",
        decoder_track_names=["1", "42", "99"],
        horizon=6,
        row=row,
        require_trainable=True,
        include_stop=True,
    )
    assert int(fields["cf/sdc_semantic_label_id"]) >= 0
    assert fields["cf/sdc_family_path_polylines_world"].shape == (2, 3, 2)
    assert fields["cf/sdc_family_path_tangents_world"].shape == (2, 3, 2)
    assert fields["cf/sdc_family_arc_lengths"].shape == (2, 3)
    assert fields["cf/sdc_family_path_mask"].shape == (2, 3)
    assert fields["cf/sdc_family_divergence_onsets"].shape == (2,)
    assert fields["cf/decision_agent_mask"].tolist() == [0.0, 1.0, 0.0]
    assert int(fields["cf/sdc_control_available"]) == 1


def test_project_points_to_family_paths_and_gate_are_monotonic():
    points = torch.tensor([[[0.0, 0.5], [0.0, 2.5], [0.0, 5.5]]], dtype=torch.float32)
    family_xy = torch.tensor(
        [[
            [[0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [0.0, 6.0]],
            [[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],
        ]],
        dtype=torch.float32,
    )
    family_mask = torch.tensor([[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32)
    tangents = torch.tensor(
        [[
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            [[0.44, 0.89], [0.44, 0.89], [0.44, 0.89], [0.44, 0.89]],
        ]],
        dtype=torch.float32,
    )
    arc = torch.tensor([[[0.0, 2.0, 4.0, 6.0], [0.0, 2.2, 4.4, 6.6]]], dtype=torch.float32)
    projection = project_points_to_family_paths_torch(
        points,
        family_path_polylines_world=family_xy,
        family_path_mask=family_mask,
        family_path_tangents_world=tangents,
        family_path_arc_lengths=arc,
    )
    nearest_arc = projection["nearest_arc"]
    assert nearest_arc.shape == (1, 3, 2)
    assert torch.all(nearest_arc[:, 1:, 0] >= nearest_arc[:, :-1, 0])
    gate = compute_family_gate_torch(nearest_arc, torch.tensor([[2.0, 4.0]], dtype=torch.float32), bandwidth_m=1.0)
    assert gate.shape == (1, 3, 2)
    assert float(gate[0, 0, 0]) < float(gate[0, 2, 0])


def test_policy_teacher_sync_copies_loaded_student_weights():
    class _Dummy:
        pass

    student = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
    teacher = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))

    with torch.no_grad():
        for parameter in student.parameters():
            parameter.fill_(1.25)
        for parameter in teacher.parameters():
            parameter.zero_()

    dummy = _Dummy()
    dummy.model = student
    dummy.policy_teacher = teacher
    dummy.policy_teacher_sync_report = None

    report = MotionLMLightning.sync_policy_teacher_from_student(dummy)

    assert report["teacher_present"] is True
    assert report["fully_synced"] is True
    assert report["num_teacher_keys"] == report["num_loaded_keys"]
    for key, value in teacher.state_dict().items():
        assert torch.allclose(value, student.state_dict()[key])
