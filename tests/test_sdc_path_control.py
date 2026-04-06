from __future__ import annotations

import numpy as np
import torch

from bmt.counterfactual.sdc_path_control import (
    ResampledLocalPath,
    build_sdc_path_dataset_fields,
    compute_path_separability_profile,
    polyline_arc_lengths,
    polyline_headings,
    project_points_to_path_torch,
    resample_polyline_xy,
)


def test_resample_and_separability_profile_increases_after_divergence():
    selected_xy = resample_polyline_xy(
        np.asarray(
            [
                [0.0, 0.0],
                [0.0, 6.0],
                [0.0, 12.0],
                [3.0, 18.0],
            ],
            dtype=np.float32,
        ),
        spacing_m=2.0,
    )
    competing_xy = resample_polyline_xy(
        np.asarray(
            [
                [0.0, 0.0],
                [0.0, 6.0],
                [0.0, 12.0],
                [-3.0, 18.0],
            ],
            dtype=np.float32,
        ),
        spacing_m=2.0,
    )
    selected = ResampledLocalPath(
        waypoints_xy=selected_xy,
        headings=polyline_headings(selected_xy),
        arc_lengths_m=polyline_arc_lengths(selected_xy),
    )
    competing = {
        "alt": ResampledLocalPath(
            waypoints_xy=competing_xy,
            headings=polyline_headings(competing_xy),
            arc_lengths_m=polyline_arc_lengths(competing_xy),
        )
    }
    profile = compute_path_separability_profile(selected, competing, scale_m=4.0, heading_weight_m=0.0)
    separability = np.asarray(profile["separability"], dtype=np.float32)
    assert separability.shape[0] == selected_xy.shape[0]
    assert separability[0] <= 0.05
    assert separability[-1] > separability[0]


def test_build_sdc_path_dataset_fields_emits_expected_keys():
    row = {
        "schema_version": "sdc_path_control_v1",
        "sdc_id": "42",
        "semantic_label": "left",
        "semantic_confidence": 0.9,
        "use_for_training": True,
        "source_kind": "alternative_sdc_path",
        "selected_path_id": "sdc_path_3",
        "candidate_count": 4,
        "selected_path_waypoints_local_xy": [[0.0, 0.0], [0.0, 2.0], [0.5, 4.0]],
        "selected_path_waypoints_local_heading": [1.5708, 1.5708, 1.2],
        "selected_path_arc_lengths_m": [0.0, 2.0, 4.1],
        "selected_path_separability": [0.0, 0.2, 0.8],
    }
    fields = build_sdc_path_dataset_fields(
        scenario_id="scene_a",
        decoder_track_names=["1", "42", "99"],
        horizon=6,
        row=row,
        require_trainable=True,
        include_stop=True,
    )
    assert int(fields["cf/sdc_semantic_label_id"]) >= 0
    assert fields["cf/sdc_path_waypoints"].shape == (3, 5)
    assert fields["cf/sdc_path_waypoint_mask"].shape == (3,)
    assert fields["cf/sdc_path_separability"].shape == (3,)
    assert fields["cf/decision_agent_mask"].tolist() == [0.0, 1.0, 0.0]
    assert int(fields["cf/sdc_control_available"]) == 1


def test_project_points_to_path_torch_returns_monotonic_arc_for_forward_points():
    points = torch.tensor([[[0.0, 0.5], [0.0, 2.0], [0.0, 4.5]]], dtype=torch.float32)
    path_xy = torch.tensor([[[0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [0.0, 6.0]]], dtype=torch.float32)
    path_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    path_heading = torch.tensor([[1.5708, 1.5708, 1.5708, 1.5708]], dtype=torch.float32)
    path_arc = torch.tensor([[0.0, 2.0, 4.0, 6.0]], dtype=torch.float32)
    path_sep = torch.tensor([[0.0, 0.2, 0.6, 1.0]], dtype=torch.float32)
    projection = project_points_to_path_torch(
        points,
        path_waypoints_local_xy=path_xy,
        path_waypoint_mask=path_mask,
        path_waypoint_heading=path_heading,
        path_waypoint_arc=path_arc,
        path_waypoint_separability=path_sep,
    )
    nearest_arc = projection["nearest_arc"]
    assert nearest_arc.shape == (1, 3)
    assert torch.all(nearest_arc[:, 1:] >= nearest_arc[:, :-1])
