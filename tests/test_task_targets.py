from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from align_cubes_ompl_only import _build_aligned_cube_targets
from align_tabung_ompl_only import _build_aligned_cylinder_targets
from ctamp_task_utils import prepare_scene_variant, write_summary_csv


def test_static_no_obstacle_cube_targets_are_reachable():
    scene = prepare_scene_variant("group_no_obs")
    _, move_order, slots, skipped = _build_aligned_cube_targets(str(scene))
    assert move_order == ["cube1", "cube2", "cube3", "cube4"]
    assert not skipped
    assert all(object_id in slots for object_id in move_order)


def test_static_no_obstacle_cylinder_targets_are_reachable():
    scene = prepare_scene_variant("group_no_obs")
    _, move_order, slots, skipped = _build_aligned_cylinder_targets(str(scene))
    assert move_order == ["circle1", "circle2", "circle3", "circle4"]
    assert not skipped
    assert all(object_id in slots for object_id in move_order)


def test_ompl_dependent_regression_is_explicitly_skipped_when_missing():
    if importlib.util.find_spec("ompl") is None:
        pytest.skip("OMPL Python bindings are not installed in this environment")
    assert Path("models/panda.xml").exists()


def test_summary_csv_records_failure_reason_counts(tmp_path):
    csv_path = write_summary_csv(
        "align_cubes",
        "group_no_obs",
        {
            "success": False,
            "objects_moved": 3,
            "objects_total": 4,
            "failed": [{"object_id": "cube4", "stage": "pick", "failure_reason": "object_not_lifted_after_pick"}],
            "duration_ms": 10,
        },
        tmp_path,
    )
    with csv_path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    assert row["object_success_rate"] == "0.75"
    assert row["no_obs_success_rate"] == "0.75"
    assert json.loads(row["failure_reason_counts_json"]) == {"object_not_lifted_after_pick": 1}
