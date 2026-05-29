from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from align_cubes_ompl_only import _build_aligned_cube_targets, _obstacle_status
from align_tabung_ompl_only import _build_aligned_cylinder_targets
from ctamp_task_utils import OBSTACLE_POSITIONS, prepare_scene_variant, write_summary_csv
from stack_cubes_ompl_only import _build_cube_stack_targets


def test_static_no_obstacle_cube_targets_are_reachable():
    scene = prepare_scene_variant("group_no_obs")
    _, move_order, slots, skipped = _build_aligned_cube_targets(str(scene))
    assert move_order == ["cube1", "cube2", "cube3", "cube4"]
    assert not skipped
    assert all(object_id in slots for object_id in move_order)


def test_static_no_obstacle_cylinder_targets_are_reachable():
    scene = prepare_scene_variant("group_no_obs")
    _, move_order, slots, skipped = _build_aligned_cylinder_targets(str(scene))
    assert move_order == ["circle4", "circle3", "circle2", "circle1"]
    assert not skipped
    assert all(object_id in slots for object_id in move_order)


def test_group_obstacle_targets_keep_all_objects_after_validated_mapping():
    scene = prepare_scene_variant("group_obs")
    cube_world, cube_order, cube_slots, cube_skipped = _build_aligned_cube_targets(str(scene))
    cylinder_world, cylinder_order, cylinder_slots, cylinder_skipped = _build_aligned_cylinder_targets(str(scene))

    assert set(cube_order) == {"cube1", "cube2", "cube3", "cube4"}
    assert cylinder_order == ["circle4", "circle3", "circle2", "circle1"]
    assert not cube_skipped
    assert not cylinder_skipped
    assert all(object_id in cube_slots for object_id in cube_order)
    assert all(object_id in cylinder_slots for object_id in cylinder_order)

    cube_statuses = {
        obj["id"]: _obstacle_status(obj, cube_world)
        for obj in cube_world["movable_objects"]
        if obj.get("class") == "cube"
    }
    cylinder_statuses = {
        obj["id"]: _obstacle_status(obj, cylinder_world)
        for obj in cylinder_world["movable_objects"]
        if obj.get("class") in {"circle", "cylinder"} or obj["id"].lower().startswith("circle")
    }
    assert OBSTACLE_POSITIONS["obstacle1"][1] < 0 < OBSTACLE_POSITIONS["obstacle2"][1]
    assert OBSTACLE_POSITIONS["obstacle1"][0] < OBSTACLE_POSITIONS["obstacle2"][0]
    assert cube_statuses["cube2"] == "NEAR"
    assert cylinder_statuses["circle4"] == "CLEAR"


def test_ungroup_obstacle_targets_keep_all_objects_with_near_challenges():
    scene = prepare_scene_variant("ungroup_obs")
    cube_world, cube_order, cube_slots, cube_skipped = _build_aligned_cube_targets(str(scene))
    cylinder_world, cylinder_order, cylinder_slots, cylinder_skipped = _build_aligned_cylinder_targets(str(scene))

    assert set(cube_order) == {"cube1", "cube2", "cube3", "cube4"}
    assert set(cylinder_order) == {"circle1", "circle2", "circle3", "circle4"}
    assert not cube_skipped
    assert not cylinder_skipped
    assert all(object_id in cube_slots for object_id in cube_order)
    assert all(object_id in cylinder_slots for object_id in cylinder_order)

    cube_statuses = {
        obj["id"]: _obstacle_status(obj, cube_world)
        for obj in cube_world["movable_objects"]
        if obj.get("class") == "cube"
    }
    cylinder_statuses = {
        obj["id"]: _obstacle_status(obj, cylinder_world)
        for obj in cylinder_world["movable_objects"]
        if obj.get("class") in {"circle", "cylinder"} or obj["id"].lower().startswith("circle")
    }
    assert cube_statuses["cube4"] == "NEAR"
    assert cylinder_statuses["circle1"] == "NEAR"
    assert cylinder_statuses["circle4"] == "CLEAR"


def test_cube_stack_targets_have_three_base_cubes_and_one_top_cube_for_all_scenes():
    for scene_key in ["group_no_obs", "ungroup_no_obs", "group_obs", "ungroup_obs"]:
        scene = prepare_scene_variant(scene_key)
        _, move_order, slots, skipped = _build_cube_stack_targets(str(scene))

        assert set(move_order) == {"cube1", "cube2", "cube3", "cube4"}
        assert not skipped
        levels = [slots[object_id]["level"] for object_id in move_order]
        assert levels.count(0) == 3
        assert levels.count(1) == 1
        for object_id in move_order:
            slot = slots[object_id]
            if slot["level"] == 0:
                assert slot["support_object_id"] is None
            else:
                assert slot["support_object_id"] in move_order
                assert slots[slot["support_object_id"]]["level"] == 0


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
