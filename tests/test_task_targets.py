from __future__ import annotations

import csv
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from align_cubes_ompl_only import (
    ALIGN_PLACE_Y_TOLERANCE_M,
    ALIGN_TARGET_ROW_BAND_M,
    _build_aligned_cube_targets,
    _check_aligned_target,
    _obstacle_status,
    _validate_aligned_targets,
)
from align_tabung_ompl_only import _build_aligned_cylinder_targets
from ctamp_task_utils import CUBE_HALF_EXTENTS, LONG_OBSTACLE_HALF_HEIGHT, OBSTACLE_POSITIONS, prepare_scene_variant, write_summary_csv
from stack_cubes_ompl_only import _build_cube_stack_targets, _rebuild_order_after_disturbance, _validate_moved_stack


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
    assert cube_statuses["cube2"] == "CLEAR"
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
    assert cube_statuses["cube4"] == "CLEAR"
    assert cylinder_statuses["circle1"] == "NEAR"
    assert cylinder_statuses["circle4"] == "CLEAR"


def test_align_targets_stay_on_one_goal_row_for_obstacle_scenes():
    for scene_key in ["group_obs", "ungroup_obs"]:
        scene = prepare_scene_variant(scene_key)
        _, cube_order, cube_slots, _ = _build_aligned_cube_targets(str(scene))
        _, cylinder_order, cylinder_slots, _ = _build_aligned_cylinder_targets(str(scene))

        cube_y_values = [cube_slots[object_id]["target_pose"][1] for object_id in cube_order]
        cylinder_y_values = [cylinder_slots[object_id]["target_pose"][1] for object_id in cylinder_order]

        assert max(cube_y_values) - min(cube_y_values) <= ALIGN_TARGET_ROW_BAND_M * 2
        assert max(cylinder_y_values) - min(cylinder_y_values) <= ALIGN_TARGET_ROW_BAND_M * 2


def test_aligned_target_validation_rejects_loose_y_drift():
    class FakeMujoco:
        @staticmethod
        def mj_forward(model, data):
            return None

    class FakeExecutor:
        mujoco = FakeMujoco()
        model = object()
        name_to_cube = {"cube1": 0, "cube2": 1}

        class data:
            xpos = [
                [0.22, -0.01, 0.83],
                [0.33, -0.06, 0.83],
            ]

    slots = {
        "cube1": {"target_pose": [0.22, -0.06, 0.83]},
        "cube2": {"target_pose": [0.33, -0.06, 0.83]},
    }

    aligned, details = _check_aligned_target(FakeExecutor(), "cube1", slots["cube1"]["target_pose"])
    invalid = _validate_aligned_targets(FakeExecutor(), ["cube1", "cube2"], slots)

    assert not aligned
    assert details["y_error"] > ALIGN_PLACE_Y_TOLERANCE_M
    assert invalid[0]["object_id"] == "cube1"
    assert invalid[0]["failure_reason"] == "object_not_aligned_with_target_line"


def test_long_obstacle_variants_keep_explicit_target_allocation_behavior():
    for scene_key in ["group_long_obs", "ungroup_long_obs"]:
        scene = prepare_scene_variant(scene_key)
        cube_world, cube_order, cube_slots, cube_skipped = _build_aligned_cube_targets(str(scene))
        cylinder_world, cylinder_order, cylinder_slots, cylinder_skipped = _build_aligned_cylinder_targets(str(scene))

        assert set(cube_order).issubset({"cube1", "cube2", "cube3", "cube4"})
        assert set(cylinder_order).issubset({"circle1", "circle2", "circle3", "circle4"})
        assert all(object_id in cube_slots for object_id in cube_order)
        assert all(object_id in cylinder_slots for object_id in cylinder_order)
        assert all(item.get("failure_reason") for item in cube_skipped + cylinder_skipped)
        assert any(
            item.get("failure_reason") == "object_blocked_by_long_obstacle_side_access"
            for item in cube_skipped + cylinder_skipped
        )
        assert cube_world["counts"]["ceramic"] == 2
        assert cylinder_world["counts"]["ceramic"] == 2
        assert all(
            obstacle["position"][2] == pytest.approx(0.806 + LONG_OBSTACLE_HALF_HEIGHT, abs=0.0001)
            for obstacle in cube_world["ceramic_obstacles"]
        )
        root = ET.parse(scene).getroot()
        long_obstacles = [
            body
            for body in root.findall(".//body")
            if body.get("name", "").startswith("obstacle")
        ]
        assert long_obstacles
        assert all(body.find("joint") is None for body in long_obstacles)


def test_cube_stack_targets_have_four_vertical_layers_for_all_scenes():
    for scene_key in ["group_no_obs", "ungroup_no_obs", "group_obs", "ungroup_obs"]:
        scene = prepare_scene_variant(scene_key)
        world_state, move_order, slots, skipped = _build_cube_stack_targets(str(scene))

        assert move_order == ["cube1", "cube2", "cube3", "cube4"]
        assert not skipped
        assert [slots[object_id]["level"] for object_id in move_order] == [0, 1, 2, 3]
        assert slots["cube1"]["support_object_id"] is None
        assert slots["cube2"]["support_object_id"] == "cube1"
        assert slots["cube3"]["support_object_id"] == "cube2"
        assert slots["cube4"]["support_object_id"] == "cube3"

        base_x, base_y, base_z = slots["cube1"]["target_pose"][:3]
        goal_x, goal_y, _ = world_state["goal_center"]
        assert (base_x, base_y) == pytest.approx((goal_x, goal_y), abs=0.0001)
        for index, object_id in enumerate(move_order):
            x, y, z = slots[object_id]["target_pose"][:3]
            assert (x, y) == (base_x, base_y)
            assert z == pytest.approx(base_z + index * 0.06, abs=0.0001)


def test_generated_cube_footprint_is_wider_but_layer_height_stays_same():
    scene = prepare_scene_variant("group_no_obs")
    root = ET.parse(scene).getroot()
    cube_geom = root.find(".//body[@name='cube1']/geom")
    assert cube_geom is not None
    assert tuple(float(value) for value in cube_geom.get("size").split()) == pytest.approx(CUBE_HALF_EXTENTS)
    assert CUBE_HALF_EXTENTS[0] > CUBE_HALF_EXTENTS[2]
    assert CUBE_HALF_EXTENTS[1] > CUBE_HALF_EXTENTS[2]


def test_stack_integrity_detects_disturbed_tower_without_repositioning():
    class FakeMujoco:
        @staticmethod
        def mj_forward(model, data):
            return None

    class FakeExecutor:
        mujoco = FakeMujoco()
        model = object()
        name_to_cube = {"cube1": 0, "cube2": 1, "cube3": 2, "cube4": 3}

        class data:
            xpos = [
                [0.22, -0.06, 0.83],
                [0.30, -0.12, 0.83],
                [0.30, -0.12, 0.89],
                [0.30, -0.12, 0.95],
            ]

    slots = {
        "cube1": {"target_pose": [0.22, -0.06, 0.83], "support_object_id": None},
        "cube2": {"target_pose": [0.22, -0.06, 0.89], "support_object_id": "cube1"},
        "cube3": {"target_pose": [0.22, -0.06, 0.95], "support_object_id": "cube2"},
        "cube4": {"target_pose": [0.22, -0.06, 1.01], "support_object_id": "cube3"},
    }

    invalid = _validate_moved_stack(
        FakeExecutor(),
        ["cube1", "cube2", "cube3"],
        slots,
        table_top=0.80,
        failure_reason="placed_stack_disturbed_after_place",
    )

    assert invalid[0]["object_id"] == "cube2"
    assert invalid[0]["failure_reason"] == "placed_stack_disturbed_after_place"
    assert _rebuild_order_after_disturbance(["cube1", "cube2", "cube3", "cube4"], invalid) == [
        "cube2",
        "cube3",
        "cube4",
    ]


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
