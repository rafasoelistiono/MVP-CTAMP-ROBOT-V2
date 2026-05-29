"""
Tests for src/scene_state.py.

These tests use a lightweight mock executor so they run without MuJoCo,
and a separate integration test that requires the full WSL environment.
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for relative in ("src", "scripts"):
    path = str(ROOT / relative)
    if path not in sys.path:
        sys.path.insert(0, path)

import scene_state
from scene_state import (
    FALLEN_Z_THRESHOLD,
    OBSTACLE_NEAR_M,
    OBSTACLE_TOO_CLOSE_M,
    REACH_HARD_M,
    _class_from_name,
    _detect_held_object,
    _min_obstacle_dist,
    _obstacle_status_from_dist,
    _radius_for_class,
    _reach_status_from_dist,
    get_live_scene,
)


@pytest.fixture(autouse=True)
def patch_mj_forward(monkeypatch):
    """mujoco.mj_forward requires real MjModel/MjData — replace with no-op for unit tests."""
    monkeypatch.setattr(scene_state.mujoco, "mj_forward", lambda m, d: None)


# ---------------------------------------------------------------------------
# Helpers to build a mock executor
# ---------------------------------------------------------------------------

def _make_model(table_body_id=1, goal_body_id=2):
    model = MagicMock()

    # body() lookup by name
    def body_by_name(name):
        b = MagicMock()
        if name == "table":
            b.id = table_body_id
        elif name == "goal_area":
            b.id = goal_body_id
        else:
            raise KeyError(name)
        return b

    model.body.side_effect = body_by_name
    model.nbody = 10

    # body_pos: table at origin with half-size [0.6, 0.8, 0.05]
    body_pos = np.zeros((10, 3))
    body_pos[table_body_id] = [0.0, 0.0, 0.775]
    body_pos[goal_body_id] = [0.22, -0.06, 0.806]
    model.body_pos = body_pos

    # body_geomadr: geom index for table body
    model.body_geomadr = np.zeros(10, dtype=int)
    model.body_geomadr[table_body_id] = 0

    # geom_pos and geom_size for table
    model.geom_pos = np.zeros((5, 3))
    model.geom_size = np.zeros((5, 3))
    model.geom_size[0] = [0.6, 0.8, 0.05]

    return model


def _make_executor(object_positions: dict, obstacle_positions: dict = None,
                   finger_width: float = 0.04, ee_xyz=None):
    """
    Build a mock executor module.

    object_positions: {name: [x, y, z]}
    obstacle_positions: {name: [x, y, z]}
    """
    if obstacle_positions is None:
        obstacle_positions = {}

    # All body ids: 0=table, 1=goal, 2+ objects, then obstacles
    all_bodies = list(object_positions.keys()) + list(obstacle_positions.keys())
    body_id_map = {name: i + 10 for i, name in enumerate(all_bodies)}

    n_bodies = 10 + len(all_bodies) + 5
    xpos = np.zeros((n_bodies, 3))
    for name, xyz in object_positions.items():
        xpos[body_id_map[name]] = xyz
    for name, xyz in obstacle_positions.items():
        xpos[body_id_map[name]] = xyz

    ee_id = None
    if ee_xyz is not None:
        ee_id = n_bodies - 1
        xpos[ee_id] = ee_xyz

    model = _make_model()

    data = MagicMock()
    data.xpos = xpos
    ctrl = np.zeros(10)
    ctrl[7] = finger_width  # finger_ctrl_adr = 7
    data.ctrl = ctrl

    ex = MagicMock()
    ex.model = model
    ex.data = data
    ex.name_to_cube = {name: body_id_map[name] for name in object_positions}
    ex._obstacle_ids = {name: body_id_map[name] for name in obstacle_positions}
    ex.ee_id = ee_id
    ex.finger_ctrl_adr = 7

    return ex


# ---------------------------------------------------------------------------
# Unit tests — helper functions
# ---------------------------------------------------------------------------

def test_class_from_name():
    assert _class_from_name("cube1") == "cube"
    assert _class_from_name("circle2") == "circle"
    assert _class_from_name("cylinder3") == "cylinder"
    assert _class_from_name("unknown") == "object"


def test_radius_for_class():
    assert _radius_for_class("cube") == pytest.approx(0.043)
    assert _radius_for_class("circle") == pytest.approx(0.030)
    assert _radius_for_class("cylinder") == pytest.approx(0.030)


def test_reach_status():
    assert _reach_status_from_dist(0.40) == "OK"
    assert _reach_status_from_dist(0.79) == "BORDERLINE"
    assert _reach_status_from_dist(0.90) == "HARD"


def test_obstacle_status():
    assert _obstacle_status_from_dist(0.30) == "CLEAR"
    assert _obstacle_status_from_dist(0.15) == "NEAR"
    assert _obstacle_status_from_dist(0.10) == "TOO_CLOSE"
    assert _obstacle_status_from_dist(OBSTACLE_TOO_CLOSE_M) == "TOO_CLOSE"
    assert _obstacle_status_from_dist(OBSTACLE_NEAR_M) == "NEAR"


def test_min_obstacle_dist_no_obstacles():
    ex = _make_executor({"cube1": [0.2, 0.0, 0.83]}, obstacle_positions={})
    dist = _min_obstacle_dist([0.2, 0.0], ex)
    assert dist == pytest.approx(999.0)


def test_min_obstacle_dist_with_obstacle():
    ex = _make_executor(
        {"cube1": [0.2, 0.0, 0.83]},
        obstacle_positions={"obstacle1": [0.2, 0.15, 0.83]},
    )
    dist = _min_obstacle_dist([0.2, 0.0], ex)
    assert dist == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Unit tests — detect_held_object
# ---------------------------------------------------------------------------

def test_detect_held_object_gripper_open():
    ex = _make_executor({"cube1": [0.2, 0.0, 0.97]}, finger_width=0.04)
    assert _detect_held_object(ex, {}) is None


def test_detect_held_object_gripper_closed_object_elevated():
    ex = _make_executor({"cube1": [0.2, 0.0, 0.97]}, finger_width=0.011)
    assert _detect_held_object(ex, {}) == "cube1"


def test_detect_held_object_gripper_closed_object_on_table():
    # Gripper closed but object still on table (z=0.83) — not held
    ex = _make_executor({"cube1": [0.2, 0.0, 0.83]}, finger_width=0.011)
    assert _detect_held_object(ex, {}) is None


def test_detect_held_object_picks_highest():
    ex = _make_executor({
        "cube1": [0.2, 0.0, 0.83],
        "cube2": [0.3, 0.0, 0.97],
        "cube3": [0.1, 0.0, 0.93],
    }, finger_width=0.011)
    assert _detect_held_object(ex, {}) == "cube2"


# ---------------------------------------------------------------------------
# Integration tests — get_live_scene
# ---------------------------------------------------------------------------

def test_get_live_scene_basic_structure():
    ex = _make_executor({
        "cube1": [0.2, 0.0, 0.83],
        "cube2": [0.3, 0.1, 0.83],
    })
    scene = get_live_scene(ex)

    assert "table" in scene
    assert "robot" in scene
    assert "movable_objects" in scene
    assert "fallen_objects" in scene
    assert "ceramic_obstacles" in scene
    assert "held_object" in scene
    assert "counts" in scene
    assert "safety" in scene


def test_get_live_scene_object_positions_are_live():
    ex = _make_executor({"cube1": [0.25, 0.10, 0.83]})
    scene = get_live_scene(ex)
    obj = next(o for o in scene["movable_objects"] if o["id"] == "cube1")
    assert obj["position"] == pytest.approx([0.25, 0.10, 0.83], abs=1e-3)


def test_get_live_scene_fallen_object_excluded_from_movable():
    ex = _make_executor({
        "cube1": [0.2, 0.0, 0.83],
        "cube2": [0.3, 0.1, 0.50],  # below FALLEN_Z_THRESHOLD
    })
    scene = get_live_scene(ex)
    movable_ids = [o["id"] for o in scene["movable_objects"]]
    fallen_ids = [o["id"] for o in scene["fallen_objects"]]

    assert "cube1" in movable_ids
    assert "cube2" not in movable_ids
    assert "cube2" in fallen_ids
    assert scene["counts"]["fallen"] == 1


def test_get_live_scene_held_object_detected():
    ex = _make_executor({
        "cube1": [0.2, 0.0, 0.97],
        "cube2": [0.3, 0.1, 0.83],
    }, finger_width=0.011)
    scene = get_live_scene(ex)
    assert scene["held_object"] == "cube1"


def test_get_live_scene_no_held_when_open():
    ex = _make_executor({"cube1": [0.2, 0.0, 0.97]}, finger_width=0.04)
    scene = get_live_scene(ex)
    assert scene["held_object"] is None


def test_get_live_scene_reach_status_computed():
    # cube at reach ~0.81m (borderline) from base [-0.4, 0.0]
    ex = _make_executor({"cube1": [0.40, 0.05, 0.83]})
    scene = get_live_scene(ex)
    obj = scene["movable_objects"][0]
    expected_dist = math.dist([0.40, 0.05], [-0.4, 0.0])
    assert obj["reach_dist_m"] == pytest.approx(expected_dist, abs=0.01)
    assert obj["reach_status"] in ("OK", "BORDERLINE", "HARD")


def test_get_live_scene_obstacle_status_computed():
    ex = _make_executor(
        {"cube1": [0.2, 0.0, 0.83]},
        obstacle_positions={"obs1": [0.2, 0.15, 0.83]},
    )
    scene = get_live_scene(ex)
    obj = scene["movable_objects"][0]
    assert obj["obstacle_dist_m"] == pytest.approx(0.15, abs=0.01)
    assert obj["obstacle_status"] == "NEAR"


def test_get_live_scene_obstacle_positions_are_live():
    ex = _make_executor(
        {"cube1": [0.2, 0.0, 0.83]},
        obstacle_positions={"obs1": [-0.18, -0.30, 0.83]},
    )
    scene = get_live_scene(ex)
    assert len(scene["ceramic_obstacles"]) == 1
    obs = scene["ceramic_obstacles"][0]
    assert obs["position"] == pytest.approx([-0.18, -0.30, 0.83], abs=1e-3)


def test_get_live_scene_counts_correct():
    ex = _make_executor({
        "cube1": [0.2, 0.0, 0.83],
        "cube2": [0.3, 0.1, 0.40],  # fallen
        "circle1": [0.15, -0.1, 0.83],
    }, obstacle_positions={"obs1": [-0.18, -0.30, 0.83]})
    scene = get_live_scene(ex)
    assert scene["counts"]["movable"] == 2
    assert scene["counts"]["fallen"] == 1
    assert scene["counts"]["ceramic"] == 1


def test_get_live_scene_ee_xyz_reported():
    ex = _make_executor({"cube1": [0.2, 0.0, 0.83]}, ee_xyz=[0.21, 0.01, 1.10])
    scene = get_live_scene(ex)
    assert scene["robot"]["ee_xyz"] == pytest.approx([0.21, 0.01, 1.10], abs=1e-3)


def test_get_live_scene_robot_base_xy():
    ex = _make_executor({"cube1": [0.2, 0.0, 0.83]})
    scene = get_live_scene(ex)
    assert scene["robot"]["base_xy"] == [-0.4, 0.0]
    assert scene["robot"]["reach_min_m"] == pytest.approx(0.30)
    assert scene["robot"]["reach_max_m"] == pytest.approx(0.82)


def test_get_live_scene_safety_regions_use_live_obstacle_positions():
    ex = _make_executor(
        {"cube1": [0.2, 0.0, 0.83]},
        obstacle_positions={"obs1": [0.42, 0.18, 0.83]},
    )
    scene = get_live_scene(ex)
    regions = scene["safety"]["inflated_ceramic_regions"]
    assert len(regions) == 1
    assert regions[0]["center_xy"] == pytest.approx([0.42, 0.18], abs=1e-3)


def test_get_live_scene_schema_compatible_with_parse_scene():
    """Keys required by task scripts and llm_planner must all be present."""
    ex = _make_executor({"cube1": [0.2, 0.0, 0.83]})
    scene = get_live_scene(ex)

    required_top = {"table", "robot", "goal_center", "objects",
                    "movable_objects", "ceramic_obstacles", "counts", "safety"}
    assert required_top.issubset(scene.keys())

    required_table = {"x_range", "y_range", "z_top"}
    assert required_table.issubset(scene["table"].keys())

    required_robot = {"base_xy", "reach_min_m", "reach_max_m"}
    assert required_robot.issubset(scene["robot"].keys())

    if scene["movable_objects"]:
        required_obj = {"id", "class", "position", "radius", "movable", "fragile"}
        assert required_obj.issubset(scene["movable_objects"][0].keys())
