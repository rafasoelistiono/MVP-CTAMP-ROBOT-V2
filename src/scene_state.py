"""
Live scene state reader.

get_live_scene(executor) reads actual MuJoCo physics state (data.xpos) and
returns a world-state dict with the same schema as _parse_scene() in
align_cubes_ompl_only.py, so the LLM planner and task scripts can consume
it without caring whether data came from XML or live sim.

Key difference from _parse_scene():
  - Object positions reflect the current simulation state, not the XML defaults.
  - Includes arm end-effector position and current finger width.
  - Includes a 'held' field per object (True if object is in the gripper).
  - Obstacle positions are also live (detects if an obstacle was displaced).
"""

from __future__ import annotations

import math
import mujoco


# Mirror the constants used in align_cubes_ompl_only.py.
REACH_MIN_M = 0.30
REACH_MAX_M = 0.82
REACH_BORDERLINE_M = 0.78
REACH_HARD_M = 0.82
OBSTACLE_TOO_CLOSE_M = 0.12
OBSTACLE_NEAR_M = 0.18
TARGET_CERAMIC_BUFFER_M = 0.13
BASE_XY = [-0.4, 0.0]

# Z threshold below which an object is considered fallen off the table.
FALLEN_Z_THRESHOLD = 0.70


def get_live_scene(executor) -> dict:
    """
    Read the current MuJoCo simulation state and return a world-state dict.

    Parameters
    ----------
    executor : module
        The imported executor module (has .model, .data, .name_to_cube,
        .ee_id, .finger_ctrl_adr, .arm_ctrl_adr).

    Returns
    -------
    dict with keys:
        table           — same schema as _parse_scene()
        robot           — base_xy, reach_min_m, reach_max_m, ee_xyz, finger_width
        goal_center     — [x, y, z]  (unchanged from model; not physics-driven)
        objects         — list of all tracked objects (movable + ceramic)
        movable_objects — subset of objects that are movable and not fallen
        fallen_objects  — objects that have fallen off the table
        ceramic_obstacles — fragile/static obstacle objects with live positions
        held_object     — object_id of the currently held object, or None
        counts          — {movable, ceramic, fallen}
        safety          — inflated ceramic regions (live positions)
    """
    model = executor.model
    data = executor.data
    mujoco.mj_forward(model, data)

    # --- end-effector ---
    ee_xyz = None
    if executor.ee_id is not None:
        p = data.xpos[executor.ee_id]
        ee_xyz = [round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)]

    # --- finger width (proxy for whether gripper is open/holding) ---
    finger_width = None
    try:
        finger_width = round(float(data.ctrl[executor.finger_ctrl_adr]), 4)
    except Exception:
        pass

    # --- table bounds (static — read from model geometry once) ---
    table = _read_table_bounds(model)

    # --- goal center (static — read from model body if present) ---
    goal_center = _read_goal_center(model)

    # --- determine held object ---
    held_object = _detect_held_object(executor, table)

    # --- movable objects (cubes / circles) ---
    objects = []
    for obj_id, body_id in executor.name_to_cube.items():
        pos = data.xpos[body_id]
        xyz = [round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3)]
        obj_class = _class_from_name(obj_id)
        radius = _radius_for_class(obj_class)
        reach_dist = math.dist(xyz[:2], BASE_XY)
        fallen = xyz[2] < FALLEN_Z_THRESHOLD
        held = (obj_id == held_object)

        obstacle_dist = _min_obstacle_dist(xyz[:2], executor)
        obstacle_status = _obstacle_status_from_dist(obstacle_dist)
        reach_status = _reach_status_from_dist(reach_dist)

        objects.append({
            "id": obj_id,
            "class": obj_class,
            "position": xyz,
            "radius": radius,
            "movable": True,
            "fragile": False,
            "held": held,
            "fallen": fallen,
            "reach_dist_m": round(reach_dist, 4),
            "reach_status": reach_status,
            "obstacle_dist_m": round(obstacle_dist, 4),
            "obstacle_status": obstacle_status,
        })

    # --- ceramic obstacles (live positions) ---
    ceramic_obstacles = []
    for name, body_id in executor._obstacle_ids.items():
        pos = data.xpos[body_id]
        xyz = [round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3)]
        ceramic_obstacles.append({
            "id": name,
            "class": "obstacle",
            "position": xyz,
            "radius": 0.05,
            "movable": False,
            "fragile": True,
        })

    movable_objects = [o for o in objects if not o["fallen"]]
    fallen_objects = [o for o in objects if o["fallen"]]

    return {
        "table": table,
        "robot": {
            "base_xy": BASE_XY,
            "reach_min_m": REACH_MIN_M,
            "reach_max_m": REACH_MAX_M,
            "ee_xyz": ee_xyz,
            "finger_width": finger_width,
        },
        "goal_center": goal_center,
        "objects": objects,
        "movable_objects": movable_objects,
        "fallen_objects": fallen_objects,
        "ceramic_obstacles": ceramic_obstacles,
        "held_object": held_object,
        "counts": {
            "movable": len(movable_objects),
            "ceramic": len(ceramic_obstacles),
            "fallen": len(fallen_objects),
        },
        "safety": {
            "inflated_ceramic_regions": [
                {
                    "id": o["id"],
                    "center_xy": o["position"][:2],
                    "radius": o["radius"] + TARGET_CERAMIC_BUFFER_M,
                }
                for o in ceramic_obstacles
            ]
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_held_object(executor, table: dict) -> str | None:
    """
    Heuristic: an object is held if its z is above HELD_Z_THRESHOLD and the
    gripper is partially closed (finger_width < 0.035).
    """
    try:
        finger_width = float(executor.data.ctrl[executor.finger_ctrl_adr])
    except Exception:
        finger_width = 0.04

    if finger_width >= 0.035:
        return None  # gripper open — not holding anything

    data = executor.data
    best_id = None
    best_z = 0.90  # must be above this to count as held
    for obj_id, body_id in executor.name_to_cube.items():
        z = float(data.xpos[body_id][2])
        if z > best_z:
            best_z = z
            best_id = obj_id
    return best_id


def _min_obstacle_dist(xy: list, executor) -> float:
    if not executor._obstacle_ids:
        return 999.0
    data = executor.data
    min_dist = 999.0
    for body_id in executor._obstacle_ids.values():
        obs_pos = data.xpos[body_id]
        d = math.dist(xy, [float(obs_pos[0]), float(obs_pos[1])])
        if d < min_dist:
            min_dist = d
    return min_dist


def _obstacle_status_from_dist(dist: float) -> str:
    if dist <= OBSTACLE_TOO_CLOSE_M:
        return "TOO_CLOSE"
    if dist <= OBSTACLE_NEAR_M:
        return "NEAR"
    return "CLEAR"


def _reach_status_from_dist(dist: float) -> str:
    if dist > REACH_HARD_M:
        return "HARD"
    if dist > REACH_BORDERLINE_M:
        return "BORDERLINE"
    return "OK"


def _class_from_name(name: str) -> str:
    lower = name.lower()
    if "cube" in lower:
        return "cube"
    if "circle" in lower:
        return "circle"
    if "cylinder" in lower:
        return "cylinder"
    return "object"


def _radius_for_class(obj_class: str) -> float:
    return 0.043 if obj_class == "cube" else 0.030


def _read_table_bounds(model) -> dict:
    try:
        body_id = model.body("table").id
        geom_id = model.body_geomadr[body_id]
        body_pos = model.body_pos[body_id]
        geom_pos = model.geom_pos[geom_id]
        size = model.geom_size[geom_id]
        cx = body_pos[0] + geom_pos[0]
        cy = body_pos[1] + geom_pos[1]
        cz = body_pos[2] + geom_pos[2]
        return {
            "x_range": [round(cx - size[0] + 0.05, 3), round(cx + size[0] - 0.05, 3)],
            "y_range": [round(cy - size[1] + 0.05, 3), round(cy + size[1] - 0.05, 3)],
            "z_top": round(cz + size[2], 3),
        }
    except Exception:
        return {"x_range": [-0.55, 0.55], "y_range": [-0.75, 0.75], "z_top": 0.80}


def _read_goal_center(model) -> list:
    try:
        body_id = model.body("goal_area").id
        pos = model.body_pos[body_id]
        return [round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4)]
    except Exception:
        return [0.22, -0.06, 0.806]
