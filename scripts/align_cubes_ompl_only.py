#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ctamp_task_utils import (
    apply_conservative_motion_defaults,
    apply_scene_motion_defaults,
    make_event_log_path,
    normalize_scene_key,
    obstacle_mode_for_scene,
    prepare_scene_variant,
    safe_process_exit,
    write_summary_csv,
)

REACH_BORDERLINE_M = 0.78
REACH_HARD_M = 0.82
OBSTACLE_TOO_CLOSE_M = 0.10
OBSTACLE_NEAR_M = 0.18
MAX_PICK_RETRY_SOURCE_SHIFT_M = 0.18
TARGET_CERAMIC_BUFFER_M = 0.13
PICK_OBSTACLE_BLOCKED_M = 0.10
LONG_OBSTACLE_SIDE_ACCESS_BLOCKED_M = 0.22
ALIGN_TARGET_ROW_BAND_M = 0.018
ALIGN_PLACE_X_TOLERANCE_M = 0.055
ALIGN_PLACE_Y_TOLERANCE_M = 0.035
ALIGN_ROW_SPREAD_TOLERANCE_M = 0.045
ALIGN_PLACED_Z_THRESHOLD_M = 0.87


def _long_obstacle_mode(world_state) -> bool:
    table_top = float(world_state["table"]["z_top"])
    return any(
        float(obstacle["position"][2]) - table_top > 0.20
        for obstacle in world_state.get("ceramic_obstacles", [])
    )


def _min_ceramic_distance_xy(obj, world_state) -> float:
    ox, oy = obj["position"][:2]
    distances = []
    for obstacle in world_state.get("ceramic_obstacles", []):
        cx, cy = obstacle["position"][:2]
        distances.append(math.dist((ox, oy), (cx, cy)))
    return min(distances) if distances else 999.0


def _obstacle_status(obj, world_state) -> str:
    distance = _min_ceramic_distance_xy(obj, world_state)
    if distance <= OBSTACLE_TOO_CLOSE_M:
        return "TOO_CLOSE"
    if distance <= OBSTACLE_NEAR_M:
        return "NEAR"
    return "CLEAR"


def _reach_distance_xy(obj, world_state) -> float:
    return math.dist(obj["position"][:2], world_state["robot"]["base_xy"])


def _reach_status(obj, world_state) -> str:
    distance = _reach_distance_xy(obj, world_state)
    if distance > REACH_HARD_M:
        return "HARD"
    if distance > REACH_BORDERLINE_M:
        return "BORDERLINE"
    return "OK"


def _runtime_object_pose(executor, object_id: str):
    executor.mujoco.mj_forward(executor.model, executor.data)
    pos = executor.data.xpos[executor.name_to_cube[object_id]]
    return [float(pos[0]), float(pos[1]), float(pos[2])]


def _terminal_pick_failure_reason(executor, object_id: str, world_state) -> str | None:
    pos = _runtime_object_pose(executor, object_id)
    table_top = float(world_state["table"]["z_top"])
    reach_distance = math.dist(pos[:2], world_state["robot"]["base_xy"])
    source = next((obj for obj in world_state["movable_objects"] if obj["id"] == object_id), None)
    source_shift = math.dist(pos[:2], source["position"][:2]) if source else 0.0
    if pos[2] < table_top - 0.08:
        return "object_displaced_below_table"
    if reach_distance > world_state["robot"]["reach_max_m"] + 0.10:
        return "object_outside_robot_reach_after_displacement"
    if source_shift > MAX_PICK_RETRY_SOURCE_SHIFT_M:
        return "object_displaced_after_failed_pick"
    return None


def _blocked_grasp_failure_reason(executor, object_id: str) -> str | None:
    pos = _runtime_object_pose(executor, object_id)
    obstacle_distance_fn = getattr(executor, "_min_obstacle_xy_distance", None)
    if obstacle_distance_fn is None:
        return None
    obstacle_distance = obstacle_distance_fn(pos)
    if obstacle_distance < PICK_OBSTACLE_BLOCKED_M:
        return "object_blocked_by_obstacle_grasp_clearance"
    return None


def _reachable(x: float, y: float, world_state) -> bool:
    robot = world_state["robot"]
    bx, by = robot["base_xy"]
    distance = math.dist((x, y), (bx, by))
    return robot["reach_min_m"] <= distance <= robot["reach_max_m"]


def _clear_from_ceramic(x: float, y: float, radius: float, world_state) -> bool:
    for region in world_state["safety"]["inflated_ceramic_regions"]:
        if math.dist((x, y), tuple(region["center_xy"])) < region["radius"] + radius:
            return False
    return True


def _target_xy_ok(x: float, y: float, radius: float, world_state, occupied=()) -> bool:
    table = world_state["table"]
    if not (table["x_range"][0] < x < table["x_range"][1] and table["y_range"][0] < y < table["y_range"][1]):
        return False
    if not _reachable(x, y, world_state):
        return False
    if not _clear_from_ceramic(x, y, radius, world_state):
        return False
    for ox, oy, other_radius in occupied:
        if math.dist((x, y), (ox, oy)) < radius + other_radius + 0.018:
            return False
    return True


def _search_safe_target_xy(
    base_x: float,
    base_y: float,
    radius: float,
    world_state,
    occupied=(),
    y_min: float = None,
    y_max: float = None,
    x_min: float = None,
    x_max: float = None,
):
    table = world_state["table"]
    table_x_min, table_x_max = table["x_range"]
    table_y_min, table_y_max = table["y_range"]
    candidates = []
    for ring in range(0, 7):
        for dx in range(-ring, ring + 1):
            for dy in range(-ring, ring + 1):
                if ring and abs(dx) != ring and abs(dy) != ring:
                    continue
                x = min(max(base_x + dx * 0.035, table_x_min + radius), table_x_max - radius)
                y = min(max(base_y + dy * 0.035, table_y_min + radius), table_y_max - radius)
                candidates.append((x, y))

    seen = set()
    unique = []
    for x, y in candidates:
        key = (round(x, 4), round(y, 4))
        if key not in seen:
            seen.add(key)
            unique.append((x, y))

    unique.sort(key=lambda xy: abs(xy[1] - base_y) * 3.0 + abs(xy[0] - base_x))
    for x, y in unique:
        if y_min is not None and y < y_min:
            continue
        if y_max is not None and y > y_max:
            continue
        if x_min is not None and x < x_min:
            continue
        if x_max is not None and x > x_max:
            continue
        if _target_xy_ok(x, y, radius, world_state, occupied):
            return x, y
    return None


def _check_aligned_target(
    executor,
    object_id: str,
    target_pose,
    z_threshold: float = ALIGN_PLACED_Z_THRESHOLD_M,
):
    actual = _runtime_object_pose(executor, object_id)
    target_x = float(target_pose[0])
    target_y = float(target_pose[1])
    x_error = abs(actual[0] - target_x)
    y_error = abs(actual[1] - target_y)
    on_table = actual[2] < z_threshold
    aligned = (
        on_table
        and x_error <= ALIGN_PLACE_X_TOLERANCE_M
        and y_error <= ALIGN_PLACE_Y_TOLERANCE_M
    )
    details = {
        "object_id": object_id,
        "actual_xyz": [round(v, 4) for v in actual],
        "target_xyz": [round(float(v), 4) for v in target_pose[:3]],
        "x_error": round(x_error, 4),
        "y_error": round(y_error, 4),
        "on_table": on_table,
        "failure_reason": None if aligned else "object_not_aligned_with_target_line",
    }
    return aligned, details


def _validate_aligned_targets(
    executor,
    object_ids,
    slots_by_object,
    z_threshold: float = ALIGN_PLACED_Z_THRESHOLD_M,
):
    invalid = []
    invalid_ids = set()
    actual_y_by_object = {}

    for object_id in object_ids:
        if object_id not in slots_by_object:
            continue
        aligned, details = _check_aligned_target(
            executor,
            object_id,
            slots_by_object[object_id]["target_pose"],
            z_threshold=z_threshold,
        )
        actual_y_by_object[object_id] = details["actual_xyz"][1]
        if not aligned:
            invalid.append(details)
            invalid_ids.add(object_id)

    if len(actual_y_by_object) > 1:
        y_values = list(actual_y_by_object.values())
        y_spread = max(y_values) - min(y_values)
        if y_spread > ALIGN_ROW_SPREAD_TOLERANCE_M:
            for object_id, actual_y in actual_y_by_object.items():
                if object_id in invalid_ids:
                    continue
                target_pose = slots_by_object[object_id]["target_pose"]
                invalid.append({
                    "object_id": object_id,
                    "actual_xyz": _runtime_object_pose(executor, object_id),
                    "target_xyz": [round(float(v), 4) for v in target_pose[:3]],
                    "x_error": 0.0,
                    "y_error": round(abs(actual_y - float(target_pose[1])), 4),
                    "row_y_spread": round(y_spread, 4),
                    "on_table": True,
                    "failure_reason": "aligned_row_not_straight",
                })
                invalid_ids.add(object_id)

    return invalid


def _parse_scene(scene_file: str):
    scene_path = Path(scene_file)
    if not scene_path.exists():
        scene_path = ROOT_DIR / "models" / scene_file
    scene_path = scene_path.resolve()
    roots = _load_roots(scene_path)
    bodies = []
    for root in roots:
        worldbody = root.find("worldbody")
        if worldbody is not None:
            bodies.extend(_iter_bodies(worldbody))

    table = _extract_table(bodies)
    objects = []
    goal_center = [0.22, -0.06, 0.806]
    for body in bodies:
        name = body.get("name", "")
        if name == "goal_area":
            goal_center = _vec(body.get("pos"), goal_center)
            continue
        if not name or name == "table" or _is_robot_body(name):
            continue
        geoms = body.findall("geom")
        if not geoms:
            continue
        pos = _vec(body.get("pos"), [0.0, 0.0, 0.0])
        shape = geoms[0].get("type", "mesh")
        radius = _radius_from_geoms(geoms)
        fragile = any(token in name.lower() for token in ("obstacle", "vase", "glass", "ceramic"))
        movable = any(joint.get("type") == "free" for joint in body.findall("joint")) and not fragile
        objects.append({
            "id": name,
            "class": _class_from_name_shape(name, shape),
            "position": pos,
            "radius": radius,
            "movable": movable,
            "fragile": fragile,
        })

    movable_objects = [o for o in objects if o["movable"]]
    ceramic_obstacles = [o for o in objects if o["fragile"]]
    return {
        "table": table,
        "robot": {"base_xy": [-0.4, 0.0], "reach_min_m": 0.30, "reach_max_m": 0.82},
        "goal_center": goal_center,
        "objects": objects,
        "movable_objects": movable_objects,
        "ceramic_obstacles": ceramic_obstacles,
        "counts": {"movable": len(movable_objects), "ceramic": len(ceramic_obstacles)},
        "safety": {
            "inflated_ceramic_regions": [
                {"id": o["id"], "center_xy": o["position"][:2], "radius": o["radius"] + TARGET_CERAMIC_BUFFER_M}
                for o in ceramic_obstacles
            ]
        },
    }


def _load_roots(scene_path: Path):
    root = ET.parse(scene_path).getroot()
    roots = [root]
    for include in root.findall("include"):
        include_file = include.get("file")
        if include_file:
            include_path = (scene_path.parent / include_file).resolve()
            if include_path.exists():
                roots.extend(_load_roots(include_path))
    return roots


def _iter_bodies(parent):
    for body in parent.findall("body"):
        yield body
        yield from _iter_bodies(body)


def _extract_table(bodies):
    for body in bodies:
        if body.get("name") != "table":
            continue
        geom = body.find("geom")
        if geom is None:
            break
        body_pos = _vec(body.get("pos"), [0.0, 0.0, 0.0])
        geom_pos = _vec(geom.get("pos"), [0.0, 0.0, 0.0])
        size = _vec(geom.get("size"), [0.6, 0.8, 0.05])
        center = [body_pos[i] + geom_pos[i] for i in range(3)]
        return {
            "x_range": [center[0] - size[0] + 0.05, center[0] + size[0] - 0.05],
            "y_range": [center[1] - size[1] + 0.05, center[1] + size[1] - 0.05],
            "z_top": center[2] + size[2],
        }
    return {"x_range": [-0.55, 0.55], "y_range": [-0.75, 0.75], "z_top": 0.80}


def _radius_from_geoms(geoms) -> float:
    radius = 0.03
    for geom in geoms:
        size = _vec(geom.get("size"), [])
        geom_type = geom.get("type", "mesh")
        if geom_type == "box" and len(size) >= 2:
            radius = max(radius, math.hypot(size[0], size[1]))
        elif geom_type in {"cylinder", "sphere", "capsule"} and size:
            radius = max(radius, size[0])
        elif geom_type == "ellipsoid" and len(size) >= 2:
            radius = max(radius, size[0], size[1])
    return radius


def _class_from_name_shape(name: str, shape: str) -> str:
    lower = name.lower()
    if "cube" in lower or shape == "box":
        return "cube"
    if "circle" in lower:
        return "circle"
    if "cylinder" in lower or shape == "cylinder":
        return "cylinder"
    if shape == "ellipsoid":
        return "circle"
    return shape


def _is_robot_body(name: str) -> bool:
    return name.startswith("link") or name in {"hand", "left_finger", "right_finger"}


def _vec(raw, default):
    if raw is None or raw.strip() == "":
        return list(default)
    return [float(part) for part in raw.split()]


def _build_aligned_cube_targets(scene_file: str, spacing: float = 0.125):
    world_state = _parse_scene(scene_file)
    cubes = [
        obj for obj in world_state["movable_objects"]
        if obj.get("class") == "cube"
    ]
    cubes.sort(key=lambda obj: (
        _obstacle_status(obj, world_state) == "TOO_CLOSE",
        _obstacle_status(obj, world_state) == "NEAR",
        _reach_status(obj, world_state) == "HARD",
        _reach_status(obj, world_state) == "BORDERLINE",
        _reach_distance_xy(obj, world_state),
        obj["position"][0],
        obj["position"][1],
    ))
    skipped = []
    eligible_cubes = []
    long_obstacle_mode = _long_obstacle_mode(world_state)
    for cube in cubes:
        obstacle_status = _obstacle_status(cube, world_state)
        reach_status = _reach_status(cube, world_state)
        obstacle_distance = _min_ceramic_distance_xy(cube, world_state)
        if long_obstacle_mode and obstacle_distance <= LONG_OBSTACLE_SIDE_ACCESS_BLOCKED_M:
            skipped.append({
                "object_id": cube["id"],
                "stage": "precheck",
                "failure_reason": "object_blocked_by_long_obstacle_side_access",
                "obstacle_distance": round(obstacle_distance, 4),
                "obstacle_status": obstacle_status,
            })
            continue
        if obstacle_status == "TOO_CLOSE":
            skipped.append({
                "object_id": cube["id"],
                "stage": "precheck",
                "failure_reason": "object_too_close_to_obstacle",
                "obstacle_distance": round(_min_ceramic_distance_xy(cube, world_state), 4),
                "obstacle_status": obstacle_status,
            })
            continue
        if reach_status == "HARD":
            skipped.append({
                "object_id": cube["id"],
                "stage": "precheck",
                "failure_reason": "object_outside_conservative_reach",
                "reach_distance": round(_reach_distance_xy(cube, world_state), 4),
                "reach_status": reach_status,
            })
            continue
        eligible_cubes.append(cube)

    table = world_state["table"]
    goal_x, goal_y, _ = world_state["goal_center"]
    row_y = goal_y
    start_x = goal_x - spacing * (len(eligible_cubes) - 1) / 2.0
    slots = {}
    occupied = []
    target_cubes = sorted(eligible_cubes, key=lambda obj: (obj["position"][0], obj["position"][1], obj["id"]))
    for index, cube in enumerate(target_cubes):
        radius = cube.get("radius", 0.043)
        base_x = start_x + index * spacing
        target_xy = _search_safe_target_xy(
            base_x,
            row_y,
            radius,
            world_state,
            occupied,
            y_min=row_y - ALIGN_TARGET_ROW_BAND_M,
            y_max=row_y + ALIGN_TARGET_ROW_BAND_M,
        )
        if target_xy is None:
            skipped.append({
                "object_id": cube["id"],
                "stage": "target_allocation",
                "failure_reason": "no_safe_aligned_target_found",
                "desired_xy": [round(base_x, 4), round(row_y, 4)],
            })
            continue
        x, y = target_xy
        occupied.append((x, y, radius))
        slots[cube["id"]] = {
            "slot_id": f"line_slot_{index + 1:02d}",
            "target_pose": [round(x, 4), round(y, 4), round(table["z_top"] + 0.03, 4), 1.0, 0.0, 0.0, 0.0],
        }

    return world_state, list(slots.keys()), slots, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Arrange cubes into one straight aligned row using OMPL + MuJoCo collision checking only."
    )
    parser.add_argument(
        "--object",
        nargs="+",
        default=["group", "no", "obs"],
        help="Scene: group no obs, ungroup no obs, group obs, ungroup obs, group long obs, ungroup long obs.",
    )
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-hint-cache", action="store_true", help="Disable HintCache adaptive learning (use fixed defaults).")
    parser.add_argument("--place-retries", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--settle-after-place", type=int, default=300, help=argparse.SUPPRESS)
    args = parser.parse_args()

    scene_key = normalize_scene_key(args.object)
    scene_file = prepare_scene_variant(scene_key)
    event_csv_path = make_event_log_path("align_cubes", scene_key, args.log_dir)
    os.environ["MODEL_FILE"] = str(scene_file)
    os.environ["CTAMP_EVENT_LOG_CSV"] = str(event_csv_path)
    os.environ["CTAMP_SCENARIO_TYPE"] = "static"
    os.environ["CTAMP_OBSTACLE_MODE"] = obstacle_mode_for_scene(scene_key)
    os.environ.setdefault("OMPL_ENABLED", "true")
    os.environ.setdefault("USE_IK_FALLBACK", "false")
    apply_scene_motion_defaults(scene_key)
    apply_conservative_motion_defaults()
    os.environ["ENABLE_VIEWER"] = "false" if args.no_viewer else "true"

    try:
        from ompl import base, geometric  # noqa: F401
    except ImportError:
        print("[ALIGN_CUBES] OMPL Python binding belum tersedia.")
        print("  pip install ompl")
        print("  python -c \"from ompl import base, geometric; print('ompl ok')\"")
        return 2

    try:
        world_state, move_order, slots_by_object, skipped_precheck = _build_aligned_cube_targets(
            scene_file=str(scene_file),
        )
    except ValueError as exc:
        print(f"[ALIGN_CUBES] Target line invalid: {exc}")
        return 1

    if not move_order:
        print("[ALIGN_CUBES] Tidak ada cube dengan akses aman untuk disusun.")
        if skipped_precheck:
            print("[ALIGN_CUBES] Skipped before target allocation:")
            for item in skipped_precheck:
                print(f"  - {item['object_id']}: {item['failure_reason']}")
        return 1

    print("[ALIGN_CUBES] Task          : susun kubus bersusun sejajar")
    print(f"[ALIGN_CUBES] Scene variant : {scene_key} ({scene_file})")
    print(f"[ALIGN_CUBES] Event CSV     : {event_csv_path}")
    print(f"[ALIGN_CUBES] Scene objects  : movable={world_state['counts']['movable']} ceramic={world_state['counts']['ceramic']}")
    print(f"[ALIGN_CUBES] Ceramic avoid  : {[o['id'] for o in world_state['ceramic_obstacles']]}")
    print(f"[ALIGN_CUBES] Move order     : {move_order}")
    if skipped_precheck:
        print("[ALIGN_CUBES] Skipped before target allocation:")
        for item in skipped_precheck:
            print(f"  - {item['object_id']}: {item['failure_reason']}")
    print("[ALIGN_CUBES] Target line:")
    for object_id in move_order:
        x, y, z = slots_by_object[object_id]["target_pose"][:3]
        obj = next(o for o in world_state["movable_objects"] if o["id"] == object_id)
        clearance = _min_ceramic_distance_xy(obj, world_state)
        print(
            f"  - {object_id:>6} -> ({x:.3f}, {y:.3f}, {z:.3f}) "
            f"reach={_reach_distance_xy(obj, world_state):.3f}m status={_reach_status(obj, world_state)} "
            f"clearance={clearance:.3f}m obs_status={_obstacle_status(obj, world_state)}"
        )

    import executor
    import feedback
    from exec_trace import flush as flush_trace
    from exec_trace import log_event

    if not getattr(executor, "_OMPL_AVAILABLE", False):
        print("[ALIGN_CUBES] Executor tidak melihat OMPL planner aktif.")
        return 2

    if args.no_hint_cache:
        print("[ALIGN_CUBES] HintCache disabled (--no-hint-cache).")
    else:
        executor.init_hint_cache(log_dir=args.log_dir, scene_filter=scene_key)

    log_event(
        "TASK_CONTEXT",
        "START",
        phase="align_cubes",
        scene=scene_key,
        model_file=str(scene_file),
        move_order=move_order,
        targets={object_id: slots_by_object[object_id]["target_pose"] for object_id in move_order},
        skipped_precheck=skipped_precheck,
        movable_count=world_state["counts"]["movable"],
        ceramic_count=world_state["counts"]["ceramic"],
    )

    started = time.perf_counter()
    moved = []
    failed = list(skipped_precheck)
    for item in skipped_precheck:
        log_event(
            "OBJECT_PRECHECK",
            "SKIP",
            object_id=item["object_id"],
            phase="align_cubes",
            failure_reason=item["failure_reason"],
            obstacle_distance=item.get("obstacle_distance"),
            reach_distance=item.get("reach_distance"),
        )
    placed_targets = {}  # {object_id: (target_x, target_y)} for sweep monitoring

    max_pick_attempts = 3
    pick_attempts = {object_id: 0 for object_id in move_order}
    pending = list(move_order)
    object_index = {object_id: index for index, object_id in enumerate(move_order, start=1)}

    retry_round = 0
    while pending:
        retry_round += 1
        current_round = pending
        pending = []
        print(f"\n[ROUND {retry_round}] candidates={current_round}")
        log_event("TASK_ROUND", "START", phase="align_cubes", round=retry_round, candidates=current_round)

        for object_id in current_round:
            if object_id in moved:
                continue

            index = object_index[object_id]
            target_pose = slots_by_object[object_id]["target_pose"]
            x, y = float(target_pose[0]), float(target_pose[1])
            obj = next(o for o in world_state["movable_objects"] if o["id"] == object_id)
            obstacle_distance = _min_ceramic_distance_xy(obj, world_state)
            obstacle_status = _obstacle_status(obj, world_state)
            reach_status = _reach_status(obj, world_state)
            if obstacle_status == "TOO_CLOSE" or reach_status == "HARD":
                reason = (
                    "object_too_close_to_obstacle"
                    if obstacle_status == "TOO_CLOSE"
                    else "object_outside_conservative_reach"
                )
                print(
                    f"\n[{index:02d}] SKIP_OBJECT object={object_id} "
                    f"reason={reason} obstacle_distance={obstacle_distance:.3f}m "
                    f"reach={_reach_distance_xy(obj, world_state):.3f}m; try_next_candidate"
                )
                log_event(
                    "OBSTACLE_PROXIMITY",
                    "FAILED",
                    object_id=object_id,
                    phase="precheck",
                    failure_reason=reason,
                    obstacle_distance=round(obstacle_distance, 4),
                    threshold=OBSTACLE_NEAR_M,
                )
                failed.append({
                    "object_id": object_id,
                    "stage": "precheck",
                    "failure_reason": reason,
                    "obstacle_distance": round(obstacle_distance, 4),
                })
                continue
            if obstacle_status == "NEAR":
                print(
                    f"\n[{index:02d}] CAUTIOUS_OBJECT object={object_id} "
                    f"obstacle_distance={obstacle_distance:.3f}m; execute_with_high_clearance"
                )
                log_event(
                    "OBSTACLE_PROXIMITY",
                    "NEAR_CAUTIOUS",
                    object_id=object_id,
                    phase="precheck",
                    obstacle_distance=round(obstacle_distance, 4),
                    threshold=OBSTACLE_NEAR_M,
                )
            pick_attempts[object_id] += 1
            print(f"\n[{index:02d}] ALIGN_CUBE START object={object_id} target=({x:.3f}, {y:.3f})")
            print(f"[{index:02d}] PICK_ATTEMPT {pick_attempts[object_id]}/{max_pick_attempts}")

            try:
                executor.pick(object_id)
            except RuntimeError as exc:
                reason = "obstacle_displaced" if "obstacle displacement" in str(exc) else "executor_runtime_error"
                print(f"[{index:02d}] PICK_EXCEPTION object={object_id} reason={reason}: {exc}")
                log_event(
                    "TASK_EXCEPTION",
                    "FAILED",
                    object_id=object_id,
                    phase="pick",
                    failure_reason=reason,
                    error=str(exc),
                )
                failed.append({
                    "object_id": object_id,
                    "stage": "pick",
                    "failure_reason": reason,
                    "error": str(exc),
                    "attempts": pick_attempts[object_id],
                })
                continue
            pick_ok, pick_z = feedback.check_pick(executor.model, executor.data, executor.name_to_cube[object_id])
            print(f"[{index:02d}] CHECK_PICK   {'OK' if pick_ok else 'FAIL'} attempt={pick_attempts[object_id]} z={pick_z}")
            log_event(
                "CHECK_PICK",
                "OK" if pick_ok else "FAILED",
                object_id=object_id,
                phase="pick",
                attempt=pick_attempts[object_id],
                object_z=pick_z,
                failure_reason=None if pick_ok else "object_not_lifted_after_pick",
            )
            if not pick_ok:
                terminal_reason = _terminal_pick_failure_reason(executor, object_id, world_state)
                executor.drop(object_id)
                _settle(executor, args.settle_after_place)
                terminal_reason = terminal_reason or _terminal_pick_failure_reason(executor, object_id, world_state)
                if terminal_reason is None and pick_attempts[object_id] < max_pick_attempts:
                    print(
                        f"[{index:02d}] DEFER_OBJECT object={object_id} reason=pick_failed "
                        f"attempts={pick_attempts[object_id]}/{max_pick_attempts}; try_next_candidate"
                    )
                    log_event(
                        "DEFER_OBJECT",
                        "RETRY_LATER",
                        object_id=object_id,
                        phase="pick",
                        failure_reason="pick_failed_try_other_candidate",
                        attempt=pick_attempts[object_id],
                        max_attempts=max_pick_attempts,
                    )
                    pending.append(object_id)
                else:
                    actual_pos = _runtime_object_pose(executor, object_id)
                    failure_reason = terminal_reason or "object_not_lifted_after_pick"
                    if terminal_reason is None and pick_attempts[object_id] >= max_pick_attempts:
                        failure_reason = _blocked_grasp_failure_reason(executor, object_id) or failure_reason
                    failed.append({
                        "object_id": object_id,
                        "stage": "pick",
                        "failure_reason": failure_reason,
                        "z": pick_z,
                        "actual": [round(v, 4) for v in actual_pos],
                        "attempts": pick_attempts[object_id],
                    })
                continue

            # Sweep already-placed objects for arm-induced disturbances.
            disturbed = feedback.check_placed_objects(
                executor.model, executor.data, executor.name_to_cube, placed_targets
            )
            for d in disturbed:
                did = d["object_id"]
                print(f"[{index:02d}] PLACED_OBJECT_DISTURBED object={did} reason={d['reason']} actual={d['actual_pos']}")
                log_event(
                    "PLACED_OBJECT_DISTURBED",
                    "DETECTED",
                    object_id=did,
                    phase="pick",
                    actual_xyz=d["actual_pos"],
                    failure_reason=d["reason"],
                    triggered_by=object_id,
                )
                moved.remove(did)
                placed_targets.pop(did, None)
                if pick_attempts.get(did, 0) < max_pick_attempts:
                    pending.insert(0, did)
                else:
                    failed.append({
                        "object_id": did,
                        "stage": "place",
                        "failure_reason": d["reason"],
                        "actual": d["actual_pos"],
                    })

            place_ok = False
            actual = None
            for retry in range(args.place_retries + 1):
                if retry > 0:
                    print(f"[{index:02d}] PLACE_RETRY  retry={retry}")
                try:
                    executor.place(x, y, object_id)
                except RuntimeError as exc:
                    reason = "obstacle_displaced" if "obstacle displacement" in str(exc) else "executor_runtime_error"
                    print(f"[{index:02d}] PLACE_EXCEPTION object={object_id} reason={reason}: {exc}")
                    log_event(
                        "TASK_EXCEPTION",
                        "FAILED",
                        object_id=object_id,
                        phase="place",
                        failure_reason=reason,
                        error=str(exc),
                    )
                    failed.append({
                        "object_id": object_id,
                        "stage": "place",
                        "failure_reason": reason,
                        "error": str(exc),
                    })
                    place_ok = False
                    break
                _settle(executor, args.settle_after_place)
                place_ok, actual = feedback.check_place(executor.model, executor.data, executor.name_to_cube[object_id], x, y)
                align_ok, align_details = _check_aligned_target(
                    executor,
                    object_id,
                    target_pose,
                    z_threshold=feedback.PLACED_Z_THRESHOLD,
                )
                print(f"[{index:02d}] CHECK_PLACE  {'OK' if place_ok else 'FAIL'} retry={retry} actual={actual}")
                print(
                    f"[{index:02d}] CHECK_ALIGN_PLACE {'OK' if align_ok else 'FAIL'} "
                    f"x_error={align_details['x_error']:.4f} y_error={align_details['y_error']:.4f}"
                )
                place_error = math.dist((actual[0], actual[1]), (x, y)) if actual else None
                log_event(
                    "CHECK_PLACE",
                    "OK" if place_ok else "FAILED",
                    object_id=object_id,
                    phase="place",
                    attempt=retry,
                    target_xyz=[round(x, 4), round(y, 4), round(float(target_pose[2]), 4)],
                    actual_xyz=actual,
                    distance_to_target=round(place_error, 4) if place_error is not None else None,
                    failure_reason=None if place_ok else "object_not_on_target_after_place",
                )
                log_event(
                    "CHECK_ALIGN_PLACE",
                    "OK" if align_ok else "FAILED",
                    object_id=object_id,
                    phase="place",
                    attempt=retry,
                    target_xyz=align_details["target_xyz"],
                    actual_xyz=align_details["actual_xyz"],
                    x_error=align_details["x_error"],
                    y_error=align_details["y_error"],
                    x_tolerance=ALIGN_PLACE_X_TOLERANCE_M,
                    y_tolerance=ALIGN_PLACE_Y_TOLERANCE_M,
                    failure_reason=None if align_ok else align_details["failure_reason"],
                )
                place_ok = place_ok and align_ok
                actual = align_details["actual_xyz"]
                if place_ok:
                    break
                executor.drop(object_id)
                _settle(executor, args.settle_after_place)
                break

            if place_ok:
                moved.append(object_id)
                placed_targets[object_id] = (x, y)
                _settle(executor, args.settle_after_place)
            else:
                already_failed = any(item.get("object_id") == object_id and item.get("stage") == "place" for item in failed)
                terminal_reason = _terminal_pick_failure_reason(executor, object_id, world_state)
                if not already_failed and terminal_reason is None and pick_attempts[object_id] < max_pick_attempts:
                    print(
                        f"[{index:02d}] DEFER_OBJECT object={object_id} reason=place_failed_repick "
                        f"attempts={pick_attempts[object_id]}/{max_pick_attempts}; try_next_candidate"
                    )
                    log_event(
                        "DEFER_OBJECT",
                        "RETRY_LATER",
                        object_id=object_id,
                        phase="place",
                        failure_reason="place_failed_repick_required",
                        attempt=pick_attempts[object_id],
                        max_attempts=max_pick_attempts,
                    )
                    pending.append(object_id)
                elif not already_failed:
                    failed.append({
                        "object_id": object_id,
                        "stage": "place",
                        "failure_reason": terminal_reason or "object_not_on_target_after_place",
                        "actual": actual,
                    })

        invalid_alignment = _validate_aligned_targets(
            executor,
            moved,
            slots_by_object,
            z_threshold=feedback.PLACED_Z_THRESHOLD,
        )
        for item in invalid_alignment:
            object_id = item["object_id"]
            if object_id not in moved:
                continue
            print(
                f"[ROUND {retry_round}] ALIGNMENT_RECHECK FAIL object={object_id} "
                f"reason={item['failure_reason']} actual={item['actual_xyz']}"
            )
            log_event(
                "ALIGNMENT_RECHECK",
                "FAILED",
                object_id=object_id,
                phase="post_round",
                actual_xyz=item["actual_xyz"],
                target_xyz=item["target_xyz"],
                x_error=item.get("x_error"),
                y_error=item.get("y_error"),
                row_y_spread=item.get("row_y_spread"),
                failure_reason=item["failure_reason"],
            )
            moved.remove(object_id)
            placed_targets.pop(object_id, None)
            already_failed = any(entry.get("object_id") == object_id for entry in failed)
            if pick_attempts.get(object_id, 0) < max_pick_attempts:
                if object_id not in pending:
                    pending.append(object_id)
                log_event(
                    "DEFER_OBJECT",
                    "RETRY_LATER",
                    object_id=object_id,
                    phase="alignment",
                    failure_reason=item["failure_reason"],
                    attempt=pick_attempts.get(object_id, 0),
                    max_attempts=max_pick_attempts,
                )
            elif not already_failed:
                failed.append({
                    "object_id": object_id,
                    "stage": "alignment",
                    "failure_reason": item["failure_reason"],
                    "actual": item["actual_xyz"],
                    "target": item["target_xyz"],
                    "attempts": pick_attempts.get(object_id, 0),
                })

    duration_ms = int((time.perf_counter() - started) * 1000)
    failed_ids = {item.get("object_id") for item in failed}
    for object_id in move_order:
        if object_id not in moved and object_id not in failed_ids:
            failed.append({
                "object_id": object_id,
                "stage": "alignment",
                "failure_reason": "object_missing_from_final_aligned_row",
                "attempts": pick_attempts.get(object_id, 0),
            })
    success = len(failed) == 0 and len(moved) == len(move_order)
    print("\n=== Align cubes OMPL-only summary ===")
    print(f"success={success}")
    print(f"objects_moved={len(moved)}")
    print(f"failed={failed}")
    print(f"duration_ms={duration_ms}")
    print("collision_policy=MuJoCo contacts via PandaOMPLPlanner/CollisionPolicy")
    print("llm_used=false")
    csv_path = write_summary_csv(
        "align_cubes",
        scene_key,
        {
            "success": success,
            "objects_moved": len(moved),
            "objects_total": len(move_order) + len(skipped_precheck),
            "failed": failed,
            "duration_ms": duration_ms,
            "scenario_type": "static",
            "obstacle_mode": obstacle_mode_for_scene(scene_key),
        },
        args.log_dir,
    )
    print(f"csv_log={csv_path}")
    print(f"event_csv_log={event_csv_path}")
    flush_trace()
    executor.shutdown_runtime()
    return 0 if not failed else 1


def _settle(executor, steps: int) -> None:
    if steps <= 0:
        return
    for _ in range(steps):
        executor.mujoco.mj_step(executor.model, executor.data)
        executor.viewer.sync()


if __name__ == "__main__":
    safe_process_exit(main())
