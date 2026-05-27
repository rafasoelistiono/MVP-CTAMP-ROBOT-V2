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
    write_summary_csv,
)

REACH_BORDERLINE_M = 0.78
REACH_HARD_M = 0.82
OBSTACLE_TOO_CLOSE_M = 0.12
OBSTACLE_NEAR_M = 0.18
MAX_PICK_RETRY_SOURCE_SHIFT_M = 0.18
TARGET_CERAMIC_BUFFER_M = 0.13


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


def _search_safe_target_xy(base_x: float, base_y: float, radius: float, world_state, occupied=()):
    table = world_state["table"]
    x_min, x_max = table["x_range"]
    y_min, y_max = table["y_range"]
    candidates = []
    for ring in range(0, 7):
        for dx in range(-ring, ring + 1):
            for dy in range(-ring, ring + 1):
                if ring and abs(dx) != ring and abs(dy) != ring:
                    continue
                x = min(max(base_x + dx * 0.035, x_min + radius), x_max - radius)
                y = min(max(base_y + dy * 0.035, y_min + radius), y_max - radius)
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
        if _target_xy_ok(x, y, radius, world_state, occupied):
            return x, y
    return None


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


def _build_aligned_cube_targets(scene_file: str, spacing: float = 0.11):
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
    for cube in cubes:
        obstacle_status = _obstacle_status(cube, world_state)
        reach_status = _reach_status(cube, world_state)
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
    row_y = goal_y - 0.065
    start_x = goal_x - spacing * (len(eligible_cubes) - 1) / 2.0
    slots = {}
    occupied = []
    for index, cube in enumerate(eligible_cubes):
        radius = cube.get("radius", 0.043)
        base_x = start_x + index * spacing
        target_xy = _search_safe_target_xy(base_x, row_y, radius, world_state, occupied)
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
    parser.add_argument("--object", nargs="+", default=["group", "no", "obs"], help="Scene: group no obs, ungroup no obs, group obs, ungroup obs.")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--no-viewer", action="store_true")
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
        print("[ALIGN_CUBES] Tidak ada cube untuk disusun.")
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
                    failed.append({
                        "object_id": object_id,
                        "stage": "pick",
                        "failure_reason": terminal_reason or "object_not_lifted_after_pick",
                        "z": pick_z,
                        "actual": [round(v, 4) for v in actual_pos],
                        "attempts": pick_attempts[object_id],
                    })
                continue

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
                print(f"[{index:02d}] CHECK_PLACE  {'OK' if place_ok else 'FAIL'} retry={retry} actual={actual}")
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
                if place_ok:
                    break
                executor.drop(object_id)
                _settle(executor, args.settle_after_place)
                break

            if place_ok:
                moved.append(object_id)
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

    duration_ms = int((time.perf_counter() - started) * 1000)
    success = len(failed) == 0
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
    return 0 if not failed else 1


def _settle(executor, steps: int) -> None:
    if steps <= 0:
        return
    for _ in range(steps):
        executor.mujoco.mj_step(executor.model, executor.data)
        executor.viewer.sync()


if __name__ == "__main__":
    raise SystemExit(main())
