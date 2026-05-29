#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from align_cubes_ompl_only import (
    OBSTACLE_NEAR_M,
    _blocked_grasp_failure_reason,
    _min_ceramic_distance_xy,
    _obstacle_status,
    _parse_scene,
    _reach_distance_xy,
    _reach_status,
    _runtime_object_pose,
    _search_safe_target_xy,
    _settle,
    _terminal_pick_failure_reason,
)
from ctamp_task_utils import (
    apply_conservative_motion_defaults,
    apply_scene_motion_defaults,
    make_event_log_path,
    normalize_scene_key,
    obstacle_mode_for_scene,
    prepare_scene_variant,
    write_summary_csv,
)

CUBE_RADIUS_M = 0.043
CUBE_HALF_SIZE_M = 0.03
STACK_XY_TOLERANCE_M = 0.10
STACK_Z_TOLERANCE_M = 0.045
STACK_TOP_PLACE_X_BIAS_M = 0.010


def _build_cube_stack_targets(scene_file: str):
    """Build a stable stack: three base cubes and one top cube."""
    world_state = _parse_scene(scene_file)
    cubes = [obj for obj in world_state["movable_objects"] if obj.get("class") == "cube"]
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
    eligible = []
    for cube in cubes:
        obstacle_status = _obstacle_status(cube, world_state)
        reach_status = _reach_status(cube, world_state)
        if obstacle_status == "TOO_CLOSE":
            skipped.append({
                "object_id": cube["id"],
                "stage": "precheck",
                "failure_reason": "object_too_close_to_obstacle",
                "obstacle_distance": round(_min_ceramic_distance_xy(cube, world_state), 4),
            })
            continue
        if reach_status == "HARD":
            skipped.append({
                "object_id": cube["id"],
                "stage": "precheck",
                "failure_reason": "object_outside_conservative_reach",
                "reach_distance": round(_reach_distance_xy(cube, world_state), 4),
            })
            continue
        eligible.append(cube)

    if len(eligible) < 4:
        skipped.append({
            "object_id": "cube_stack",
            "stage": "target_allocation",
            "failure_reason": "not_enough_reachable_cubes",
            "eligible_count": len(eligible),
        })
        return world_state, [], {}, skipped

    table = world_state["table"]
    goal_x, goal_y, _ = world_state["goal_center"]
    desired_centers = [
        (goal_x - 0.105, goal_y - 0.025),
        (goal_x, goal_y - 0.025),
        (goal_x + 0.105, goal_y - 0.025),
    ]
    occupied = []
    tower_centers = []
    for base_x, base_y in desired_centers:
        target_xy = _search_safe_target_xy(base_x, base_y, CUBE_RADIUS_M, world_state, occupied)
        if target_xy is None:
            skipped.append({
                "object_id": "cube_stack",
                "stage": "target_allocation",
                "failure_reason": "no_safe_stack_target_found",
                "desired_xy": [round(base_x, 4), round(base_y, 4)],
            })
            continue
        x, y = target_xy
        occupied.append((x, y, CUBE_RADIUS_M))
        tower_centers.append((x, y))

    if len(tower_centers) < 3:
        return world_state, [], {}, skipped

    base_z = table["z_top"] + CUBE_HALF_SIZE_M
    top_z = base_z + 2.0 * CUBE_HALF_SIZE_M
    slot_defs = [
        {"tower_id": "tower_1", "level": 0, "xy": tower_centers[0], "target_z": base_z},
        {"tower_id": "tower_2", "level": 0, "xy": tower_centers[1], "target_z": base_z},
        {"tower_id": "tower_3", "level": 0, "xy": tower_centers[2], "target_z": base_z},
        {"tower_id": "tower_2", "level": 1, "xy": tower_centers[1], "target_z": top_z},
    ]

    slots = {}
    base_by_tower = {}
    target_cubes = sorted(eligible[:4], key=lambda obj: (obj["position"][0], obj["position"][1], obj["id"]))
    for cube, slot in zip(target_cubes, slot_defs):
        x, y = slot["xy"]
        support_object_id = base_by_tower.get(slot["tower_id"])
        slots[cube["id"]] = {
            "slot_id": f"{slot['tower_id']}_level_{slot['level']}",
            "tower_id": slot["tower_id"],
            "level": slot["level"],
            "support_object_id": support_object_id,
            "target_pose": [
                round(x, 4),
                round(y, 4),
                round(slot["target_z"], 4),
                1.0,
                0.0,
                0.0,
                0.0,
            ],
        }
        if slot["level"] == 0:
            base_by_tower[slot["tower_id"]] = cube["id"]

    return world_state, list(slots.keys()), slots, skipped


def _check_stack_place(executor, object_id: str, x: float, y: float, z: float):
    executor.mujoco.mj_forward(executor.model, executor.data)
    pos = executor.data.xpos[executor.name_to_cube[object_id]]
    actual = [round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3)]
    xy_error = float(math.dist((float(pos[0]), float(pos[1])), (x, y)))
    z_error = abs(float(pos[2]) - z)
    return xy_error <= STACK_XY_TOLERANCE_M and z_error <= STACK_Z_TOLERANCE_M, actual, xy_error, z_error


def _refresh_supported_target_from_base(executor, slot: dict) -> None:
    support = slot.get("support_object_id")
    if not support:
        return
    support_pos = _runtime_object_pose(executor, support)
    slot["target_pose"][0] = round(float(support_pos[0]), 4)
    slot["target_pose"][1] = round(float(support_pos[1]), 4)
    slot["target_pose"][2] = round(float(support_pos[2] + 2.0 * CUBE_HALF_SIZE_M), 4)


def _refresh_children_after_base_place(executor, base_object_id: str, slots_by_object: dict) -> None:
    base_pos = _runtime_object_pose(executor, base_object_id)
    for slot in slots_by_object.values():
        if slot.get("support_object_id") != base_object_id:
            continue
        slot["target_pose"][0] = round(float(base_pos[0]), 4)
        slot["target_pose"][1] = round(float(base_pos[1]), 4)
        slot["target_pose"][2] = round(float(base_pos[2] + 2.0 * CUBE_HALF_SIZE_M), 4)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stack four cubes as three base cubes plus one top cube using OMPL + MuJoCo collision checking only."
    )
    parser.add_argument("--object", nargs="+", default=["group", "no", "obs"], help="Scene: group no obs, ungroup no obs, group obs, ungroup obs.")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--settle-after-place", type=int, default=360, help=argparse.SUPPRESS)
    args = parser.parse_args()

    scene_key = normalize_scene_key(args.object)
    scene_file = prepare_scene_variant(scene_key)
    event_csv_path = make_event_log_path("stack_cubes", scene_key, args.log_dir)
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
        print("[STACK_CUBES] OMPL Python binding belum tersedia.")
        print("  pip install ompl")
        print("  python -c \"from ompl import base, geometric; print('ompl ok')\"")
        return 2

    world_state, move_order, slots_by_object, skipped_precheck = _build_cube_stack_targets(str(scene_file))
    if not move_order:
        print("[STACK_CUBES] Tidak ada target stack aman untuk cube.")
        return 1

    print("[STACK_CUBES] Task          : stack kubus 3 base + 1 top")
    print(f"[STACK_CUBES] Scene variant : {scene_key} ({scene_file})")
    print(f"[STACK_CUBES] Event CSV     : {event_csv_path}")
    print(f"[STACK_CUBES] Ceramic avoid  : {[o['id'] for o in world_state['ceramic_obstacles']]}")
    print(f"[STACK_CUBES] Move order     : {move_order}")
    print("[STACK_CUBES] Stack targets:")
    for object_id in move_order:
        slot = slots_by_object[object_id]
        x, y, z = slot["target_pose"][:3]
        obj = next(o for o in world_state["movable_objects"] if o["id"] == object_id)
        print(
            f"  - {object_id:>6} -> {slot['slot_id']} "
            f"({x:.3f}, {y:.3f}, {z:.3f}) support={slot['support_object_id']} "
            f"reach={_reach_distance_xy(obj, world_state):.3f}m "
            f"obs_status={_obstacle_status(obj, world_state)}"
        )

    import executor
    from exec_trace import flush as flush_trace
    from exec_trace import log_event
    import feedback

    if not getattr(executor, "_OMPL_AVAILABLE", False):
        print("[STACK_CUBES] Executor tidak melihat OMPL planner aktif.")
        return 2

    log_event(
        "TASK_CONTEXT",
        "START",
        phase="stack_cubes",
        scene=scene_key,
        model_file=str(scene_file),
        move_order=move_order,
        targets={object_id: slots_by_object[object_id]["target_pose"] for object_id in move_order},
        skipped_precheck=skipped_precheck,
    )

    started = time.perf_counter()
    moved = []
    failed = list(skipped_precheck)
    max_pick_attempts = 3
    pick_attempts = {object_id: 0 for object_id in move_order}
    pending = list(move_order)
    object_index = {object_id: index for index, object_id in enumerate(move_order, start=1)}
    base_object_ids = [
        object_id
        for object_id in move_order
        if slots_by_object[object_id]["level"] == 0
    ]

    round_id = 0
    while pending:
        round_id += 1
        current_round = pending
        pending = []
        print(f"\n[ROUND {round_id}] candidates={current_round}")
        for object_id in current_round:
            if object_id in moved:
                continue
            slot = slots_by_object[object_id]
            support = slot["support_object_id"]
            if slot["level"] > 0 and any(base_id not in moved for base_id in base_object_ids):
                pending.append(object_id)
                continue
            if support and support not in moved:
                pending.append(object_id)
                continue
            if support:
                _refresh_supported_target_from_base(executor, slot)

            index = object_index[object_id]
            x, y, z = [float(v) for v in slot["target_pose"][:3]]
            obj = next(o for o in world_state["movable_objects"] if o["id"] == object_id)
            if _obstacle_status(obj, world_state) == "TOO_CLOSE" or _reach_status(obj, world_state) == "HARD":
                reason = "object_too_close_to_obstacle" if _obstacle_status(obj, world_state) == "TOO_CLOSE" else "object_outside_conservative_reach"
                failed.append({"object_id": object_id, "stage": "precheck", "failure_reason": reason})
                continue
            if _obstacle_status(obj, world_state) == "NEAR":
                distance = _min_ceramic_distance_xy(obj, world_state)
                print(f"\n[{index:02d}] CAUTIOUS_OBJECT object={object_id} obstacle_distance={distance:.3f}m")
                log_event(
                    "OBSTACLE_PROXIMITY",
                    "NEAR_CAUTIOUS",
                    object_id=object_id,
                    phase="precheck",
                    obstacle_distance=round(distance, 4),
                    threshold=OBSTACLE_NEAR_M,
                )

            pick_attempts[object_id] += 1
            print(f"\n[{index:02d}] STACK_CUBE START object={object_id} slot={slot['slot_id']}")
            print(f"[{index:02d}] PICK_ATTEMPT {pick_attempts[object_id]}/{max_pick_attempts}")
            executor.pick(object_id)
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
                    })
                continue

            release_lift = 0.050 if slot["level"] == 0 else 0.008
            place_x = x + (STACK_TOP_PLACE_X_BIAS_M if slot["level"] > 0 else 0.0)
            executor.place(
                place_x,
                y,
                object_id,
                target_z=z,
                release_lift=release_lift,
                post_place_ignored_body_names=[],
            )
            _settle(executor, args.settle_after_place)
            place_ok, actual, xy_error, z_error = _check_stack_place(executor, object_id, x, y, z)
            print(
                f"[{index:02d}] CHECK_STACK {'OK' if place_ok else 'FAIL'} "
                f"actual={actual} xy_err={xy_error:.3f} z_err={z_error:.3f}"
            )
            log_event(
                "CHECK_STACK_PLACE",
                "OK" if place_ok else "FAILED",
                object_id=object_id,
                phase="place",
                target_xyz=[round(x, 4), round(y, 4), round(z, 4)],
                actual_xyz=actual,
                distance_to_target=round(xy_error, 4),
                z_error=round(z_error, 4),
                failure_reason=None if place_ok else "object_not_on_stack_target",
            )
            if place_ok:
                moved.append(object_id)
                if slot["level"] == 0:
                    _refresh_children_after_base_place(executor, object_id, slots_by_object)
            elif pick_attempts[object_id] < max_pick_attempts:
                pending.append(object_id)
            else:
                failed.append({
                    "object_id": object_id,
                    "stage": "place",
                    "failure_reason": "object_not_on_stack_target",
                    "actual": actual,
                    "target": [round(x, 4), round(y, 4), round(z, 4)],
                })

        if current_round and pending == current_round:
            for object_id in pending:
                if not any(item.get("object_id") == object_id for item in failed):
                    failed.append({
                        "object_id": object_id,
                        "stage": "dependency",
                        "failure_reason": "support_cube_not_available",
                    })
            pending = []

    final_failed = []
    for object_id in moved:
        x, y, z = [float(v) for v in slots_by_object[object_id]["target_pose"][:3]]
        ok, actual, xy_error, z_error = _check_stack_place(executor, object_id, x, y, z)
        if not ok:
            final_failed.append({
                "object_id": object_id,
                "stage": "final_validation",
                "failure_reason": "object_moved_after_stack",
                "actual": actual,
                "target": [round(x, 4), round(y, 4), round(z, 4)],
                "xy_error": round(xy_error, 4),
                "z_error": round(z_error, 4),
            })
    failed.extend(final_failed)

    duration_ms = int((time.perf_counter() - started) * 1000)
    success = len(failed) == 0
    print("\n=== Stack cubes OMPL-only summary ===")
    print(f"success={success}")
    print(f"objects_moved={len(moved) - len(final_failed)}")
    print(f"failed={failed}")
    print(f"duration_ms={duration_ms}")
    print("collision_policy=MuJoCo contacts via PandaOMPLPlanner/CollisionPolicy")
    print("llm_used=false")
    csv_path = write_summary_csv(
        "stack_cubes",
        scene_key,
        {
            "success": success,
            "objects_moved": len(moved) - len(final_failed),
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


if __name__ == "__main__":
    raise SystemExit(main())
