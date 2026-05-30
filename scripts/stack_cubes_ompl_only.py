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
    CUBE_HALF_EXTENTS,
    apply_conservative_motion_defaults,
    apply_scene_motion_defaults,
    make_event_log_path,
    normalize_scene_key,
    obstacle_mode_for_scene,
    prepare_scene_variant,
    safe_process_exit,
    write_summary_csv,
)

CUBE_RADIUS_M = math.hypot(CUBE_HALF_EXTENTS[0], CUBE_HALF_EXTENTS[1])
CUBE_HALF_SIZE_M = CUBE_HALF_EXTENTS[2]
STACK_XY_TOLERANCE_M = 0.045
STACK_Z_TOLERANCE_M = 0.035
STACK_UPPER_PLACE_X_BIAS_M = 0.0


def _configure_stack_ready_pose(executor) -> None:
    """Keep the ready pose above the central tower, not inside it."""
    if not hasattr(executor, "HOME") or not hasattr(executor, "GRASP_READY"):
        return
    executor.GRASP_READY = executor.HOME.copy()
    executor.log_event(
        "STACK_READY_POSE",
        "SET",
        phase="stack_cubes",
        target="HOME",
        q=[round(float(v), 4) for v in executor.GRASP_READY],
    )


def _build_cube_stack_targets(scene_file: str):
    """Build one four-layer cube tower: cube1 <- cube2 <- cube3 <- cube4."""
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

    required_order = ["cube1", "cube2", "cube3", "cube4"]
    eligible_by_id = {cube["id"]: cube for cube in eligible}
    missing_required = [object_id for object_id in required_order if object_id not in eligible_by_id]
    if missing_required:
        skipped.append({
            "object_id": "cube_stack",
            "stage": "target_allocation",
            "failure_reason": "required_cube_not_reachable_for_four_layer_stack",
            "eligible_count": len(eligible),
            "missing_required": missing_required,
        })
        return world_state, [], {}, skipped

    table = world_state["table"]
    goal_x, goal_y, _ = world_state["goal_center"]
    desired_base_x = goal_x
    desired_base_y = goal_y
    base_xy = _search_safe_target_xy(desired_base_x, desired_base_y, CUBE_RADIUS_M, world_state, [])
    if base_xy is None:
        skipped.append({
            "object_id": "cube_stack",
            "stage": "target_allocation",
            "failure_reason": "no_safe_four_layer_stack_target_found",
            "desired_xy": [round(desired_base_x, 4), round(desired_base_y, 4)],
        })
        return world_state, [], {}, skipped

    base_z = table["z_top"] + CUBE_HALF_SIZE_M
    slot_defs = [
        {
            "tower_id": "tower_1",
            "level": level,
            "xy": base_xy,
            "target_z": base_z + level * 2.0 * CUBE_HALF_SIZE_M,
            "support_object_id": required_order[level - 1] if level > 0 else None,
        }
        for level in range(len(required_order))
    ]

    slots = {}
    target_cubes = [eligible_by_id[object_id] for object_id in required_order]
    for cube, slot in zip(target_cubes, slot_defs):
        x, y = slot["xy"]
        slots[cube["id"]] = {
            "slot_id": f"{slot['tower_id']}_level_{slot['level']}",
            "tower_id": slot["tower_id"],
            "level": slot["level"],
            "support_object_id": slot["support_object_id"],
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

    return world_state, required_order, slots, skipped


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


def _expected_stack_pose(executor, slot: dict, table_top: float):
    support = slot.get("support_object_id")
    if support:
        support_pos = _runtime_object_pose(executor, support)
        return (
            float(support_pos[0]),
            float(support_pos[1]),
            float(support_pos[2] + 2.0 * CUBE_HALF_SIZE_M),
        )
    x, y = [float(v) for v in slot["target_pose"][:2]]
    return x, y, float(table_top + CUBE_HALF_SIZE_M)


def _check_stack_slot(executor, object_id: str, slots_by_object: dict, table_top: float):
    x, y, z = _expected_stack_pose(executor, slots_by_object[object_id], table_top)
    return _check_stack_place(executor, object_id, x, y, z)


def _validate_moved_stack(
    executor,
    moved: list[str],
    slots_by_object: dict,
    table_top: float,
    failure_reason: str = "placed_stack_disturbed",
):
    invalid = []
    for object_id in moved:
        ok, actual, xy_error, z_error = _check_stack_slot(executor, object_id, slots_by_object, table_top)
        if not ok:
            invalid.append({
                "object_id": object_id,
                "stage": "stack_integrity",
                "failure_reason": failure_reason,
                "actual": actual,
                "xy_error": round(xy_error, 4),
                "z_error": round(z_error, 4),
                "support_object_id": slots_by_object[object_id].get("support_object_id"),
            })
    return invalid


def _log_stack_integrity_failures(log_event_fn, invalid_items: list[dict], phase: str) -> None:
    for item in invalid_items:
        log_event_fn(
            "STACK_INTEGRITY",
            "FAILED",
            object_id=item["object_id"],
            phase=phase,
            actual_xyz=item["actual"],
            distance_to_target=item["xy_error"],
            z_error=item["z_error"],
            failure_reason=item["failure_reason"],
            support_object_id=item["support_object_id"],
        )


def _rebuild_order_after_disturbance(move_order: list[str], invalid_items: list[dict]) -> list[str]:
    invalid_ids = {item["object_id"] for item in invalid_items}
    first_invalid_index = min(index for index, object_id in enumerate(move_order) if object_id in invalid_ids)
    return move_order[first_invalid_index:]


def _check_four_layer_tower_final(executor, moved: list[str], slots_by_object: dict, table_top: float):
    failed = []
    for object_id in moved:
        slot = slots_by_object[object_id]
        support = slot.get("support_object_id")
        if support:
            support_pos = _runtime_object_pose(executor, support)
            expected_x, expected_y = support_pos[:2]
            expected_z = float(support_pos[2] + 2.0 * CUBE_HALF_SIZE_M)
        else:
            expected_x, expected_y = [float(v) for v in slot["target_pose"][:2]]
            expected_z = float(table_top + CUBE_HALF_SIZE_M)

        ok, actual, xy_error, z_error = _check_stack_place(
            executor,
            object_id,
            float(expected_x),
            float(expected_y),
            float(expected_z),
        )
        if not ok:
            failed.append({
                "object_id": object_id,
                "stage": "final_validation",
                "failure_reason": "object_not_in_four_layer_stack",
                "actual": actual,
                "target": [round(float(expected_x), 4), round(float(expected_y), 4), round(float(expected_z), 4)],
                "xy_error": round(xy_error, 4),
                "z_error": round(z_error, 4),
                "support_object_id": support,
            })
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stack four cubes into one vertical four-layer tower using OMPL + MuJoCo collision checking only."
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

    print("[STACK_CUBES] Task          : stack kubus 4 layer dalam 1 tower")
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

    _configure_stack_ready_pose(executor)

    if args.no_hint_cache:
        print("[STACK_CUBES] HintCache disabled (--no-hint-cache).")
    else:
        executor.init_hint_cache(log_dir=args.log_dir, scene_filter=scene_key)

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
    max_stack_rebuild_attempts = 2
    pick_attempts = {object_id: 0 for object_id in move_order}
    stack_rebuild_attempts = 0
    pending = list(move_order)
    object_index = {object_id: index for index, object_id in enumerate(move_order, start=1)}
    table_top = float(world_state["table"]["z_top"])

    def handle_stack_disturbance(invalid_items: list[dict], phase: str) -> bool:
        nonlocal moved, pending, stack_rebuild_attempts
        if not invalid_items:
            return False
        _log_stack_integrity_failures(log_event, invalid_items, phase)
        if stack_rebuild_attempts < max_stack_rebuild_attempts:
            stack_rebuild_attempts += 1
            rebuild_order = _rebuild_order_after_disturbance(move_order, invalid_items)
            rebuild_set = set(rebuild_order)
            moved = [moved_id for moved_id in moved if moved_id not in rebuild_set]
            pending = rebuild_order
            log_event(
                "STACK_REBUILD",
                "RETRY_LATER",
                phase=phase,
                failure_reason="stack_disturbed_rebuild_from_current_pose",
                attempt=stack_rebuild_attempts,
                max_attempts=max_stack_rebuild_attempts,
                rebuild_order=rebuild_order,
                stable_prefix=moved,
            )
        else:
            for item in invalid_items:
                if not any(existing.get("object_id") == item["object_id"] for existing in failed):
                    failed.append(item)
            pending = []
        return True

    round_id = 0
    while pending:
        round_id += 1
        current_round = pending
        pending = []
        dependency_blocked = set()
        print(f"\n[ROUND {round_id}] candidates={current_round}")
        for object_id in current_round:
            if object_id in moved:
                continue
            slot = slots_by_object[object_id]
            support = slot["support_object_id"]
            if support and support not in moved:
                pending.append(object_id)
                dependency_blocked.add(object_id)
                continue
            if moved:
                prefix_invalid = _validate_moved_stack(
                    executor,
                    moved,
                    slots_by_object,
                    table_top,
                    failure_reason="placed_stack_disturbed_before_next_pick",
                )
                if handle_stack_disturbance(prefix_invalid, phase=f"pre_pick_{object_id}"):
                    break
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
            pick_lifted, pick_z = feedback.check_pick(executor.model, executor.data, executor.name_to_cube[object_id])
            held_ok = getattr(executor, "_held_object_name", None) == object_id
            pick_ok = pick_lifted and held_ok
            pick_failure_reason = None
            if not pick_ok:
                pick_failure_reason = "object_not_held_after_pick" if pick_lifted and not held_ok else "object_not_lifted_after_pick"
            print(f"[{index:02d}] CHECK_PICK   {'OK' if pick_ok else 'FAIL'} attempt={pick_attempts[object_id]} z={pick_z}")
            log_event(
                "CHECK_PICK",
                "OK" if pick_ok else "FAILED",
                object_id=object_id,
                phase="pick",
                attempt=pick_attempts[object_id],
                object_z=pick_z,
                failure_reason=pick_failure_reason,
            )
            if not pick_ok:
                terminal_reason = _terminal_pick_failure_reason(executor, object_id, world_state)
                executor.drop(object_id)
                _settle(executor, args.settle_after_place)
                terminal_reason = terminal_reason or _terminal_pick_failure_reason(executor, object_id, world_state)
                unrecoverable_pick_failure = terminal_reason in {
                    "object_displaced_below_table",
                    "object_outside_robot_reach_after_displacement",
                }
                if not unrecoverable_pick_failure and pick_attempts[object_id] < max_pick_attempts:
                    log_event(
                        "DEFER_OBJECT",
                        "RETRY_LATER",
                        object_id=object_id,
                        phase="pick",
                        failure_reason="pick_failed_retry_from_current_pose",
                        attempt=pick_attempts[object_id],
                        max_attempts=max_pick_attempts,
                    )
                    pending.append(object_id)
                else:
                    actual_pos = _runtime_object_pose(executor, object_id)
                    failure_reason = terminal_reason or pick_failure_reason or "object_not_lifted_after_pick"
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
            place_x = x + (STACK_UPPER_PLACE_X_BIAS_M if slot["level"] > 0 else 0.0)
            executor.place(
                place_x,
                y,
                object_id,
                target_z=z,
                release_lift=release_lift,
                post_place_ignored_body_names=[],
            )
            _settle(executor, args.settle_after_place)
            x, y, z = [float(v) for v in slot["target_pose"][:3]]
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
            post_place_moved = moved + [object_id] if place_ok else list(moved)
            post_place_invalid = _validate_moved_stack(
                executor,
                post_place_moved,
                slots_by_object,
                table_top,
                failure_reason="placed_stack_disturbed_after_place",
            )
            if handle_stack_disturbance(post_place_invalid, phase=f"post_place_{object_id}"):
                break
            if place_ok:
                moved.append(object_id)
                _refresh_children_after_base_place(executor, object_id, slots_by_object)
            elif pick_attempts[object_id] < max_pick_attempts:
                log_event(
                    "DEFER_OBJECT",
                    "RETRY_LATER",
                    object_id=object_id,
                    phase="place",
                    failure_reason="place_failed_retry_from_current_pose",
                    attempt=pick_attempts[object_id],
                    max_attempts=max_pick_attempts,
                    actual_xyz=actual,
                    target_xyz=[round(x, 4), round(y, 4), round(z, 4)],
                )
                pending.append(object_id)
            else:
                failed.append({
                    "object_id": object_id,
                    "stage": "place",
                    "failure_reason": "object_not_on_stack_target",
                    "actual": actual,
                    "target": [round(x, 4), round(y, 4), round(z, 4)],
                })

        if current_round and pending == current_round and all(object_id in dependency_blocked for object_id in pending):
            for object_id in pending:
                if not any(item.get("object_id") == object_id for item in failed):
                    failed.append({
                        "object_id": object_id,
                        "stage": "dependency",
                        "failure_reason": "support_cube_not_available",
                    })
            pending = []

    final_failed = _check_four_layer_tower_final(
        executor,
        moved,
        slots_by_object,
        float(world_state["table"]["z_top"]),
    )
    failed.extend(final_failed)
    final_failed_ids = {item["object_id"] for item in final_failed}
    objects_moved_final = sum(1 for object_id in moved if object_id not in final_failed_ids)

    duration_ms = int((time.perf_counter() - started) * 1000)
    success = len(failed) == 0
    print("\n=== Stack cubes OMPL-only summary ===")
    print(f"success={success}")
    print(f"objects_moved={objects_moved_final}")
    print(f"failed={failed}")
    print(f"duration_ms={duration_ms}")
    print("collision_policy=MuJoCo contacts via PandaOMPLPlanner/CollisionPolicy")
    print("llm_used=false")
    csv_path = write_summary_csv(
        "stack_cubes",
        scene_key,
        {
            "success": success,
            "objects_moved": objects_moved_final,
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


if __name__ == "__main__":
    safe_process_exit(main())
