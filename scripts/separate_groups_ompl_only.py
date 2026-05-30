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
    safe_process_exit,
    write_summary_csv,
)

# Cubes go to the red (square) row: goal_y - 0.065
# Circles go to the green (circle) row: goal_y + 0.065
# These match the visual goal_area markers in ctamp_task_utils._goal_area_body().
CUBE_ROW_OFFSET_Y = -0.065
CIRCLE_ROW_OFFSET_Y = +0.065
CUBE_SPACING = 0.11
CIRCLE_SPACING = 0.105


def _build_separate_groups_targets(scene_file: str):
    world_state = _parse_scene(scene_file)

    cubes = [o for o in world_state["movable_objects"] if o.get("class") == "cube"]
    circles = [
        o for o in world_state["movable_objects"]
        if o.get("class") in {"circle", "cylinder"} or o["id"].lower().startswith("circle")
    ]

    def sort_key(obj):
        return (
            _obstacle_status(obj, world_state) == "TOO_CLOSE",
            _obstacle_status(obj, world_state) == "NEAR",
            _reach_status(obj, world_state) == "HARD",
            _reach_status(obj, world_state) == "BORDERLINE",
            _reach_distance_xy(obj, world_state),
            obj["position"][0],
            obj["position"][1],
        )

    cubes.sort(key=sort_key)
    circles.sort(key=sort_key)

    skipped = []
    eligible_cubes = []
    eligible_circles = []

    for obj in cubes:
        status = _obstacle_status(obj, world_state)
        reach = _reach_status(obj, world_state)
        if status == "TOO_CLOSE":
            skipped.append({
                "object_id": obj["id"],
                "group": "cube",
                "stage": "precheck",
                "failure_reason": "object_too_close_to_obstacle",
                "obstacle_distance": round(_min_ceramic_distance_xy(obj, world_state), 4),
                "obstacle_status": status,
            })
        elif reach == "HARD":
            skipped.append({
                "object_id": obj["id"],
                "group": "cube",
                "stage": "precheck",
                "failure_reason": "object_outside_conservative_reach",
                "reach_distance": round(_reach_distance_xy(obj, world_state), 4),
                "reach_status": reach,
            })
        else:
            eligible_cubes.append(obj)

    for obj in circles:
        status = _obstacle_status(obj, world_state)
        reach = _reach_status(obj, world_state)
        if status == "TOO_CLOSE":
            skipped.append({
                "object_id": obj["id"],
                "group": "circle",
                "stage": "precheck",
                "failure_reason": "object_too_close_to_obstacle",
                "obstacle_distance": round(_min_ceramic_distance_xy(obj, world_state), 4),
                "obstacle_status": status,
            })
        elif reach == "HARD":
            skipped.append({
                "object_id": obj["id"],
                "group": "circle",
                "stage": "precheck",
                "failure_reason": "object_outside_conservative_reach",
                "reach_distance": round(_reach_distance_xy(obj, world_state), 4),
                "reach_status": reach,
            })
        else:
            eligible_circles.append(obj)

    table = world_state["table"]
    goal_x, goal_y, _ = world_state["goal_center"]
    cube_row_y = goal_y + CUBE_ROW_OFFSET_Y
    circle_row_y = goal_y + CIRCLE_ROW_OFFSET_Y

    # Shared occupied list so cube and circle targets don't overlap each other.
    occupied = []
    slots = {}

    # Row separation tolerance: displaced targets may not cross into the other row.
    # Cubes stay at or below cube_row_y + 0.025; circles stay at or above circle_row_y - 0.025.
    _cube_y_max = cube_row_y + 0.025
    _circle_y_min = circle_row_y - 0.025

    cube_start_x = goal_x - CUBE_SPACING * (len(eligible_cubes) - 1) / 2.0
    for index, obj in enumerate(eligible_cubes):
        radius = obj.get("radius", 0.043)
        base_x = cube_start_x + index * CUBE_SPACING
        target_xy = _search_safe_target_xy(base_x, cube_row_y, radius, world_state, occupied, y_max=_cube_y_max)
        if target_xy is None:
            skipped.append({
                "object_id": obj["id"],
                "group": "cube",
                "stage": "target_allocation",
                "failure_reason": "no_safe_aligned_target_found",
                "desired_xy": [round(base_x, 4), round(cube_row_y, 4)],
            })
            continue
        x, y = target_xy
        occupied.append((x, y, radius))
        slots[obj["id"]] = {
            "group": "cube",
            "slot_id": f"cube_slot_{index + 1:02d}",
            "target_pose": [round(x, 4), round(y, 4), round(table["z_top"] + 0.03, 4), 1.0, 0.0, 0.0, 0.0],
        }

    circle_start_x = goal_x - CIRCLE_SPACING * (len(eligible_circles) - 1) / 2.0
    for index, obj in enumerate(eligible_circles):
        radius = obj.get("radius", 0.03)
        base_x = circle_start_x + index * CIRCLE_SPACING
        target_xy = _search_safe_target_xy(base_x, circle_row_y, radius, world_state, occupied, y_min=_circle_y_min)
        if target_xy is None:
            skipped.append({
                "object_id": obj["id"],
                "group": "circle",
                "stage": "target_allocation",
                "failure_reason": "no_safe_aligned_target_found",
                "desired_xy": [round(base_x, 4), round(circle_row_y, 4)],
            })
            continue
        x, y = target_xy
        occupied.append((x, y, radius))
        slots[obj["id"]] = {
            "group": "circle",
            "slot_id": f"circle_slot_{index + 1:02d}",
            "target_pose": [round(x, 4), round(y, 4), round(table["z_top"] + 0.04, 4), 1.0, 0.0, 0.0, 0.0],
        }

    # Interleave move order by difficulty so easiest objects from both groups go first.
    all_eligible = [o for o in eligible_cubes + eligible_circles if o["id"] in slots]
    all_eligible.sort(key=sort_key)
    move_order = [o["id"] for o in all_eligible]

    return world_state, move_order, slots, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Separate cubes to one row and circles to another using OMPL + MuJoCo collision checking only."
    )
    parser.add_argument(
        "--object", nargs="+", default=["ungroup", "no", "obs"],
        help="Scene: group no obs, ungroup no obs, group obs, ungroup obs.",
    )
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-hint-cache", action="store_true", help="Disable HintCache adaptive learning (use fixed defaults).")
    parser.add_argument("--place-retries", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--settle-after-place", type=int, default=300, help=argparse.SUPPRESS)
    args = parser.parse_args()

    scene_key = normalize_scene_key(args.object)
    scene_file = prepare_scene_variant(scene_key)
    event_csv_path = make_event_log_path("separate_groups", scene_key, args.log_dir)
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
        print("[SEPARATE_GROUPS] OMPL Python binding belum tersedia.")
        print("  pip install ompl")
        print("  python -c \"from ompl import base, geometric; print('ompl ok')\"")
        return 2

    try:
        world_state, move_order, slots_by_object, skipped_precheck = _build_separate_groups_targets(
            scene_file=str(scene_file),
        )
    except ValueError as exc:
        print(f"[SEPARATE_GROUPS] Target allocation invalid: {exc}")
        return 1

    if not move_order:
        print("[SEPARATE_GROUPS] Tidak ada objek yang bisa dipindahkan.")
        return 1

    cube_targets = {oid: s for oid, s in slots_by_object.items() if s["group"] == "cube"}
    circle_targets = {oid: s for oid, s in slots_by_object.items() if s["group"] == "circle"}

    print("[SEPARATE_GROUPS] Task          : pisahkan kubus (row-) dan lingkaran (row+)")
    print(f"[SEPARATE_GROUPS] Scene variant : {scene_key} ({scene_file})")
    print(f"[SEPARATE_GROUPS] Event CSV     : {event_csv_path}")
    print(f"[SEPARATE_GROUPS] Scene objects  : movable={world_state['counts']['movable']} ceramic={world_state['counts']['ceramic']}")
    print(f"[SEPARATE_GROUPS] Ceramic avoid  : {[o['id'] for o in world_state['ceramic_obstacles']]}")
    print(f"[SEPARATE_GROUPS] Move order     : {move_order}")
    if skipped_precheck:
        print("[SEPARATE_GROUPS] Skipped before target allocation:")
        for item in skipped_precheck:
            print(f"  - {item['object_id']} ({item.get('group', '?')}): {item['failure_reason']}")
    print("[SEPARATE_GROUPS] Cube targets (row-):")
    for oid, slot in cube_targets.items():
        x, y, z = slot["target_pose"][:3]
        obj = next(o for o in world_state["movable_objects"] if o["id"] == oid)
        print(
            f"  - {oid:>6} -> ({x:.3f}, {y:.3f}, {z:.3f}) "
            f"reach={_reach_distance_xy(obj, world_state):.3f}m "
            f"obs_status={_obstacle_status(obj, world_state)}"
        )
    print("[SEPARATE_GROUPS] Circle targets (row+):")
    for oid, slot in circle_targets.items():
        x, y, z = slot["target_pose"][:3]
        obj = next(o for o in world_state["movable_objects"] if o["id"] == oid)
        print(
            f"  - {oid:>6} -> ({x:.3f}, {y:.3f}, {z:.3f}) "
            f"reach={_reach_distance_xy(obj, world_state):.3f}m "
            f"obs_status={_obstacle_status(obj, world_state)}"
        )

    import executor
    import feedback
    from exec_trace import flush as flush_trace
    from exec_trace import log_event

    if args.no_hint_cache:
        print("[SEPARATE_GROUPS] HintCache disabled (--no-hint-cache).")
    else:
        executor.init_hint_cache(log_dir=args.log_dir, scene_filter=scene_key)

    if not getattr(executor, "_OMPL_AVAILABLE", False):
        print("[SEPARATE_GROUPS] Executor tidak melihat OMPL planner aktif.")
        return 2

    log_event(
        "TASK_CONTEXT",
        "START",
        phase="separate_groups",
        scene=scene_key,
        model_file=str(scene_file),
        move_order=move_order,
        targets={oid: slots_by_object[oid]["target_pose"] for oid in move_order},
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
            phase="separate_groups",
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
        log_event("TASK_ROUND", "START", phase="separate_groups", round=retry_round, candidates=current_round)

        for object_id in current_round:
            if object_id in moved:
                continue

            index = object_index[object_id]
            slot = slots_by_object[object_id]
            target_pose = slot["target_pose"]
            group = slot["group"]
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
                    f"\n[{index:02d}] SKIP_OBJECT object={object_id} group={group} "
                    f"reason={reason} obstacle_distance={obstacle_distance:.3f}m"
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
                    "group": group,
                    "stage": "precheck",
                    "failure_reason": reason,
                    "obstacle_distance": round(obstacle_distance, 4),
                })
                continue

            if obstacle_status == "NEAR":
                print(
                    f"\n[{index:02d}] CAUTIOUS_OBJECT object={object_id} group={group} "
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
            print(
                f"\n[{index:02d}] SEPARATE_GROUPS START object={object_id} group={group} "
                f"target=({x:.3f}, {y:.3f})"
            )
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
                    "group": group,
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
                        "group": group,
                        "stage": "pick",
                        "failure_reason": terminal_reason or "object_not_lifted_after_pick",
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
                        "group": slots_by_object[did]["group"],
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
                        "group": group,
                        "stage": "place",
                        "failure_reason": reason,
                        "error": str(exc),
                    })
                    place_ok = False
                    break
                _settle(executor, args.settle_after_place)
                place_ok, actual = feedback.check_place(
                    executor.model, executor.data, executor.name_to_cube[object_id], x, y
                )
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
                placed_targets[object_id] = (x, y)
                _settle(executor, args.settle_after_place)
            else:
                already_failed = any(
                    item.get("object_id") == object_id and item.get("stage") == "place"
                    for item in failed
                )
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
                        "group": group,
                        "stage": "place",
                        "failure_reason": terminal_reason or "object_not_on_target_after_place",
                        "actual": actual,
                    })

    duration_ms = int((time.perf_counter() - started) * 1000)
    success = len(failed) == 0
    print("\n=== Separate groups OMPL-only summary ===")
    print(f"success={success}")
    print(f"objects_moved={len(moved)}")
    print(f"failed={failed}")
    print(f"duration_ms={duration_ms}")
    print("collision_policy=MuJoCo contacts via PandaOMPLPlanner/CollisionPolicy")
    print("llm_used=false")
    csv_path = write_summary_csv(
        "separate_groups",
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


if __name__ == "__main__":
    safe_process_exit(main())
