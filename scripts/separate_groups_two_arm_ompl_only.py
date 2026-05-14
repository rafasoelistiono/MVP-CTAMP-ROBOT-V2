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
    make_event_log_path,
    normalize_scene_key,
    prepare_two_arm_scene_variant,
    write_summary_csv,
)

REACH_BORDERLINE_M = 0.78
REACH_HARD_M = 0.82
OBSTACLE_TOO_CLOSE_M = 0.12
OBSTACLE_NEAR_M = 0.18


def _vec(raw, default):
    if raw is None or raw.strip() == "":
        return list(default)
    return [float(part) for part in raw.split()]


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


def _is_robot_body(name: str) -> bool:
    if name.startswith(("link", "right_link")):
        return True
    return name in {
        "hand",
        "left_finger",
        "right_finger",
        "right_hand",
        "right_left_finger",
        "right_right_finger",
    }


def _class_from_name_shape(name: str, shape: str) -> str:
    lower = name.lower()
    if "cube" in lower or shape == "box":
        return "cube"
    if "circle" in lower or shape in {"cylinder", "ellipsoid"}:
        return "circle"
    return shape


def _parse_scene(scene_file: str):
    scene_path = Path(scene_file)
    if not scene_path.exists():
        scene_path = ROOT_DIR / "models" / scene_file
    roots = _load_roots(scene_path.resolve())
    bodies = []
    for root in roots:
        worldbody = root.find("worldbody")
        if worldbody is not None:
            bodies.extend(_iter_bodies(worldbody))

    objects = []
    goal_center = [0.22, -0.06, 0.806]
    arm_bases = {}
    for body in bodies:
        name = body.get("name", "")
        if name == "goal_area":
            goal_center = _vec(body.get("pos"), goal_center)
            continue
        if name == "link0":
            arm_bases["left"] = _vec(body.get("pos"), [-0.4, -0.45, 0.8])[:2]
            continue
        if name == "right_link0":
            arm_bases["right"] = _vec(body.get("pos"), [-0.4, 0.45, 0.8])[:2]
            continue
        if not name or name == "table" or _is_robot_body(name):
            continue
        geoms = body.findall("geom")
        if not geoms:
            continue
        shape = geoms[0].get("type", "mesh")
        fragile = any(token in name.lower() for token in ("obstacle", "vase", "glass", "ceramic"))
        movable = any(joint.get("type") == "free" for joint in body.findall("joint")) and not fragile
        pos = _vec(body.get("pos"), [0.0, 0.0, 0.0])
        objects.append({
            "id": name,
            "class": _class_from_name_shape(name, shape),
            "position": pos,
            "movable": movable,
            "fragile": fragile,
        })

    if "left" not in arm_bases:
        arm_bases["left"] = [-0.4, -0.45]
    if "right" not in arm_bases:
        arm_bases["right"] = [-0.4, 0.45]

    return {
        "objects": objects,
        "goal_center": goal_center,
        "arm_bases": arm_bases,
        "movable_objects": [o for o in objects if o["movable"]],
        "ceramic_obstacles": [o for o in objects if o["fragile"]],
    }


def _min_obstacle_distance_xy(obj, scene) -> float:
    ox, oy = obj["position"][:2]
    distances = []
    for obstacle in scene.get("ceramic_obstacles", []):
        cx, cy = obstacle["position"][:2]
        distances.append(math.dist((ox, oy), (cx, cy)))
    return min(distances) if distances else 999.0


def _obstacle_status(obj, scene) -> str:
    distance = _min_obstacle_distance_xy(obj, scene)
    if distance <= OBSTACLE_TOO_CLOSE_M:
        return "TOO_CLOSE"
    if distance <= OBSTACLE_NEAR_M:
        return "NEAR"
    return "CLEAR"


def _reach_distance_xy(x: float, y: float, arm: str, scene) -> float:
    return math.dist((x, y), tuple(scene["arm_bases"][arm]))


def _reach_status_xy(x: float, y: float, arm: str, scene) -> str:
    distance = _reach_distance_xy(x, y, arm, scene)
    if distance > REACH_HARD_M:
        return "HARD"
    if distance > REACH_BORDERLINE_M:
        return "BORDERLINE"
    return "OK"


def _object_reach_status(obj, arm: str, scene) -> str:
    x, y = obj["position"][:2]
    return _reach_status_xy(x, y, arm, scene)


def _arm_validation(obj, target_xy, arm: str, scene, arm_load: int, blocked_arms=None):
    blocked_arms = set(blocked_arms or [])
    ox, oy = obj["position"][:2]
    tx, ty = target_xy
    pick_distance = _reach_distance_xy(ox, oy, arm, scene)
    place_distance = _reach_distance_xy(tx, ty, arm, scene)
    pick_status = _reach_status_xy(ox, oy, arm, scene)
    place_status = _reach_status_xy(tx, ty, arm, scene)
    obstacle_status = _obstacle_status(obj, scene)
    ok = (
        arm not in blocked_arms
        and obstacle_status == "CLEAR"
        and pick_status != "HARD"
        and place_status != "HARD"
    )
    score = (
        pick_distance
        + 0.65 * place_distance
        + 0.12 * arm_load
        + (0.20 if pick_status == "BORDERLINE" else 0.0)
        + (0.12 if place_status == "BORDERLINE" else 0.0)
    )
    return {
        "arm": arm,
        "ok": ok,
        "score": score,
        "pick_distance": pick_distance,
        "place_distance": place_distance,
        "pick_status": pick_status,
        "place_status": place_status,
        "obstacle_status": obstacle_status,
        "blocked": arm in blocked_arms,
    }


def _select_arm(obj, target_xy, scene, arm_loads, blocked_arms=None):
    validations = [
        _arm_validation(obj, target_xy, arm, scene, arm_loads.get(arm, 0), blocked_arms)
        for arm in scene["arm_bases"]
    ]
    valid = [item for item in validations if item["ok"]]
    if not valid:
        return None, validations
    return min(valid, key=lambda item: item["score"]), validations


def _easy_first_two_arm(obj, scene):
    statuses = [_object_reach_status(obj, arm, scene) for arm in scene["arm_bases"]]
    best_distance = min(
        _reach_distance_xy(obj["position"][0], obj["position"][1], arm, scene)
        for arm in scene["arm_bases"]
    )
    return (
        all(status == "HARD" for status in statuses),
        all(status != "OK" for status in statuses),
        best_distance,
        obj["position"][0],
        obj["position"][1],
    )


def _build_two_arm_group_targets(scene_file: str, spacing: float):
    scene = _parse_scene(scene_file)
    goal_x, goal_y, _ = scene["goal_center"]
    skipped = []
    eligible = []
    for obj in scene["movable_objects"]:
        obstacle_status = _obstacle_status(obj, scene)
        if obstacle_status != "CLEAR":
            skipped.append({
                "object_id": obj["id"],
                "stage": "precheck",
                "failure_reason": (
                    "object_too_close_to_obstacle"
                    if obstacle_status == "TOO_CLOSE"
                    else "object_near_obstacle_safety_skip"
                ),
                "obstacle_distance": round(_min_obstacle_distance_xy(obj, scene), 4),
                "obstacle_status": obstacle_status,
            })
            continue
        if all(_object_reach_status(obj, arm, scene) == "HARD" for arm in scene["arm_bases"]):
            skipped.append({
                "object_id": obj["id"],
                "stage": "precheck",
                "failure_reason": "object_outside_all_arm_reach",
            })
            continue
        eligible.append(obj)

    groups = {
        "cube": sorted([o for o in eligible if o["class"] == "cube"], key=lambda obj: _easy_first_two_arm(obj, scene)),
        "circle": sorted([o for o in eligible if o["class"] == "circle"], key=lambda obj: _easy_first_two_arm(obj, scene)),
    }

    row_y = {
        "cube": goal_y - 0.065,
        "circle": goal_y + 0.065,
    }
    slots = {}
    arm_loads = {arm: 0 for arm in scene["arm_bases"]}
    assignment = {}
    for class_name, objects in groups.items():
        n = len(objects)
        start_x = goal_x - spacing * (n - 1) / 2.0
        for index, obj in enumerate(objects):
            target_pose = [
                round(start_x + index * spacing, 4),
                round(row_y[class_name], 4),
                0.83,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
            selected, validations = _select_arm(obj, target_pose[:2], scene, arm_loads)
            if selected is None:
                skipped.append({
                    "object_id": obj["id"],
                    "stage": "precheck",
                    "failure_reason": "no_arm_passed_minimum_validator",
                    "arm_validations": _round_validations(validations),
                })
                continue
            slots[obj["id"]] = {"class": class_name, "target_pose": target_pose}
            assignment[obj["id"]] = selected["arm"]
            arm_loads[selected["arm"]] += 1

    move_order = [obj["id"] for obj in groups["cube"] if obj["id"] in slots]
    move_order += [obj["id"] for obj in groups["circle"] if obj["id"] in slots]
    return scene, move_order, slots, assignment, skipped


def _round_validations(validations):
    rounded = []
    for item in validations:
        rounded.append({
            "arm": item["arm"],
            "ok": item["ok"],
            "pick_distance": round(item["pick_distance"], 4),
            "place_distance": round(item["place_distance"], 4),
            "pick_status": item["pick_status"],
            "place_status": item["place_status"],
            "obstacle_status": item["obstacle_status"],
            "blocked": item["blocked"],
        })
    return rounded


def _settle(executor, steps: int) -> None:
    for _ in range(max(0, steps)):
        executor.mujoco.mj_step(executor.model, executor.data)
        executor.viewer.sync()


def _execute_object_once(
    executor,
    feedback,
    object_id: str,
    x: float,
    y: float,
    settle_steps: int,
    pick_attempt: int,
    max_pick_attempts: int,
):
    print(f"    PICK_ATTEMPT {pick_attempt}/{max_pick_attempts} arm={executor.ACTIVE_ARM}")
    executor.pick(object_id)
    pick_ok, pick_z = feedback.check_pick(executor.model, executor.data, executor.name_to_cube[object_id])
    print(f"    CHECK_PICK  {'OK' if pick_ok else 'FAIL'} arm={executor.ACTIVE_ARM} z={pick_z}")
    if not pick_ok:
        executor.drop()
        _settle(executor, settle_steps)
        return False, {"object_id": object_id, "stage": "pick", "z": pick_z, "attempts": pick_attempt, "arm": executor.ACTIVE_ARM}

    executor.place(x, y, object_id)
    _settle(executor, settle_steps)
    place_ok, actual = feedback.check_place(executor.model, executor.data, executor.name_to_cube[object_id], x, y)
    print(f"    CHECK_PLACE {'OK' if place_ok else 'FAIL'} arm={executor.ACTIVE_ARM} actual={actual}")
    if not place_ok:
        executor.drop()
        _settle(executor, settle_steps)
        return False, {"object_id": object_id, "stage": "place", "actual": actual, "arm": executor.ACTIVE_ARM}
    return True, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Separate square/circle objects using a two-arm scene and side-switch validator.")
    parser.add_argument("--object", nargs="+", default=["ungroup", "obs"], help="Scene: group no obs, ungroup no obs, group obs, ungroup obs.")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--settle-steps", type=int, default=500, help=argparse.SUPPRESS)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    scene_key = normalize_scene_key(args.object)
    scene_file = prepare_two_arm_scene_variant(scene_key)
    event_csv_path = make_event_log_path("separate_groups_two_arm", scene_key, args.log_dir)
    os.environ["MODEL_FILE"] = str(scene_file)
    os.environ["CTAMP_EVENT_LOG_CSV"] = str(event_csv_path)
    os.environ.setdefault("OMPL_ENABLED", "true")
    os.environ.setdefault("USE_IK_FALLBACK", "false")
    os.environ.setdefault("ACTIVE_ARM", "left")
    apply_conservative_motion_defaults()
    os.environ["ENABLE_VIEWER"] = "false" if args.no_viewer else "true"

    try:
        from ompl import base, geometric  # noqa: F401
    except ImportError:
        print("[SEPARATE_GROUPS_TWO_ARM] OMPL Python binding belum tersedia.")
        print("  pip install ompl")
        print("  python -c \"from ompl import base, geometric; print('ompl ok')\"")
        return 2

    scene, move_order, slots, assignment, skipped_precheck = _build_two_arm_group_targets(str(scene_file), spacing=0.11)

    print("[SEPARATE_GROUPS_TWO_ARM] Task         : pisahkan square dan circle dengan two-arm validator")
    print(f"[SEPARATE_GROUPS_TWO_ARM] Scene variant: {scene_key} ({scene_file})")
    print(f"[SEPARATE_GROUPS_TWO_ARM] Event CSV    : {event_csv_path}")
    print(f"[SEPARATE_GROUPS_TWO_ARM] Arm bases    : {scene['arm_bases']}")
    print(f"[SEPARATE_GROUPS_TWO_ARM] Ceramic avoid: {[o['id'] for o in scene['ceramic_obstacles']]}")
    if skipped_precheck:
        print("[SEPARATE_GROUPS_TWO_ARM] Skipped before target allocation:")
        for item in skipped_precheck:
            print(f"  - {item['object_id']}: {item['failure_reason']}")
    print("[SEPARATE_GROUPS_TWO_ARM] Targets:")
    for object_id in move_order:
        slot = slots[object_id]
        x, y, z = slot["target_pose"][:3]
        obj = next(o for o in scene["movable_objects"] if o["id"] == object_id)
        selected, validations = _select_arm(obj, (x, y), scene, {arm: 0 for arm in scene["arm_bases"]})
        print(
            f"  - {object_id:>7} ({slot['class']:<6}) -> ({x:.3f}, {y:.3f}, {z:.3f}) "
            f"arm={assignment[object_id]} validators={_round_validations(validations)}"
        )

    import executor
    import feedback
    from exec_trace import flush as flush_trace
    from exec_trace import log_event

    if "right" not in executor.available_arms():
        print("[SEPARATE_GROUPS_TWO_ARM] Scene tidak memuat right arm di executor.")
        return 1

    started = time.perf_counter()
    moved = []
    failed = list(skipped_precheck)
    max_pick_attempts = 4
    pick_attempts = {object_id: 0 for object_id in move_order}
    blocked_arms_by_object = {object_id: set() for object_id in move_order}
    pending = list(move_order)
    object_index = {object_id: index for index, object_id in enumerate(move_order, start=1)}
    arm_loads = {arm: 0 for arm in scene["arm_bases"]}

    for item in skipped_precheck:
        log_event("OBJECT_PRECHECK", "SKIP", object_id=item["object_id"], phase="separate_groups_two_arm", failure_reason=item["failure_reason"])

    retry_round = 0
    while pending:
        retry_round += 1
        current_round = pending
        pending = []
        print(f"\n[ROUND {retry_round}] candidates={current_round}")
        log_event("TASK_ROUND", "START", phase="separate_groups_two_arm", round=retry_round, candidates=current_round)

        for object_id in current_round:
            if object_id in moved:
                continue
            index = object_index[object_id]
            x, y = slots[object_id]["target_pose"][:2]
            obj = next(o for o in scene["movable_objects"] if o["id"] == object_id)
            selected, validations = _select_arm(
                obj,
                (x, y),
                scene,
                arm_loads,
                blocked_arms=blocked_arms_by_object[object_id],
            )
            if selected is None:
                print(f"\n[{index:02d}] SKIP_OBJECT object={object_id} reason=no_arm_passed_minimum_validator")
                failure = {
                    "object_id": object_id,
                    "stage": "precheck",
                    "failure_reason": "no_arm_passed_minimum_validator",
                    "arm_validations": _round_validations(validations),
                }
                log_event("ARM_VALIDATION", "FAILED", object_id=object_id, validations=failure["arm_validations"])
                failed.append(failure)
                continue

            selected_arm = selected["arm"]
            executor.set_active_arm(selected_arm)
            pick_attempts[object_id] += 1
            print(
                f"\n[{index:02d}] MOVE object={object_id} arm={selected_arm} -> ({x:.3f}, {y:.3f}) "
                f"validator={_round_validations(validations)}"
            )
            log_event(
                "ARM_VALIDATION",
                "OK",
                object_id=object_id,
                selected_arm=selected_arm,
                validations=_round_validations(validations),
                attempt=pick_attempts[object_id],
            )

            ok, failure = _execute_object_once(
                executor,
                feedback,
                object_id,
                float(x),
                float(y),
                args.settle_steps,
                pick_attempts[object_id],
                max_pick_attempts,
            )
            if ok:
                moved.append(object_id)
                arm_loads[selected_arm] += 1
                continue

            if failure and failure.get("stage") == "pick":
                blocked_arms_by_object[object_id].add(selected_arm)
                alternate, _ = _select_arm(
                    obj,
                    (x, y),
                    scene,
                    arm_loads,
                    blocked_arms=blocked_arms_by_object[object_id],
                )
                if alternate is not None and pick_attempts[object_id] < max_pick_attempts:
                    print(
                        f"    SWITCH_ARM object={object_id} failed_arm={selected_arm} "
                        f"next_arm={alternate['arm']}; try_next_candidate_first"
                    )
                    log_event(
                        "SWITCH_ARM",
                        "RETRY_LATER",
                        object_id=object_id,
                        failed_arm=selected_arm,
                        next_arm=alternate["arm"],
                        attempt=pick_attempts[object_id],
                    )
                    pending.append(object_id)
                    continue

            failed.append(failure)

    duration_ms = int((time.perf_counter() - started) * 1000)
    success = len(failed) == 0
    print("\n=== Separate groups two-arm summary ===")
    print(f"success={success}")
    print(f"objects_moved={len(moved)}")
    print(f"failed={failed}")
    print(f"duration_ms={duration_ms}")
    print("llm_used=false")
    csv_path = write_summary_csv(
        "separate_groups_two_arm",
        scene_key,
        {
            "success": success,
            "objects_moved": len(moved),
            "objects_total": len(move_order) + len(skipped_precheck),
            "failed": failed,
            "duration_ms": duration_ms,
        },
        args.log_dir,
    )
    print(f"csv_log={csv_path}")
    print(f"event_csv_log={event_csv_path}")
    flush_trace()
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
