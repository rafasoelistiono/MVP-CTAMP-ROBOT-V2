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
    prepare_scene_variant,
    write_summary_csv,
)

ROBOT_BASE_XY = (-0.4, 0.0)
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
    goal_center = [0.23, -0.18, 0.806]
    for body in bodies:
        name = body.get("name", "")
        if name == "goal_area":
            goal_center = _vec(body.get("pos"), goal_center)
            continue
        if not name or name == "table" or name.startswith("link") or name in {"hand", "left_finger", "right_finger"}:
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

    return {
        "objects": objects,
        "goal_center": goal_center,
        "movable_objects": [o for o in objects if o["movable"]],
        "ceramic_obstacles": [o for o in objects if o["fragile"]],
    }


def _class_from_name_shape(name: str, shape: str) -> str:
    lower = name.lower()
    if "cube" in lower or shape == "box":
        return "cube"
    if "circle" in lower or shape in {"cylinder", "ellipsoid"}:
        return "circle"
    return shape


def _distance_to_goal(obj, goal_center) -> float:
    return math.dist(obj["position"][:2], goal_center[:2])


def _reach_distance(obj) -> float:
    return math.dist(obj["position"][:2], ROBOT_BASE_XY)


def _reach_status(obj) -> str:
    distance = _reach_distance(obj)
    if distance > REACH_HARD_M:
        return "HARD"
    if distance > REACH_BORDERLINE_M:
        return "BORDERLINE"
    return "OK"


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


def _left_to_right(obj):
    x, y = obj["position"][:2]
    return x, y


def _easy_first(obj):
    status = _reach_status(obj)
    return (status == "HARD", status == "BORDERLINE", _reach_distance(obj), obj["position"][0], obj["position"][1])


def _build_group_targets(scene_file: str, per_class: int | None, spacing: float):
    scene = _parse_scene(scene_file)
    goal_x, goal_y, _ = scene["goal_center"]
    skipped = []
    eligible = []
    for obj in scene["movable_objects"]:
        obstacle_status = _obstacle_status(obj, scene)
        reach_status = _reach_status(obj)
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
        if reach_status == "HARD":
            skipped.append({
                "object_id": obj["id"],
                "stage": "precheck",
                "failure_reason": "object_outside_conservative_reach",
                "reach_distance": round(_reach_distance(obj), 4),
                "reach_status": reach_status,
            })
            continue
        eligible.append(obj)
    groups = {
        "cube": sorted(
            [o for o in eligible if o["class"] == "cube"],
            key=lambda obj: (
                _obstacle_status(obj, scene) == "TOO_CLOSE",
                _obstacle_status(obj, scene) == "NEAR",
                _easy_first(obj),
            ),
        ),
        "circle": sorted(
            [o for o in eligible if o["class"] == "circle"],
            key=lambda obj: (
                _obstacle_status(obj, scene) == "TOO_CLOSE",
                _obstacle_status(obj, scene) == "NEAR",
                _easy_first(obj),
            ),
        ),
    }
    if per_class is not None:
        groups = {name: objs[:per_class] for name, objs in groups.items()}

    row_y = {
        "cube": goal_y - 0.065,
        "circle": goal_y + 0.065,
    }
    slots = {}
    for class_name, objects in groups.items():
        n = len(objects)
        start_x = goal_x - spacing * (n - 1) / 2.0
        for index, obj in enumerate(objects):
            slots[obj["id"]] = {
                "class": class_name,
                "target_pose": [round(start_x + index * spacing, 4), round(row_y[class_name], 4), 0.83, 1.0, 0.0, 0.0, 0.0],
            }
    move_order = [obj["id"] for obj in groups["cube"]] + [obj["id"] for obj in groups["circle"]]
    return scene, move_order, slots, skipped


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
    print(f"    PICK_ATTEMPT {pick_attempt}/{max_pick_attempts}")
    executor.pick(object_id)
    pick_ok, pick_z = feedback.check_pick(executor.model, executor.data, executor.name_to_cube[object_id])
    print(f"    CHECK_PICK  {'OK' if pick_ok else 'FAIL'} attempt={pick_attempt} z={pick_z}")
    if not pick_ok:
        executor.drop()
        _settle(executor, settle_steps)
        return False, {"object_id": object_id, "stage": "pick", "z": pick_z, "attempts": pick_attempt}

    executor.place(x, y, object_id)
    _settle(executor, settle_steps)
    place_ok, actual = feedback.check_place(executor.model, executor.data, executor.name_to_cube[object_id], x, y)
    print(f"    CHECK_PLACE {'OK' if place_ok else 'FAIL'} actual={actual}")
    if not place_ok:
        executor.drop()
        _settle(executor, settle_steps)
        return False, {"object_id": object_id, "stage": "place", "actual": actual}
    return True, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Separate random square/circle objects into two neat groups in the goal area.")
    parser.add_argument("--object", nargs="+", default=["group", "no", "obs"], help="Scene: group no obs, ungroup no obs, group obs, ungroup obs.")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--settle-steps", type=int, default=500, help=argparse.SUPPRESS)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    scene_key = normalize_scene_key(args.object)
    scene_file = prepare_scene_variant(scene_key)
    event_csv_path = make_event_log_path("separate_groups", scene_key, args.log_dir)
    os.environ["MODEL_FILE"] = str(scene_file)
    os.environ["CTAMP_EVENT_LOG_CSV"] = str(event_csv_path)
    os.environ.setdefault("OMPL_ENABLED", "true")
    os.environ.setdefault("USE_IK_FALLBACK", "false")
    apply_conservative_motion_defaults()
    os.environ["ENABLE_VIEWER"] = "false" if args.no_viewer else "true"

    try:
        from ompl import base, geometric  # noqa: F401
    except ImportError:
        print("[SEPARATE_GROUPS] OMPL Python binding belum tersedia.")
        print("  pip install ompl")
        print("  python -c \"from ompl import base, geometric; print('ompl ok')\"")
        return 2

    scene, move_order, slots, skipped_precheck = _build_group_targets(str(scene_file), per_class=None, spacing=0.11)

    print("[SEPARATE_GROUPS] Task         : pisahkan square dan circle ke goal area")
    print(f"[SEPARATE_GROUPS] Scene variant: {scene_key} ({scene_file})")
    print(f"[SEPARATE_GROUPS] Event CSV    : {event_csv_path}")
    print(f"[SEPARATE_GROUPS] Goal center  : {scene['goal_center'][:2]}")
    print(f"[SEPARATE_GROUPS] Ceramic avoid: {[o['id'] for o in scene['ceramic_obstacles']]}")
    if skipped_precheck:
        print("[SEPARATE_GROUPS] Skipped before target allocation:")
        for item in skipped_precheck:
            print(f"  - {item['object_id']}: {item['failure_reason']}")
    print("[SEPARATE_GROUPS] Targets:")
    for object_id in move_order:
        slot = slots[object_id]
        x, y, z = slot["target_pose"][:3]
        obj = next(o for o in scene["movable_objects"] if o["id"] == object_id)
        obstacle_distance = _min_obstacle_distance_xy(obj, scene)
        print(
            f"  - {object_id:>7} ({slot['class']:<6}) -> ({x:.3f}, {y:.3f}, {z:.3f}) "
            f"reach={_reach_distance(obj):.3f}m status={_reach_status(obj)} "
            f"obs={obstacle_distance:.3f}m obs_status={_obstacle_status(obj, scene)}"
        )

    import executor
    import feedback
    from exec_trace import flush as flush_trace
    from exec_trace import log_event

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
            x, y = slots[object_id]["target_pose"][:2]
            obj = next(o for o in scene["movable_objects"] if o["id"] == object_id)
            obstacle_distance = _min_obstacle_distance_xy(obj, scene)
            obstacle_status = _obstacle_status(obj, scene)
            reach_status = _reach_status(obj)
            if obstacle_status != "CLEAR" or reach_status == "HARD":
                reason = (
                    "object_too_close_to_obstacle"
                    if obstacle_status == "TOO_CLOSE"
                    else "object_near_obstacle_safety_skip"
                    if obstacle_status == "NEAR"
                    else "object_outside_conservative_reach"
                )
                print(
                    f"\n[{index:02d}] SKIP_OBJECT object={object_id} "
                    f"reason={reason} obstacle_distance={obstacle_distance:.3f}m "
                    f"reach={_reach_distance(obj):.3f}m; try_next_candidate"
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
            pick_attempts[object_id] += 1
            print(f"\n[{index:02d}] MOVE {object_id} -> ({x:.3f}, {y:.3f})")
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
                continue
            if failure and failure.get("stage") == "pick" and pick_attempts[object_id] < max_pick_attempts:
                print(
                    f"    DEFER_OBJECT object={object_id} reason=pick_failed "
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
                continue
            failed.append(failure)

    duration_ms = int((time.perf_counter() - started) * 1000)
    success = len(failed) == 0
    print("\n=== Separate groups summary ===")
    print(f"success={success}")
    print(f"objects_moved={len(moved)}")
    print(f"failed={failed}")
    print(f"duration_ms={duration_ms}")
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
        },
        args.log_dir,
    )
    print(f"csv_log={csv_path}")
    print(f"event_csv_log={event_csv_path}")
    flush_trace()
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
