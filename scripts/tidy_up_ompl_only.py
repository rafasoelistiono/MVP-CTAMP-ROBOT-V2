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

OBSTACLE_TOO_CLOSE_M = 0.12
OBSTACLE_NEAR_M = 0.18


def _build_cube_tidy_plan(scene_file: str):
    world_state = _parse_scene(scene_file)
    candidates = sorted(
        world_state["movable_objects"],
        key=lambda obj: (
            _obstacle_status(obj, world_state) == "TOO_CLOSE",
            _obstacle_status(obj, world_state) == "NEAR",
            _reach_status(obj, world_state) == "HARD",
            _reach_status(obj, world_state) == "BORDERLINE",
            0 if obj.get("class") == "cube" else 1,
            obj["position"][0],
            obj["position"][1],
        ),
    )
    selected = []
    skipped = []
    for obj in candidates:
        obstacle_status = _obstacle_status(obj, world_state)
        reach_status = _reach_status(obj, world_state)
        if obstacle_status != "CLEAR":
            skipped.append({
                "object_id": obj["id"],
                "stage": "precheck",
                "failure_reason": (
                    "object_too_close_to_obstacle"
                    if obstacle_status == "TOO_CLOSE"
                    else "object_near_obstacle_safety_skip"
                ),
                "obstacle_distance": round(_min_ceramic_distance_xy(obj, world_state), 4),
                "obstacle_status": obstacle_status,
            })
            continue
        if reach_status == "HARD":
            skipped.append({
                "object_id": obj["id"],
                "stage": "precheck",
                "failure_reason": "object_outside_conservative_reach",
                "reach_distance": round(_reach_distance_xy(obj, world_state), 4),
                "reach_status": reach_status,
            })
            continue
        selected.append(obj)
    object_ids = [o["id"] for o in selected]

    plan = {
        "task_name": "ompl_only_tidy_up_cubes",
        "movable_objects": object_ids,
        "obstacles_to_avoid": [o["id"] for o in world_state["ceramic_obstacles"]],
        "goal_layout": {
            "type": "aligned_rows",
            "primary_axis": "x",
            "grouping": [
                {"class": "cube", "row": 0, "order": "left_to_right"},
                {"class": "circle", "row": 1, "order": "left_to_right"},
            ],
            "spacing_policy": "object_diameter_plus_clearance",
            "avoid_regions": ["inflated_ceramic_regions"],
        },
        "preferred_order": ["nearest_first", "objects_blocking_target_area_first"],
        "success_criteria": {
            "all_movable_objects_aligned": True,
            "no_ceramic_contact": True,
            "objects_stable_on_table": True,
        },
    }
    slots = _generate_slots(plan, world_state)
    slots_by_object = {slot["assigned_object_id"]: slot for slot in slots}
    return world_state, plan, slots_by_object, skipped


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
                {"id": o["id"], "center_xy": o["position"][:2], "radius": o["radius"] + 0.08}
                for o in ceramic_obstacles
            ]
        },
    }


def _generate_slots(plan, world_state):
    table = world_state["table"]
    movable_by_id = {o["id"]: o for o in world_state["movable_objects"]}
    selected = [movable_by_id[i] for i in plan["movable_objects"] if i in movable_by_id]
    x_min, x_max = table["x_range"]
    y_min, y_max = table["y_range"]
    goal_x, goal_y, _ = world_state["goal_center"]
    spacing = 0.11
    row_y = {
        "cube": goal_y - 0.065,
        "circle": goal_y + 0.065,
    }
    grouped = {
        "cube": [obj for obj in selected if obj.get("class") == "cube"],
        "circle": [obj for obj in selected if obj.get("class") == "circle"],
    }
    slots = []
    occupied = []
    for class_name, objects in grouped.items():
        if not objects:
            continue
        start_x = goal_x - spacing * (len(objects) - 1) / 2.0
        for idx, obj in enumerate(objects):
            start_y = row_y[class_name]
            x, y = _search_safe_xy(start_x + idx * spacing, start_y, x_min, x_max, y_min, y_max, occupied, world_state, obj["radius"])
            clear = _clearance_ok(x, y, obj["radius"], occupied, world_state)
            slots.append({
                "slot_id": f"slot_{obj['id']}",
                "assigned_object_id": obj["id"],
                "target_pose": [round(x, 4), round(y, 4), round(table["z_top"] + 0.03, 4), 1.0, 0.0, 0.0, 0.0],
                "clearance_ok": clear,
            })
            occupied.append((x, y, obj["radius"]))
    return slots


def _search_safe_xy(start_x, start_y, x_min, x_max, y_min, y_max, occupied, world_state, radius):
    candidates = [(start_x, start_y)]
    for ring in range(1, 7):
        for dx in range(-ring, ring + 1):
            for dy in range(-ring, ring + 1):
                if abs(dx) == ring or abs(dy) == ring:
                    candidates.append((start_x + dx * 0.06, start_y + dy * 0.06))
    for x, y in candidates:
        x = min(max(x, x_min), x_max)
        y = min(max(y, y_min), y_max)
        if _clearance_ok(x, y, radius, occupied, world_state):
            return x, y
    return min(max(start_x, x_min), x_max), min(max(start_y, y_min), y_max)


def _clearance_ok(x, y, radius, occupied, world_state):
    bx, by = world_state["robot"]["base_xy"]
    reach = math.dist((x, y), (bx, by))
    if not (world_state["robot"]["reach_min_m"] <= reach <= world_state["robot"]["reach_max_m"]):
        return False
    for ox, oy, other_radius in occupied:
        if math.dist((x, y), (ox, oy)) < radius + other_radius + 0.06:
            return False
    for region in world_state["safety"]["inflated_ceramic_regions"]:
        if math.dist((x, y), tuple(region["center_xy"])) < region["radius"] + radius:
            return False
    return True


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


def _min_ceramic_distance_xy(obj, world_state) -> float:
    ox, oy = obj["position"][:2]
    distances = []
    for obstacle in world_state.get("ceramic_obstacles", []):
        cx, cy = obstacle["position"][:2]
        distances.append(((ox - cx) ** 2 + (oy - cy) ** 2) ** 0.5)
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
    if distance > world_state["robot"]["reach_max_m"]:
        return "HARD"
    if distance > world_state["robot"]["reach_max_m"] - 0.04:
        return "BORDERLINE"
    return "OK"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tidy-up study case using only OMPL + MuJoCo collision checking."
    )
    parser.add_argument("--object", nargs="+", default=["group", "no", "obs"], help="Scene: group no obs, ungroup no obs, group obs, ungroup obs.")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    scene_key = normalize_scene_key(args.object)
    scene_file = prepare_scene_variant(scene_key)
    event_csv_path = make_event_log_path("tidy_up", scene_key, args.log_dir)
    os.environ["MODEL_FILE"] = str(scene_file)
    os.environ["CTAMP_EVENT_LOG_CSV"] = str(event_csv_path)
    os.environ.setdefault("OMPL_ENABLED", "true")
    os.environ.setdefault("USE_IK_FALLBACK", "false")
    apply_conservative_motion_defaults()
    os.environ["ENABLE_VIEWER"] = "false" if args.no_viewer else "true"

    try:
        from ompl import base, geometric  # noqa: F401
    except ImportError:
        print("[OMPL_ONLY] OMPL Python binding belum tersedia.")
        print("[OMPL_ONLY] Install di WSL venv:")
        print("  pip install ompl")
        print("  python -c \"from ompl import base, geometric; print('ompl ok')\"")
        return 2

    world_state, plan, slots_by_object, skipped_precheck = _build_cube_tidy_plan(
        str(scene_file),
    )
    if not plan["movable_objects"]:
        print("[OMPL_ONLY] Tidak ada object movable untuk tidy up.")
        return 1

    unsafe_slots = [s for s in slots_by_object.values() if not s["clearance_ok"]]
    if unsafe_slots:
        print("[OMPL_ONLY] Slot target tidak aman dari inflated ceramic region:")
        for slot in unsafe_slots:
            print(f"  - {slot['slot_id']} {slot['target_pose'][:2]}")
        return 1

    print("[OMPL_ONLY] Task          : tidy up cubes into aligned safe slots")
    print(f"[OMPL_ONLY] Scene variant : {scene_key} ({scene_file})")
    print(f"[OMPL_ONLY] Event CSV     : {event_csv_path}")
    print(f"[OMPL_ONLY] Scene objects  : movable={world_state['counts']['movable']} ceramic={world_state['counts']['ceramic']}")
    print(f"[OMPL_ONLY] Ceramic avoid  : {plan['obstacles_to_avoid']}")
    print(f"[OMPL_ONLY] Move order     : {plan['movable_objects']}")
    if skipped_precheck:
        print("[OMPL_ONLY] Skipped before target allocation:")
        for item in skipped_precheck:
            print(f"  - {item['object_id']}: {item['failure_reason']}")
    print("[OMPL_ONLY] Target slots:")
    for object_id in plan["movable_objects"]:
        slot = slots_by_object[object_id]
        x, y, z = slot["target_pose"][:3]
        obj = next(o for o in world_state["movable_objects"] if o["id"] == object_id)
        obstacle_distance = _min_ceramic_distance_xy(obj, world_state)
        print(
            f"  - {object_id:>6} -> {slot['slot_id']:<12} ({x:.3f}, {y:.3f}, {z:.3f}) "
            f"obs={obstacle_distance:.3f}m obs_status={_obstacle_status(obj, world_state)}"
        )

    # Importing executor starts MuJoCo, creates viewer if enabled, warms the arm,
    # and wires the existing Panda OMPL planner + collision policy.
    import executor
    import feedback
    from exec_trace import flush as flush_trace
    from exec_trace import log_event

    if not getattr(executor, "_OMPL_AVAILABLE", False):
        print("[OMPL_ONLY] Executor tidak melihat OMPL planner aktif.")
        print("[OMPL_ONLY] Pastikan OMPL_ENABLED=true dan import OMPL sukses.")
        return 2

    started = time.perf_counter()
    moved = []
    failed = list(skipped_precheck)
    for item in skipped_precheck:
        log_event(
            "OBJECT_PRECHECK",
            "SKIP",
            object_id=item["object_id"],
            phase="tidy_up",
            failure_reason=item["failure_reason"],
            obstacle_distance=item.get("obstacle_distance"),
            reach_distance=item.get("reach_distance"),
        )

    for index, object_id in enumerate(plan["movable_objects"], start=1):
        slot = slots_by_object[object_id]
        x, y = slot["target_pose"][:2]
        obj = next(o for o in world_state["movable_objects"] if o["id"] == object_id)
        obstacle_distance = _min_ceramic_distance_xy(obj, world_state)
        obstacle_status = _obstacle_status(obj, world_state)
        reach_status = _reach_status(obj, world_state)
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
        print(f"\n[{index:02d}] EXEC_OBJECT START object={object_id} target=({x:.3f}, {y:.3f})")

        pick_ok = False
        pick_z = 0.0
        for pick_attempt in range(1, 4):
            print(f"[{index:02d}] PICK_ATTEMPT {pick_attempt}/3")
            executor.pick(object_id)
            pick_ok, pick_z = feedback.check_pick(
                executor.model,
                executor.data,
                executor.name_to_cube[object_id],
            )
            print(f"[{index:02d}] CHECK_PICK   {'OK' if pick_ok else 'FAIL'} attempt={pick_attempt} z={pick_z}")
            if pick_ok:
                break
            executor.drop()
        if not pick_ok:
            failed.append({"object_id": object_id, "stage": "pick", "z": pick_z, "attempts": 3})
            continue

        executor.place(float(x), float(y), object_id)
        place_ok, actual = feedback.check_place(
            executor.model,
            executor.data,
            executor.name_to_cube[object_id],
            float(x),
            float(y),
        )
        print(f"[{index:02d}] CHECK_PLACE  {'OK' if place_ok else 'FAIL'} actual={actual}")
        if place_ok:
            moved.append(object_id)
        else:
            failed.append({"object_id": object_id, "stage": "place", "actual": actual})

    duration_ms = int((time.perf_counter() - started) * 1000)
    success = len(failed) == 0
    print("\n=== OMPL-only tidy summary ===")
    print(f"success={success}")
    print(f"objects_moved={len(moved)}")
    print(f"failed={failed}")
    print(f"duration_ms={duration_ms}")
    print("collision_policy=MuJoCo contacts via PandaOMPLPlanner/CollisionPolicy")
    print("llm_used=false")
    csv_path = write_summary_csv(
        "tidy_up",
        scene_key,
        {
            "success": success,
            "objects_moved": len(moved),
            "objects_total": len(plan["movable_objects"]) + len(skipped_precheck),
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
