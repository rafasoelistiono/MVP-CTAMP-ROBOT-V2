#!/usr/bin/env python3
"""
Arena runner for the tidy-up (sort-by-type) task.

Goal: move every scattered object into the goal area, sorted by type -- cubes
into the cube row, circles/cylinders into the circle row. The planner only
decides which row each object belongs in and in what order (the symbolic, high-
level part); the executor owns the exact collision-free slot inside the row.

Row slots are precomputed with the same proven allocator the existing
separate_groups task uses (centered spacing along each row, shared occupancy so
cubes and circles never overlap), so every object has a stable home slot. The
arena seam (closed-loop planner + region-membership scoring) sits on top.

    python scripts/run_arena_tidy.py --object ungroup no obs --no-viewer
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
SCRIPT_DIR = ROOT_DIR / "scripts"
for _p in (SRC_DIR, SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ctamp_task_utils import (  # noqa: E402
    apply_conservative_motion_defaults,
    apply_scene_motion_defaults,
    make_event_log_path,
    normalize_scene_key,
    obstacle_mode_for_scene,
    prepare_scene_variant,
    safe_process_exit,
    write_summary_csv,
)
from align_cubes_ompl_only import (  # noqa: E402
    _min_ceramic_distance_xy,
    _obstacle_status,
    _parse_scene,
    _reach_distance_xy,
    _reach_status,
    _search_safe_target_xy,
)

# Cubes -> red (square) row below goal centre; circles -> green row above.
# Matches the visual markers in ctamp_task_utils._goal_area_body().
CUBE_ROW_OFFSET_Y = -0.065
CIRCLE_ROW_OFFSET_Y = +0.065
GOAL_X_HALF = 0.26
GOAL_Y_HALF = 0.20
SEP_MARGIN = 0.02      # dead-zone half-width between the two rows (matches evaluator)
CUBE_SPACING = 0.11
CIRCLE_SPACING = 0.105
# "Touches the strip": strip half-height (~0.025) + object half-width. Must match
# the TidyEvaluator tolerances so the allocator only targets slots the evaluator
# will credit.
CUBE_ROW_TOL = 0.061
CIRCLE_ROW_TOL = 0.051
# Keep target slots within comfortable reach of the arm. Reach thresholds are
# BORDERLINE 0.78 / HARD 0.82; placing/grasping is unreliable that far out, and
# obstacles tend to shove slots toward the workspace edge. 0.76 sits just under
# BORDERLINE and is wide enough that a full row of 4 objects still fits without
# any one being pushed past it (a tighter cap backfires -> the last slot
# overflows to the HARD edge).
COMFORT_REACH_M = 0.76


def _build_tidy_slots(world_state, goal_x, goal_y):
    """Allocate one home slot per object INSIDE the goal zone, sorted by type.

    Unlike the generic separate_groups allocator (which only respects table
    bounds), this clamps every slot to the goal area and to its type's half, so
    the executor never places where the evaluator would reject -- even when an
    obstacle pushes a slot off-centre. Objects that can't be picked (too close to
    an obstacle / out of reach) or can't get a collision-free in-zone slot are
    skipped honestly.
    """
    def _is_circle(o):
        return o.get("class") in {"circle", "cylinder"} or o["id"].lower().startswith("circle")

    def _sort_key(o):
        return (
            _obstacle_status(o, world_state) == "TOO_CLOSE",
            _obstacle_status(o, world_state) == "NEAR",
            _reach_status(o, world_state) == "HARD",
            _reach_status(o, world_state) == "BORDERLINE",
            _reach_distance_xy(o, world_state),
            o["position"][0], o["position"][1],
        )

    table_z = float(world_state["table"]["z_top"])
    cube_row_y = goal_y + CUBE_ROW_OFFSET_Y
    circle_row_y = goal_y + CIRCLE_ROW_OFFSET_Y
    cubes = sorted((o for o in world_state["movable_objects"] if o.get("class") == "cube"), key=_sort_key)
    circles = sorted((o for o in world_state["movable_objects"] if _is_circle(o)), key=_sort_key)

    occupied: list = []   # shared so cube and circle slots never overlap
    slots: dict = {}
    skipped: list = []
    base_xy = world_state["robot"]["base_xy"]

    def _max_comfortable_x(row_y):
        """Largest x on this row line within COMFORT_REACH_M of the robot base."""
        rem = COMFORT_REACH_M ** 2 - (row_y - base_xy[1]) ** 2
        return base_xy[0] + math.sqrt(rem) if rem > 0 else base_xy[0]

    def _allocate(objs, group, row_y, y_lo, y_hi, z_off, spacing):
        eligible = []
        for o in objs:
            if _obstacle_status(o, world_state) == "TOO_CLOSE":
                skipped.append({"object_id": o["id"], "group": group,
                                "failure_reason": "object_too_close_to_obstacle",
                                "obstacle_distance": round(_min_ceramic_distance_xy(o, world_state), 4)})
            elif _reach_status(o, world_state) == "HARD":
                skipped.append({"object_id": o["id"], "group": group,
                                "failure_reason": "object_outside_conservative_reach"})
            else:
                eligible.append(o)
        # Pull the row toward the robot so the FARTHEST slot stays within
        # comfortable reach (obstacles otherwise spread slots to the workspace
        # edge, where grasp/IK is unreliable for every planner). Stay centred on
        # the goal when reach allows it.
        span = spacing * max(len(eligible) - 1, 0)
        x_comfort = _max_comfortable_x(row_y)
        center = min(goal_x, x_comfort - span / 2.0)
        start_x = center - span / 2.0
        for i, o in enumerate(eligible):
            r = float(o.get("radius", 0.04))
            x_lo = goal_x - GOAL_X_HALF + r
            x_hi = goal_x + GOAL_X_HALF - r
            base_x = min(max(start_x + i * spacing, x_lo), x_hi)
            txy = _search_safe_target_xy(
                base_x, row_y, r, world_state, occupied,
                y_min=y_lo, y_max=y_hi, x_min=x_lo, x_max=min(x_hi, x_comfort),
            )
            if txy is None:  # nothing comfortable -> allow the full zone (a far slot beats skipping)
                txy = _search_safe_target_xy(
                    base_x, row_y, r, world_state, occupied,
                    y_min=y_lo, y_max=y_hi, x_min=x_lo, x_max=x_hi,
                )
            if txy is None:
                skipped.append({"object_id": o["id"], "group": group,
                                "failure_reason": "no_safe_in_zone_target"})
                continue
            x, y = txy
            occupied.append((x, y, r))
            slots[o["id"]] = {
                "group": group,
                "target_pose": [round(x, 4), round(y, 4), round(table_z + z_off, 4), 1.0, 0.0, 0.0, 0.0],
            }

    # Keep placements ON the row strip (within the evaluator's touch tolerance),
    # not anywhere in the half -- so what the executor places is exactly what the
    # evaluator credits (an object that can't fit on the strip is skipped, not
    # dumped in a second row off the line).
    _allocate(cubes, "cube", cube_row_y, cube_row_y - CUBE_ROW_TOL, cube_row_y + CUBE_ROW_TOL, 0.03, CUBE_SPACING)
    _allocate(circles, "circle", circle_row_y, circle_row_y - CIRCLE_ROW_TOL, circle_row_y + CIRCLE_ROW_TOL, 0.04, CIRCLE_SPACING)

    move_order = sorted(
        slots.keys(),
        key=lambda oid: _sort_key(next(o for o in world_state["movable_objects"] if o["id"] == oid)),
    )
    return move_order, slots, skipped

GOAL_TEXT = (
    "Tidy the table: move every object into the goal area, sorted by type. "
    "Cubes go in the cube row (region 'cube_row'); circles/cylinders go in the "
    "circle row (region 'circle_row')."
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tidy-up / sort-by-type via the arena seam (swappable planner).")
    parser.add_argument("--object", nargs="+", default=["ungroup", "no", "obs"],
                        help="Scene: group no obs, ungroup no obs, group obs, ungroup obs, group long obs, ungroup long obs.")
    parser.add_argument("--planner", default="scripted", choices=["scripted", "llm"],
                        help="Planner under test: 'scripted' baseline or 'llm' (any chat model).")
    parser.add_argument("--model", default="claude-opus-4-8",
                        help="Model id for --planner llm (e.g. claude-opus-4-8, claude-sonnet-4-6, gpt-4o, gpt-4.1).")
    parser.add_argument("--provider", default=None, choices=["anthropic", "openai"],
                        help="Override provider (default: inferred from --model).")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable adaptive thinking (Anthropic LLM planner only; cheaper/faster).")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-hint-cache", action="store_true")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--settle-after-place", type=int, default=300, help=argparse.SUPPRESS)
    args = parser.parse_args()

    scene_key = normalize_scene_key(args.object)
    scene_file = prepare_scene_variant(scene_key)
    event_csv_path = make_event_log_path("arena_tidy", scene_key, args.log_dir)
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
        print("[ARENA_TIDY] OMPL Python binding belum tersedia. `pip install ompl`")
        return 2

    world_state = _parse_scene(str(scene_file))
    goal_x, goal_y, _ = world_state["goal_center"]
    move_order, slots_by_object, skipped = _build_tidy_slots(world_state, goal_x, goal_y)
    if not move_order:
        print("[ARENA_TIDY] Tidak ada objek yang bisa dipindahkan.")
        return 1

    table_z = float(world_state["table"]["z_top"])
    cube_row_y = goal_y + CUBE_ROW_OFFSET_Y
    circle_row_y = goal_y + CIRCLE_ROW_OFFSET_Y

    # Score EVERY movable object: ones that can't be placed on a strip count as
    # misses, rather than being silently dropped from the denominator.
    all_objects = world_state["movable_objects"]
    object_ids = [o["id"] for o in all_objects]
    class_by_id = {o["id"]: ("cube" if o.get("class") == "cube" else "circle") for o in all_objects}
    target_by_id = {
        oid: (float(s["target_pose"][0]), float(s["target_pose"][1]), float(s["target_pose"][2]))
        for oid, s in slots_by_object.items()
    }

    def place_resolver(object_id, region, env):
        # Each object has a stable precomputed home slot; region is informational.
        return target_by_id.get(object_id)

    regions = {
        "cube_row": (float(goal_x), float(cube_row_y), table_z),
        "circle_row": (float(goal_x), float(circle_row_y), table_z),
    }

    by_id = {o["id"]: o for o in world_state["movable_objects"]}
    print(f"[ARENA_TIDY] Scene variant : {scene_key} ({scene_file})")
    print(f"[ARENA_TIDY] Planner       : {args.planner}")
    print(f"[ARENA_TIDY] Objects       : {len(object_ids)} "
          f"(cubes={sum(1 for c in class_by_id.values() if c == 'cube')}, "
          f"circles={sum(1 for c in class_by_id.values() if c == 'circle')})")
    print(f"[ARENA_TIDY] cube_row y={cube_row_y:.3f}  circle_row y={circle_row_y:.3f}")
    print(f"[ARENA_TIDY] Event CSV     : {event_csv_path}")
    if skipped:
        print(f"[ARENA_TIDY] Skipped (unreachable/blocked): {[s['object_id'] for s in skipped]}")
    for oid in object_ids:
        o = by_id[oid]
        slot = target_by_id.get(oid)
        slot_str = f"slot=({slot[0]:.3f}, {slot[1]:.3f})" if slot else "NO STRIP SLOT (counts as miss)"
        print(f"  - {oid:>7} [{class_by_id[oid]:>6}] reach={_reach_distance_xy(o, world_state):.3f}m "
              f"status={_reach_status(o, world_state)} obs={_obstacle_status(o, world_state)} -> {slot_str}")

    import executor  # noqa: E402
    import feedback  # noqa: E402
    from exec_trace import flush as flush_trace  # noqa: E402
    from exec_trace import log_event  # noqa: E402

    if not getattr(executor, "_OMPL_AVAILABLE", False):
        print("[ARENA_TIDY] Executor tidak melihat OMPL planner aktif.")
        return 2
    if not args.no_hint_cache:
        executor.init_hint_cache(log_dir=args.log_dir, scene_filter=scene_key)

    # Park the arm high (HOME) between placements so it retracts up instead of
    # sweeping low over already-tidied cubes/cylinders and nudging them.
    home_pose = executor.HOME.copy() if hasattr(executor, "HOME") else None
    if home_pose is not None:
        print("[ARENA_TIDY] Ready pose  : HOME (high park, clear of placed objects)")

    from arena import TaskSpec, run  # noqa: E402
    from arena.evaluators import TidyEvaluator  # noqa: E402
    from arena.mujoco_env import MujocoEnvironment  # noqa: E402
    from arena.planners import ScriptedTidyPlanner  # noqa: E402

    env = MujocoEnvironment(
        executor=executor,
        feedback=feedback,
        world_state=world_state,
        regions=regions,
        place_resolver=place_resolver,
        obstacle_status_fn=_obstacle_status,
        reach_status_fn=_reach_status,
        settle_steps=args.settle_after_place,
        ready_pose=home_pose,
        log_event=log_event,
    )
    evaluator = TidyEvaluator(
        object_ids=object_ids,
        class_by_id=class_by_id,
        goal_x=float(goal_x),
        cube_row_y=float(cube_row_y),
        circle_row_y=float(circle_row_y),
        goal_x_half=GOAL_X_HALF,
        cube_row_tol_m=CUBE_ROW_TOL,
        circle_row_tol_m=CIRCLE_ROW_TOL,
    )
    if args.planner == "llm":
        from arena.llm import LLMPlanner, make_llm_client  # noqa: E402
        client = make_llm_client(args.model, provider=args.provider, thinking=not args.no_thinking)
        planner = LLMPlanner(client, log_event=log_event)
        print(f"[ARENA_TIDY] LLM planner  : provider={args.provider or 'auto'} model={args.model}")
    else:
        planner = ScriptedTidyPlanner()
    task = TaskSpec("tidy_table", scene_key, GOAL_TEXT, max_steps=args.max_steps)

    log_event("TASK_CONTEXT", "START", phase="arena_tidy", scene=scene_key,
              planner=args.planner, model_file=str(scene_file), objects=object_ids)

    def on_step(step, action, result):
        print(f"[STEP {step:02d}] {action.describe():34s} -> "
              f"{'OK' if result.ok else 'FAIL'} {result.failure_reason or ''}")

    record = run(task, planner, env, evaluator, on_step=on_step)

    print("\n=== Arena tidy summary ===")
    print(f"planner={record.planner_name}")
    print(f"success={record.success}")
    print(f"objects_placed={record.objects_placed}/{record.objects_total}")
    print(f"score={record.score}")
    print(f"steps_used={record.steps_used}")
    print(f"duration_ms={record.duration_ms}")
    print(f"failures={record.failures}")

    csv_path = write_summary_csv(
        "arena_tidy",
        scene_key,
        {
            "success": record.success,
            "objects_moved": record.objects_placed,
            "objects_total": record.objects_total,
            "failed": record.failures,
            "duration_ms": record.duration_ms,
            "scenario_type": "static",
            "obstacle_mode": obstacle_mode_for_scene(scene_key),
        },
        args.log_dir,
    )
    print(f"csv_log={csv_path}")
    print(f"event_csv_log={event_csv_path}")
    flush_trace()
    return 0 if record.success else 1


if __name__ == "__main__":
    safe_process_exit(main())
