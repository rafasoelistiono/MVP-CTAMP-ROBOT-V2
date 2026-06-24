#!/usr/bin/env python3
"""
Arena runner for the cube-stacking task.

This is the *thin harness* version of stack_cubes: it owns no planning logic.
It sets up the scene + executor, then hands control to a swappable Planner
through the arena seam (Planner -> Environment -> Evaluator). Today it runs the
scripted baseline; swapping `--planner llm` later changes only which Planner is
constructed, nothing else.

    python scripts/run_arena_stack.py --object group no obs --no-viewer
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
SCRIPT_DIR = ROOT_DIR / "scripts"
for _p in (SRC_DIR, SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import os  # noqa: E402

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

REGION = "tower_base"


def _goal_text(order: list[str]) -> str:
    chain = ", then ".join(f"{order[i]} on {order[i - 1]}" for i in range(1, len(order)))
    return (
        f"Build one vertical tower of {len(order)} cubes in the goal area: "
        f"{order[0]} on the table as the base, then {chain}."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Cube stacking via the arena seam (swappable planner).")
    parser.add_argument("--object", nargs="+", default=["group", "no", "obs"],
                        help="Scene: group no obs, ungroup no obs, group obs, ungroup obs, group long obs, ungroup long obs.")
    parser.add_argument("--planner", default="scripted", choices=["scripted"],
                        help="Planner under test (llm planner lands in Phase 2).")
    parser.add_argument("--height", type=int, default=3, choices=[2, 3, 4],
                        help="Tower height. 3 is reliable; 4 is hard mode (the top cube is tippy).")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-hint-cache", action="store_true")
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--settle-after-place", type=int, default=360, help=argparse.SUPPRESS)
    args = parser.parse_args()

    stack_order = [f"cube{i}" for i in range(1, args.height + 1)]
    goal_text = _goal_text(stack_order)

    scene_key = normalize_scene_key(args.object)
    scene_file = prepare_scene_variant(scene_key)
    event_csv_path = make_event_log_path("arena_stack", scene_key, args.log_dir)
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
        print("[ARENA_STACK] OMPL Python binding belum tersedia. `pip install ompl`")
        return 2

    world_state = _parse_scene(str(scene_file))
    goal = world_state["goal_center"]
    table_z = float(world_state["table"]["z_top"])
    base_xy = _search_safe_target_xy(goal[0], goal[1], 0.051, world_state, []) or (goal[0], goal[1])
    base_x, base_y = float(base_xy[0]), float(base_xy[1])
    regions = {REGION: (base_x, base_y, table_z)}

    def stack_place_resolver(object_id, region, env):
        # The tower base is a fixed point at the goal; upper cubes use `stack`.
        return (base_x, base_y, table_z + 0.03)

    print(f"[ARENA_STACK] Scene variant : {scene_key} ({scene_file})")
    print(f"[ARENA_STACK] Planner       : {args.planner}")
    print(f"[ARENA_STACK] Tower height  : {args.height} ({' -> '.join(stack_order)})")
    print(f"[ARENA_STACK] Tower base    : ({base_x:.3f}, {base_y:.3f})")
    print(f"[ARENA_STACK] Event CSV     : {event_csv_path}")
    for object_id in stack_order:
        obj = next((o for o in world_state["movable_objects"] if o["id"] == object_id), None)
        if obj is None:
            print(f"  - {object_id:>6}: NOT IN SCENE")
            continue
        print(f"  - {object_id:>6}: reach={_reach_distance_xy(obj, world_state):.3f}m "
              f"status={_reach_status(obj, world_state)} "
              f"clearance={_min_ceramic_distance_xy(obj, world_state):.3f}m "
              f"obs={_obstacle_status(obj, world_state)}")

    import executor  # noqa: E402  (must follow env-var setup)
    import feedback  # noqa: E402
    from exec_trace import flush as flush_trace  # noqa: E402
    from exec_trace import log_event  # noqa: E402

    if not getattr(executor, "_OMPL_AVAILABLE", False):
        print("[ARENA_STACK] Executor tidak melihat OMPL planner aktif.")
        return 2
    if not args.no_hint_cache:
        executor.init_hint_cache(log_dir=args.log_dir, scene_filter=scene_key)

    # Park the arm high (HOME) between picks; the env applies this on reset so the
    # hand retracts up instead of sweeping low over the tower / placed objects.
    home_pose = executor.HOME.copy() if hasattr(executor, "HOME") else None
    if home_pose is not None:
        print("[ARENA_STACK] Ready pose  : HOME (high park, clear of tower)")

    from arena import TaskSpec, run  # noqa: E402
    from arena.evaluators import StackEvaluator  # noqa: E402
    from arena.mujoco_env import MujocoEnvironment  # noqa: E402
    from arena.planners import ScriptedStackPlanner  # noqa: E402

    env = MujocoEnvironment(
        executor=executor,
        feedback=feedback,
        world_state=world_state,
        regions=regions,
        place_resolver=stack_place_resolver,
        obstacle_status_fn=_obstacle_status,
        reach_status_fn=_reach_status,
        settle_steps=args.settle_after_place,
        ready_pose=home_pose,
        log_event=log_event,
    )
    evaluator = StackEvaluator(stack_order, REGION)
    planner = ScriptedStackPlanner(stack_order, region=REGION)
    task = TaskSpec("stack_cubes", scene_key, goal_text, max_steps=args.max_steps)

    log_event("TASK_CONTEXT", "START", phase="arena_stack", scene=scene_key,
              planner=args.planner, model_file=str(scene_file), order=stack_order)

    def on_step(step, action, result):
        print(f"[STEP {step:02d}] {action.describe():32s} -> "
              f"{'OK' if result.ok else 'FAIL'} {result.failure_reason or ''}")

    record = run(task, planner, env, evaluator, on_step=on_step)

    print("\n=== Arena stack summary ===")
    print(f"planner={record.planner_name}")
    print(f"success={record.success}")
    print(f"objects_placed={record.objects_placed}/{record.objects_total}")
    print(f"score={record.score}")
    print(f"steps_used={record.steps_used}")
    print(f"duration_ms={record.duration_ms}")
    print(f"failures={record.failures}")

    csv_path = write_summary_csv(
        "arena_stack",
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
