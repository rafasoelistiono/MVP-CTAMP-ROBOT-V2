"""
LLM-based CTAMP entry point.

Usage:
    python main.py                              # default goal: "tidy up"
    python main.py "move all cubes to the right side"
    python main.py "place all cubes to the left side"
"""

import sys
import math
import mujoco
from executor import model, data, name_to_cube, pick, place, drop
import scene    as sc
import feedback as fb
from llm_planner import plan

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GOAL        = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "tidy up"
MAX_REPLAN  = 10  # pre-validation makes retries cheap — allow more rounds

# ---------------------------------------------------------------------------
# Pre-validation
# ---------------------------------------------------------------------------

def pre_validate_plan(steps, remaining, placed):
    """
    Simulate the full plan against current cube positions and return every
    proximity/reachability violation found — without running any robot action.

    Simulates plan progression correctly:
      - When a cube is picked it is removed from the obstacle set.
      - When a cube is placed it is added at its new position as a fixed obstacle.
      - already_placed cubes are permanent obstacles (cannot be picked).

    Returns a list of failure dicts (empty list = plan is valid).
    """
    obs   = {}  # name → (x, y)  — full set of current obstacles
    fixed = set()  # names of cubes that cannot be moved (already placed)

    for o in remaining:
        obs[o["name"]] = (o["x"], o["y"])
    for o in placed:
        obs[o["name"]] = (o["x"], o["y"])
        fixed.add(o["name"])

    violations = []
    held = None

    for step in steps:
        action = step.get("action")

        if action == "pick":
            held = step.get("object", "")
            obs.pop(held, None)   # picked cube is no longer a table obstacle

        elif action == "place" and held:
            x = float(step.get("x", 0))
            y = float(step.get("y", 0))

            if not sc.is_reachable(x, y):
                dist = math.sqrt((x + 0.4)**2 + y**2)
                violations.append({
                    "action":    "place",
                    "object":    held,
                    "reason":    (f"({x}, {y}) dist={dist:.3f} m from arm base; "
                                  f"must be {sc.ARM_MIN_REACH}–{sc.ARM_MAX_REACH} m."),
                    "requested": [x, y],
                })
            else:
                for cname, (cx, cy) in obs.items():
                    d = math.sqrt((x - cx)**2 + (y - cy)**2)
                    if d < sc.CUBE_CLEARANCE:
                        if cname in fixed:
                            note = (f"{cname} is already placed (immovable). "
                                    f"Choose a slot ≥ {sc.CUBE_CLEARANCE} m from "
                                    f"its position ({cx:.3f}, {cy:.3f}).")
                        else:
                            note = (f"{cname} has not been picked yet. "
                                    f"Either pick {cname} before placing here, "
                                    f"or choose a slot ≥ {sc.CUBE_CLEARANCE} m from "
                                    f"its current position ({cx:.3f}, {cy:.3f}).")
                        violations.append({
                            "action":    "place",
                            "object":    held,
                            "reason":    (f"({x:.3f}, {y:.3f}) is {d:.3f} m from "
                                          f"{cname} at ({cx:.3f}, {cy:.3f}); "
                                          f"must be ≥ {sc.CUBE_CLEARANCE} m. {note}"),
                            "requested": [x, y],
                        })
                        break

            # Add placed position as fixed obstacle for subsequent checks,
            # even if the placement was invalid (to catch follow-on violations).
            obs[held] = (x, y)
            fixed.add(held)
            held = None

    return violations

# ---------------------------------------------------------------------------
# Closed-loop CTAMP execution
# ---------------------------------------------------------------------------

print(f"\n=== CTAMP  |  goal: \"{GOAL}\" ===\n")

completed = set()   # names of cubes successfully placed
failures  = []      # accumulated failure reports fed back to LLM

for replan_round in range(MAX_REPLAN):

    # -- Observe current scene ------------------------------------------------
    scene_state = sc.get_scene(model, data, name_to_cube)
    all_objects = scene_state["objects"]
    remaining   = [o for o in all_objects if o["name"] not in completed]
    placed      = [o for o in all_objects if o["name"] in completed]

    if not remaining:
        print("Goal achieved — all cubes placed.")
        break

    # Give the LLM the actual (x, y) positions of already-placed cubes so it
    # can treat them as fixed obstacles when checking the 0.10 m clearance rule.
    scene_state["objects"]        = remaining
    scene_state["already_placed"] = placed

    print(f"--- Round {replan_round + 1}  |  {len(remaining)} cube(s) remaining ---")
    for o in remaining:
        print(f"    {o['name']} ({o['color']})  x={o['x']}  y={o['y']}  "
              f"dist={o['distance_from_arm_base_m']} m")

    # -- Ask LLM for task+motion plan -----------------------------------------
    result = plan(GOAL, scene_state, failures)

    if result is None:
        print("[main] LLM planning failed — aborting.")
        break

    print(f"[LLM] {result.get('reasoning', '(no reasoning)')}")

    steps = result.get("plan", [])

    # -- Pre-validate entire plan before executing anything -------------------
    # Catches all proximity/reachability violations without burning pick-drop
    # cycles.  Each bad round costs only an LLM call, not simulation time.
    pre_violations = pre_validate_plan(steps, remaining, placed)
    if pre_violations:
        print(f"[main] Plan pre-validation: {len(pre_violations)} violation(s) "
              f"— skipping execution, replanning")
        for v in pre_violations:
            req = v.get("requested", "?")
            print(f"  [!] {v['object']} @ {req}: {v['reason']}")
        failures.extend(pre_violations)
        continue   # straight to next replan round — no robot action taken

    # -- Execute plan step by step --------------------------------------------
    held_cube  = None   # which cube the arm is currently carrying
    plan_broke = False

    i = 0
    while i < len(steps):
        step   = steps[i]
        action = step.get("action")

        # ---- PICK -----------------------------------------------------------
        if action == "pick":
            obj = step.get("object", "")

            if obj in completed:
                print(f"[main] skip pick({obj}) — already placed")
                i += 1
                continue

            if obj not in name_to_cube:
                print(f"[main] skip pick({obj}) — unknown object")
                i += 1
                continue

            pick(obj)

            ok, z = fb.check_pick(model, data, name_to_cube[obj])
            if ok:
                held_cube = obj
                print(f"[main] pick({obj}) OK  (z={z})")
            else:
                print(f"[main] pick({obj}) FAILED  (z={z} <= {fb.HELD_Z_THRESHOLD})")
                failures.append({
                    "action": "pick",
                    "object": obj,
                    "reason": (f"cube z={z} after lift; expected > "
                               f"{fb.HELD_Z_THRESHOLD}. "
                               "Arm may have missed — retry or try a different cube first."),
                })
                print(f"[main] opening gripper after failed pick")
                drop()
                held_cube  = None
                plan_broke = True
                break

        # ---- PLACE ----------------------------------------------------------
        elif action == "place":
            if held_cube is None:
                print("[main] skip place — nothing held")
                i += 1
                continue

            x = float(step.get("x", 0))
            y = float(step.get("y", 0))

            # Safety net: reachability (pre-validation should have caught this)
            if not sc.is_reachable(x, y):
                dist = math.sqrt((x - (-0.4))**2 + y**2)
                print(f"[main] place({x:.2f}, {y:.2f}) REJECTED — "
                      f"dist={dist:.3f} m out of reach")
                failures.append({
                    "action":    "place",
                    "object":    held_cube,
                    "reason":    (f"({x}, {y}) dist={dist:.3f} m from arm base; "
                                  f"must be {sc.ARM_MIN_REACH}–{sc.ARM_MAX_REACH} m."),
                    "requested": [x, y],
                })
                print(f"[main] dropping {held_cube} before replan")
                drop()
                held_cube  = None
                plan_broke = True
                break

            # Safety net: proximity (pre-validation should have caught this;
            # re-check with live physics positions in case a cube shifted)
            mujoco.mj_forward(model, data)
            blocker = None
            for cname, cid in name_to_cube.items():
                if cname == held_cube:
                    continue
                cpos = data.xpos[cid]
                if float(cpos[2]) < fb.HELD_Z_THRESHOLD:
                    d = math.sqrt((x - float(cpos[0]))**2 + (y - float(cpos[1]))**2)
                    if d < sc.CUBE_CLEARANCE:
                        blocker = cname
                        break

            if blocker is not None:
                print(f"[main] place({x:.2f}, {y:.2f}) REJECTED — "
                      f"within {sc.CUBE_CLEARANCE} m of {blocker}")
                failures.append({
                    "action":    "place",
                    "object":    held_cube,
                    "reason":    (f"({x}, {y}) is within {sc.CUBE_CLEARANCE} m of "
                                  f"{blocker}'s current position ({sc.CUBE_CLEARANCE} m "
                                  "clearance required from ALL cubes)."),
                    "requested": [x, y],
                })
                print(f"[main] dropping {held_cube} before replan")
                drop()
                held_cube  = None
                plan_broke = True
                break

            place(x, y)

            ok, actual = fb.check_place(model, data, name_to_cube[held_cube], x, y)
            if ok:
                print(f"[main] place({x:.2f}, {y:.2f}) OK  "
                      f"— {held_cube} now at {actual}")
                completed.add(held_cube)
            else:
                print(f"[main] place({x:.2f}, {y:.2f}) FAILED  "
                      f"— {held_cube} ended at {actual}")
                failures.append({
                    "action":    "place",
                    "object":    held_cube,
                    "reason":    (f"{held_cube} ended at {actual} instead of "
                                  f"({x}, {y}). Its new position is in "
                                  "the next scene state — plan from there."),
                    "requested": [x, y],
                    "actual":    actual,
                })
                plan_broke = True

            held_cube = None

            if plan_broke:
                break

        # ---- UNKNOWN --------------------------------------------------------
        else:
            print(f"[main] unknown action '{action}' — skipping")

        i += 1

# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

all_cubes = set(name_to_cube.keys())
print(f"\n=== Final report ===")
print(f"Placed:     {sorted(completed)}")
print(f"Not placed: {sorted(all_cubes - completed)}")
