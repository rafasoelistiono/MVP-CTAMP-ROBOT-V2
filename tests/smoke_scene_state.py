#!/usr/bin/env python3
"""
Smoke test for scene_state.get_live_scene() against a real MuJoCo simulation.

Runs without any pick/place — just loads the sim, reads the live scene,
and validates every field against known ground truth from the XML.

Usage:
    source .venv/bin/activate
    python tests/smoke_scene_state.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Point executor at a known scene before importing it.
from ctamp_task_utils import prepare_scene_variant, apply_conservative_motion_defaults

scene_key = "group_no_obs"
scene_file = prepare_scene_variant(scene_key)
os.environ["MODEL_FILE"] = str(scene_file)
os.environ["CTAMP_EVENT_LOG_CSV"] = ""       # no log file
os.environ["CTAMP_TRACE_CONSOLE"] = "false"  # silence event trace
os.environ["ENABLE_VIEWER"] = "false"
os.environ["OMPL_ENABLED"] = "true"
os.environ.setdefault("USE_IK_FALLBACK", "false")
apply_conservative_motion_defaults()

import executor
from scene_state import get_live_scene

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

failures = []

def check(label: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}" + (f"  ({detail})" if detail else ""))
        failures.append(label)

print("\n=== smoke_scene_state: loading real MuJoCo sim ===")
print(f"  scene: {scene_key}  ({scene_file})\n")

scene = get_live_scene(executor)

# ------------------------------------------------------------------
# 1. Top-level keys
# ------------------------------------------------------------------
print("[1] Top-level schema")
for key in ("table", "robot", "goal_center", "objects", "movable_objects",
            "fallen_objects", "ceramic_obstacles", "held_object", "counts", "safety"):
    check(f"key '{key}' present", key in scene)

# ------------------------------------------------------------------
# 2. Table bounds sanity
# ------------------------------------------------------------------
print("\n[2] Table bounds")
t = scene["table"]
check("x_range is list of 2", isinstance(t["x_range"], list) and len(t["x_range"]) == 2)
check("y_range is list of 2", isinstance(t["y_range"], list) and len(t["y_range"]) == 2)
check("x_range sensible", -1.0 < t["x_range"][0] < t["x_range"][1] < 1.0)
check("y_range sensible", -1.5 < t["y_range"][0] < t["y_range"][1] < 1.5)
check("z_top near 0.80", 0.75 < t["z_top"] < 0.90,
      f"z_top={t['z_top']:.4f}")

# ------------------------------------------------------------------
# 3. Robot
# ------------------------------------------------------------------
print("\n[3] Robot")
r = scene["robot"]
check("base_xy == [-0.4, 0.0]", r["base_xy"] == [-0.4, 0.0], str(r["base_xy"]))
check("reach_min_m == 0.30", abs(r["reach_min_m"] - 0.30) < 1e-6)
check("reach_max_m == 0.82", abs(r["reach_max_m"] - 0.82) < 1e-6)
check("ee_xyz is None (executor.ee_id not set at module level)",
      r["ee_xyz"] is None or isinstance(r["ee_xyz"], list))
check("finger_width present and float",
      isinstance(r.get("finger_width"), float),
      str(r.get("finger_width")))

# ------------------------------------------------------------------
# 4. Movable objects — group_no_obs has cube1..4 and circle1..4
# ------------------------------------------------------------------
print("\n[4] Movable objects")
movable_ids = {o["id"] for o in scene["movable_objects"]}
expected_ids = {"cube1", "cube2", "cube3", "cube4",
                "circle1", "circle2", "circle3", "circle4"}
check("all 8 objects present", expected_ids == movable_ids,
      f"got {sorted(movable_ids)}")
check("no fallen objects at start", len(scene["fallen_objects"]) == 0,
      str(scene["fallen_objects"]))
check("counts.movable == 8", scene["counts"]["movable"] == 8,
      str(scene["counts"]))
check("counts.fallen == 0", scene["counts"]["fallen"] == 0)

# ------------------------------------------------------------------
# 5. Per-object fields and values
# ------------------------------------------------------------------
print("\n[5] Per-object fields")
for obj in scene["movable_objects"]:
    oid = obj["id"]
    check(f"{oid}: has all required fields",
          {"id","class","position","radius","movable","fragile",
           "held","fallen","reach_dist_m","reach_status",
           "obstacle_dist_m","obstacle_status"}.issubset(obj.keys()))
    check(f"{oid}: position is [x,y,z]",
          isinstance(obj["position"], list) and len(obj["position"]) == 3)
    check(f"{oid}: z on table (~0.83)",
          0.78 < obj["position"][2] < 0.90,
          f"z={obj['position'][2]:.4f}")
    check(f"{oid}: radius > 0", obj["radius"] > 0)
    check(f"{oid}: movable=True", obj["movable"] is True)
    check(f"{oid}: held=False at start", obj["held"] is False)
    check(f"{oid}: fallen=False at start", obj["fallen"] is False)
    reach = math.dist(obj["position"][:2], [-0.4, 0.0])
    check(f"{oid}: reach_dist_m matches position",
          abs(obj["reach_dist_m"] - reach) < 0.01,
          f"stored={obj['reach_dist_m']:.4f} computed={reach:.4f}")
    check(f"{oid}: reach_status valid",
          obj["reach_status"] in ("OK", "BORDERLINE", "HARD"),
          obj["reach_status"])
    check(f"{oid}: obstacle_status valid",
          obj["obstacle_status"] in ("CLEAR", "NEAR", "TOO_CLOSE"),
          obj["obstacle_status"])

# ------------------------------------------------------------------
# 6. No obstacles in group_no_obs
# ------------------------------------------------------------------
print("\n[6] Obstacles (group_no_obs — none expected)")
check("ceramic_obstacles empty for no_obs scene",
      len(scene["ceramic_obstacles"]) == 0,
      str([o["id"] for o in scene["ceramic_obstacles"]]))
check("counts.ceramic == 0", scene["counts"]["ceramic"] == 0)
check("safety regions empty", len(scene["safety"]["inflated_ceramic_regions"]) == 0)

# ------------------------------------------------------------------
# 7. held_object is None at start (gripper open, nothing lifted)
# ------------------------------------------------------------------
print("\n[7] Held object")
check("held_object is None at start", scene["held_object"] is None,
      str(scene["held_object"]))

# ------------------------------------------------------------------
# 8. goal_center plausible
# ------------------------------------------------------------------
print("\n[8] Goal center")
gc = scene["goal_center"]
check("goal_center is list of 3", isinstance(gc, list) and len(gc) == 3)
check("goal_center x in table range", t["x_range"][0] < gc[0] < t["x_range"][1],
      f"x={gc[0]}")
check("goal_center y in table range", t["y_range"][0] < gc[1] < t["y_range"][1],
      f"y={gc[1]}")

# ------------------------------------------------------------------
# 9. Quick repeat-call consistency
# ------------------------------------------------------------------
print("\n[9] Repeat-call consistency")
scene2 = get_live_scene(executor)
for obj in scene["movable_objects"]:
    obj2 = next(o for o in scene2["movable_objects"] if o["id"] == obj["id"])
    check(f"{obj['id']}: position stable across calls",
          obj["position"] == obj2["position"],
          f"{obj['position']} vs {obj2['position']}")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print(f"\n{'='*52}")
if failures:
    print(f"RESULT: {len(failures)} FAILED")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    total = sum(1 for line in open(__file__) if "check(" in line)
    print(f"RESULT: ALL CHECKS PASSED")
    sys.exit(0)
