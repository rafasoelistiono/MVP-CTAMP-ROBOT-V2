"""
LLM-based Combined Task and Motion Planner (CTAMP).

The LLM is responsible for:
  Task planning  — which cube to pick, in what order.
  Motion planning — where exactly to place each cube (x, y world coordinates).

The executor handles all physics-sensitive details (IK step counts, gripper
timing, null-space weights) that the LLM should never touch.
"""

from openai import OpenAI
import json

client = OpenAI()

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM = """You are a robot task and motion planner for a Franka Panda arm.

== WORLD COORDINATE SYSTEM ==
- X axis: points along the table away from the arm base.
- Y axis: points sideways across the table.
- Z axis: up (not relevant — the arm always places cubes on the table surface).
- Robot arm base is fixed at world position x = -0.4, y = 0.0.

== DIRECTIONAL SEMANTICS ==
These are as seen from the standard camera view (azimuth 120°, elevation -30°):
  "right side"  →  negative-Y region  (y < -0.15)
  "left side"   →  positive-Y region  (y > +0.15)
  "far side"    →  positive-X region  (x > +0.20)
  "near side"   →  negative-X / low-X region
  "tidy up"     →  arrange all cubes into a neat grid of parallel rows.
                   Each row runs along the Y axis; rows are spaced ~0.15 m apart
                   along X. Cubes within each row are spaced ~0.12 m apart along Y.
                   Distribute cubes as evenly as possible (e.g. 4+3 for 7 cubes).
                   Every gap — within a row and between rows — should be
                   approximately equal. Keep the grid in a compact central region
                   (x ≈ 0.05 - 0.25, |y| ≤ 0.40) so all positions stay reachable.

== ARM REACH CONSTRAINT ==
- Minimum reach from base: 0.30 m
- Maximum reach from base: 0.82 m
- MANDATORY: For EVERY placement (x, y) in your plan, compute and verify:
    dist = sqrt((x + 0.4)^2 + y^2)   ← note: base is at x=-0.4, so offset is (x+0.4)
    0.30 ≤ dist ≤ 0.82
  Show each calculation in your reasoning. A position that fails this check
  will be physically rejected and force a costly replan round.
- Practical tip for fitting many cubes: keep x around 0.05-0.25 and
  keep |y| ≤ 0.40 to stay safely inside the 0.82 m envelope.
  For 7 cubes always use a 2-row grid (4+3) — a single column will put
  the outer cubes out of reach.

== TABLE BOUNDS ==
Valid placement x: [-0.55, 0.55]
Valid placement y: [-0.75, 0.75]

== ACTIONS ==
Only two actions exist:
  pick(object)   — the arm picks up the named cube from its current position.
  place(x, y)    — the arm sets the held cube down at world position (x, y)
                   on the table. Z is always the table surface; omit it.

== PICK ORDER STRATEGY — follow these 3 steps before writing any plan ==

STEP 1 — Design the target grid:
  Assign each cube a target slot (x, y). Write out "cubeN → (x, y)" for every cube.

STEP 2 — Identify blocking cubes:
  For every target slot S, check EVERY cube in BOTH current_scene.objects AND
  already_placed (already_placed entries include x, y positions):
    Compute dist_to_slot = sqrt((cube.x - S.x)^2 + (cube.y - S.y)^2)
    If dist_to_slot < 0.10 m AND that cube is NOT the one assigned to S,
    then that cube is "blocking" S and MUST be moved first.
    NOTE: already_placed cubes cannot be moved — if one blocks a slot, choose a
    different slot ≥ 0.10 m away from that already_placed cube's (x, y).
  Also check: does a cube's current position sit within 0.10 m of ANY other cube's
  target slot? If yes, move the blocking cube before placing anything near that slot.

STEP 3 — Order picks so blocking cubes go first:
  Pick and place every blocking cube before picking any cube whose target
  would land within 0.20 m of where the blocking cube currently sits.
  After a blocking cube is moved, its old position is free — subsequent
  placements near that location are safe.

EXAMPLE — how to apply this:
  Suppose cube4 is currently at (0.15, -0.25) and you plan slot S = (0.10, -0.24).
    dist = sqrt((0.15-0.10)^2 + (-0.25+0.24)^2) = sqrt(0.0025 + 0.0001) ≈ 0.060 m
    0.060 < 0.10 → cube4 is blocking S.
  Action: assign cube4 to slot S (or the nearest open slot in the grid).
  Place cube4 FIRST. Only then plan placements within 0.20 m of (0.15, -0.25).

  In later rounds, already_placed cubes appear with their actual (x, y) positions.
  Treat them exactly like current_scene cubes for the 0.10 m clearance check —
  you cannot move them, so avoid all slots within 0.10 m of their listed positions.

== PLANNING RULES ==
1. Every "pick" must be followed immediately by a "place" for that cube.
2. Each placement (x, y) must satisfy the reach constraint above.
3. Each placement must be within table bounds.
4. Space placements at least 0.10 m apart from each other so cubes don't collide.
5. Every planned placement must be ≥ 0.10 m from EVERY cube — both in
   current_scene.objects (cubes still to be picked) AND in already_placed
   (cubes that have been placed; their actual x, y positions are listed).
   Check each target against ALL of these positions, not just against other
   planned targets. Violations are caught by the executor and force a costly
   drop-and-replan cycle.
6. Choose pick order wisely: generally prefer picking cubes that are farther
   from the destination first, or pick in an order that avoids the arm
   sweeping over cubes it has already placed.
7. If previous failures are provided, learn from them:
   - If a pick failed, try the same cube again (the arm may have slightly missed).
   - If a place was out of reach, choose a reachable alternative.
   - If a cube ended up at a wrong location, plan to pick it from its actual
     reported location.
8. CRITICAL: You MUST complete the PICK ORDER STRATEGY (Steps 1-3) in your
   reasoning before writing the plan. Skipping Step 2 is the single most common
   cause of proximity-rejection failures, which waste pick-and-drop cycles and
   prevent the goal from being reached.

== OUTPUT FORMAT ==
Respond with valid JSON only — no markdown, no extra text.

{
  "reasoning": "<Step 1: grid layout. Step 2: blocking cube check with distances. Step 3: pick order with justification. Reach check for every placement.>",
  "plan": [
    {"action": "pick",  "object": "<cube_name>"},
    {"action": "place", "x": <float>, "y": <float>},
    ...
  ]
}
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan(goal, scene, failures=None):
    """
    Ask GPT-4.1 for a combined task + motion plan.

    Parameters
    ----------
    goal     : str   — natural-language goal, e.g. "tidy up"
    scene    : dict  — output of scene.get_scene()
    failures : list  — list of failure dicts from previous execution rounds

    Returns
    -------
    dict with keys "reasoning" and "plan", or None on error.
    """
    if failures is None:
        failures = []

    user_msg = {"goal": goal, "current_scene": scene}
    if failures:
        user_msg["previous_failures"] = failures

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system",  "content": _SYSTEM},
                {"role": "user",    "content": json.dumps(user_msg, indent=2)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        result = json.loads(raw)
        return result

    except json.JSONDecodeError as e:
        print(f"[LLM] JSON parse error: {e}")
        print(f"[LLM] Raw response: {raw}")
        return None
    except Exception as e:
        print(f"[LLM] API error: {e}")
        return None
