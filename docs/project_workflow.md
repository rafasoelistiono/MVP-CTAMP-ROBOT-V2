# MVP-CTAMP-ROBOT-V2 — Full Project Workflow
# Purpose: Use this document to generate a detailed workflow chart of the entire project.

---

## WHAT THIS PROJECT IS

A benchmarking arena that evaluates multiple LLM/AI models on their ability to plan
robotic tabletop tasks (stacking cubes, sorting objects). The LLM does high-level
symbolic planning (decide WHAT to do); a real robot arm simulator (MuJoCo + OMPL + IK)
does low-level execution (decide HOW to do it physically). The system runs closed-loop:
after every action the LLM sees the updated physical scene and decides the next move.

---

## TOP-LEVEL COMPONENTS (boxes in the chart)

| Component | File | Role |
|---|---|---|
| Runner Script | scripts/run_arena_stack.py OR run_arena_tidy.py | Entry point. Wires everything together. |
| Harness | src/arena/harness.py | The closed-loop controller. Drives every episode. |
| LLM Planner | src/arena/llm.py | Calls the AI model API to decide next action. |
| MuJoCo Environment | src/arena/mujoco_env.py | Translates symbolic actions into real robot moves. |
| Executor | src/executor.py | Low-level arm control: OMPL path planning + IK solving. |
| Collision Policy | src/collision_policy.py | Tells OMPL whether a joint config is collision-free. |
| Evaluator | src/arena/evaluators.py | Judges whether the goal is achieved. Scores the run. |
| HintCache | src/hint_cache.py | Reads past run logs to adaptively tune IK behavior. |
| Contracts | src/arena/contracts.py | Shared data types: Action, Observation, WorldSnapshot, RunRecord. |

---

## PHASE 0 — STARTUP (runs once before the episode)

### Node: Parse CLI Arguments
- Inputs: command-line flags
- Key flags: --planner (scripted or llm), --model (e.g. gpt-4o, claude-opus-4-8),
  --object (scene variant: group/ungroup, obs/no_obs), --no-viewer, --height (2/3/4)
- Output: parsed args object

### Node: Set Environment Variables
- Sets MODEL_FILE (path to MuJoCo XML scene file)
- Sets CTAMP_EVENT_LOG_CSV (path for event log)
- Sets OMPL_ENABLED=true
- Sets CTAMP_OBSTACLE_MODE (none / cautious / long)
- MUST happen BEFORE importing executor (executor reads these at import time)

### Node: Import Executor
- executor.py is imported AFTER env vars are set
- On import: loads MuJoCo model from MODEL_FILE, initializes OMPL planner,
  initializes IK backend (Pinocchio or MuJoCo DLS), sets joint limits

### Node: Initialize HintCache (optional)
- Reads all past *_events.csv log files for this scene
- Computes 3 adaptive hints from past failure patterns:
  (1) preferred_backend: skip Pinocchio if failure rate >= 70%
  (2) pos_err_tolerance: widen IK threshold for hard-reach zones
  (3) preferred_grasp_profile: best grip width per object class
- If not enough data (< 5 samples), hints stay inactive

### Node: Parse Scene XML
- Reads the MuJoCo XML file directly to extract:
  - All movable objects: id, class (cube/circle), position, radius
  - Table surface Z height
  - Goal center coordinates
  - Ceramic/vase/glass obstacles and their positions
- Output: world_state dict

### Node: Compute Regions
- Calculates named target locations (e.g. "tower_base", "cube_row", "circle_row")
- For stacking: finds a safe XY point at the goal center
- For tidy: builds per-type slots clamped to goal zone boundaries
- Output: regions dict { region_name: (x, y, z) }

### Node: Construct place_resolver
- A closure (function) that converts (object_id, region_name) → (x, y, z) coordinates
- For stacking: always returns the fixed tower_base point
- For tidy: computes collision-free slot within the correct row using live object positions
- The LLM never sees coordinates — only this resolver knows them

### Node: Construct Three Seam Components
COMPONENT A — MujocoEnvironment
  - Wraps executor + feedback + world_state + regions + place_resolver
  - Sets ready_pose = executor.HOME (high park pose to avoid arm sweeping over placed objects)

COMPONENT B — Evaluator (StackEvaluator or TidyEvaluator)
  - StackEvaluator: knows the required tower order [cube1, cube2, cube3]
  - TidyEvaluator: knows which objects are cubes vs circles, and the goal zone bounds

COMPONENT C — Planner
  - If --planner scripted: ScriptedStackPlanner or ScriptedTidyPlanner (hard-coded logic)
  - If --planner llm:
      make_llm_client(model, provider) →
        if model starts with "claude": AnthropicClient (uses anthropic SDK, adaptive thinking ON)
        if model starts with "gpt":    OpenAIClient   (uses openai SDK, JSON mode ON)
      LLMPlanner(client) wraps the client in retry/validation logic

### Node: Create TaskSpec
- task_id: "stack_cubes" or "tidy_table"
- scene_key: e.g. "group_no_obs"
- goal_text: natural-language goal string (e.g. "Build a tower: cube1 as base, then cube2, cube3")
- max_steps: 24

### Node: Call harness.run(task, planner, env, evaluator)
- This is where the closed loop starts

---

## PHASE 1 — EPISODE INITIALIZATION (inside harness.run, runs once)

### Node: env.reset()
- Sets executor.REST_POSE = HOME (arm will park high between moves)
- Clears per-object pick attempt counters

### Node: env.snapshot() — first snapshot
- Calls mujoco.mj_forward() to sync physics state
- For each movable object: reads live XYZ position from MuJoCo data.xpos[]
- Computes reach status per object:
    reach distance < 0.65m  → "OK"
    0.65m to 0.80m          → "BORDERLINE"
    > 0.80m                 → "HARD"
- Computes obstacle proximity per object:
    nearest ceramic > 0.15m → "CLEAR"
    0.08m to 0.15m          → "NEAR"
    < 0.08m                 → "TOO_CLOSE"
- Output: WorldSnapshot (objects, obstacles, regions, held_object)

### Node: evaluator.placed_status(snapshot) — initial check
- Checks each object's position against goal geometry
- At start: all objects return placed=False

### Node: build_observation(task, snapshot, placed_status, step=0, history=[])
- Folds placed_status flags into each ObjectView
- Builds the Observation the planner will receive:
    - goal (natural language)
    - step number and max_steps budget
    - all objects with id, class, position, reach, obstacle, on_table, held, placed
    - all obstacles with id and position
    - regions dict
    - action vocabulary (the 3 allowed actions)
    - history (empty at step 0)
- This is the ONLY place Observations are built — every planner gets identical format

### Node: planner.reset(task, obs)
- For LLMPlanner: no-op (stateless)
- For ScriptedPlanner: resets internal step counter

---

## PHASE 2 — CLOSED LOOP (repeats up to max_steps=24 times)

### Decision: Is goal satisfied?
- evaluator.is_goal_satisfied(snapshot)
- For StackEvaluator: all cubes in the order are placed=True (tower is complete)
- For TidyEvaluator: all objects are in their correct zone
- YES → EXIT LOOP → go to Phase 3
- NO  → continue to next step

---

### Node: planner.act(obs) — THE DECISION POINT

#### Branch A: ScriptedPlanner
- Reads hard-coded logic (e.g. "place cube1 first, then stack cube2 on cube1")
- Returns Action immediately with no API call

#### Branch B: LLMPlanner (main focus)

##### Sub-Node: Build user message
- Serializes the entire Observation to JSON (obs.to_dict())
- Wraps it: "Goal: ...\nObservation (JSON): {...}\nWhat is your single next action?"

##### Sub-Node: Retry loop (up to 3 attempts)

  ATTEMPT 1 (and 2, 3 if needed):

  Sub-Node: client.complete(system_prompt, user_message)
    - AnthropicClient path:
        Calls anthropic SDK: client.messages.create(
          model=model,
          max_tokens=4096,
          system=SYSTEM_PROMPT,
          thinking={"type": "adaptive"},   ← extended reasoning ON
          messages=[{"role": "user", "content": user_message}]
        )
        Extracts text blocks from response (ignores thinking blocks)
        Returns raw text string

    - OpenAIClient path:
        Calls openai SDK: client.chat.completions.create(
          model=model,
          max_tokens=2048,
          messages=[{system}, {user}],
          response_format={"type": "json_object"}
        )
        Returns response.choices[0].message.content

  Decision: Did the API call succeed?
    - NO (network error, refusal, timeout):
        Append correction: "The previous attempt errored. Reply with ONLY a valid JSON action."
        Retry
    - YES: continue

  Sub-Node: extract_json_object(reply)
    - Scans the text for the first "{"
    - Walks character-by-character tracking brace depth and string escapes
    - Returns the first balanced {...} JSON block
    - Tolerates: code fences, markdown, prose around the JSON
    Decision: Found valid JSON?
      - NO: raise ValueError → correction appended → retry
      - YES: continue

  Sub-Node: parse_action(json_dict)
    - Reads "action" field (stack / place / stop)
    - For "stack": requires "object" and "on" fields
    - For "place": requires "object" and "region" fields
    - For "stop": optional "reason" field
    Decision: Valid action type and required fields?
      - NO: raise ActionParseError → correction appended → retry
      - YES: continue

  Sub-Node: _parse_and_validate(action, obs) — ground check
    - Checks action.object_id is in obs.objects_by_id (LLM can't hallucinate objects)
    - For stack: checks action.on is also in obs.objects_by_id
    - For place: checks action.region is in obs.regions
    Decision: All IDs/regions exist in the live observation?
      - NO: raise ActionParseError → correction appended → retry
      - YES: return Action

  Decision: All 3 attempts failed?
    - YES: return Action.stop(reason="llm_no_valid_action:...")  ← safe fallback, never crashes
    - NO:  return the valid Action

---

### Decision: Is the action STOP?
- YES → append to history, EXIT LOOP → go to Phase 3
- NO  → continue to execution

---

### Node: env.execute(action) → MujocoEnvironment

#### Branch: action.type == PLACE
  - Calls place_resolver(object_id, region, env)
  - Resolver returns (x, y, z) target coordinates
  - Decision: Did resolver return a valid slot?
      - NO: return StepResult(ok=False, failure_reason="no_safe_slot_in_region")
      - YES: call _transport(action, object_id, x, y, z, release_lift=0.050)

#### Branch: action.type == STACK
  - Reads live position of the base object from MuJoCo: base_xyz = data.xpos[base_id]
  - Computes target_z = base_xyz[2] + 0.06  (one cube height above base)
  - Calls _transport(action, object_id, base_x, base_y, target_z, release_lift=0.008)

#### Node: _transport(action, object_id, x, y, z)

  Sub-Node: Precheck
    Decision: Is obstacle_status == "TOO_CLOSE"?
      - YES: return StepResult(ok=False, stage="precheck", failure_reason="object_too_close_to_obstacle")
    Decision: Is reach_status == "HARD"?
      - YES: return StepResult(ok=False, stage="precheck", failure_reason="object_outside_conservative_reach")

  Sub-Node: executor.pick(object_id)
    (Details in PHASE 2A below)
    Decision: Did pick raise RuntimeError?
      - YES: settle physics, return StepResult(ok=False, stage="pick", failure_reason="executor_runtime_error")

  Sub-Node: feedback.check_pick(model, data, cube_body_id)
    - Checks object Z height is above table (object was lifted)
    - Checks executor._held_object_name == object_id (gripper is holding it)
    Decision: pick_lifted=True AND held=True?
      - NO: executor.drop(), settle, return StepResult(ok=False, stage="pick", failure_reason="object_not_lifted_after_pick" or "object_not_held_after_pick")
      - YES: continue

  Sub-Node: executor.place(x, y, object_id, target_z=z, release_lift=...)
    (Details in PHASE 2B below)
    Decision: Did place raise RuntimeError?
      - YES: settle physics, return StepResult(ok=False, stage="place", failure_reason="executor_runtime_error")

  Sub-Node: _settle() — physics simulation
    - Runs 360 MuJoCo time steps (mj_step) so object settles under gravity

  Sub-Node: Verify final position
    - Reads actual object XYZ from MuJoCo after settling
    - xy_error = distance(actual[:2], target[:2])
    - z_error  = |actual[2] - target[2]|
    Decision: xy_error <= 0.045m AND z_error <= 0.035m?
      - YES: return StepResult(ok=True, stage="stack"/"place")
      - NO:  return StepResult(ok=False, failure_reason="object_not_at_target_after_place")

---

## PHASE 2A — executor.pick(object_id) — Low-level pick pipeline

### Node: Compute grasp pose
- Reads object's current XYZ from MuJoCo data
- Selects grasp profile (grip width, offset, clearance bonus) based on:
    - HintCache preferred_grasp_profile hint (if active)
    - Object class (cube vs cylinder) and reach distance
- Applies FAR_PICK_XY_DISTANCE boost if object is beyond 0.74m

### Node: Set ignored bodies for collision policy
- collision_policy.set_ignored_bodies([object_id])
- Tells the collision checker: "it's OK to touch this object"
- Rebuild geom-to-team sets (refresh())

### Node: Plan path to pre-grasp hover (OMPL)
- Uses RRTConnect algorithm to find a collision-free joint-space path
- Start: current arm joint angles
- Goal: joint angles that position end-effector at hover point above object
- At each sampled configuration:
    Sub-Node: collision_policy.check_contacts(data)
      - For each active contact pair in MuJoCo:
          Classify: robot geom vs environment geom?
          If YES:
            - Is environment body in ignored list? → ALLOW (skip)
            - Is environment body a movable object?
                Finger + penetration <= 0.018m? → ALLOW
                Otherwise?                      → BLOCK (invalid config)
            - Is environment body the table?
                Finger + penetration <= 0.005m? → ALLOW
                Otherwise?                      → BLOCK
            - Is environment body an obstacle?
                Penetration <= 0.003m?          → ALLOW
                Otherwise?                      → BLOCK
            - None matched?                     → BLOCK
      - All contacts passed?                    → VALID config (OMPL accepts it)
- Output: sequence of joint configurations (path)

### Node: Execute path to pre-grasp hover
- Interpolates along the OMPL path, steps MuJoCo simulation
- Arm moves to hover position above the object

### Node: IK solve for grasp pose
Decision: Which IK backend?
  - Check HintCache.preferred_backend() — forces mujoco_dls if Pinocchio fails >= 70%
  - IK_BACKEND=pinocchio or auto → try Pinocchio first

  Branch A: Pinocchio IK
    - Loads Franka Panda URDF model (separate from MuJoCo)
    - Converts target XYZ from MuJoCo world frame to Panda base frame
    - Runs Jacobian pseudo-inverse with Levenberg-Marquardt damping (damping=1e-4)
    - Iterates up to 120 steps until position error < 0.006m
    - FK validation: plugs result back into MuJoCo, checks position error
    Decision: FK validation passed (error <= pos_limit)?
      - YES: use Pinocchio result
      - NO:  fall through to MuJoCo DLS

  Branch B: MuJoCo DLS (Damped Least Squares)
    - Runs entirely inside MuJoCo's physics model
    - Iterates up to 600 steps:
        mj_forward() to compute Jacobian
        J_pinv = Jᵀ(JJᵀ + 0.05·I)⁻¹   (damped pseudo-inverse)
        dq_task = J_pinv · error         (move toward target)
        dq_null = null_proj · (GRASP_READY - q) · 0.05  (pull toward preferred pose)
        q += clip(dq_task + dq_null, -0.08, 0.08) · 0.015
    - Until position error < 0.008m AND orientation error < 0.20 rad
    - Output: joint angles for grasp pose

### Node: Move to grasp pose and close gripper
- Execute IK result: move arm to exact grasp position
- Close gripper fingers (target width narrows to grip sequence: 0.015 → 0.011 → 0.008 m)
- Set executor._held_object_name = object_id

### Node: Lift object to clearance height
- Plan and execute upward motion to safe clearance above other objects

### Node: Return arm to REST_POSE (HOME)
- Parks arm at high safe configuration
- Prevents arm from sweeping over already-placed objects during transit

---

## PHASE 2B — executor.place(x, y, object_id, target_z) — Low-level place pipeline

### Node: Plan path to hover above target (OMPL)
- Same OMPL + CollisionPolicy pipeline as pick
- Goal: end-effector hovering over (x, y) at height above target_z

### Node: IK solve for release pose
- Same Pinocchio → DLS cascade as pick
- Target: end-effector at (x, y, target_z)

### Node: Move to release pose
- Execute IK result: arm moves to release position
- Open gripper (release object)
- Object falls under gravity to surface

### Node: Post-place lift
- Arm lifts up by release_lift distance (0.050m for place, 0.008m for stack)
- release_lift is small for stacking (don't knock the tower)

### Node: Return arm to REST_POSE (HOME)
- Parks arm at high safe configuration again

---

## PHASE 2 CONTINUED — After execution

### Node: Append result to history
- StepResult(action, ok, stage, failure_reason, detail) → history list

### Node: env.snapshot() — re-read physical state
- Re-reads all live object positions from MuJoCo after settling
- Recomputes reach and obstacle status

### Node: evaluator.placed_status(snapshot) — re-evaluate
- StackEvaluator: re-checks tower geometry
    - Base cube: within 4.5cm XY of tower_base region?
    - Level N cube: within 4.5cm XY AND 3.5cm Z of (base.z + N×0.06m)?
    - If any lower cube is knocked off → all cubes above it become placed=False again
- TidyEvaluator: re-checks zone membership
    - Cube: in lower half of goal area (y <= goal_y - 0.02)?
    - Circle: in upper half of goal area (y >= goal_y + 0.02)?

### Node: No-progress guard check
- progressed = (new placed count) > (previous placed count)
- no_progress counter:
    - Resets to 0 if: result.ok=True AND at least one new object is placed
    - Increments otherwise
- Decision: no_progress >= 6?
    - YES: append stop action to history, EXIT LOOP → go to Phase 3
    - NO:  continue loop

### Node: build_observation (updated)
- Rebuilds Observation with:
    - Updated object positions and placed flags
    - Full history of all steps so far (each entry: step, action, ok, stage, failure_reason)
    - Incremented step counter
- The LLM sees: what just happened, what failed, what is now placed, what is still on the table

### LOOP BACK to "Decision: Is goal satisfied?"

---

## PHASE 3 — SCORING AND TEARDOWN (runs once after loop exits)

### Node: evaluator.score(task, snapshot, history, steps_used, duration_ms, planner.name)
- Re-runs placed_status on the final snapshot
- Builds RunRecord:
    - task_id, scene_key, planner_name
    - success: True if ALL objects are placed
    - objects_placed / objects_total
    - score: objects_placed / objects_total (0.0 to 1.0)
    - steps_used, duration_ms
    - failures list: for each unplaced object → {object_id, failure_reason from history}
    - extra: list of placed objects, list of all actions attempted

### Node: env.close()
- Calls executor.shutdown_runtime() to clean up MuJoCo viewer and physics

### Node: Return RunRecord to runner script

### Node: Runner prints summary
- success, objects_placed, score, steps_used, duration_ms, failures

### Node: Write to CSV files
- Summary CSV: one row per run (success, count, duration, scene, obstacle_mode)
- Events CSV: every IK attempt, pick/place check, collision, transport result
  (Used by HintCache in future runs to adapt behavior)

---

## FULL DATA FLOW SUMMARY

Runner Script
  → sets env vars
  → constructs [MujocoEnvironment, Evaluator, LLMPlanner, TaskSpec]
  → calls harness.run()

harness.run()
  → env.reset()                           setup
  → env.snapshot()                        read physical world
  → evaluator.placed_status()             check goal
  → build_observation()                   package for planner
  → loop:
      evaluator.is_goal_satisfied()       early exit check
      planner.act(obs)                    LLM decides action
        → LLMClient.complete()            API call (1 per step)
        → extract_json_object()           parse reply
        → parse_action()                  validate vocab
        → _parse_and_validate()           validate against live scene
      env.execute(action)                 physical execution
        → place_resolver()                get real coordinates
        → executor.pick()                 OMPL + IK + gripper
            → CollisionPolicy             per-sample validity
            → Pinocchio IK or DLS IK      joint angle solving
        → feedback.check_pick()           verify pickup
        → executor.place()               OMPL + IK + release
        → settle physics                  wait for object to stabilize
        → verify final position           check tolerance
      env.snapshot()                      re-read world
      evaluator.placed_status()           re-check goal
      no_progress guard                   detect stuck planner
      build_observation()                 update for next step
  → evaluator.score()                     final scoring
  → env.close()
  → return RunRecord

RunRecord
  → printed to console
  → written to summary CSV
  → events written to events CSV
  → HintCache reads events CSV on NEXT run to improve IK behavior

---

## KEY DESIGN DECISIONS (for chart annotations)

1. ANTI-BIAS: build_observation() in harness.py is called in ONE place only.
   Every model (LLM or scripted) receives the identical Observation struct.
   The evaluator knows nothing about which planner ran.
   The executor is shared and identical for all models.

2. SYMBOL/GEOMETRY SEPARATION: LLM emits only symbols (stack/place/stop + object name).
   All coordinates, IK, and collision logic live in the Environment and Executor.
   The LLM never sees or outputs XYZ coordinates.

3. CLOSED-LOOP RECOVERY: placed=False propagates up the tower each step.
   If a cube is knocked off, the LLM automatically sees placed=False and can re-stack.
   No special recovery logic needed — it emerges from the loop structure.

4. PROVIDER SWAP: Only make_llm_client() changes when switching models.
   AnthropicClient and OpenAIClient both implement the same LLMClient protocol.
   Planner, Harness, Environment, Evaluator are all untouched.

5. IK CASCADE: Pinocchio (accurate, URDF-based) runs first.
   MuJoCo DLS (robust, always converges) is the fallback.
   HintCache can force DLS if Pinocchio historically fails for this scene.

6. SAFETY NETS:
   - LLM errors (bad JSON, hallucinated objects) → retry up to 3x → safe stop fallback
   - Planner crash → harness catches exception → safe stop
   - Stuck planner → no-progress guard after 6 consecutive non-advancing steps
   - Attempt cap per object in planners → max 3 attempts per object before moving on
