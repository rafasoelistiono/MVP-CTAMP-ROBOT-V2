# End-to-End CTAMP Pipeline: Obstacle, Goal, and Log POV

Dokumen ini merangkum alur runtime task OMPL-only setelah pull terbaru,
termasuk HintCache, obstacle handling, goal allocation, dan alasan yang
terlihat dari event log.

## Script Coverage

| Script | Object classes | Scene conditions | HintCache |
|---|---|---|---|
| `scripts/align_cubes_ompl_only.py` | cube | group/ungroup, obs/no_obs | yes, default on |
| `scripts/align_tabung_ompl_only.py` | circle/cylinder | group/ungroup, obs/no_obs | yes, default on |
| `scripts/separate_groups_ompl_only.py` | cube + circle | group/ungroup, obs/no_obs | yes, default on |
| `scripts/stack_cubes_ompl_only.py` | cube | group/ungroup, obs/no_obs | no |

HintCache can be disabled in supported scripts with `--no-hint-cache`.
Stacking does not currently call `executor.init_hint_cache()`.

## End-to-End Pipeline

```mermaid
flowchart TD
    A[CLI task script] --> B[normalize scene key]
    B --> C[prepare_scene_variant]
    C --> D[hard-coded object and obstacle placement]
    D --> E[parse scene: robot, table, goal, objects, obstacles]
    E --> F[target builder]
    F --> G{precheck object}
    G -->|too close obstacle| G1[skip with object_too_close_to_obstacle]
    G -->|outside reach| G2[skip with object_outside_conservative_reach]
    G -->|valid| H[move order]
    H --> I[executor import]
    I --> J{HintCache supported and enabled?}
    J -->|yes| K[load past *_events.csv]
    J -->|no| L[fixed defaults]
    K --> M[pick object]
    L --> M
    M --> N[IK candidates]
    N --> O[MuJoCo FK validation]
    O --> P[CollisionPolicy and OMPL state validity]
    P --> Q[OMPL plan]
    Q --> R[trajectory execution]
    R --> S[pick feedback]
    S -->|not lifted| S1[retry or fail with reason]
    S -->|lifted| T[place at allocated goal]
    T --> U[place feedback]
    U -->|not on target| U1[retry or fail with reason]
    U -->|ok| V[final validation]
    V --> W[summary CSV and event CSV]
```

## HintCache Decision Flow

```mermaid
flowchart TD
    A[past event logs] --> B[HintCache]
    B --> C[pinocchio fallback rate]
    B --> D[near-miss IK rejection buckets]
    B --> E[pick profile success buckets]
    C -->|rate high and enough samples| C1[preferred_backend = mujoco_dls]
    D -->|near miss frequent and enough samples| D1[widen pos_err tolerance]
    E -->|profile has enough successes| E1[start from preferred grasp profile]
    C1 --> F[executor._solve_ik_to]
    D1 --> G[executor._ranked_ik_goals]
    E1 --> H[executor.pick]
```

Cold start is safe: if there are not enough samples, HintCache returns `None`
and executor uses fixed defaults.

## Obstacle and Goal Handling

```mermaid
flowchart TD
    A[object source pose] --> B[min obstacle XY distance]
    B --> C{distance band}
    C -->|less than MIN_PICK_OBSTACLE_CLEARANCE| C1[cancel pick or precheck skip]
    C -->|near obstacle| C2[cautious high-clearance motion]
    C -->|clear| C3[normal profile]
    C2 --> D[extra IK target candidates away from obstacle]
    C2 --> E[tighter grip for near cube/cylinder]
    C3 --> F[normal IK candidates]
    D --> G[OMPL collision-checked path]
    E --> G
    F --> G
    G --> H[goal target]
    H --> I{goal is safe from table, obstacle, and existing occupied targets?}
    I -->|no| I1[search nearby safe target]
    I -->|yes| J[place]
    I1 --> J
```

Object and goal positions are deterministic, not random. Scene variants are
generated from hard-coded placements in `ctamp_task_utils.py`, then target
builders compute deterministic goal slots from the goal area plus safety search.

## Stack-Specific Flow

```mermaid
flowchart TD
    A[eligible cubes] --> B[allocate 3 base slots]
    B --> C[allocate 1 top slot]
    C --> D[place base cubes first]
    D --> E{all base cubes moved?}
    E -->|no| D
    E -->|yes| F[refresh top target from support cube actual pose]
    F --> G[place top cube]
    G --> H[final validation for all 4 cubes]
```

The important refactor is that the top cube waits for all base cubes, not only
its direct support. This prevents a later base retry from knocking down the top
cube after it was already placed.

## Event Log POV

| Log stage | What the robot is doing | Typical reason field |
|---|---|---|
| `TASK_CONTEXT` | Records scene, move order, targets, skipped precheck | none |
| `OBSTACLE_PROXIMITY` | Marks object as near obstacle or blocked | `object_too_close_to_obstacle` |
| `HINT_CACHE` | Loads learned hints from previous logs | `LOADED`, `FAILED` |
| `PICK_PROFILE` | Chooses grip, grasp offset, clearance profile | profile metadata |
| `PICK_PRECHECK` | Reads object pose and obstacle distance before motion | pose or clearance data |
| `IK_CANDIDATE` | Tests candidate pose/joint solution | `ik_error_above_limit`, `ik_goal_collision_invalid` |
| `IK_SOLVE` | Selects IK backend or falls back | `pinocchio_fk_validation_failed` |
| `OMPL_PLAN` | Plans collision-checked joint path | `ompl_failed_no_fallback` |
| `TRAJECTORY_EXEC` | Executes path and watches contacts | `collision_at_waypoint_*` |
| `OBSTACLE_MONITOR` | Fails fast if obstacle is displaced | `obstacle_displaced` |
| `CHECK_PICK` | Verifies object was lifted | `object_not_lifted_after_pick` |
| `CHECK_PLACE` | Verifies object reached target row | `object_not_on_target_after_place` |
| `CHECK_STACK_PLACE` | Verifies stack placement and height | `object_not_on_stack_target` |
| final validation | Ensures final state remains valid | `object_moved_after_stack` |

## Current Verification

Latest local verification after pull:

```text
pytest -q
13 passed
```

Runtime validations from the latest completed logs before this document:

| Task | Scene | Result |
|---|---|---|
| align cubes | group_obs | 4/4 |
| align cubes | ungroup_obs | 4/4 |
| align tabung | group_obs | 4/4 |
| align tabung | ungroup_obs | 4/4 |
| stack cubes | group_no_obs | 4/4 |
| stack cubes | ungroup_no_obs | 4/4 |
| stack cubes | group_obs | 4/4 |
| stack cubes | ungroup_obs | 4/4 |

