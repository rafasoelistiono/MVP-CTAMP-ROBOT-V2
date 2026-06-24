# Robot Arena — Benchmark Framework Design

## Goal

Turn this project from a single hard-coded task solver into a **robust, anti-bias
arena** that benchmarks multiple models (LLMs/chatbots) on **high-level,
long-horizon tabletop task planning** (cube stacking, tidy-up), and ranks them
best → worst on a leaderboard. The model does the *symbolic* planning; the
existing OMPL + Pinocchio IK stack stays as the shared low-level executor.

## The core shift: the task script must stop being the planner

Today each task script *is* the plan. In `stack_cubes_ompl_only.py`,
`_build_cube_stack_targets()` hard-codes the goal (`["cube1","cube2","cube3","cube4"]`),
the move ordering, the tower geometry, the retry policy, and the success
tolerances. There is no model in the loop — `llm_used=false` is a constant.

For an arena, that hard-coded intelligence is exactly **the thing under test**.
So we cut one seam:

```
   Planner            Environment              Evaluator
 (under test)   →   (shared, identical)   →   (shared, blind)
 decides WHAT       DOES it + reports         JUDGES it
 to do              physical state            goal + score
```

- **Planner** — an LLM or a baseline. Sees the live scene, returns the next
  symbolic action. This is what we swap and rank.
- **Environment** — translates symbolic actions into pick/place + IK/OMPL,
  reports state. Identical for every model.
- **Evaluator** — checks the goal predicate and scores. Knows nothing about
  which model ran.

## Two design decisions

1. **Closed-loop.** The planner sees the live scene, emits the *next* action,
   sees the result, repeats — rather than emitting one plan up front. This is
   what makes "long-horizon" meaningful: a model can notice a knocked-over cube
   and re-stack it. Recovery falls out of the loop for free.

2. **Symbolic actions only.** The vocabulary is transport-level:
   - `stack(object, on)` — put `object` on top of `on`
   - `place(object, region)` — put `object` into a named region
   - `stop(reason)` — declare the task finished

   The model never emits coordinates or gripper commands. All geometry and IK
   belong to the Environment. This keeps the benchmark about **reasoning and
   ordering**, and keeps every model on identical geometric footing.

## Why this is structurally anti-bias

Fairness is not a rule we remember to enforce — it is where the code lives:

- The **Observation** every planner sees is assembled in exactly one place
  (`harness.build_observation`). Same fields, same serialization, for all models.
- Every model's output flows through one **action parser** (`parse_action`);
  malformed / out-of-vocabulary actions are handled identically.
- One **Evaluator** scores all models, blind to model identity.
- Same task suite, same scenes, same step budget, same seeds.

Swapping `--planner scripted` for `--planner llm` changes only which Planner is
constructed — nothing about the scene, the actions, or the scoring.

## Status & roadmap

- **Phase 1 — Carve the seam (done, additive, nothing existing touched).**
  `src/arena/` defines the contracts, the closed-loop harness, a scripted
  baseline planner, the stack evaluator, a physics-free `FakeEnvironment` (for
  fast tests), and the real `MujocoEnvironment` adapter. 5/5 seam tests pass with
  no simulator. **Next concrete step: the parity run** — confirm
  `run_arena_stack.py` (scripted baseline through the seam) reproduces today's
  `stack_cubes` result on a real MuJoCo run.
- **Phase 2 — `LLMPlanner`** implementing the same `Planner` protocol. Same
  observation, same actions, same scorer. One model, one task, end-to-end.
- **Phase 3 — Generalize** the evaluator + tasks (tidy / align / separate /
  stack); finish moving scattered constants into explicit config; add seeds and
  randomized layouts.
- **Phase 4 — Multi-model runner** (model × task × seed) → standardized
  `RunRecord`s → leaderboard aggregator + an anti-bias checklist.

## File map (`src/arena/`)

| File | Role |
|------|------|
| `contracts.py` | Action, Observation, TaskSpec, RunRecord; Planner/Environment/Evaluator protocols; `parse_action` |
| `harness.py` | The closed-loop `run()`; the single place Observations are built |
| `evaluators.py` | `StackEvaluator` — goal predicate + scoring (explicit tolerances) |
| `planners.py` | `ScriptedStackPlanner` — the oracle baseline |
| `environment_fake.py` | `FakeEnvironment` — physics-free, for tests + recovery scenarios |
| `mujoco_env.py` | `MujocoEnvironment` — real adapter over the executor (pick/place + feedback) |

Runner: `scripts/run_arena_stack.py`. Tests: `tests/test_arena_seam.py`.
