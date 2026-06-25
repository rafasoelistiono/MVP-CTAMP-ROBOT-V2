"""
Closed-loop benchmark harness.

run() is the entire control loop, and the ONLY place an Observation is built.
Because every planner is driven through this one function, every planner gets:
  - the identical Observation (same fields, same serialization),
  - the identical action vocabulary,
  - the identical step budget and scoring.

The harness talks to an Environment (physical execution) and an Evaluator
(goal predicate + score). It never touches MuJoCo directly, so the loop is
testable with a FakeEnvironment.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from .contracts import (
    Action,
    ActionType,
    Environment,
    Evaluator,
    HistoryEntry,
    Observation,
    ObjectView,
    Planner,
    RunRecord,
    StepResult,
    TaskSpec,
    WorldSnapshot,
)

ACTION_VOCABULARY = (
    "stack(object, on)  -- pick `object` and place it on top of `on`",
    "place(object, region)  -- pick `object` and place it into named `region`",
    "stop(reason)  -- declare the task finished",
)

# A planner that loops forever (e.g. repeatedly emitting an action that never
# succeeds) must still terminate the run. We cap consecutive no-progress steps.
DEFAULT_MAX_NO_PROGRESS = 6


def build_observation(
    task: TaskSpec,
    snapshot: WorldSnapshot,
    placed_status: dict[str, bool],
    step: int,
    history: list[StepResult],
) -> Observation:
    """Single source of truth for what a planner sees. Folds the evaluator's
    placed-status into each ObjectView so disturbances are reflected live."""
    objects = tuple(
        ObjectView(
            id=o.id,
            klass=o.klass,
            position=o.position,
            reach=o.reach,
            obstacle=o.obstacle,
            on_table=o.on_table,
            held=o.held,
            placed=placed_status.get(o.id, False),
        )
        for o in snapshot.objects
    )
    hist = tuple(
        HistoryEntry(
            step=i,
            action=r.action.to_dict(),
            ok=r.ok,
            stage=r.stage,
            failure_reason=r.failure_reason,
        )
        for i, r in enumerate(history)
    )
    return Observation(
        task_id=task.task_id,
        goal=task.goal_text,
        step=step,
        max_steps=task.max_steps,
        held_object=snapshot.held_object,
        objects=objects,
        obstacles=snapshot.obstacles,
        regions=snapshot.regions,
        action_vocabulary=ACTION_VOCABULARY,
        history=hist,
    )


def run(
    task: TaskSpec,
    planner: Planner,
    env: Environment,
    evaluator: Evaluator,
    *,
    max_no_progress: int = DEFAULT_MAX_NO_PROGRESS,
    on_step: Optional[Callable[[int, Action, StepResult], None]] = None,
) -> RunRecord:
    """Drive one closed-loop episode and return a standardized RunRecord."""
    started = time.perf_counter()
    env.reset()

    snapshot = env.snapshot()
    placed = evaluator.placed_status(snapshot)
    obs = build_observation(task, snapshot, placed, step=0, history=[])
    planner.reset(task, obs)

    history: list[StepResult] = []
    no_progress = 0
    steps_used = 0

    for step in range(task.max_steps):
        if evaluator.is_goal_satisfied(snapshot):
            break

        try:
            action = planner.act(obs)
        except Exception as exc:  # an LLM planner must never crash the run
            action = Action.stop(f"planner_error:{type(exc).__name__}:{exc}")
        if not isinstance(action, Action):
            action = Action.stop("planner_returned_non_action")
        if action.type is ActionType.STOP:
            history.append(StepResult(action=action, ok=True, stage="stop",
                                      failure_reason=action.reason))
            break

        result = env.execute(action)
        history.append(result)
        steps_used += 1
        if on_step is not None:
            on_step(step, action, result)

        prev_placed = placed
        snapshot = env.snapshot()
        placed = evaluator.placed_status(snapshot)

        progressed = sum(placed.values()) > sum(prev_placed.values())
        no_progress = 0 if (result.ok and progressed) else no_progress + 1

        obs = build_observation(task, snapshot, placed, step=step + 1, history=history)

        if no_progress >= max_no_progress:
            history.append(StepResult(
                action=Action.stop("max_no_progress_reached"),
                ok=False, stage="harness",
                failure_reason="max_no_progress_reached",
            ))
            break

    duration_ms = int((time.perf_counter() - started) * 1000)
    record = evaluator.score(
        task, snapshot, history, steps_used, duration_ms, planner.name,
    )
    env.close()
    return record
