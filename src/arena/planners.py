"""
Baseline planners.

ScriptedStackPlanner is the *oracle baseline*: a deterministic closed-loop
planner that reproduces the hand-coded stacking strategy. It exists so we can
(a) prove the seam works and matches today's behavior, and (b) give every LLM a
fixed reference to be ranked against on the leaderboard.

It is intentionally dumb: it only knows the dependency order (place the base,
then stack each cube on the previous). All geometry is the Environment's job.
An LLMPlanner will implement the same `Planner` protocol and plug in here.
"""

from __future__ import annotations

from typing import Sequence

from .contracts import Action, Observation, TaskSpec


def _consecutive_failures(obs: Observation, object_id: str) -> int:
    """How many times this object has failed *in a row* since its last success.

    Counting consecutive (not total) failures lets a baseline give up on a
    genuinely stuck object, while NOT penalizing one that succeeded earlier and
    later needs redoing -- e.g. a tower cube that was placed, knocked over, and
    must be re-stacked. That is what lets the closed loop rebuild a toppled tower
    instead of exhausting a total-attempt budget mid-rebuild."""
    count = 0
    for h in reversed(obs.history):
        if h.action.get("object") != object_id:
            continue
        if h.ok:
            break
        count += 1
    return count


class ScriptedStackPlanner:
    """Emits the next transport needed to build the tower, reacting to live state.

    Because it reads `placed` from the Observation each step, it automatically
    re-issues a stack if a lower cube was disturbed -- the closed loop gives
    recovery for free, without any special-case code.
    """

    name = "scripted_stack"

    def __init__(self, order: Sequence[str], region: str = "tower_base", max_attempts: int = 3) -> None:
        self.order = list(order)
        self.region = region
        self.max_attempts = max_attempts

    def reset(self, task: TaskSpec, obs: Observation) -> None:  # noqa: ARG002
        return None

    def act(self, obs: Observation) -> Action:
        for level, object_id in enumerate(self.order):
            obj = obs.object(object_id)
            if obj is None:
                # Required cube missing from the scene -> tower impossible.
                return Action.stop(reason=f"{object_id}_not_in_scene")
            if obj.placed:
                continue
            # The tower is strictly ordered: if this level keeps failing in a row,
            # the tower can't be finished -- stop instead of looping. (Consecutive,
            # so a knock-and-rebuild of an earlier-placed cube doesn't count here.)
            if _consecutive_failures(obs, object_id) >= self.max_attempts:
                return Action.stop(reason=f"{object_id}_unplaceable_after_{self.max_attempts}_failures")
            if level == 0:
                return Action.place(object_id, region=self.region)
            base = self.order[level - 1]
            base_view = obs.object(base)
            if base_view is None or not base_view.placed:
                # Dependency not ready yet; address the base first.
                return Action.place(base, region=self.region) if level - 1 == 0 \
                    else Action.stack(base, on=self.order[level - 2])
            return Action.stack(object_id, on=base)
        return Action.stop(reason="tower_complete")


class ScriptedTidyPlanner:
    """Baseline for the tidy/sort task: route each object to its type's row.

    Closed-loop logic: each step, among objects not yet in their row (and not
    physically blocked), pick the easiest reachable one and emit
    place(object, its_region). Reading `placed` live means a disturbed object is
    automatically re-tidied. If the last action failed, it rotates to a different
    object so one stubborn item doesn't stall the whole run.
    """

    name = "scripted_tidy"

    def __init__(self, cube_region: str = "cube_row", circle_region: str = "circle_row",
                 max_attempts: int = 3) -> None:
        self.cube_region = cube_region
        self.circle_region = circle_region
        self.max_attempts = max_attempts

    def reset(self, task: TaskSpec, obs: Observation) -> None:  # noqa: ARG002
        return None

    def _region(self, klass: str) -> str:
        return self.cube_region if klass == "cube" else self.circle_region

    def act(self, obs: Observation) -> Action:
        fails = {o.id: _consecutive_failures(obs, o.id) for o in obs.objects}
        candidates = [
            o for o in obs.objects
            if not o.placed and not o.held
            and o.reach != "HARD" and o.obstacle != "TOO_CLOSE"
            and fails[o.id] < self.max_attempts
        ]
        if not candidates:
            return Action.stop(reason="no_more_reachable_objects")

        # Fewest recent failures first (round-robin so one stubborn item doesn't
        # stall the run), then easiest: clear of obstacles, comfortably in reach.
        candidates.sort(key=lambda o: (fails[o.id], o.obstacle == "NEAR",
                                       o.reach == "BORDERLINE", o.id))
        choice = candidates[0]
        return Action.place(choice.id, region=self._region(choice.klass))
