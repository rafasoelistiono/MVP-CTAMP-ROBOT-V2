"""
Arena seam contracts.

This module defines the fixed interfaces that separate the three roles of the
benchmark:

    Planner   -- decides WHAT to do (the thing under test; an LLM or a baseline)
    Environment -- DOES it and reports physical state (shared, identical for all)
    Evaluator -- JUDGES it (shared goal predicate + scoring, identical for all)

The whole point of putting these contracts in one place is *anti-bias*: every
model receives the same Observation, may only emit the same Action vocabulary,
and is scored by the same Evaluator. Fairness is structural, not a thing we have
to remember to enforce per model.

Nothing in here imports MuJoCo / OMPL / executor, so it stays pure and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable


# --------------------------------------------------------------------------- #
# Actions  --  the symbolic, high-level vocabulary a planner may emit.
#
# Actions are transport-level on purpose: the planner says "put cube2 on cube1",
# never "open gripper / move joint 3". All geometry and IK belong to the
# Environment. This keeps the benchmark about reasoning + ordering, and keeps
# every model on equal geometric footing.
# --------------------------------------------------------------------------- #


class ActionType(str, Enum):
    STACK = "stack"   # pick `object_id` and place it on top of `on`
    PLACE = "place"   # pick `object_id` and place it into named `region`
    STOP = "stop"     # declare the task finished (no more actions)


@dataclass(frozen=True)
class Action:
    type: ActionType
    object_id: Optional[str] = None
    on: Optional[str] = None          # STACK: base object to stack onto
    region: Optional[str] = None      # PLACE: named target region
    reason: Optional[str] = None      # free annotation (esp. for STOP)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"action": self.type.value}
        if self.object_id is not None:
            out["object"] = self.object_id
        if self.on is not None:
            out["on"] = self.on
        if self.region is not None:
            out["region"] = self.region
        if self.reason is not None:
            out["reason"] = self.reason
        return out

    def describe(self) -> str:
        if self.type is ActionType.STACK:
            return f"stack({self.object_id}, on={self.on})"
        if self.type is ActionType.PLACE:
            return f"place({self.object_id}, region={self.region})"
        return f"stop({self.reason or ''})"

    # Convenience constructors ------------------------------------------------
    @staticmethod
    def stack(object_id: str, on: str) -> "Action":
        return Action(ActionType.STACK, object_id=object_id, on=on)

    @staticmethod
    def place(object_id: str, region: str) -> "Action":
        return Action(ActionType.PLACE, object_id=object_id, region=region)

    @staticmethod
    def stop(reason: Optional[str] = None) -> "Action":
        return Action(ActionType.STOP, reason=reason)


class ActionParseError(ValueError):
    """Raised when a planner (e.g. an LLM) emits an action we cannot honour."""


def parse_action(raw: Any) -> Action:
    """
    Turn an LLM/JSON-style dict into a validated Action.

    Every model's output flows through this one parser, so malformed or
    out-of-vocabulary actions are handled identically for all models.
    """
    if isinstance(raw, Action):
        return raw
    if not isinstance(raw, dict):
        raise ActionParseError(f"action must be an object, got {type(raw).__name__}")

    kind = str(raw.get("action", "")).strip().lower()
    obj = raw.get("object") or raw.get("object_id")
    reason = raw.get("reason")

    if kind == ActionType.STACK.value:
        on = raw.get("on") or raw.get("base")
        if not obj or not on:
            raise ActionParseError("stack requires 'object' and 'on'")
        return Action.stack(str(obj), str(on))
    if kind == ActionType.PLACE.value:
        region = raw.get("region") or raw.get("target")
        if not obj or not region:
            raise ActionParseError("place requires 'object' and 'region'")
        return Action.place(str(obj), str(region))
    if kind == ActionType.STOP.value:
        return Action.stop(str(reason) if reason else None)
    raise ActionParseError(f"unknown action '{kind}' (valid: stack, place, stop)")


# --------------------------------------------------------------------------- #
# Observation  --  the single, shared view a planner gets each step.
# Assembled in exactly one place (the harness) so every model sees the same thing.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ObjectView:
    id: str
    klass: str                       # "cube" | "circle" | "cylinder"
    position: tuple[float, float, float]
    reach: str                       # "OK" | "BORDERLINE" | "HARD"
    obstacle: str                    # "CLEAR" | "NEAR" | "TOO_CLOSE"
    on_table: bool
    held: bool
    placed: bool                     # currently satisfies its goal location


@dataclass(frozen=True)
class ObstacleView:
    id: str
    position: tuple[float, float, float]


@dataclass(frozen=True)
class HistoryEntry:
    step: int
    action: dict[str, Any]
    ok: bool
    stage: str
    failure_reason: Optional[str]


@dataclass(frozen=True)
class Observation:
    task_id: str
    goal: str                        # natural-language goal statement
    step: int
    max_steps: int
    held_object: Optional[str]
    objects: tuple[ObjectView, ...]
    obstacles: tuple[ObstacleView, ...]
    regions: dict[str, tuple[float, float, float]]
    action_vocabulary: tuple[str, ...]
    history: tuple[HistoryEntry, ...] = ()

    def object(self, object_id: str) -> Optional[ObjectView]:
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        return None

    @property
    def objects_by_id(self) -> dict[str, ObjectView]:
        return {obj.id: obj for obj in self.objects}

    def to_dict(self) -> dict[str, Any]:
        """Serializable view -- this is what an LLM planner is shown as text."""
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "step": self.step,
            "max_steps": self.max_steps,
            "held_object": self.held_object,
            "objects": [
                {
                    "id": o.id,
                    "class": o.klass,
                    "position": [round(v, 3) for v in o.position],
                    "reach": o.reach,
                    "obstacle": o.obstacle,
                    "on_table": o.on_table,
                    "held": o.held,
                    "placed": o.placed,
                }
                for o in self.objects
            ],
            "obstacles": [
                {"id": ob.id, "position": [round(v, 3) for v in ob.position]}
                for ob in self.obstacles
            ],
            "regions": {k: [round(v, 3) for v in xyz] for k, xyz in self.regions.items()},
            "action_vocabulary": list(self.action_vocabulary),
            "history": [asdict(h) for h in self.history],
        }


# --------------------------------------------------------------------------- #
# Snapshot  --  raw physical state the Environment exposes; the harness turns
# this into an Observation (so Observation assembly lives in one place).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WorldSnapshot:
    objects: tuple[ObjectView, ...]
    obstacles: tuple[ObstacleView, ...]
    regions: dict[str, tuple[float, float, float]]
    held_object: Optional[str]


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class StepResult:
    action: Action
    ok: bool
    stage: str                       # "pick" | "place" | "stack" | "precheck" | "stop"
    failure_reason: Optional[str] = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRecord:
    """One (planner x task x seed) result -- the unit the leaderboard aggregates."""
    task_id: str
    scene_key: str
    planner_name: str
    success: bool
    objects_placed: int
    objects_total: int
    score: float
    steps_used: int
    duration_ms: int
    seed: Optional[int] = None
    failures: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Task spec  --  declarative "what the task is", decoupled from "how to solve it".
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TaskSpec:
    task_id: str                     # e.g. "stack_cubes"
    scene_key: str                   # e.g. "group_no_obs"
    goal_text: str                   # natural-language goal shown to the planner
    max_steps: int = 24
    params: dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None


# --------------------------------------------------------------------------- #
# Roles  --  the three swappable/fixed pieces of the seam.
# --------------------------------------------------------------------------- #


@runtime_checkable
class Planner(Protocol):
    """The thing under test. Closed-loop: sees the live Observation, returns the
    next Action. An LLM planner and a scripted baseline both implement this."""

    name: str

    def reset(self, task: TaskSpec, obs: Observation) -> None: ...

    def act(self, obs: Observation) -> Action: ...


@runtime_checkable
class Environment(Protocol):
    """Executes symbolic actions and reports physical state. Identical for every
    model -- this is where fairness physically lives."""

    def reset(self) -> None: ...

    def snapshot(self) -> WorldSnapshot: ...

    def execute(self, action: Action) -> StepResult: ...

    def close(self) -> None: ...


@runtime_checkable
class Evaluator(Protocol):
    """Shared goal predicate + scoring. Knows nothing about which planner ran."""

    def placed_status(self, snapshot: WorldSnapshot) -> dict[str, bool]: ...

    def is_goal_satisfied(self, snapshot: WorldSnapshot) -> bool: ...

    def score(
        self,
        task: TaskSpec,
        snapshot: WorldSnapshot,
        history: list[StepResult],
        steps_used: int,
        duration_ms: int,
        planner_name: str,
    ) -> RunRecord: ...
