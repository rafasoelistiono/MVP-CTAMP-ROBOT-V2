"""
FakeEnvironment: a physics-free Environment for testing the seam.

It bookkeeps object positions instead of running MuJoCo, so the harness loop,
planners, and evaluators can be exercised end-to-end in milliseconds with no
simulator. It also supports scripted failures/disturbances so closed-loop
recovery can be tested deterministically.
"""

from __future__ import annotations

from typing import Callable, Optional

from .contracts import (
    Action,
    ActionType,
    ObjectView,
    ObstacleView,
    StepResult,
    WorldSnapshot,
)

CUBE_FULL_HEIGHT_M = 0.06


class FakeEnvironment:
    def __init__(
        self,
        objects: dict[str, tuple[float, float, float]],
        regions: dict[str, tuple[float, float, float]],
        table_z: float = 0.80,
        classes: Optional[dict[str, str]] = None,
        obstacles: Optional[dict[str, tuple[float, float, float]]] = None,
        on_execute: Optional[Callable[["FakeEnvironment", Action, int], Optional[StepResult]]] = None,
    ) -> None:
        self._initial = {k: tuple(v) for k, v in objects.items()}
        self.positions = {k: tuple(v) for k, v in objects.items()}
        self.regions = {k: tuple(v) for k, v in regions.items()}
        self.table_z = table_z
        self.classes = classes or {k: "cube" for k in objects}
        self.obstacles = {k: tuple(v) for k, v in (obstacles or {}).items()}
        self.held: Optional[str] = None
        self._on_execute = on_execute
        self._calls = 0

    def reset(self) -> None:
        self.positions = {k: tuple(v) for k, v in self._initial.items()}
        self.held = None
        self._calls = 0

    def snapshot(self) -> WorldSnapshot:
        objects = tuple(
            ObjectView(
                id=object_id,
                klass=self.classes.get(object_id, "cube"),
                position=pos,
                reach="OK",
                obstacle="CLEAR",
                on_table=abs(pos[2] - (self.table_z + CUBE_FULL_HEIGHT_M / 2)) < 0.02,
                held=(object_id == self.held),
                placed=False,  # filled in by the harness via the evaluator
            )
            for object_id, pos in self.positions.items()
        )
        obstacles = tuple(ObstacleView(id=k, position=v) for k, v in self.obstacles.items())
        return WorldSnapshot(
            objects=objects,
            obstacles=obstacles,
            regions=dict(self.regions),
            held_object=self.held,
        )

    def execute(self, action: Action) -> StepResult:
        self._calls += 1
        if self._on_execute is not None:
            injected = self._on_execute(self, action, self._calls)
            if injected is not None:
                return injected

        if action.type is ActionType.PLACE:
            region_xy = self.regions.get(action.region)
            if region_xy is None or action.object_id not in self.positions:
                return StepResult(action, ok=False, stage="place",
                                  failure_reason="unknown_object_or_region")
            self.positions[action.object_id] = (
                region_xy[0], region_xy[1], self.table_z + CUBE_FULL_HEIGHT_M / 2,
            )
            return StepResult(action, ok=True, stage="place")

        if action.type is ActionType.STACK:
            if action.object_id not in self.positions or action.on not in self.positions:
                return StepResult(action, ok=False, stage="stack",
                                  failure_reason="unknown_object")
            base = self.positions[action.on]
            self.positions[action.object_id] = (
                base[0], base[1], base[2] + CUBE_FULL_HEIGHT_M,
            )
            return StepResult(action, ok=True, stage="stack")

        return StepResult(action, ok=True, stage="stop")

    def close(self) -> None:
        return None
