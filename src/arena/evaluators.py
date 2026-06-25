"""
Task evaluators: the shared goal predicate + scoring for a task.

An Evaluator knows nothing about which planner produced the actions -- it only
looks at physical state. That blindness is what keeps scoring unbiased across
models. Tolerances live here as explicit constructor args (no magic numbers
buried in control flow), so they can be moved into per-task config later.
"""

from __future__ import annotations

import math
from typing import Sequence

from .contracts import (
    RunRecord,
    StepResult,
    TaskSpec,
    WorldSnapshot,
)


class StackEvaluator:
    """Goal: build a single vertical tower in `order`, base resting in `region`.

    Placement is judged purely from live positions, using the base cube's own
    pose as the tower reference (robust to table height and to the base settling
    a few millimetres). A cube counts as placed only if every cube below it is
    also placed -- so a knocked-over lower cube invalidates everything above it,
    which is exactly what we want a closed-loop planner to notice and fix.
    """

    def __init__(
        self,
        order: Sequence[str],
        region: str,
        cube_full_height_m: float = 0.06,
        xy_tolerance_m: float = 0.045,
        z_tolerance_m: float = 0.035,
    ) -> None:
        self.order = list(order)
        self.region = region
        self.cube_full_height_m = cube_full_height_m
        self.xy_tolerance_m = xy_tolerance_m
        self.z_tolerance_m = z_tolerance_m

    def placed_status(self, snapshot: WorldSnapshot) -> dict[str, bool]:
        by_id = {o.id: o for o in snapshot.objects}
        status = {o.id: False for o in snapshot.objects}
        region_xy = snapshot.regions.get(self.region)

        base_id = self.order[0] if self.order else None
        base = by_id.get(base_id) if base_id else None

        lower_ok = True
        for level, object_id in enumerate(self.order):
            obj = by_id.get(object_id)
            if obj is None or not lower_ok:
                status[object_id] = False
                lower_ok = False
                continue

            if level == 0:
                ok = obj.on_table and (
                    region_xy is None
                    or math.dist(obj.position[:2], region_xy[:2]) <= self.xy_tolerance_m
                )
            else:
                expected_z = base.position[2] + level * self.cube_full_height_m
                ok = (
                    math.dist(obj.position[:2], base.position[:2]) <= self.xy_tolerance_m
                    and abs(obj.position[2] - expected_z) <= self.z_tolerance_m
                )
            status[object_id] = ok
            lower_ok = lower_ok and ok
        return status

    def is_goal_satisfied(self, snapshot: WorldSnapshot) -> bool:
        status = self.placed_status(snapshot)
        return bool(self.order) and all(status.get(o, False) for o in self.order)

    def score(
        self,
        task: TaskSpec,
        snapshot: WorldSnapshot,
        history: list[StepResult],
        steps_used: int,
        duration_ms: int,
        planner_name: str,
    ) -> RunRecord:
        status = self.placed_status(snapshot)
        placed = [o for o in self.order if status.get(o, False)]
        total = len(self.order)
        success = len(placed) == total

        failures = []
        present = {o.id for o in snapshot.objects}
        for object_id in self.order:
            if status.get(object_id, False):
                continue
            reason = self._last_failure_reason(history, object_id)
            failures.append({
                "object_id": object_id,
                "in_scene": object_id in present,
                "failure_reason": reason or "object_not_in_tower",
            })

        return RunRecord(
            task_id=task.task_id,
            scene_key=task.scene_key,
            planner_name=planner_name,
            success=success,
            objects_placed=len(placed),
            objects_total=total,
            score=round(len(placed) / total, 4) if total else 0.0,
            steps_used=steps_used,
            duration_ms=duration_ms,
            seed=task.seed,
            failures=failures,
            extra={
                "placed": placed,
                "actions_attempted": [r.action.describe() for r in history],
            },
        )

    @staticmethod
    def _last_failure_reason(history: list[StepResult], object_id: str) -> str | None:
        for result in reversed(history):
            if result.action.object_id == object_id and not result.ok:
                return result.failure_reason
        return None


class TidyEvaluator:
    """Goal: every object sitting ON its type's row strip.

    'Tidied' means the object's footprint overlaps the coloured row strip it
    belongs to: a cube must touch the cube (red) strip, a circle/cylinder the
    circle (blue) strip. An object dumped between the strips, in a second row off
    the line, or outside the goal area does NOT count -- which matches the visual
    "the cube is on the red line, the cylinder is on the blue line".

    A strip is centred at its row_y with a half-height of ~0.025 m; the default
    tolerances (cube 0.061, circle 0.051) are that half-height plus the object's
    own half-width, i.e. "the object overlaps the strip". Membership is judged
    from live positions, so scoring is identical across planners.
    """

    def __init__(
        self,
        object_ids: Sequence[str],
        class_by_id: dict[str, str],
        goal_x: float,
        cube_row_y: float,
        circle_row_y: float,
        goal_x_half: float = 0.26,
        cube_row_tol_m: float = 0.061,
        circle_row_tol_m: float = 0.051,
    ) -> None:
        self.object_ids = list(object_ids)
        self.class_by_id = dict(class_by_id)
        self.goal_x = goal_x
        self.cube_row_y = cube_row_y
        self.circle_row_y = circle_row_y
        self.goal_x_half = goal_x_half
        self.cube_row_tol_m = cube_row_tol_m
        self.circle_row_tol_m = circle_row_tol_m

    def region_for(self, object_id: str) -> str:
        return "cube_row" if self.class_by_id.get(object_id) == "cube" else "circle_row"

    def _on_strip(self, obj, klass: str) -> bool:
        if not (obj.on_table and abs(obj.position[0] - self.goal_x) <= self.goal_x_half):
            return False
        if klass == "cube":
            return abs(obj.position[1] - self.cube_row_y) <= self.cube_row_tol_m
        return abs(obj.position[1] - self.circle_row_y) <= self.circle_row_tol_m

    def placed_status(self, snapshot: WorldSnapshot) -> dict[str, bool]:
        by_id = {o.id: o for o in snapshot.objects}
        status = {o.id: False for o in snapshot.objects}
        for object_id in self.object_ids:
            obj = by_id.get(object_id)
            if obj is None:
                continue
            status[object_id] = self._on_strip(obj, self.class_by_id.get(object_id, "cube"))
        return status

    def is_goal_satisfied(self, snapshot: WorldSnapshot) -> bool:
        status = self.placed_status(snapshot)
        return bool(self.object_ids) and all(status.get(o, False) for o in self.object_ids)

    def score(
        self,
        task: TaskSpec,
        snapshot: WorldSnapshot,
        history: list[StepResult],
        steps_used: int,
        duration_ms: int,
        planner_name: str,
    ) -> RunRecord:
        status = self.placed_status(snapshot)
        placed = [o for o in self.object_ids if status.get(o, False)]
        total = len(self.object_ids)
        present = {o.id for o in snapshot.objects}

        failures = []
        for object_id in self.object_ids:
            if status.get(object_id, False):
                continue
            failures.append({
                "object_id": object_id,
                "in_scene": object_id in present,
                "intended_region": self.region_for(object_id),
                "failure_reason": StackEvaluator._last_failure_reason(history, object_id)
                or "object_not_in_its_row",
            })

        return RunRecord(
            task_id=task.task_id,
            scene_key=task.scene_key,
            planner_name=planner_name,
            success=len(placed) == total,
            objects_placed=len(placed),
            objects_total=total,
            score=round(len(placed) / total, 4) if total else 0.0,
            steps_used=steps_used,
            duration_ms=duration_ms,
            seed=task.seed,
            failures=failures,
            extra={
                "placed": placed,
                "actions_attempted": [r.action.describe() for r in history],
            },
        )
