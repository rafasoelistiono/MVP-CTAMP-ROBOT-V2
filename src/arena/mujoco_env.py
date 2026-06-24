"""
MujocoEnvironment: the real Environment adapter.

It translates the symbolic Action vocabulary (stack / place) into the existing
executor primitives (pick / place + feedback checks), and reports live physical
state as a WorldSnapshot. All geometry lives here -- the planner only ever names
objects and regions, never coordinates.

Dependencies (executor, feedback, scene-status helpers) are injected so the
arena package stays free of MuJoCo/OMPL/script imports; the run script wires
them in after it has set the scene env vars and imported the executor runtime.

NOTE: unlike the rest of the package, this adapter can only be validated against
a live MuJoCo run (the parity check), since it drives the real arm.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

from .contracts import (
    Action,
    ActionType,
    ObjectView,
    ObstacleView,
    StepResult,
    WorldSnapshot,
)


class MujocoEnvironment:
    def __init__(
        self,
        *,
        executor: Any,
        feedback: Any,
        world_state: dict,
        regions: dict[str, tuple[float, float, float]],
        place_resolver: Callable[[str, str, "MujocoEnvironment"], Optional[tuple[float, float, float]]],
        obstacle_status_fn: Callable[[dict, dict], str],
        reach_status_fn: Callable[[dict, dict], str],
        settle_steps: int = 360,
        cube_full_height_m: float = 0.06,
        max_pick_attempts: int = 3,
        ready_pose: Any = None,
        log_event: Optional[Callable[..., None]] = None,
    ) -> None:
        self.executor = executor
        self.feedback = feedback
        self.world_state = world_state
        # `regions` are named target centers shown to the planner (informational).
        # `place_resolver` owns the real geometry: given (object, region) it returns
        # the concrete (x, y, z) to place at -- a fixed point for stacking, or a
        # collision-free slot inside a row for tidy. Returning None = no valid slot.
        self.regions = dict(regions)
        self.place_resolver = place_resolver
        self._obstacle_status = obstacle_status_fn
        self._reach_status = reach_status_fn
        self.settle_steps = settle_steps
        self.cube_full_height_m = cube_full_height_m
        self.max_pick_attempts = max_pick_attempts
        # The arm parks at executor.GRASP_READY between every pick/place. The
        # default is a low, elbow-down pose that sweeps over just-placed objects,
        # so tasks should pass a high park pose (e.g. executor.HOME) here.
        self.ready_pose = ready_pose
        self._log_event = log_event or (lambda *a, **k: None)

        self._static_by_id = {o["id"]: o for o in world_state["movable_objects"]}
        self.table_z = float(world_state["table"]["z_top"])
        self._pick_attempts: dict[str, int] = {}

    def live_pose(self, object_id: str) -> tuple[float, float, float]:
        """Public accessor for a place_resolver to read live object positions."""
        return self._runtime_pose(object_id)

    def object_radius(self, object_id: str) -> float:
        return float(self._static_by_id.get(object_id, {}).get("radius", 0.04))

    # -- lifecycle ---------------------------------------------------------- #

    def reset(self) -> None:
        self._pick_attempts = {}
        # Raise the inter-action park pose so the arm retracts high instead of
        # sweeping low over already-placed objects. Prefer REST_POSE (parking
        # only) so GRASP_READY stays intact as the tuned IK seed for low grasps;
        # fall back to GRASP_READY for executors without REST_POSE.
        if self.ready_pose is not None:
            if hasattr(self.executor, "REST_POSE"):
                self.executor.REST_POSE = self.ready_pose
            elif hasattr(self.executor, "GRASP_READY"):
                self.executor.GRASP_READY = self.ready_pose
            self._log_event("ARENA_REST_POSE", "SET", phase="reset",
                            q=[round(float(v), 4) for v in self.ready_pose])

    def close(self) -> None:
        try:
            self.executor.shutdown_runtime()
        except Exception:  # pragma: no cover - teardown best effort
            pass

    # -- state -------------------------------------------------------------- #

    def _runtime_pose(self, object_id: str) -> tuple[float, float, float]:
        self.executor.mujoco.mj_forward(self.executor.model, self.executor.data)
        pos = self.executor.data.xpos[self.executor.name_to_cube[object_id]]
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def snapshot(self) -> WorldSnapshot:
        held = getattr(self.executor, "_held_object_name", None)
        objects = []
        for object_id, static in self._static_by_id.items():
            pos = self._runtime_pose(object_id)
            objects.append(ObjectView(
                id=object_id,
                klass=static.get("class", "cube"),
                position=pos,
                reach=self._reach_status(static, self.world_state),
                obstacle=self._obstacle_status(static, self.world_state),
                on_table=pos[2] < self.table_z + 0.07,
                held=(object_id == held),
                placed=False,  # the harness fills this via the evaluator
            ))
        obstacles = tuple(
            ObstacleView(id=o["id"], position=tuple(float(v) for v in o["position"]))
            for o in self.world_state.get("ceramic_obstacles", [])
        )
        goal = self.world_state["goal_center"]
        regions = {name: tuple(float(v) for v in xyz) for name, xyz in self.regions.items()}
        regions.setdefault("goal_center", (float(goal[0]), float(goal[1]), self.table_z))
        return WorldSnapshot(
            objects=tuple(objects),
            obstacles=obstacles,
            regions=regions,
            held_object=held,
        )

    # -- execution ---------------------------------------------------------- #

    def execute(self, action: Action) -> StepResult:
        if action.type is ActionType.PLACE:
            try:
                target = self.place_resolver(action.object_id, action.region, self)
            except Exception as exc:  # resolver is task-supplied; never crash the run
                return StepResult(action, ok=False, stage="precheck",
                                  failure_reason="place_resolver_error", detail={"error": str(exc)})
            if target is None:
                return StepResult(action, ok=False, stage="precheck",
                                  failure_reason=f"no_safe_slot_in_region:{action.region}")
            return self._transport(action, action.object_id, target[0], target[1],
                                   target[2], release_lift=0.050, stage="place")

        if action.type is ActionType.STACK:
            if action.on not in self._static_by_id:
                return StepResult(action, ok=False, stage="precheck",
                                  failure_reason=f"unknown_base:{action.on}")
            base = self._runtime_pose(action.on)
            target_z = base[2] + self.cube_full_height_m
            return self._transport(action, action.object_id, base[0], base[1],
                                   target_z, release_lift=0.008, stage="stack")

        return StepResult(action, ok=True, stage="stop")

    def _transport(
        self,
        action: Action,
        object_id: str,
        x: float,
        y: float,
        z: float,
        *,
        release_lift: float,
        stage: str,
    ) -> StepResult:
        if object_id not in self._static_by_id:
            return StepResult(action, ok=False, stage="precheck",
                              failure_reason=f"unknown_object:{object_id}")

        static = self._static_by_id[object_id]
        if self._obstacle_status(static, self.world_state) == "TOO_CLOSE":
            return StepResult(action, ok=False, stage="precheck",
                              failure_reason="object_too_close_to_obstacle")
        if self._reach_status(static, self.world_state) == "HARD":
            return StepResult(action, ok=False, stage="precheck",
                              failure_reason="object_outside_conservative_reach")

        self._pick_attempts[object_id] = self._pick_attempts.get(object_id, 0) + 1

        # --- pick --------------------------------------------------------- #
        try:
            self.executor.pick(object_id)
        except RuntimeError as exc:
            self._settle()
            return StepResult(action, ok=False, stage="pick",
                              failure_reason="executor_runtime_error",
                              detail={"error": str(exc)})

        pick_lifted, pick_z = self.feedback.check_pick(
            self.executor.model, self.executor.data, self.executor.name_to_cube[object_id]
        )
        held_ok = getattr(self.executor, "_held_object_name", None) == object_id
        if not (pick_lifted and held_ok):
            reason = "object_not_held_after_pick" if pick_lifted else "object_not_lifted_after_pick"
            self.executor.drop(object_id)
            self._settle()
            self._log_event("CHECK_PICK", "FAILED", object_id=object_id,
                            phase=stage, failure_reason=reason, object_z=pick_z)
            return StepResult(action, ok=False, stage="pick",
                              failure_reason=reason, detail={"z": pick_z})
        self._log_event("CHECK_PICK", "OK", object_id=object_id, phase=stage, object_z=pick_z)

        # --- place -------------------------------------------------------- #
        try:
            self.executor.place(
                x, y, object_id,
                target_z=z,
                release_lift=release_lift,
                post_place_ignored_body_names=[],
            )
        except RuntimeError as exc:
            self._settle()
            return StepResult(action, ok=False, stage="place",
                              failure_reason="executor_runtime_error",
                              detail={"error": str(exc)})
        self._settle()

        actual = self._runtime_pose(object_id)
        xy_err = math.dist(actual[:2], (x, y))
        z_err = abs(actual[2] - z)
        ok = xy_err <= 0.045 and z_err <= 0.035
        self._log_event(
            "CHECK_TRANSPORT", "OK" if ok else "FAILED", object_id=object_id, phase=stage,
            target_xyz=[round(x, 4), round(y, 4), round(z, 4)],
            actual_xyz=[round(v, 4) for v in actual],
            distance_to_target=round(xy_err, 4), z_error=round(z_err, 4),
            failure_reason=None if ok else "object_not_at_target_after_place",
        )
        return StepResult(
            action, ok=ok, stage=stage,
            failure_reason=None if ok else "object_not_at_target_after_place",
            detail={"actual": [round(v, 4) for v in actual],
                    "xy_error": round(xy_err, 4), "z_error": round(z_err, 4)},
        )

    def _settle(self) -> None:
        for _ in range(self.settle_steps):
            self.executor.mujoco.mj_step(self.executor.model, self.executor.data)
            self.executor.viewer.sync()
