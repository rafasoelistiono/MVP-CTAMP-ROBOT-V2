from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import mujoco

from collision_policy import CollisionPolicy, DEFAULT_ROBOT_BODIES

try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError as e:
    raise ImportError(
        "OMPL Python bindings are not installed or not importable. "
        "Install OMPL and make sure 'from ompl import base, geometric' works."
    ) from e


class ClearanceObjective(ob.StateCostIntegralObjective):
    """
    Clearance-biased objective.

    Smaller cost = better.
    We use reciprocal clearance, so paths that stay farther from obstacles
    are cheaper.
    """

    def __init__(self, si, clearance_fn):
        super().__init__(si, True)
        self._clearance_fn = clearance_fn

    def stateCost(self, s):
        clr = max(float(self._clearance_fn(s)), 1e-3)
        return ob.Cost(1.0 / clr)


@dataclass
class OMPLConfig:
    planner_name: str = "BITstar"
    fragile_planner_name: str = "BITstar"
    time_limit: float = 2.0
    state_validity_resolution: float = 0.005
    sampler_range: float = 0.08
    waypoint_step: float = 0.015
    goal_tolerance: float = 1e-3


class PandaOMPLPlanner:
    """
    Joint-space OMPL planner for the 7-DoF Franka Panda in MuJoCo.

    Planned state:
        q = [joint1, joint2, joint3, joint4, joint5, joint6, joint7]

    Collision model:
        - checks robot vs environment contacts in MuJoCo
        - environment = all non-robot bodies by default
        - optional ignored_body_names lets you exclude the grasp target
          during a specific planning phase
    """

    DEFAULT_ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: Optional[OMPLConfig] = None,
        robot_body_names: Optional[Sequence[str]] = None,
        arm_joint_names: Optional[Sequence[str]] = None,
    ):
        self.model = model
        self.live_data = data
        self.cfg = config or OMPLConfig()

        self.robot_body_names = set(robot_body_names or DEFAULT_ROBOT_BODIES)
        self.collision_policy = CollisionPolicy(
            model=self.model,
            robot_body_names=tuple(self.robot_body_names),
        )
        self.arm_joint_names: List[str] = list(arm_joint_names or self.DEFAULT_ARM_JOINTS)

        self.arm_qpos_adr = np.array(
            [self.model.joint(n).qposadr[0] for n in self.arm_joint_names],
            dtype=int,
        )
        self.arm_ranges = np.array(
            [self.model.joint(n).range for n in self.arm_joint_names],
            dtype=float,
        )

        self.ndof = len(self.arm_joint_names)
        self.lower = self.arm_ranges[:, 0].copy()
        self.upper = self.arm_ranges[:, 1].copy()

        # Internal planning data: never mutate the live sim state during planning.
        self.plan_data = mujoco.MjData(self.model)
        self._sync_from_live_data()

        # OMPL state space
        self.space = ob.RealVectorStateSpace(self.ndof)
        bounds = ob.RealVectorBounds(self.ndof)
        for i in range(self.ndof):
            bounds.setLow(i, float(self.lower[i]))
            bounds.setHigh(i, float(self.upper[i]))
        self.space.setBounds(bounds)

        self.ssi = ob.SpaceInformation(self.space)
        self.ssi.setStateValidityCheckingResolution(self.cfg.state_validity_resolution)
        self.ssi.setStateValidityChecker(self._is_state_valid)

        self._ignored_body_names: set[str] = set()
        self.collision_policy.set_ignored_bodies(self._ignored_body_names)

        # Set by plan() before each solve so the validity checker can
        # exempt the start state (the live arm configuration is always
        # physically realized and must never be rejected by OMPL).
        self._planning_start_q: Optional[np.ndarray] = None

        self._last_invalid_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync_live_data(self, data: mujoco.MjData) -> None:
        """Call this before planning if the simulation state has changed."""
        self.live_data = data
        self._sync_from_live_data()

    def plan(
        self,
        start_q: Sequence[float],
        goal_q: Sequence[float],
        time_limit: Optional[float] = None,
        planner_name: Optional[str] = None,
        ignored_body_names: Optional[Sequence[str]] = None,
        simplify: bool = True,
        fragile_mode: bool = False,
    ) -> Tuple[Optional[np.ndarray], dict]:
        """
        Plan a collision-free joint trajectory from start_q to goal_q.

        Returns:
            (trajectory, info)
            trajectory: np.ndarray of shape (N, 7) or None
            info: dict with metadata
        """
        start_q = np.asarray(start_q, dtype=float).reshape(-1)
        goal_q = np.asarray(goal_q, dtype=float).reshape(-1)

        if start_q.shape[0] != self.ndof or goal_q.shape[0] != self.ndof:
            raise ValueError(
                f"Expected {self.ndof}-D q vectors, got {start_q.shape} and {goal_q.shape}"
            )

        solve_time = float(time_limit if time_limit is not None else self.cfg.time_limit)

        # Update live scene snapshot used by the collision checker.
        self._sync_from_live_data()

        if ignored_body_names is not None:
            self._ignored_body_names = set(ignored_body_names)
        else:
            self._ignored_body_names = set()

        self.collision_policy.set_ignored_bodies(self._ignored_body_names)

        # Record the start configuration BEFORE setStartAndGoalStates() is
        # called so that _is_state_valid can exempt it unconditionally.
        self._planning_start_q = start_q.copy()

        planner_used = (
            self.cfg.fragile_planner_name if fragile_mode
            else (planner_name or self.cfg.planner_name)
        )

        goal_candidates = self._goal_candidates(goal_q)
        per_try_time = max(0.35, solve_time / max(len(goal_candidates), 1))

        last_info = {
            "solved": False,
            "planner_name": planner_used,
            "time_limit": solve_time,
            "ignored_body_names": sorted(list(self._ignored_body_names)),
            "start_q": start_q.tolist(),
            "goal_q": goal_q.tolist(),
            "goal_attempts": [],
        }

        try:
            for goal_idx, goal_try in enumerate(goal_candidates):
                goal_state = self._q_to_state(goal_try)

                # Skip invalid candidate goals immediately.
                if not self._is_state_valid(goal_state):
                    last_info["goal_attempts"].append({
                        "idx": goal_idx,
                        "goal_q": goal_try.tolist(),
                        "status": "invalid_goal",
                        "reason": self._last_invalid_reason,
                    })
                    continue

                ss = og.SimpleSetup(self.space)
                ss.setStateValidityChecker(self._is_state_valid)
                ss.getSpaceInformation().setStateValidityCheckingResolution(
                    self.cfg.state_validity_resolution
                )

                objective = self._make_objective(ss.getSpaceInformation(), fragile_mode)
                self._attach_objective(ss, objective)

                start = self._q_to_state(start_q)
                goal = self._q_to_state(goal_try)
                ss.setStartAndGoalStates(start, goal, self.cfg.goal_tolerance)

                planner = self._make_planner(
                    ss.getSpaceInformation(),
                    planner_name=planner_used,
                    fragile_mode=fragile_mode,
                )
                ss.setPlanner(planner)

                solved = ss.solve(per_try_time)

                last_info["goal_attempts"].append({
                    "idx": goal_idx,
                    "goal_q": goal_try.tolist(),
                    "status": "solved" if solved else "no_solution",
                })

                if not solved:
                    continue

                if simplify:
                    try:
                        ss.simplifySolution()
                    except Exception:
                        pass

                path = ss.getSolutionPath()
                raw = self._extract_path(path)
                dense = self._densify_path(raw, step=self.cfg.waypoint_step)

                last_info.update({
                    "solved": True,
                    "selected_goal_q": goal_try.tolist(),
                    "num_waypoints": int(dense.shape[0]),
                    "path_length_joint_space": float(self._path_length(dense)),
                })
                return dense, last_info

            return None, last_info

        finally:
            self._planning_start_q = None

    def is_state_valid_q(
        self,
        q: Sequence[float],
        ignored_body_names: Optional[Sequence[str]] = None,
    ) -> bool:
        """Direct validity check for a 7-DoF joint vector."""
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != self.ndof:
            return False
        if ignored_body_names is None:
            return self._is_state_valid(self._q_to_state(q))

        previous_ignored = set(self._ignored_body_names)
        self._ignored_body_names = set(ignored_body_names)
        self.collision_policy.set_ignored_bodies(self._ignored_body_names)
        try:
            return self._is_state_valid(self._q_to_state(q))
        finally:
            self._ignored_body_names = previous_ignored
            self.collision_policy.set_ignored_bodies(self._ignored_body_names)

    # ------------------------------------------------------------------
    # OMPL helpers
    # ------------------------------------------------------------------

    def _attach_objective(self, ss: og.SimpleSetup, objective) -> None:
        """
        Attach an optimization objective to the planning problem.

        Some OMPL Python builds expose setOptimizationObjective on SimpleSetup;
        others are safer via ProblemDefinition.
        """
        if hasattr(ss, "setOptimizationObjective"):
            try:
                ss.setOptimizationObjective(objective)
                return
            except Exception:
                pass

        pdef = ss.getProblemDefinition()
        pdef.setOptimizationObjective(objective)

    def _make_planner(self, si: ob.SpaceInformation, planner_name: str, fragile_mode: bool = False):
        if fragile_mode:
            # Prefer a more conservative planner for fragile scenes.
            planner = og.BITstar(si)
            if hasattr(planner, "setRange"):
                try:
                    planner.setRange(float(min(self.cfg.sampler_range, 0.06)))
                except Exception:
                    pass
            return planner

        name = planner_name.strip().lower()

        if name in {"rrtconnect", "rrt_connect"}:
            planner = og.RRTConnect(si)
        elif name in {"rrtstar", "rrt_star"}:
            planner = og.RRTstar(si)
        elif name in {"bitstar", "bit_star", "bit*"}:
            planner = og.BITstar(si)
        elif name in {"prmstar", "prm_star", "prm*"}:
            planner = og.PRMstar(si)
        elif name in {"kpiece", "kpiece1"}:
            planner = og.KPIECE1(si)
        elif name in {"est"}:
            planner = og.EST(si)
        else:
            raise ValueError(f"Unsupported OMPL planner_name: {planner_name}")

        if hasattr(planner, "setRange"):
            try:
                planner.setRange(float(self.cfg.sampler_range))
            except Exception:
                pass

        return planner

    def _q_to_state(self, q: np.ndarray):
        state = self.space.allocState()
        for i in range(self.ndof):
            state[i] = float(q[i])
        return state

    def _state_to_q(self, state) -> np.ndarray:
        return np.array([float(state[i]) for i in range(self.ndof)], dtype=float)

    def _extract_path(self, path) -> np.ndarray:
        n = int(path.getStateCount())
        waypoints = np.zeros((n, self.ndof), dtype=float)
        for i in range(n):
            waypoints[i] = self._state_to_q(path.getState(i))
        return waypoints

    def _densify_path(self, waypoints: np.ndarray, step: float = 0.03) -> np.ndarray:
        if waypoints.shape[0] <= 1:
            return waypoints.copy()

        dense = [waypoints[0].copy()]
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            dist = float(np.linalg.norm(b - a))
            n = max(1, int(np.ceil(dist / max(step, 1e-6))))
            for k in range(1, n + 1):
                alpha = k / n
                dense.append((1.0 - alpha) * a + alpha * b)
        return np.asarray(dense, dtype=float)

    def _path_length(self, waypoints: np.ndarray) -> float:
        if waypoints.shape[0] <= 1:
            return 0.0
        diffs = np.diff(waypoints, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def _clip_arm(self, q):
        return np.clip(np.asarray(q, dtype=float).reshape(-1), self.lower, self.upper)

    def _goal_candidates(self, goal_q: np.ndarray) -> list[np.ndarray]:
        """
        Small joint-space goal perturbations to rescue near-valid grasp goals.
        These are tiny on purpose: they are only for invalid-goal recovery.
        """
        goal_q = self._clip_arm(goal_q)

        offsets = [
            np.zeros(self.ndof),
            np.array([0.015, 0.010, 0.000, 0.000, 0.000, 0.010, 0.000]),
            np.array([-0.015, -0.010, 0.000, 0.000, 0.000, -0.010, 0.000]),
            np.array([0.000, 0.020, 0.000, 0.000, 0.000, -0.020, 0.000]),
            np.array([0.000, -0.020, 0.000, 0.000, 0.000, 0.020, 0.000]),
            np.array([0.000, 0.000, 0.000, 0.000, 0.000, 0.020, 0.000]),
            np.array([0.000, 0.000, 0.000, 0.000, 0.000, -0.020, 0.000]),
        ]
        return [self._clip_arm(goal_q + off) for off in offsets]

    def _make_objective(self, si: ob.SpaceInformation, fragile_mode: bool = False):
        """
        In non-fragile mode: plain path length.
        In fragile mode: path length + clearance bias.
        """
        length_obj = ob.PathLengthOptimizationObjective(si)

        if not fragile_mode:
            return length_obj

        clear_obj = ClearanceObjective(si, self._state_clearance)

        opt = ob.MultiOptimizationObjective(si)
        opt.addObjective(length_obj, 1.0)
        opt.addObjective(clear_obj, 3.0)
        return opt

    # ------------------------------------------------------------------
    # Collision checking
    # ------------------------------------------------------------------

    def _sync_from_live_data(self) -> None:
        self.plan_data.qpos[:] = self.live_data.qpos[:]
        self.plan_data.qvel[:] = self.live_data.qvel[:]
        mujoco.mj_forward(self.model, self.plan_data)

    def _state_clearance(self, state) -> float:
        """
        Conservative clearance proxy:
        minimum Euclidean distance between robot body centers and environment body centers.
        This is a biasing signal, not a formal safety proof.
        """
        q = self._state_to_q(state)

        self.plan_data.qpos[:] = self.live_data.qpos[:]
        self.plan_data.qvel[:] = 0.0
        self.plan_data.qpos[self.arm_qpos_adr] = q
        mujoco.mj_forward(self.model, self.plan_data)

        return self.collision_policy.minimum_body_center_clearance(self.plan_data)

    def _is_state_valid(self, state) -> bool:
        q = self._state_to_q(state)
        self._last_invalid_reason = None

        # The planning start state is the live arm configuration — it is
        # physically realized and must never be rejected regardless of any
        # contacts that IK may have left behind. All waypoints and the goal
        # still get the full collision check.
        if (
            self._planning_start_q is not None
            and np.allclose(q, self._planning_start_q, atol=1e-4)
        ):
            return True

        # Joint limits check
        if np.any(q < (self.lower - 1e-9)) or np.any(q > (self.upper + 1e-9)):
            self._last_invalid_reason = "joint_limit"
            return False

        # Write candidate joint state into the planning copy of the sim.
        self.plan_data.qpos[:] = self.live_data.qpos[:]
        self.plan_data.qvel[:] = 0.0
        self.plan_data.qpos[self.arm_qpos_adr] = q
        mujoco.mj_forward(self.model, self.plan_data)

        report = self.collision_policy.check_contacts(self.plan_data)
        if not report.valid:
            self._last_invalid_reason = report.reason
            return False

        return True


def make_default_panda_planner(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    planner_name: str = "BITstar",
    fragile_planner_name: str = "BITstar",
    time_limit: float = 2.0,
    state_validity_resolution: float = 0.005,
    sampler_range: float = 0.08,
    waypoint_step: float = 0.015,
    goal_tolerance: float = 1e-3,
    robot_body_names: Optional[Sequence[str]] = None,
    arm_joint_names: Optional[Sequence[str]] = None,
) -> PandaOMPLPlanner:
    return PandaOMPLPlanner(
        model=model,
        data=data,
        robot_body_names=robot_body_names,
        arm_joint_names=arm_joint_names,
        config=OMPLConfig(
            planner_name=planner_name,
            fragile_planner_name=fragile_planner_name,
            time_limit=time_limit,
            state_validity_resolution=state_validity_resolution,
            sampler_range=sampler_range,
            waypoint_step=waypoint_step,
            goal_tolerance=goal_tolerance,
        ),
    )
