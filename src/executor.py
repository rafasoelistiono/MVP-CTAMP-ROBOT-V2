from __future__ import annotations

import time
import os
from typing import Optional, Sequence, Tuple

import mujoco
import mujoco.viewer
import numpy as np

from config import CONFIG
from collision_policy import CollisionPolicy
from exec_trace import log_event

try:
    from ompl_planner import make_default_panda_planner
except ImportError:
    make_default_panda_planner = None


# =============================
# LOAD SIMULATION (ONCE)
# =============================

model = mujoco.MjModel.from_xml_path(str(CONFIG.model_file))
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
log_event("SIM_LOAD", "OK", model_file=str(CONFIG.model_file), bodies=model.nbody, geoms=model.ngeom)

class _NullViewer:
    def sync(self) -> None:
        pass


if CONFIG.enable_viewer:
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.distance = 2.5
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -30
    viewer.cam.lookat[:] = [0, 0, 0.7]
else:
    viewer = _NullViewer()
log_event("VIEWER_INIT", "OK", enabled=CONFIG.enable_viewer)

# Planning-side copy. Never use the live MuJoCo state directly inside IK.
_plan_data = mujoco.MjData(model)

# =============================
# ARM SETUP
# =============================

HOME = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7854])
GRASP_READY = np.array([0.0, 0.529, 0.0, -1.98, 0.0, 2.495, -0.75])
_DESIRED_Z = np.array([0.0, 0.0, -1.0])
GRASP_OFFSET = 0.10
APPROACH_CLEARANCE = 0.30
NEAR_OBSTACLE_XY_DISTANCE = 0.18
MIN_PICK_OBSTACLE_CLEARANCE = float(os.getenv("MIN_PICK_OBSTACLE_CLEARANCE", "0.18"))
CAUTIOUS_OBSTACLE_CLEARANCE = float(os.getenv("CAUTIOUS_OBSTACLE_CLEARANCE", "0.24"))

# Elbow-up null-space reference: joint2=0.20 keeps link2 well above the table.
# Used as a secondary IK seed when the primary converges to an elbow-down
# configuration (joint2 > ~0.55) that would place link2 below z=0.80.
_ELBOW_UP_REF = np.array([0.0, 0.20, 0.0, -2.40, 0.0, 2.60, -0.75])

# When fragile objects are present, never use IK fallback for physical motion.
USE_IK_FALLBACK = CONFIG.use_ik_fallback

# Keep OMPL paths dense and the commanded motion slow.
DEFAULT_PLANNER_NAME = CONFIG.ompl_planner_name
DEFAULT_TIME_LIMIT = CONFIG.ompl_time_limit
DEFAULT_SETTLE_STEPS_PER_WP = CONFIG.settle_steps_per_waypoint
DEFAULT_FINAL_SETTLE_STEPS = CONFIG.final_settle_steps
PICK_GRIP_SEQUENCE = (0.015, 0.011, 0.008)
PICK_GRASP_OFFSET_SEQUENCE = (0.10, 0.085, 0.075)
PICK_CLEARANCE_BONUS_SEQUENCE = (0.0, 0.035, 0.06)
COMPACT_CYLINDER_PICK_GRIP_SEQUENCE = (0.014, 0.012, 0.010)
COMPACT_CYLINDER_PICK_GRASP_OFFSET_SEQUENCE = (0.105, 0.095, 0.085)
HELD_Z_THRESHOLD = 0.90
IK_PLAN_POS_ERR_LIMIT = 0.035
IK_PREGRASP_POS_ERR_LIMIT = 0.055
FAR_PICK_XY_DISTANCE = 0.74

_BASE_ROBOT_BODY_NAMES = (
    "link0",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "hand",
    "left_finger",
    "right_finger",
)

ACTIVE_ARM = os.getenv("ACTIVE_ARM", "left").strip().lower()
_planner_by_arm = {}
_live_collision_policy = None
ee_id = None
BASE_XY = np.array([-0.4, 0.0])
arm_joint_names = []
arm_qpos_adr = np.array([], dtype=int)
arm_dof_adr = np.array([], dtype=int)
arm_ranges = np.zeros((0, 2), dtype=float)
arm_ctrl_adr = np.array([], dtype=int)
finger_ctrl_adr = 7
finger_joint_name = "finger_joint1"
active_robot_body_names = _BASE_ROBOT_BODY_NAMES


def _arm_prefix(arm: str) -> str:
    normalized = (arm or "left").strip().lower()
    if normalized in {"left", "l", ""}:
        return ""
    if normalized in {"right", "r"}:
        return "right_"
    raise ValueError(f"unknown arm '{arm}', expected left/right")


def _body_exists(name: str) -> bool:
    try:
        model.body(name)
        return True
    except KeyError:
        return False


def _joint_ctrl_index(joint_name: str) -> int:
    joint_id = model.joint(joint_name).id
    for actuator_id in range(model.nu):
        if int(model.actuator_trnid[actuator_id][0]) == int(joint_id):
            return actuator_id
    raise RuntimeError(f"no actuator controls joint '{joint_name}'")


def _robot_body_names_for_prefix(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}{name}" for name in _BASE_ROBOT_BODY_NAMES)


def available_arms() -> list[str]:
    arms = ["left"]
    if _body_exists("right_link0"):
        arms.append("right")
    return arms


def set_active_arm(arm: str) -> None:
    global ACTIVE_ARM, BASE_XY, ee_id, arm_joint_names, arm_qpos_adr, arm_dof_adr
    global arm_ranges, arm_ctrl_adr, finger_ctrl_adr, finger_joint_name
    global active_robot_body_names, _live_collision_policy

    prefix = _arm_prefix(arm)
    if prefix and not _body_exists(f"{prefix}link0"):
        raise RuntimeError(f"model has no '{arm}' arm")

    ACTIVE_ARM = "right" if prefix else "left"
    active_robot_body_names = _robot_body_names_for_prefix(prefix)
    arm_joint_names = [f"{prefix}joint{i}" for i in range(1, 8)]
    finger_joint_name = f"{prefix}finger_joint1"

    ee_id = model.body(f"{prefix}hand").id
    BASE_XY = np.asarray(model.body(f"{prefix}link0").pos[:2], dtype=float)
    arm_qpos_adr = np.array([model.joint(n).qposadr[0] for n in arm_joint_names], dtype=int)
    arm_dof_adr = np.array([model.joint(n).dofadr[0] for n in arm_joint_names], dtype=int)
    arm_ranges = np.array([model.joint(n).range for n in arm_joint_names], dtype=float)
    arm_ctrl_adr = np.array([_joint_ctrl_index(n) for n in arm_joint_names], dtype=int)
    finger_ctrl_adr = _joint_ctrl_index(finger_joint_name)
    _live_collision_policy = CollisionPolicy(model, robot_body_names=active_robot_body_names)
    log_event("ARM_SELECT", "OK", arm=ACTIVE_ARM, base_xy=[round(float(v), 4) for v in BASE_XY])


def _initialize_available_arms() -> None:
    arms = available_arms()
    requested = ACTIVE_ARM if ACTIVE_ARM in arms else "left"
    for arm in arms:
        set_active_arm(arm)
        data.qpos[arm_qpos_adr] = HOME
        _set_arm_ctrl(HOME, 0.04)
    set_active_arm(requested)

# =============================
# FIND CUBES
# =============================

name_to_cube = {}
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name and (name.startswith("cube") or name.startswith("circle")):
        name_to_cube[name] = model.body(name).id

# =============================
# OPTIONAL OBSTACLE MONITORING
# =============================

_obstacle_ids: dict[str, int] = {}
for _i in range(model.nbody):
    _n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, _i)
    if _n and any(token in _n.lower() for token in ("obstacle", "vase", "glass", "ceramic")):
        _obstacle_ids[_n] = model.body(_n).id

_obstacle_init_z: dict[str, float] = {}
_obstacle_init_xy: dict[str, np.ndarray] = {}


def _init_obstacle_monitoring() -> None:
    mujoco.mj_forward(model, data)
    for name, bid in _obstacle_ids.items():
        pos = data.xpos[bid].copy()
        _obstacle_init_z[name] = float(pos[2])
        _obstacle_init_xy[name] = pos[:2].copy()
        print(f"[init] obstacle '{name}' z0={pos[2]:.4f} xy0=({pos[0]:.3f},{pos[1]:.3f})")
        log_event(
            "OBSTACLE_MONITOR",
            "INIT",
            object_id=name,
            target_xyz=[round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4)],
        )


def _check_obstacles_fallen(context: str) -> None:
    if not _obstacle_ids:
        return
    mujoco.mj_forward(model, data)
    for name, bid in _obstacle_ids.items():
        init_z = _obstacle_init_z.get(name)
        init_xy = _obstacle_init_xy.get(name)
        if init_z is None or init_xy is None:
            continue
        pos = data.xpos[bid]
        z = float(pos[2])
        xy_dist = float(np.linalg.norm(pos[:2] - init_xy))
        z_drop = init_z - z
        if z_drop > 0.06 or xy_dist > 0.08:
            cur_pos = [round(float(pos[0]), 3), round(float(pos[1]), 3), round(z, 3)]
            ee_pos = [round(float(data.xpos[ee_id][0]), 3), round(float(data.xpos[ee_id][1]), 3), round(float(data.xpos[ee_id][2]), 3)]
            print(
                f"[OBSTACLE] {name} DISPLACED during '{context}' | "
                f"pos={cur_pos}  z_drop={z_drop:.3f} m  xy_shift={xy_dist:.3f} m "
                f"| arm_ee={ee_pos}"
            )
            log_event(
                "OBSTACLE_MONITOR",
                "FATAL",
                object_id=name,
                phase=context,
                target_xyz=cur_pos,
                failure_reason="obstacle_displaced",
                z_drop=round(z_drop, 4),
                xy_shift=round(xy_dist, 4),
                ee_pos=ee_pos,
            )
            raise RuntimeError(f"fatal obstacle displacement: {name} during {context}")


def _min_obstacle_xy_distance(pos: np.ndarray) -> float:
    if not _obstacle_ids:
        return float("inf")
    mujoco.mj_forward(model, data)
    xy = np.asarray(pos[:2], dtype=float)
    distances = []
    for bid in _obstacle_ids.values():
        distances.append(float(np.linalg.norm(xy - data.xpos[bid][:2])))
    return min(distances) if distances else float("inf")


def _object_lifted(obj: str) -> tuple[bool, float]:
    body_id = name_to_cube.get(obj)
    if body_id is None:
        return False, 0.0
    mujoco.mj_forward(model, data)
    z = float(data.xpos[body_id][2])
    return z > HELD_Z_THRESHOLD, z


# =============================
# OMPL PLANNER
# =============================

_ompl_planner = None
_OMPL_AVAILABLE = CONFIG.ompl_enabled and make_default_panda_planner is not None


def _get_ompl_planner():
    global _ompl_planner
    if not _OMPL_AVAILABLE:
        return None
    planner_key = ACTIVE_ARM
    planner = _planner_by_arm.get(planner_key)
    if planner is None:
        planner = make_default_panda_planner(
            model=model,
            data=data,
            planner_name=DEFAULT_PLANNER_NAME,
            fragile_planner_name=CONFIG.ompl_fragile_planner_name,
            time_limit=DEFAULT_TIME_LIMIT,
            state_validity_resolution=CONFIG.ompl_state_validity_resolution,
            sampler_range=CONFIG.ompl_sampler_range,
            waypoint_step=CONFIG.ompl_waypoint_step,
            goal_tolerance=CONFIG.ompl_goal_tolerance,
            robot_body_names=active_robot_body_names,
            arm_joint_names=arm_joint_names,
        )
        _planner_by_arm[planner_key] = planner
    else:
        planner.sync_live_data(data)
    _ompl_planner = planner
    return planner


# =============================
# HELPERS
# =============================

def clip_arm(q: np.ndarray) -> np.ndarray:
    return np.clip(q, arm_ranges[:, 0], arm_ranges[:, 1])


def current_q() -> np.ndarray:
    return data.qpos[arm_qpos_adr].copy()


def cube_null_ref(pos: np.ndarray) -> np.ndarray:
    """GRASP_READY with joint1 aimed at the cube bearing from the arm base."""
    dx = pos[0] - BASE_XY[0]
    dy = pos[1] - BASE_XY[1]
    j1 = float(np.clip(np.arctan2(dy, dx), arm_ranges[0, 0], arm_ranges[0, 1]))
    ref = GRASP_READY.copy()
    ref[0] = j1
    return ref


def _sync_plan_data_from_live() -> None:
    _plan_data.qpos[:] = data.qpos[:]
    _plan_data.qvel[:] = data.qvel[:]
    mujoco.mj_forward(model, _plan_data)


def _set_arm_ctrl(q: np.ndarray, grip: float) -> None:
    data.ctrl[arm_ctrl_adr] = clip_arm(np.asarray(q, dtype=float))
    data.ctrl[finger_ctrl_adr] = float(grip)


def _step_sim(steps: int, q: Optional[np.ndarray] = None, grip: Optional[float] = None) -> None:
    for _ in range(steps):
        if q is not None:
            data.ctrl[arm_ctrl_adr] = clip_arm(np.asarray(q, dtype=float))
        if grip is not None:
            data.ctrl[finger_ctrl_adr] = float(grip)
        mujoco.mj_step(model, data)
        viewer.sync()


def _is_table_finger_pair(report) -> bool:
    bodies = {report.body1, report.body2}
    return "table" in bodies and any(str(body).endswith(("left_finger", "right_finger")) for body in bodies)


def _check_live_collision(
    context: str,
    ignored_body_names: Optional[Sequence[str]] = None,
    allow_start_table_finger: bool = False,
) -> bool:
    _live_collision_policy.set_ignored_bodies(ignored_body_names)
    mujoco.mj_forward(model, data)
    report = _live_collision_policy.check_contacts(data)
    if report.valid:
        return True
    if allow_start_table_finger and _is_table_finger_pair(report):
        log_event(
            "COLLISION_CHECK",
            "IGNORED_START",
            phase=context,
            failure_reason=report.reason,
            collision_pair=[report.body1, report.body2],
            ignored_body_names=list(ignored_body_names or []),
        )
        return True
    print(f"[collision] blocked during {context}: {report.reason}")
    log_event(
        "COLLISION_CHECK",
        "BLOCKED",
        phase=context,
        failure_reason=report.reason,
        collision_pair=[report.body1, report.body2],
        ignored_body_names=list(ignored_body_names or []),
    )
    return False


def _finger_pos() -> float:
    finger_qposadr = model.joint(finger_joint_name).qposadr[0]
    return float(data.qpos[finger_qposadr])


def go_to_cube_ready(cube_pos, steps=400):
    """Deprecated for fragile scenes. Kept only for backward compatibility."""
    target = cube_null_ref(np.asarray(cube_pos, dtype=float))
    for _ in range(steps):
        data.ctrl[arm_ctrl_adr] += 0.01 * (target - data.ctrl[arm_ctrl_adr])
        data.ctrl[finger_ctrl_adr] = 0.04
        mujoco.mj_step(model, data)
        viewer.sync()


# =============================
# IK SOLVER (planning only; does NOT move the live robot)
# =============================

def _ik_solve_to(
    target_xyz: Sequence[float],
    null_ref: Optional[np.ndarray] = None,
    q_seed: Optional[Sequence[float]] = None,
    steps: int = 600,
    pos_tol: float = 0.008,
    ori_tol: float = 0.20,
) -> Tuple[np.ndarray, dict]:
    if null_ref is None:
        null_ref = GRASP_READY

    q_target = np.array(q_seed if q_seed is not None else current_q(), dtype=float).reshape(-1)
    target_xyz = np.array(target_xyz, dtype=float).reshape(3)

    info = {
        "converged": False,
        "pos_err_norm": None,
        "ori_err_norm": None,
        "iters": 0,
    }

    for it in range(steps):
        _plan_data.qpos[:] = data.qpos[:]
        _plan_data.qvel[:] = 0.0
        _plan_data.qpos[arm_qpos_adr] = q_target
        mujoco.mj_forward(model, _plan_data)

        pos_error = target_xyz - _plan_data.xpos[ee_id]
        R = _plan_data.xmat[ee_id].reshape(3, 3)
        ori_error = np.cross(R[:, 2], _DESIRED_Z)

        pos_norm = float(np.linalg.norm(pos_error))
        ori_norm = float(np.linalg.norm(ori_error))

        info["pos_err_norm"] = pos_norm
        info["ori_err_norm"] = ori_norm
        info["iters"] = it + 1

        if pos_norm < pos_tol and ori_norm < ori_tol:
            info["converged"] = True
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, _plan_data, jacp, jacr, ee_id)
        J_pos = jacp[:, arm_dof_adr]
        J_rot = jacr[:, arm_dof_adr]

        w = 0.5
        J6 = np.vstack([J_pos, w * J_rot])
        err6 = np.concatenate([pos_error, w * ori_error])

        lam = 0.05
        JJT = J6 @ J6.T + lam * np.eye(6)
        J_pinv = J6.T @ np.linalg.solve(JJT, np.eye(6))

        dq_task = J_pinv @ err6
        null_proj = np.eye(7) - J_pinv @ J6
        dq_null = null_proj @ (np.asarray(null_ref, dtype=float) - q_target) * 0.05

        dq = np.clip(dq_task + dq_null, -0.08, 0.08)
        q_target = clip_arm(q_target + 0.015 * dq)

    return clip_arm(q_target), info

def _solve_safe_goal_candidates(target_xyz, base_null_ref, label=""):
    planner = _get_ompl_planner()
    candidate_refs = [
        base_null_ref.copy(),
        _ELBOW_UP_REF.copy(),
    ]

    # Small null-space variations help escape one bad IK basin.
    for delta in [0.12, -0.12]:
        ref = base_null_ref.copy()
        ref[1] = np.clip(ref[1] + delta, arm_ranges[1, 0], arm_ranges[1, 1])
        candidate_refs.append(ref)

    for null_ref in candidate_refs:
        goal_q, info = _ik_solve_to(target_xyz, null_ref=null_ref, steps=800)
        if not info["converged"]:
            continue
        if planner is None or planner.is_state_valid_q(goal_q):
            return goal_q, info

    return None, None


def _target_xyz_candidates(target_xyz: Sequence[float], label: str) -> list[np.ndarray]:
    base = np.asarray(target_xyz, dtype=float).reshape(3)
    candidates = [base]
    if label.startswith("pick("):
        xy = base[:2]
        base_distance = float(np.linalg.norm(xy - BASE_XY))
        radial = BASE_XY - xy
        radial_norm = float(np.linalg.norm(radial))
        radial_dir = radial / radial_norm if radial_norm > 1e-6 else np.array([-1.0, 0.0])
        tangent_dir = np.array([-radial_dir[1], radial_dir[0]])

        xy_offsets = [
            np.array([0.025, 0.0, 0.0]),
            np.array([-0.025, 0.0, 0.0]),
            np.array([0.0, 0.025, 0.0]),
            np.array([0.0, -0.025, 0.0]),
        ]
        candidates.extend(base + offset for offset in xy_offsets)

        # For far objects, approach from the robot-facing side and avoid an
        # unnecessarily high vertical pregrasp that pushes IK outside workspace.
        if base_distance > FAR_PICK_XY_DISTANCE:
            for radial_offset in (0.045, 0.075):
                candidates.append(base + np.r_[radial_dir * radial_offset, 0.0])
            if "pregrasp" in label:
                for z_drop in (0.045, 0.075):
                    candidates.append(base + np.array([0.0, 0.0, -z_drop]))
                    candidates.append(base + np.r_[radial_dir * 0.055, -z_drop])

        # Cylinders are more forgiving with a radial side-biased contact than a
        # pure top-center target, especially near workspace limits.
        if label.startswith("pick(circle"):
            for radial_offset in (0.035, 0.06, 0.085):
                candidates.append(base + np.r_[radial_dir * radial_offset, 0.0])
            for side_offset in (0.035, -0.035):
                candidates.append(base + np.r_[radial_dir * 0.055 + tangent_dir * side_offset, 0.0])

        if "grasp" in label and "pregrasp" not in label:
            candidates.extend(base + offset + np.array([0.0, 0.0, -0.01]) for offset in xy_offsets[:2])
    return candidates


def _ik_pos_limit_for_label(label: str) -> float:
    if label.startswith("pick(") and "pregrasp" in label:
        return IK_PREGRASP_POS_ERR_LIMIT
    return IK_PLAN_POS_ERR_LIMIT


def _null_ref_candidates(null_ref: Optional[np.ndarray]) -> list[np.ndarray]:
    base_ref = np.asarray(null_ref if null_ref is not None else GRASP_READY, dtype=float).copy()
    refs = [base_ref, _ELBOW_UP_REF.copy()]
    for delta in (0.18, -0.18):
        ref = base_ref.copy()
        ref[1] = np.clip(ref[1] + delta, arm_ranges[1, 0], arm_ranges[1, 1])
        refs.append(ref)
    return refs


def _select_ik_goal(target_xyz: Sequence[float], null_ref: Optional[np.ndarray], label: str):
    planner = _get_ompl_planner()
    best = None
    best_score = float("inf")
    pos_limit = _ik_pos_limit_for_label(label)

    for candidate_idx, xyz in enumerate(_target_xyz_candidates(target_xyz, label)):
        for ref_idx, ref in enumerate(_null_ref_candidates(null_ref)):
            q, info = _ik_solve_to(xyz, null_ref=ref, q_seed=ref if ref_idx else None)
            pos_err = float(info["pos_err_norm"] or float("inf"))
            ori_err = float(info["ori_err_norm"] or float("inf"))
            score = pos_err + 0.25 * ori_err
            log_event(
                "IK_CANDIDATE",
                "OK" if pos_err <= IK_PLAN_POS_ERR_LIMIT else "REJECT",
                phase=label,
                target_xyz=[round(float(v), 4) for v in xyz],
                pos_err=round(pos_err, 5),
                ori_err=round(ori_err, 5),
                candidate_idx=candidate_idx,
                ref_idx=ref_idx,
                pos_limit=pos_limit,
            )

            if score < best_score:
                best = (q, info, xyz)
                best_score = score

            if pos_err > pos_limit:
                continue
            if planner is not None and not planner.is_state_valid_q(q):
                log_event("IK_CANDIDATE", "INVALID_GOAL", phase=label, candidate_idx=candidate_idx, ref_idx=ref_idx)
                continue
            return q, info, xyz

    if best is None:
        return None, None, None
    q, info, xyz = best
    if float(info["pos_err_norm"] or float("inf")) > pos_limit:
        return None, info, xyz
    if planner is not None and not planner.is_state_valid_q(q):
        log_event("IK_CANDIDATE", "NO_VALID_GOAL", phase=label, pos_limit=pos_limit)
        return None, info, xyz
    return q, info, xyz

# =============================
# OMPL EXECUTION
# =============================

def _execute_joint_trajectory(
    traj: np.ndarray,
    grip: float,
    ignored_body_names: Optional[Sequence[str]] = None,
    settle_steps_per_wp: int = DEFAULT_SETTLE_STEPS_PER_WP,
    final_settle_steps: int = DEFAULT_FINAL_SETTLE_STEPS,
) -> bool:
    if traj is None or len(traj) == 0:
        log_event("TRAJECTORY_EXEC", "SKIP", waypoints=0, grip=grip)
        return True

    log_event(
        "TRAJECTORY_EXEC",
        "START",
        waypoints=len(traj),
        grip=grip,
        ignored_body_names=list(ignored_body_names or []),
        settle_steps_per_wp=settle_steps_per_wp,
        final_settle_steps=final_settle_steps,
    )
    started = time.perf_counter()
    for waypoint_index, q in enumerate(traj):
        q = clip_arm(np.asarray(q, dtype=float))
        if waypoint_index in {0, len(traj) - 1} or waypoint_index % max(1, len(traj) // 4) == 0:
            log_event(
                "TRAJECTORY_WAYPOINT",
                "EXEC",
                waypoints=f"{waypoint_index + 1}/{len(traj)}",
                grip=grip,
                q=[round(float(v), 4) for v in q],
            )
        for _ in range(settle_steps_per_wp):
            _set_arm_ctrl(q, grip)
            mujoco.mj_step(model, data)
            viewer.sync()
            if not _check_live_collision(
                context=f"trajectory waypoint {waypoint_index}",
                ignored_body_names=ignored_body_names,
                allow_start_table_finger=waypoint_index == 0,
            ):
                log_event(
                    "TRAJECTORY_EXEC",
                    "FAILED",
                    waypoints=len(traj),
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    failure_reason=f"collision_at_waypoint_{waypoint_index}",
                )
                return False

    if final_settle_steps > 0:
        final_q = clip_arm(np.asarray(traj[-1], dtype=float))
        for _ in range(final_settle_steps):
            _set_arm_ctrl(final_q, grip)
            mujoco.mj_step(model, data)
            viewer.sync()
            if not _check_live_collision(
                context="trajectory final settle",
                ignored_body_names=ignored_body_names,
            ):
                log_event(
                    "TRAJECTORY_EXEC",
                    "FAILED",
                    waypoints=len(traj),
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    failure_reason="collision_during_final_settle",
                )
                return False

    log_event(
        "TRAJECTORY_EXEC",
        "OK",
        waypoints=len(traj),
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    return True


def _move_with_ompl(
    goal_q: Sequence[float],
    grip: float,
    ignored_body_names: Optional[Sequence[str]] = None,
    planner_name: str = DEFAULT_PLANNER_NAME,
    time_limit: float = DEFAULT_TIME_LIMIT,
    label: str = "",
    settle_steps_per_wp: int = DEFAULT_SETTLE_STEPS_PER_WP,
    final_settle_steps: int = DEFAULT_FINAL_SETTLE_STEPS,
) -> bool:
    goal_q = np.asarray(goal_q, dtype=float).reshape(7)
    start_q = current_q()

    if not _OMPL_AVAILABLE:
        print("[exec][OMPL] unavailable")
        log_event("OMPL_PLAN", "UNAVAILABLE", phase=label, failure_reason="ompl_unavailable")
        return False

    planner = _get_ompl_planner()
    if planner is None:
        print("[exec][OMPL] planner init failed")
        log_event("OMPL_PLAN", "FAILED", phase=label, failure_reason="planner_init_failed")
        return False

    try:
        log_event(
            "OMPL_PLAN",
            "START",
            phase=label,
            planner=planner_name,
            grip=grip,
            ignored_body_names=list(ignored_body_names or []),
            time_limit=time_limit,
            start_q=[round(float(v), 4) for v in start_q],
            goal_q=[round(float(v), 4) for v in goal_q],
        )
        started = time.perf_counter()
        traj, info = planner.plan(
            start_q=start_q,
            goal_q=goal_q,
            time_limit=time_limit,
            planner_name=planner_name,
            ignored_body_names=ignored_body_names,
            fragile_mode=True,
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        if traj is None:
            print(f"[exec][OMPL] planning failed: {info}")
            log_event(
                "OMPL_PLAN",
                "FAILED",
                phase=label,
                planner=info.get("planner_name", planner_name),
                duration_ms=duration_ms,
                failure_reason="no_solution_path",
                goal_attempts=info.get("goal_attempts"),
            )
            return False

        print(
            f"[exec][OMPL] solved={info.get('solved')} "
            f"planner={info.get('planner_name')} waypoints={info.get('num_waypoints')}"
        )
        log_event(
            "OMPL_PLAN",
            "OK",
            phase=label,
            planner=info.get("planner_name"),
            waypoints=info.get("num_waypoints"),
            duration_ms=duration_ms,
            path_length=round(float(info.get("path_length_joint_space", 0.0)), 4),
            selected_goal_q=info.get("selected_goal_q"),
        )
        return _execute_joint_trajectory(
            traj,
            grip=grip,
            ignored_body_names=ignored_body_names,
            settle_steps_per_wp=settle_steps_per_wp,
            final_settle_steps=final_settle_steps,
        )
    except Exception as e:
        print(f"[exec][OMPL] error: {e}")
        log_event("OMPL_PLAN", "ERROR", phase=label, failure_reason=str(e))
        return False


def _move_pose_safe(
    target_xyz: Sequence[float],
    grip: float,
    null_ref: Optional[np.ndarray] = None,
    ignored_body_names: Optional[Sequence[str]] = None,
    label: str = "",
    cautious_motion: bool = False,
) -> bool:
    log_event(
        "MOVE_POSE",
        "START",
        phase=label,
        target_xyz=[round(float(v), 4) for v in target_xyz],
        grip=grip,
        ignored_body_names=list(ignored_body_names or []),
    )
    selected_goal_q, selected_ik_info, selected_xyz = _select_ik_goal(target_xyz, null_ref, label)
    if selected_goal_q is None:
        pos_err = float((selected_ik_info or {}).get("pos_err_norm") or float("inf"))
        ori_err = float((selected_ik_info or {}).get("ori_err_norm") or float("inf"))
        pos_limit = _ik_pos_limit_for_label(label)
        print(
            f"[exec][IK] reject {label or 'pose'}: "
            f"best_pos={pos_err:.4f} best_ori={ori_err:.4f} limit={pos_limit:.3f}"
        )
        log_event(
            "IK_SOLVE",
            "FAILED",
            phase=label,
            target_xyz=[round(float(v), 4) for v in target_xyz],
            failure_reason="ik_error_above_plan_limit",
            pos_err=round(pos_err, 5) if np.isfinite(pos_err) else None,
            ori_err=round(ori_err, 5) if np.isfinite(ori_err) else None,
            pos_limit=pos_limit,
        )
        return False

    log_event(
        "IK_SOLVE",
        "OK",
        phase=label,
        target_xyz=[round(float(v), 4) for v in selected_xyz],
        pos_err=round(float(selected_ik_info["pos_err_norm"] or 0.0), 5),
        ori_err=round(float(selected_ik_info["ori_err_norm"] or 0.0), 5),
        iterations=selected_ik_info["iters"],
    )
    ok = _move_with_ompl(
        goal_q=selected_goal_q,
        grip=grip,
        ignored_body_names=ignored_body_names,
        planner_name=DEFAULT_PLANNER_NAME,
        time_limit=DEFAULT_TIME_LIMIT * (1.7 if cautious_motion else 1.0),
        label=label,
        settle_steps_per_wp=DEFAULT_SETTLE_STEPS_PER_WP * (2 if cautious_motion else 1),
        final_settle_steps=DEFAULT_FINAL_SETTLE_STEPS * (2 if cautious_motion else 1),
    )
    if ok:
        log_event("MOVE_POSE", "OK", phase=label)
        return True
    if USE_IK_FALLBACK:
        print(f"[exec] OMPL failed for {label or 'pose'}; using IK fallback")
        log_event("MOVE_POSE", "IK_FALLBACK", phase=label)
        move_ee_to(selected_xyz, grip=grip, steps=300, null_ref=null_ref)
        return True
    print(f"[exec] OMPL failed for {label or 'pose'}; no fallback in fragile-scene mode")
    log_event("MOVE_POSE", "FAILED", phase=label, failure_reason="ompl_failed_no_fallback")
    return False

    goal_q, ik_info = _ik_solve_to(target_xyz, null_ref=null_ref)
    log_event(
        "IK_SOLVE",
        "OK" if ik_info["converged"] else "WARN",
        phase=label,
        target_xyz=[round(float(v), 4) for v in target_xyz],
        duration_ms=None,
        pos_err=round(float(ik_info["pos_err_norm"] or 0.0), 5),
        ori_err=round(float(ik_info["ori_err_norm"] or 0.0), 5),
        iterations=ik_info["iters"],
    )
    if not ik_info["converged"]:
        print(
            f"[exec][IK] warning for {label or 'pose'}: "
            f"pos={ik_info['pos_err_norm']:.4f} ori={ik_info['ori_err_norm']:.4f} iters={ik_info['iters']}"
        )

    # If primary IK converged to an elbow-down config (joint2 > 0.55 → link2
    # may drop below the table at z=0.80) or did not converge, try an elbow-up
    # seed.  A large joint2 happens for close targets (x≈0.2) where the default
    # GRASP_READY null-ref drives joint2 to ~0.65, putting link2 at z≈0.74 —
    # below the table — so OMPL rejects every goal candidate as invalid_goal.
    _EU_J2_THRESHOLD = 0.55
    if goal_q[1] > _EU_J2_THRESHOLD or not ik_info["converged"]:
        eu_ref = _ELBOW_UP_REF.copy()
        if null_ref is not None:
            eu_ref[0] = null_ref[0]  # match joint1 direction to target
        goal_q_eu, ik_info_eu = _ik_solve_to(target_xyz, null_ref=eu_ref, q_seed=eu_ref)
        log_event(
            "IK_ELBOW_UP",
            "OK" if ik_info_eu["converged"] else "FAILED",
            phase=label,
            pos_err=round(float(ik_info_eu["pos_err_norm"] or 0.0), 5),
            ori_err=round(float(ik_info_eu["ori_err_norm"] or 0.0), 5),
            iterations=ik_info_eu["iters"],
        )
        if ik_info_eu["converged"] and (
            not ik_info["converged"]
            or ik_info_eu["pos_err_norm"] < ik_info["pos_err_norm"]
        ):
            goal_q, ik_info = goal_q_eu, ik_info_eu
            print(
                f"[exec][IK] elbow-up solution for {label or 'pose'}: "
                f"pos={ik_info['pos_err_norm']:.4f}"
            )

    ok = _move_with_ompl(
        goal_q=goal_q,
        grip=grip,
        ignored_body_names=ignored_body_names,
        planner_name=DEFAULT_PLANNER_NAME,
        time_limit=DEFAULT_TIME_LIMIT * (1.7 if cautious_motion else 1.0),
        label=label,
        settle_steps_per_wp=DEFAULT_SETTLE_STEPS_PER_WP * (2 if cautious_motion else 1),
        final_settle_steps=DEFAULT_FINAL_SETTLE_STEPS * (2 if cautious_motion else 1),
    )

    if ok:
        log_event("MOVE_POSE", "OK", phase=label)
        return True

    if USE_IK_FALLBACK:
        print(f"[exec] OMPL failed for {label or 'pose'}; using IK fallback")
        log_event("MOVE_POSE", "IK_FALLBACK", phase=label)
        move_ee_to(target_xyz, grip=grip, steps=300, null_ref=null_ref)
        return True

    print(f"[exec] OMPL failed for {label or 'pose'}; no fallback in fragile-scene mode")
    log_event("MOVE_POSE", "FAILED", phase=label, failure_reason="ompl_failed_no_fallback")
    return False


# =============================
# SAFETY / GRIPPER HELPERS
# =============================

def _recover_to_safe_hover() -> None:
    """Move away from table/contact states before retrying another pick."""
    log_event("RECOVERY", "START", phase="safe_hover")
    ok = _move_with_ompl(
        goal_q=GRASP_READY,
        grip=0.04,
        ignored_body_names=None,
        planner_name=DEFAULT_PLANNER_NAME,
        time_limit=max(DEFAULT_TIME_LIMIT, 2.0),
        label="recovery safe_hover",
        settle_steps_per_wp=DEFAULT_SETTLE_STEPS_PER_WP,
        final_settle_steps=DEFAULT_FINAL_SETTLE_STEPS,
    )
    if ok:
        log_event("RECOVERY", "OK", phase="safe_hover")
        return

    # Last-resort controller recovery. This is only used after releasing an
    # object, where the main risk is repeatedly starting near table contact.
    log_event("RECOVERY", "DIRECT_CTRL", phase="safe_hover")
    _step_sim(260, q=GRASP_READY, grip=0.04)
    log_event("RECOVERY", "OK", phase="safe_hover_direct")


def _move_to_grasp_ready(reason: str, grip: float = 0.04) -> bool:
    if float(np.linalg.norm(current_q() - GRASP_READY)) < 0.05:
        return True
    log_event("TRANSIT", "START", phase=reason, target="GRASP_READY")
    ok = _move_with_ompl(
        goal_q=GRASP_READY,
        grip=grip,
        ignored_body_names=None,
        planner_name=DEFAULT_PLANNER_NAME,
        time_limit=max(DEFAULT_TIME_LIMIT, 3.0),
        label=f"transit {reason}",
        settle_steps_per_wp=DEFAULT_SETTLE_STEPS_PER_WP,
        final_settle_steps=DEFAULT_FINAL_SETTLE_STEPS,
    )
    log_event("TRANSIT", "OK" if ok else "FAILED", phase=reason, target="GRASP_READY")
    return ok


def set_grip(target: float, steps: int = 200):
    log_event("GRIPPER", "SET", grip=target, steps=steps)
    for _ in range(steps):
        data.ctrl[finger_ctrl_adr] = float(target)
        mujoco.mj_step(model, data)
        viewer.sync()


def drop():
    """Emergency release at the current arm position."""
    global _held_grip_target
    log_event("DROP", "START", object_id=_held_object_name)
    set_grip(0.04, steps=300)
    for _ in range(120):
        mujoco.mj_step(model, data)
        viewer.sync()
    _recover_to_safe_hover()
    _held_grip_target = 0.015
    log_event("DROP", "OK")


# =============================
# STARTUP WARMUP
# =============================

_initialize_available_arms()

# Phase 1: HOME.
log_event("WARMUP", "START", phase="home")
data.qpos[arm_qpos_adr] = HOME
_set_arm_ctrl(HOME, 0.04)
for _ in range(250):
    mujoco.mj_step(model, data)
    viewer.sync()
log_event("WARMUP", "OK", phase="home")

# Phase 2: Transition to a grasp-friendly pose.
log_event("WARMUP", "START", phase="grasp_ready")
for _ in range(350):
    data.ctrl[arm_ctrl_adr] += 0.01 * (GRASP_READY - data.ctrl[arm_ctrl_adr])
    data.ctrl[finger_ctrl_adr] = 0.04
    mujoco.mj_step(model, data)
    viewer.sync()
log_event("WARMUP", "OK", phase="grasp_ready")

_init_obstacle_monitoring()


# =============================
# HIGH-LEVEL ACTIONS
# =============================

_held_object_name: Optional[str] = None
_held_grip_target: float = 0.015
_pick_call_counts: dict[str, int] = {}


def pick(obj):
    global _held_object_name, _held_grip_target

    print(f"[exec] pick({obj})")
    log_event("PICK", "START", object_id=obj)
    if obj not in name_to_cube:
        print(f"[exec] unknown object: {obj}")
        log_event("PICK", "FAILED", object_id=obj, failure_reason="unknown_object")
        return

    if not _move_to_grasp_ready(f"before pick({obj})", grip=0.04):
        log_event("PICK", "FAILED", object_id=obj, phase="transit", failure_reason="move_to_grasp_ready_failed")
        return

    call_count = _pick_call_counts.get(obj, 0)
    profile_index = min(call_count, len(PICK_GRIP_SEQUENCE) - 1)
    _pick_call_counts[obj] = call_count + 1

    grip_target = PICK_GRIP_SEQUENCE[profile_index]
    grasp_offset = PICK_GRASP_OFFSET_SEQUENCE[profile_index]
    clearance_bonus = PICK_CLEARANCE_BONUS_SEQUENCE[profile_index]
    is_circle = obj.startswith("circle")
    if is_circle:
        grip_target = COMPACT_CYLINDER_PICK_GRIP_SEQUENCE[profile_index]
        grasp_offset = COMPACT_CYLINDER_PICK_GRASP_OFFSET_SEQUENCE[profile_index]
    log_event(
        "PICK_PROFILE",
        "SELECT",
        object_id=obj,
        profile_index=profile_index,
        grip=grip_target,
        grasp_offset=grasp_offset,
        clearance_bonus=clearance_bonus,
    )

    cube_id = name_to_cube[obj]

    # Let the system settle before planning a new motion.
    _step_sim(80, grip=0.04)
    mujoco.mj_forward(model, data)
    cube_pos = data.xpos[cube_id].copy()
    obstacle_distance = _min_obstacle_xy_distance(cube_pos)
    approach_clearance = APPROACH_CLEARANCE + clearance_bonus
    cautious_motion = False
    if obstacle_distance < MIN_PICK_OBSTACLE_CLEARANCE:
        print(
            f"[exec][OBSTACLE_AVOID] cancel pick({obj}): "
            f"distance={obstacle_distance:.3f}m threshold={MIN_PICK_OBSTACLE_CLEARANCE:.3f}m"
        )
        log_event(
            "PICK",
            "FAILED",
            object_id=obj,
            phase="precheck",
            failure_reason="object_too_close_to_obstacle",
            obstacle_distance=round(float(obstacle_distance), 4),
            threshold=MIN_PICK_OBSTACLE_CLEARANCE,
        )
        return
    if obstacle_distance < CAUTIOUS_OBSTACLE_CLEARANCE:
        cautious_motion = True
        approach_clearance += 0.06
        print(
            f"[exec][OBSTACLE_AVOID] {obj} near obstacle "
            f"distance={obstacle_distance:.3f}m; using cautious high-clearance approach"
        )
        log_event(
            "OBSTACLE_AVOID",
            "NEAR_CAUTIOUS",
            object_id=obj,
            obstacle_distance=obstacle_distance,
            approach_clearance=approach_clearance,
        )
    else:
        log_event(
            "OBSTACLE_AVOID",
            "CLEAR",
            object_id=obj,
            obstacle_distance=obstacle_distance if np.isfinite(obstacle_distance) else None,
            approach_clearance=approach_clearance,
        )

    cube_ref = cube_null_ref(cube_pos)
    object_xy_distance = float(np.linalg.norm(cube_pos[:2] - BASE_XY))
    pregrasp_clearance = approach_clearance
    if object_xy_distance > FAR_PICK_XY_DISTANCE:
        pregrasp_clearance = min(pregrasp_clearance, 0.24)
        log_event(
            "PICK_PROFILE",
            "FAR_PREGRASP",
            object_id=obj,
            obstacle_distance=obstacle_distance if np.isfinite(obstacle_distance) else None,
            approach_clearance=pregrasp_clearance,
            xy_distance=round(object_xy_distance, 4),
        )
    pregrasp_xyz = cube_pos + np.array([0.0, 0.0, pregrasp_clearance])
    grasp_xyz = cube_pos + np.array([0.0, 0.0, grasp_offset])
    lift_xyz = cube_pos + np.array([0.0, 0.0, approach_clearance])

    # 1) Move above the cube.
    if not _move_pose_safe(
        pregrasp_xyz,
        grip=0.04,
        null_ref=cube_ref,
        ignored_body_names=None,
        label=f"pick({obj}) pregrasp",
        cautious_motion=cautious_motion,
    ):
        log_event("PICK", "FAILED", object_id=obj, phase="pregrasp", failure_reason="move_pregrasp_failed")
        return

    # 2) Move to the grasp pose while ignoring the target cube only.
    if not _move_pose_safe(
        grasp_xyz,
        grip=0.04,
        null_ref=cube_ref,
        ignored_body_names=[obj],
        label=f"pick({obj}) grasp",
        cautious_motion=cautious_motion,
    ):
        log_event("PICK", "FAILED", object_id=obj, phase="grasp", failure_reason="move_grasp_failed")
        return

    # 3) Close gripper and let contact settle.
    log_event("PICK", "GRIP_CLOSE", object_id=obj, phase="close_gripper")
    set_grip(grip_target, steps=320 if cautious_motion else 260)
    for _ in range(110 if cautious_motion else 70):
        mujoco.mj_step(model, data)
        viewer.sync()

    _held_object_name = obj
    _held_grip_target = grip_target

    # 4) Lift straight up using OMPL, still ignoring the target cube.
    if not _move_pose_safe(
        lift_xyz,
        grip=grip_target,
        null_ref=cube_ref,
        ignored_body_names=[obj],
        label=f"pick({obj}) lift",
        cautious_motion=cautious_motion,
    ):
        # If lift planning fails after grasping, release immediately and let the
        # closed-loop system replan from the table state.
        drop()
        _held_object_name = None
        log_event("PICK", "FAILED", object_id=obj, phase="lift", failure_reason="move_lift_failed")
        return

    _check_obstacles_fallen(f"pick({obj})")
    lifted, z = _object_lifted(obj)
    if not lifted:
        print(f"[exec][PICK_RETRY] {obj} not lifted after grasp z={z:.3f}; release and retry")
        log_event("PICK", "FAILED", object_id=obj, phase="lift_check", failure_reason="object_not_lifted", z=round(z, 4))
        drop()
        _held_object_name = None
        return
    log_event("PICK", "OK", object_id=obj)



def place(x, y, obj=None):
    global _held_object_name, _held_grip_target

    print(f"[exec] place({x:.3f}, {y:.3f})")
    log_event("PLACE", "START", object_id=obj or _held_object_name, target_xyz=[round(float(x), 4), round(float(y), 4), 0.83])
    if _held_object_name is None:
        print("[exec] no cube is currently held")
        log_event("PLACE", "FAILED", failure_reason="no_object_held")
        return

    obj = _held_object_name

    # Settled cube centre after release.
    place_pos = np.array([x, y, 0.83])
    release_pos = np.array([x, y, 0.89])
    place_ref = cube_null_ref(place_pos)
    obstacle_distance = _min_obstacle_xy_distance(place_pos)
    approach_clearance = APPROACH_CLEARANCE
    cautious_motion = False
    if obstacle_distance < CAUTIOUS_OBSTACLE_CLEARANCE:
        cautious_motion = True
        approach_clearance += 0.06
        print(
            f"[exec][OBSTACLE_AVOID] place target near obstacle "
            f"distance={obstacle_distance:.3f}m; using cautious high-clearance preplace"
        )
        log_event(
            "OBSTACLE_AVOID",
            "NEAR_CAUTIOUS",
            object_id=obj,
            phase="place",
            obstacle_distance=obstacle_distance,
            approach_clearance=approach_clearance,
        )

    preplace_xyz = place_pos + np.array([0.0, 0.0, approach_clearance])
    release_xyz = release_pos + np.array([0.0, 0.0, GRASP_OFFSET])
    retreat_xyz = release_pos + np.array([0.0, 0.0, approach_clearance])

    # 1) Move above the goal while still holding the cube.
    if not _move_pose_safe(
        preplace_xyz,
        grip=_held_grip_target,
        null_ref=place_ref,
        ignored_body_names=[obj],
        label=f"place({x:.3f}, {y:.3f}) preplace",
        cautious_motion=cautious_motion,
    ):
        drop()
        _held_object_name = None
        log_event("PLACE", "FAILED", object_id=obj, phase="preplace", failure_reason="move_preplace_failed")
        return

    # 2) Move to the release pose.
    if not _move_pose_safe(
        release_xyz,
        grip=_held_grip_target,
        null_ref=place_ref,
        ignored_body_names=[obj],
        label=f"place({x:.3f}, {y:.3f}) release",
        cautious_motion=cautious_motion,
    ):
        drop()
        _held_object_name = None
        log_event("PLACE", "FAILED", object_id=obj, phase="release", failure_reason="move_release_failed")
        return

    # 3) Let the arm settle before opening.
    log_event("PLACE", "SETTLE_BEFORE_OPEN", object_id=obj, steps=120)
    for _ in range(120):
        mujoco.mj_step(model, data)
        viewer.sync()

    print(f"[exec] finger before open: {_finger_pos():.4f}")
    log_event("PLACE", "GRIP_OPEN", object_id=obj, finger_before=round(_finger_pos(), 4))
    set_grip(0.04, steps=450)
    print(f"[exec] finger after open:  {_finger_pos():.4f}")
    log_event("PLACE", "GRIP_OPENED", object_id=obj, finger_after=round(_finger_pos(), 4))

    # 4) Let the cube fall and settle.
    for _ in range(160):
        mujoco.mj_step(model, data)
        viewer.sync()

    # 5) Retreat upward using OMPL.
    if not _move_pose_safe(
        retreat_xyz,
        grip=0.04,
        null_ref=place_ref,
        ignored_body_names=[obj],
        label=f"place({x:.3f}, {y:.3f}) retreat",
        cautious_motion=cautious_motion,
    ):
        # Retreat failure after release is not fatal, but we still keep the
        # controller in a safe open state.
        print(f"[exec] retreat failed after place({x:.3f}, {y:.3f})")
        log_event("PLACE", "RETREAT_FAILED", object_id=obj, failure_reason="retreat_failed_after_release")

    _held_object_name = None
    _held_grip_target = 0.015
    _move_to_grasp_ready(f"after place({x:.3f}, {y:.3f})", grip=0.04)
    _check_obstacles_fallen(f"place({x:.3f}, {y:.3f})")
    log_event("PLACE", "OK", object_id=obj, target_xyz=[round(float(x), 4), round(float(y), 4), 0.83])

