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
from ik_diagnostics import (
    IKAttemptResult,
    IK_GOAL_STATE_INVALID,
    IK_SUCCESS,
    IK_UNREACHABLE,
    OMPL_TIMEOUT,
    classify_ik_attempt,
    joint_limits_valid,
    rank_ik_attempts,
)

try:
    from ompl_planner import make_default_panda_planner
except ImportError:
    make_default_panda_planner = None

try:
    import pinocchio as pin
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ImportError as _pinocchio_import_error:
    pin = None
    load_robot_description = None
    _PINOCCHIO_IMPORT_ERROR = _pinocchio_import_error
else:
    _PINOCCHIO_IMPORT_ERROR = None


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
PICK_GRASP_OFFSET_SEQUENCE = (0.10, 0.10, 0.10)
PICK_CLEARANCE_BONUS_SEQUENCE = (0.0, 0.035, 0.06)
COMPACT_CYLINDER_PICK_GRIP_SEQUENCE = (0.014, 0.012, 0.010)
COMPACT_CYLINDER_PICK_GRASP_OFFSET_SEQUENCE = (0.105, 0.095, 0.085)
CYLINDER_RETRY_MIN_GRASP_OFFSET = float(os.getenv("CYLINDER_RETRY_MIN_GRASP_OFFSET", "0.095"))
OBSTACLE_CAUTIOUS_CUBE_GRIP = float(os.getenv("OBSTACLE_CAUTIOUS_CUBE_GRIP", "0.011"))
OBSTACLE_CAUTIOUS_CYLINDER_GRIP = float(os.getenv("OBSTACLE_CAUTIOUS_CYLINDER_GRIP", "0.012"))
HELD_Z_THRESHOLD = 0.90
IK_PLAN_POS_ERR_LIMIT = CONFIG.ik_plan_pos_err_limit
IK_PREGRASP_POS_ERR_LIMIT = CONFIG.ik_pregrasp_pos_err_limit
IK_PLAN_ORI_ERR_LIMIT = CONFIG.ik_plan_ori_err_limit
IK_PREGRASP_ORI_ERR_LIMIT = CONFIG.ik_pregrasp_ori_err_limit
FAR_PICK_XY_DISTANCE = 0.74
MAX_VALID_IK_CANDIDATES = int(os.getenv("MAX_VALID_IK_CANDIDATES", "6"))
MAX_IK_ATTEMPTS_PER_SEGMENT = int(os.getenv("MAX_IK_ATTEMPTS_PER_SEGMENT", "80"))
MIN_PICK_OBJECT_Z = float(os.getenv("MIN_PICK_OBJECT_Z", "0.70"))
MAX_PICK_OBJECT_XY_DISTANCE = float(os.getenv("MAX_PICK_OBJECT_XY_DISTANCE", "0.92"))

_IK_BACKEND_NAME = "uninitialized"
_PINOCCHIO_ROBOT = None
_PINOCCHIO_MODEL = None
_PINOCCHIO_DATA = None
_PINOCCHIO_FRAME_ID = None
_LAST_SUCCESSFUL_SEED: Optional[np.ndarray] = None

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
log_event(
    "OMPL_INIT",
    "OK" if _OMPL_AVAILABLE else "FAILED",
    enabled=CONFIG.ompl_enabled,
    failure_reason=None if _OMPL_AVAILABLE else "ompl_unavailable",
)
if CONFIG.ompl_required and not _OMPL_AVAILABLE:
    raise RuntimeError(
        "OMPL_REQUIRED=true but OMPL planner bindings are unavailable. "
        "Install OMPL Python bindings in this environment or set OMPL_REQUIRED=false for diagnostics."
    )


def _initialize_ik_backend() -> None:
    global _IK_BACKEND_NAME, _PINOCCHIO_ROBOT, _PINOCCHIO_MODEL, _PINOCCHIO_DATA, _PINOCCHIO_FRAME_ID

    requested = (CONFIG.ik_backend or "auto").strip().lower()
    if requested not in {"auto", "pinocchio", "mujoco_dls"}:
        log_event("IK_INIT", "PINOCCHIO_FAILED", failure_reason=f"unknown IK_BACKEND={CONFIG.ik_backend}")
        requested = "auto"

    if requested in {"auto", "pinocchio"}:
        if pin is None or load_robot_description is None:
            log_event("IK_INIT", "PINOCCHIO_FAILED", failure_reason=str(_PINOCCHIO_IMPORT_ERROR))
            if CONFIG.ik_require_pinocchio or requested == "pinocchio":
                raise RuntimeError(
                    "Pinocchio IK requested but dependencies are unavailable. "
                    "Install pin and robot_descriptions, or set IK_BACKEND=mujoco_dls for development fallback."
                )
        else:
            try:
                _PINOCCHIO_ROBOT = load_robot_description("panda_description")
                _PINOCCHIO_MODEL = getattr(_PINOCCHIO_ROBOT, "model", _PINOCCHIO_ROBOT)
                _PINOCCHIO_DATA = _PINOCCHIO_MODEL.createData()
                for frame_name in ("panda_hand", "panda_link8", "panda_hand_tcp"):
                    frame_id = _PINOCCHIO_MODEL.getFrameId(frame_name)
                    if frame_id < len(_PINOCCHIO_MODEL.frames):
                        _PINOCCHIO_FRAME_ID = frame_id
                        break
                if _PINOCCHIO_FRAME_ID is None:
                    raise RuntimeError("Pinocchio Panda model has no panda_hand-compatible frame")
                _IK_BACKEND_NAME = "pinocchio"
                log_event("IK_INIT", "PINOCCHIO_OK", frame=_PINOCCHIO_MODEL.frames[_PINOCCHIO_FRAME_ID].name)
                return
            except Exception as exc:
                log_event("IK_INIT", "PINOCCHIO_FAILED", failure_reason=str(exc))
                if CONFIG.ik_require_pinocchio or requested == "pinocchio":
                    raise

    _IK_BACKEND_NAME = "mujoco_dls"
    log_event("IK_INIT", "MUJOCO_DLS_FALLBACK", failure_reason="pinocchio_unavailable_or_not_requested")


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


def _round_vec(values, ndigits: int = 4) -> list[float]:
    return [round(float(v), ndigits) for v in np.asarray(values, dtype=float).reshape(-1)]


def _ee_xyz() -> list[float]:
    mujoco.mj_forward(model, data)
    return _round_vec(data.xpos[ee_id], 4)


def _object_xyz(obj: Optional[str]) -> Optional[list[float]]:
    if obj is None or obj not in name_to_cube:
        return None
    mujoco.mj_forward(model, data)
    return _round_vec(data.xpos[name_to_cube[obj]], 4)


def _arm_q() -> list[float]:
    return _round_vec(current_q(), 4)


def _finger_snapshot() -> float:
    try:
        return round(_finger_pos(), 4)
    except Exception:
        return 0.0


def _distance_xy_to_base(pos: Sequence[float]) -> float:
    return round(float(np.linalg.norm(np.asarray(pos[:2], dtype=float) - BASE_XY)), 4)


def _object_pose_failure_reason(pos: Sequence[float]) -> Optional[str]:
    pos_arr = np.asarray(pos, dtype=float).reshape(3)
    if float(pos_arr[2]) < MIN_PICK_OBJECT_Z:
        return "object_displaced_below_table"
    if float(np.linalg.norm(pos_arr[:2] - BASE_XY)) > MAX_PICK_OBJECT_XY_DISTANCE:
        return "object_outside_robot_reach_after_displacement"
    return None


def _desired_orientation_error_from_matrix(rot_matrix: np.ndarray) -> float:
    z_axis = np.asarray(rot_matrix, dtype=float).reshape(3, 3)[:, 2]
    denom = max(float(np.linalg.norm(z_axis) * np.linalg.norm(_DESIRED_Z)), 1e-9)
    cos_angle = float(np.clip(np.dot(z_axis, _DESIRED_Z) / denom, -1.0, 1.0))
    return float(np.arccos(cos_angle))


def _mujoco_fk_error(q: Sequence[float], target_xyz: Sequence[float]) -> tuple[float, float, list[float]]:
    q_arr = clip_arm(np.asarray(q, dtype=float).reshape(-1))
    _plan_data.qpos[:] = data.qpos[:]
    _plan_data.qvel[:] = 0.0
    _plan_data.qpos[arm_qpos_adr] = q_arr
    mujoco.mj_forward(model, _plan_data)
    ee_pos = _plan_data.xpos[ee_id].copy()
    pos_err = float(np.linalg.norm(np.asarray(target_xyz, dtype=float).reshape(3) - ee_pos))
    ori_err = _desired_orientation_error_from_matrix(_plan_data.xmat[ee_id].reshape(3, 3))
    return pos_err, ori_err, _round_vec(ee_pos, 4)


def _log_arm_state(stage: str, status: str, **fields) -> None:
    obj = fields.get("object_id") or fields.get("held_object") or _held_object_name
    fields.setdefault("arm", ACTIVE_ARM)
    fields.setdefault("scenario_type", os.getenv("CTAMP_SCENARIO_TYPE", "static"))
    fields.setdefault("obstacle_mode", os.getenv("CTAMP_OBSTACLE_MODE", "unknown"))
    fields.setdefault("ee_xyz", _ee_xyz())
    fields.setdefault("q", _arm_q())
    fields.setdefault("finger_pos", _finger_snapshot())
    fields.setdefault("held_object", _held_object_name)
    if obj is not None and "object_xyz" not in fields:
        fields["object_xyz"] = _object_xyz(str(obj))
    log_event(stage, status, **fields)


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
        _log_arm_state(
            "COLLISION_CHECK",
            "IGNORED_START",
            phase=context,
            failure_reason=report.reason,
            collision_pair=[report.body1, report.body2],
            contact_count=int(data.ncon),
            penetration=round(float(getattr(report, "penetration", 0.0)), 5),
            ignored_body_names=list(ignored_body_names or []),
        )
        return True
    print(f"[collision] blocked during {context}: {report.reason}")
    _log_arm_state(
        "COLLISION_CHECK",
        "BLOCKED",
        phase=context,
        failure_reason=report.reason,
        collision_pair=[report.body1, report.body2],
        contact_count=int(data.ncon),
        penetration=round(float(getattr(report, "penetration", 0.0)), 5),
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
        ori_norm = _desired_orientation_error_from_matrix(R)

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

    info["backend"] = "mujoco_dls"
    return clip_arm(q_target), info


def _pinocchio_ik_solve_to(
    target_xyz: Sequence[float],
    q_seed: Optional[Sequence[float]] = None,
    steps: int = 120,
    pos_tol: float = 0.006,
    ori_tol: float = 0.25,
) -> Tuple[np.ndarray, dict]:
    if _PINOCCHIO_MODEL is None or _PINOCCHIO_DATA is None or _PINOCCHIO_FRAME_ID is None:
        raise RuntimeError("Pinocchio IK backend is not initialized")

    nq = int(_PINOCCHIO_MODEL.nq)
    nv = int(_PINOCCHIO_MODEL.nv)
    q = np.zeros(nq, dtype=float)
    seed = np.asarray(q_seed if q_seed is not None else current_q(), dtype=float).reshape(-1)
    q[: min(seed.shape[0], nq)] = seed[: min(seed.shape[0], nq)]
    if nq > 7:
        q[7:] = 0.04

    target_world_xyz = np.asarray(target_xyz, dtype=float).reshape(3)
    target_xyz = target_world_xyz - _pinocchio_base_translation()
    info = {
        "converged": False,
        "pos_err_norm": float("inf"),
        "ori_err_norm": float("inf"),
        "iters": 0,
        "backend": "pinocchio",
    }

    damping = 1e-4
    for it in range(steps):
        pin.forwardKinematics(_PINOCCHIO_MODEL, _PINOCCHIO_DATA, q)
        pin.updateFramePlacements(_PINOCCHIO_MODEL, _PINOCCHIO_DATA)
        frame = _PINOCCHIO_DATA.oMf[_PINOCCHIO_FRAME_ID]
        pos_error = target_xyz - np.asarray(frame.translation).reshape(3)
        pos_norm = float(np.linalg.norm(pos_error))
        ori_norm = _desired_orientation_error_from_matrix(np.asarray(frame.rotation))
        info.update({
            "converged": pos_norm <= pos_tol and ori_norm <= ori_tol,
            "pos_err_norm": pos_norm,
            "ori_err_norm": ori_norm,
            "iters": it + 1,
        })
        if info["converged"]:
            break

        jac = pin.computeFrameJacobian(
            _PINOCCHIO_MODEL,
            _PINOCCHIO_DATA,
            q,
            _PINOCCHIO_FRAME_ID,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        jpos = jac[:3, :nv]
        lhs = jpos @ jpos.T + damping * np.eye(3)
        dq = jpos.T @ np.linalg.solve(lhs, pos_error)
        dq = np.clip(dq, -0.12, 0.12)
        q = pin.integrate(_PINOCCHIO_MODEL, q, dq)

    return clip_arm(q[:7]), info


def _pinocchio_base_translation() -> np.ndarray:
    prefix = _arm_prefix(ACTIVE_ARM)
    return np.asarray(model.body(f"{prefix}link0").pos[:3], dtype=float)


def _solve_ik_to(
    target_xyz,
    null_ref=None,
    q_seed=None,
    steps=800,
    pos_limit: Optional[float] = None,
    ori_limit: Optional[float] = None,
):
    if _IK_BACKEND_NAME == "pinocchio":
        try:
            seed = q_seed if q_seed is not None else null_ref
            pin_q, pin_info = _pinocchio_ik_solve_to(target_xyz, q_seed=seed, steps=180)
            fk_pos_err, fk_ori_err, _ = _mujoco_fk_error(pin_q, target_xyz)
            max_pos = float(pos_limit if pos_limit is not None else IK_PLAN_POS_ERR_LIMIT)
            max_ori = float(ori_limit if ori_limit is not None else IK_PLAN_ORI_ERR_LIMIT)
            if fk_pos_err <= max_pos and fk_ori_err <= max_ori:
                return pin_q, pin_info

            dls_q, dls_info = _ik_solve_to(target_xyz, null_ref=null_ref, q_seed=q_seed, steps=steps)
            dls_pos_err, dls_ori_err, _ = _mujoco_fk_error(dls_q, target_xyz)
            use_dls = (
                (dls_pos_err <= max_pos and dls_ori_err <= max_ori)
                or (dls_pos_err + 0.15 * dls_ori_err < fk_pos_err + 0.15 * fk_ori_err)
            )
            log_event(
                "IK_SOLVE",
                "BACKEND_FALLBACK",
                backend="pinocchio",
                failure_reason="pinocchio_fk_validation_failed",
                pos_err=round(float(fk_pos_err), 5),
                ori_err=round(float(fk_ori_err), 5),
                pos_limit=max_pos,
                ori_limit=max_ori,
                fallback_backend="mujoco_dls",
                dls_pos_err=round(float(dls_pos_err), 5),
                dls_ori_err=round(float(dls_ori_err), 5),
                use_dls=use_dls,
            )
            if use_dls:
                return dls_q, dls_info
            return pin_q, pin_info
        except Exception as exc:
            log_event("IK_SOLVE", "BACKEND_FALLBACK", backend="pinocchio", failure_reason=str(exc))
    return _ik_solve_to(target_xyz, null_ref=null_ref, q_seed=q_seed, steps=steps)


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
    elif label.startswith("place("):
        xy_offsets = [
            np.array([0.018, 0.0, 0.0]),
            np.array([-0.018, 0.0, 0.0]),
            np.array([0.0, 0.018, 0.0]),
            np.array([0.0, -0.018, 0.0]),
        ]
        candidates.extend(base + offset for offset in xy_offsets)
        if "release" in label:
            candidates.extend(base + np.array([0.0, 0.0, z]) for z in (0.015, -0.015))
    return candidates


def _ik_pos_limit_for_label(label: str) -> float:
    if label.startswith("pick(") and "pregrasp" in label:
        return IK_PREGRASP_POS_ERR_LIMIT
    return IK_PLAN_POS_ERR_LIMIT


def _ik_ori_limit_for_label(label: str) -> float:
    if label.startswith("pick(") and "pregrasp" in label:
        return IK_PREGRASP_ORI_ERR_LIMIT
    return IK_PLAN_ORI_ERR_LIMIT


def _null_ref_candidates(null_ref: Optional[np.ndarray]) -> list[np.ndarray]:
    base_ref = np.asarray(null_ref if null_ref is not None else GRASP_READY, dtype=float).copy()
    refs = [current_q(), base_ref, GRASP_READY.copy(), _ELBOW_UP_REF.copy()]
    if _LAST_SUCCESSFUL_SEED is not None:
        refs.append(_LAST_SUCCESSFUL_SEED.copy())
    for delta in (0.18, -0.18):
        ref = base_ref.copy()
        ref[1] = np.clip(ref[1] + delta, arm_ranges[1, 0], arm_ranges[1, 1])
        refs.append(ref)
    unique = []
    seen = set()
    for ref in refs:
        clipped = clip_arm(ref)
        key = tuple(np.round(clipped, 4))
        if key not in seen:
            seen.add(key)
            unique.append(clipped)
    return unique


def _ranked_ik_goals(
    target_xyz: Sequence[float],
    null_ref: Optional[np.ndarray],
    label: str,
    ignored_body_names: Optional[Sequence[str]] = None,
) -> list[IKAttemptResult]:
    planner = _get_ompl_planner()
    pos_limit = _ik_pos_limit_for_label(label)
    ori_limit = _ik_ori_limit_for_label(label)
    attempts: list[IKAttemptResult] = []

    for candidate_idx, xyz in enumerate(_target_xyz_candidates(target_xyz, label)):
        for seed_idx, ref in enumerate(_null_ref_candidates(null_ref)):
            q_seed = current_q() if seed_idx == 0 else ref
            q, info = _solve_ik_to(
                xyz,
                null_ref=ref,
                q_seed=q_seed,
                pos_limit=pos_limit,
                ori_limit=ori_limit,
            )
            fk_pos_err, fk_ori_err, fk_ee_xyz = _mujoco_fk_error(q, xyz)
            pos_err = fk_pos_err
            ori_err = fk_ori_err
            joint_ok = joint_limits_valid(q, arm_ranges[:, 0], arm_ranges[:, 1])
            state_valid = None
            state_reason = None
            if planner is not None and joint_ok:
                state_valid = planner.is_state_valid_q(q, ignored_body_names=ignored_body_names)
                state_reason = getattr(planner, "_last_invalid_reason", None)
            reason = classify_ik_attempt(
                pos_err=pos_err,
                ori_err=ori_err,
                pos_limit=pos_limit,
                ori_limit=ori_limit,
                joint_limit_valid=joint_ok,
                state_valid=state_valid,
                state_invalid_reason=state_reason,
                converged=bool(info.get("converged", False)) or np.isfinite(pos_err),
            )
            score = (
                pos_err
                + 0.15 * ori_err
                + 0.02 * float(np.linalg.norm(np.asarray(q, dtype=float) - current_q()))
            )
            if label.startswith("pick(") and "grasp" in label and "pregrasp" not in label:
                score += 1.5 * float(np.linalg.norm(np.asarray(xyz, dtype=float) - np.asarray(target_xyz, dtype=float)))
            attempt = IKAttemptResult(
                q=q,
                target_xyz=np.asarray(xyz, dtype=float),
                backend=str(info.get("backend", _IK_BACKEND_NAME)),
                candidate_id=candidate_idx,
                seed_id=seed_idx,
                pos_err=pos_err,
                ori_err=ori_err,
                iterations=int(info.get("iters", 0)),
                converged=bool(info.get("converged", False)),
                joint_limit_valid=joint_ok,
                state_valid=state_valid,
                state_invalid_reason=state_reason,
                failure_reason=reason,
                score=score,
            )
            attempts.append(attempt)
            log_event(
                "IK_CANDIDATE",
                "OK" if reason == IK_SUCCESS else "REJECT",
                arm=ACTIVE_ARM,
                phase=label,
                target_xyz=[round(float(v), 4) for v in xyz],
                actual_xyz=fk_ee_xyz,
                ee_xyz=_ee_xyz(),
                q=_round_vec(q, 4),
                q_error_norm=round(float(np.linalg.norm(np.asarray(q, dtype=float) - current_q())), 5),
                pos_err=round(pos_err, 5),
                ori_err=round(ori_err, 5),
                iterations=info.get("iters", 0),
                candidate_idx=candidate_idx,
                candidate_id=candidate_idx,
                seed_id=seed_idx,
                backend=info.get("backend", _IK_BACKEND_NAME),
                pos_limit=pos_limit,
                ori_limit=ori_limit,
                converged=info.get("converged", False),
                joint_limit_valid=joint_ok,
                state_valid=state_valid,
                state_invalid_reason=state_reason,
                failure_reason=None if reason == IK_SUCCESS else reason,
                score=round(score, 5),
            )
            valid_count = sum(1 for item in attempts if item.failure_reason == IK_SUCCESS)
            if valid_count >= MAX_VALID_IK_CANDIDATES:
                return rank_ik_attempts(attempts)
            if len(attempts) >= MAX_IK_ATTEMPTS_PER_SEGMENT and valid_count > 0:
                return rank_ik_attempts(attempts)

    return rank_ik_attempts(attempts)


def _select_ik_goal(target_xyz: Sequence[float], null_ref: Optional[np.ndarray], label: str):
    ranked = _ranked_ik_goals(target_xyz, null_ref, label)
    valid = [item for item in ranked if item.failure_reason == IK_SUCCESS]
    if not valid:
        return None, None, None, IK_UNREACHABLE
    best = valid[0]
    info = {
        "backend": best.backend,
        "pos_err_norm": best.pos_err,
        "ori_err_norm": best.ori_err,
        "iters": best.iterations,
        "converged": best.converged,
        "candidate_id": best.candidate_id,
        "seed_id": best.seed_id,
    }
    return best.q, info, best.target_xyz, IK_SUCCESS

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
        _log_arm_state("TRAJECTORY_EXEC", "SKIP", waypoints=0, grip=grip)
        return True

    _log_arm_state(
        "TRAJECTORY_EXEC",
        "START",
        waypoints=len(traj),
        grip=grip,
        q_target=_round_vec(traj[-1], 4),
        q_error_norm=round(float(np.linalg.norm(clip_arm(np.asarray(traj[-1], dtype=float)) - current_q())), 5),
        ignored_body_names=list(ignored_body_names or []),
        settle_steps_per_wp=settle_steps_per_wp,
        final_settle_steps=final_settle_steps,
    )
    started = time.perf_counter()
    for waypoint_index, q in enumerate(traj):
        q = clip_arm(np.asarray(q, dtype=float))
        if waypoint_index in {0, len(traj) - 1} or waypoint_index % max(1, len(traj) // 4) == 0:
            _log_arm_state(
                "TRAJECTORY_WAYPOINT",
                "EXEC",
                waypoints=f"{waypoint_index + 1}/{len(traj)}",
                grip=grip,
                q=[round(float(v), 4) for v in q],
                q_target=[round(float(v), 4) for v in traj[-1]],
                q_error_norm=round(float(np.linalg.norm(q - current_q())), 5),
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
                _log_arm_state(
                    "TRAJECTORY_EXEC",
                    "FAILED",
                    waypoints=len(traj),
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    failure_reason=f"collision_at_waypoint_{waypoint_index}",
                    q_target=_round_vec(q, 4),
                    q_error_norm=round(float(np.linalg.norm(q - current_q())), 5),
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
                _log_arm_state(
                    "TRAJECTORY_EXEC",
                    "FAILED",
                    waypoints=len(traj),
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    failure_reason="collision_during_final_settle",
                    q_target=_round_vec(final_q, 4),
                    q_error_norm=round(float(np.linalg.norm(final_q - current_q())), 5),
                )
                return False

    _log_arm_state(
        "TRAJECTORY_EXEC",
        "OK",
        waypoints=len(traj),
        duration_ms=int((time.perf_counter() - started) * 1000),
        q_target=_round_vec(traj[-1], 4),
        q_error_norm=round(float(np.linalg.norm(clip_arm(np.asarray(traj[-1], dtype=float)) - current_q())), 5),
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
        _log_arm_state("OMPL_PLAN", "UNAVAILABLE", phase=label, failure_reason="ompl_unavailable", q_target=_round_vec(goal_q, 4))
        return False

    planner = _get_ompl_planner()
    if planner is None:
        print("[exec][OMPL] planner init failed")
        _log_arm_state("OMPL_PLAN", "FAILED", phase=label, failure_reason="planner_init_failed", q_target=_round_vec(goal_q, 4))
        return False

    planner_attempts = []
    for candidate in (planner_name, CONFIG.ompl_fragile_planner_name, "BITstar", "RRTConnect"):
        if candidate and candidate not in planner_attempts:
            planner_attempts.append(candidate)

    start_state_valid = planner.is_state_valid_q(start_q)
    start_state_reason = getattr(planner, "_last_invalid_reason", None)
    last_info = None
    try:
        _log_arm_state(
            "OMPL_PLAN",
            "START",
            phase=label,
            planner=planner_attempts[0],
            grip=grip,
            ignored_body_names=list(ignored_body_names or []),
            time_limit=time_limit,
            start_q=[round(float(v), 4) for v in start_q],
            goal_q=[round(float(v), 4) for v in goal_q],
            q_target=_round_vec(goal_q, 4),
            q_error_norm=round(float(np.linalg.norm(goal_q - start_q)), 5),
            planner_attempts=planner_attempts,
            start_state_valid=start_state_valid,
            start_state_invalid_reason=start_state_reason,
        )
        started = time.perf_counter()
        for attempt_index, attempt_planner in enumerate(planner_attempts, start=1):
            traj, info = planner.plan(
                start_q=start_q,
                goal_q=goal_q,
                time_limit=time_limit,
                planner_name=attempt_planner,
                ignored_body_names=ignored_body_names,
                fragile_mode=attempt_planner == CONFIG.ompl_fragile_planner_name,
            )
            last_info = info
            if traj is None:
                attempt_duration_ms = int((time.perf_counter() - started) * 1000)
                goal_attempts = info.get("goal_attempts") or []
                statuses = [item.get("status") for item in goal_attempts if isinstance(item, dict)]
                if statuses and all(status == "invalid_goal" for status in statuses):
                    failure_reason = IK_GOAL_STATE_INVALID
                elif attempt_duration_ms >= int(time_limit * 950):
                    failure_reason = OMPL_TIMEOUT
                else:
                    failure_reason = "ompl_no_path_found"
                _log_arm_state(
                    "OMPL_PLAN_ATTEMPT",
                    "FAILED",
                    phase=label,
                    planner=info.get("planner_name", attempt_planner),
                    attempt=attempt_index,
                    failure_reason=failure_reason,
                    ompl_result=failure_reason,
                    goal_attempts=goal_attempts,
                    q_target=_round_vec(goal_q, 4),
                    q_error_norm=round(float(np.linalg.norm(goal_q - current_q())), 5),
                )
                continue

            duration_ms = int((time.perf_counter() - started) * 1000)
            print(
                f"[exec][OMPL] solved={info.get('solved')} "
                f"planner={info.get('planner_name')} waypoints={info.get('num_waypoints')}"
            )
            _log_arm_state(
                "OMPL_PLAN",
                "OK",
                phase=label,
                planner=info.get("planner_name"),
                waypoints=info.get("num_waypoints"),
                duration_ms=duration_ms,
                path_length=round(float(info.get("path_length_joint_space", 0.0)), 4),
                selected_goal_q=info.get("selected_goal_q"),
                q_target=_round_vec(info.get("selected_goal_q", goal_q), 4),
                q_error_norm=round(float(np.linalg.norm(np.asarray(info.get("selected_goal_q", goal_q), dtype=float) - current_q())), 5),
                ompl_result="success",
            )
            return _execute_joint_trajectory(
                traj,
                grip=grip,
                ignored_body_names=ignored_body_names,
                settle_steps_per_wp=settle_steps_per_wp,
                final_settle_steps=final_settle_steps,
            )

        duration_ms = int((time.perf_counter() - started) * 1000)
        goal_attempts = (last_info or {}).get("goal_attempts") or []
        statuses = [item.get("status") for item in goal_attempts if isinstance(item, dict)]
        if statuses and all(status == "invalid_goal" for status in statuses):
            failure_reason = IK_GOAL_STATE_INVALID
        elif duration_ms >= int(time_limit * 950):
            failure_reason = OMPL_TIMEOUT
        else:
            failure_reason = "ompl_no_path_found"
        print(f"[exec][OMPL] planning failed: {last_info}")
        _log_arm_state(
            "OMPL_PLAN",
            "FAILED",
            phase=label,
            planner=(last_info or {}).get("planner_name", planner_name),
            duration_ms=duration_ms,
            failure_reason=failure_reason,
            ompl_result=failure_reason,
            goal_attempts=goal_attempts,
            q_target=_round_vec(goal_q, 4),
            q_error_norm=round(float(np.linalg.norm(goal_q - current_q())), 5),
        )
        return False
    except Exception as e:
        print(f"[exec][OMPL] error: {e}")
        _log_arm_state("OMPL_PLAN", "ERROR", phase=label, failure_reason=str(e), q_target=_round_vec(goal_q, 4))
        return False


def _move_pose_safe(
    target_xyz: Sequence[float],
    grip: float,
    null_ref: Optional[np.ndarray] = None,
    ignored_body_names: Optional[Sequence[str]] = None,
    label: str = "",
    cautious_motion: bool = False,
) -> bool:
    global _LAST_SUCCESSFUL_SEED
    _log_arm_state(
        "MOVE_POSE",
        "START",
        phase=label,
        target_xyz=[round(float(v), 4) for v in target_xyz],
        grip=grip,
        ignored_body_names=list(ignored_body_names or []),
    )
    ranked_attempts = _ranked_ik_goals(
        target_xyz,
        null_ref,
        label,
        ignored_body_names=ignored_body_names,
    )
    valid_attempts = [item for item in ranked_attempts if item.failure_reason == IK_SUCCESS]
    if not valid_attempts:
        best_attempt = ranked_attempts[0] if ranked_attempts else None
        pos_err = float(best_attempt.pos_err if best_attempt is not None else float("inf"))
        ori_err = float(best_attempt.ori_err if best_attempt is not None else float("inf"))
        failure_reason = best_attempt.failure_reason if best_attempt is not None else IK_UNREACHABLE
        pos_limit = _ik_pos_limit_for_label(label)
        ori_limit = _ik_ori_limit_for_label(label)
        print(
            f"[exec][IK] reject {label or 'pose'}: "
            f"best_pos={pos_err:.4f} best_ori={ori_err:.4f} "
            f"pos_limit={pos_limit:.3f} ori_limit={ori_limit:.3f} reason={failure_reason}"
        )
        _log_arm_state(
            "IK_SOLVE",
            "FAILED",
            phase=label,
            target_xyz=[round(float(v), 4) for v in target_xyz],
            failure_reason=failure_reason,
            pos_err=round(pos_err, 5) if np.isfinite(pos_err) else None,
            ori_err=round(ori_err, 5) if np.isfinite(ori_err) else None,
            pos_limit=pos_limit,
            ori_limit=ori_limit,
            backend=best_attempt.backend if best_attempt is not None else _IK_BACKEND_NAME,
            candidate_id=best_attempt.candidate_id if best_attempt is not None else None,
            seed_id=best_attempt.seed_id if best_attempt is not None else None,
        )
        return False

    for goal_attempt, attempt in enumerate(valid_attempts, start=1):
        selected_goal_q = attempt.q
        selected_xyz = attempt.target_xyz
        _log_arm_state(
            "IK_SOLVE",
            "OK",
            phase=label,
            target_xyz=[round(float(v), 4) for v in selected_xyz],
            q_target=_round_vec(selected_goal_q, 4),
            pos_err=round(float(attempt.pos_err), 5),
            ori_err=round(float(attempt.ori_err), 5),
            iterations=attempt.iterations,
            backend=attempt.backend,
            candidate_id=attempt.candidate_id,
            seed_id=attempt.seed_id,
            goal_attempt=goal_attempt,
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
            _LAST_SUCCESSFUL_SEED = np.asarray(selected_goal_q, dtype=float).copy()
            _log_arm_state(
                "MOVE_POSE",
                "OK",
                phase=label,
                target_xyz=[round(float(v), 4) for v in selected_xyz],
                backend=attempt.backend,
                candidate_id=attempt.candidate_id,
                seed_id=attempt.seed_id,
                execution_result="success",
            )
            return True
        _log_arm_state(
            "MOVE_POSE",
            "RETRY_NEXT_IK_GOAL",
            phase=label,
            target_xyz=[round(float(v), 4) for v in selected_xyz],
            backend=attempt.backend,
            candidate_id=attempt.candidate_id,
            seed_id=attempt.seed_id,
            failure_reason="ompl_or_execution_failed",
        )

    if USE_IK_FALLBACK:
        print(f"[exec] OMPL failed for {label or 'pose'}; using IK fallback")
        fallback_xyz = valid_attempts[0].target_xyz
        _log_arm_state("MOVE_POSE", "IK_FALLBACK", phase=label, target_xyz=[round(float(v), 4) for v in fallback_xyz])
        move_ee_to(fallback_xyz, grip=grip, steps=300, null_ref=null_ref)
        return True
    print(f"[exec] OMPL failed for {label or 'pose'}; no fallback in fragile-scene mode")
    _log_arm_state(
        "MOVE_POSE",
        "FAILED",
        phase=label,
        target_xyz=[round(float(v), 4) for v in valid_attempts[0].target_xyz],
        failure_reason="ompl_failed_no_fallback",
        execution_result="execution_failed",
    )
    return False


# =============================
# SAFETY / GRIPPER HELPERS
# =============================

def _recover_to_safe_hover(ignored_body_names: Optional[Sequence[str]] = None) -> None:
    """Move away from table/contact states before retrying another pick."""
    log_event("RECOVERY", "START", phase="safe_hover", ignored_body_names=list(ignored_body_names or []))
    ok = _move_with_ompl(
        goal_q=GRASP_READY,
        grip=0.04,
        ignored_body_names=ignored_body_names,
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


def _move_to_grasp_ready(
    reason: str,
    grip: float = 0.04,
    ignored_body_names: Optional[Sequence[str]] = None,
) -> bool:
    if float(np.linalg.norm(current_q() - GRASP_READY)) < 0.05:
        return True
    log_event("TRANSIT", "START", phase=reason, target="GRASP_READY")
    ok = _move_with_ompl(
        goal_q=GRASP_READY,
        grip=grip,
        ignored_body_names=ignored_body_names,
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


def drop(ignored_body_name: Optional[str] = None):
    """Emergency release at the current arm position."""
    global _held_grip_target
    ignored = [ignored_body_name] if ignored_body_name else None
    _log_arm_state("DROP", "START", object_id=_held_object_name or ignored_body_name, ignored_body_names=ignored or [])
    set_grip(0.04, steps=300)
    for _ in range(120):
        mujoco.mj_step(model, data)
        viewer.sync()
    _recover_to_safe_hover(ignored_body_names=ignored)
    _held_grip_target = 0.015
    _log_arm_state("DROP", "OK")


# =============================
# STARTUP WARMUP
# =============================

_initialize_available_arms()
_initialize_ik_backend()

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
    _log_arm_state("PICK", "START", object_id=obj)
    if obj not in name_to_cube:
        print(f"[exec] unknown object: {obj}")
        _log_arm_state("PICK", "FAILED", object_id=obj, failure_reason="unknown_object")
        return

    if not _move_to_grasp_ready(f"before pick({obj})", grip=0.04):
        _log_arm_state("PICK", "FAILED", object_id=obj, phase="transit", failure_reason="move_to_grasp_ready_failed")
        return

    call_count = _pick_call_counts.get(obj, 0)
    profile_index = min(call_count, len(PICK_GRIP_SEQUENCE) - 1)
    _pick_call_counts[obj] = call_count + 1

    grip_target = PICK_GRIP_SEQUENCE[profile_index]
    grasp_offset = PICK_GRASP_OFFSET_SEQUENCE[profile_index]
    clearance_bonus = PICK_CLEARANCE_BONUS_SEQUENCE[profile_index]
    is_circle = obj.startswith("circle")
    if is_circle:
        # Cylinders skip profile 0 (grasp_offset=0.105 overshoots the cylinder body).
        # Start at profile 1 (grasp_offset=0.095) which is already proven reliable.
        cyl_index = min(profile_index + 1, len(COMPACT_CYLINDER_PICK_GRIP_SEQUENCE) - 1)
        profile_index = cyl_index
        grip_target = COMPACT_CYLINDER_PICK_GRIP_SEQUENCE[cyl_index]
        grasp_offset = max(COMPACT_CYLINDER_PICK_GRASP_OFFSET_SEQUENCE[cyl_index], CYLINDER_RETRY_MIN_GRASP_OFFSET)
        clearance_bonus = PICK_CLEARANCE_BONUS_SEQUENCE[cyl_index]
    _log_arm_state(
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
    pose_failure_reason = _object_pose_failure_reason(cube_pos)
    if pose_failure_reason is not None:
        _log_arm_state(
            "PICK",
            "FAILED",
            object_id=obj,
            phase="precheck",
            object_xyz=_round_vec(cube_pos, 4),
            distance_to_target=_distance_xy_to_base(cube_pos),
            failure_reason=pose_failure_reason,
        )
        return
    obstacle_distance = _min_obstacle_xy_distance(cube_pos)
    approach_clearance = APPROACH_CLEARANCE + clearance_bonus
    cautious_motion = False
    _log_arm_state(
        "PICK_PRECHECK",
        "START",
        object_id=obj,
        object_xyz=_round_vec(cube_pos, 4),
        obstacle_distance=obstacle_distance if np.isfinite(obstacle_distance) else None,
        distance_to_target=_distance_xy_to_base(cube_pos),
        approach_clearance=approach_clearance,
    )
    if obstacle_distance < MIN_PICK_OBSTACLE_CLEARANCE:
        print(
            f"[exec][OBSTACLE_AVOID] cancel pick({obj}): "
            f"distance={obstacle_distance:.3f}m threshold={MIN_PICK_OBSTACLE_CLEARANCE:.3f}m"
        )
        _log_arm_state(
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
        grip_target = min(
            grip_target,
            OBSTACLE_CAUTIOUS_CYLINDER_GRIP if is_circle else OBSTACLE_CAUTIOUS_CUBE_GRIP,
        )
        print(
            f"[exec][OBSTACLE_AVOID] {obj} near obstacle "
            f"distance={obstacle_distance:.3f}m; using cautious high-clearance approach"
        )
        _log_arm_state(
            "OBSTACLE_AVOID",
            "NEAR_CAUTIOUS",
            object_id=obj,
            obstacle_distance=obstacle_distance,
            approach_clearance=approach_clearance,
            grip=grip_target,
        )
    else:
        _log_arm_state(
            "OBSTACLE_AVOID",
            "CLEAR",
            object_id=obj,
            obstacle_distance=obstacle_distance if np.isfinite(obstacle_distance) else None,
            approach_clearance=approach_clearance,
        )

    cube_ref = cube_null_ref(cube_pos)
    object_xy_distance = float(np.linalg.norm(cube_pos[:2] - BASE_XY))
    if object_xy_distance > FAR_PICK_XY_DISTANCE:
        grasp_offset += 0.02
    pregrasp_clearance = approach_clearance
    if object_xy_distance > FAR_PICK_XY_DISTANCE:
        pregrasp_clearance = min(pregrasp_clearance, 0.24)
        _log_arm_state(
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
    _log_arm_state(
        "PICK_TARGETS",
        "SET",
        object_id=obj,
        object_xyz=_round_vec(cube_pos, 4),
        pregrasp_xyz=_round_vec(pregrasp_xyz, 4),
        grasp_xyz=_round_vec(grasp_xyz, 4),
        lift_xyz=_round_vec(lift_xyz, 4),
        cautious_motion=cautious_motion,
    )

    # 1) Move above the cube.
    if not _move_pose_safe(
        pregrasp_xyz,
        grip=0.04,
        null_ref=cube_ref,
        ignored_body_names=None,
        label=f"pick({obj}) pregrasp",
        cautious_motion=cautious_motion,
    ):
        _log_arm_state("PICK", "FAILED", object_id=obj, phase="pregrasp", target_xyz=_round_vec(pregrasp_xyz, 4), failure_reason="move_pregrasp_failed")
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
        _log_arm_state("PICK", "FAILED", object_id=obj, phase="grasp", target_xyz=_round_vec(grasp_xyz, 4), failure_reason="move_grasp_failed")
        return

    # 3) Close gripper and let contact settle.
    _log_arm_state("PICK", "GRIP_CLOSE", object_id=obj, phase="close_gripper", target_xyz=_round_vec(grasp_xyz, 4))
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
        drop(obj)
        _held_object_name = None
        _log_arm_state("PICK", "FAILED", object_id=obj, phase="lift", target_xyz=_round_vec(lift_xyz, 4), failure_reason="move_lift_failed")
        return

    _check_obstacles_fallen(f"pick({obj})")
    lifted, z = _object_lifted(obj)
    if not lifted:
        print(f"[exec][PICK_RETRY] {obj} not lifted after grasp z={z:.3f}; release and retry")
        _log_arm_state("PICK", "FAILED", object_id=obj, phase="lift_check", failure_reason="object_not_lifted", object_z=round(z, 4))
        drop(obj)
        _held_object_name = None
        return
    _log_arm_state("PICK", "OK", object_id=obj, object_z=round(z, 4))



def place(x, y, obj=None):
    global _held_object_name, _held_grip_target

    print(f"[exec] place({x:.3f}, {y:.3f})")
    _log_arm_state("PLACE", "START", object_id=obj or _held_object_name, target_xyz=[round(float(x), 4), round(float(y), 4), 0.83])
    if _held_object_name is None:
        print("[exec] no cube is currently held")
        _log_arm_state("PLACE", "FAILED", object_id=obj, failure_reason="no_object_held")
        return

    obj = _held_object_name

    # Settled cube centre after release.
    place_pos = np.array([x, y, 0.83])
    release_pos = np.array([x, y, 0.89])
    place_ref = cube_null_ref(place_pos)
    obstacle_distance = _min_obstacle_xy_distance(place_pos)
    approach_clearance = APPROACH_CLEARANCE
    cautious_motion = False
    _log_arm_state(
        "PLACE_PRECHECK",
        "START",
        object_id=obj,
        target_xyz=_round_vec(place_pos, 4),
        obstacle_distance=obstacle_distance if np.isfinite(obstacle_distance) else None,
        distance_to_target=round(float(np.linalg.norm(np.asarray([x, y]) - np.asarray(_object_xyz(obj)[:2]))), 4) if _object_xyz(obj) else None,
        approach_clearance=approach_clearance,
    )
    if obstacle_distance < CAUTIOUS_OBSTACLE_CLEARANCE:
        cautious_motion = True
        approach_clearance += 0.06
        print(
            f"[exec][OBSTACLE_AVOID] place target near obstacle "
            f"distance={obstacle_distance:.3f}m; using cautious high-clearance preplace"
        )
        _log_arm_state(
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
    _log_arm_state(
        "PLACE_TARGETS",
        "SET",
        object_id=obj,
        target_xyz=_round_vec(place_pos, 4),
        preplace_xyz=_round_vec(preplace_xyz, 4),
        release_xyz=_round_vec(release_xyz, 4),
        retreat_xyz=_round_vec(retreat_xyz, 4),
        cautious_motion=cautious_motion,
    )

    # 1) Move above the goal while still holding the cube.
    if not _move_pose_safe(
        preplace_xyz,
        grip=_held_grip_target,
        null_ref=place_ref,
        ignored_body_names=[obj],
        label=f"place({x:.3f}, {y:.3f}) preplace",
        cautious_motion=cautious_motion,
    ):
        drop(obj)
        _held_object_name = None
        _log_arm_state("PLACE", "FAILED", object_id=obj, phase="preplace", target_xyz=_round_vec(preplace_xyz, 4), failure_reason="move_preplace_failed")
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
        drop(obj)
        _held_object_name = None
        _log_arm_state("PLACE", "FAILED", object_id=obj, phase="release", target_xyz=_round_vec(release_xyz, 4), failure_reason="move_release_failed")
        return

    # 3) Let the arm settle before opening.
    _log_arm_state("PLACE", "SETTLE_BEFORE_OPEN", object_id=obj, target_xyz=_round_vec(release_xyz, 4), steps=120)
    for _ in range(120):
        mujoco.mj_step(model, data)
        viewer.sync()

    print(f"[exec] finger before open: {_finger_pos():.4f}")
    _log_arm_state("PLACE", "GRIP_OPEN", object_id=obj, target_xyz=_round_vec(release_xyz, 4), finger_before=round(_finger_pos(), 4))
    set_grip(0.04, steps=450)
    print(f"[exec] finger after open:  {_finger_pos():.4f}")
    _log_arm_state("PLACE", "GRIP_OPENED", object_id=obj, target_xyz=_round_vec(place_pos, 4), finger_after=round(_finger_pos(), 4))

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
        _log_arm_state("PLACE", "RETREAT_FAILED", object_id=obj, target_xyz=_round_vec(retreat_xyz, 4), failure_reason="retreat_failed_after_release")

    _held_object_name = None
    _held_grip_target = 0.015
    _move_to_grasp_ready(f"after place({x:.3f}, {y:.3f})", grip=0.04, ignored_body_names=[obj])
    _check_obstacles_fallen(f"place({x:.3f}, {y:.3f})")
    _log_arm_state("PLACE", "OK", object_id=obj, target_xyz=[round(float(x), 4), round(float(y), 4), 0.83])

