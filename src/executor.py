from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import mujoco
import mujoco.viewer
import numpy as np

try:
    from ompl_planner import make_default_panda_planner
except Exception:
    make_default_panda_planner = None


# =============================
# LOAD SIMULATION (ONCE)
# =============================

_MODELS = Path(__file__).parent.parent / "models"
model = mujoco.MjModel.from_xml_path(str(_MODELS / "panda.xml"))
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

viewer = mujoco.viewer.launch_passive(model, data)
viewer.cam.distance = 2.5
viewer.cam.azimuth = 120
viewer.cam.elevation = -30
viewer.cam.lookat[:] = [0, 0, 0.7]

# Planning-side copy. Never use the live MuJoCo state directly inside IK.
_plan_data = mujoco.MjData(model)

# =============================
# ARM SETUP
# =============================

ee_id = model.body("hand").id
BASE_XY = np.array([-0.4, 0.0])

arm_joint_names = [f"joint{i}" for i in range(1, 8)]
arm_qpos_adr = np.array([model.joint(n).qposadr[0] for n in arm_joint_names], dtype=int)
arm_dof_adr = np.array([model.joint(n).dofadr[0] for n in arm_joint_names], dtype=int)
arm_ranges = np.array([model.joint(n).range for n in arm_joint_names], dtype=float)

HOME = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7854])
GRASP_READY = np.array([0.0, 0.529, 0.0, -1.98, 0.0, 2.495, -0.75])
_DESIRED_Z = np.array([0.0, 0.0, -1.0])
GRASP_OFFSET = 0.10
APPROACH_CLEARANCE = 0.30

# When fragile objects are present, never use IK fallback for physical motion.
USE_IK_FALLBACK = False

# Keep OMPL paths dense and the commanded motion slow.
DEFAULT_PLANNER_NAME = "RRTConnect"
DEFAULT_TIME_LIMIT = 3.0
DEFAULT_SETTLE_STEPS_PER_WP = 8
DEFAULT_FINAL_SETTLE_STEPS = 15

# =============================
# FIND CUBES
# =============================

name_to_cube = {}
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name and name.startswith("cube"):
        name_to_cube[name] = model.body(name).id

# =============================
# OPTIONAL OBSTACLE MONITORING
# =============================

_obstacle_ids: dict[str, int] = {}
for _n in ["vase1", "glass1"]:
    try:
        _obstacle_ids[_n] = model.body(_n).id
    except Exception:
        pass

_obstacle_init_z: dict[str, float] = {}
_obstacle_init_xy: dict[str, np.ndarray] = {}


def _init_obstacle_monitoring() -> None:
    mujoco.mj_forward(model, data)
    for name, bid in _obstacle_ids.items():
        pos = data.xpos[bid].copy()
        _obstacle_init_z[name] = float(pos[2])
        _obstacle_init_xy[name] = pos[:2].copy()
        print(f"[init] obstacle '{name}' z0={pos[2]:.4f} xy0=({pos[0]:.3f},{pos[1]:.3f})")


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
        if z_drop > 0.03 or xy_dist > 0.05:
            cur_pos = [round(float(pos[0]), 3), round(float(pos[1]), 3), round(z, 3)]
            ee_pos = [round(float(data.xpos[ee_id][0]), 3), round(float(data.xpos[ee_id][1]), 3), round(float(data.xpos[ee_id][2]), 3)]
            print(
                f"[OBSTACLE] {name} DISPLACED during '{context}' | "
                f"pos={cur_pos}  z_drop={z_drop:.3f} m  xy_shift={xy_dist:.3f} m "
                f"| arm_ee={ee_pos}"
            )


# =============================
# OMPL PLANNER
# =============================

_ompl_planner = None
_OMPL_AVAILABLE = make_default_panda_planner is not None


def _get_ompl_planner():
    global _ompl_planner
    if not _OMPL_AVAILABLE:
        return None
    if _ompl_planner is None:
        _ompl_planner = make_default_panda_planner(
            model=model,
            data=data,
            planner_name=DEFAULT_PLANNER_NAME,
            time_limit=DEFAULT_TIME_LIMIT,
        )
    else:
        _ompl_planner.sync_live_data(data)
    return _ompl_planner


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
    data.ctrl[:7] = clip_arm(np.asarray(q, dtype=float))
    data.ctrl[7] = float(grip)


def _step_sim(steps: int, q: Optional[np.ndarray] = None, grip: Optional[float] = None) -> None:
    for _ in range(steps):
        if q is not None:
            data.ctrl[:7] = clip_arm(np.asarray(q, dtype=float))
        if grip is not None:
            data.ctrl[7] = float(grip)
        mujoco.mj_step(model, data)
        viewer.sync()


def _finger_pos() -> float:
    finger_qposadr = model.joint("finger_joint1").qposadr[0]
    return float(data.qpos[finger_qposadr])


def go_to_cube_ready(cube_pos, steps=400):
    """Deprecated for fragile scenes. Kept only for backward compatibility."""
    target = cube_null_ref(np.asarray(cube_pos, dtype=float))
    for _ in range(steps):
        data.ctrl[:7] += 0.01 * (target - data.ctrl[:7])
        data.ctrl[7] = 0.04
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


# =============================
# OMPL EXECUTION
# =============================

def _execute_joint_trajectory(
    traj: np.ndarray,
    grip: float,
    settle_steps_per_wp: int = DEFAULT_SETTLE_STEPS_PER_WP,
    final_settle_steps: int = DEFAULT_FINAL_SETTLE_STEPS,
) -> None:
    if traj is None or len(traj) == 0:
        return

    for q in traj:
        q = clip_arm(np.asarray(q, dtype=float))
        for _ in range(settle_steps_per_wp):
            _set_arm_ctrl(q, grip)
            mujoco.mj_step(model, data)
            viewer.sync()

    if final_settle_steps > 0:
        final_q = clip_arm(np.asarray(traj[-1], dtype=float))
        for _ in range(final_settle_steps):
            _set_arm_ctrl(final_q, grip)
            mujoco.mj_step(model, data)
            viewer.sync()


def _move_with_ompl(
    goal_q: Sequence[float],
    grip: float,
    ignored_body_names: Optional[Sequence[str]] = None,
    planner_name: str = DEFAULT_PLANNER_NAME,
    time_limit: float = DEFAULT_TIME_LIMIT,
) -> bool:
    goal_q = np.asarray(goal_q, dtype=float).reshape(7)
    start_q = current_q()

    if not _OMPL_AVAILABLE:
        print("[exec][OMPL] unavailable")
        return False

    planner = _get_ompl_planner()
    if planner is None:
        print("[exec][OMPL] planner init failed")
        return False

    try:
        traj, info = planner.plan(
            start_q=start_q,
            goal_q=goal_q,
            time_limit=time_limit,
            planner_name=planner_name,
            ignored_body_names=ignored_body_names,
            fragile_mode=True,
        )
        if traj is None:
            print(f"[exec][OMPL] planning failed: {info}")
            return False

        print(
            f"[exec][OMPL] solved={info.get('solved')} "
            f"planner={info.get('planner_name')} waypoints={info.get('num_waypoints')}"
        )
        _execute_joint_trajectory(traj, grip=grip)
        return True
    except Exception as e:
        print(f"[exec][OMPL] error: {e}")
        return False


def _move_pose_safe(
    target_xyz: Sequence[float],
    grip: float,
    null_ref: Optional[np.ndarray] = None,
    ignored_body_names: Optional[Sequence[str]] = None,
    label: str = "",
) -> bool:
    goal_q, ik_info = _ik_solve_to(target_xyz, null_ref=null_ref)
    if not ik_info["converged"]:
        print(
            f"[exec][IK] warning for {label or 'pose'}: "
            f"pos={ik_info['pos_err_norm']:.4f} ori={ik_info['ori_err_norm']:.4f} iters={ik_info['iters']}"
        )

    ok = _move_with_ompl(
        goal_q=goal_q,
        grip=grip,
        ignored_body_names=ignored_body_names,
        planner_name=DEFAULT_PLANNER_NAME,
        time_limit=DEFAULT_TIME_LIMIT,
    )

    if ok:
        return True

    if USE_IK_FALLBACK:
        print(f"[exec] OMPL failed for {label or 'pose'}; using IK fallback")
        move_ee_to(target_xyz, grip=grip, steps=300, null_ref=null_ref)
        return True

    print(f"[exec] OMPL failed for {label or 'pose'}; no fallback in fragile-scene mode")
    return False


# =============================
# SAFETY / GRIPPER HELPERS
# =============================

def set_grip(target: float, steps: int = 200):
    for _ in range(steps):
        data.ctrl[7] = float(target)
        mujoco.mj_step(model, data)
        viewer.sync()


def drop():
    """Emergency release at the current arm position."""
    set_grip(0.04, steps=300)
    for _ in range(120):
        mujoco.mj_step(model, data)
        viewer.sync()


# =============================
# STARTUP WARMUP
# =============================

# Phase 1: HOME.
data.qpos[arm_qpos_adr] = HOME
_set_arm_ctrl(HOME, 0.04)
for _ in range(250):
    mujoco.mj_step(model, data)
    viewer.sync()

# Phase 2: Transition to a grasp-friendly pose.
for _ in range(350):
    data.ctrl[:7] += 0.01 * (GRASP_READY - data.ctrl[:7])
    data.ctrl[7] = 0.04
    mujoco.mj_step(model, data)
    viewer.sync()

_init_obstacle_monitoring()


# =============================
# HIGH-LEVEL ACTIONS
# =============================

_held_object_name: Optional[str] = None


def pick(obj):
    global _held_object_name

    print(f"[exec] pick({obj})")
    if obj not in name_to_cube:
        print(f"[exec] unknown object: {obj}")
        return

    cube_id = name_to_cube[obj]

    # Let the system settle before planning a new motion.
    _step_sim(80, grip=0.04)
    mujoco.mj_forward(model, data)
    cube_pos = data.xpos[cube_id].copy()

    cube_ref = cube_null_ref(cube_pos)
    pregrasp_xyz = cube_pos + np.array([0.0, 0.0, APPROACH_CLEARANCE])
    grasp_xyz = cube_pos + np.array([0.0, 0.0, GRASP_OFFSET])
    lift_xyz = cube_pos + np.array([0.0, 0.0, APPROACH_CLEARANCE])

    # 1) Move above the cube.
    if not _move_pose_safe(
        pregrasp_xyz,
        grip=0.04,
        null_ref=cube_ref,
        ignored_body_names=None,
        label=f"pick({obj}) pregrasp",
    ):
        return

    # 2) Move to the grasp pose while ignoring the target cube only.
    if not _move_pose_safe(
        grasp_xyz,
        grip=0.04,
        null_ref=cube_ref,
        ignored_body_names=[obj],
        label=f"pick({obj}) grasp",
    ):
        return

    # 3) Close gripper and let contact settle.
    set_grip(0.015, steps=260)
    for _ in range(70):
        mujoco.mj_step(model, data)
        viewer.sync()

    _held_object_name = obj

    # 4) Lift straight up using OMPL, still ignoring the target cube.
    if not _move_pose_safe(
        lift_xyz,
        grip=0.015,
        null_ref=cube_ref,
        ignored_body_names=[obj],
        label=f"pick({obj}) lift",
    ):
        # If lift planning fails after grasping, release immediately and let the
        # closed-loop system replan from the table state.
        drop()
        _held_object_name = None
        return

    _check_obstacles_fallen(f"pick({obj})")



def place(x, y, obj=None):
    global _held_object_name

    print(f"[exec] place({x:.3f}, {y:.3f})")
    if _held_object_name is None:
        print("[exec] no cube is currently held")
        return

    obj = _held_object_name

    # Settled cube centre after release.
    place_pos = np.array([x, y, 0.83])
    release_pos = np.array([x, y, 0.89])
    place_ref = cube_null_ref(place_pos)

    preplace_xyz = place_pos + np.array([0.0, 0.0, APPROACH_CLEARANCE])
    release_xyz = release_pos + np.array([0.0, 0.0, GRASP_OFFSET])
    retreat_xyz = release_pos + np.array([0.0, 0.0, APPROACH_CLEARANCE])

    # 1) Move above the goal while still holding the cube.
    if not _move_pose_safe(
        preplace_xyz,
        grip=0.015,
        null_ref=place_ref,
        ignored_body_names=[obj],
        label=f"place({x:.3f}, {y:.3f}) preplace",
    ):
        drop()
        _held_object_name = None
        return

    # 2) Move to the release pose.
    if not _move_pose_safe(
        release_xyz,
        grip=0.015,
        null_ref=place_ref,
        ignored_body_names=[obj],
        label=f"place({x:.3f}, {y:.3f}) release",
    ):
        drop()
        _held_object_name = None
        return

    # 3) Let the arm settle before opening.
    for _ in range(120):
        mujoco.mj_step(model, data)
        viewer.sync()

    print(f"[exec] finger before open: {_finger_pos():.4f}")
    set_grip(0.04, steps=450)
    print(f"[exec] finger after open:  {_finger_pos():.4f}")

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
    ):
        # Retreat failure after release is not fatal, but we still keep the
        # controller in a safe open state.
        print(f"[exec] retreat failed after place({x:.3f}, {y:.3f})")

    _held_object_name = None
    _check_obstacles_fallen(f"place({x:.3f}, {y:.3f})")

