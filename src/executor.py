import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

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

# =============================
# ARM SETUP
# =============================

ee_id   = model.body("hand").id
BASE_XY = np.array([-0.4, 0.0])   # arm base XY in world frame

arm_joint_names = [f"joint{i}" for i in range(1, 8)]
arm_qpos_adr = np.array([model.joint(n).qposadr[0] for n in arm_joint_names], dtype=int)
arm_dof_adr  = np.array([model.joint(n).dofadr[0]  for n in arm_joint_names], dtype=int)
arm_ranges   = np.array([model.joint(n).range       for n in arm_joint_names], dtype=float)

# joint4 range is -3.07 to -0.07 — zero is OUTSIDE it, so always init from HOME.
HOME        = np.array([0.0, 0.0,   0.0, -1.5708, 0.0, 1.5708, -0.7854])
GRASP_READY = np.array([0.0, 0.529, 0.0, -1.98,   0.0, 2.495,  -0.75  ])
_DESIRED_Z   = np.array([0.0, 0.0, -1.0])
GRASP_OFFSET = 0.10   # hand-body → finger-pad distance (0.0584 + 0.0445 m)

# =============================
# FIND CUBES
# =============================

name_to_cube = {}
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name and name.startswith("cube"):
        name_to_cube[name] = model.body(name).id

# =============================
# HELPERS
# =============================

def clip_arm(q):
    return np.clip(q, arm_ranges[:, 0], arm_ranges[:, 1])


def cube_null_ref(pos):
    """GRASP_READY with joint1 aimed at pos's bearing from arm base."""
    dx = pos[0] - BASE_XY[0]
    dy = pos[1] - BASE_XY[1]
    j1 = float(np.clip(np.arctan2(dy, dx), arm_ranges[0, 0], arm_ranges[0, 1]))
    ref = GRASP_READY.copy()
    ref[0] = j1
    return ref

# =============================
# IK CONTROLLER (6-DOF)
# =============================

def move_ee_to(target_xyz, grip=0.04, steps=250, null_ref=None):
    """6-DOF Jacobian IK — position + orientation (gripper z → [0,0,-1])."""
    if null_ref is None:
        null_ref = GRASP_READY
    q_target = data.qpos[arm_qpos_adr].copy()

    for _ in range(steps):
        mujoco.mj_forward(model, data)

        pos_error = target_xyz - data.xpos[ee_id]
        R         = data.xmat[ee_id].reshape(3, 3)
        ori_error = np.cross(R[:, 2], _DESIRED_Z)

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, ee_id)
        J_pos = jacp[:, arm_dof_adr]
        J_rot = jacr[:, arm_dof_adr]

        w    = 0.5
        J6   = np.vstack([J_pos, w * J_rot])
        err6 = np.concatenate([pos_error, w * ori_error])

        lam    = 0.05
        JJT    = J6 @ J6.T + lam * np.eye(6)
        J_pinv = J6.T @ np.linalg.solve(JJT, np.eye(6))

        dq_task  = J_pinv @ err6
        null_proj = np.eye(7) - J_pinv @ J6
        dq_null  = null_proj @ (null_ref - q_target) * 0.05

        dq       = np.clip(dq_task + dq_null, -0.08, 0.08)
        q_target = clip_arm(q_target + 0.015 * dq)

        data.ctrl[:7] = q_target
        data.ctrl[7]  = grip
        mujoco.mj_step(model, data)
        viewer.sync()


def set_grip(target, steps=200):
    """Arm holds; only gripper moves."""
    for _ in range(steps):
        data.ctrl[7] = target
        mujoco.mj_step(model, data)
        viewer.sync()


def drop():
    """
    Emergency release at the current arm position.
    Call this when a place was rejected/failed and the arm is still holding a cube,
    so the cube lands back on the table before the next replanning round.
    """
    set_grip(0.04, steps=300)
    for _ in range(120):   # let physics settle with cube on table
        mujoco.mj_step(model, data)
        viewer.sync()


def go_to_cube_ready(cube_pos, steps=400):
    """Smoothly pre-aim joint1 at cube_pos so IK starts correctly oriented."""
    target = cube_null_ref(cube_pos)
    for _ in range(steps):
        data.ctrl[:7] += 0.01 * (target - data.ctrl[:7])
        data.ctrl[7]   = 0.04
        mujoco.mj_step(model, data)
        viewer.sync()

# =============================
# STARTUP WARMUP (runs on import)
# =============================

# Phase 1: HOME — joint4 starts inside its valid range; cubes settle to z≈0.83.
data.qpos[arm_qpos_adr] = HOME
data.ctrl[:7] = HOME
data.ctrl[7]  = 0.04
for _ in range(400):
    mujoco.mj_step(model, data)
    viewer.sync()

# Phase 2: Transition joint6 from 1.57 → 2.495 (GRASP_READY).
# At IK step-size 0.015×clip 0.08 the IK alone can't close a 1.13 rad gap in time.
for _ in range(600):
    data.ctrl[:7] += 0.01 * (GRASP_READY - data.ctrl[:7])
    data.ctrl[7]   = 0.04
    mujoco.mj_step(model, data)
    viewer.sync()

# =============================
# ACTIONS
# =============================

_finger_qposadr = None   # cached on first use

def _finger_pos():
    """Current finger_joint1 position (0 = closed, 0.04 = fully open)."""
    global _finger_qposadr
    if _finger_qposadr is None:
        _finger_qposadr = model.joint("finger_joint1").qposadr[0]
    return float(data.qpos[_finger_qposadr])


def pick(obj):
    print(f"[exec] pick({obj})")

    if obj not in name_to_cube:
        print(f"[exec] unknown object: {obj}")
        return

    cube_id = name_to_cube[obj]

    # Keep gripper commanded open during the settle so the arm actuators
    # don't leave it in a partially-closed state from a previous place().
    for _ in range(150):
        data.ctrl[7] = 0.04
        mujoco.mj_step(model, data)
        viewer.sync()
    mujoco.mj_forward(model, data)
    cube_pos  = data.xpos[cube_id].copy()

    cube_ref  = cube_null_ref(cube_pos)
    grasp_xyz = cube_pos + np.array([0.0, 0.0, GRASP_OFFSET])

    go_to_cube_ready(cube_pos, steps=500)
    set_grip(0.04, steps=120)
    print(f"[exec] finger before close: {_finger_pos():.4f}")
    move_ee_to(cube_pos + [0, 0, 0.25], grip=0.04, steps=380, null_ref=cube_ref)
    move_ee_to(grasp_xyz,               grip=0.04,  steps=380, null_ref=cube_ref)
    set_grip(0.015, steps=320)
    for _ in range(80):
        mujoco.mj_step(model, data)
        viewer.sync()
    move_ee_to(cube_pos + [0, 0, 0.25], grip=0.015, steps=300, null_ref=cube_ref)


def place(x, y):
    """Place the held cube at world coordinates (x, y) on the table."""
    print(f"[exec] place({x:.3f}, {y:.3f})")

    # place_pos : final settled cube centre (table top 0.80 + half-size 0.03 = 0.83).
    # release_pos: cube centre when gripper opens — 6 cm above settled z.
    #   cube bottom = 0.89 - 0.03 = 0.86, which is 6 cm clear of the table (0.80).
    place_pos   = np.array([x, y, 0.83])
    release_pos = np.array([x, y, 0.89])
    place_ref   = cube_null_ref(place_pos)
    release_xyz = release_pos + np.array([0.0, 0.0, GRASP_OFFSET])  # hand at z=0.99

    move_ee_to(place_pos + [0, 0, 0.25], grip=0.015, steps=340, null_ref=place_ref)
    move_ee_to(release_xyz,              grip=0.015, steps=280, null_ref=place_ref)

    # Let arm dynamics settle BEFORE opening.
    # move_ee_to stops after a fixed step count with residual joint-velocity;
    # the high-kp arm actuators (kp=2000-5000) then vibrate the hand body,
    # creating dynamic loads on the cube that jam it against the fingertip pads.
    # 150 steps (~0.3 s) lets the arm damp to near-stationary.
    for _ in range(150):
        mujoco.mj_step(model, data)
        viewer.sync()

    print(f"[exec] finger before open: {_finger_pos():.4f}")
    set_grip(0.04, steps=600)
    print(f"[exec] finger after open:  {_finger_pos():.4f}")

    # Let cube fall 6 cm and settle on the table.
    for _ in range(200):
        mujoco.mj_step(model, data)
        viewer.sync()

    # Retreat straight up from the release point — safe, predictable path.
    move_ee_to(release_pos + [0, 0, 0.30], grip=0.04, steps=350, null_ref=place_ref)
