import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

_MODELS = Path(__file__).parent.parent / "models"
model = mujoco.MjModel.from_xml_path(str(_MODELS / "panda.xml"))
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

arm_joint_names = [f"joint{i}" for i in range(1, 8)]
arm_qpos_adr = np.array([model.joint(n).qposadr[0] for n in arm_joint_names], dtype=int)
arm_dof_adr  = np.array([model.joint(n).dofadr[0] for n in arm_joint_names], dtype=int)
arm_ranges   = np.array([model.joint(n).range for n in arm_joint_names], dtype=float)

ee_id    = model.body("hand").id
BASE_XY  = np.array([-0.4, 0.0])   # arm base position in world XY

cube_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    for i in range(model.nbody)
    if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) and
       mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i).startswith("cube")
]
cube_ids = [model.body(name).id for name in cube_names]

HOME        = np.array([0.0, 0.0,   0.0, -1.5708, 0.0, 1.5708, -0.7854])
GRASP_READY = np.array([0.0, 0.529, 0.0, -1.98,   0.0, 2.495,  -0.75  ])

_DESIRED_Z   = np.array([0.0, 0.0, -1.0])
GRASP_OFFSET = 0.10   # hand-body → finger-pad distance along hand z (0.0584 + 0.0445 m)


def clip_arm(q):
    return np.clip(q, arm_ranges[:, 0], arm_ranges[:, 1])


def cube_null_ref(cube_pos):
    """
    GRASP_READY with joint1 aimed at the cube's bearing from the arm base.
    This gives every cube its own null-space target so the IK never has to
    fight joint1 back to 0 — critical for close or angled cubes like the blue one.
    """
    dx = cube_pos[0] - BASE_XY[0]
    dy = cube_pos[1] - BASE_XY[1]
    j1 = float(np.clip(np.arctan2(dy, dx), arm_ranges[0, 0], arm_ranges[0, 1]))
    ref = GRASP_READY.copy()
    ref[0] = j1
    return ref


def move_ee_to(target_xyz, grip=0.04, steps=250, null_ref=None, viewer=None):
    """
    6-DOF Jacobian IK (position + orientation).
    null_ref: the joint configuration the null-space attracts toward.
              Pass cube_null_ref(cube_pos) so joint1 is already aimed correctly.
    """
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
        J6   = np.vstack([J_pos,  w * J_rot])
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
        if viewer is not None:
            viewer.sync()


def set_grip(target, steps=200, viewer=None):
    """Arm holds; only gripper moves."""
    for _ in range(steps):
        data.ctrl[7] = target
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()


def go_to_cube_ready(cube_pos, viewer=None, steps=400):
    """
    Drive arm to GRASP_READY with joint1 pre-aimed at the next cube.
    Without this, the IK for a close/angled cube (like blue at (0,-0.2))
    starts with joint1=0 and the null-space fights the correction the whole time.
    """
    target = cube_null_ref(cube_pos)
    for _ in range(steps):
        data.ctrl[:7] += 0.01 * (target - data.ctrl[:7])
        data.ctrl[7]   = 0.04
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()


with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.5
    viewer.cam.azimuth  = 120
    viewer.cam.elevation = -30
    viewer.cam.lookat[:] = [0, 0, 0.7]

    # Phase 1: HOME — joint4 starts inside its valid range (-3.07 to -0.07).
    # Cubes settle from z=0.78 to z≈0.83 during these steps.
    data.qpos[arm_qpos_adr] = HOME
    data.ctrl[:7] = HOME
    data.ctrl[7]  = 0.04
    for _ in range(400):
        mujoco.mj_step(model, data)
        viewer.sync()

    # Phase 2: Transition joint6 from 1.57 → 2.495 (GRASP_READY).
    # Must do this before the loop; at step-size 0.015×clip 0.08 the IK can only
    # travel 0.0012 rad/step — not enough to close the 1.13 rad gap in 300 steps.
    for _ in range(600):
        data.ctrl[:7] += 0.01 * (GRASP_READY - data.ctrl[:7])
        data.ctrl[7]   = 0.04
        mujoco.mj_step(model, data)
        viewer.sync()

    # Phase 3: Pre-aim joint1 at the first cube before the loop starts.
    mujoco.mj_forward(model, data)
    go_to_cube_ready(data.xpos[cube_ids[0]].copy(), viewer=viewer, steps=300)

    # ── Pick-and-place loop ──────────────────────────────────────────────────
    for idx, cube_id in enumerate(cube_ids):
        mujoco.mj_forward(model, data)
        cube_pos  = data.xpos[cube_id].copy()

        # Place row at x=0.0, spread along y.
        # Previous x=0.3 caused idx=2 (0.83 m) and idx=3 (0.89 m) to exceed the
        # Panda's ~0.855 m max reach, making those IK targets unreachable.
        # At x=0.0: all four distances are 0.53–0.72 m — comfortably within reach.
        place_pos = np.array([0.0, 0.35 + 0.08 * idx, cube_pos[2]])

        # Two separate null-space references:
        #  cube_ref  — joint1 aimed at the CUBE  (pickup approach + lift)
        #  place_ref — joint1 aimed at PLACE pos (transit + lower + release)
        # Using the right reference for each phase stops the arm from fighting
        # itself when it needs to swing from the cube's side to the place side.
        cube_ref  = cube_null_ref(cube_pos)
        place_ref = cube_null_ref(place_pos)

        grasp_xyz   = cube_pos  + np.array([0.0, 0.0, GRASP_OFFSET])
        # Release 6 cm above settled position: cube bottom at z=cube_z+0.03,
        # 6 cm above table top (0.80) — fingers spread freely with no table friction.
        release_pos = place_pos + np.array([0.0, 0.0, 0.06])
        place_xyz   = release_pos + np.array([0.0, 0.0, GRASP_OFFSET])

        set_grip(0.04, steps=60,  viewer=viewer)
        move_ee_to(cube_pos  + [0, 0, 0.25],  grip=0.04,  steps=300, null_ref=cube_ref,  viewer=viewer)
        move_ee_to(grasp_xyz,                  grip=0.04,  steps=380, null_ref=cube_ref,  viewer=viewer)
        set_grip(0.015, steps=320, viewer=viewer)
        # Let physics settle with the cube gripped before lifting.
        for _ in range(80):
            mujoco.mj_step(model, data)
            viewer.sync()
        move_ee_to(cube_pos  + [0, 0, 0.25],  grip=0.015, steps=300, null_ref=cube_ref,  viewer=viewer)
        move_ee_to(place_pos + [0, 0, 0.25],  grip=0.015, steps=340, null_ref=place_ref, viewer=viewer)
        move_ee_to(place_xyz,                  grip=0.015, steps=280, null_ref=place_ref, viewer=viewer)
        # Let arm dynamics settle before opening — residual joint velocity from the
        # IK vibrates the hand body and jams the cube against the fingertip pads.
        for _ in range(150):
            mujoco.mj_step(model, data)
            viewer.sync()
        # Open gripper — cube hangs 6 cm above table, fingers spread freely.
        set_grip(0.04,  steps=600, viewer=viewer)
        for _ in range(200):
            mujoco.mj_step(model, data)
            viewer.sync()
        move_ee_to(release_pos + [0, 0, 0.30], grip=0.04, steps=350, null_ref=place_ref, viewer=viewer)

        # Between cubes: pre-aim joint1 at the NEXT cube so IK starts already oriented.
        if idx + 1 < len(cube_ids):
            mujoco.mj_forward(model, data)
            next_cube_pos = data.xpos[cube_ids[idx + 1]].copy()
            go_to_cube_ready(next_cube_pos, viewer=viewer, steps=400)
        else:
            go_to_cube_ready(cube_pos, viewer=viewer, steps=300)
