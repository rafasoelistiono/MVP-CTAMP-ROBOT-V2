import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

_MODELS = Path(__file__).parent.parent / "models"
model = mujoco.MjModel.from_xml_path(str(_MODELS / "panda.xml"))
data = mujoco.MjData(model)

# Reset simulation
mujoco.mj_resetData(model, data)

# Set cube pose (on table)
data.qpos[7:10] = [0.2, 0, 0.78]
data.qpos[10:14] = [1, 0, 0, 0]

mujoco.mj_forward(model, data)

# IDs
cube_id = model.body("cube1").id
ee_id = model.body("hand").id

# ===== YOUR RECORDED POSES =====
target_above = np.array([
    0.0,
    0.529,
    0.0,
    -1.98,
    0.0,
    2.495,
    -0.75
])

target_down = np.array([
    0.0,
    0.599,
    0.0,
    -2.07,
    0.0,
    2.7,
    -0.74
])

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.5
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -30
    viewer.cam.lookat[:] = [0, 0, 0.7]

    step = 0

    while viewer.is_running():
        step += 1

        alpha = 0.005  # smoothness (reduce if still too fast)

        # ===== FSM PIPELINE =====

        # 1. Move above cube
        if step < 1000:
            target = target_above
            grip = 255   # OPEN

        # 2. Move down
        elif step < 1400:
            target = target_down
            grip = 255   # still OPEN

        # 3. Close gripper (wait here for stable grasp)
        elif step < 1700:
            target = target_down
            grip = 0     # CLOSE

        # 4. Lift
        else:
            target = target_above
            grip = 0     # HOLD

        # ===== Smooth joint control =====
        data.ctrl[:7] = data.ctrl[:7] + alpha * (target - data.ctrl[:7])

        # ===== Gripper control =====
        data.ctrl[7] = grip

        # Step simulation
        mujoco.mj_step(model, data)
        viewer.sync()