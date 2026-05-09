import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

_MODELS = Path(__file__).parent.parent / "models"
model = mujoco.MjModel.from_xml_path(str(_MODELS / "panda.xml"))
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)

# set cube pose manually
data.qpos[7:10] = [0.2, 0, 0.78]
data.qpos[10:14] = [1, 0, 0, 0]

mujoco.mj_forward(model, data)

cube_id = model.body("cube1").id
ee_id = model.body("hand").id

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.5
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -30
    viewer.cam.lookat[:] = [0, 0, 0.7]

    step = 0
    auto_mode = True

    while viewer.is_running():
        step += 1

        target_down = np.array([
            0.0,
            0.599,
            0.0,
            -2.07,
            0.0,
            2.7,
            -0.74
        ])

        alpha = 0.005

        # -------- AUTO PHASE --------
        if auto_mode:
            data.ctrl[:7] = data.ctrl[:7] + alpha * (target_down - data.ctrl[:7])
            data.ctrl[7] = 255  # open gripper

            # stop auto after some time
            if step > 1000:
                auto_mode = False
                print("Now you can move joints manually")

        # -------- MANUAL PHASE --------
        else:
            # DO NOTHING → sliders take control
            pass

        mujoco.mj_step(model, data)
        viewer.sync()