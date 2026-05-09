import mujoco
import mujoco.viewer
from pathlib import Path

_MODELS = Path(__file__).parent.parent / "models"
model = mujoco.MjModel.from_xml_path(str(_MODELS / "scene.xml"))
data = mujoco.MjData(model)

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()