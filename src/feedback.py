import numpy as np
import mujoco

# Cube z when held by gripper after lift ≈ 0.97–0.98.
# Cube z on table (settled)              ≈ 0.83.
# Cube z stacked on another cube         ≈ 0.89  (0.83 + 2×half-size 0.03).
#
# PLACED_Z_THRESHOLD must sit between 0.83 (table) and 0.89 (stacked):
#   0.87 gives a 4 cm margin above the table resting height and correctly
#   rejects stacked cubes.  Previous value of 0.95 let stacking pass silently.
HELD_Z_THRESHOLD   = 0.90   # pick:  cube must be ABOVE this to count as gripped
PLACED_Z_THRESHOLD = 0.87   # place: cube must be BELOW this to count as on table

# Acceptable XY distance between target and actual cube position after place.
PLACE_TOLERANCE_M = 0.12


def check_pick(model, data, cube_id):
    """
    Call immediately after executor.pick() returns.
    Returns (success: bool, cube_z: float).
    success=True means the cube is elevated — it's in the gripper.
    """
    mujoco.mj_forward(model, data)
    z = float(data.xpos[cube_id][2])
    return z > HELD_Z_THRESHOLD, round(z, 3)


def check_place(model, data, cube_id, target_x, target_y):
    """
    Call immediately after executor.place() returns.
    Returns (success: bool, actual_pos: [x, y, z]).
    success=True means the cube is on the table and near the target.
    Uses PLACED_Z_THRESHOLD (0.95) so a cube still bouncing after release
    (z ≈ 0.83–0.93) is not mistaken for "still airborne".
    """
    mujoco.mj_forward(model, data)
    pos = data.xpos[cube_id]
    actual = [round(float(pos[0]), 3),
              round(float(pos[1]), 3),
              round(float(pos[2]), 3)]

    on_table  = float(pos[2]) < PLACED_Z_THRESHOLD
    near_goal = float(np.linalg.norm(
        pos[:2] - np.array([target_x, target_y])
    )) < PLACE_TOLERANCE_M

    return on_table and near_goal, actual
