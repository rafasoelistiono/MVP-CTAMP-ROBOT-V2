import numpy as np
import mujoco

# ---------------------------------------------------------------------------
# Constants derived from panda.xml
# ---------------------------------------------------------------------------

ARM_BASE_XY = np.array([-0.4, 0.0])

# Empirically confirmed reach envelope for this setup.
ARM_MIN_REACH = 0.30   # m — arm can't reach very close targets
ARM_MAX_REACH = 0.82   # m — Panda max reach from base

# Table geom: size="0.6 0.8 0.05" pos="0 0 0.75" → top at z=0.80.
# Safety margin of 0.05 m kept from each edge so cubes don't hang off.
TABLE_X = (-0.55, 0.55)
TABLE_Y = (-0.75, 0.75)

CUBE_Z_ON_TABLE  = 0.83   # settled cube centre (table top 0.80 + half-size 0.03)
CUBE_CLEARANCE   = 0.10   # min XY gap between any two cube centres

# Colour map — order matches rgba in panda.xml
CUBE_COLORS = {
    "cube1": "red",
    "cube2": "green",
    "cube3": "blue",
    "cube4": "yellow",
    "cube5": "white",
    "cube6": "orange",
    "cube7": "purple",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dist_from_arm(x, y):
    return float(np.linalg.norm(np.array([x, y]) - ARM_BASE_XY))


def is_reachable(x, y):
    """Return True if (x, y) is within the arm's reach and on the table."""
    d = _dist_from_arm(x, y)
    on_table = TABLE_X[0] <= x <= TABLE_X[1] and TABLE_Y[0] <= y <= TABLE_Y[1]
    return on_table and ARM_MIN_REACH <= d <= ARM_MAX_REACH


# ---------------------------------------------------------------------------
# Scene state for the LLM
# ---------------------------------------------------------------------------

def get_scene(model, data, name_to_cube):
    """
    Return a JSON-serialisable dict describing the current simulation state.
    This is the full context given to the LLM at every planning call.
    """
    mujoco.mj_forward(model, data)

    objects = []
    for name in sorted(name_to_cube):
        cube_id = name_to_cube[name]
        pos = data.xpos[cube_id]
        objects.append({
            "name":  name,
            "color": CUBE_COLORS.get(name, "unknown"),
            "x":     round(float(pos[0]), 3),
            "y":     round(float(pos[1]), 3),
            "distance_from_arm_base_m": round(_dist_from_arm(pos[0], pos[1]), 3),
        })

    return {
        "arm_base_xy":       list(ARM_BASE_XY),
        "arm_reach_min_m":   ARM_MIN_REACH,
        "arm_reach_max_m":   ARM_MAX_REACH,
        "table_x_range":     list(TABLE_X),
        "table_y_range":     list(TABLE_Y),
        "cube_z_on_table":   CUBE_Z_ON_TABLE,
        "objects":           objects,
    }
