from __future__ import annotations

import csv
import copy
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
GENERATED_DIR = MODELS_DIR

CONSERVATIVE_MOTION_DEFAULTS = {
    "OMPL_TIME_LIMIT": "5.0",
    "OMPL_WAYPOINT_STEP": "0.012",
    "SETTLE_STEPS_PER_WAYPOINT": "14",
    "FINAL_SETTLE_STEPS": "40",
    "MIN_PICK_OBSTACLE_CLEARANCE": "0.18",
    "CAUTIOUS_OBSTACLE_CLEARANCE": "0.24",
    "ALLOW_MOVABLE_OBJECT_CONTACT": "false",
}

SCENE_ALIASES = {
    "group_no_obs": "group_no_obs",
    "group-no-obs": "group_no_obs",
    "group no obs": "group_no_obs",
    "group": "group_no_obs",
    "ungroup_no_obs": "ungroup_no_obs",
    "ungroup-no-obs": "ungroup_no_obs",
    "ungroup no obs": "ungroup_no_obs",
    "ungroup": "ungroup_no_obs",
    "group_obs": "group_obs",
    "group-obs": "group_obs",
    "group obs": "group_obs",
    "ungroup_obs": "ungroup_obs",
    "ungroup-obs": "ungroup_obs",
    "ungroup obs": "ungroup_obs",
}

GOAL_CENTER = (0.22, -0.06, 0.806)
GOAL_HALF_SIZE_X = 0.26
GOAL_HALF_SIZE_Y = 0.20
GOAL_EXCLUSION_MARGIN = 0.04
COMPACT_CYLINDER_CENTER_Z = 0.84
COMPACT_CYLINDER_SIZE = (0.026, 0.04)
TWO_ARM_BASES = {
    "left": (-0.4, -0.45, 0.8),
    "right": (-0.4, 0.45, 0.8),
}

VARIANT_OBJECTS = {
    "group_no_obs": {
        "cube1": (-0.02, -0.46, 0.83),
        "cube2": (0.10, -0.46, 0.83),
        "cube3": (0.22, -0.40, 0.83),
        "cube4": (0.32, -0.34, 0.83),
        "circle1": (-0.02, 0.24, COMPACT_CYLINDER_CENTER_Z),
        "circle2": (0.10, 0.32, COMPACT_CYLINDER_CENTER_Z),
        "circle3": (0.22, 0.26, COMPACT_CYLINDER_CENTER_Z),
        "circle4": (0.32, 0.34, COMPACT_CYLINDER_CENTER_Z),
    },
    "ungroup_no_obs": {
        "cube1": (-0.02, -0.46, 0.83),
        "circle1": (0.12, -0.42, COMPACT_CYLINDER_CENTER_Z),
        "cube2": (0.28, -0.40, 0.83),
        "circle2": (0.34, -0.32, COMPACT_CYLINDER_CENTER_Z),
        "cube3": (0.00, 0.24, 0.83),
        "circle3": (0.16, 0.32, COMPACT_CYLINDER_CENTER_Z),
        "cube4": (0.30, 0.24, 0.83),
        "circle4": (0.34, 0.34, COMPACT_CYLINDER_CENTER_Z),
    },
    "group_obs": {
        "cube1": (-0.02, -0.46, 0.83),
        "cube2": (0.10, -0.46, 0.83),
        "cube3": (0.22, -0.40, 0.83),
        "cube4": (0.32, -0.34, 0.83),
        "circle1": (-0.02, 0.24, COMPACT_CYLINDER_CENTER_Z),
        "circle2": (0.10, 0.32, COMPACT_CYLINDER_CENTER_Z),
        "circle3": (0.22, 0.26, COMPACT_CYLINDER_CENTER_Z),
        "circle4": (0.32, 0.34, COMPACT_CYLINDER_CENTER_Z),
    },
    "ungroup_obs": {
        "cube1": (-0.02, -0.46, 0.83),
        "circle1": (0.12, -0.42, COMPACT_CYLINDER_CENTER_Z),
        "cube2": (0.28, -0.40, 0.83),
        "circle2": (0.34, -0.32, COMPACT_CYLINDER_CENTER_Z),
        "cube3": (0.00, 0.24, 0.83),
        "circle3": (0.16, 0.32, COMPACT_CYLINDER_CENTER_Z),
        "cube4": (0.30, 0.24, 0.83),
        "circle4": (0.34, 0.34, COMPACT_CYLINDER_CENTER_Z),
    },
}

OBSTACLE_POSITIONS = {
    "obstacle1": (0.08, -0.30, 0.89),
    "obstacle2": (0.32, 0.19, 0.89),
}


def normalize_scene_key(raw: str | Iterable[str] | None) -> str:
    if raw is None:
        return "group_no_obs"
    if isinstance(raw, str):
        key = raw
    else:
        key = " ".join(raw)
    key = " ".join(key.strip().lower().replace("-", " ").replace("_", " ").split())
    normalized = SCENE_ALIASES.get(key) or SCENE_ALIASES.get(key.replace(" ", "_"))
    if normalized is None:
        valid = ", ".join(sorted({"group no obs", "ungroup no obs", "group obs", "ungroup obs"}))
        raise ValueError(f"unknown --object '{key}'. Valid: {valid}")
    return normalized


def prepare_scene_variant(raw: str | Iterable[str] | None) -> Path:
    scene_key = normalize_scene_key(raw)
    _validate_variant(scene_key)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GENERATED_DIR / f"panda_{scene_key}.xml"

    tree = ET.parse(MODELS_DIR / "panda.xml")
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("models/panda.xml has no worldbody")

    removable_prefixes = ("cube", "circle", "obstacle", "vase", "glass", "ceramic")
    for body in list(worldbody.findall("body")):
        name = body.get("name", "")
        if name == "goal_area" or name.startswith(removable_prefixes):
            worldbody.remove(body)

    link0_index = 0
    for idx, body in enumerate(list(worldbody)):
        if body.tag == "body" and body.get("name") == "link0":
            link0_index = idx
            break

    inserts = [_goal_area_body()]
    for object_name, pos in VARIANT_OBJECTS[scene_key].items():
        inserts.append(_movable_body(object_name, pos))
    if scene_key in {"group_obs", "ungroup_obs"}:
        for obstacle_name, pos in OBSTACLE_POSITIONS.items():
            inserts.append(_obstacle_body(obstacle_name, pos))

    for offset, body in enumerate(inserts):
        worldbody.insert(link0_index + offset, body)

    _indent(root)
    tmp_path = out_path.with_suffix(".tmp")
    tree.write(tmp_path, encoding="utf-8", xml_declaration=False)
    tmp_path.replace(out_path)
    return out_path


def prepare_two_arm_scene_variant(raw: str | Iterable[str] | None) -> Path:
    scene_key = normalize_scene_key(raw)
    _validate_variant(scene_key)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GENERATED_DIR / f"panda_two_arm_{scene_key}.xml"

    tree = ET.parse(MODELS_DIR / "panda.xml")
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("models/panda.xml has no worldbody")

    removable_prefixes = ("cube", "circle", "obstacle", "vase", "glass", "ceramic")
    for body in list(worldbody.findall("body")):
        name = body.get("name", "")
        if name == "goal_area" or name.startswith(removable_prefixes):
            worldbody.remove(body)

    left_arm = None
    link0_index = 0
    for idx, body in enumerate(list(worldbody)):
        if body.tag == "body" and body.get("name") == "link0":
            left_arm = body
            link0_index = idx
            break
    if left_arm is None:
        raise RuntimeError("models/panda.xml has no link0 arm body")

    left_arm.set("pos", _fmt_vec(TWO_ARM_BASES["left"]))
    right_arm = copy.deepcopy(left_arm)
    right_arm.set("pos", _fmt_vec(TWO_ARM_BASES["right"]))
    _prefix_robot_tree(right_arm, "right_")

    inserts = [_goal_area_body()]
    for object_name, pos in VARIANT_OBJECTS[scene_key].items():
        inserts.append(_movable_body(object_name, pos))
    if scene_key in {"group_obs", "ungroup_obs"}:
        for obstacle_name, pos in OBSTACLE_POSITIONS.items():
            inserts.append(_obstacle_body(obstacle_name, pos))

    for offset, body in enumerate(inserts):
        worldbody.insert(link0_index + offset, body)
    worldbody.insert(link0_index + len(inserts) + 1, right_arm)

    _duplicate_right_arm_controls(root)
    _remove_keyframes(root)

    _indent(root)
    tmp_path = out_path.with_suffix(".tmp")
    tree.write(tmp_path, encoding="utf-8", xml_declaration=False)
    tmp_path.replace(out_path)
    return out_path


def write_summary_csv(task_name: str, scene_key: str, summary: dict, log_dir: str | Path = "logs") -> Path:
    out_dir = Path(log_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{task_name}_{scene_key}_{timestamp}.csv"
    row = {
        "task": task_name,
        "scene": scene_key,
        "success": summary.get("success"),
        "objects_moved": summary.get("objects_moved"),
        "objects_total": summary.get("objects_total"),
        "failed_count": len(summary.get("failed", [])),
        "failed_json": json.dumps(summary.get("failed", []), ensure_ascii=False),
        "duration_ms": summary.get("duration_ms"),
        "llm_used": "false",
    }
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return out_path


def make_event_log_path(task_name: str, scene_key: str, log_dir: str | Path = "logs") -> Path:
    out_dir = Path(log_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return out_dir / f"{task_name}_{scene_key}_{timestamp}_events.csv"


def apply_conservative_motion_defaults() -> None:
    """Use slower, stricter motion defaults unless the caller overrides them."""
    import os

    for name, value in CONSERVATIVE_MOTION_DEFAULTS.items():
        os.environ.setdefault(name, value)


def _validate_variant(scene_key: str) -> None:
    for name, pos in VARIANT_OBJECTS[scene_key].items():
        if _inside_goal_area(pos[0], pos[1]):
            raise RuntimeError(
                f"{scene_key}: initial object {name} is inside goal area at {(pos[0], pos[1])}"
            )


def _inside_goal_area(x: float, y: float) -> bool:
    gx, gy, _ = GOAL_CENTER
    return (
        gx - GOAL_HALF_SIZE_X - GOAL_EXCLUSION_MARGIN
        <= x
        <= gx + GOAL_HALF_SIZE_X + GOAL_EXCLUSION_MARGIN
        and gy - GOAL_HALF_SIZE_Y - GOAL_EXCLUSION_MARGIN
        <= y
        <= gy + GOAL_HALF_SIZE_Y + GOAL_EXCLUSION_MARGIN
    )


def _goal_area_body() -> ET.Element:
    return ET.fromstring(
        """
        <body name="goal_area" pos="0.22 -0.06 0.806">
          <geom name="goal_area_base" type="box" size="0.26 0.20 0.003" rgba="0.05 0.35 0.95 0.22" contype="0" conaffinity="0"/>
          <geom name="goal_square_row" type="box" size="0.24 0.025 0.004" pos="0 -0.065 0.004" rgba="0.95 0.25 0.15 0.35" contype="0" conaffinity="0"/>
          <geom name="goal_circle_row" type="box" size="0.24 0.025 0.004" pos="0 0.065 0.004" rgba="0.1 0.8 0.45 0.35" contype="0" conaffinity="0"/>
        </body>
        """
    )


def _movable_body(name: str, pos: tuple[float, float, float]) -> ET.Element:
    rgba = {
        "cube1": "1 0 0 1",
        "cube2": "0 1 0 1",
        "cube3": "0 0 1 1",
        "cube4": "1 1 0 1",
        "circle1": "0.0 0.9 0.9 1",
        "circle2": "0.1 0.7 1.0 1",
        "circle3": "0.2 1.0 0.45 1",
        "circle4": "0.3 0.9 0.2 1",
    }.get(name, "1 1 1 1")
    if name.startswith("cube"):
        geom = f'<geom type="box" size="0.03 0.03 0.03" mass="0.1" friction="2 1 0.5" contype="1" conaffinity="1" rgba="{rgba}"/>'
    else:
        radius, half_height = COMPACT_CYLINDER_SIZE
        geom = f'<geom type="cylinder" size="{radius} {half_height}" mass="0.08" friction="3 1.5 0.8" contype="1" conaffinity="1" rgba="{rgba}"/>'
    return ET.fromstring(
        f"""
        <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">
          <joint type="free"/>
          {geom}
        </body>
        """
    )


def _obstacle_body(name: str, pos: tuple[float, float, float]) -> ET.Element:
    return ET.fromstring(
        f"""
        <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">
          <joint type="free"/>
          <geom type="cylinder" size="0.035 0.085" mass="0.4" friction="2 1 0.5" rgba="0.9 0.75 0.2 0.75" contype="1" conaffinity="1"/>
        </body>
        """
    )


def _fmt_vec(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:g}" for value in values)


def _prefix_robot_tree(elem: ET.Element, prefix: str) -> None:
    name = elem.get("name")
    if name:
        elem.set("name", f"{prefix}{name}")
    for child in elem:
        _prefix_robot_tree(child, prefix)


def _duplicate_right_arm_controls(root: ET.Element) -> None:
    actuator = root.find("actuator")
    if actuator is not None:
        for ctrl in list(actuator):
            duplicated = copy.deepcopy(ctrl)
            joint_name = duplicated.get("joint")
            if joint_name:
                duplicated.set("joint", f"right_{joint_name}")
            name = duplicated.get("name")
            if name:
                duplicated.set("name", f"right_{name}")
            actuator.append(duplicated)

    tendon = root.find("tendon")
    if tendon is not None:
        for fixed in list(tendon):
            duplicated = copy.deepcopy(fixed)
            name = duplicated.get("name")
            duplicated.set("name", f"right_{name or 'split'}")
            for joint in duplicated.findall(".//joint"):
                joint_name = joint.get("joint")
                if joint_name:
                    joint.set("joint", f"right_{joint_name}")
            tendon.append(duplicated)

    equality = root.find("equality")
    if equality is not None:
        for constraint in list(equality):
            duplicated = copy.deepcopy(constraint)
            for attr in ("joint1", "joint2", "body1", "body2"):
                value = duplicated.get(attr)
                if value:
                    duplicated.set(attr, f"right_{value}")
            equality.append(duplicated)

    contact = root.find("contact")
    if contact is not None:
        for exclude in list(contact):
            duplicated = copy.deepcopy(exclude)
            for attr in ("body1", "body2"):
                value = duplicated.get(attr)
                if value:
                    duplicated.set(attr, f"right_{value}")
            contact.append(duplicated)


def _remove_keyframes(root: ET.Element) -> None:
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)


def _indent(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i
