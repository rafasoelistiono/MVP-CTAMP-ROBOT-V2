from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Set

import mujoco
import numpy as np


DEFAULT_ROBOT_BODIES = (
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


@dataclass(frozen=True)
class CollisionReport:
    valid: bool
    geom1: Optional[int] = None
    geom2: Optional[int] = None
    body1: Optional[str] = None
    body2: Optional[str] = None

    @property
    def reason(self) -> str:
        if self.valid:
            return "valid"
        return f"robot-env contact: {self.body1}/{self.geom1} <-> {self.body2}/{self.geom2}"


class CollisionPolicy:
    """
    MuJoCo contact policy for OMPL state validity.

    The planner treats named robot bodies as the robot and every other non-ignored
    body as environment. A state is invalid when MuJoCo reports any contact between
    a robot geom and an environment geom.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        robot_body_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.model = model
        self.robot_body_names: Set[str] = set(robot_body_names or DEFAULT_ROBOT_BODIES)
        self.ignored_body_names: Set[str] = set()
        self.robot_geom_ids: Set[int] = set()
        self.env_geom_ids: Set[int] = set()
        self.env_body_ids: Set[int] = set()
        self.robot_body_ids: Set[int] = {
            self.model.body(name).id
            for name in self.robot_body_names
            if self._has_body(name)
        }
        self.refresh()

    def set_ignored_bodies(self, body_names: Optional[Iterable[str]]) -> None:
        self.ignored_body_names = set(body_names or [])
        self.refresh()

    def refresh(self) -> None:
        self.robot_geom_ids = set()
        self.env_geom_ids = set()
        self.env_body_ids = set()

        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            )
            if body_name is None:
                continue

            if body_name in self.robot_body_names:
                self.robot_geom_ids.add(geom_id)
            elif body_name not in self.ignored_body_names:
                self.env_geom_ids.add(geom_id)
                self.env_body_ids.add(body_id)

    def check_contacts(self, data: mujoco.MjData) -> CollisionReport:
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            geom1_robot = geom1 in self.robot_geom_ids
            geom2_robot = geom2 in self.robot_geom_ids
            geom1_env = geom1 in self.env_geom_ids
            geom2_env = geom2 in self.env_geom_ids

            if (geom1_robot and geom2_env) or (geom2_robot and geom1_env):
                body1 = self._body_name_for_geom(geom1)
                body2 = self._body_name_for_geom(geom2)
                return CollisionReport(
                    valid=False,
                    geom1=geom1,
                    geom2=geom2,
                    body1=body1,
                    body2=body2,
                )

        return CollisionReport(valid=True)

    def minimum_body_center_clearance(self, data: mujoco.MjData) -> float:
        if not self.env_body_ids or not self.robot_body_ids:
            return 1.0

        best = float("inf")
        for robot_body_id in self.robot_body_ids:
            robot_pos = data.xpos[robot_body_id]
            for env_body_id in self.env_body_ids:
                distance = float(np.linalg.norm(robot_pos - data.xpos[env_body_id]))
                if distance < best:
                    best = distance

        if not np.isfinite(best):
            return 1.0
        return best

    def _body_name_for_geom(self, geom_id: int) -> Optional[str]:
        body_id = int(self.model.geom_bodyid[geom_id])
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)

    def _has_body(self, name: str) -> bool:
        try:
            self.model.body(name)
            return True
        except KeyError:
            return False
