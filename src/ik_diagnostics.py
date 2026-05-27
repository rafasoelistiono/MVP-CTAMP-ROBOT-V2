from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


IK_SUCCESS = "success"
IK_ERROR_ABOVE_LIMIT = "ik_error_above_limit"
IK_ORIENTATION_ERROR_ABOVE_LIMIT = "ik_orientation_error_above_limit"
IK_UNREACHABLE = "ik_unreachable"
IK_JOINT_LIMIT_INVALID = "ik_joint_limit_invalid"
IK_GOAL_COLLISION_INVALID = "ik_goal_collision_invalid"
IK_GOAL_STATE_INVALID = "ik_goal_state_invalid"
OMPL_FAILED = "ompl_failed"
OMPL_TIMEOUT = "ompl_timeout"
EXECUTION_FAILED = "execution_failed"


@dataclass
class IKAttemptResult:
    q: Optional[np.ndarray]
    target_xyz: np.ndarray
    backend: str
    candidate_id: int
    seed_id: int
    pos_err: float
    ori_err: float
    iterations: int
    converged: bool
    joint_limit_valid: bool
    state_valid: Optional[bool]
    state_invalid_reason: Optional[str]
    failure_reason: str
    score: float


def joint_limits_valid(q: Sequence[float], lower: Sequence[float], upper: Sequence[float]) -> bool:
    q_arr = np.asarray(q, dtype=float)
    return bool(
        q_arr.shape == np.asarray(lower).shape
        and np.all(q_arr >= np.asarray(lower, dtype=float) - 1e-9)
        and np.all(q_arr <= np.asarray(upper, dtype=float) + 1e-9)
    )


def classify_ik_attempt(
    *,
    pos_err: float,
    ori_err: float,
    pos_limit: float,
    ori_limit: float,
    joint_limit_valid: bool,
    state_valid: Optional[bool],
    state_invalid_reason: Optional[str],
    converged: bool = True,
) -> str:
    if not converged and not np.isfinite(pos_err):
        return IK_UNREACHABLE
    if pos_err > pos_limit:
        return IK_ERROR_ABOVE_LIMIT
    if ori_err > ori_limit:
        return IK_ORIENTATION_ERROR_ABOVE_LIMIT
    if not joint_limit_valid:
        return IK_JOINT_LIMIT_INVALID
    if state_valid is False:
        reason = (state_invalid_reason or "").lower()
        if "contact" in reason or "collision" in reason:
            return IK_GOAL_COLLISION_INVALID
        return IK_GOAL_STATE_INVALID
    return IK_SUCCESS


def rank_ik_attempts(attempts: Sequence[IKAttemptResult]) -> list[IKAttemptResult]:
    return sorted(
        attempts,
        key=lambda item: (
            item.failure_reason != IK_SUCCESS,
            item.score,
            item.pos_err,
            item.ori_err,
            item.candidate_id,
            item.seed_id,
        ),
    )
