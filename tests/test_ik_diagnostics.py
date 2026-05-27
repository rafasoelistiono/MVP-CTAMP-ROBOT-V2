from __future__ import annotations

import numpy as np

from ik_diagnostics import (
    IK_ERROR_ABOVE_LIMIT,
    IK_GOAL_COLLISION_INVALID,
    IK_GOAL_STATE_INVALID,
    IK_JOINT_LIMIT_INVALID,
    IK_ORIENTATION_ERROR_ABOVE_LIMIT,
    IK_SUCCESS,
    classify_ik_attempt,
    joint_limits_valid,
)


def test_numeric_position_error_classification():
    reason = classify_ik_attempt(
        pos_err=0.04,
        ori_err=0.10,
        pos_limit=0.02,
        ori_limit=0.35,
        joint_limit_valid=True,
        state_valid=True,
        state_invalid_reason=None,
    )
    assert reason == IK_ERROR_ABOVE_LIMIT


def test_orientation_error_classification():
    reason = classify_ik_attempt(
        pos_err=0.01,
        ori_err=0.60,
        pos_limit=0.02,
        ori_limit=0.35,
        joint_limit_valid=True,
        state_valid=True,
        state_invalid_reason=None,
    )
    assert reason == IK_ORIENTATION_ERROR_ABOVE_LIMIT


def test_collision_invalid_goal_is_not_numeric_ik_error():
    reason = classify_ik_attempt(
        pos_err=0.01,
        ori_err=0.10,
        pos_limit=0.02,
        ori_limit=0.35,
        joint_limit_valid=True,
        state_valid=False,
        state_invalid_reason="robot-env contact: cube4/8 <-> hand/77",
    )
    assert reason == IK_GOAL_COLLISION_INVALID


def test_state_invalid_goal_classification():
    reason = classify_ik_attempt(
        pos_err=0.01,
        ori_err=0.10,
        pos_limit=0.02,
        ori_limit=0.35,
        joint_limit_valid=True,
        state_valid=False,
        state_invalid_reason="bounds mismatch",
    )
    assert reason == IK_GOAL_STATE_INVALID


def test_joint_limit_validation():
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    assert joint_limits_valid([0.0, 0.5], lower, upper)
    assert not joint_limits_valid([0.0, 1.5], lower, upper)


def test_success_classification():
    reason = classify_ik_attempt(
        pos_err=0.01,
        ori_err=0.10,
        pos_limit=0.02,
        ori_limit=0.35,
        joint_limit_valid=True,
        state_valid=True,
        state_invalid_reason=None,
    )
    assert reason == IK_SUCCESS
