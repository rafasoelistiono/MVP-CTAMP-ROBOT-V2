"""
Pure-Python tests for the arena seam: no MuJoCo, no OMPL.

These prove the closed-loop contract composes correctly -- a planner drives an
Environment through the harness and is scored by an Evaluator -- including
closed-loop recovery after a disturbance. The real MujocoEnvironment is exercised
separately by the parity run.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arena import Action, ActionParseError, TaskSpec, parse_action, run  # noqa: E402
from arena.contracts import ObjectView, WorldSnapshot  # noqa: E402
from arena.environment_fake import FakeEnvironment  # noqa: E402
from arena.evaluators import StackEvaluator, TidyEvaluator  # noqa: E402
from arena.planners import ScriptedStackPlanner, ScriptedTidyPlanner  # noqa: E402

ORDER = ["cube1", "cube2", "cube3", "cube4"]
REGION = "tower_base"
REGIONS = {REGION: (0.22, -0.06, 0.806)}
TABLE_Z = 0.80


def _scene(cubes=ORDER):
    spread = {
        "cube1": (-0.02, -0.46, 0.83),
        "cube2": (0.10, -0.46, 0.83),
        "cube3": (0.22, -0.40, 0.83),
        "cube4": (0.32, -0.34, 0.83),
    }
    return {c: spread[c] for c in cubes}


def _make(env_kwargs=None, max_steps=24):
    env = FakeEnvironment(
        objects=_scene(),
        regions=REGIONS,
        table_z=TABLE_Z,
        **(env_kwargs or {}),
    )
    planner = ScriptedStackPlanner(ORDER, region=REGION)
    evaluator = StackEvaluator(ORDER, REGION)
    task = TaskSpec("stack_cubes", "group_no_obs", "Build one 4-cube tower.", max_steps=max_steps)
    return task, planner, env, evaluator


# --- action parsing -------------------------------------------------------- #

def test_action_round_trip():
    a = Action.stack("cube2", on="cube1")
    assert parse_action(a.to_dict()) == a
    p = parse_action({"action": "place", "object": "cube1", "region": "tower_base"})
    assert p == Action.place("cube1", "tower_base")


def test_action_parse_rejects_garbage():
    with pytest.raises(ActionParseError):
        parse_action({"action": "teleport", "object": "cube1"})
    with pytest.raises(ActionParseError):
        parse_action({"action": "stack", "object": "cube2"})  # missing 'on'


# --- closed-loop happy path ------------------------------------------------ #

def test_scripted_planner_builds_tower():
    task, planner, env, evaluator = _make()
    record = run(task, planner, env, evaluator)
    assert record.success is True
    assert record.objects_placed == 4
    assert record.score == 1.0
    assert record.steps_used == 4  # place base + 3 stacks, no wasted moves
    assert record.failures == []


# --- missing object -> graceful partial credit ----------------------------- #

def test_missing_required_cube_stops_with_partial_score():
    env = FakeEnvironment(objects=_scene(["cube1", "cube2", "cube4"]), regions=REGIONS, table_z=TABLE_Z)
    planner = ScriptedStackPlanner(ORDER, region=REGION)
    evaluator = StackEvaluator(ORDER, REGION)
    task = TaskSpec("stack_cubes", "group_no_obs", "Build one 4-cube tower.")
    record = run(task, planner, env, evaluator)
    assert record.success is False
    assert record.objects_placed == 2  # cube1, cube2 before the missing cube3 blocks
    assert any(f["object_id"] == "cube3" and not f["in_scene"] for f in record.failures)


# --- closed-loop recovery after a disturbance ------------------------------ #

def test_recovers_after_base_knocked_off():
    def knock_base_once(env, action, calls):
        if calls == 4:  # right as the last cube is being stacked, base gets nudged
            env.positions["cube1"] = (-0.30, -0.40, TABLE_Z + 0.03)
        return None

    task, planner, env, evaluator = _make(env_kwargs={"on_execute": knock_base_once})
    record = run(task, planner, env, evaluator)
    assert record.success is True
    assert record.objects_placed == 4
    assert record.steps_used > 4  # had to notice and redo work


# --- tidy / sort-by-type --------------------------------------------------- #

TIDY_GOAL_X = 0.22
_GOAL_Y = -0.06
TIDY_CUBE_ROW_Y = _GOAL_Y - 0.065     # cubes go below goal centre
TIDY_CIRCLE_ROW_Y = _GOAL_Y + 0.065   # circles go above goal centre


def _tidy_setup():
    objects = {
        "cube1": (-0.02, -0.46, 0.83),
        "cube2": (0.10, -0.46, 0.83),
        "circle1": (0.12, 0.30, 0.84),
        "circle2": (0.20, 0.32, 0.84),
    }
    classes = {"cube1": "cube", "cube2": "cube", "circle1": "circle", "circle2": "circle"}
    regions = {
        "cube_row": (TIDY_GOAL_X, TIDY_CUBE_ROW_Y, 0.80),
        "circle_row": (TIDY_GOAL_X, TIDY_CIRCLE_ROW_Y, 0.80),
    }
    env = FakeEnvironment(objects=objects, regions=regions, table_z=0.80, classes=classes)
    planner = ScriptedTidyPlanner()
    evaluator = TidyEvaluator(list(objects), classes, TIDY_GOAL_X, TIDY_CUBE_ROW_Y, TIDY_CIRCLE_ROW_Y)
    task = TaskSpec("tidy_table", "ungroup_no_obs", "Sort by type.", max_steps=20)
    return task, planner, env, evaluator


def test_tidy_sorts_every_object_by_type():
    task, planner, env, evaluator = _tidy_setup()
    record = run(task, planner, env, evaluator)
    assert record.success is True
    assert record.objects_placed == 4
    assert record.score == 1.0


def test_tidy_evaluator_rejects_object_in_wrong_row():
    ev = TidyEvaluator(["cube1", "circle1"], {"cube1": "cube", "circle1": "circle"},
                       TIDY_GOAL_X, TIDY_CUBE_ROW_Y, TIDY_CIRCLE_ROW_Y)
    snap = WorldSnapshot(
        objects=(
            # cube in its own (cube) row -> tidied
            ObjectView("cube1", "cube", (TIDY_GOAL_X, TIDY_CUBE_ROW_Y, 0.83),
                       "OK", "CLEAR", True, False, False),
            # circle dumped in the CUBE row -> not tidied (sorting matters)
            ObjectView("circle1", "circle", (TIDY_GOAL_X, TIDY_CUBE_ROW_Y, 0.84),
                       "OK", "CLEAR", True, False, False),
        ),
        obstacles=(), regions={}, held_object=None,
    )
    status = ev.placed_status(snap)
    assert status["cube1"] is True
    assert status["circle1"] is False
    assert ev.is_goal_satisfied(snap) is False
