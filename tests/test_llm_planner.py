"""
Tests for the LLM planner -- no API calls. A ScriptedClient stands in for the
model so the full parse -> validate -> act -> harness path is exercised offline.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arena import Action, TaskSpec, run  # noqa: E402
from arena.environment_fake import FakeEnvironment  # noqa: E402
from arena.evaluators import TidyEvaluator  # noqa: E402
from arena.llm import LLMPlanner, ScriptedClient, extract_json_object, infer_provider  # noqa: E402

GOAL_X, GOAL_Y = 0.22, -0.06
CUBE_ROW_Y, CIRCLE_ROW_Y = GOAL_Y - 0.065, GOAL_Y + 0.065


def _obs_from_user(user: str) -> dict:
    after = user.split("Observation (JSON):\n", 1)[1]
    return json.loads(after.split("\n\nWhat is your single next action?", 1)[0])


def _tidy_world():
    objects = {
        "cube1": (-0.02, -0.46, 0.83),
        "cube2": (0.10, -0.46, 0.83),
        "circle1": (0.12, 0.30, 0.84),
        "circle2": (0.20, 0.32, 0.84),
    }
    classes = {"cube1": "cube", "cube2": "cube", "circle1": "circle", "circle2": "circle"}
    regions = {"cube_row": (GOAL_X, CUBE_ROW_Y, 0.80), "circle_row": (GOAL_X, CIRCLE_ROW_Y, 0.80)}
    env = FakeEnvironment(objects=objects, regions=regions, table_z=0.80, classes=classes)
    evaluator = TidyEvaluator(list(objects), classes, GOAL_X, CUBE_ROW_Y, CIRCLE_ROW_Y)
    task = TaskSpec("tidy_table", "ungroup_no_obs", "Sort objects by type.", max_steps=20)
    return task, env, evaluator


# --- JSON extraction ------------------------------------------------------- #

def test_infer_provider_routes_by_model_name():
    assert infer_provider("claude-opus-4-8") == "anthropic"
    assert infer_provider("claude-sonnet-4-6") == "anthropic"
    assert infer_provider("gpt-4o") == "openai"
    assert infer_provider("gpt-4.1") == "openai"
    assert infer_provider("o3-mini") == "openai"


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json_object('```json\n{"action": "stop"}\n```') == {"action": "stop"}
    assert extract_json_object('Sure: {"action": "stack", "object": "a", "on": "b"} done') == {
        "action": "stack", "object": "a", "on": "b",
    }
    with pytest.raises(ValueError):
        extract_json_object("no json here")


# --- a competent model solves the task ------------------------------------- #

def _good_responder(system, user):
    obs = _obs_from_user(user)
    for o in obs["objects"]:
        if o["placed"] or o["held"] or o["reach"] == "HARD" or o["obstacle"] == "TOO_CLOSE":
            continue
        region = "cube_row" if o["class"] == "cube" else "circle_row"
        return json.dumps({"action": "place", "object": o["id"], "region": region})
    return json.dumps({"action": "stop", "reason": "done"})


def test_llm_planner_drives_tidy_to_goal():
    task, env, evaluator = _tidy_world()
    planner = LLMPlanner(ScriptedClient(_good_responder))
    record = run(task, planner, env, evaluator)
    assert record.success is True
    assert record.objects_placed == 4
    assert record.planner_name.startswith("llm:")


# --- recovers from one bad reply via re-ask -------------------------------- #

def test_llm_planner_reasks_then_succeeds():
    calls = {"n": 0}

    def flaky(system, user):
        calls["n"] += 1
        if calls["n"] == 1:
            return "I think we should... (no json)"          # unparseable
        if calls["n"] == 2:
            return json.dumps({"action": "place", "object": "ghost", "region": "cube_row"})  # hallucinated id
        return _good_responder(system, user)

    task, env, evaluator = _tidy_world()
    planner = LLMPlanner(ScriptedClient(flaky), max_retries=5)
    record = run(task, planner, env, evaluator)
    assert record.success is True  # recovered after the two bad replies


# --- a persistently broken model never crashes the run --------------------- #

def test_llm_planner_falls_back_to_stop_on_garbage():
    task, env, evaluator = _tidy_world()
    planner = LLMPlanner(ScriptedClient(lambda s, u: "never valid json"), max_retries=2)
    record = run(task, planner, env, evaluator)
    assert record.success is False
    assert record.objects_placed == 0  # stopped safely, no crash


def test_harness_survives_a_planner_that_raises():
    class Boom:
        name = "boom"

        def reset(self, task, obs):
            return None

        def act(self, obs):
            raise RuntimeError("kaboom")

    task, env, evaluator = _tidy_world()
    record = run(task, Boom(), env, evaluator)  # must not raise
    assert record.success is False
