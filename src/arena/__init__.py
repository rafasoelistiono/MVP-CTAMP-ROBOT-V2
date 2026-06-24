"""
Arena: a model-agnostic benchmark seam for robotic high-level task planning.

The package separates three roles so that multiple planners (LLMs or baselines)
can be ranked fairly on the same tasks:

    Planner      -- decides WHAT to do        (the thing under test)
    Environment  -- DOES it, reports state    (shared, identical for all)
    Evaluator    -- JUDGES it                 (shared goal predicate + scoring)

See contracts.py for the interfaces and harness.run() for the closed loop.
"""

from .contracts import (  # noqa: F401
    Action,
    ActionParseError,
    ActionType,
    Environment,
    Evaluator,
    Observation,
    ObjectView,
    ObstacleView,
    Planner,
    RunRecord,
    StepResult,
    TaskSpec,
    WorldSnapshot,
    parse_action,
)
from .harness import build_observation, run  # noqa: F401
