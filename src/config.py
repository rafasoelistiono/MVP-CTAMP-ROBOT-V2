from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"

if load_dotenv is not None:
    load_dotenv(ROOT_DIR / ".env")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class AppConfig:
    model_file: Path = MODELS_DIR / "panda.xml"
    enable_viewer: bool = True

    ompl_enabled: bool = True
    ompl_planner_name: str = "RRTConnect"
    ompl_fragile_planner_name: str = "BITstar"
    ompl_time_limit: float = 3.0
    ompl_state_validity_resolution: float = 0.005
    ompl_sampler_range: float = 0.08
    ompl_waypoint_step: float = 0.015
    ompl_goal_tolerance: float = 1e-3

    use_ik_fallback: bool = False
    settle_steps_per_waypoint: int = 8
    final_settle_steps: int = 15


def load_config() -> AppConfig:
    model_path = os.getenv("MODEL_FILE")
    model_file = Path(model_path) if model_path else MODELS_DIR / "panda.xml"
    if not model_file.is_absolute():
        model_file = ROOT_DIR / model_file

    return AppConfig(
        model_file=model_file,
        enable_viewer=_env_bool("ENABLE_VIEWER", True),
        ompl_enabled=_env_bool("OMPL_ENABLED", True),
        ompl_planner_name=os.getenv("OMPL_PLANNER_NAME", "RRTConnect"),
        ompl_fragile_planner_name=os.getenv("OMPL_FRAGILE_PLANNER_NAME", "BITstar"),
        ompl_time_limit=_env_float("OMPL_TIME_LIMIT", 3.0),
        ompl_state_validity_resolution=_env_float("OMPL_STATE_VALIDITY_RESOLUTION", 0.005),
        ompl_sampler_range=_env_float("OMPL_SAMPLER_RANGE", 0.08),
        ompl_waypoint_step=_env_float("OMPL_WAYPOINT_STEP", 0.015),
        ompl_goal_tolerance=_env_float("OMPL_GOAL_TOLERANCE", 1e-3),
        use_ik_fallback=_env_bool("USE_IK_FALLBACK", False),
        settle_steps_per_waypoint=_env_int("SETTLE_STEPS_PER_WAYPOINT", 8),
        final_settle_steps=_env_int("FINAL_SETTLE_STEPS", 15),
    )


CONFIG = load_config()
