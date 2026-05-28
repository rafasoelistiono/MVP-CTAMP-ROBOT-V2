from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ── Bucket definitions ────────────────────────────────────────────────────────

_REACH_BUCKETS = [
    ("near",       0.0,  0.50),
    ("mid",        0.50, 0.70),
    ("far",        0.70, 0.85),
    ("borderline", 0.85, math.inf),
]

_OBSTACLE_BUCKETS = [
    ("clear",     0.28, math.inf),
    ("near",      0.12, 0.28),
    ("too_close", 0.0,  0.12),
]

# ARM_BASE_XY matches executor.py BASE_XY = [-0.4, 0.0]
_ARM_BASE_XY = (-0.4, 0.0)

# ── Tunable defaults (all overridable via env vars at runtime) ─────────────────
#
#   HINT_MIN_SAMPLES          int   min events per bucket before a hint fires
#   HINT_PINOCCHIO_SKIP_RATE  float fallback/total ratio above which Pinocchio
#                                    is skipped for the whole run
#   HINT_NEAR_MISS_RATE       float fraction of near-miss rejections that
#                                    triggers a widened pos_err tolerance
#   HINT_NEAR_MISS_FACTOR     float a rejection is a "near miss" when
#                                    pos_err in (limit, limit*(1+factor))
#   HINT_TOLERANCE_HEADROOM   float widened = median_near_miss * headroom
#   HINT_MAX_TOLERANCE_FACTOR float widened is capped at limit * this value
#   HINT_MAX_LOG_AGE_DAYS     int   ignore logs older than this many days

_D_MIN_SAMPLES          = 5
_D_PINOCCHIO_SKIP_RATE  = 0.70
_D_NEAR_MISS_RATE       = 0.40
_D_NEAR_MISS_FACTOR     = 0.50
_D_TOLERANCE_HEADROOM   = 1.10
_D_MAX_TOLERANCE_FACTOR = 1.60
_D_MAX_LOG_AGE_DAYS     = 14


def _reach_bucket(dist: float) -> str:
    for name, lo, hi in _REACH_BUCKETS:
        if lo <= dist < hi:
            return name
    return "borderline"


def _obstacle_bucket(dist: float) -> str:
    for name, lo, hi in _OBSTACLE_BUCKETS:
        if lo <= dist < hi:
            return name
    return "clear"


def _parse_object_from_phase(phase: str) -> Optional[str]:
    """'pick(cube2) pregrasp'  →  'cube2'"""
    if not phase:
        return None
    try:
        start = phase.index("(") + 1
        end = phase.index(")")
        return phase[start:end]
    except ValueError:
        return None


class HintCache:
    """
    Reads past *_events.csv logs and surfaces three adaptive hints:

      preferred_backend()        skip Pinocchio if FK failure rate is too high
      pos_err_tolerance()        widen pos_err limit for known-hard workspace buckets
      preferred_grasp_profile()  start with the profile that worked fastest

    All hints return None when there is insufficient data (cold-start safe).

    Tuning knobs (see _D_* defaults above and docs/hintcache_plan.txt):
      Constructor params mirror the env-var names without the HINT_ prefix,
      in snake_case.  Env vars always take precedence over constructor args.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        max_log_age_days: int = _D_MAX_LOG_AGE_DAYS,
        scene_filter: Optional[str] = None,
        min_samples: int = _D_MIN_SAMPLES,
        pinocchio_skip_rate: float = _D_PINOCCHIO_SKIP_RATE,
        near_miss_rate: float = _D_NEAR_MISS_RATE,
        near_miss_factor: float = _D_NEAR_MISS_FACTOR,
        tolerance_headroom: float = _D_TOLERANCE_HEADROOM,
        max_tolerance_factor: float = _D_MAX_TOLERANCE_FACTOR,
    ):
        self._min_samples          = int(os.getenv("HINT_MIN_SAMPLES",          str(min_samples)))
        self._pinocchio_skip_rate  = float(os.getenv("HINT_PINOCCHIO_SKIP_RATE", str(pinocchio_skip_rate)))
        self._near_miss_rate       = float(os.getenv("HINT_NEAR_MISS_RATE",      str(near_miss_rate)))
        self._near_miss_factor     = float(os.getenv("HINT_NEAR_MISS_FACTOR",    str(near_miss_factor)))
        self._tolerance_headroom   = float(os.getenv("HINT_TOLERANCE_HEADROOM",  str(tolerance_headroom)))
        self._max_tol_factor       = float(os.getenv("HINT_MAX_TOLERANCE_FACTOR",str(max_tolerance_factor)))

        self._pinocchio_skip: bool = False
        self._pos_err_hints: dict[tuple, float] = {}
        self._profile_hints: dict[tuple, int] = {}
        self._stats: dict = {
            "logs_loaded": 0,
            "rows_loaded": 0,
            "pinocchio_skip": False,
            "pinocchio_fallback_rate": None,
            "pos_err_hints": {},
            "profile_hints": {},
        }

        rows = self._load_rows(log_dir, max_log_age_days, scene_filter)
        if rows:
            self._compute_pinocchio_hint(rows)
            self._compute_pos_err_hints(rows)
            self._compute_profile_hints(rows)

    # ── Public API ─────────────────────────────────────────────────────────────

    def preferred_backend(
        self,
        reach_dist: float = 0.0,
        obstacle_dist: float = math.inf,
    ) -> Optional[str]:
        """Returns 'mujoco_dls' to skip Pinocchio, or None to use default."""
        if self._pinocchio_skip:
            return "mujoco_dls"
        return None

    def pos_err_tolerance(
        self,
        reach_dist: float,
        obstacle_dist: float,
    ) -> Optional[float]:
        """Returns widened pos_err limit, or None to use default."""
        key = (_reach_bucket(reach_dist), _obstacle_bucket(obstacle_dist))
        return self._pos_err_hints.get(key)

    def preferred_grasp_profile(
        self,
        obj_class: str,
        reach_dist: float,
    ) -> Optional[int]:
        """Returns effective profile index (post circle-adjustment), or None."""
        key = (obj_class, _reach_bucket(reach_dist))
        return self._profile_hints.get(key)

    def summary(self) -> dict:
        return dict(self._stats)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _load_rows(
        self,
        log_dir: str,
        max_age_days: int,
        scene_filter: Optional[str],
    ) -> list[dict]:
        cutoff = datetime.now() - timedelta(days=max_age_days)
        rows: list[dict] = []
        logs_loaded = 0
        try:
            log_path = Path(log_dir)
            if not log_path.is_absolute():
                here = Path(__file__).resolve().parent
                log_path = here.parent / log_dir
            for path in sorted(log_path.glob("*_events.csv")):
                if scene_filter and scene_filter not in path.name:
                    continue
                try:
                    if datetime.fromtimestamp(path.stat().st_mtime) < cutoff:
                        continue
                    with path.open(newline="", encoding="utf-8") as f:
                        file_rows = list(csv.DictReader(f))
                    rows.extend(file_rows)
                    logs_loaded += 1
                except Exception:
                    pass
        except Exception:
            pass
        self._stats["logs_loaded"] = logs_loaded
        self._stats["rows_loaded"] = len(rows)
        return rows

    # ── Hint 1: Pinocchio backend skip ────────────────────────────────────────

    def _compute_pinocchio_hint(self, rows: list[dict]) -> None:
        fallback_count = sum(
            1 for r in rows
            if r.get("stage") == "IK_SOLVE"
            and r.get("status") == "BACKEND_FALLBACK"
            and r.get("failure_reason") == "pinocchio_fk_validation_failed"
        )
        ik_candidate_count = sum(
            1 for r in rows if r.get("stage") == "IK_CANDIDATE"
        )
        if ik_candidate_count < self._min_samples:
            return
        rate = fallback_count / ik_candidate_count
        self._stats["pinocchio_fallback_rate"] = round(rate, 3)
        self._stats["pinocchio_fallback_count"] = fallback_count
        self._stats["ik_candidate_count"] = ik_candidate_count
        if rate >= self._pinocchio_skip_rate:
            self._pinocchio_skip = True
            self._stats["pinocchio_skip"] = True

    # ── Hint 2: pos_err tolerance widening ────────────────────────────────────

    def _compute_pos_err_hints(self, rows: list[dict]) -> None:
        # Build object_id → (reach_dist, obstacle_dist) from PICK_PRECHECK events.
        obj_context: dict[str, dict] = {}
        for r in rows:
            if r.get("stage") != "PICK_PRECHECK":
                continue
            oid = r.get("object_id", "")
            if not oid:
                continue
            try:
                xyz = json.loads(r.get("object_xyz", "[]"))
                obs = float(r.get("obstacle_distance", "inf") or "inf")
                reach = math.dist(xyz[:2], list(_ARM_BASE_XY))
                obj_context[oid] = {"reach_dist": reach, "obstacle_dist": obs}
            except Exception:
                pass

        # Per (reach_bkt, obs_bkt): collect (pos_err, pos_limit) pairs from
        # IK_CANDIDATE REJECT events with failure_reason=ik_error_above_limit.
        bucket_samples: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
        for r in rows:
            if r.get("stage") != "IK_CANDIDATE" or r.get("status") != "REJECT":
                continue
            if r.get("failure_reason") != "ik_error_above_limit":
                continue
            oid = _parse_object_from_phase(r.get("phase", ""))
            if not oid or oid not in obj_context:
                continue
            try:
                pos_err = float(r.get("pos_err", "nan"))
                extra = json.loads(r.get("extra_json", "{}") or "{}")
                pos_limit = float(extra.get("pos_limit", 0.03))
            except Exception:
                continue
            ctx = obj_context[oid]
            key = (_reach_bucket(ctx["reach_dist"]), _obstacle_bucket(ctx["obstacle_dist"]))
            bucket_samples[key].append((pos_err, pos_limit))

        for key, samples in bucket_samples.items():
            if len(samples) < self._min_samples:
                continue
            near_misses = [
                pe for pe, pl in samples
                if pl < pe <= pl * (1.0 + self._near_miss_factor)
            ]
            if not near_misses:
                continue
            if len(near_misses) / len(samples) < self._near_miss_rate:
                continue
            near_misses.sort()
            median = near_misses[len(near_misses) // 2]
            max_limit = max(pl for _, pl in samples) * self._max_tol_factor
            widened = round(min(median * self._tolerance_headroom, max_limit), 4)
            self._pos_err_hints[key] = widened

        self._stats["pos_err_hints"] = {str(k): v for k, v in self._pos_err_hints.items()}

    # ── Hint 3: grasp profile preference ──────────────────────────────────────

    def _compute_profile_hints(self, rows: list[dict]) -> None:
        # Build object_id → reach_dist from PICK_PRECHECK events.
        obj_reach: dict[str, float] = {}
        for r in rows:
            if r.get("stage") != "PICK_PRECHECK":
                continue
            oid = r.get("object_id", "")
            if not oid:
                continue
            try:
                xyz = json.loads(r.get("object_xyz", "[]"))
                obj_reach[oid] = math.dist(xyz[:2], list(_ARM_BASE_XY))
            except Exception:
                pass

        # Build object_id → [profile_index per attempt] from PICK_PROFILE SELECT.
        obj_profiles: dict[str, list[int]] = defaultdict(list)
        for r in rows:
            if r.get("stage") != "PICK_PROFILE" or r.get("status") != "SELECT":
                continue
            oid = r.get("object_id", "")
            if not oid:
                continue
            try:
                extra = json.loads(r.get("extra_json", "{}") or "{}")
                obj_profiles[oid].append(int(extra["profile_index"]))
            except Exception:
                pass

        # Build object_id → [pick_ok per attempt] from CHECK_PICK events.
        obj_pick_ok: dict[str, list[bool]] = defaultdict(list)
        for r in rows:
            if r.get("stage") != "CHECK_PICK":
                continue
            oid = r.get("object_id", "")
            if not oid:
                continue
            obj_pick_ok[oid].append(r.get("status") == "OK")

        # Per (obj_class, reach_bkt): accumulate (profile_index, success) pairs.
        bucket_stats: dict[tuple, dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
        for oid, profiles in obj_profiles.items():
            reach = obj_reach.get(oid)
            if reach is None:
                continue
            obj_class = "circle" if oid.startswith("circle") else "cube"
            key = (obj_class, _reach_bucket(reach))
            successes = obj_pick_ok.get(oid, [])
            for i, pi in enumerate(profiles):
                outcome = successes[i] if i < len(successes) else False
                bucket_stats[key][pi].append(outcome)

        for key, profile_data in bucket_stats.items():
            best_pi, best_score = None, -1.0
            for pi, outcomes in profile_data.items():
                if len(outcomes) < self._min_samples:
                    continue
                score = sum(outcomes) / len(outcomes)
                if score > best_score:
                    best_score, best_pi = score, pi
            if best_pi is not None and best_score > 0.0:
                self._profile_hints[key] = best_pi

        self._stats["profile_hints"] = {str(k): v for k, v in self._profile_hints.items()}
