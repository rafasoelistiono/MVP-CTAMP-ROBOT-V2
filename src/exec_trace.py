from __future__ import annotations

import csv
import json
import os
import time
import atexit
from pathlib import Path
from typing import Any


_EVENT_ID = 0
_CSV_PATH = os.getenv("CTAMP_EVENT_LOG_CSV", "").strip()
_CONSOLE = os.getenv("CTAMP_TRACE_CONSOLE", "true").strip().lower() not in {"0", "false", "no", "off"}
_BUFFER: list[dict[str, Any]] = []
_FLUSH_EVERY = int(os.getenv("CTAMP_TRACE_FLUSH_EVERY", "1"))
_FIELDNAMES = [
    "event_id",
    "timestamp",
    "stage",
    "status",
    "arm",
    "object_id",
    "phase",
    "label",
    "scenario_type",
    "obstacle_mode",
    "backend",
    "planner",
    "attempt",
    "candidate_id",
    "seed_id",
    "waypoints",
    "duration_ms",
    "grip",
    "target_xyz",
    "actual_xyz",
    "ee_xyz",
    "object_xyz",
    "object_z",
    "held_object",
    "finger_pos",
    "q",
    "q_target",
    "q_error_norm",
    "pos_err",
    "ori_err",
    "iterations",
    "distance_to_target",
    "joint_limit_valid",
    "state_valid",
    "state_invalid_reason",
    "ompl_result",
    "execution_result",
    "ignored_body_names",
    "failure_reason",
    "collision_pair",
    "contact_count",
    "penetration",
    "obstacle_distance",
    "extra_json",
]


def log_event(stage: str, status: str, **fields: Any) -> None:
    global _EVENT_ID
    _EVENT_ID += 1
    row = {
        "event_id": _EVENT_ID,
        "timestamp": _timestamp(),
        "stage": stage,
        "status": status,
    }
    extra = {}
    for key, value in fields.items():
        if key in _FIELDNAMES:
            row[key] = _format_value(value)
        else:
            extra[key] = value
    row["extra_json"] = json.dumps(extra, ensure_ascii=False, default=str) if extra else ""

    if _CONSOLE:
        print(_format_console(row))
    if _CSV_PATH:
        _BUFFER.append(row)
        if len(_BUFFER) >= _FLUSH_EVERY:
            flush()


def flush() -> None:
    if not _CSV_PATH or not _BUFFER:
        return
    path = Path(_CSV_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if not exists:
            writer.writeheader()
        for row in _BUFFER:
            writer.writerow({key: row.get(key, "") for key in _FIELDNAMES})
    _BUFFER.clear()


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()) + f".{int((time.time() % 1) * 1000):03d}"


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _format_console(row: dict[str, Any]) -> str:
    parts = [
        f"[TRACE {int(row['event_id']):03d}]",
        f"{str(row['stage'])[:24]:<24}",
        f"{str(row['status'])[:8]:<8}",
    ]
    for key in ("arm", "object_id", "phase", "backend", "planner", "attempt", "waypoints", "duration_ms", "failure_reason"):
        value = row.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _write_csv(row: dict[str, Any]) -> None:
    path = Path(_CSV_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in _FIELDNAMES})


atexit.register(flush)
