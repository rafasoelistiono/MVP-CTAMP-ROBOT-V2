#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "logs"
OUT_DIR = ROOT_DIR / "docs" / "log_analysis_comparison"

SCENARIO_LABELS = {
    "before": "Sebelum improvements",
    "after": "Sesudah improvements",
    "two_arm": "Two arm",
}

EVENT_SCENARIO_OVERRIDES = {
    "align_cubes_ungroup_obs_20260514_060004_events_sebelum improvements.csv": "before",
    "separate_groups_ungroup_obs_20260514_061115_events_sebelum improvements.csv": "before",
    "align_cubes_ungroup_obs_20260514_071534_events_sesudah improvements.csv": "after",
    "separate_groups_ungroup_obs_20260514_071701_events.csv": "after",
    "separate_groups_two_arm_ungroup_obs_20260514_081634_events.csv": "two_arm",
}

SUMMARY_SCENARIO_OVERRIDES = {
    "align_cubes_ungroup_obs_20260514_060054.csv": "before",
    "separate_groups_ungroup_obs_20260514_061257.csv": "before",
    "align_cubes_ungroup_obs_20260514_071645.csv": "after",
    "separate_groups_ungroup_obs_20260514_071919_sesudah improvements.csv": "after",
    "separate_groups_two_arm_ungroup_obs_20260514_081838.csv": "two_arm",
}

SCENARIO_ORDER = ["before", "after", "two_arm"]


def _task_name(path: Path) -> str:
    name = path.name
    if name.startswith("separate_groups_two_arm"):
        return "separate_groups_two_arm"
    if name.startswith("separate_groups"):
        return "separate_groups"
    if name.startswith("align_cubes"):
        return "align_cubes"
    if name.startswith("tidy_up"):
        return "tidy_up"
    return name.split("_")[0]


def _scenario_for(path: Path, overrides: dict[str, str]) -> str | None:
    if path.name in overrides:
        return overrides[path.name]
    lower = path.name.lower()
    if "two_arm" in lower or "two-arm" in lower:
        return "two_arm"
    if "sebelum" in lower:
        return "before"
    if "sesudah" in lower:
        return "after"
    return None


def _run_id(path: Path) -> str:
    return path.name.replace(".csv", "")


def _object_from_phase(phase: str) -> str:
    if not isinstance(phase, str):
        return ""
    match = re.search(r"\(([^),]+)", phase)
    return match.group(1).strip() if match else ""


def _load_events() -> pd.DataFrame:
    frames = []
    for path in sorted(LOG_DIR.glob("*_events*.csv")):
        scenario = _scenario_for(path, EVENT_SCENARIO_OVERRIDES)
        if scenario is None:
            continue
        df = pd.read_csv(path, keep_default_na=False)
        df["scenario"] = scenario
        df["scenario_label"] = SCENARIO_LABELS[scenario]
        df["task"] = _task_name(path)
        df["run_id"] = _run_id(path)
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    events = pd.concat(frames, ignore_index=True)
    events["duration_ms_num"] = pd.to_numeric(events.get("duration_ms", ""), errors="coerce").fillna(0)
    events["object_from_phase"] = events.get("phase", "").map(_object_from_phase)
    object_id = events.get("object_id", "")
    events["object_resolved"] = object_id.where(object_id != "", events["object_from_phase"])
    return events


def _load_summaries() -> pd.DataFrame:
    frames = []
    for path in sorted(LOG_DIR.glob("*.csv")):
        if "_events" in path.name:
            continue
        scenario = _scenario_for(path, SUMMARY_SCENARIO_OVERRIDES)
        if scenario is None:
            continue
        try:
            df = pd.read_csv(path, keep_default_na=False)
        except pd.errors.EmptyDataError:
            continue
        df["scenario"] = scenario
        df["scenario_label"] = SCENARIO_LABELS[scenario]
        df["task"] = _task_name(path)
        df["source_file"] = path.name
        for col in ("objects_moved", "objects_total", "failed_count", "duration_ms"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _parse_failed_json(raw: str) -> list[dict]:
    if not isinstance(raw, str) or raw.strip() == "":
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            return []
    return parsed if isinstance(parsed, list) else []


def _summary_failures(summaries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if summaries.empty or "failed_json" not in summaries.columns:
        return pd.DataFrame(rows)
    for _, row in summaries.iterrows():
        for item in _parse_failed_json(row.get("failed_json", "")):
            if not isinstance(item, dict):
                continue
            rows.append({
                "scenario": row["scenario"],
                "scenario_label": row["scenario_label"],
                "task": row["task"],
                "object_id": item.get("object_id", ""),
                "stage": item.get("stage", ""),
                "failure_reason": item.get("failure_reason", item.get("stage", "")),
                "arm": item.get("arm", ""),
            })
    return pd.DataFrame(rows)


def _scenario_metrics(events: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIO_ORDER:
        ev = events[events["scenario"].eq(scenario)]
        su = summaries[summaries["scenario"].eq(scenario)]
        if ev.empty and su.empty:
            continue
        ompl = ev[ev["stage"].eq("OMPL_PLAN")]
        traj = ev[ev["stage"].eq("TRAJECTORY_EXEC")]
        collision = ev[ev["stage"].eq("COLLISION_CHECK")]
        moved = int(su["objects_moved"].sum()) if "objects_moved" in su.columns else 0
        total = int(su["objects_total"].sum()) if "objects_total" in su.columns else 0
        failed = int(su["failed_count"].sum()) if "failed_count" in su.columns else 0
        rows.append({
            "scenario": scenario,
            "scenario_label": SCENARIO_LABELS[scenario],
            "runs": int(su["source_file"].nunique()) if not su.empty else int(ev["source_file"].nunique()),
            "objects_moved": moved,
            "objects_total": total,
            "failed_count": failed,
            "success_rate_objects": moved / total if total else 0.0,
            "ompl_start": int((ompl["status"] == "START").sum()),
            "ompl_ok": int((ompl["status"] == "OK").sum()),
            "ompl_failed": int((ompl["status"] == "FAILED").sum()),
            "trajectory_start": int((traj["status"] == "START").sum()),
            "trajectory_ok": int((traj["status"] == "OK").sum()),
            "trajectory_failed": int((traj["status"] == "FAILED").sum()),
            "pick_failed": int(((ev["stage"] == "PICK") & (ev["status"] == "FAILED")).sum()),
            "place_ok_events": int(((ev["stage"] == "PLACE") & (ev["status"] == "OK")).sum()),
            "collision_blocked": int((collision["status"] == "BLOCKED").sum()),
            "precheck_skips": int(((ev["stage"] == "OBJECT_PRECHECK") & (ev["status"] == "SKIP")).sum()),
        })
    return pd.DataFrame(rows)


def _task_metrics(summaries: pd.DataFrame) -> pd.DataFrame:
    if summaries.empty:
        return pd.DataFrame()
    grouped = summaries.groupby(["scenario", "scenario_label", "task"], as_index=False).agg(
        objects_moved=("objects_moved", "sum"),
        objects_total=("objects_total", "sum"),
        failed_count=("failed_count", "sum"),
        duration_ms=("duration_ms", "sum"),
    )
    grouped["success_rate_objects"] = grouped["objects_moved"] / grouped["objects_total"].replace(0, pd.NA)
    grouped["success_rate_objects"] = grouped["success_rate_objects"].fillna(0)
    return grouped


def _failure_labels(events: pd.DataFrame) -> pd.DataFrame:
    bad = events[events["status"].isin(["FAILED", "ERROR", "BLOCKED", "FATAL"])].copy()
    rows = []
    for _, row in bad.iterrows():
        label = row.get("failure_reason", "")
        if not label:
            label = row.get("stage", "")
        phase = row.get("phase", "")
        if phase and label in {"move_pregrasp_failed", "move_grasp_failed", "move_lift_failed", "ompl_failed_no_fallback"}:
            label = f"{label} | {phase}"
        rows.append({
            "scenario": row["scenario"],
            "scenario_label": row["scenario_label"],
            "task": row["task"],
            "stage": row.get("stage", ""),
            "failure_label": label,
            "object_id": row.get("object_resolved", ""),
        })
    return pd.DataFrame(rows)


def _save_bar(data: pd.DataFrame, x: str, y: str, hue: str, title: str, out: Path, ylabel: str = "") -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    pivot = data.pivot_table(index=x, columns=hue, values=y, aggfunc="sum", fill_value=0)
    pivot = pivot.reindex([SCENARIO_LABELS[s] for s in SCENARIO_ORDER if SCENARIO_LABELS[s] in pivot.index])
    plt.figure(figsize=(11, 5.8))
    pivot.plot(kind="bar", ax=plt.gca(), width=0.78)
    plt.title(title)
    plt.ylabel(ylabel or y)
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()


def _plot_object_outcomes(metrics: pd.DataFrame) -> Path:
    rows = []
    for _, row in metrics.iterrows():
        rows.extend([
            {"scenario_label": row["scenario_label"], "outcome": "Moved", "count": row["objects_moved"]},
            {"scenario_label": row["scenario_label"], "outcome": "Failed / skipped", "count": row["failed_count"]},
        ])
    data = pd.DataFrame(rows)
    out = OUT_DIR / "object_outcomes_by_scenario.png"
    _save_bar(data, "scenario_label", "count", "outcome", "Object Outcomes by Scenario", out, "objects")
    return out


def _plot_pipeline(metrics: pd.DataFrame) -> Path:
    rows = []
    stages = [
        ("OMPL OK", "ompl_ok"),
        ("OMPL failed", "ompl_failed"),
        ("Trajectory OK", "trajectory_ok"),
        ("Trajectory failed", "trajectory_failed"),
        ("Pick failed", "pick_failed"),
        ("Place OK", "place_ok_events"),
        ("Precheck skips", "precheck_skips"),
    ]
    for _, row in metrics.iterrows():
        for label, col in stages:
            rows.append({"scenario_label": row["scenario_label"], "stage": label, "count": row[col]})
    data = pd.DataFrame(rows)
    out = OUT_DIR / "pipeline_counts_by_scenario.png"
    pivot = data.pivot_table(index="stage", columns="scenario_label", values="count", aggfunc="sum", fill_value=0)
    pivot = pivot[[label for label in SCENARIO_LABELS.values() if label in pivot.columns]]
    plt.figure(figsize=(12, 6.2))
    pivot.plot(kind="bar", ax=plt.gca(), width=0.78)
    plt.title("Pipeline Event Counts by Scenario")
    plt.ylabel("event count")
    plt.xlabel("")
    plt.xticks(rotation=35, ha="right")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    return out


def _plot_task_success(task_metrics: pd.DataFrame) -> Path:
    out = OUT_DIR / "task_success_rate.png"
    data = task_metrics.copy()
    data["scenario_task"] = data["scenario_label"] + "\n" + data["task"]
    plt.figure(figsize=(12, 5.8))
    colors = ["#4f79a7" if "two_arm" not in t else "#6f9f4a" for t in data["task"]]
    plt.bar(data["scenario_task"], data["success_rate_objects"] * 100, color=colors)
    for idx, row in enumerate(data.itertuples()):
        plt.text(idx, row.success_rate_objects * 100 + 1, f"{int(row.objects_moved)}/{int(row.objects_total)}", ha="center", fontsize=9)
    plt.title("Object Success Rate per Task Run")
    plt.ylabel("moved / total objects (%)")
    plt.ylim(0, 105)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    return out


def _plot_failure_reasons(summary_failures: pd.DataFrame, event_failures: pd.DataFrame) -> Path:
    out = OUT_DIR / "failure_reasons_by_scenario.png"
    if not summary_failures.empty:
        data = summary_failures.copy()
        data["failure_reason"] = data["failure_reason"].replace("", "unknown")
    else:
        data = event_failures.rename(columns={"failure_label": "failure_reason"})
    counts = data.groupby(["scenario_label", "failure_reason"]).size().reset_index(name="count")
    top = counts.groupby("failure_reason")["count"].sum().sort_values(ascending=False).head(10).index
    counts = counts[counts["failure_reason"].isin(top)]
    pivot = counts.pivot_table(index="failure_reason", columns="scenario_label", values="count", aggfunc="sum", fill_value=0)
    pivot = pivot[[label for label in SCENARIO_LABELS.values() if label in pivot.columns]]
    pivot = pivot.loc[pivot.sum(axis=1).sort_values().index]
    plt.figure(figsize=(12, max(5.0, len(pivot) * 0.45)))
    pivot.plot(kind="barh", ax=plt.gca(), width=0.78)
    plt.title("Failure Reasons by Scenario")
    plt.xlabel("count")
    plt.ylabel("")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    return out


def _plot_two_arm_usage(events: pd.DataFrame, summary_failures: pd.DataFrame) -> Path:
    out = OUT_DIR / "two_arm_usage_and_failures.png"
    two = events[events["scenario"].eq("two_arm")]
    rows = []
    if not two.empty:
        arm_events = two[two["stage"].isin(["ARM_SELECT", "ARM_VALIDATION", "SWITCH_ARM"])]
        for _, row in arm_events.iterrows():
            extra = {}
            try:
                extra = json.loads(row.get("extra_json", "") or "{}")
            except Exception:
                extra = {}
            arm = extra.get("arm") or extra.get("selected_arm") or extra.get("failed_arm") or ""
            if arm:
                rows.append({"category": f"{row['stage']} {row['status']}", "arm": arm, "count": 1})
    sf = summary_failures[summary_failures["scenario"].eq("two_arm")] if not summary_failures.empty else pd.DataFrame()
    if not sf.empty and "arm" in sf.columns:
        for _, row in sf[sf["arm"].ne("")].iterrows():
            rows.append({"category": f"summary_failed_{row['stage']}", "arm": row["arm"], "count": 1})
    data = pd.DataFrame(rows)
    if data.empty:
        data = pd.DataFrame([{"category": "no two-arm events", "arm": "n/a", "count": 0}])
    pivot = data.pivot_table(index="category", columns="arm", values="count", aggfunc="sum", fill_value=0)
    plt.figure(figsize=(11, max(4.8, len(pivot) * 0.45)))
    pivot.plot(kind="barh", ax=plt.gca(), width=0.78)
    plt.title("Two-arm Arm Usage and Failures")
    plt.xlabel("event count")
    plt.ylabel("")
    plt.legend(title="arm")
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    return out


def _md_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if df.empty:
        return "_Tidak ada data._"
    view = df[columns].copy() if columns else df.copy()
    cols = list(view.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in view.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                value = f"{value:.2f}" if not value.is_integer() else str(int(value))
            values.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def analyze() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events = _load_events()
    summaries = _load_summaries()
    if events.empty:
        raise FileNotFoundError(f"No comparable event logs found in {LOG_DIR}")

    metrics = _scenario_metrics(events, summaries)
    task_metrics = _task_metrics(summaries)
    summary_failures = _summary_failures(summaries)
    event_failures = _failure_labels(events)

    images = {
        "object_outcomes": _plot_object_outcomes(metrics),
        "pipeline": _plot_pipeline(metrics),
        "task_success": _plot_task_success(task_metrics),
        "failure_reasons": _plot_failure_reasons(summary_failures, event_failures),
        "two_arm_usage": _plot_two_arm_usage(events, summary_failures),
    }

    metrics_out = metrics.copy()
    metrics_out["success_rate_objects"] = (metrics_out["success_rate_objects"] * 100).round(1)
    task_out = task_metrics.copy()
    task_out["success_rate_objects"] = (task_out["success_rate_objects"] * 100).round(1)

    top_event_failures = (
        event_failures.groupby(["scenario_label", "stage", "failure_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["scenario_label", "count"], ascending=[True, False])
        .groupby("scenario_label")
        .head(8)
    )

    summary_failure_counts = (
        summary_failures.groupby(["scenario_label", "task", "stage", "failure_reason", "arm"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["scenario_label", "count"], ascending=[True, False])
        if not summary_failures.empty
        else pd.DataFrame()
    )

    metrics.to_csv(OUT_DIR / "scenario_metrics.csv", index=False)
    task_metrics.to_csv(OUT_DIR / "task_metrics.csv", index=False)
    summary_failure_counts.to_csv(OUT_DIR / "summary_failure_counts.csv", index=False)
    top_event_failures.to_csv(OUT_DIR / "top_event_failures.csv", index=False)

    image_lines = "\n".join(
        f"![{name}]({path.relative_to(OUT_DIR).as_posix()})"
        for name, path in images.items()
    )
    report = f"""# Improvement Log Comparison

Perbandingan ini memakai log yang tersedia di `logs/` dan membagi run menjadi tiga skenario:

- **Sebelum improvements**: run awal `align_cubes` dan `separate_groups`.
- **Sesudah improvements**: run single-arm setelah perubahan precheck/recovery/IK.
- **Two arm**: run `separate_groups_two_arm`.

## Scenario Metrics

{_md_table(metrics_out, [
    "scenario_label",
    "runs",
    "objects_moved",
    "objects_total",
    "failed_count",
    "success_rate_objects",
    "ompl_start",
    "ompl_ok",
    "ompl_failed",
    "trajectory_ok",
    "trajectory_failed",
    "pick_failed",
    "place_ok_events",
    "collision_blocked",
    "precheck_skips",
])}

`success_rate_objects` dalam persen.

## Task Metrics

{_md_table(task_out, [
    "scenario_label",
    "task",
    "objects_moved",
    "objects_total",
    "failed_count",
    "success_rate_objects",
    "duration_ms",
])}

## Summary Failure Counts

{_md_table(summary_failure_counts, [
    "scenario_label",
    "task",
    "stage",
    "failure_reason",
    "arm",
    "count",
] if not summary_failure_counts.empty else None)}

## Event Failure Hotspots

{_md_table(top_event_failures, [
    "scenario_label",
    "stage",
    "failure_label",
    "count",
])}

## Visualizations

{image_lines}

## Sintesa

1. **Sebelum improvements**, planner sering berhasil, tetapi eksekusi trajectory masih rapuh. Banyak kegagalan terjadi setelah `OMPL_PLAN OK`, terutama saat live trajectory validation dan pick pregrasp/grasp.
2. **Sesudah improvements single-arm**, kandidat berisiko mulai disaring lewat precheck. Ini menurunkan percobaan pada object dekat obstacle, tetapi metrik sukses object masih dibatasi oleh reachable workspace dan kualitas pick/place.
3. **Two arm** menambah fallback sisi: validator memilih arm dengan reach pick/place terbaik dan dapat switch jika pick di sisi pertama gagal. Pada log yang ada, two-arm membantu memindahkan sebagian pekerjaan ke arm kanan untuk object atas, tetapi run masih belum sukses penuh karena kegagalan pick/place fisik masih muncul setelah validasi reach lulus.
4. Bottleneck terbaru bukan sekadar OMPL menemukan path. Bottleneck bergeser ke kualitas grasp/lift dan akurasi place, sehingga next improvement paling berdampak adalah per-arm grasp candidate sampling, validasi IK pose grasp sebelum OMPL, dan strategi release/place yang lebih stabil.
"""
    report_path = OUT_DIR / "comparison_summary.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> int:
    report_path = analyze()
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
