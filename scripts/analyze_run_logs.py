#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = ROOT_DIR / "logs"
DEFAULT_OUT_DIR = ROOT_DIR / "docs" / "log_analysis"


def _run_id(path: Path) -> str:
    return path.name.replace("_events.csv", "")


def _task_name(path: Path) -> str:
    name = path.name
    if name.startswith("separate_groups"):
        return "separate_groups"
    if name.startswith("align_cubes"):
        return "align_cubes"
    if name.startswith("tidy_up"):
        return "tidy_up"
    return name.split("_")[0]


def _object_from_phase(phase: str) -> str:
    if not isinstance(phase, str):
        return ""
    match = re.search(r"\(([^),]+)", phase)
    return match.group(1).strip() if match else ""


def _load_events(log_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(log_dir.glob("*_events.csv")):
        df = pd.read_csv(path, keep_default_na=False)
        df["run_id"] = _run_id(path)
        df["task"] = _task_name(path)
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    events = pd.concat(frames, ignore_index=True)
    events["duration_ms_num"] = pd.to_numeric(events.get("duration_ms", ""), errors="coerce").fillna(0)
    events["object_from_phase"] = events.get("phase", "").map(_object_from_phase)
    events["object_resolved"] = events.get("object_id", "").where(events.get("object_id", "") != "", events["object_from_phase"])
    return events


def _load_summaries(log_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(log_dir.glob("*.csv")):
        if path.name.endswith("_events.csv"):
            continue
        try:
            df = pd.read_csv(path, keep_default_na=False)
        except pd.errors.EmptyDataError:
            continue
        df["source_file"] = path.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _shorten(value: str, limit: int = 70) -> str:
    value = str(value)
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _save_barh(series: pd.Series, title: str, out: Path, color: str = "#c05a36") -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    data = series.sort_values().tail(12)
    plt.figure(figsize=(11, max(4.5, len(data) * 0.46)))
    plt.barh([_shorten(idx, 78) for idx in data.index], data.values, color=color)
    plt.title(title)
    plt.xlabel("count")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _plot_failure_hotspots(events: pd.DataFrame, out_dir: Path) -> Path:
    bad = events[events["status"].isin(["FAILED", "ERROR", "BLOCKED", "FATAL"])]
    labels = bad.apply(
        lambda r: " | ".join(
            part
            for part in [
                r.get("stage", ""),
                r.get("failure_reason", ""),
                r.get("collision_pair", ""),
                r.get("phase", ""),
            ]
            if str(part).strip()
        ),
        axis=1,
    )
    out = out_dir / "failure_hotspots.png"
    _save_barh(labels.value_counts(), "Failure Hotspots from Event Logs", out)
    return out


def _plot_ompl_outcomes(events: pd.DataFrame, out_dir: Path) -> Path:
    ompl = events[events["stage"].eq("OMPL_PLAN") & events["status"].isin(["OK", "FAILED", "ERROR"])]
    pivot = ompl.pivot_table(index="run_id", columns="status", values="event_id", aggfunc="count", fill_value=0)
    for col in ["OK", "FAILED", "ERROR"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["OK", "FAILED", "ERROR"]]
    out = out_dir / "ompl_outcomes_by_run.png"
    plt.figure(figsize=(12, max(4.5, len(pivot) * 0.55)))
    pivot.plot(kind="barh", stacked=True, color=["#2e8b57", "#d08b26", "#922b21"], ax=plt.gca())
    plt.title("OMPL Plan Outcomes by Run")
    plt.xlabel("OMPL plan count")
    plt.ylabel("run")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def _plot_pipeline_funnel(events: pd.DataFrame, out_dir: Path) -> Path:
    labels = [
        ("PICK start", (events["stage"].eq("PICK") & events["status"].eq("START")).sum()),
        ("PICK ok", (events["stage"].eq("PICK") & events["status"].eq("OK")).sum()),
        ("PICK failed", (events["stage"].eq("PICK") & events["status"].eq("FAILED")).sum()),
        ("OMPL start", (events["stage"].eq("OMPL_PLAN") & events["status"].eq("START")).sum()),
        ("OMPL ok", (events["stage"].eq("OMPL_PLAN") & events["status"].eq("OK")).sum()),
        ("Trajectory start", (events["stage"].eq("TRAJECTORY_EXEC") & events["status"].eq("START")).sum()),
        ("Trajectory ok", (events["stage"].eq("TRAJECTORY_EXEC") & events["status"].eq("OK")).sum()),
        ("Trajectory failed", (events["stage"].eq("TRAJECTORY_EXEC") & events["status"].eq("FAILED")).sum()),
        ("PLACE ok", (events["stage"].eq("PLACE") & events["status"].eq("OK")).sum()),
    ]
    out = out_dir / "pipeline_funnel.png"
    plt.figure(figsize=(12, 5))
    plt.bar([x[0] for x in labels], [x[1] for x in labels], color="#4f79a7")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("event count")
    plt.title("Execution Funnel Across Logged Runs")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def _plot_pick_attempts(events: pd.DataFrame, out_dir: Path) -> Path:
    picks = events[events["stage"].eq("PICK_PROFILE") & events["status"].eq("SELECT")].copy()
    if picks.empty:
        series = pd.Series(dtype=int)
    else:
        series = picks["object_id"].replace("", "unknown").value_counts()
    out = out_dir / "pick_attempts_by_object.png"
    _save_barh(series, "Pick Attempts by Object", out, color="#6d6ab7")
    return out


def _metrics(events: pd.DataFrame, summaries: pd.DataFrame) -> dict:
    failure_mask = events["status"].isin(["FAILED", "ERROR", "BLOCKED", "FATAL"])
    ompl = events[events["stage"].eq("OMPL_PLAN")]
    traj = events[events["stage"].eq("TRAJECTORY_EXEC")]
    ik_warn = events[(events["stage"].eq("IK_SOLVE")) & (events["status"].eq("WARN"))]
    collision = events[events["stage"].eq("COLLISION_CHECK")]
    table_finger = collision[collision["failure_reason"].str.contains("table/1 <-> left_finger|left_finger/.+table/1", regex=True, na=False)]
    return {
        "event_files": int(events["source_file"].nunique()) if not events.empty else 0,
        "summary_files": int(summaries["source_file"].nunique()) if not summaries.empty else 0,
        "events_total": int(len(events)),
        "ompl_start": int((ompl["status"] == "START").sum()),
        "ompl_ok": int((ompl["status"] == "OK").sum()),
        "ompl_failed": int((ompl["status"] == "FAILED").sum()),
        "ompl_error": int((ompl["status"] == "ERROR").sum()),
        "trajectory_start": int((traj["status"] == "START").sum()),
        "trajectory_ok": int((traj["status"] == "OK").sum()),
        "trajectory_failed": int((traj["status"] == "FAILED").sum()),
        "ik_warn": int(len(ik_warn)),
        "collision_blocked": int((collision["status"] == "BLOCKED").sum()),
        "table_finger_waypoint0": int((table_finger["phase"] == "trajectory waypoint 0").sum()),
        "pick_failed": int(((events["stage"] == "PICK") & (events["status"] == "FAILED")).sum()),
        "place_ok": int(((events["stage"] == "PLACE") & (events["status"] == "OK")).sum()),
    }


def _top_failures(events: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    bad = events[events["status"].isin(["FAILED", "ERROR", "BLOCKED", "FATAL"])].copy()
    if bad.empty:
        return pd.DataFrame(columns=["count", "stage", "status", "failure_reason", "phase", "collision_pair"])
    cols = ["stage", "status", "failure_reason", "phase", "collision_pair"]
    return bad.groupby(cols, dropna=False).size().reset_index(name="count").sort_values("count", ascending=False).head(n)


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("\n", " ").replace("|", "\\|") for col in cols]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _markdown_report(events: pd.DataFrame, summaries: pd.DataFrame, out_dir: Path, images: dict[str, Path]) -> str:
    m = _metrics(events, summaries)
    top = _top_failures(events)
    top_md = _df_to_markdown(top) if not top.empty else "Tidak ada failure event."
    summary_cols = ["task", "scene", "success", "objects_moved", "objects_total", "failed_count", "duration_ms", "source_file"]
    summary_view = summaries[[c for c in summary_cols if c in summaries.columns]].copy() if not summaries.empty else summaries
    summary_md = _df_to_markdown(summary_view) if not summary_view.empty else "Tidak ada summary CSV."
    rel = {k: v.as_posix().replace((ROOT_DIR.as_posix() + "/"), "") for k, v in images.items()}
    return f"""# Log Analysis Snapshot

Generated from `{m['event_files']}` event CSV files and `{m['summary_files']}` summary CSV files.

## Aggregate Metrics

| Metric | Value |
|---|---:|
| Total events | {m['events_total']} |
| OMPL starts | {m['ompl_start']} |
| OMPL OK | {m['ompl_ok']} |
| OMPL failed | {m['ompl_failed']} |
| OMPL error | {m['ompl_error']} |
| Trajectory starts | {m['trajectory_start']} |
| Trajectory OK | {m['trajectory_ok']} |
| Trajectory failed | {m['trajectory_failed']} |
| IK warnings | {m['ik_warn']} |
| Collision blocked | {m['collision_blocked']} |
| Table-left_finger blocked at waypoint 0 | {m['table_finger_waypoint0']} |
| Pick failed events | {m['pick_failed']} |
| Place OK events | {m['place_ok']} |

## Summary Runs

{summary_md}

## Top Failure Hotspots

{top_md}

## Figures

![Failure hotspots]({rel['failures']})

![OMPL outcomes]({rel['ompl']})

![Pipeline funnel]({rel['funnel']})

![Pick attempts]({rel['pick_attempts']})
"""


def analyze(log_dir: Path, out_dir: Path) -> tuple[dict, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    events = _load_events(log_dir)
    summaries = _load_summaries(log_dir)
    if events.empty:
        raise FileNotFoundError(f"No event CSV files found in {log_dir}")
    images = {
        "failures": _plot_failure_hotspots(events, out_dir),
        "ompl": _plot_ompl_outcomes(events, out_dir),
        "funnel": _plot_pipeline_funnel(events, out_dir),
        "pick_attempts": _plot_pick_attempts(events, out_dir),
    }
    metrics = _metrics(events, summaries)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = _markdown_report(events, summaries, out_dir, images)
    report_path = out_dir / "analysis_summary.md"
    report_path.write_text(report, encoding="utf-8")
    return metrics, report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze CTAMP event CSV logs and generate PNG figures + markdown summary.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    metrics, report = analyze(args.log_dir, args.out_dir)
    print(f"analysis_report={report}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
