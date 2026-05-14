#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


def _latest_event_csv(task: str | None) -> Path:
    pattern = f"{task}_*_events.csv" if task else "*_events.csv"
    candidates = list((ROOT_DIR / "logs").glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"no event CSV found for pattern logs/{pattern}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _parse_extra(row: dict[str, str]) -> dict:
    raw = row.get("extra_json") or ""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _event_ms(row: dict[str, str]) -> int:
    try:
        return int(row.get("duration_ms") or 0)
    except ValueError:
        return 0


def _phase_label(row: dict[str, str]) -> str:
    return row.get("phase") or row.get("object_id") or row.get("stage") or "-"


def build_report(events_path: Path, out_path: Path) -> Path:
    rows = _read_rows(events_path)
    status_counts = Counter((r["stage"], r["status"]) for r in rows)
    failure_counts = Counter(
        (
            r["stage"],
            r["status"],
            r.get("failure_reason", ""),
            r.get("phase", ""),
            r.get("collision_pair", ""),
        )
        for r in rows
        if r["status"] in {"FAILED", "ERROR", "BLOCKED", "FATAL", "WARN"}
    )

    ompl_rows = [r for r in rows if r["stage"] == "OMPL_PLAN" and r["status"] in {"OK", "FAILED", "ERROR"}]
    object_counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        obj = row.get("object_id") or _object_from_phase(row.get("phase", ""))
        if obj:
            object_counts[obj][f"{row['stage']}:{row['status']}"] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        _html_document(events_path, rows, status_counts, failure_counts, ompl_rows, object_counts),
        encoding="utf-8",
    )
    return out_path


def _object_from_phase(phase: str) -> str:
    if "(" not in phase or ")" not in phase:
        return ""
    inside = phase.split("(", 1)[1].split(")", 1)[0]
    if "," in inside:
        return ""
    return inside.strip()


def _html_document(
    events_path: Path,
    rows: list[dict[str, str]],
    status_counts: Counter,
    failure_counts: Counter,
    ompl_rows: list[dict[str, str]],
    object_counts: dict[str, Counter],
) -> str:
    timeline = _timeline_svg(rows)
    failure_table = _failure_table(failure_counts)
    ompl_table = _ompl_table(ompl_rows)
    object_table = _object_table(object_counts)
    status_table = _status_table(status_counts)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CTAMP OMPL Run Visualization</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 28px; color: #17202a; background: #f7f4ed; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .card {{ background: #fffdf8; border: 1px solid #dacfb9; border-radius: 14px; padding: 18px; margin: 16px 0; box-shadow: 0 2px 8px #0001; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ text-align: left; border-bottom: 1px solid #eadfcb; padding: 7px 8px; vertical-align: top; }}
    th {{ background: #efe4cf; }}
    code {{ background: #eee2ca; padding: 2px 5px; border-radius: 5px; }}
    .ok {{ color: #237a3b; }}
    .bad {{ color: #a13324; }}
    .muted {{ color: #6e6252; }}
  </style>
</head>
<body>
  <h1>CTAMP OMPL Run Visualization</h1>
  <p class="muted">Source: <code>{html.escape(str(events_path))}</code>, events: {len(rows)}</p>
  <div class="card">
    <h2>Timeline</h2>
    {timeline}
  </div>
  <div class="card">
    <h2>Failure Hotspots</h2>
    {failure_table}
  </div>
  <div class="card">
    <h2>OMPL Plan Results</h2>
    {ompl_table}
  </div>
  <div class="card">
    <h2>Object Activity</h2>
    {object_table}
  </div>
  <div class="card">
    <h2>Stage Counts</h2>
    {status_table}
  </div>
</body>
</html>
"""


def _timeline_svg(rows: list[dict[str, str]]) -> str:
    width = 1180
    height = max(160, min(900, 30 + len(rows) * 6))
    max_events = 140
    selected = rows if len(rows) <= max_events else rows[:70] + rows[-70:]
    gap_note = "" if len(rows) <= max_events else f"<text x='16' y='20' font-size='12' fill='#7b6a55'>Showing first/last {max_events} events from {len(rows)} total.</text>"
    items = [gap_note]
    x0, y0 = 16, 34
    step = max(3, (width - 32) / max(len(selected), 1))
    color = {
        "OK": "#2e8b57",
        "START": "#4a78b8",
        "EXEC": "#7b61b5",
        "WARN": "#d08b26",
        "FAILED": "#c0392b",
        "ERROR": "#922b21",
        "BLOCKED": "#922b21",
        "FATAL": "#581845",
    }
    for i, row in enumerate(selected):
        x = x0 + i * step
        y = y0 + (i % 18) * 6
        c = color.get(row["status"], "#6c757d")
        title = html.escape(f"{row.get('event_id')} {row['stage']} {row['status']} {_phase_label(row)} {row.get('failure_reason','')}")
        items.append(f"<circle cx='{x:.1f}' cy='{y}' r='3' fill='{c}'><title>{title}</title></circle>")
    legend_y = height - 22
    legend = " ".join(
        f"<rect x='{16 + idx * 118}' y='{legend_y}' width='10' height='10' fill='{c}'/><text x='{30 + idx * 118}' y='{legend_y + 10}' font-size='11'>{s}</text>"
        for idx, (s, c) in enumerate(color.items())
    )
    return f"<svg width='100%' viewBox='0 0 {width} {height}' role='img'>{''.join(items)}{legend}</svg>"


def _failure_table(counts: Counter) -> str:
    rows = []
    for (stage, status, reason, phase, pair), count in counts.most_common(20):
        rows.append(
            f"<tr><td>{count}</td><td>{html.escape(stage)}</td><td>{html.escape(status)}</td>"
            f"<td>{html.escape(reason)}</td><td>{html.escape(phase)}</td><td>{html.escape(pair)}</td></tr>"
        )
    return "<table><tr><th>Count</th><th>Stage</th><th>Status</th><th>Reason</th><th>Phase</th><th>Pair</th></tr>" + "".join(rows) + "</table>"


def _ompl_table(rows: list[dict[str, str]]) -> str:
    html_rows = []
    for row in rows[-80:]:
        extra = _parse_extra(row)
        html_rows.append(
            f"<tr><td>{html.escape(row.get('event_id',''))}</td><td>{html.escape(row.get('status',''))}</td>"
            f"<td>{html.escape(row.get('phase',''))}</td><td>{html.escape(row.get('planner',''))}</td>"
            f"<td>{html.escape(row.get('waypoints',''))}</td><td>{html.escape(row.get('duration_ms',''))}</td>"
            f"<td>{html.escape(row.get('failure_reason',''))}</td><td>{html.escape(str(extra.get('path_length','')))}</td></tr>"
        )
    return "<table><tr><th>ID</th><th>Status</th><th>Phase</th><th>Planner</th><th>Waypoints</th><th>ms</th><th>Reason</th><th>Path Len</th></tr>" + "".join(html_rows) + "</table>"


def _object_table(counts: dict[str, Counter]) -> str:
    rows = []
    for obj, counter in sorted(counts.items()):
        rows.append(f"<tr><td>{html.escape(obj)}</td><td>{html.escape(json.dumps(counter, ensure_ascii=False))}</td></tr>")
    return "<table><tr><th>Object</th><th>Events</th></tr>" + "".join(rows) + "</table>"


def _status_table(counts: Counter) -> str:
    rows = []
    for (stage, status), count in counts.most_common():
        rows.append(f"<tr><td>{count}</td><td>{html.escape(stage)}</td><td>{html.escape(status)}</td></tr>")
    return "<table><tr><th>Count</th><th>Stage</th><th>Status</th></tr>" + "".join(rows) + "</table>"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an HTML visualization for align/separate OMPL event CSV logs.")
    parser.add_argument("--events", type=Path)
    parser.add_argument("--task", choices=["align_cubes", "separate_groups"])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    events = args.events or _latest_event_csv(args.task)
    out = args.out or events.with_suffix(".html")
    report = build_report(events, out)
    print(f"visualization={report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
