#!/usr/bin/env python3
"""
HintCache log collector.

Runs task scripts repeatedly until HintCache reaches full readiness
(all three hints active for every target scene variant).

Usage:
    source .venv/bin/activate
    python tools/collect_logs.py                        # all variants, auto-stop
    python tools/collect_logs.py --scenes separate_groups_ungroup_obs --rounds 5
    python tools/collect_logs.py --dry-run              # show plan, don't run

What "ready" means:
    Hint1 (pinocchio_skip)  : fires after >= MIN_SAMPLES IK_CANDIDATE events
                              and Pinocchio failure rate >= threshold
    Hint2 (pos_err_hints)   : fires per (reach, obstacle) bucket when near-miss
                              rate >= threshold  — may never fire if IK is clean
    Hint3 (profile_hints)   : fires per (obj_class, reach) bucket when >= MIN_SAMPLES
                              CHECK_PICK events — NEEDS ~3 runs per variant

The collector stops a variant once Hint1 + Hint3 are both active (Hint2 is
optional — it only fires for genuinely hard IK buckets).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["CTAMP_TRACE_CONSOLE"] = "false"
os.environ["CTAMP_EVENT_LOG_CSV"] = ""

from hint_cache import HintCache

# ---------------------------------------------------------------------------
# All runnable variants: (script, scene_args, scene_filter_key)
# ---------------------------------------------------------------------------
ALL_VARIANTS = [
    # separate_groups
    ("separate_groups_ompl_only.py", ["--object", "ungroup", "obs"],    "separate_groups_ungroup_obs"),
    ("separate_groups_ompl_only.py", ["--object", "group",   "obs"],    "separate_groups_group_obs"),
    ("separate_groups_ompl_only.py", ["--object", "ungroup", "no", "obs"], "separate_groups_ungroup_no_obs"),
    ("separate_groups_ompl_only.py", ["--object", "group",   "no", "obs"], "separate_groups_group_no_obs"),
    # align_cubes
    ("align_cubes_ompl_only.py",     ["--object", "ungroup", "obs"],    "align_cubes_ungroup_obs"),
    ("align_cubes_ompl_only.py",     ["--object", "group",   "obs"],    "align_cubes_group_obs"),
    ("align_cubes_ompl_only.py",     ["--object", "ungroup", "no", "obs"], "align_cubes_ungroup_no_obs"),
    ("align_cubes_ompl_only.py",     ["--object", "group",   "no", "obs"], "align_cubes_group_no_obs"),
    # align_tabung
    ("align_tabung_ompl_only.py",    ["--object", "ungroup", "obs"],    "align_tabung_ungroup_obs"),
    ("align_tabung_ompl_only.py",    ["--object", "group",   "obs"],    "align_tabung_group_obs"),
    ("align_tabung_ompl_only.py",    ["--object", "ungroup", "no", "obs"], "align_tabung_ungroup_no_obs"),
    ("align_tabung_ompl_only.py",    ["--object", "group",   "no", "obs"], "align_tabung_group_no_obs"),
]


def _readiness(scene_filter: str) -> dict:
    """Return HintCache summary for a scene variant."""
    hc = HintCache(log_dir=str(ROOT / "logs"), scene_filter=scene_filter)
    s = hc.summary()
    hint1 = s["pinocchio_skip"]
    hint3 = bool(s["profile_hints"])
    hint2 = bool(s["pos_err_hints"])
    return {
        "logs": s["logs_loaded"],
        "rows": s["rows_loaded"],
        "hint1_pinocchio_skip": hint1,
        "hint2_pos_err": hint2,
        "hint3_profiles": hint3,
        "pinocchio_rate": s["pinocchio_fallback_rate"],
        "profile_hints": s["profile_hints"],
        "pos_err_hints": s["pos_err_hints"],
        "ready": hint1 and hint3,   # hint2 is optional
    }


def _print_readiness_table(variants: list[tuple]) -> dict[str, dict]:
    print(f"\n{'─'*72}")
    print(f"{'SCENE VARIANT':<42} {'LOGS':>4}  H1-PIN  H2-TOL  H3-PROF  READY")
    print(f"{'─'*72}")
    states = {}
    for _, _, scene_key in variants:
        r = _readiness(scene_key)
        states[scene_key] = r
        h1 = "YES " if r["hint1_pinocchio_skip"] else (f"{r['pinocchio_rate']:.2f}" if r["pinocchio_rate"] is not None else "----")
        h2 = "YES " if r["hint2_pos_err"] else "----"
        h3 = "YES " if r["hint3_profiles"] else "----"
        rd = "READY" if r["ready"] else "     "
        print(f"  {scene_key:<40} {r['logs']:>4}  {h1:<6}  {h2:<6}  {h3:<6}  {rd}")
    print(f"{'─'*72}\n")
    return states


def _run_variant(script: str, scene_args: list[str], round_n: int, dry_run: bool) -> bool:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / script),
        *scene_args,
        "--no-viewer",
    ]
    label = f"{script} {' '.join(scene_args)}"
    print(f"  [round {round_n}] {label}")
    if dry_run:
        print(f"           DRY-RUN — skipping execution")
        return True
    t0 = time.perf_counter()
    env = os.environ.copy()
    env["CTAMP_TRACE_CONSOLE"] = "false"
    env["ENABLE_VIEWER"] = "false"
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    elapsed = time.perf_counter() - t0
    ok = result.returncode == 0
    print(f"           {'OK' if ok else 'FAILED (returncode=' + str(result.returncode) + ')'}"
          f"  ({elapsed:.0f}s)")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect HintCache training logs.")
    parser.add_argument(
        "--scenes", nargs="+", default=None,
        help="Scene filter keys to collect (default: all). "
             "E.g. --scenes separate_groups_ungroup_obs align_cubes_group_obs",
    )
    parser.add_argument(
        "--rounds", type=int, default=5,
        help="Max rounds per variant (default: 5). Stops early when ready.",
    )
    parser.add_argument(
        "--no-early-stop", action="store_true",
        help="Keep running even after a variant is marked ready.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without executing any scripts.",
    )
    args = parser.parse_args()

    # Filter variants
    variants = ALL_VARIANTS
    if args.scenes:
        variants = [v for v in ALL_VARIANTS if v[2] in args.scenes]
        if not variants:
            print(f"[ERROR] No variants matched: {args.scenes}")
            print(f"Available: {[v[2] for v in ALL_VARIANTS]}")
            return 1

    print(f"\n{'='*72}")
    print(f"  HintCache log collector")
    print(f"  variants={len(variants)}  max_rounds={args.rounds}  dry_run={args.dry_run}")
    print(f"{'='*72}")

    # Initial state
    print("\n[BEFORE] Current readiness:")
    _print_readiness_table(variants)

    total_runs = 0
    for round_n in range(1, args.rounds + 1):
        print(f"\n{'='*72}")
        print(f"  ROUND {round_n} / {args.rounds}")
        print(f"{'='*72}")

        ran_any = False
        for script, scene_args, scene_key in variants:
            r = _readiness(scene_key)
            if r["ready"] and not args.no_early_stop:
                print(f"  [skip] {scene_key} already ready")
                continue
            _run_variant(script, scene_args, round_n, args.dry_run)
            total_runs += 1
            ran_any = True

        print(f"\n[AFTER round {round_n}] Readiness:")
        states = _print_readiness_table(variants)

        all_ready = all(s["ready"] for s in states.values())
        if all_ready and not args.no_early_stop:
            print(f"  All variants ready after {round_n} round(s). Stopping.\n")
            break

        if not ran_any:
            print(f"  All variants already ready. Nothing to run.\n")
            break

    print(f"\n{'='*72}")
    print(f"  Collection complete.  Total runs: {total_runs}")
    print(f"\n  To verify hints are firing, run:")
    print(f"    python tools/check_hintcache.py")
    print(f"\n  To run with HintCache active:")
    print(f"    python scripts/separate_groups_ompl_only.py --object ungroup obs --no-viewer")
    print(f"{'='*72}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
