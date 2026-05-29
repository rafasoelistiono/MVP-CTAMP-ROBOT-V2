#!/usr/bin/env python3
"""Print current HintCache readiness for every scene variant."""
import sys, os, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["CTAMP_TRACE_CONSOLE"] = "false"
os.environ["CTAMP_EVENT_LOG_CSV"] = ""

from hint_cache import HintCache

SCENES = [
    "separate_groups_ungroup_obs",
    "separate_groups_group_obs",
    "separate_groups_group_no_obs",
    "separate_groups_ungroup_no_obs",
    "align_cubes_ungroup_obs",
    "align_cubes_group_obs",
    "align_cubes_group_no_obs",
    "align_cubes_ungroup_no_obs",
    "align_tabung_ungroup_obs",
    "align_tabung_group_obs",
    "align_tabung_group_no_obs",
    "align_tabung_ungroup_no_obs",
]

MIN_SAMPLES = int(os.getenv("HINT_MIN_SAMPLES", "5"))

print(f"\n{'='*70}")
print(f"HintCache readiness report  (MIN_SAMPLES={MIN_SAMPLES})")
print(f"{'='*70}")

for scene in SCENES:
    hc = HintCache(log_dir=str(ROOT / "logs"), scene_filter=scene)
    s = hc.summary()
    logs = s["logs_loaded"]
    rows = s["rows_loaded"]
    if logs == 0:
        print(f"\n{scene}")
        print(f"  No logs found — cold start")
        continue

    rate = s["pinocchio_fallback_rate"]
    skip = s["pinocchio_skip"]
    pos  = s["pos_err_hints"]
    prof = s["profile_hints"]

    pin_status = "WILL SKIP" if skip else (f"rate={rate:.2f} (threshold 0.70)" if rate is not None else "insufficient data")

    print(f"\n{scene}")
    print(f"  logs={logs}  rows={rows}")
    print(f"  Hint1 pinocchio_skip : {pin_status}")
    print(f"  Hint2 pos_err_hints  : {pos if pos else 'none active'}")
    print(f"  Hint3 profile_hints  : {prof if prof else 'none active'}")

print(f"\n{'='*70}\n")
