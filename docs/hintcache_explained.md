# Log-Based Adaptive Rules — What They Are and Why They Work

*Based on real log data from 2026-05-29 runs on MVP-CTAMP-ROBOT-V2.*

---

## The Problem These Rules Were Built To Solve

After Refactor 3, all 8 align variants ran at 4/4. But when the new
`separate_groups` task was added, the first run of `ungroup_obs` took
**22.5 minutes** — nearly 3× longer than expected (~8 min).

The root cause: for the `ungroup_obs` scene specifically, the robot base
position in `panda_ungroup_obs.xml` causes Pinocchio IK's FK validation to
fail on **65–68% of all IK attempts**. Every failure means:

1. Pinocchio tries to solve IK → produces a joint config
2. MuJoCo FK re-validates it → position error 0.38–0.85 m (way above 0.02 m limit)
3. System falls back to MuJoCo DLS → DLS solves it correctly but takes ~350ms per attempt
4. This happens **hundreds of times per run** — burning time on a known-bad backend

The system had no way to remember this between runs. Every run started from
scratch, hit the same wall, and paid the same cost.

The solution: read past event logs at startup, compute statistics, and use
those statistics to skip known-bad paths before they waste time again.

---

## How the Rules Work

The logic lives in `src/hint_cache.py` and is called from `src/executor.py`.
It is a plain Python class — no library, no ML model, no trained weights.
It reads `*_events.csv` log files, counts events, computes rates and medians,
and produces up to three scalar outputs. Every decision it makes is visible
in the `HINT_CACHE LOADED` event at the start of each run.

**Cold-start safe:** if there are fewer than `MIN_SAMPLES=5` data points
for a bucket, the rule returns `None` and the executor uses its defaults.
The first run always works — it just doesn't benefit from the rules yet.

---

## Workspace Buckets

The rules don't learn globally — they learn per **workspace region**,
because the arm behaves differently depending on where an object is.

**Reach buckets** (distance from arm base):
| Bucket | Range |
|---|---|
| `near` | < 0.50 m |
| `mid` | 0.50 – 0.70 m |
| `far` | 0.70 – 0.85 m |
| `borderline` | > 0.85 m |

**Obstacle buckets** (distance to nearest ceramic obstacle):
| Bucket | Range |
|---|---|
| `clear` | > 0.28 m |
| `near` | 0.12 – 0.28 m |
| `too_close` | < 0.12 m |

---

## Rule 1 — Skip Pinocchio When It Keeps Failing

**What it does:** if Pinocchio FK validation has been failing at a rate
≥ 70% across all IK attempts in past logs, skip Pinocchio entirely for
the whole run and go straight to MuJoCo DLS.

**How it's computed:**
```
fallback_count    = IK_SOLVE BACKEND_FALLBACK events (pinocchio_fk_validation_failed)
ik_candidate_count = total IK_CANDIDATE events
rate = fallback_count / ik_candidate_count
if rate >= 0.70 → skip Pinocchio
```

**Why it works:** Pinocchio solves IK in its own coordinate frame and
sometimes produces joint configs that are correct in Pinocchio space but
wrong in MuJoCo space (position error 0.38–0.85 m). When this happens
consistently for a given scene, there is no point trying Pinocchio at all —
MuJoCo DLS will just be called anyway after every failure, wasting ~350ms
per attempt. The rule detects the pattern from logs and eliminates the
wasted overhead upfront.

**Real numbers from logs (2026-05-29):**
```
separate_groups_ungroup_obs:
  fallback_count     = 3333
  ik_candidate_count = 5087
  rate               = 0.655   ← not yet at threshold 0.70
  rule active        = False   ← needs ~2 more runs
```

**When it fires:** executor skips Pinocchio entirely, no FK validation
overhead, DLS runs directly. Expected speedup: ~40–50% runtime reduction
for `ungroup_obs`.

---

## Rule 2 — Widen IK Tolerance for Hard Workspace Corners

**What it does:** for a specific (reach, obstacle) bucket where many IK
candidates are "near misses" (position error just slightly above the limit),
widens the acceptance threshold so those candidates are accepted instead
of rejected.

**How it's computed:**
- Looks at `IK_CANDIDATE REJECT` events with `failure_reason=ik_error_above_limit`
- A "near miss" is: `pos_limit < pos_err ≤ pos_limit × 1.50`
- If near-miss rate ≥ 40% in a bucket → widen threshold
- Widened value = `median(near_miss_errors) × 1.10`, capped at `pos_limit × 1.60`

**Why it works:** some workspace corners (mid reach, near obstacle) produce
IK solutions that are geometrically close enough to execute safely, but get
rejected by the default tolerance. Logs reveal when this is happening
systematically. Widening the tolerance only for those specific buckets
recovers otherwise-discarded valid solutions without relaxing the global limit.

**Real numbers from logs (2026-05-29):**
```
separate_groups_group_obs:
  ('mid', 'near') bucket → widened to 0.0228 m
```

All other variants: no near-miss pattern detected → rule not active.
This rule is the most situational — it only fires in genuinely hard
workspace corners.

---

## Rule 3 — Start With the Grasp Profile That Actually Works

**What it does:** instead of always starting with grasp profile 0 and
retrying on failure, start with whichever profile has the highest
historical success rate for this object class and reach distance.

**How it's computed:**
- Pairs `PICK_PROFILE SELECT` events with subsequent `CHECK_PICK` outcomes
- Groups by `(obj_class, reach_bucket)`
- Returns the profile index with the highest success rate (needs ≥ 5 samples)

**Why it works:** the executor has three grasp profiles per object type
(different grip widths and grasp offsets). For cylinders, profile 0 is
almost always wrong — the grasp offset overshoots the cylinder body. Without
this rule, every cylinder pick wastes one full failed attempt before landing
on the right profile. Logs make the pattern obvious: profile 1 wins for
cylinders at mid and far reach, profile 2 for borderline reach. The rule
just skips the guaranteed failure.

**Real numbers from logs (2026-05-29):**
```
separate_groups_ungroup_obs:
  ('cube', 'near')         → profile 0
  ('cube', 'mid')          → profile 0
  ('cube', 'far')          → profile 0
  ('circle', 'mid')        → profile 1  (grasp_offset=0.095)
  ('circle', 'far')        → profile 1
  ('circle', 'borderline') → profile 2  (wider clearance for far cylinders)
```

---

## What These Rules Solved (and What They Didn't)

### What they solved

| Problem | Fix |
|---|---|
| Cylinders always fail profile 0 first | Rule 3 starts at profile 1 directly |
| `circle_far` needs extra clearance | Rule 3 jumps to profile 2 for borderline reach |
| `group_obs` has near-obstacle IK near-misses | Rule 2 widens tolerance for `(mid, near)` bucket |

### What they have NOT solved

| Problem | Status |
|---|---|
| Pinocchio wasting time on `ungroup_obs` | Rate 0.655, threshold 0.70 — **needs ~2 more runs** |
| circle2 being flung by circle4's trajectory | **Cannot be fixed here** — it's an OMPL collision world issue |
| `ungroup_no_obs` still failing 2/8 | circle2 + circle4 displacement — same OMPL issue |

---

## The Real Remaining Problem: OMPL Collision World

These rules are an **IK-layer optimization**. They cannot fix physics-level
failures.

The `circle2` failure that keeps appearing is:
1. circle2 placed successfully → `CHECK_PLACE OK`
2. Arm picks circle4 → OMPL plans a trajectory **not knowing circle2 exists**
3. Arm physically sweeps through circle2 during the circle4 trajectory
4. `PLACED_OBJECT_DISTURBED` detects the nudge → re-queues circle2
5. circle2 is already at `z=0.022` (flung off table) → pick attempt fails
6. `attempts=2` hits `max_pick_attempts=3` → permanent failure

The fix requires registering placed objects as collision bodies in the
OMPL planner after each `CHECK_PLACE OK`. This is the next engineering
priority and cannot be addressed by log-based rules.

---

## Current Status (2026-05-29)

```
Variant                        Logs  Rule1-Pinocchio  Rule2-Tolerance  Rule3-Profile
──────────────────────────────────────────────────────────────────────────────────────
separate_groups_ungroup_obs      7   0.655 / 0.70     none             5 buckets ✓
separate_groups_group_obs        5   0.540 / 0.70     (mid,near) ✓     4 buckets ✓
separate_groups_group_no_obs     5   0.420 / 0.70     none             4 buckets ✓
separate_groups_ungroup_no_obs   5   0.580 / 0.70     none             5 buckets ✓
align_* variants                 0   cold start       cold start       cold start
```

Rule 3 (profile) is active for all separate_groups variants.
Rule 1 (Pinocchio skip) is not yet active for any variant.

To check current status:
```bash
python tools/check_hintcache.py
```

To collect more logs until Rule 1 fires:
```bash
python tools/collect_logs.py --scenes separate_groups_ungroup_obs --rounds 3
```

---

## Summary

This is a lightweight adaptive layer that sits between the event logs and
the executor. It reads CSV files, counts events, computes rates and medians,
and returns three scalar values to guide execution. It is fully auditable —
every decision traces back to a specific count ratio in the logs.

Its most impactful rule (Rule 1 — Pinocchio skip) is not yet active because
the Pinocchio failure rate (0.65) is just below the threshold (0.70). Once
it crosses that line with 1–2 more runs, the `ungroup_obs` scene is expected
to drop from ~4.5 min to ~2.5 min.

The cylinder-flung-off-table failure is outside the scope of these rules.
That is an OMPL collision world registration problem and requires a separate
fix in `ompl_planner.py`.
