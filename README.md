# CTAMP Robot Arm Log Analysis

Dokumen ini merangkum analisis run OMPL + MuJoCo terbaru untuk task `align_cubes` dan `separate_groups`.

Analisis dibuat dari event CSV di `logs/` menggunakan `pandas` + `matplotlib` melalui:

```bash
python scripts/analyze_run_logs.py
```

Output analisis:

```text
docs/log_analysis/analysis_summary.md
docs/log_analysis/failure_hotspots.png
docs/log_analysis/ompl_outcomes_by_run.png
docs/log_analysis/pipeline_funnel.png
docs/log_analysis/pick_attempts_by_object.png
docs/log_analysis/metrics.json
```

## Dataset Log

Log yang dianalisis:

```text
align_cubes_ungroup_obs_20260514_060004_events.csv
separate_groups_ungroup_obs_20260514_055533_events.csv
separate_groups_ungroup_obs_20260514_060627_events.csv
separate_groups_ungroup_obs_20260514_060855_events.csv
separate_groups_ungroup_obs_20260514_061115_events.csv
```

Aggregate metrics:

| Metric | Value |
|---|---:|
| Total events | 2111 |
| OMPL starts | 132 |
| OMPL OK | 126 |
| OMPL failed | 4 |
| OMPL error | 0 |
| Trajectory starts | 126 |
| Trajectory OK | 74 |
| Trajectory failed | 51 |
| IK warnings | 108 |
| Collision blocked | 51 |
| Table-left_finger blocked at waypoint 0 | 51 |
| Pick failed events | 57 |
| Place OK events | 8 |

Summary run:

| Task | Scene | Success | Objects moved | Objects total | Failed |
|---|---|---:|---:|---:|---:|
| align_cubes | ungroup_obs | False | 2 | 4 | 2 |
| separate_groups | ungroup_obs | False | 3 | 8 | 5 |
| separate_groups | ungroup_obs | False | 2 | 8 | 6 |

## Visualisasi Data

Failure hotspot:

![Failure hotspots](docs/log_analysis/failure_hotspots.png)

OMPL outcome per run:

![OMPL outcomes](docs/log_analysis/ompl_outcomes_by_run.png)

Pipeline funnel:

![Pipeline funnel](docs/log_analysis/pipeline_funnel.png)

Pick attempts per object:

![Pick attempts](docs/log_analysis/pick_attempts_by_object.png)

## Insight

1. OMPL mayoritas berhasil menemukan path.

   Dari 132 `OMPL_PLAN START`, ada 126 `OMPL_PLAN OK`. Hanya 4 event yang benar-benar `OMPL_PLAN FAILED`. Jadi bottleneck utama bukan planner tidak menemukan solusi.

2. Banyak path gagal setelah masuk executor.

   Ada 126 `TRAJECTORY_EXEC START`, tetapi hanya 74 `TRAJECTORY_EXEC OK` dan 51 `TRAJECTORY_EXEC FAILED`. Ini menunjukkan path yang sudah ditemukan OMPL sering ditolak saat validasi live trajectory.

3. Failure paling dominan adalah table-finger collision di waypoint pertama.

   Semua 51 `COLLISION_CHECK BLOCKED` adalah:

   ```text
   robot-env contact: table/1 <-> left_finger/82
   phase=trajectory waypoint 0
   ```

   Ini berarti robot belum sempat bergerak. Path langsung diblok di start waypoint.

4. Retry grip memperlihatkan masalah start-state, bukan hanya masalah gripping.

   Setelah retry/drop, gripper sering berada dekat meja. Saat retry berikutnya dimulai, waypoint 0 masih mengandung kontak `left_finger` dengan table. Executor memblokir sebelum arm bergerak.

5. IK masih sering tidak presisi.

   Ada 108 `IK_SOLVE WARN`. Beberapa object, terutama `circle*` dan object jauh, memiliki `pos_err` besar. Artinya OMPL bisa planning ke joint goal, tetapi end-effector belum tentu tepat di pose grasp yang diinginkan.

## Penyebab Failed

Penyebab utama:

```text
live trajectory checker memblokir waypoint 0 karena table <-> left_finger contact
```

Detail teknis:

1. OMPL planner menganggap start state valid karena itu state robot saat ini.
2. Executor tetap melakukan collision check pada waypoint 0.
3. Jika gripper sedang menyentuh/melewati meja sedikit setelah `drop()` atau retry, waypoint 0 langsung gagal.
4. Karena gagal di waypoint 0, robot terlihat stuck walaupun `OMPL_PLAN OK`.

Penyebab sekunder:

```text
IK target kurang presisi dan retry grasp belum mengubah strategi geometri secara cukup
```

Retry saat ini mengubah grip, grasp height, dan clearance. Namun belum mengubah approach direction, XY offset, side grasp, atau sampling grasp candidate.

Penyebab yang bukan dominan:

```text
obstacle dekat
```

Obstacle tetap penting untuk safety, tetapi dari log terbaru failure terbanyak bukan obstacle collision. Failure terbanyak adalah table-finger collision di start waypoint.

## Improvements Berikutnya

Prioritas 1: perbaiki validasi waypoint 0.

Rekomendasi:

```text
Jika waypoint_index == 0 dan q sama dengan current_q, jangan block karena table-finger contact kecil.
Tetap block obstacle/vase/glass collision.
Mulai strict validation dari waypoint 1.
```

Efek yang diharapkan:

```text
OMPL_PLAN OK benar-benar dilanjutkan ke gerakan robot.
Trajectory failed karena table-finger waypoint 0 turun signifikan.
```

Prioritas 2: recovery pose setelah drop.

Rekomendasi:

```text
Setelah drop(), buka gripper lalu move arm ke safe hover atau GRASP_READY.
Jangan retry pick dari pose jari dekat meja.
```

Efek yang diharapkan:

```text
Retry pick tidak dimulai dari state collision.
PICK failed karena move_pregrasp_failed berkurang.
```

Prioritas 3: validasi IK sebelum OMPL.

Rekomendasi:

```text
Jika IK pos_err > 0.03 m untuk pregrasp/grasp, jangan langsung panggil OMPL.
Coba null-space seed tambahan, XY pregrasp offset, atau target pose alternatif.
```

Efek yang diharapkan:

```text
OMPL tidak membuang waktu planning ke goal yang secara task-space buruk.
Object lebih sering benar-benar ter-grip.
```

Prioritas 4: grasp sampler nyata.

Rekomendasi:

```text
Cube: top grasp + offset kecil dari 4 sisi.
Circle/cylinder: top grasp + radial side grasp.
Retry: ubah pose grasp, bukan hanya grip width.
```

Efek yang diharapkan:

```text
PICK failed object_not_lifted berkurang.
Circle lebih mudah diangkat.
```

Prioritas 5: improve OMPL secara langsung.

Rekomendasi:

```text
Gunakan goal region, bukan satu goal joint hasil IK.
Tambahkan clearance objective terhadap obstacle dan table.
Tambahkan multiple seeds per phase.
Gunakan path shortcut/smoothing yang tetap collision-aware.
```

Efek yang diharapkan:

```text
No-solution OMPL berkurang.
Path lebih jauh dari obstacle dan table.
```

Prioritas 6: opsi arsitektur two-arm untuk task padat.

Jika scene makin crowded, one-arm pick-place akan sering melewati area sempit dan sulit menjaga stabilitas object. Two-arm dapat dipertimbangkan untuk:

```text
satu arm melakukan stabilisasi / guarding
satu arm melakukan pick-place
atau satu arm memindahkan obstacle non-fragile/staging object jika task diperluas
```

Untuk constraint saat ini, obstacle fragile tetap tidak boleh disentuh.

## Cara Reproduce Analisis

1. Jalankan task:

```bash
python scripts/separate_groups_ompl_only.py --object ungroup obs
python scripts/align_cubes_ompl_only.py --object ungroup obs
```

2. Generate analisis:

```bash
python scripts/analyze_run_logs.py
```

3. Buka gambar di:

```text
docs/log_analysis/
```

4. Baca ringkasan otomatis:

```text
docs/log_analysis/analysis_summary.md
```
