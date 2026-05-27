# MVP CTAMP Robot - OMPL Only

Repository ini difokuskan untuk simulasi Franka Panda di MuJoCo dengan IK dan
OMPL joint-space planning. Jalur normal gerak arm adalah:

```text
task script -> scene variant -> IK candidates -> MuJoCo FK validation
            -> joint/state/collision validity -> OMPL plan -> trajectory execution
            -> pick/place feedback -> summary + event logs
```

Direct IK movement tetap bukan jalur utama. IK hanya menghasilkan `goal_q`;
gerak fisik tetap lewat OMPL.

## Status Validasi

Validasi final no-obstacle dari log terbaru:

![Final no-obstacle success matrix](docs/final_no_obs_success_matrix.svg)

| Task | Scene | Result | Log summary |
|---|---|---:|---|
| Cubes | `group_no_obs` | 4/4 | `align_cubes_group_no_obs_20260527_204059.csv` |
| Cubes | `ungroup_no_obs` | 4/4 | `align_cubes_ungroup_no_obs_20260527_205745.csv` |
| Tabung | `group_no_obs` | 4/4 | `align_tabung_group_no_obs_20260527_204857.csv` |
| Tabung | `ungroup_no_obs` | 4/4 | `align_tabung_ungroup_no_obs_20260527_205443.csv` |

Obstacle scenario dipisahkan dari tabel ini karena safety precheck memang dapat
men-skip objek yang terlalu dekat obstacle. Contoh: `tabung_group_obs` terakhir
menolak beberapa objek karena `object_near_obstacle_safety_skip`; itu berbeda
dari arm gagal bergerak.

## Cara Run

Di WSL Ubuntu 22.04:

```bash
cd /mnt/c/projek/MVP-CTAMP-ROBOT
source .venv/bin/activate
python -m pytest -q
python scripts/align_cubes_ompl_only.py --object group no obs --no-viewer
python scripts/align_cubes_ompl_only.py --object ungroup no obs --no-viewer
python scripts/align_tabung_ompl_only.py --object group no obs --no-viewer
python scripts/align_tabung_ompl_only.py --object ungroup no obs --no-viewer
```

Varian obstacle yang masih didukung:

```bash
python scripts/align_cubes_ompl_only.py --object group obs --no-viewer
python scripts/align_cubes_ompl_only.py --object ungroup obs --no-viewer
python scripts/align_tabung_ompl_only.py --object group obs --no-viewer
python scripts/align_tabung_ompl_only.py --object ungroup obs --no-viewer
```

Dependency utama:

```bash
pip install -r requirements.txt
python -c "from ompl import base, geometric; print('ompl ok')"
python -c "import pinocchio, robot_descriptions; print('pinocchio ok')"
```

Jika OMPL Python binding tidak tersedia, script OMPL-only berhenti dengan pesan
setup yang eksplisit.

## Progress Refactor

![Refactor success progress](docs/refactor_success_progress.svg)

![Failure focus by refactor](docs/failure_focus_by_refactor.svg)

### 1. Sebelum Refactor

Fokus awal:

- Menjalankan task OMPL-only untuk cubes/tabung.
- Mengurangi file di luar jalur utama.
- Membaca kegagalan dari log lama.

Pipeline saat itu:

```text
task script
  -> target pose statis
  -> IK MuJoCo DLS legacy
  -> satu goal dominan
  -> OMPL plan
  -> execute trajectory
  -> check_pick/check_place
```

Masalah utama dari log:

- `align_cubes_group_no_obs_20260526_232400.csv`: 3/4, `cube4` gagal pick.
- `align_tabung_group_no_obs_20260526_231404.csv`: 2/4, `circle3` dan
  `circle4` gagal pick.
- Banyak kegagalan masih bercampur sebagai `ik_error_above_plan_limit`, padahal
  beberapa sebenarnya collision/state invalid.
- Grasp untuk object jauh atau borderline reach sering terlalu rendah atau
  offset-nya tidak sesuai.
- Log belum cukup eksplisit untuk membedakan:
  - IK numeric error
  - goal collision invalid
  - OMPL no path
  - trajectory execution collision
  - object not lifted

Notes dari capture log:

- No-obstacle failure bukan karena obstacle avoidance.
- Kegagalan cube paling sering muncul di object paling jauh (`cube4`).
- Tabung/cylinder sensitif terhadap radial contact dan residual contact setelah
  place.

### 2. Refactor 1

Fokus utama:

- Membuat backend initialization eksplisit.
- Menambahkan Pinocchio sebagai IK primary.
- Menambahkan MuJoCo FK validation.
- Menambahkan taxonomy failure reason.
- Menambahkan event logging yang machine-readable.

Pipeline setelah Refactor 1:

```text
task script
  -> target pose
  -> Pinocchio IK primary
  -> MuJoCo FK validation
  -> joint limit validation
  -> planner state validity
  -> OMPL plan
  -> trajectory execution
  -> feedback + summary metrics
```

Perubahan teknis:

- Startup log:
  - `IK_INIT=PINOCCHIO_OK`
  - `IK_INIT=PINOCCHIO_FAILED`
  - `IK_INIT=MUJOCO_DLS_FALLBACK`
  - `OMPL_INIT=OK`
  - `OMPL_INIT=FAILED`
- Failure taxonomy:
  - `ik_error_above_limit`
  - `ik_orientation_error_above_limit`
  - `ik_unreachable`
  - `ik_joint_limit_invalid`
  - `ik_goal_collision_invalid`
  - `ik_goal_state_invalid`
  - `ompl_timeout`
  - `ompl_no_path_found`
  - `execution_failed`
  - `success`
- Event log ditambah kolom backend, candidate id, seed id, FK error,
  state-validity result, OMPL result, dan execution result.

Hasil penting:

- Refactor 1 belum langsung membuat success naik. Pada
  `align_cubes_group_no_obs_20260527_201418.csv`, result turun menjadi 0/4.
- Justru dari log ini akar masalah besar terlihat jelas:
  Pinocchio tersedia dan dipilih, tetapi target dikirim dalam frame world MuJoCo.
  `panda_description` Pinocchio memakai frame base Panda.
- MuJoCo `link0` berada di `[-0.4, 0.0, 0.8]`, sehingga hasil Pinocchio meleset
  sekitar 0.6-0.9 m ketika divalidasi memakai MuJoCo FK.

Notes dari capture log:

- `IK_CANDIDATE` banyak `REJECT` dengan `ik_error_above_limit`.
- `state_valid=True`, tetapi `pos_err` sangat besar. Artinya ini bukan collision
  issue; ini frame mismatch.
- Refactor 1 berguna sebagai tahap observability: error yang sebelumnya kabur
  menjadi bisa dibaca secara spesifik.

### 3. Refactor 2

Fokus utama:

- Membuat Pinocchio dan MuJoCo konsisten frame.
- Memastikan valid grasp contact tidak salah dianggap collision.
- Mengurangi retry yang tidak produktif.
- Menstabilkan grasp object jauh/borderline.

Pipeline final:

```text
task script
  -> target pose world MuJoCo
  -> candidate generator
      - nominal target
      - small XY offsets
      - radial/side offsets untuk object jauh/cylinder
      - z variants untuk pregrasp/release
  -> Pinocchio IK in Panda base frame
  -> MuJoCo FK validation in world frame
  -> fallback MuJoCo DLS per candidate jika Pinocchio FK gagal
  -> joint limit validation
  -> planner state validity with ignored target body for grasp/release
  -> ranked valid candidates
  -> OMPL RRTConnect plan
  -> dense trajectory execution with collision checking
  -> pick/place feedback
  -> summary metrics + event logs
```

Perubahan teknis:

- Target Pinocchio dikonversi dari MuJoCo world frame ke active arm base frame.
- Setiap hasil Pinocchio tetap divalidasi ulang dengan MuJoCo FK.
- Jika Pinocchio FK gagal threshold segmen, kandidat yang sama dicoba ulang
  dengan MuJoCo DLS.
- `planner.is_state_valid_q()` menerima `ignored_body_names`, sehingga target
  object yang sedang digenggam boleh disentuh pada fase grasp/release.
- Candidate valid per segment dibatasi agar run tidak menghabiskan waktu pada
  kandidat redundant.
- Retry pick dihentikan jika object jatuh di bawah meja atau keluar workspace.
- Ranking grasp cube diberi penalti untuk offset besar, jadi center grasp lebih
  diprioritaskan.
- Grasp object jauh/borderline dinaikkan sedikit untuk menghindari
  `table <-> finger` contact tanpa menonaktifkan collision checking.

Hasil setelah Refactor 2:

- `align_cubes_group_no_obs_20260527_204059.csv`: 4/4.
- `align_cubes_ungroup_no_obs_20260527_205745.csv`: 4/4.
- `align_tabung_group_no_obs_20260527_204857.csv`: 4/4.
- `align_tabung_ungroup_no_obs_20260527_205443.csv`: 4/4.

Notes dari capture log:

- OMPL berhasil membuat path untuk goal valid.
- Failure yang tersisa sebelum final mostly bukan OMPL no-path, melainkan grasp
  height/ranking dan target-object collision classification.
- Setelah ignore-list dan far-grasp height fix, `cube4` group no-obs naik dari
  3/4 ke 4/4.

## Insight dari Log

1. No-obstacle failure bukan berarti planner obstacle avoidance buruk.
   Pada run lama, no-obstacle gagal karena IK goal invalid, grasp terlalu rendah,
   atau object tidak terangkat.

2. Label failure harus memisahkan IK numeric dan planner validity.
   Goal dengan FK bagus tetapi collision invalid tidak boleh disebut
   `ik_error_above_limit`.

3. Pinocchio harus selalu divalidasi dengan MuJoCo FK.
   Pinocchio dan MuJoCo dapat memakai frame/model offset berbeda. FK validation
   adalah guard utama sebelum goal masuk OMPL.

4. Grasp target tidak sama dengan transit target.
   Pada grasp/release, target object memang boleh disentuh. State-validity check
   harus memakai ignore-list yang sama dengan OMPL.

5. Object jauh seperti `cube4` perlu treatment khusus.
   Di workspace borderline, sedikit kenaikan grasp height lebih benar daripada
   melonggarkan collision checker.

6. Retry harus berhenti jika object sudah tidak valid.
   Jika object jatuh di bawah meja atau keluar reach, retry IK/OMPL hanya
   membuang waktu dan memperkeruh log.

## Log dan Metrik

Setiap run menghasilkan:

- summary CSV: `logs/<task>_<scene>_<timestamp>.csv`
- event CSV: `logs/<task>_<scene>_<timestamp>_events.csv`

Kolom penting untuk tracing:

- `stage`, `status`, `phase`, `failure_reason`
- `object_id`, `held_object`, `object_xyz`, `object_z`
- `target_xyz`, `actual_xyz`, `distance_to_target`
- `ee_xyz`, `q`, `q_target`, `q_error_norm`, `finger_pos`
- `backend`, `candidate_id`, `seed_id`
- `planner`, `waypoints`, `pos_err`, `ori_err`, `iterations`
- `joint_limit_valid`, `state_valid`, `state_invalid_reason`
- `ompl_result`, `execution_result`
- `collision_pair`, `contact_count`, `penetration`

Urutan baca praktis:

```text
1. Cari status=FAILED.
2. Baca stage + phase.
3. Baca failure_reason.
4. Jika IK_CANDIDATE, cek pos_err/ori_err/state_valid.
5. Jika OMPL_PLAN, cek ompl_result dan goal_attempts di extra_json.
6. Jika TRAJECTORY_EXEC, cek collision_pair dan waypoint.
7. Cocokkan dengan CHECK_PICK/CHECK_PLACE untuk outcome object.
```

Data visualisasi dibuat dari log dan disimpan di:

- `docs/refactor_success_progress.svg`
- `docs/final_no_obs_success_matrix.svg`
- `docs/failure_focus_by_refactor.svg`
- `docs/refactor_metrics.csv`

Notebook analisis lengkap tersedia di:

- `docs/ompl_log_analysis.ipynb`

Notebook tersebut membuat dataframe:

- `summary_df`
- `event_df`
- `tracking_df`
- `error_df`
- `ik_df`
- `ompl_df`
- `trajectory_df`
- `per_log_analysis_df`

Output dataframe dan gambar dari notebook disimpan di `docs/notebook_outputs/`.

## Konfigurasi Penting

Default ada di `.env.example`:

```env
IK_BACKEND=auto
IK_REQUIRE_PINOCCHIO=false
IK_PLAN_POS_ERR_LIMIT=0.020
IK_PREGRASP_POS_ERR_LIMIT=0.030
IK_PLAN_ORI_ERR_LIMIT=0.35
IK_PREGRASP_ORI_ERR_LIMIT=0.50
MAX_VALID_IK_CANDIDATES=6
MAX_IK_ATTEMPTS_PER_SEGMENT=80
OMPL_ENABLED=true
OMPL_REQUIRED=false
OMPL_PLANNER_NAME=RRTConnect
OMPL_TIME_LIMIT=6.0
USE_IK_FALLBACK=false
```

Untuk production-style run, `USE_IK_FALLBACK=false` harus tetap dipertahankan
agar arm tidak bypass OMPL.

## Test

```bash
python -m pytest -q
```

Hasil terakhir:

- Windows interpreter: `9 passed, 1 skipped`
- WSL Ubuntu 22.04 `.venv`: `10 passed`

Skip di Windows terjadi karena OMPL/Pinocchio tidak tersedia di interpreter
aktif Windows, sedangkan WSL `.venv` sudah lengkap.
