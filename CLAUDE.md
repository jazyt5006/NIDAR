# NIDAR AirMouse: Survivor Localisation and Grid Tagging

## Scope
Personal repository. There is no shared team repo. This is a standalone Python 3 + numpy library for one subsystem:
detection + depth -> 3D position -> SLAM map frame -> official NIDAR grid cell -> reliable tag.
- Core library has NO ROS dependency. ROS 2 wrappers come later in a separate package (see docs/design/architecture.md section g).
- Consumers: the controls team will drive it from their simulation; later it runs as ROS 2 nodes on a Raspberry Pi 5.
- Do NOT build (spec Part 1.2): human detection / YOLO model, flight controller, motor control, stabilisation, obstacle avoidance,
  path planning, autonomous navigation, a complete GCS, a complete mission manager, or a replacement SLAM. Consume their outputs
  through defined interfaces. Do not duplicate other teams' functionality.

## Authoritative documents
- Master spec: `docs/specs/survivor_tagging_master_spec.md`. Read parts on demand.
- Rules: `docs/reference/NIDAR 26-27 - Rulebook.pdf` (page refs below are the printed "Page N of 25" numbers; AirMouse
  rules 8.20-8.31 on pp.12-13, AirMouse scoring p.19, penalties p.20) and `docs/reference/Mission Brief - NIDAR AirMouse.pdf`.
- Design: `docs/design/architecture.md`, `docs/design/conventions.md` (frames, time, units), `docs/design/teammate_questions.md`.
- Module map (purpose, public API, invariants, import bans, test files, commands): `docs/design/CODEMAP.md`.
  Library: `types`, `config`, `geometry` (L1), `transforms` (L2), `grid` (L3), `tracking` (L4), `footprint`,
  `calibration`, `output` (telemetry codec), `pipeline`. Harness side: `sim`, `scenarios`, `harness`, `baseline`
  (NaivePipeline control, never reuse), `sim_pipelines`, `diagnostics`, `experiments` (3.1b), `experiments_calib`
  (3.1c). Harness results are SIMULATION tested only. Test config: `configs/example_synthetic.json` (TEST VALUES ONLY).
- `l1.body_depth_offset_m` stays 0.0 until measured on real bodies (Step 5); never set it from simulator results.
- Pipeline order: capture-time pose -> floor plane -> Layer 1 -> camera-to-map -> footprint -> Layer 3 -> Layer 4.
  Floor removal (`l1.floor_removal`, needs `map.floor_z_m`) and footprint refinement (`l1b.footprint_refinement`) are
  OFF by default: both failed their pre-set decision rules (3.1b).
- Layer 4 ray sigma = sqrt(`l4.ray_bias_sigma_m`^2 + `l1.body_depth_offset_sigma_m`^2). End-of-mission policy
  (`final_tags`) stays configurable until O2 is answered; `expected_value` threshold P/(40+P), P =
  `score.wrong_tag_penalty_assumed` (unset = 0.5 + note). 3.1c calibration sweep FAILED its rule; redo in Step 5.
- Five-layer architecture is FIXED: (1) robust survivor geometry, (2) time-aligned coordinate transformation, (3) NIDAR grid
  registration, (4) temporal consistency and uncertainty, (5) correction handling and live map output.
  Changing it needs a written technical reason and my approval.

## Standing rules
- Start every task by reading CLAUDE.md and `docs/design/CODEMAP.md`, then only the files you will change or that
  directly depend on them. Read spec sections only when the task names them.
- Python 3.10+, numpy only, seeded randomness. Library code must not import sim, scenarios, baseline or harness.
- Never invent hardware or competition values. New tuning defaults are labelled ASSUMPTION and listed in the report.
  Unknowns stay configurable.
- Never weaken or skip a test. During development run only the affected tests; run the full suite once at the end.
- Do not commit unless asked.
- Evaluations: cache expensive intermediate results; print only summary tables to the terminal; save detailed output
  under `docs/reports/` (not a temp folder).
- Update `docs/design/*.md` only at the end of a step (when the task says so). In between, append a few lines per task
  to `docs/reports/NOTES.md` (decisions, new parameters, deviations).
- Do not re-run the full evaluation or the full test suite after the final successful run unless code changed since.
- Include only the tables the task asks for in the report.
- If the context was compacted, check git status and the report files first instead of re-running everything.
- Reporting: write the full report to `docs/reports/<task id>.md` using `docs/reports/TEMPLATE.md`. In chat print only:
  a one-line PASS/FAIL, the report path, the test counts, and at most 10 lines of key numbers or decisions.
- Decision rules given in a task are followed as written, even if the results disagree with expectations.
- Results are labelled by evidence level: unit, simulation, Gazebo, recorded-data, hardware.

## Competition facts that constrain the design
Verified against the PDFs in `docs/reference/` on 2026-11-08 unless marked otherwise.
- Scoring: 40 marks per survivor correctly detected AND correctly tagged on the generated 2D map; tagging means marking
  the correct grid reference on the 1 m x 1 m reference grid; max 6 survivors, 240 marks (rulebook Phase 4D item 1, p.19).
  The metric is correct grid cell, not detection count. Separate 220-mark item for 2D grid mapping accuracy of 1 m cells
  in the 15 m x 15 m grid zone (p.19, not this subsystem).
- Manual intervention includes manual navigation assistance, map editing, localisation correction, survivor tagging or
  path-planning input (rulebook 8.28, p.13; brief s.4, s.5). Penalty -50 per instance for manual navigation, flight
  control, path correction, survivor tagging, map correction input or reset after start, except safety abort or
  emergency recall; total penalties may be capped at 150 (rulebook Phase 4D penalties item 4, p.20). Tagging must be autonomous.
- No wrong-tag or extra-tag penalty is listed in the AirMouse penalty table (p.20). This does not settle open question 2.
- Detection uses onboard sensing and processing (brief s.6). Mapping, localisation, processing and transfer to the
  GCS happen during flight; no post-flight time (brief s.4, s.5).
- No GSM, LTE, 5G, public Wi-Fi, internet, cloud or other external network during the mission (rulebook 8.4, p.11;
  brief s.7). All communication via locally deployed systems; no tethers or wired links to the drone in flight (rulebook 8.5, p.11).
- GCS must display at minimum: mission status, live camera feed, drone position or estimate, generated map, tagged
  survivor locations, mission progress (rulebook 8.27, pp.12-13). The brief additionally requires the grid coordinate or
  grid box containing each detected survivor (brief s.5, s.6).
- Arena at most 15 m x 15 m on a modular grid; corridors at least 1 m wide; standard room 2 m x 2 m; same entry and exit
  point (rulebook 8.21, 8.22, p.12; brief s.3).
- [ASSUMPTION] Telemetry link is low bandwidth (tens of kbps) until the radio is chosen. Compact tag messages plus periodic
  full snapshots. No images or point clouds over telemetry.
- Cell size is 1 m (p.19), but grid origin, axes, numbering and entry cell are NOT known (open question 1).

## Hardware (spec Part 2) - VERIFY, DO NOT ASSUME
- Raspberry Pi 5, 16 GB, shared with SLAM, avoidance, planning, detection, thermal processing. Must stay lightweight.
- Hailo-8 / Hailo-8L (variant unconfirmed): used by detection. Not available for geometry.
- Intel RealSense D435i: primary RGB, depth, IMU. Intrinsics, extrinsics, depth scale, resolution, frame rate and mounting
  come from the device/driver or measured calibration, never hardcoded. Firmware/librealsense/driver versions unverified.
- RPLIDAR C1: owned by SLAM.
- Pi Camera Module 3 Wide, VL53L5CX x2, VL53L1X, MicoAir MTF-01, ESP32-S3, OV5647, MLX90640, FC + telemetry: not in this
  pipeline unless a role, calibration, frame and timestamps are confirmed. No automatic fusion.

## Non-negotiable rules (spec 1.3, 4.3, 6.4, 10, 11, 14.3, 14.4)
- Never assume: a detection gives a correct position; a depth sample belongs to the person; the SLAM origin is the grid
  origin. Never confirm a cell on insufficient evidence.
- No invented values: camera mount angle, extrinsics, intrinsics, altitude source, pose availability, ROS 2 distro, SLAM
  correction API, grid origin, grid orientation, grid indexing, sensor timestamps, pose rate, penalties. Unknown = config
  parameter or question. Any default is labelled "ASSUMPTION". No silent placeholders.
- Use the pose at image capture time. Never the latest pose. Never callback time as capture time. Fail explicitly when
  the historical pose is unavailable. Do not compare clocks from different domains without a known relation.
- Every failure returns an explicit reason code (architecture.md section d). Never zeros or fabricated positions.
- Detection confidence is not localisation confidence. Never use an arbitrary confidence number as position uncertainty.
- No fabricated results. Label every result: unit tested / simulation tested / recorded-data tested / hardware tested /
  flight validated. No claims about Pi performance, real-time behaviour, driver/camera function or accuracy without measurement.
- Lightweight: no second detector, no full-resolution copies, no point clouds unless required, bounded history; measure
  CPU, RAM, latency and drops.
- Thresholds are configurable and experimentally validated, never hardcoded from discussion.

## Code conventions
- Python 3, numpy only in the core. src layout: `src/survivor_tagging/`. Tests in `tests/` (pytest).
- Portable: develop on Windows (Git Bash), run on Linux (Pi 5). Use pathlib; no OS-specific calls.
- SI units in code: metres, radians, nanoseconds (int) for timestamps. Unit in every field name or docstring.

## Working agreement
- One numbered substep at a time. Test, report, then wait for my approval.
- Inspect before modifying. Reuse existing code. Do not overwrite working modules.
- Ask first before: adding dependencies, installing anything, changing build files, touching files outside this repo,
  committing, git push or force operations, deleting or moving files.
- Every substep report also states: functionality, dependencies added, tests run (see Standing rules: Reporting).
- Keep README, architecture, interface, configuration and test docs current. Record every assumption.

## Build, test, run
Verified on dev machine (Windows, Git Bash, Python 3.13.5, numpy 2.4.2, pytest 9.2.0) on 2026-11-08:
- `python -m pytest`  (pyproject sets testpaths=tests and pythonpath=src; no install step needed)
- `PYTHONPATH=src python -c "import survivor_tagging"`
- `PYTHONPATH=src python -m survivor_tagging.harness`  (naive baseline, all scenarios x 5 seeds, ~17 s on dev laptop)
- `PYTHONPATH=src python -m survivor_tagging.experiments`  (substep 3.1b tables, ~215 s on dev laptop)
- `PYTHONPATH=src python -m survivor_tagging.experiments_calib`  (substep 3.1c, ~800 s, 11 processes)
Not verified on Python 3.10/3.11 or on the Pi.

## Frames, topics, messages
Core library uses no topics. Frames: camera optical frame, body frame, SLAM map frame, official grid frame (definitions in
architecture.md). Proposed ROS 2 boundary in architecture.md section g; none of it exists yet.

## Open questions (full list with owners and default assumptions: docs/design/teammate_questions.md)
1. Official grid origin, axes, indexing, entry cell, whether walls lie on grid lines (organisers).
2. Are wrong or extra tags penalised? Affects the end-of-mission commit policy (organisers).
3. Dummy appearance and posture. Will the detector recognise dummies? (detection)
4. Do detections come from the D435i RGB stream or another camera? (detection)
5. Is the pose full 3D or planar? Does SLAM expose corrected historical poses? (SLAM)
6. Camera mounting extrinsics (hardware).
7. Telemetry radio and its data rate (hardware).
8. How will the controls simulation feed depth, pose and detections into this library? (controls)
9. Pi OS, ROS 2 distro, RealSense driver status (hardware).
10. O5: if a survivor lies across two grid cells, which point decides the cell? Default ASSUMPTION: the cell
    containing the centre of the body (organisers).
11. O6: how are two survivors in the same cell scored? Default ASSUMPTION: one tag per survivor (organisers).

## Candidate ideas (from design discussion; NOT requirements; do not implement without approval)
- Grid registration from wall alignment: histogram wall coordinates modulo cell size for sub-cell offset and heading
  modulo 90 degrees (needs question 1).
- Ray check: if an estimate lands on a wall, step back along the camera ray toward the drone to the nearest free cell.
- Cross-check the depth position against a floor projection (altitude, attitude, camera tilt).
- Per-cell probability by integrating the position uncertainty over each cell.
- Store observations in pose-relative form so tags are recomputed after SLAM corrections (re-derive, do not patch).
- End-of-mission commit policy for unconfirmed candidates (depends on question 2).
