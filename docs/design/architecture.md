# Survivor Tagging Core Library: Architecture

Status: Layers 1-5 are implemented (unit and simulation tested only; no recorded, Gazebo or hardware evidence). The sections
(a) to (g) below are the original design (substep 1.1) with in-place corrections; the sections "... as implemented" at the
end describe what the code does and where it deviates. Last updated: end of step 3 (substep 3.3b, 2026-11-09). The complete
list of parameter values with source and evidence level is `configs/frozen_synthetic_v1.json`.
Source of requirements: `survivor_tagging_master_spec.md` (Parts referenced as "spec N.M").
Labels: **ASSUMPTION** = chosen by us, not from hardware, rules or measurement. **needs measurement** = must come from
calibration, the device/driver, a teammate, the organisers or an experiment before it is trusted.

Conventions used throughout: metres, radians, timestamps as integer nanoseconds (`*_ns`) with an explicit clock domain,
right-handed frames. Frames: `camera` (optical frame of the camera that produced the depth; axis convention needs
measurement from the driver), `body`, `map` (SLAM), `grid` (official NIDAR grid).

---

## (a) Module layout mapped to the five layers

```
src/survivor_tagging/
  __init__.py      version only
  types.py         all data types below + FailureReason + TagStatus enums. No logic.
  config.py        ConfigParam registry, loading/validation from configs/*.yaml|json, "unset required" detection.
  geometry.py      Layer 1  robust survivor geometry: bbox validation, depth validation, region extraction,
                            clustering, back-projection, ground reference, measurement covariance.
  transforms.py    Layer 2  T_body_camera handling, PoseBuffer (bounded), capture-time lookup + interpolation,
                            camera->map transform, covariance propagation.
  grid.py          Layer 3  GridRegistration (T_grid_map, versioned), map->grid, cell index, boundary proximity,
                            out-of-grid, per-cell probability.
  tracking.py      Layer 4  SurvivorTrack store, association, decorrelated fusion, status state machine, duplicates.
  output.py        Layer 5  compact telemetry codec (25-byte record, snapshot), live output (PublishedTag, TelemetryScheduler).
  corrections.py   Layer 5  observation store for re-derivation, correction history and fold, TagBook (survivor ids).
  pipeline.py      wiring: Layer123Pipeline, Layer1234Pipeline, Layer5Pipeline (as implemented; the original plan had no
                            pipeline module).
  footprint.py, calibration.py   optional footprint refinement (default off) and reliability/ECE metrics.
  harness.py       Offline driver: feeds (detections, depth, intrinsics, poses, corrections) into the pipeline in
                   timestamp order. Entry point for unit/simulation tests and for the controls team's simulator.
                   Later reused by the ROS 2 wrapper.
configs/           parameter files (none yet). Every hardware value ships unset.
tests/             one test module per layer + harness end-to-end tests.
```

Dependency direction: `types` <- `config` <- layers 1..5 <- `harness`. Layers depend only on lower layers and `types`;
no layer imports ROS, OpenCV or I/O libraries. Only numpy.

---

## (b) Data types (names, fields, units; no implementation)

### Detection (input; spec 4.1)
| Field | Type / unit | Notes |
|---|---|---|
| detection_id | int | unique per image |
| tracking_id | int or None | optional detector track id; never relied on exclusively |
| class_label | str or None | optional |
| confidence | float [0,1] | detector confidence, NOT localisation quality |
| bbox_px | (x_min, y_min, x_max, y_max) float, pixels | image coordinates of `image_frame_id` |
| image_stamp_ns | int ns | capture time of the image, not callback time |
| clock_domain | str | e.g. "camera_hw", "ros_system"; relation to pose clock must be known |
| image_frame_id | str | camera that produced the image (D435i colour or other; open question D2) |
| image_size_px | (width, height) int | from the stream, not assumed |

### DepthMeasurement (input; spec 4.2. Only the region needed, no full-frame copy retained)
| Field | Type / unit | Notes |
|---|---|---|
| depth_stamp_ns | int ns | capture time of depth frame |
| clock_domain | str | |
| frame_id | str | must equal the frame the bbox refers to (aligned depth) or an explicit calibrated mapping exists |
| depth_raw | 2D array, uint16 raw units (view, not copy) | |
| depth_scale_m_per_unit | float m/unit | from device |
| intrinsics | fx, fy, cx, cy (px); distortion model + coeffs; width, height (px) | from CameraInfo/device |
| aligned_to_color | bool | |
| valid_range_m | (min, max) m | from config (see (c)) |

### PoseSample (input; spec 4.3)
| Field | Type / unit | Notes |
|---|---|---|
| stamp_ns | int ns | |
| clock_domain | str | |
| frame_id / child_frame_id | str | e.g. map / body |
| position_m | (x, y, z) m | z source must be stated if SLAM is planar |
| orientation | unit quaternion (w, x, y, z) | roll/pitch source must be stated if SLAM is planar |
| covariance | 6x6 (m², rad²) or None | |
| valid | bool | |
| pose_revision | int | increments when SLAM revises history |
| source | str | e.g. "slam_2d+fc_attitude+rangefinder"; provenance for diagnostics |

### SurvivorObservation (output of layers 1-3 for one detection; stored for re-derivation, spec 9.2)
| Field | Type / unit | Notes |
|---|---|---|
| observation_id | int | |
| detection_id, tracking_id, confidence | copied | |
| capture_stamp_ns | int ns | |
| p_camera_m, cov_camera_m2 | (3,) m, 3x3 m² | Layer 1 |
| ground_ref_camera_m | (3,) m or None | None = not reliably estimable (spec 5.7) |
| n_valid_px, valid_fraction, depth_quality | int, float [0,1], float [0,1] | |
| extrinsic_version | int | |
| pose_stamps_ns, interp_alpha | (t0, t1) ns, float | pose association for re-derivation |
| pose_revision | int | |
| p_map_m, cov_map_m2 | (3,) m, 3x3 m² or None | Layer 2 |
| p_grid_m, cell (i, j), cell_probs | (2,) m, ints, {cell: prob} or None | Layer 3 |
| registration_version | int | |
| boundary_distance_m | float m | distance to nearest cell edge |
| failure | FailureReason or None | first failure that stopped processing |
| warnings | list[FailureReason] | non-fatal degradations |

### SurvivorTrack (Layer 4; spec 8.3)
| Field | Type / unit | Notes |
|---|---|---|
| survivor_id | int | stable id we assign |
| detector_tracking_ids | set[int] | all seen, for re-association |
| first_seen_ns, last_seen_ns | int ns | |
| p_map_m, cov_map_m2 | (3,) m, 3x3 m² | fused |
| p_grid_m, cell, cell_probs | (2,) m, (i, j), {cell: prob} | |
| max_detection_confidence | float | |
| localisation_quality | float [0,1] | derived from covariance, not detector confidence |
| observation_count, effective_independent_count | int, float | latter discounts correlated frames |
| observation_ids | bounded deque[int] | bounded history |
| status | TagStatus | CANDIDATE, PROVISIONAL, CONFIRMED, UPDATED, UNRELIABLE |
| last_pose_revision, registration_version | int | |
| reasons | list[FailureReason] | why it is not CONFIRMED |

### Tag (Layer 5 output; full form in (e))
Derived view of a SurvivorTrack at an update time: see (e).

### ConfigParam
| Field | Type | Notes |
|---|---|---|
| name | str | dotted, e.g. `l1.depth_min_m` |
| unit | str | |
| value | number / array / str or None | None = unset |
| required | bool | required + unset blocks the affected stage with `CONFIG_MISSING` |
| source | enum | `needs measurement` or `ASSUMPTION` (later: `measured`, `rulebook`, `teammate` with reference) |
| reference | str | where the value came from (doc, date, measurement id) |
| valid_range | (min, max) or None | load-time validation |

---

## (c) Configurable parameters

No hardware or competition value has a default. Tuning defaults exist only so tests can run, and are all ASSUMPTION
until validated experimentally (spec 8.6: thresholds are experimentally validated).

### Sensor / calibration (all required, no default)
| Name | Unit | Meaning | Source |
|---|---|---|---|
| cam.intrinsics_source | enum | `device` (CameraInfo / librealsense) or `file` | needs measurement |
| cam.depth_scale_m_per_unit | m/unit | from device at runtime; config only as cross-check | needs measurement |
| cam.depth_frame_id, cam.color_frame_id | str | frames used by depth and detections | needs measurement |
| cam.detections_use_depth_aligned_frame | bool | true only if detections come from D435i colour with aligned depth | needs measurement (Q D2) |
| extr.T_body_camera | 4x4 (m, rad) | camera-to-body transform | needs measurement (Q H4) |
| extr.T_body_camera_cov | 6x6 (m², rad²) | extrinsic uncertainty | needs measurement |
| extr.version | int | bumped on any calibration change | needs measurement |
| clock.camera_to_pose_offset_ns | ns | offset between camera and pose clock domains | needs measurement (Q S3) |
| clock.offset_uncertainty_ns | ns | | needs measurement |

### Layer 1: geometry
| Name | Unit | Meaning | Source |
|---|---|---|---|
| l1.depth_min_m, l1.depth_max_m | m | accepted depth range | needs measurement (D435i range at our altitude) |
| l1.bbox_inset_fraction | ratio | inset of bbox per side before sampling | ASSUMPTION: 0.15 |
| l1.min_bbox_size_px | px | reject tiny boxes | ASSUMPTION: 8 |
| l1.sample_stride_px | px | subsample stride to bound CPU | ASSUMPTION: 2 |
| l1.min_valid_pixels | count | below = DEPTH_TOO_FEW_PIXELS | ASSUMPTION: 50 |
| l1.min_valid_fraction | ratio | valid / sampled | ASSUMPTION: 0.2 |
| l1.cluster_gap_m | m | depth gap separating clusters | ASSUMPTION: 0.3 |
| l1.min_cluster_fraction | ratio | cluster must hold this share of valid pixels | ASSUMPTION: 0.3 |
| l1.ambiguity_ratio | ratio | 2nd/1st cluster size above this = ambiguous | ASSUMPTION: 0.6 |
| l1.background_fraction_max | ratio | share of pixels in far cluster above this = background dominant | ASSUMPTION: 0.7 |
| l1.depth_noise_model | coeffs | depth sigma as a function of depth (e.g. a·z²) | needs measurement |
| l1.ground_ref_method | enum | `none`, `bbox_bottom_ray`, `keypoints` (later) | ASSUMPTION: `none` (provisional only) |

### Layer 2: time alignment and transforms
| Name | Unit | Meaning | Source |
|---|---|---|---|
| l2.pose_buffer_duration_s | s | history kept | ASSUMPTION: 10 |
| l2.pose_buffer_max_samples | count | hard memory bound | ASSUMPTION: 2000 |
| l2.max_interp_gap_s | s | max gap between bracketing poses | needs measurement (depends on pose rate, Q S2) |
| l2.allow_extrapolation_s | s | extrapolation beyond newest pose | ASSUMPTION: 0 (never extrapolate) |
| l2.max_detection_age_s | s | detection older than this vs. now = stale | ASSUMPTION: 1.0 |
| l2.max_rgb_depth_skew_s | s | RGB vs depth stamp tolerance | needs measurement |
| l2.pose_mode | enum | `full_3d` or `planar_plus_attitude_altitude` | needs measurement (Q S1) |

### Layer 3: grid registration (all required, no default; Q O1)
| Name | Unit | Meaning | Source |
|---|---|---|---|
| grid.cell_size_m | m | cell edge length | needs measurement (rules; ASSUMPTION for tests only: 1.0) |
| grid.n_cells_x, grid.n_cells_y | count | grid extent | needs measurement |
| grid.origin_in_map_m | (x, y) m | official origin expressed in map frame | needs measurement |
| grid.yaw_map_to_grid_rad | rad | rotation | needs measurement |
| grid.axis_flip_x, grid.axis_flip_y | bool | axis inversion | needs measurement |
| grid.index_base | 0 or 1 | first index | needs measurement |
| grid.numbering | enum | `ij` (column, row), `row_major_label`, `alnum` (e.g. "C4") | needs measurement |
| grid.boundary_rule | enum | which cell owns a point exactly on an edge | ASSUMPTION: lower-left-closed (`floor`) |
| grid.registration_version | int | bumped on any change | (bookkeeping) |
| grid.registration_valid | bool | false = never publish CONFIRMED official cells | needs measurement |
| grid.boundary_margin_m | m | within this of an edge, apply boundary handling | ASSUMPTION: derived from covariance; floor 0.1 |

### Layer 4: tracking and confirmation (substep 3.1; replaces the 2.x placeholders `decorrelation_time_s`,
`min_viewpoint_baseline_m`, `confirm_min_independent_obs`, `confirm_min_cell_probability`, `confirm_max_sigma_m`,
`outlier_mahalanobis`, `unreliable_after_inconsistent_obs`)
| Name | Unit | Meaning | Source |
|---|---|---|---|
| l4.assoc_gate_mahalanobis | chi² | association gate | ASSUMPTION: 9.21 (2-DoF 99%) |
| l4.assoc_max_distance_m | m | hard cap on association distance | ASSUMPTION: 1.0 |
| l4.assoc_ambiguity_margin | chi² | rival within this of the best = ambiguous | ASSUMPTION: 2.0 |
| l4.track_id_hint_bonus | chi² | cost reduction for a matching detector id | ASSUMPTION: 1.0 |
| l4.merge_gate_mahalanobis | chi² | two tracks consistent below this | ASSUMPTION: 9.21 |
| l4.min_separate_survivor_distance_m | m | same-frame tracks closer than this are flagged | ASSUMPTION: 0.5 |
| l4.obs_model_sigma_m | m | isotropic per-observation model error | ASSUMPTION: 0.05 |
| l4.view_bin_deg | deg | bearing bin width | ASSUMPTION: 30 |
| l4.view_bin_max_gap_s | s | silence that starts a new bin (revisit) | ASSUMPTION: 5.0 |
| l4.ray_bias_sigma_m | m | common-mode error along the ray per bin | ASSUMPTION: 0.1 |
| l4.max_effective_obs_per_bin | count | in-bin averaging limit | ASSUMPTION: 3 |
| l4.provisional_min_obs | count | CANDIDATE -> PROVISIONAL | ASSUMPTION: 2 |
| l4.confirm_min_obs | count | frames to confirm | ASSUMPTION: 5 |
| l4.confirm_min_view_bins | count | view bins to confirm | ASSUMPTION: 1 |
| l4.confirm_min_span_s | s | time span to confirm | ASSUMPTION: 0.5 |
| l4.confirm_prob | prob | top-cell probability to confirm | ASSUMPTION: 0.9 |
| l4.cell_change_margin | prob | hysteresis margin | ASSUMPTION: 0.2 |
| l4.cell_change_updates | count | hysteresis consecutive updates | ASSUMPTION: 3 |
| l4.unreliable_nis | chi² | mean NIS over the window above this = contradiction | ASSUMPTION: 6.0 |
| l4.nis_window | count | NIS window | ASSUMPTION: 5 |
| l4.unreliable_ambiguous_streak | count | consecutive ambiguous frames -> UNRELIABLE | ASSUMPTION: 3 |
| l4.candidate_timeout_s | s | silence before dropping a never-confirmed track | ASSUMPTION: 2.0 |
| l4.track_history_max | count | observations kept per track | ASSUMPTION: 50 |
| l4.max_tracks | count | tracks kept | ASSUMPTION: 64 |

### Layer 5: corrections and output (updated in 3.2; values and evidence levels: `configs/frozen_synthetic_v1.json`)
| Name | Unit | Meaning | Source |
|---|---|---|---|
| l5.correction_source | enum | `none`, `history`, `frame_correction` (the old `pose_revision_stream`/`keyframe_updates` were never used) | needs measurement (Q S4) |
| l5.observation_store_max | count | bounded re-derivation store | ASSUMPTION: 2000 |
| l5.frame_correction_cov_inflation, l5.uncorrected_cov_inflation | factor | variance inflation for a frame jump / observations without a corrected pose | ASSUMPTION: 4.0 |
| l5.correction_history_max, l5.tag_history_max, l5.removed_tags_max | count | bounds | ASSUMPTION: 8, 8, 64 |
| l5.missing_correction_timeout_s | s | a pose_revision bump without an event for this long adds a "missing correction" marker | ASSUMPTION: 2.0 |
| l5.recompute_mode | enum | `full` (exact, default) or `affected_tracks` (approximate) | ASSUMPTION: full |
| l5.telemetry_snapshot_period_s, l5.telemetry_update_min_interval_s | s | snapshot cadence / minimum interval between updates of one tag | ASSUMPTION: 5, 1 |
| l5.telemetry_max_tags_per_snapshot, l5.telemetry_send_candidates, l5.telemetry_link_budget_bps, l5.telemetry_log_max | - | snapshot size, candidates on the radio, design link budget, diagnostic log bound | ASSUMPTION: 32, false, 20000 (Q H5), 4096 |
| l5.end_of_mission_commit_policy | enum | `confirmed_only`, `confirmed_plus_provisional_top1`, `expected_value` | unset (rules, Q O2); interim simulation choice in "Substep 3.3b" |
| l5.commit_min_obs | count | observations a track needs for the `expected_value` policy | ASSUMPTION: 3 |
| score.correct_tag_points / score.wrong_tag_penalty_assumed | points | 40 (rulebook p.19) / ASSUMED penalty for evaluation, unset | rulebook / unset (Q O2) |

---

## (d) Failure-reason codes

Severity: **REJECT** = observation discarded; **BLOCK** = processing continues but CONFIRMED is forbidden;
**DEGRADE** = kept with reduced quality / provisional; **EVENT** = informational, not a failure.

| Code | Severity | Spec Part 10 row | Required behaviour |
|---|---|---|---|
| DEPTH_NO_VALID | REJECT | No valid depth | reject measurement |
| DEPTH_TOO_FEW_PIXELS | DEGRADE | Too few valid depth pixels | mark unreliable |
| DEPTH_BACKGROUND_DOMINANT | REJECT | Background dominates bounding box | reject ambiguous estimate |
| DEPTH_AMBIGUOUS_CLUSTERS | REJECT | Multiple plausible depth clusters | do not pick one |
| DETECTION_STALE | REJECT | Detection timestamp stale | reject or mark stale |
| RGB_DEPTH_TIME_MISMATCH | REJECT | Depth and RGB timestamps mismatch | reject unless validated sync |
| POSE_UNAVAILABLE_AT_CAPTURE | BLOCK | Historical pose unavailable | do not confirm |
| EXTRINSIC_MISSING | BLOCK | Camera extrinsic missing | no confirmed map position |
| CAMERA_ORIENTATION_UNKNOWN | DEGRADE | Camera orientation unknown | mark unreliable |
| GRID_UNREGISTERED | BLOCK | Grid registration unavailable | no confirmed official cell |
| CELL_BOUNDARY_AMBIGUOUS | DEGRADE | Position near cell boundary | stay provisional |
| TRACK_DUPLICATE_MERGED | EVENT | Duplicate detection | update existing track |
| TRACK_ID_REASSOCIATED | EVENT | Detector tracking ID changes | spatial association |
| SLAM_CORRECTION_APPLIED | EVENT | SLAM correction | recalculate affected observations |
| TRACK_NOT_OBSERVED | EVENT | Survivor leaves camera view | keep last status |
| CALIBRATION_INVALID | BLOCK | Invalid calibration | prevent confirmed localisation |
| OUT_OF_GRID | DEGRADE | Position outside official grid | report out-of-grid |

Additional codes (not Part 10 rows, needed for explicit diagnostics):
BBOX_INVALID (REJECT), BBOX_OUTSIDE_IMAGE (REJECT), DEPTH_SCALE_INVALID (REJECT), INTRINSICS_MISSING (REJECT),
DEPTH_FRAME_MISMATCH (REJECT: bbox camera ≠ depth camera without calibrated mapping), CLOCK_DOMAIN_UNKNOWN (REJECT),
POSE_INVALID (BLOCK), POSE_GAP_TOO_LARGE (BLOCK), GROUND_REF_UNAVAILABLE (DEGRADE), OBSERVATION_OUTLIER (REJECT),
TRACK_INCONSISTENT (DEGRADE → UNRELIABLE), CORRECTION_SOURCE_UNAVAILABLE (DEGRADE: limitation logged once),
CONFIG_MISSING (BLOCK), CAPACITY_EXCEEDED (EVENT: bounded store evicted oldest).
Added in 3.1b-3.3a: FLOOR_UNKNOWN, EXTENDED_CLUSTER (Layer 1 warnings); POSE_NOT_CORRECTED, OBSERVATION_NOT_REDERIVABLE,
FRAME_CORRECTION_APPLIED (DEGRADE, Layer 5); CORRECTION_INVALID, CORRECTION_CONFLICT, CORRECTION_STALE (REJECT, Layer 5
events); NEAR_DUPLICATE_OBSERVATION (EVENT, Layer 4, cleared by a Layer 5 re-derivation). Severities: `types.FAILURE_SEVERITY`.

---

## (e) Tag messages

### Full Tag (onboard / ROS 2 / logs; spec 9.4)
`survivor_id, status, cell (i, j) and official label, cell_probs (top 4 cells), p_map_m (x, y, z),
cov_map_m2 (3x3), p_grid_m (x, y), sigma_horizontal_m, boundary_distance_m, out_of_grid, max_detection_confidence,
localisation_quality, observation_count, effective_independent_count, first_seen_ns, last_seen_ns, capture_stamp_ns
(latest contributing), update_stamp_ns, map_revision (pose_revision), registration_version, extrinsic_version,
reasons (list of FailureReason)`.

### Compact telemetry record (radio). Fixed layout, little-endian, 25 bytes
| Offset | Size | Field | Encoding |
|---|---|---|---|
| 0 | 1 | msg_type | u8: 0x01 = tag update, 0x02 = snapshot header |
| 1 | 1 | seq | u8, wraps; lets the GCS detect drops |
| 2 | 1 | survivor_id | u8 |
| 3 | 1 | status | u8: 0 CANDIDATE, 1 PROVISIONAL, 2 CONFIRMED, 3 UPDATED, 4 UNRELIABLE |
| 4 | 1 | flags | bit0 registration_valid, bit1 out_of_grid, bit2 boundary_ambiguous, bit3 pose_blocked, bit4 removed, bits5-7 reserved |
| 5 | 2 | cell_i | i16; 0x7FFF = no cell |
| 7 | 2 | cell_j | i16; 0x7FFF = no cell |
| 9 | 1 | cell_prob | u8, value/255 |
| 10 | 1 | det_conf | u8, value/255 (detector confidence) |
| 11 | 2 | sigma_cm | u16, horizontal 1-sigma in cm; 0xFFFF = unknown |
| 13 | 2 | grid_x_cm | i16, cm in grid frame; 0x7FFF = unknown |
| 15 | 2 | grid_y_cm | i16 |
| 17 | 4 | update_ms | u32, ms since library start (mission clock); wraps after ~49.7 days |
| 21 | 1 | map_rev | u8, low byte of pose_revision |
| 22 | 1 | reg_rev | u8, low byte of registration_version |
| 23 | 2 | crc16 | CRC-16/CCITT-FALSE over bytes 0..22 |

Total **25 bytes**. If the radio protocol (e.g. MAVLink) already provides CRC and sequence, those 3 bytes can be dropped
later; layout version stays explicit via msg_type.

### Snapshot header (msg_type 0x02). 12 bytes, followed by N tag records
`msg_type u8 | seq u8 | snapshot_id u16 | n_tags u8 | map_rev u8 | reg_rev u8 | flags u8 | update_ms u32` = 12 bytes,
then N × 25-byte records, then crc16 over the whole frame (+2). Size = 14 + 25·N bytes.

### Send rule
1. Event update: send one record whenever a tag's status, cell, or flags change (rate-limited to 1 per tag per second,
   ASSUMPTION).
2. Full snapshot: every `l5.telemetry_snapshot_period_s` (ASSUMPTION: 5 s) **and** immediately after any map_rev or
   reg_rev change. Snapshot contains all non-CANDIDATE tags (CANDIDATE only if `l5.telemetry_send_candidates`).
3. The GCS treats the latest snapshot as the truth, and event updates as deltas on top of it.
Budget check (ASSUMPTION link 20 kbit/s): snapshot of 16 tags = 414 B every 5 s ≈ 0.66 kbit/s, plus events. This is
far below budget before link framing overhead, which needs measurement once the radio is known.

---

## (f) Thin-slice order (simplest path to one correct tag; every layer minimal; unit + simulation tested only)

1. `types.py` + `config.py`: data types, FailureReason, TagStatus, ConfigParam with required/unset checks.
2. Layer 1 minimal: bbox validation → inset region → reject zero/NaN/inf/out-of-range → median depth → back-project
   the inset-region centre pixel with given intrinsics. Fail codes for no-valid / too-few. No clustering yet.
3. Layer 2 minimal: PoseBuffer with bounded size; exact or linear/slerp interpolation between bracketing poses;
   fail on no bracket. T_body_camera from config. Covariance pass-through omitted (flagged).
4. Layer 3 minimal: configured 2D rigid T_grid_map + cell size + `floor` + index base; GRID_UNREGISTERED and OUT_OF_GRID.
5. Layer 4 minimal: single track by nearest-distance gate; status CANDIDATE → PROVISIONAL → CONFIRMED by
   observation count + same cell (no probabilities yet).
6. Layer 5 minimal: full Tag build + compact record encode/decode round-trip. No corrections yet.
7. `harness.py`: synthetic scene (known survivor in grid cell, known poses, synthetic depth patch) → assert one
   CONFIRMED tag in the expected cell. Label: **simulation tested (synthetic)**.
Then deepen layer by layer per spec Phases 2-7 (clustering, covariance propagation, cell probabilities, decorrelation,
corrections).

---

## (g) Future ROS 2 wrapper boundary (proposed, not created)

Separate packages, so that the core stays ROS-free:
- `survivor_tagging_msgs`: SurvivorTag.msg, SurvivorTagArray.msg, RegistrationState.msg, (Detection msgs if the
  detection teammate has none; prefer `vision_msgs/Detection2DArray` if they already use it).
- `survivor_tagging_ros` (ament_python): node `survivor_tagging_node` wraps `harness`-style ingestion; node
  `survivor_telemetry_bridge` encodes compact records for the radio link.

Proposed topics (names to be aligned with teammates; RealSense names depend on driver version, needs measurement):
| Direction | Topic | Type |
|---|---|---|
| sub | `/detection/humans` (teammate's actual topic) | vision_msgs/Detection2DArray or teammate msg |
| sub | `/camera/camera/aligned_depth_to_color/image_raw` | sensor_msgs/Image |
| sub | `/camera/camera/aligned_depth_to_color/camera_info` | sensor_msgs/CameraInfo |
| sub | pose via tf2 (`map`→`base_link`) or `/slam/pose` | tf2 / geometry_msgs/PoseWithCovarianceStamped |
| sub | `/slam/correction` (only if backend exposes it) | backend-specific; adapter |
| pub | `/survivor_tagging/tags` | survivor_tagging_msgs/SurvivorTagArray |
| pub | `/survivor_tagging/markers` | visualization_msgs/MarkerArray |
| pub | `/survivor_tagging/registration_state` | survivor_tagging_msgs/RegistrationState |
| pub | `/diagnostics` | diagnostic_msgs/DiagnosticArray |
| pub | `/survivor_tagging/telemetry_frames` | std_msgs/UInt8MultiArray (to radio bridge) |

Static extrinsic `body`→`camera` published by the hardware owner via tf2 static transform, mirrored in config.

---

## Layer 1 as implemented (substeps 2.3 and 2.3b, `geometry.py`)

Status: unit tested and simulation tested on synthetic data only (box-shaped survivors).

`estimate_survivor_point(det, depth, intrinsics, Layer1Params, extractor=inset_box_region) -> Layer1Result` is a
pure function. Steps:
1. **Input checks**: equal clock domains (else CLOCK_DOMAIN_UNKNOWN); detection frame id = depth frame id, depth
   aligned to colour, image size = depth shape (else DEPTH_FRAME_MISMATCH); intrinsics valid and matching the depth
   size (else INTRINSICS_MISSING); depth scale finite and > 0 (else DEPTH_SCALE_INVALID); bbox finite, ordered,
   inside the image, at least `l1.min_bbox_size_px` (BBOX_INVALID / BBOX_OUTSIDE_IMAGE).
2. **Region**: bbox inset by `l1.bbox_inset_fraction` per side, sampled every `l1.sample_stride_px` as a strided
   view (no frame copy). Replaceable through the `RegionExtractor` interface (mask or keypoints later).
3. **Validity**: finite, > 0, inside [`l1.depth_min_m`, `l1.depth_max_m`]. No valid pixel: DEPTH_NO_VALID; fewer
   than `l1.min_valid_pixels` or `l1.min_valid_fraction`: DEPTH_TOO_FEW_PIXELS (no point returned).
4. **Density clustering (2.3b)**: sort depths; keep only samples with at least `l1.min_density_fraction` of the
   valid samples within +/- `l1.cluster_gap_m`/2 (1D DBSCAN core points), so a sloped floor or flying pixels cannot
   bridge person and wall; split the kept samples at gaps > `l1.cluster_gap_m`. No dense sample: AMBIGUOUS.
5. **Cluster selection** (`select_cluster` docstring has the exact rule): candidates >= `l1.min_cluster_fraction`;
   second/first > `l1.ambiguity_ratio`: AMBIGUOUS; else the largest D. If >= `l1.background_near_min_fraction` of the
   pixels are nearer than D and `l1.cluster_rule = spatial` (2.3b): nearer pixels forming a band along a box edge
   (>= `l1.occluder_edge_span_min` of that edge) = occluder, accept D; nearer pixels forming a strip with D on both
   sides in >= `l1.strip_rows_fraction_min` of their rows = thin person in front of background, choose the largest
   nearer cluster. Otherwise (or `fraction_only`, the 2.3 rule) D > `l1.background_fraction_max`: BACKGROUND_DOMINANT.
6. **Point** (`l1.point_method`): `median_pixel` = median depth of the cluster at the median (u, v) of the cluster's
   pixels; `cluster_centroid` (2.3b) = per-axis median of all back-projected cluster points. Then moved along the ray
   by `l1.body_depth_offset_m` (default 0.0; must be measured on real bodies, never set from the simulator).
7. **Covariance**: median_pixel: sigma_z^2 = noise(z)^2 + (1.4826 MAD_z)^2, sigma_u^2 = stride^2/12 + (1.4826
   MAD_u)^2/n (same for v), propagated by the back-projection Jacobian. cluster_centroid: Cov(cluster points) +
   noise(z)^2 r r^T. Box within `l1.truncation_margin_px` of the image border: warning BBOX_TRUNCATED (DEGRADE) and
   covariance x `l1.truncated_cov_scale`. depth_quality = valid_fraction x cluster share / (1 + 1.4826 MAD_z / gap).
8. **Ground reference**: interface only; "none" -> None + GROUND_REF_UNAVAILABLE warning (all methods for now).

Deviations / limits:
- RGB-depth timestamp skew (RGB_DEPTH_TIME_MISMATCH) is checked in Layer 2, not here.
- DEPTH_TOO_FEW_PIXELS returns no point (the table marks it DEGRADE).
- No lens undistortion: correct only for zero-distortion intrinsics (verify, Q H3).
- Both point methods measure the visible surface, so a standing body is about half its depth short of its centre
  (0.25 m for the simulator's boxes). `cluster_centroid` does not remove this; only the body offset does.
- Per-axis medians are pixel-weighted: a prone body's near end has more pixels, so prone estimates fall short
  (0.49 m horizontal in simulation). A percentile mid-range (p5..p95 centre) cut this to 0.29 m in simulation but
  worsened the p90 error in tilt_45 and low_occluder; not adopted, candidate for later.
- Density clustering separates floor from the person in steep views (tilt_45), which then become AMBIGUOUS
  (39 % in simulation) instead of being merged: a safe failure.
- Added parameters (all ASSUMPTION): `l1.background_near_min_fraction` 0.05 (2.3); `l1.point_method` median_pixel,
  `l1.cluster_rule` spatial, `l1.occluder_edge_span_min` 0.6, `l1.strip_rows_fraction_min` 0.6,
  `l1.min_density_fraction` 0.05, `l1.truncation_margin_px` 1.0, `l1.truncated_cov_scale` 4.0 (2.3b).

Layer 1 follow-up (substep 2.4): `l1.point_method = cluster_area_weighted` (depth^2-weighted per-axis median) cut the
prone horizontal error from 0.487 to 0.322 m in simulation but worsened the tilt_45 p90 by 0.418 m (rule: no other
scenario may worsen p90 by more than 0.02 m), so the default stays `median_pixel`. The option remains available.

---

## Layer 2 as implemented (substep 2.4, `transforms.py`)

Status: unit tested and simulation tested on synthetic data only.

- Utilities: quaternion (w, x, y, z) normalise/multiply/to-from matrix/slerp, Exp map, 4x4 compose/invert/apply.
- `extrinsic_from_config`: unset `extr.T_body_camera` or `extr.version` -> EXTRINSIC_MISSING; not a proper rigid
  transform, or covariance not symmetric/non-negative -> CALIBRATION_INVALID.
- `PoseBuffer`: sorted by stamp, out-of-order inserts, same-stamp replacement only by a not-older `pose_revision`,
  bounded by `l2.pose_buffer_duration_s` (relative to the newest stamp) and `l2.pose_buffer_max_samples`; invalid
  poses -> POSE_INVALID; a second clock domain -> CLOCK_DOMAIN_UNKNOWN.
- `lookup(t)`: exact stamp or the two bracketing poses (gap <= `l2.max_interp_gap_s`, else POSE_GAP_TOO_LARGE), linear
  position, slerp orientation, linear covariance. Before the oldest or after the newest pose (beyond
  `l2.allow_extrapolation_s`, default 0) -> POSE_UNAVAILABLE_AT_CAPTURE. The latest pose is never substituted.
  Bracketing poses with different revisions add the warning SLAM_CORRECTION_APPLIED.
- Pending queue: an observation whose capture time is newer than every received pose waits until a newer pose
  arrives, at most `l2.max_pose_wait_s` after its own arrival (then POSE_UNAVAILABLE_AT_CAPTURE); bounded by
  `l2.pending_queue_max` (overflow evicts the oldest with CAPACITY_EXCEEDED); pending items at finalize fail explicitly.
- Timestamp checks, in order: `l2.max_rgb_depth_skew_s` unset -> CONFIG_MISSING; detection/depth clock domains differ
  -> CLOCK_DOMAIN_UNKNOWN; |RGB - depth stamp| > skew -> RGB_DEPTH_TIME_MISMATCH; image vs pose clock domain differs
  -> add `clock.camera_to_pose_offset_ns`, or CLOCK_DOMAIN_UNKNOWN if unset; arrival - capture >
  `l2.max_detection_age_s` -> DETECTION_STALE.
- `camera_to_map`: `P_map = R_mb (R_bc p_c + t_bc) + t_mb`; covariance = measurement + pose (6x6, map-frame
  perturbation) + extrinsic (6x6, body-frame perturbation) through analytic Jacobians, verified against finite
  differences in `tests/test_transforms.py`.

Deviations / limits:
- ASSUMPTION: arrival times are in the pose clock domain (needed for staleness and the pending deadline).
- `l2.pose_mode` is not used yet: the pose is taken as given (full 3D). A planar SLAM pose must be completed with
  altitude/attitude upstream (Q S1).
- `clock.offset_uncertainty_ns` is not propagated into the covariance yet.
- A pose without covariance contributes no pose uncertainty (the simulator provides none).

## Layer 3 as implemented (substep 2.4, `grid.py`)

Status: unit tested and simulation tested on synthetic data only.

- `GridRegistration` (frozen, versioned; `updated()` returns version + 1) from `grid.*`; map -> grid definition and a
  worked example in conventions.md. `map_to_grid` / `grid_to_map` are exact inverses.
- Cells: `floor` rule, `grid.index_base`, OUT_OF_GRID from `grid.n_cells_x/y` (unknown extent -> not decided);
  labels `ij` ("i,j") and `row_major_label` (row * n_cells_x + col + base); `alnum` raises a ConfigError (O1).
- `boundary_distance_m`: distance to the nearest cell edge.
- `cell_probabilities`: grid-frame Gaussian; exact erf product for a diagonal covariance; correlated covariance =
  exact conditional normal CDF in y, 24-node Gauss-Legendre in x per cell; cells within mean +/- 5 sigma; top
  `grid.max_candidate_cells` returned with the out-of-grid mass (sum <= 1).
- Unregistered (`grid.registration_valid` false or any registration parameter unset): a provisional registration
  (origin 0, yaw 0, no flips, base 0, "ij") gives a diagnostic cell with warning GRID_UNREGISTERED (BLOCK); the
  Layer123 pipeline then emits PROVISIONAL tags only.
- `fit_registration_from_points`: closed-form 2D rigid least squares with known flips; returns registration and RMS.

Deviations / limits:
- Probability computation uses Python `math.erf` loops: 0.44 ms median per observation on the dev laptop.
- `grid.boundary_margin_m` is not used yet (boundary decisions belong to Layer 4 via the cell probabilities).

## Pipeline (`pipeline.py`) and simulator glue (`sim_pipelines.py`)
`Layer123Pipeline(cfg)` chains Layers 1-3 with the baseline's one-tag-per-tracking-id policy (Layer 4 replaces it).
`sim_pipelines.scenario_config` builds the Config from a scenario through the simulator's public API only (camera
mount, `Registration.map_from_grid`), with SIMULATION values for `l2.max_interp_gap_s` (0.05) and
`l2.max_rgb_depth_skew_s` (0.005).

---

## Layer 4 as implemented (substep 3.1, `tracking.py`)

Status: unit tested and simulation tested on synthetic data only. Spec: Part 8 "LAYER 4: TEMPORAL CONSISTENCY AND
UNCERTAINTY" (8.1-8.8) and 12.4 "Layer 4 tests - Temporal consistency".

Per frame batch (observations with one capture stamp, already through Layers 1-3):
1. **Measurement model**: map-plane position; covariance = Layer 1-3 covariance + `l4.obs_model_sigma_m`^2 I (where
   on the body the point lands varies frame to frame; without it the thin, ray-aligned Layer 1 covariances split one
   person into several tracks in simulation) + per view bin a common-mode bias `l4.ray_bias_sigma_m`^2 along the ray.
2. **Association**: innovation `S = P_track + R + B_ray`, gate `l4.assoc_gate_mahalanobis` and
   `l4.assoc_max_distance_m`, at most one observation per track per frame, optimal assignment by the Hungarian
   algorithm (Kuhn-Munkres with potentials, own implementation, checked against brute force). A matching detector
   `tracking_id` lowers the cost by `l4.track_id_hint_bonus` only. If an unassigned rival track's cost is within
   `l4.assoc_ambiguity_margin` of the best, the observation is ASSOCIATION_AMBIGUOUS and not fused. Ungated
   observations start new tracks.
3. **Fusion**: view bins = bearing (survivor -> camera) relative to the track's first bearing, `l4.view_bin_deg`
   wide; silence > `l4.view_bin_max_gap_s` in a bin starts a new visit. Within a bin: information-weighted mean, noise
   reduced by at most `l4.max_effective_obs_per_bin` observations, plus the ray bias. Bins are fused by information
   weighting. Robustness = the association gate (a wild observation never enters an existing track; its own
   candidate cannot confirm and times out). Chosen over Huber weighting: simpler, and fusion stays a plain average.
4. **Cell probabilities** from the fused Gaussian via `grid.cell_probabilities` (out-of-grid mass included). Pose
   covariance enters through Layer 2.
5. **Status** (table below), hysteresis, UNRELIABLE: mean NIS of the last `l4.nis_window` associated observations >
   `l4.unreliable_nis` (TRACK_INCONSISTENT), or `l4.unreliable_ambiguous_streak` consecutive ambiguous frames.
6. **Merges**: two tracks never observed in the same frame whose estimates are consistent
   (`l4.merge_gate_mahalanobis`) are merged; the higher status survives (ties: older); event TRACK_DUPLICATE_MERGED
   and history on both. Tracks in one frame closer than `l4.min_separate_survivor_distance_m` are flagged
   (ASSOCIATION_AMBIGUOUS, history "too_close_same_frame"), never merged. Since 3.3a two observations of ONE frame
   closer than that distance that fit the same track are one body seen twice: the better one is kept and the other is
   dropped (NEAR_DUPLICATE_OBSERVATION), see "Substep 3.3a" below. A new track created from an observation
   whose detector id belongs to an existing track logs "split_from" / "split_to".
7. **False positives**: never-confirmed tracks with fewer than `l4.confirm_min_obs` frames are dropped after
   `l4.candidate_timeout_s` without observations (bounded `dropped` log of 100).
8. **Memory**: at most `l4.max_tracks` tracks (evict lowest status rank, then fewest frames, then oldest last
   observation; event CAPACITY_EXCEEDED) and `l4.track_history_max` observation references per track (evict the
   oldest of the most populated view bin; warning CAPACITY_EXCEEDED). References keep `pose_revision` for Layer 5.
9. **Output**: `Tag` (status, confirmed or top cell, candidate cells with probabilities, sigma, revision, reasons,
   confirmed time, cell changes); `final_tags(policy, threshold)` with `confirmed_only`,
   `confirmed_plus_provisional_top1` or `expected_value` (3.1c). No default policy is claimed: it depends on O2; the
   interim simulation choice is in "Substep 3.3b".

### Status transitions
| From \ To | CANDIDATE | PROVISIONAL | CONFIRMED | UPDATED | UNRELIABLE |
|---|---|---|---|---|---|
| CANDIDATE | - | >= `provisional_min_obs` frames | illegal | illegal | contradiction / ambiguity |
| PROVISIONAL | illegal | - | all confirm criteria | illegal | contradiction / ambiguity |
| CONFIRMED | illegal | a BLOCK warning appears | - | new cell beats old by `cell_change_margin` for `cell_change_updates` updates | contradiction / ambiguity |
| UPDATED | illegal | a BLOCK warning appears | illegal | another hysteresis change | contradiction / ambiguity |
| UNRELIABLE | illegal | condition cleared | illegal | illegal | - |

Confirm criteria (all): >= `confirm_min_obs` frames, >= `confirm_min_view_bins` view bins, span >=
`confirm_min_span_s`, top in-grid cell probability >= `confirm_prob`, no BLOCK-severity warning (an invalid
registration always blocks). Every legal and illegal pair is tested (`test_every_legal_and_illegal_transition`).

Deviations / limits:
- Survivors are modelled as static; a moving target starts new tracks instead of dragging one.
- View bins have fixed edges relative to the first view: a viewpoint drifting across an edge counts as a second bin.
- `l4.confirm_min_view_bins` defaults to 1, because most views in a mission come from one side; the intermittent
  false positive is suppressed by `confirm_min_obs` and the timeout, not by view diversity.
- In simulation the cell probabilities are over-confident when the Layer 1 body-depth offset is wrong (0.25 m bias vs
  ray-bias sigma 0.1 m): mean predicted 0.85 vs 0.63 observed at offset 0.0; 0.83 vs 0.74 at offset 0.2.
- Loop-closure errors baked into earlier observations are not corrected (Layer 5); they appear as a duplicate track.
- Layer 4 costs about 0.6 ms per observation on the dev laptop (pure Python).

---

## Substep 3.1b: pipeline order, floor-aware Layer 1, footprint refinement

Status: unit tested and simulation tested on synthetic data only.

**Pipeline order** (`pipeline.Layer123Pipeline._resolve`): timestamp checks -> pending queue until a pose newer than
the capture time has arrived -> capture-time pose lookup (`Layer2.lookup`; missing pose fails exactly as before, the
latest pose is never substituted) -> floor plane -> Layer 1 -> camera-to-map (`Layer2.apply`) -> footprint refinement
-> Layer 3. The pending queue now holds (observation, detection, depth frame, intrinsics); at most
`l2.pending_queue_max` depth-frame references are kept.

**Floor plane** (`transforms.floor_plane_in_camera`, `geometry.FloorPlane`): from the capture-time pose,
`T_body_camera` and `map.floor_z_m` (venue config; unset -> FLOOR_UNKNOWN warning and no removal). Pixels with height
above the floor < `l1.floor_margin_m` + `l1.floor_margin_pose_sigma_k` x sigma_z (pose covariance) are removed before
clustering. The sensor-validity checks (`min_valid_fraction`) use the pixels before removal; `min_valid_pixels` is
re-checked after removal. Attitude uncertainty does not widen the margin yet.

**Footprint refinement** (`footprint.py`): Layer 1 keeps up to `l1b.max_points` cluster points (deterministic,
row-major subsample, camera frame, before the body offset; only when the refinement is enabled). After the camera-to-map
transform they are projected onto the floor plane; principal axis from the 2x2 covariance; robust extents from the
`l1b.extent_percentile_lo/hi` percentiles. Extent along the axis >= `l1b.extended_min_m` -> EXTENDED: centre = midpoint
of the robust extent (or the depth^2-weighted median, `l1b.method = area_weighted`) along the axis, median across it;
body offset only across the axis: `+ offset * (v . across) * across` with v the unit horizontal view direction;
covariance `+ (l1b.extended_axis_sigma_fraction x extent)^2` along the axis; warning EXTENDED_CLUSTER. Non-extended
observations are left bit-identical. Layer 4 reads `obs.cov_map_m2` in `Layer4._make_item`, so it uses the inflated
covariance (checked in code).

**Decision results** (rules fixed before running; each had to pass at body offsets 0.0 and 0.2):
- Floor removal: FAIL, default off. tilt_45 acceptance rose 0.61 -> 1.00 and its p90 fell 0.522 -> 0.254 m, but at
  offset 0.0 the wrong-cell rate rose in prone (+0.059) and tilt_45 (+0.063), and the prone p90 worsened by 0.025 m
  (0.028 m at offset 0.2): with the floor removed, the low prone body loses its lowest band.
- Footprint refinement: FAIL, default off. Prone horizontal median 0.530 -> 0.329 m (offset 0.0) and 0.370 -> 0.333 m
  (offset 0.2), not below 0.30 m; prone correct CONFIRMED stayed 0. Likely cause (not verified): the 15 % bbox inset
  trims the far end of a lying body, so the visible extent is short on the far side. In a generated geometric check
  without inset the midpoint error is -0.076 m vs -0.324 m for the median.
- Both remain options (`l1.floor_removal`, `l1b.footprint_refinement`).

**Diagnostics** (`diagnostics.py`, part A): survivors not CONFIRMED in the true cell are almost never a
scenario-length artifact (1 of 65 at offset 0.0, 1 of 55 at offset 0.2). The dominant reason is a top-cell probability
below 0.9 (45/65, 52/55), and 40/45 and 49/52 of those have the fused estimate within 0.15 m of a cell edge: from one
viewpoint the ray-bias floor (sigma about 0.1 m) cannot support p >= 0.9 that close to an edge. At offset 0.0 another
16 are confirmed in the wrong cell (the uncorrected 0.25 m front-surface bias).

---

## Substep 3.1c: honest cell probabilities, expected-value commit policy

Status: unit tested and simulation tested on synthetic data only. All calibration must be redone on recorded data in
Step 5.

**Bias-uncertainty model.** Layer 4's per-view-bin common-mode term along the camera ray uses
`ray_sigma = sqrt(l4.ray_bias_sigma_m^2 + l1.body_depth_offset_sigma_m^2)` (`Layer4Params.ray_sigma_m`), both in the
association innovation and in the bin fusion. Who does what (no double counting):
- `l1.body_depth_offset_m`: the correction applied in Layer 1 (0.0 until measured).
- `l1.body_depth_offset_sigma_m` (ASSUMPTION 0.10 m): uncertainty of that correction only (true front-surface bias
  minus the configured offset). Shrink it only when the offset is measured on real bodies (Step 5).
- `l4.ray_bias_sigma_m` (0.10 m): every other common-mode ray error (pose, extrinsic, depth scale, surface shape).
- `l4.obs_model_sigma_m` (0.05 m): isotropic frame-to-frame variation, averaged within a bin up to
  `l4.max_effective_obs_per_bin`.
With `l1.body_depth_offset_sigma_m = 0` the behaviour is exactly the pre-3.1c one (tested). The new default 0.10 m
changes the Layer 4 default output (effective sigma 0.141 m instead of 0.10 m); `experiments.py` pins the 3.1b values.

**Calibration method.** `calibration.py` (library): equal-width reliability table, ECE, over-confidence flags.
`experiments_calib.py` (harness): per-update samples of matched tracks; 60 settings (floor {0.05..0.20} x offset sigma
{0..0.20} x obs model {0.03, 0.05, 0.08}) x configured offset {0, 0.1, 0.2, 0.3} x true model {fixed, uniform depth
0.2-0.5 m}; Layers 1-3 cached, Layer 4 replayed; 796 s on the dev laptop (11 processes).

**Decision (pre-registered rule; FAIL).** No setting was ACCEPTABLE on seeds 0-2 (ECE(p >= 0.5) <= 0.05 and no bin with
>= 20 samples over-confident by > 0.10, in all four combinations with |offset - true mean bias| <= 0.1 m). Best by
summed shortfall: floor 0.05, offset sigma 0.10, obs model 0.08 (effective 0.112 m); seeds 0-2: ECE 0.045 / 0.057 /
0.081 / 0.092, worst over-confidence +0.127 / +0.123 / -0.019 / -0.018; held-out seeds 3-7: ECE 0.040 / 0.061 / 0.070 /
0.059, worst over-confidence +0.149 / +0.209 / +0.027 / -0.026: FAIL. Miscalibration goes both ways (over-confident
around p 0.6-0.9, under-confident above 0.9 when the offset is close), so no single isotropic or ray sigma fixes it;
a wrong offset (0.0 configured, true 0.25) gives +0.45 over-confidence that a 0.10 m offset sigma cannot cover.
Defaults NOT changed by the sweep (floor 0.10, offset sigma 0.10 as specified, obs model 0.05).

**Final-tag policy `expected_value`** (`Layer4.final_tag_set`): every track that is not UNRELIABLE with at least
`l5.commit_min_obs` (ASSUMPTION 3) observations is committed to its top cell if p_top >= P / (40 + P), P =
`score.wrong_tag_penalty_assumed` (unset -> 0.5 with a "penalty unknown" note; 40 = `score.correct_tag_points`, rulebook
p.19). For P < 40, 0.5 is conservative; for P > 40 it over-commits. Simulation: expected_value(0.5) scores at least as
well as confirmed_only for P <= 40 in all four offset/model cases, but the formula threshold was the best of the tested
thresholds only at P = 0 and P = 80 (and P = 60 in two cases); for P = 20-40 a higher threshold (0.5-0.67) was better,
consistent with the measured over-confidence. confirm_prob stays 0.9 (pre-set rule: 0.7 scores more whenever the
offset is right, but less at offset 0.0 and P = 80).

---

## Layer 5 as implemented (substep 3.2, `corrections.py`, `output.py`, `pipeline.Layer5Pipeline`)

Status: unit tested and simulation tested on synthetic data only. Spec: Part 9 and 12.5. Details and numbers:
`docs/reports/3.2.md`.

- **Pose-independent store.** Every accepted observation is stored (`StoredObservation`: camera-frame point and
  covariance, capture stamp, the base pose lookup used, the Layer 4 batch id). Bounded by `l5.observation_store_max`.
- **Corrections.** One interface, `PoseCorrection`, with two sources chosen by `l5.correction_source`: `history`
  (corrected historical poses for a time range) and `frame_correction` (one map-frame jump applied over the window
  [previous correction stamp, event stamp)). Events are folded in revision order over the stored base poses, so arrival
  order does not matter. Observations without a corrected pose get POSE_NOT_CORRECTED and inflated covariance; a bracket
  of mixed revisions counts as uncorrected (`PoseLookup.min_revision`). A pose_revision bump with no event for
  `l5.missing_correction_timeout_s` flags older observations; the real event replaces the marker.
- **Recompute = re-derive, never patch.** A correction or a registration update re-derives Layer 2 (effective
  capture-time pose) and Layer 3 (current registration) for every stored observation and replays Layer 4 from scratch
  in the ORIGINAL batches (replaying per capture stamp would not be exact: Layer 4 timeouts run once per update).
  Result equals an offline run on the corrected poses (R1: 0.0 m). Observations evicted from the store but still held by
  a track are replayed with their old position and inflated covariance (OBSERVATION_NOT_REDERIVABLE).
- **Survivor identity.** `TagBook` assigns a survivor id when a track is first published (status above CANDIDATE),
  never reuses it, and after a rebuild matches tracks by shared observation ids. A merge announces the losing id once
  as REMOVED. A confirmed-cell change publishes UPDATED with tag revision + 1. Ids and revisions are sent as their low
  byte on the radio (u8 fields): more than 256 published survivors would collide.
- **Live output.** `output.py` keeps the 25-byte record and 12-byte snapshot header unchanged; `PublishedTag` and
  `TelemetryScheduler` (snapshot every `l5.telemetry_snapshot_period_s` and on map/registration revision changes,
  per-tag rate limit) were appended. Simulation: mean 243 bit/s, worst steady case 2.5 kbit/s against the ASSUMED 20 kbit/s.
- **Deviations from the original plan:** the plan put the store in `output.py` and listed `pose_revision_stream` /
  `keyframe_updates` as sources; both were replaced as above. No persistence (a restart starts clean, tested).
  `l5.recompute_mode = full` is exact but slow (R3 FAIL below); `affected_tracks` is an approximation and stays optional.

## Substep 3.3a: near-duplicate observations (Layer 4) and honest pose covariance (simulator)

Status: unit tested and simulation tested only. Report: `docs/reports/3.3a.md`.
- Two observations of one frame closer than `l4.min_separate_survivor_distance_m` that fit the same track (or no
  track) are one body seen twice: the higher-confidence one is kept, the other gets NEAR_DUPLICATE_OBSERVATION (EVENT)
  and does not count as ambiguity. `_merge` may merge two tracks of one body whose shared frames are all near-duplicates,
  and runs before the proximity flag. An unambiguous assignment resets the ambiguity streak of every track it gates with
  (this removes the 3.2 deadlock where two tracks of one body made every later observation ambiguous).
- Setting the distance to 0 switches the drop off (the 3.2 behaviour; 3.3b verified this against the saved 3.3a "before" rows).
- Simulator: `double_detection` scenario (`DetectorSim.dup_prob`), and live/corrected poses after a closure now carry a
  covariance that matches their drawn error (mean normalised error squared 1.86-2.19, expected 2.0).
- Accepted limitation: a real second person within 0.5 m whose box fits the same track loses that frame's observation;
  a detector false positive within 0.5 m of a real person is dropped (this caused A3 FAIL; quantified in 3.3b R-A).

## Substep 3.3b: end-to-end evaluation, interim final-tag policy, parameter freeze

Status: simulation tested on synthetic data only (5 seeds; body offsets 0.0 and 0.2 m; true body model "fixed" unless
stated). Report: `docs/reports/3.3b.md`; full tables `docs/reports/3.3b_eval.txt`; raw rows `3.3b_cache_*.json`.
Scoring in all points tables is ASSUMED (O2): +40 per correct committed tag, -P per other committed tag.

- **R-A (A3 quantification, no decision).** Pooled over revisit_after_gap, opposite_views_boundary and
  thin_person_in_front_of_wall (30 runs): net change at P = 40 is **-4.00 points per run** (13.33 -> 9.33), at P = 0
  0.00. FLAGGED (threshold -1.0). It is 3 of 30 runs: a detector false positive near the person used to start a second
  track that survived as a tag; the person is now found but tagged in the neighbouring cell (front-surface bias, offset
  0.0 uncorrected). The change was accepted before the run; this is its cost.
- **R-B (merge distance).** Family `two_close_survivors` (8 pair types, centres 0.30-0.60 m apart, adjacent and same
  cell, standing/sitting). Missed rate at min_sep 0 / 0.3 / 0.4 / 0.5: 0.013 / 0 / 0 / 0 (offset 0.0), 0 at all values
  (offset 0.2). Largest value within +0.02: **0.5 = current default, unchanged**. The test barely discriminates (one miss
  in 80 survivors per cell); it does show that the 0.5 m drop did not lose real neighbours 0.30 m apart in simulation.
  Pairs closer than 0.30 m were not tested (the boxes would overlap).
- **R-C (INTERIM final-tag policy, per-P maximin).** Layers 1-5, 34 scenarios x 5 seeds, the four 3.1c calibration
  combinations, P in {0,20,40,60,80}. Worst-case points per run: confirmed_only 30.6, expected_value 0.5: 31.5, 0.6:
  35.1, **0.667: 36.9 (chosen)**. The maximin is dominated by P = 80; at P = 0 it gives up 3-5 points per run against
  threshold 0.5. Not applied as a default: `l5.end_of_mission_commit_policy` stays unset until O2 is answered.
- **Calibration per scenario (A).** At offset 0.2 (fixed) the 0.6-0.7 probability bin is over-confident (predicted 0.645,
  observed 0.511) almost only because of cell_boundary (80 of 368 samples, 0 correct; 56 % of the over-confidence mass);
  without it the bin is calibrated (0.648 vs 0.653) and pooled ECE falls 0.073 -> 0.052. At offset 0.0 the bin is
  calibrated pooled (0.649 vs 0.640); the large errors are in the 0.8-1.0 bins (uncorrected 0.25 m bias, e.g.
  cell_boundary ECE 0.87, latency_fast_yaw 0.48). Calibration on recorded data (Step 5) is still required.
- **Ablation (pooled, 170 runs per row, points at P = 40).** naive -32.0 (it commits every tag, including 0.81
  duplicates per run), Layer 1 -29.4 / -23.3 (offset 0.0 / 0.2), Layers 1-3 -31.3 / -24.2, Layers 1-4 25.9 / 37.7,
  Layers 1-5 28.5 / 40.9 under the interim policy. At P = 0 the naive pipelines score as much (41-45) as Layers 1-4
  (36-42) because they commit everything. Layers 1-5 vs 1-4 differ only in the correction scenarios (loop_closure
  duplicates 1.00 -> 0.00).
- **Freeze.** `configs/frozen_synthetic_v1.json` lists all 116 registry parameters (81 with values, 35 unset) with
  source and evidence level, the simulation-only constants, the interim choice and the unfrozen unknowns.

## Decision rules and results, substeps 3.1b to 3.3b (all simulation evidence)

| step | rule (as pre-registered) | result | consequence |
|---|---|---|---|
| 3.1b | floor removal default on iff tilt_45 acceptance rises, wrong-cell rate does not rise, p90 not worse than 0.02 m, at both offsets | FAIL (prone wrong-cell +0.059, p90 +0.025 m) | `l1.floor_removal` OFF |
| 3.1b | footprint refinement default on iff prone h-median < 0.30 m, prone correct CONFIRMED > 0, p90 not worse | FAIL (0.329 m, 0 confirmed) | `l1b.footprint_refinement` OFF |
| 3.1c | a (floor, offset sigma, obs model) setting with ECE(p >= 0.5) <= 0.05 and no bin >= 20 samples over-confident by > 0.10 in all rule combinations | FAIL (no setting; best held-out ECE 0.04-0.07, over-confidence up to +0.21) | defaults unchanged, redo on recorded data |
| 3.1c | change `l4.confirm_prob` iff an alternative scores more at every P > 0, both offsets, both models | keep 0.9 | unchanged |
| 3.2 R1 | tags after a correction equal an offline run on the corrected poses | PASS (35/35, 0.0 m) | exact recompute |
| 3.2 R2 | loop_closure duplicates <= 0.1/run and pre-closure error <= post + 0.03 m | PASS (0.00; 0.253 vs 0.281) | - |
| 3.2 R3 | full recompute at max store <= 250 ms | FAIL (12.8 s; incremental 10.3 s whole history, 0.52 s for a 4 s window) | `full` stays default; speed-up open |
| 3.3a A1 | double_detection >= 0.8 correct; real-neighbour scenarios within +-0.02 | PASS (1.00; worst 0.000) | - |
| 3.3a A2 | sim pose NIS in [1, 3]; repeated_corrections duplicates <= 0.2 | PASS (1.86-2.19; 0.0) | - |
| 3.3a A3 | every number outside the targeted scenarios unchanged | FAIL (46 of 1040 numbers, 3 scenarios) | accepted by the user; costed in 3.3b R-A |
| 3.3b R-A | flag if pooled net change at P = 40 below -1.0 point/run | **FLAGGED: -4.00** | reported, no code change |
| 3.3b R-B | largest min_sep with missed rate <= off value + 0.02 at both offsets | 0.5 | default unchanged |
| 3.3b R-C | candidate with the best worst-case points over 4 combos x 5 penalties | expected_value 0.667 (36.9) | INTERIM only |

## Known gaps (at the end of step 3)
- Nothing here is validated on recorded data, Gazebo or hardware. Probabilities are over-confident when the body-depth
  offset is wrong; the offset stays 0.0 until measured.
- R3: a full recompute after a loop closure is not affordable (seconds, laptop, simulation).
- Layer 4 assumes static survivors; moving survivors would start new tracks.
- The SLAM correction interface, pose type, grid definition, penalty rule and survivor spacing are open questions
  (`teammate_questions.md`).
