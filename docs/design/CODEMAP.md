# CODEMAP: `src/survivor_tagging/`

Names verified against the code on 2026-11-10 (substep 4.1b; docs/design/*.md are refreshed at the end of Step 4, task 4.2). "Library" = runs on the drone: numpy + stdlib only and must NOT import
`sim`, `scenarios`, `baseline`, `harness`, `sim_pipelines`, `diagnostics` or `experiments*` (checked by
`tests/test_pipeline.py` independence test for types, config, geometry, transforms, grid, pipeline, output, corrections,
tracking, footprint, calibration; it also bans `experiments_calib` and `experiments_layer5`). "Harness" = test world and evaluation, may import anything.

## Library
**`types.py`**: data types, status enums, failure codes (architecture.md b, d). Imports nothing from the package.
- `Severity`, `TagStatus` (IntEnum), `FailureReason`, `FAILURE_SEVERITY`; frozen dataclasses `Detection`,
  `CameraIntrinsics`, `DepthMeasurement`, `PoseSample` (+ `covariance_assumed`, 4.1b), `PoseCorrection`, `Tag`, `TagResult`;
  mutable `SurvivorObservation` (`cell_probs` is lazy since 3.4), `SurvivorTrack`.
- New codes since 3.3b: FRAME_MISMATCH, TIMESTAMP_INVALID, IMAGE_FORMAT_INVALID (4.1a, REJECT), POSE_COV_ASSUMED (4.1b, DEGRADE).
- Invariant: every failure is a `FailureReason`; timestamps are int ns with a clock-domain string.

**`config.py`**: parameter registry and JSON loader (architecture.md c). Imports `types`.
- `ParamSource`, `ConfigError`, `ConfigParam`, `Config` (`value(name)`, `set`, `params`, `is_required`,
  `missing_required(stage)`, `is_stage_ready`, `stage_failure`, `apply_overrides(overrides, reference)`),
  `coerce_value`, `default_config()`, `load_config(path, base=None)`, `STAGES`, `STAGE_DEPENDENCIES`.
- Invariants: hardware/competition values ship unset (`required`, None); tuning defaults are `ASSUMPTION`; rulebook
  values only `grid.cell_size_m` and `score.correct_tag_points`; overrides validated all-or-nothing.

**`geometry.py`** (Layer 1): robust survivor point from bbox + depth. Imports `config`, `types`.
- `Layer1Params.from_config`, `Layer1Result`, `FloorPlane`, `estimate_survivor_point(det, depth, intr, params,
  extractor=inset_box_region, floor=None) -> Layer1Result`; helpers `subsample_points`, `validate_inputs`,
  `validate_bbox`, `bbox_truncated`, `inset_box_region`, `region_depths_m`, `dense_mask`, `cluster_sorted_depths`,
  `edge_span`, `strip_fraction`, `select_cluster`, `robust_depth`, `back_project`, `apply_body_offset`,
  `ground_reference`, `weighted_median`, `depth_noise_sigma`, `measurement_covariance`.
- Density clustering, spatial occluder/strip rule, point methods median_pixel / cluster_centroid /
  cluster_area_weighted, BBOX_TRUNCATED. Invariant: failure -> reason code, never a fabricated point.

**`transforms.py`** (Layer 2): pose buffer, capture-time interpolation, pending queue, covariance. Imports `config`,
`geometry`, `types`.
- Quaternion/transform helpers (`quat_*`, `make_transform`, `invert_transform`, `compose`, `apply_transform`,
  `is_rigid`, `pose_matrix`); `Extrinsic`, `extrinsic_from_config`; `Layer2Params.from_config`; `PoseLookup`;
  `interpolate_pose`; `PoseBuffer` (`add`, `lookup(t_ns, max_gap_s, allow_extrapolation_s=0.0)`); `PendingItem`,
  `PendingQueue` (`push`, `pop_ready`, `pop_expired`, `drain`); `capture_in_pose_clock`, `check_timestamps`,
  `camera_to_map`, `floor_plane_in_camera`; `Layer2` (`add_pose`, `check`, `waiting_for_pose`, `lookup`, `apply`,
  `transform`).
- Invariants: pose at capture time only, never the latest; no extrapolation by default; different clock domains
  need `clock.camera_to_pose_offset_ns`; bounded buffer and queue.

**`grid.py`** (Layer 3): map -> official grid, cell probabilities, registration fit. Imports `config`, `types`.
- `GridRegistration` (`from_config`, `updated`, `map_to_grid`, `grid_to_map`, `cell_index`, `zero_based_cell`,
  `is_out_of_grid`, `cell_label`, `boundary_distance_m`); `CellProbabilities` (`as_dict`);
  `cell_probabilities(reg, mean_map_xy, cov_map_xy, max_cells=4)`; `apply_grid(obs, reg, max_cells=4)`;
  `fit_registration_from_points`.
- Invariants: unset registration -> `valid=False`, cells carry GRID_UNREGISTERED, never CONFIRMED; erf integration,
  out-of-grid mass kept.

**`tracking.py`** (Layer 4): association, de-correlated fusion, status machine, merges, bounded memory, final tags.
Imports `config`, `grid`, `types`.
- `Layer4Params.from_config(cfg, **overrides)` (property `ray_sigma_m` = hypot(`ray_bias_sigma_m`, `offset_sigma_m`));
  `hungarian(cost)`; `TrackObs`; `view_bins`; `fuse(items, p) -> (x, P, n_bins)`; `Track`; `Layer4` (`update(batch)`,
  `advance(now_ns)`, `tags()`, `final_tags(policy, threshold=None)`, `final_tag_set(...) -> FinalTagSet`,
  `transition`, `blocked`, `unreliable_reason`, `confirmable`); `expected_value_threshold(P, C)`; `FinalTagSet`;
  `LEGAL_TRANSITIONS`, `IllegalTransition`, `PENALTY_UNKNOWN_NOTE`.
- Also `Layer4.reconcile()` (affected-tracks recompute) and `_near_duplicates` (3.3a: one body seen twice in a frame is
  kept once, NEAR_DUPLICATE_OBSERVATION; `l4.min_separate_survivor_distance_m = 0` disables it).
- Policies: `confirmed_only`, `confirmed_plus_provisional_top1`, `expected_value` (not UNRELIABLE, >=
  `l5.commit_min_obs`, p_top >= threshold, default P/(40+P)). Invariants: only legal transitions; bounded tracks/observations; offset
  sigma 0 reproduces pre-3.1c behaviour exactly.

**`footprint.py`**: map-frame footprint refinement for extended (prone) clusters, default OFF. Imports `config`,
`geometry`, `types`.
- `FootprintParams.from_config`, `Footprint`, `principal_axis(xy)`, `analyse_footprint(xy, p, weights=None)`,
  `refine_observation(obs, t_map_camera, p) -> bool`. Invariant: non-extended observations stay bit-identical.

**`calibration.py`**: reliability and ECE, reusable on recorded data (Step 5). Imports nothing from the package:
`BinRow`, `reliability_table(pred, correct, bins=10)`, `ece(pred, correct, bins=10, p_min=None)`,
  `over_confidence_flags(..., min_count=20, max_over=0.10)`, `worst_over_confidence`, `format_reliability`.

**`output.py`**: Layer 5 telemetry codec and live output. Imports `types`.
- Codec: `TagFlags`, `SnapshotFlags`, `TagRecord`, `Snapshot`, `encode_tag_record`/`decode_tag_record`,
  `encode_snapshot`/`decode_snapshot`, `snapshot_size`, `crc16_ccitt_false`, `TelemetryDecodeError`.
- Live output (3.2): `TagChange`, `PublishedTag`, `record_flags`, `tag_record`, `full_tag_dict`, `TelemetryParams`,
  `TelemetryMessage`, `TelemetryStats`, `TelemetryScheduler`.
- Invariant: fixed 25-byte tag record, CRC-16/CCITT-FALSE; no images or point clouds.

**`corrections.py`** (Layer 5, 3.2): pose-independent observation store, correction fold, survivor ids. Imports `config`,
`grid`, `tracking`, `transforms`, `types`, `geometry`.
- `StoredObservation`, `stored_from`, `ObservationStore` (`add`, `get`, `replace`), `AppliedCorrection`, `EffectivePose`,
  `CorrectionOutcome`, `validate_correction`, `CorrectionHistory` (`effective`, `add`, `bake_excess`, `map_revision`),
  `corrected_lookups`, `TagState`, `TagBook` (`assign`, `remap`, `publish`, `pop_removed`), `item_reasons`.
- Invariants: events are folded in revision order over the stored base poses (arrival order does not matter); bounded
  store; tags are re-derived from stored observations, never patched.

**`pipeline.py`**: wiring, configured only from a `Config`. Imports library modules only.
- `Layer123Pipeline(cfg)` (`process_pose`, `process_frame`, `finalize`, `observations`; naive one-tag-per-tracking-id
  policy for layer comparison); `Layer1234Pipeline(cfg, output_statuses=..., **layer4_overrides)` (`final_tags(policy,
  threshold=None)`); `Layer5Pipeline` (adds `process_correction(PoseCorrection, arrival_ns)`,
  `process_registration(reg, arrival_ns)`, `published()`, `map_revision()`, `final_tags(policy=None, threshold=None)`;
  `l5.recompute_mode` full / affected_tracks; optional telemetry `sink`); `Layer5Params`, `telemetry_params`,
  `tag_result(tag) -> TagResult`.
- `Layer5Pipeline` also: `advance(arrival_ns)` (clock step without data: pose-wait expiry, flush, telemetry schedule),
  `publish_seq` (counter of republished tag lists), `job_sink` / `_snapshot` / `complete_rebuild(job)` (off-thread full recompute,
  3.4), `RebuildJob.run()` (any thread, snapshot only), `RebuiltState`.
- Order: capture-time pose -> floor plane -> Layer 1 -> camera-to-map -> footprint -> Layer 3 -> Layer 4.
## Harness side (simulation only)
- **`sim.py`**: arena, survivors, RGB-D camera, detector, poses. `SurvivorSpec` (`depth_m`), `Scenario`
  (`body_depth_range_m`), `place_survivors(scn, seed)`, `build_events(scn, seed)`, `Frame`, `Event`, `Closure`,
  `RegistrationUpdate`, `DetectorSim.dup_prob`, ray casting. Seven seeded streams per (seed, scenario name); adding a
  stream never changes earlier ones. One common placement shift per run keeps the authored spacing of a pair.
- **`scenarios.py`**: 26 named scenarios (+ 8 two_close members on request), `all_scenarios(placement="random_in_cell",
  body_depth="fixed", with_two_close=False)`, `pre_32_scenarios()` (the original 20), `NEW_IN_32`, `NEW_IN_33A`,
  `TWO_CLOSE_PAIRS`, `two_close_names()`, `with_correction_mode`, `without_correction_events`, `BODY_DEPTH_MODELS`,
  `make_scenario`, `make_ideal`, `sim_layer1_config`. All values are SIMULATION values.
- **`harness.py`**: `Pipeline` protocol, `score_tags`, `run_events`, `run_scenario`, `evaluate_many(pipelines,
  scenarios, seeds, estimators=None, hooks=())`, `evaluate`, `summarise`, `score_observations`,
  `score_camera_frame`, table formatters, `reliability_table` (fixed bins), `mission_points`, `MATCH_RADIUS_M`.
- **`baseline.py`**: `NaivePipeline` CONTROL (never reuse), `naive_factory`, `naive_layer1_factory`,
  `centre_pixel_estimator`, `layer1_estimator`.
- **`sim_pipelines.py`**: scenario -> Config glue: `scenario_config(scn, overrides=None)`, `registration_overrides`,
  `layer123_factory`, `layer1234_factory(overrides=None, **layer4_overrides)`, `layer5_factory`, `SimLayer5Pipeline`,
  `naive_l1_factory`.
- **`diagnostics.py`**: `diagnose_run`, `format_diag`, `DiagSummary`: why survivors are not confirmed (3.1b).
- **`experiments.py`**: 3.1b evaluation (pins pre-3.1c Layer 4 values in `PRE_31C_LAYER4`).
- **`experiments_calib.py`**: 3.1c sweep: `CachePipeline`, `cache_run`, `replay`, `rule_combos`, `acceptable`.
- **`golden.py`** (3.4: `record` / `check`; golden file `tests/golden/layer5_golden.json.gz`, 234 runs, behaviour gate for
  optimisations), **`experiments_34.py`** (3.4 timing, profile).
- **`experiments_layer5.py`** (3.2: R1-R3, F1-F4 tables), **`experiments_33a.py`** (3.3a: `run before|after`, `report`),
  **`experiments_33b.py`** (3.3b: `run`, `report`; stages `calib`, `sweep`, `rc`, `ablation`, `ra`, cached as
  `docs/reports/3.3b_cache_<stage>.json`; a stage is skipped if its cache file exists).

## Tests (pytest, `tests/`)
| test file | covers |
|---|---|
| `test_types.py`, `test_smoke.py`, `test_config.py` | `types`, package import / version, `config` |
| `test_geometry.py`, `test_transforms.py` | `geometry`, `transforms` |
| `test_grid.py` | `grid` (+ `sim_pipelines` registration glue) |
| `test_tracking.py` | `tracking` |
| `test_floor_footprint.py` | floor plane (`transforms`, `geometry`), `footprint`, pipeline order |
| `test_pipeline.py` | `pipeline`, library import-independence |
| `test_telemetry.py` | `output` |
| `test_sim.py` | `sim`, `scenarios` |
| `test_harness.py` | `harness`, `baseline`, `sim_pipelines` |
| `test_calibration.py` | `calibration`, ray sigma, expected-value policy, body-depth option, replay == pipeline, naive reference |
| `test_corrections.py` | `corrections`, `Layer5Pipeline`: R1 equivalence, correction semantics, spec 12.5 cases |
| `test_live_output.py` | `output` live output, `TelemetryScheduler` |
| `test_double_detection.py` | near-duplicate observations (3.3a), double_detection scenario |
| `test_two_close.py` | two_close_survivors family (3.3b) |
| `test_frozen_config.py` | `configs/frozen_synthetic_v1.json` matches the config registry |
| `test_golden.py`, `test_recompute_async.py` | behaviour gate (seed 0 of the golden file); off-thread rebuild and atomic swap (3.4) |
| `test_ros_adapters.py`, `test_ros_msgs.py` | ROS-free adapters, message files, mapping completeness (4.1a); helpers `ros_fakes.py` |
| `test_ros_runtime.py`, `test_ros_node_static.py`, `test_ros_smoke.py` | `TaggerRuntime` incl. AC1 end-to-end equivalence, node shell static checks, converter, smoke tool (4.1b); helper `ros_runtime_helpers.py` |
| `test_ros_node.py` | rclpy node (skipped without rclpy; NOT RUN on the dev machine) |

## Commands (from pyproject.toml, CLAUDE.md and module docstrings)
- All tests: `python -m pytest` (pyproject: testpaths=tests, pythonpath=src). One file: `python -m pytest tests/test_x.py`.
  Import check: `PYTHONPATH=src python -c "import survivor_tagging"`.
- Naive baseline table: `PYTHONPATH=src python -m survivor_tagging.harness` (~17 s).
- 3.1b evaluation: `PYTHONPATH=src python -m survivor_tagging.experiments` (~215 s).
- 3.1c evaluation: `PYTHONPATH=src python -m survivor_tagging.experiments_calib` (~800 s, multiprocess).
- 3.2 evaluation: `PYTHONPATH=src python -m survivor_tagging.experiments_layer5` (~570 s; writes `docs/reports/3.2_eval.txt`).
- 3.3a: `... experiments_33a run <before|after>`, then `... experiments_33a report`.
- 3.3b: `PYTHONPATH=src python -m survivor_tagging.experiments_33b run` (~17 min, 11 processes, writes the stage caches),
  then `... experiments_33b report` (seconds, writes `docs/reports/3.3b_eval.txt`).

## Where results go
`experiments`, `experiments_calib` and `harness` print to stdout only (3.1c output saved by hand as `3.1c_calib.txt`; the
3.1b output was not saved). `experiments_layer5`, `experiments_33a` and `experiments_33b` write under `docs/reports/`
(`3.2_eval.txt`, `3.3a_*.json|txt`, `3.3b_cache_*.json`, `3.3b_eval.txt`). Reports: `docs/reports/<task>.md` from
`docs/reports/TEMPLATE.md`; running notes: `docs/reports/NOTES.md`. Frozen parameter listing:
`configs/frozen_synthetic_v1.json`. No `3.1c.md` report exists; the 3.1c results are in architecture.md.

## ROS 2 side (`ros2/`, not part of `src/`; the library never imports it)
- `ros2/survivor_tagging_msgs` (ament_cmake): `Detection`, `DetectionArray`, `CandidateCell`, `SurvivorTag`, `SurvivorTagArray`,
  `TelemetryPacket`, `CorrectionEvent`, `SetGridRegistration.srv`, `CommitFinalTags.srv`. Never built here.
- `ros2/survivor_tagging_ros` (ament_python), module `survivor_tagging_ros/`:
  - `adapters.py` (4.1a, pure Python + numpy, duck-typed messages): `AdapterConfig`, `AdapterError`, `ConfigurationError`, stamps
    (`stamp_to_ns`, `ns_to_stamp`, `StampMonitor`), depth (`depth_image_to_metres`, `depth_image_to_raw`,
    `depth_measurement_from_image`), `intrinsics_from_camera_info`, detections (`detections_from_array`, `scale_detection_to_depth`),
    poses (`pose_from_msg`, `frd_to_flu_quaternion`, `ros_covariance_to_library`, `default_pose_covariance`), tags
    (`tag_to_msg_dict`, `tag_from_msg_dict`, `tag_array_to_msg_dict`, `tag_result_to_msg_dict`), telemetry, `correction_from_msg`,
    `registration_from_request`, `commit_request_to_args`, mapping tables `TAG_FIELD_MAP` etc.
  - `runtime.py` (4.1b): `TaggerRuntime` (`on_camera_info`, `on_depth`, `on_detections`, `on_pose`, `on_correction`,
    `call_set_registration`, `call_commit_final_tags`, `tick`, `finish`, `flush_rebuilds`, `diagnostics`), `RebuildWorker`,
    `build_markers`, `build_runtime(params, now_ns_fn, publish)`, `build_library_config`. Outputs via `publish(topic_key, payload)`.
  - `convert.py` (`dict_to_msg`), `tagger_node.py` (rclpy shell, forwards only; NOT RUN), `smoke_test.py` (`--dry-run`, ROS mode,
    `--write-sim-overrides`), `sim_bridge.py` (simulator -> message dicts; imports the simulator, test side only), `msgdef.py` (.msg parser,
    validator, CDR size), `launch/tagger.launch.py`, `config/tagger.params.yaml`, `tools/cov_default_effect.py`.
- Handoff documents: `docs/handoff/ros_interfaces.md`, `docs/handoff/ros_runbook.md`.
