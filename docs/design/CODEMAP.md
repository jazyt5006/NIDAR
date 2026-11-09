# CODEMAP: `src/survivor_tagging/`

Names verified against the code on 2026-11-09. "Library" = runs on the drone: numpy + stdlib only and must NOT import
`sim`, `scenarios`, `baseline`, `harness`, `sim_pipelines`, `diagnostics` or `experiments*` (checked by
`tests/test_pipeline.py` independence test for types, config, geometry, transforms, grid, pipeline, output, tracking,
footprint, calibration). "Harness" = test world and evaluation, may import anything.

## Library
**`types.py`**: data types, status enums, failure codes (architecture.md b, d). Imports nothing from the package.
- `Severity`, `TagStatus` (IntEnum), `FailureReason`, `FAILURE_SEVERITY`; frozen dataclasses `Detection`,
  `CameraIntrinsics`, `DepthMeasurement`, `PoseSample`, `Tag`, `TagResult`; mutable `SurvivorObservation`, `SurvivorTrack`.
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
- Policies: `confirmed_only`, `confirmed_plus_provisional_top1`, `expected_value` (not UNRELIABLE, >=
  `l5.commit_min_obs`, p_top >= P/(40+P)). Invariants: only legal transitions; bounded tracks/observations; offset
  sigma 0 reproduces pre-3.1c behaviour exactly.

**`footprint.py`**: map-frame footprint refinement for extended (prone) clusters, default OFF. Imports `config`,
`geometry`, `types`.
- `FootprintParams.from_config`, `Footprint`, `principal_axis(xy)`, `analyse_footprint(xy, p, weights=None)`,
  `refine_observation(obs, t_map_camera, p) -> bool`. Invariant: non-extended observations stay bit-identical.

**`calibration.py`**: reliability and ECE, reusable on recorded data (Step 5). Imports nothing from the package:
`BinRow`, `reliability_table(pred, correct, bins=10)`, `ece(pred, correct, bins=10, p_min=None)`,
  `over_confidence_flags(..., min_count=20, max_over=0.10)`, `worst_over_confidence`, `format_reliability`.

**`output.py`**: Layer 5 telemetry codec only. Imports `types`.
- `TagFlags`, `SnapshotFlags`, `TagRecord`, `Snapshot`, `encode_tag_record`/`decode_tag_record`,
  `encode_snapshot`/`decode_snapshot`, `snapshot_size`, `crc16_ccitt_false`, `TelemetryDecodeError`.
- Invariant: fixed 25-byte tag record, CRC-16/CCITT-FALSE; no images or point clouds.

**`pipeline.py`**: wiring, configured only from a `Config`. Imports library modules only.
- `Layer123Pipeline(cfg)` (`process_pose`, `process_frame`, `finalize`, `observations`; naive one-tag-per-tracking-id
  policy for layer comparison); `Layer1234Pipeline(cfg, output_statuses=..., **layer4_overrides)` (`final_tags(policy,
  threshold=None)`); `tag_result(tag) -> TagResult`.
- Order: capture-time pose -> floor plane -> Layer 1 -> camera-to-map -> footprint -> Layer 3 -> Layer 4.
## Harness side (simulation only)
- **`sim.py`**: arena, survivors, RGB-D camera, detector, poses. `SurvivorSpec` (`depth_m`), `Scenario`
  (`body_depth_range_m`), `place_survivors(scn, seed)`, `build_events(scn, seed)`, `Frame`, `Event`, ray casting.
  Five seeded streams per (seed, scenario name); adding a stream never changes earlier ones.
- **`scenarios.py`**: 20 named scenarios, `all_scenarios(placement="random_in_cell", body_depth="fixed")`,
  `BODY_DEPTH_MODELS`, `make_scenario`, `make_ideal`, `sim_layer1_config`. All values are SIMULATION values.
- **`harness.py`**: `Pipeline` protocol, `score_tags`, `run_events`, `run_scenario`, `evaluate_many(pipelines,
  scenarios, seeds, estimators=None, hooks=())`, `evaluate`, `summarise`, `score_observations`,
  `score_camera_frame`, table formatters, `reliability_table` (fixed bins), `mission_points`, `MATCH_RADIUS_M`.
- **`baseline.py`**: `NaivePipeline` CONTROL (never reuse), `naive_factory`, `naive_layer1_factory`,
  `centre_pixel_estimator`, `layer1_estimator`.
- **`sim_pipelines.py`**: scenario -> Config glue: `scenario_config(scn, overrides=None)`, `registration_overrides`,
  `layer123_factory`, `layer1234_factory(overrides=None, **layer4_overrides)`, `naive_l1_factory`.
- **`diagnostics.py`**: `diagnose_run`, `format_diag`, `DiagSummary`: why survivors are not confirmed (3.1b).
- **`experiments.py`**: 3.1b evaluation (pins pre-3.1c Layer 4 values in `PRE_31C_LAYER4`).
- **`experiments_calib.py`**: 3.1c sweep: `CachePipeline`, `cache_run`, `replay`, `rule_combos`, `acceptable`.

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

## Commands (from pyproject.toml, CLAUDE.md and module docstrings)
- All tests: `python -m pytest` (pyproject: testpaths=tests, pythonpath=src). One file: `python -m pytest tests/test_x.py`.
  Import check: `PYTHONPATH=src python -c "import survivor_tagging"`.
- Naive baseline table: `PYTHONPATH=src python -m survivor_tagging.harness` (~17 s).
- 3.1b evaluation: `PYTHONPATH=src python -m survivor_tagging.experiments` (~215 s).
- 3.1c evaluation: `PYTHONPATH=src python -m survivor_tagging.experiments_calib` (~800 s, multiprocess).

## Where results go
All evaluation scripts print to stdout only; none writes a file. Saved outputs (redirected by hand) live in
`docs/reports/` (e.g. `3.1c_calib.txt`). The 3.1b output was not saved to the repo. Reports: `docs/reports/<task>.md`
from `docs/reports/TEMPLATE.md`; running notes: `docs/reports/NOTES.md`.
