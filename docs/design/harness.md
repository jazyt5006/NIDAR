# Test World and Scoring Harness

All results produced with this harness are **simulation tested on synthetic data**. All numbers in `sim.py` and
`scenarios.py` are SIMULATION values, not hardware or competition values.

## Pieces
| Module | Role |
|---|---|
| `sim.py` | Synthetic world: grid-aligned arena (walls 2.4 m high, 0.05 m thick), survivor boxes (standing 0.5x0.3x1.7, sitting 0.5x0.5x1.0, prone 1.7x0.5x0.25 m), virtual RGB-D camera (vectorised ray casting, uint16 mm depth, noise), virtual detector, pose stream, event stream. |
| `scenarios.py` | Named scenarios, `make_scenario(...)` to build new ones, `make_ideal(scn)` to remove all noise and impairments. |
| `harness.py` | `Pipeline` protocol, `run_scenario`, `evaluate`, `score_tags`, `format_table`. `python -m survivor_tagging.harness` runs the naive baseline over every scenario (5 seeds). |
| `baseline.py` | `NaivePipeline`: deliberately naive control with a pluggable point estimator (`centre_pixel_estimator` default, `layer1_estimator` for the Layer 1 ablation). Never reused by the real layers. |
| `experiments.py` | Substep 2.3 tables: reference, camera-frame comparison, end-to-end ablation, offset sweep. |

The simulator's geometry is independent of the pipeline: it projects the world into the image and ray-casts depth
in the grid frame. A pipeline only ever sees image-space detections, depth pixels and poses, so frame or sign
errors in a pipeline cannot be cancelled by matching errors in the simulator.

## Event stream
`build_events(scenario, seed)` returns a list of `Event(arrival_ns, kind, payload)` sorted by arrival time
(ties: poses before frames, then generation order). Single clock domain `"sim"`, integer ns.
- `kind="pose"`: `PoseSample` stamped at its true time, `arrival_ns = stamp + pose latency`. The reported position
  can include constant drift and, before a loop-closure time, a jump error; poses after the closure have
  `pose_revision` + 1.
- `kind="frame"`: `Frame` with the capture stamp, detections (stamped with the capture time), a `DepthMeasurement`
  and intrinsics; `arrival_ns = stamp + detector latency`.
Randomness comes from `numpy.random.SeedSequence([seed, crc32(scenario name)])`, so a (scenario, seed) pair is
exactly reproducible.

**Placement.** Default `placement="random_in_cell"`: per seed, one common shift is drawn uniformly over the shifts
that keep every survivor centre inside its allowed region (its cell, or a narrower region the scenario states to
keep walls/occlusion valid), so two-survivor spacing is preserved. `all_scenarios(placement="nominal")` restores
the authored centres. `cell_boundary` keeps x exactly on the edge (y random). `place_survivors(scn, seed)` resolves
the placement; the harness scores against the placed truths.

**Frame ground truth** (scoring only, a pipeline must never read it): `det_survivor_ids` (survivor id per
detection, None = false positive), `truth_camera_m` (true box centre in the camera frame), `T_grid_camera`.

## Pipeline protocol
```
process_pose(pose: PoseSample, arrival_ns: int) -> None
process_frame(detections, depth: DepthMeasurement, intrinsics: CameraIntrinsics, arrival_ns: int) -> None
finalize() -> list[TagResult]   # survivor_id, cell (i, j) or None, status, p_map_m or None,
                                 # first_stamp_ns, confirm_stamp_ns or None
```
A pipeline gets its static configuration (camera mount, registration) from a factory `factory(scenario)`.

### Driving it from the controls team's simulator
Implement nothing on our side: in your simulator loop, convert each pose to `PoseSample` and each camera frame to
(`Detection` list, `DepthMeasurement`, `CameraIntrinsics`), then call `process_pose` / `process_frame` in the order
the data would arrive on the drone, passing the arrival time. Call `finalize()` at the end and score with
`score_tags(truths, tags)`, where each truth is `SurvivorTruth(id, centre_map_xy, cell, posture)`. Required
conventions: docs/design/conventions.md (camera optical x right / y down / z forward, body FLU, ns timestamps,
capture-time stamps on detections and depth).

## Scenarios (default 6x6 arena, registration offset (1.2, 0.8) m, yaw 0.25 rad)
| Name | Tests |
|---|---|
| standing / sitting / prone | posture; survivor in cell (3,3), about 1.8-2.8 m ahead |
| near_wall | full wall 0.01-0.3 m behind the survivor's back face (background depth) |
| cell_boundary | survivor centred exactly on the x = 3 edge (true cell 3 by floor) |
| occluded | lower half hidden behind a 1.2 m wall; centre still visible |
| two_same_cell / two_adjacent | two survivors 0.8 m apart, same cell / adjacent cells |
| tilt_0 / tilt_30 / tilt_45 | camera tilt down; survivor 1.5-2.1 m ahead of the camera |
| latency_fast_yaw | 100 ms pose latency with +/-25 deg yaw at 0.6 Hz (capture-time alignment) |
| loop_closure | 0.3 m pose error before t = 1.5 s, removed by a loop closure (correction handling) |
| low_occluder | survivor behind a 0.9 m wall (top below the chest): wall is a band across the box bottom |
| thin_person_in_front_of_wall | loose detector box (`DetectorSim.bbox_pad_frac` 1.5) around a survivor 1.0-1.5 m in front of a full wall |
| id_switch | detector assigns a new tracking id in ~30 % of frames |
| id_merge | two survivors 0.8 m apart share one detector tracking id (`DetectorSim.id_merge_dist_m`) |
| intermittent_false_positive | a 0.8 m obstacle triggers a false detection in ~20 % of frames (`phantom_box`) |
| revisit_after_gap | survivor seen ~2 s, camera turned away ~6 s while flying 2.7 m away and back, seen again (pose drift) |
| opposite_views_boundary | survivor 0.03-0.12 m from a cell edge, seen along a half circle from both sides (pose drift) |
Default impairments (SIM): depth sigma 0.002 z^2, 2 % dropout, flying pixels at depth edges, bbox jitter 3 px,
10 % misses, 0.05 false positives per frame, 2 % per-frame tracking-ID switch, 50 ms detector latency, 2 deg wobble.

## Metrics (per scenario, aggregated over seeds)
Each scored tag is assigned to its nearest survivor within 1 m (otherwise it is a **false tag**); per survivor the
nearest assigned tag is the match and the rest are **duplicates**. UNRELIABLE tags are not scored; tags with no
position count as false.
- **correct**: fraction of survivors whose matched tag has the true cell (the competition metric).
- **missed**, **false**, **dup**: mean per run.
- **err_med_m / err_p90_m**: median / 90th percentile horizontal error of matched tags (pooled over seeds).
- **t_conf_s**: median time from scenario start to the first CONFIRMED tag of each survivor.

Pose uncertainty (substep 3.1, off unless a scenario sets it): `Trajectory.pose_drift_sigma_m_per_sqrt_m` makes
the reported horizontal position error a random walk whose variance grows with distance travelled, and the
reported pose covariance is exactly that variance (honest by construction, tested); `loop_closure_variance_factor`
scales both error and variance at the closure. New random draws use a fourth seed stream, so earlier scenarios are
unchanged.

### Status-aware tag metrics (substep 3.1)
`score_tags(..., first_seen_ns)` and `format_status_table`: **corr_any** (matched tag in the true cell, any
status), **corr_conf** (CONFIRMED/UPDATED in the true cell), **missed**, **f_conf** / **f_prov** (CONFIRMED /
PROVISIONAL tags that are unmatched or in the wrong cell, per run), **dup**, **flick** (cell changes after
confirmation), **err_med**, **t_conf** (median time from the survivor's first detection to confirmation).
`reliability_table(pairs)`: predicted top-cell probability bins vs observed fraction correct.
`mission_points(truths, committed, +40, penalty)`: ASSUMED scoring for the unknown wrong-tag penalty (O2).

### Non-confirmation diagnostics (substep 3.1b, `diagnostics.py`)
`diagnose_run(placed, frames, layer1234_pipeline, summary)` (use as an `evaluate_many` hook) labels every survivor
that is not CONFIRMED in its true cell with exactly one reason, in order: confirmed_wrong_cell, split,
never_associated, blocked, min_obs, span, prob_below (with top probability and distance to the nearest cell edge),
other. `format_diag` prints per scenario: counts per label, accepted observations per survivor (min/median/max), median
top probability and edge distance of the prob_below cases. `summary.statuses` counts final track statuses plus
DROPPED_TIMEOUT / MERGED / EVICTED. min_obs and span are scenario-length artifacts; the others are not.

### Calibration metrics (substep 3.1c, `calibration.py` + `experiments_calib.py`)
- Samples: after every Layer 4 update, each track that received an observation and lies within 1 m of a true survivor
  gives (top-cell probability, top cell == true cell). Pool excludes loop_closure (needs Layer 5) and prone (reported
  separately).
- `reliability_table(pred, correct, bins=10)`: equal-width bins (p = 1 in the last bin); `ece(..., p_min=0.5)`: sum of
  (bin count / total) x |mean predicted - observed|, total = samples with p >= p_min; `over_confidence_flags` /
  `worst_over_confidence`: bins with >= 20 samples whose mean predicted exceeds observed by more than 0.10. These are
  library functions, reusable on recorded data (Step 5).
- True body-depth models (`scenarios.BODY_DEPTH_MODELS`): "fixed" (default, 0.5 m box depth, bias 0.25 m) and
  "uniform_0.2_0.5" (per-survivor depth from its own random stream, bias 0.10-0.25 m, mean 0.175 m). The pipeline
  never sees it.
- Layers 1-3 are cached per (configured offset, true model, scenario, seed) and only Layer 4 is replayed
  (`CachePipeline`, `replay`; a test checks a replay equals `Layer1234Pipeline`).
- Policy points: +40 per correct tag (rulebook) and -P per wrong, duplicate or false-positive committed tag (P ASSUMED).

### Observation-level metrics (pipelines exposing `observations()`)
`score_observations(placed, frames, observations)`: every processed detection of a real survivor (linked by capture
stamp and detection_id): **accept** rate, **correct** cell rate among accepted observations, horizontal map-frame
error (median, p90), rejection counts by reason, and accepted false-positive observations (`fp_acc`).

### Camera-frame metrics (point estimators, independent of poses, transforms and grid)
`score_camera_frame(placed, frames, estimator)` over detections of real survivors (false positives excluded):
- **valid**: fraction with a point; **ambig**: fraction with DEPTH_AMBIGUOUS_CLUSTERS.
- **false_as**: among valid points, fraction farther than 0.5 m from the survivor's box (mapped with the true pose):
  the depth came from another surface.
- **h_med / h_p90** (primary): horizontal error, i.e. the 3D error vector rotated into the grid frame with the TRUE
  camera attitude, vertical part dropped (the cell depends on horizontal position only). **<0.15 / <0.25 / <0.5**:
  share of horizontal errors below each threshold.
- **3d_med / 3d_p90**: 3D distance to the true box centre (front-surface estimates carry the half-depth of
  the box as bias). **t_med_ms / t_p95_ms**: per-detection estimator time.
`evaluate_many(pipelines, scenarios, seeds, estimators)` generates each event stream once for all of them.
