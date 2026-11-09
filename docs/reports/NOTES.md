# Running notes

## 3.2 Layer 5: corrections, revisions, live output (2026-11-09)
- New library module `corrections.py` (store, correction fold, TagBook); `Layer5Pipeline` in `pipeline.py`; live output
  (PublishedTag, tag_record, full_tag_dict, TelemetryScheduler) appended to `output.py` (codec unchanged). Library
  independence test now also covers `corrections` and bans `experiments_calib` / `experiments_layer5`.
- Recompute = re-derive every stored observation (Layer 2 from the effective capture-time pose, Layer 3 with the
  current registration) and replay Layer 4 from scratch in the ORIGINAL batches (stored batch id). Replaying per
  capture stamp would not be exact: Layer 4 timeouts run once per update call. This makes R1 exact (0.0 m).
- Correction fold is in revision order over the stored base pose, so arrival order does not matter. History events
  store, per affected observation, the corrected lookup from event poses + live poses of the SAME revision (needed at
  the end of the corrected range). A mixed-revision bracket counts as uncorrected: new `PoseLookup.min_revision`.
- Frame correction window = [previous correction stamp in revision order, event stamp), as the task states. The
  POSE_NOT_CORRECTED rule: captured before an event whose revision it did not reach.
- Missing correction: a live pose_revision bump with no event after `l5.missing_correction_timeout_s` adds a marker
  that flags older observations; the real event replaces it later.
- Evicted observations still held by a Layer 4 track are replayed with their old position, x
  `l5.uncorrected_cov_inflation`, OBSERVATION_NOT_REDERIVABLE. Only evictions before a recompute become such copies.
- Survivor id: assigned when a track is first published (> CANDIDATE), never reused; after a rebuild matched by
  shared observation ids (most shared, then lower id). The losing id of a merge is announced once with REMOVED.
  Tag revision +1 on every confirmed-cell change; confirmed tags then publish as UPDATED.
- `l5.correction_source` choices changed from (none, pose_revision_stream, keyframe_updates) [never used in code] to
  (none, history, frame_correction). Events of another source -> CORRECTION_SOURCE_UNAVAILABLE.
- New ASSUMPTION params: l5.frame_correction_cov_inflation 4, l5.uncorrected_cov_inflation 4,
  l5.correction_history_max 8, l5.missing_correction_timeout_s 2, l5.recompute_mode full, l5.tag_history_max 8,
  l5.removed_tags_max 64, l5.telemetry_update_min_interval_s 1 (architecture send rule 1), l5.telemetry_log_max 4096.
- New codes: POSE_NOT_CORRECTED, OBSERVATION_NOT_REDERIVABLE, FRAME_CORRECTION_APPLIED (DEGRADE);
  CORRECTION_INVALID, CORRECTION_CONFLICT, CORRECTION_STALE (REJECT).
- grid.py: Gauss-Legendre nodes cached in a module constant (they were recomputed on every correlated-covariance
  call); results bit-identical (all earlier tests pass). tracking.py: additive `Layer4.reconcile()` for the
  affected-tracks mode. transforms.py: `PoseLookup.min_revision`, `PoseBuffer.poses()`, `apply_frame_jump`.
- Sim: `Closure` (history / frame_correction / emit=False), drift restarts at closures, registration updates,
  6th random stream (residuals). Pre-3.2 pose/frame streams verified unchanged (digests pinned in test_sim).
  experiments.py, experiments_calib.py and the harness main now use `pre_32_scenarios()` so earlier tables are
  reproducible.
- R3 FAILED (full recompute in seconds, not 250 ms); affected_tracks mode added as the prescribed remedy, default stays
  `full` (only full is exact for R1). Cost is dominated by the Layer 4 replay (re-fusion of every item per update) and
  Layer 3 erf integration.
- Finding (Layer 4, not changed here): a detector false positive overlapping the person gives two observations of one
  body in one frame -> two tracks -> every later observation ASSOCIATION_AMBIGUOUS -> both UNRELIABLE -> dropped.
  Layer 5's correction exposed it in loop_closure seed 2 (R1 shows the offline run does the same).
- Finding: repeated_corrections duplicates rise with Layer 5 because post-closure drift (0.12 m/s, not in the pose
  covariance) leaves later observations outside the gate of the now tight corrected track.
- Frame vs history (sim): equal for constant offsets; with drift inside the window frame_correction is worse (see 3.2.md).
- Status 2026-11-09 (user check): A-D finished and tested; E tests written, new/affected files pass, full suite
  not yet run (interrupted); F evaluation finished (5 seeds, 567 s, docs/reports/3.2_eval.txt): R1 PASS 35/35,
  R2 PASS, R3 FAIL (12.8 s full at N=2000; affected_tracks 0.52 s for a 4 s window, 10.3 s whole history).
  Remaining: full test suite (background, output to file), report 3.2.md. Eval not re-run (no code change since);
  any re-run will use 3 seeds.
- Full test suite: 289 passed (167 s). Report written: docs/reports/3.2.md. 3.2 complete.

## 3.3a Layer 4 near-duplicates and honest sim pose covariance (2026-11-09)
- tracking.py: same-frame observations closer than l4.min_separate_survivor_distance_m that gate with a common track: keep the higher-confidence one, drop the other (NEAR_DUPLICATE_OBSERVATION, EVENT severity, in LAYER4_OBS_CODES so Layer 5 re-derivation clears it), no ambiguity count. _merge may merge tracks whose shared frames are all near-duplicates; merge now runs before the proximity flag; an unambiguous assignment resets the streak of all gated tracks.
- sim: DetectorSim.dup_prob/dup_offset_m (second widened box, fresh tracking id; draws only if dup_prob > 0), 7th random stream (closure drift), live drift after a closure is random with covariance (sigma dt)^2, corrected-history poses always carry sigma_res^2. double_detection added; pre_32_scenarios() excludes it (NEW_IN_33A).
- Results: A1 PASS, A2 PASS (NIS 1.86-2.19; repeated_corrections dup 0.6 -> 0.0), A3 FAIL (4/15 runs in revisit_after_gap, opposite_views_boundary, thin_person_in_front_of_wall: FP boxes over the person now dropped). See 3.3a.md.
- A3 baseline was generated by experiments_33a run before, before editing the library (3.2_eval.txt covers only 5 scenarios).

## 3.3b End-to-end evaluation, interim policy, freeze, end-of-step docs (2026-11-09/10)
- New harness module `experiments_33b.py` (stages calib, sweep, rc, ablation, ra; raw rows cached in `docs/reports/3.3b_cache_*.json`, tables in `3.3b_eval.txt`, report `3.3b.md`). No library code changed.
- Scenarios: `two_close_survivors/*` family (8 pairs, 0.30-0.60 m apart) added to `scenarios.py`; opt-in via `all_scenarios(with_two_close=True)` because `test_random_in_cell_placement_reproducible_and_inside_cell` assumes a 0.8 m spacing of every two-survivor scenario (not weakened). `pre_32_scenarios()` still returns the original 20.
- Results: R-A FLAGGED (-4.00 points/run at P = 40, 3 of 30 runs; the -1.0 threshold is below one run, so it cannot distinguish one run from none); R-B keeps `l4.min_separate_survivor_distance_m` 0.5 (no discrimination: <= 1 miss in 80); R-C interim `expected_value` 0.667 (worst case 36.94 points/run), not applied as a default.
- A: the 0.6-0.7 bin mismatch at offset 0.2 is cell_boundary (56 %); section A recomputed with current defaults because the 3.1c cache was never saved.
- `configs/frozen_synthetic_v1.json` + `tests/test_frozen_config.py`: all 116 registry parameters with source and evidence level (35 unset on purpose). Generated once from `default_config()`; if a default changes, update the file (the test fails otherwise).
- Docs updated at the end of the step: architecture.md (Layer 5, 3.3a, 3.3b, decision-rule table, known gaps), harness.md, conventions.md, teammate_questions.md (one ordered table, O7 merged into O2/O5, O8 added), CODEMAP.md, CLAUDE.md. There is no `docs/reports/3.1c.md` in the repo.

## 3.4 Layer 4 replay performance (2026-11-10, stopped early)
- Profile before (full recompute, 2000 stored): fuse 22.6 s cum of 40 s (numpy inv 189k calls), cell_probabilities 6.8 s, view_bins 2.0 s, stamps() 0.8 s. Live feed: same set plus _merge 5.4 s, _tag 5.3 s, telemetry step 4.8 s.
- Fixes (all gate-P1 clean): scalar fuse with cached R^-1, scalar erf in cell_probabilities, lazy Track.probs and SurvivorObservation.cell_probs, cached stamps, RebuildJob/complete_rebuild for off-thread recompute. Gate: `python -m survivor_tagging.golden check`; golden file tests/golden/layer5_golden.json.gz recorded BEFORE any change.
- 1.0 s target not reliably met; timings are noisy (2x). See reports/3.4.md.

## 4.1a ROS 2 message definitions and ROS-free adapters (2026-11-10)
- New top-level `ros2/`: `survivor_tagging_msgs` (9 interface files + CandidateCell helper msg, ament_cmake) and `survivor_tagging_ros` (ament_python skeleton, `adapters.py`, `msgdef.py`; no nodes). Handoff: `docs/handoff/ros_interfaces.md`. Tests: `tests/test_ros_adapters.py`, `tests/test_ros_msgs.py`, helper `tests/ros_fakes.py`. Nothing was built with rosidl/colcon (no ROS here).
- Library change: three new FailureReason codes (REJECT): FRAME_MISMATCH, TIMESTAMP_INVALID, IMAGE_FORMAT_INVALID. architecture.md (d) code list not yet updated (docs only at end of step).
- Decisions: detection_id = index in the array (message has none); tracking_id -1 = None; confidence outside 0..1 -> BBOX_INVALID; all-zero or cov[0] == -1 pose covariance = unknown (None) rather than a perfect pose; ROS 6x6 covariance == library layout (reshape), unchanged by FRD/FLU; pose_revision is supplied by the caller (not in ROS messages); SetGridRegistration carries every GridRegistration field (extent 0 = unknown) and the adapter raises CORRECTION_STALE for version < current, equal version is left to the pipeline.
- Only tuning default: AdapterConfig.image_scale_tolerance = 0.01 (ASSUMPTION). All frame / convention / alignment / depth-range parameters are unset and raise ConfigurationError.
