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
