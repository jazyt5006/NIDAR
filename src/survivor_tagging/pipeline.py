"""Layer123Pipeline: capture-time pose -> floor plane -> Layer 1 -> camera-to-map (Layer 2) -> footprint refinement
-> Layer 3, configured only from a Config. Detections wait in the bounded pending queue until a pose newer than their
capture time has arrived (or fail explicitly); the latest pose is never used as a substitute.

Implements the harness Pipeline protocol. Tag policy is deliberately the SAME as the naive baseline (one tag per
detector tracking_id, from its first fully processed observation, no fusion) so that comparisons isolate the layers;
Layer 4 replaces it later. A tag is PROVISIONAL instead of CONFIRMED if a BLOCK-severity warning is present
(e.g. GRID_UNREGISTERED). No simulator imports.
"""

from __future__ import annotations

import copy
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

from .config import Config, ConfigError
from .corrections import (LAYER4_OBS_CODES, AppliedCorrection, CorrectionHistory, CorrectionOutcome, EffectivePose,
                          ObservationStore, StoredObservation, TagBook, corrected_lookups, item_reasons, stored_from,
                          validate_correction)
from .footprint import FootprintParams, refine_observation
from .geometry import Layer1Params, estimate_survivor_point
from .grid import GridRegistration, apply_grid
from .registration import WallAlignParams, WallAlignResult, estimate_wall_alignment, update_needed
from .output import PublishedTag, TelemetryMessage, TelemetryParams, TelemetryScheduler
from .tracking import Layer4, Layer4Params
from .transforms import (NS_PER_S, Layer2, Layer2Params, PendingItem, extrinsic_from_config, floor_plane_in_camera,
                         pose_matrix)
from .types import (FAILURE_SEVERITY, CameraIntrinsics, DepthMeasurement, Detection, FailureReason, PoseCorrection,
                    PoseSample, Severity, SurvivorObservation, Tag, TagResult, TagStatus)


class Layer123Pipeline:
    def __init__(self, cfg: Config):
        self.l1 = Layer1Params.from_config(cfg)
        extr, extr_fail = extrinsic_from_config(cfg)
        self.l2 = Layer2(Layer2Params.from_config(cfg), extr, extr_fail)
        self.grid = GridRegistration.from_config(cfg)
        self.max_cells = cfg.value("grid.max_candidate_cells")
        self.floor_removal = bool(cfg.value("l1.floor_removal"))
        self.floor_z_m = cfg.value("map.floor_z_m")
        self.floor_margin_m = cfg.value("l1.floor_margin_m")
        self.floor_sigma_k = cfg.value("l1.floor_margin_pose_sigma_k")
        self.footprint = FootprintParams.from_config(cfg) if cfg.value("l1b.footprint_refinement") else None
        self.l1_ms: List[float] = []
        self.footprint_ms: List[float] = []
        self.obs: List[SurvivorObservation] = []
        self.tags: Dict[object, object] = {}
        self.now_ns: Optional[int] = None
        self.l2_ms: List[float] = []
        self.l3_ms: List[float] = []

    # --- Pipeline protocol --------------------------------------------------------------------------------
    def process_pose(self, pose: PoseSample, arrival_ns: int) -> None:
        self.now_ns = arrival_ns
        self.l2.add_pose(pose)
        for item in self.l2.queue.pop_ready(self.l2.buffer.newest_ns):
            self._resolve(item)
        self._expire()

    def process_frame(self, detections: Sequence[Detection], depth: DepthMeasurement,
                      intrinsics: CameraIntrinsics, arrival_ns: int) -> None:
        self.now_ns = arrival_ns
        self._expire()
        for det in detections:
            obs = SurvivorObservation(len(self.obs), det.detection_id, det.confidence, det.image_stamp_ns,
                                      det.tracking_id)
            self.obs.append(obs)
            capture, fail = self.l2.check(det, depth, arrival_ns)
            if fail is not None:
                obs.failure = fail
                continue
            item = PendingItem(capture, arrival_ns + int(self.l2.params.max_pose_wait_s * 1e9),
                               (obs, det, depth, intrinsics))
            if self.l2.waiting_for_pose(capture):
                evicted = self.l2.queue.push(item)
                if evicted is not None:
                    evicted.payload[0].failure = FailureReason.CAPACITY_EXCEEDED
            else:
                self._resolve(item)

    def finalize(self) -> List[object]:
        for item in self.l2.queue.drain():
            item.payload[0].failure = FailureReason.POSE_UNAVAILABLE_AT_CAPTURE
        return list(self.tags.values())

    def observations(self) -> List[SurvivorObservation]:
        return self.obs

    # --- internals ----------------------------------------------------------------------------------------
    def _expire(self) -> None:
        for item in self.l2.queue.pop_expired(self.now_ns):
            item.payload[0].failure = FailureReason.POSE_UNAVAILABLE_AT_CAPTURE

    def _resolve(self, item: PendingItem) -> None:
        obs, det, depth, intrinsics = item.payload
        t0 = time.perf_counter()
        lk = self.l2.lookup(item.capture_pose_ns)
        if lk.failure is not None:
            obs.failure = lk.failure
            self.l2_ms.append((time.perf_counter() - t0) * 1e3)
            return
        floor = None
        if self.floor_removal:
            if self.floor_z_m is None:
                obs.warnings.append(FailureReason.FLOOR_UNKNOWN)
            else:
                floor = floor_plane_in_camera(lk.pose, self.l2.extrinsic, self.floor_z_m, self.floor_margin_m,
                                              self.floor_sigma_k)
        ta = time.perf_counter()
        r = estimate_survivor_point(det, depth, intrinsics, self.l1, floor=floor)
        tb = time.perf_counter()
        self.l1_ms.append((tb - ta) * 1e3)
        obs.p_camera_m, obs.cov_camera_m2, obs.ground_ref_camera_m = r.p_camera_m, r.cov_camera_m2, \
            r.ground_ref_camera_m
        obs.n_valid_px, obs.valid_fraction, obs.depth_quality = r.n_valid_px, r.valid_fraction, r.depth_quality
        obs.floor_removed_px, obs.cluster_points_camera_m = r.floor_removed_px, r.cluster_points_camera_m
        obs.warnings.extend(r.warnings)
        n_l1_warnings = len(obs.warnings)
        if r.failure is not None:
            obs.failure = r.failure
            return
        self.l2.apply(obs, lk)
        t1 = time.perf_counter()
        self.l2_ms.append((ta - t0 + t1 - tb) * 1e3)
        if self.footprint is not None:
            refine_observation(obs, pose_matrix(lk.pose) @ self.l2.extrinsic.T_body_camera, self.footprint)
            self.footprint_ms.append((time.perf_counter() - t1) * 1e3)
            t1 = time.perf_counter()
        apply_grid(obs, self.grid, self.max_cells)
        self.l3_ms.append((time.perf_counter() - t1) * 1e3)
        self._accept(obs, lk, item.capture_pose_ns, n_l1_warnings)

    def _accept(self, obs: SurvivorObservation, lk, capture_pose_ns: int, n_l1_warnings: int) -> None:
        """An observation finished Layers 1-3 (Layer5Pipeline stores it for re-derivation first)."""
        self._tag(obs)

    def _tag(self, obs: SurvivorObservation) -> None:
        key = obs.tracking_id if obs.tracking_id is not None else ("untracked", obs.capture_stamp_ns,
                                                                   obs.detection_id)
        if key in self.tags:
            return
        blocked = any(FAILURE_SEVERITY[w] is Severity.BLOCK for w in obs.warnings)
        status = TagStatus.PROVISIONAL if blocked else TagStatus.CONFIRMED
        self.tags[key] = TagResult(len(self.tags), obs.cell, status, obs.p_map_m, obs.capture_stamp_ns,
                                         None if blocked else obs.capture_stamp_ns)


class Layer1234Pipeline(Layer123Pipeline):
    """Layers 1-4. Observations finished by Layer 3 are batched per call and handed to Layer 4, which owns the tag
    policy. finalize() returns TagResults for the statuses in `output_statuses` (default: PROVISIONAL, CONFIRMED,
    UPDATED; CANDIDATE and UNRELIABLE are not reported); final_tags(policy) gives the end-of-mission commit set."""

    def __init__(self, cfg: Config, output_statuses=(TagStatus.PROVISIONAL, TagStatus.CONFIRMED, TagStatus.UPDATED),
                 **layer4_overrides):
        super().__init__(cfg)
        self.l4 = Layer4(Layer4Params.from_config(cfg, **layer4_overrides), self.grid)
        self.output_statuses = tuple(output_statuses)
        self._batch: List[SurvivorObservation] = []
        self.l4_ms: List[float] = []

    def _tag(self, obs: SurvivorObservation) -> None:
        self._batch.append(obs)

    def _flush(self) -> None:
        if not self._batch:
            return
        t0 = time.perf_counter()
        self.l4.update(self._batch)
        per_obs = (time.perf_counter() - t0) * 1e3 / len(self._batch)
        self.l4_ms.extend([per_obs] * len(self._batch))
        self._batch = []

    def process_pose(self, pose: PoseSample, arrival_ns: int) -> None:
        super().process_pose(pose, arrival_ns)
        self._flush()

    def process_frame(self, detections: Sequence[Detection], depth: DepthMeasurement,
                      intrinsics: CameraIntrinsics, arrival_ns: int) -> None:
        super().process_frame(detections, depth, intrinsics, arrival_ns)
        self._flush()

    def finalize(self) -> List[TagResult]:
        super().finalize()
        self._flush()
        return [tag_result(t) for t in self.l4.tags() if t.status in self.output_statuses]

    def final_tags(self, policy: str, threshold: Optional[float] = None) -> List[TagResult]:
        return [tag_result(t) for t in self.l4.final_tags(policy, threshold)]


def tag_result(t: Tag) -> TagResult:
    prob = dict(t.cell_probs).get(t.cell) if t.cell is not None else None
    return TagResult(t.survivor_id, t.cell, t.status, t.p_map_m, t.first_seen_ns, t.confirmed_stamp_ns,
                     0.0 if prob is None and t.cell is not None else prob, t.cell_changes)


# ---------------------------------------------------------------------------
# Layer 5 (substep 3.2): correction handling, revisions and live output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Layer5Params:
    correction_source: Optional[str]  # l5.correction_source (None = unset)
    store_max: int  # l5.observation_store_max
    frame_inflation: float  # l5.frame_correction_cov_inflation
    uncorrected_inflation: float  # l5.uncorrected_cov_inflation
    history_max: int  # l5.correction_history_max
    missing_timeout_s: float  # l5.missing_correction_timeout_s
    tag_history_max: int  # l5.tag_history_max
    removed_max: int  # l5.removed_tags_max
    telemetry_log_max: int  # l5.telemetry_log_max
    commit_policy: Optional[str]  # l5.end_of_mission_commit_policy (None = unset, O2)
    recompute_mode: str = "full"  # l5.recompute_mode

    @classmethod
    def from_config(cls, cfg: Config) -> "Layer5Params":
        v = cfg.value
        return cls(v("l5.correction_source"), v("l5.observation_store_max"), v("l5.frame_correction_cov_inflation"),
                   v("l5.uncorrected_cov_inflation"), v("l5.correction_history_max"),
                   v("l5.missing_correction_timeout_s"), v("l5.tag_history_max"), v("l5.removed_tags_max"),
                   v("l5.telemetry_log_max"), v("l5.end_of_mission_commit_policy"), v("l5.recompute_mode"))


def telemetry_params(cfg: Config) -> TelemetryParams:
    v = cfg.value
    return TelemetryParams(v("l5.telemetry_snapshot_period_s"), v("l5.telemetry_max_tags_per_snapshot"),
                           bool(v("l5.telemetry_send_candidates")), v("l5.telemetry_update_min_interval_s"),
                           v("l5.telemetry_link_budget_bps"), v("l4.confirm_prob"))


@dataclass(frozen=True)
class RecomputeStat:
    arrival_ns: int
    cause: str
    n_rederived: int  # stored observations re-derived (Layers 2 + 3)
    n_carried: int  # evicted observations replayed with their old position (OBSERVATION_NOT_REDERIVABLE)
    n_batches: int  # Layer 4 update calls replayed
    rederive_ms: float
    replay_ms: float
    total_ms: float
    mode: str = "full"
    n_tracks_replayed: int = -1  # affected_tracks mode: tracks rebuilt (-1 = all)


def _eff_key(eff: EffectivePose) -> tuple:
    """Content key of an effective pose (affected_tracks mode decides 'changed' with it)."""
    p = eff.lookup.pose
    return (eff.revision, eff.flags, p.position_m.tobytes(), p.orientation_wxyz.tobytes())


@dataclass
class RebuiltState:
    """Result of a full recompute (substep 3.4): a complete new Layer 4 plus the derived observations. Built from a
    RebuildJob snapshot only; shares no mutable object with the live pipeline."""

    l4: Layer4
    derived: Dict[int, SurvivorObservation]
    keys: Dict[int, tuple]
    n_batches: int
    n_carried: int
    rederive_ms: float
    replay_ms: float
    total_ms: float


class RebuildJob:
    """Snapshot of everything a full recompute reads, taken on the pipeline's own thread by Layer5Pipeline
    (_snapshot). run() may be called on ANY thread, once: it reads only the snapshot (stored observation records are
    frozen; the effective poses were resolved at snapshot time; the evicted-observation copies, the registration and the
    Layer 4 parameters are immutable or private copies) and the pipeline's read-only configuration, and it writes only
    objects it creates. The result is installed with Layer5Pipeline.complete_rebuild(job) on the pipeline's thread
    (the atomic swap); if another rebuild was started meanwhile the result is stale and is discarded."""

    def __init__(self, cause: str, arrival_ns: int, epoch: int, batch_cut: int, grid: GridRegistration,
                 l4_params: Layer4Params, carried: List[Tuple[int, SurvivorObservation]],
                 items: List[Tuple[StoredObservation, EffectivePose]], derive: Callable):
        self.cause, self.arrival_ns, self.epoch, self.batch_cut = cause, arrival_ns, epoch, batch_cut
        self.grid, self.l4_params, self.carried, self.items = grid, l4_params, carried, items
        self._derive = derive
        self.state: Optional[RebuiltState] = None
        self.error: Optional[BaseException] = None

    def run(self) -> RebuiltState:
        t0 = time.perf_counter()
        try:
            batches: Dict[int, List[SurvivorObservation]] = {}
            keys: Dict[int, tuple] = {}
            for b, copy_obs in self.carried:
                batches.setdefault(b, []).append(copy_obs)
            derived: Dict[int, SurvivorObservation] = {}
            for rec, eff in self.items:
                obs = self._derive(rec, eff, self.grid)
                derived[rec.observation_id], keys[rec.observation_id] = obs, _eff_key(eff)
                batches.setdefault(rec.batch, []).append(obs)
            t1 = time.perf_counter()
            l4 = Layer4(self.l4_params, self.grid)
            for b in sorted(batches):
                l4.update(batches[b])
            t2 = time.perf_counter()
            self.state = RebuiltState(l4, derived, keys, len(batches), len(self.carried), (t1 - t0) * 1e3,
                                      (t2 - t1) * 1e3, (time.perf_counter() - t0) * 1e3)
            return self.state
        except BaseException as exc:  # noqa: BLE001 - re-raised to the caller, also kept for a worker thread
            self.error = exc
            raise


class Layer5Pipeline(Layer1234Pipeline):
    """Layers 1-5. Every accepted observation is stored (corrections.StoredObservation). A correction event or a
    registration update re-derives every stored observation (Layer 2 from the effective capture-time pose, Layer 3
    with the current registration) and rebuilds Layer 4 by replaying the observations in their original batches,
    so after a correction the state equals that of a run that had the corrected poses from the start (decision rule
    R1). Survivor ids, tag revisions and telemetry are handled on top (corrections.TagBook, output.TelemetryScheduler).
    Telemetry messages go to `sink(message)` if given and to the bounded `sent` log."""

    def __init__(self, cfg: Config, output_statuses=(TagStatus.PROVISIONAL, TagStatus.CONFIRMED, TagStatus.UPDATED),
                 sink: Optional[Callable[[TelemetryMessage], None]] = None, **layer4_overrides):
        super().__init__(cfg, output_statuses, **layer4_overrides)
        self.l5 = Layer5Params.from_config(cfg)
        self.store = ObservationStore(self.l5.store_max)
        self.corrections = CorrectionHistory(self.l5.history_max)
        self.book = TagBook(self.l5.tag_history_max, self.l5.removed_max)
        self.telemetry_params = telemetry_params(cfg)
        self.telemetry: Optional[TelemetryScheduler] = None  # starts at the first call (mission clock origin)
        self.sent: Deque[TelemetryMessage] = deque(maxlen=self.l5.telemetry_log_max)
        self.sink = sink
        self.derived: Dict[int, SurvivorObservation] = {}  # latest derivation per stored observation
        self._keys: Dict[int, tuple] = {}  # effective-pose key of that derivation (affected_tracks mode)
        self.outcomes: Deque[Tuple[int, CorrectionOutcome]] = deque(maxlen=64)
        self.recomputes: Deque[RecomputeStat] = deque(maxlen=256)
        self.final_notes: Tuple[str, ...] = ()
        self._carried: Dict[int, Tuple[int, SurvivorObservation]] = {}  # evicted but still tracked
        self._batch_no = 0
        self._live_revs: Dict[int, Tuple[int, int]] = {}  # unacknowledged live revision -> (first stamp, arrival)
        self._published: List[PublishedTag] = []
        self._dirty = True
        self._cause = "layer4"
        # Substep 3.4: if set, full recomputes are not run inline: a RebuildJob is handed to job_sink(job) (the caller
        # runs job.run() on a worker thread and then calls complete_rebuild(job) on this pipeline's thread).
        self.job_sink: Optional[Callable[["RebuildJob"], None]] = None
        self._epoch = 0  # number of rebuilds started; a job whose epoch is not the latest is stale
        self.publish_seq = 0  # incremented whenever the published tag list is rebuilt (substep 4.1b)
        # Substep 4.3: wall-alignment registration estimate (OFF unless grid.wall_alignment_enabled; assumes walls on
        # grid lines, open question O1).
        self.wall_align_enabled = bool(cfg.value("grid.wall_alignment_enabled"))
        self.wall_align_params = WallAlignParams.from_config(cfg)
        self.reg_min_yaw_rad = cfg.value("grid.reg_update_min_yaw_rad")
        self.reg_min_offset_m = cfg.value("grid.reg_update_min_offset_m")
        self.wall_alignments: Deque[WallAlignResult] = deque(maxlen=16)

    # --- inputs -------------------------------------------------------------------------------------------
    def process_pose(self, pose: PoseSample, arrival_ns: int) -> None:
        rev = pose.pose_revision
        if rev > max([self.corrections.baked_revision] + list(self.corrections.events)) and \
                rev not in self._live_revs and pose.failure() is None:
            self._live_revs[rev] = (pose.stamp_ns, arrival_ns)
        super().process_pose(pose, arrival_ns)
        self._after(arrival_ns)

    def process_frame(self, detections: Sequence[Detection], depth: DepthMeasurement,
                      intrinsics: CameraIntrinsics, arrival_ns: int) -> None:
        super().process_frame(detections, depth, intrinsics, arrival_ns)
        self._after(arrival_ns)

    def advance(self, arrival_ns: int) -> None:
        """Move the pipeline clock forward without new data (substep 4.1b): expire detections that waited too long for
        a pose, flush pending observations, apply the missing-correction timeout and run the telemetry schedule
        (snapshot period, rate-limited updates). `arrival_ns` is a receive/arrival time, never a capture time; it
        never moves backwards."""
        if self.now_ns is not None:
            arrival_ns = max(arrival_ns, self.now_ns)
        self.now_ns = arrival_ns
        self._expire()
        self._flush()
        self._after(arrival_ns)

    def process_correction(self, c: PoseCorrection, arrival_ns: int) -> CorrectionOutcome:
        """Apply a SLAM correction event (spec 9.3). Idempotent; out-of-order events are folded by revision."""
        self.now_ns = arrival_ns
        self._flush()
        out = self._apply_correction(c, arrival_ns)
        self.outcomes.append((arrival_ns, out))
        self._after(arrival_ns)
        return out

    def process_registration(self, reg: GridRegistration, arrival_ns: int) -> CorrectionOutcome:
        """New grid registration: Layer 3 for every stored observation and a Layer 4 replay. Versions are monotonic."""
        self.now_ns = arrival_ns
        self._flush()
        cur = self.grid
        if reg.version < cur.version:
            out = CorrectionOutcome(False, reg.version, FailureReason.CORRECTION_STALE,
                                    f"registration version {reg.version} < current {cur.version}")
        elif reg.version == cur.version:
            out = CorrectionOutcome(False, reg.version, None if reg == cur else FailureReason.CORRECTION_CONFLICT,
                                    "duplicate" if reg == cur else "same version, different registration")
        else:
            self.grid = reg
            self._rebuild(f"registration v{reg.version}", arrival_ns)
            out = CorrectionOutcome(True, reg.version)
        self.outcomes.append((arrival_ns, out))
        self._after(arrival_ns)
        return out

    def process_wall_points(self, points_map_xy, arrival_ns: int) -> CorrectionOutcome:
        """Estimate the grid registration from 2D wall points (map frame) with the CURRENT registration as the coarse
        one, and apply it through process_registration (new version, full Layer 5 recompute, revision ids) unless
        (a) grid.wall_alignment_enabled is false, (b) the estimator gives no update (reason code kept in
        `wall_alignments`), or (c) the change is below the hysteresis thresholds. Every refusal is an explicit
        CorrectionOutcome(applied=False, reason=...)."""
        self.now_ns = arrival_ns
        if not self.wall_align_enabled:
            out = CorrectionOutcome(False, self.grid.version, FailureReason.WALL_ALIGNMENT_DISABLED,
                                    "grid.wall_alignment_enabled is false")
            self.outcomes.append((arrival_ns, out))
            return out
        res = estimate_wall_alignment(points_map_xy, self.grid, self.wall_align_params)
        self.wall_alignments.append(res)
        if res.registration is None:
            out = CorrectionOutcome(False, self.grid.version, res.reason, res.detail)
        elif not update_needed(self.grid, res.registration, self.reg_min_yaw_rad, self.reg_min_offset_m):
            out = CorrectionOutcome(False, self.grid.version, FailureReason.REG_UPDATE_BELOW_THRESHOLD,
                                    f"change below hysteresis (yaw {self.reg_min_yaw_rad} rad, "
                                    f"offset {self.reg_min_offset_m} m)")
        else:
            return self.process_registration(res.registration, arrival_ns)
        self.outcomes.append((arrival_ns, out))
        return out

    # --- outputs ------------------------------------------------------------------------------------------
    @property
    def map_revision(self) -> int:
        return self.corrections.map_revision

    def published(self) -> List[PublishedTag]:
        return list(self._published)

    def observations(self) -> List[SurvivorObservation]:
        return [self.derived.get(o.observation_id, o) for o in self.obs]

    def finalize(self) -> List[TagResult]:
        Layer123Pipeline.finalize(self)
        self._flush()
        self._after(self.now_ns if self.now_ns is not None else 0)
        return [tag_result(pt.tag) for pt in self._published if pt.tag.status in self.output_statuses]

    def final_tags(self, policy: Optional[str] = None, threshold: Optional[float] = None) -> List[TagResult]:
        """End-of-mission commit set (Layer4.final_tag_set) with survivor ids. policy None = the configured
        l5.end_of_mission_commit_policy; unset -> ConfigError (O2 open, no policy is claimed as correct)."""
        policy = policy if policy is not None else self.l5.commit_policy
        if policy is None:
            raise ConfigError(f"{FailureReason.CONFIG_MISSING.value}: l5.end_of_mission_commit_policy is unset "
                              "(organiser question O2); pass a policy explicitly")
        fs = self.l4.final_tag_set(policy, threshold)
        self.final_notes = fs.notes
        pub = {pt.survivor_id: pt for pt in self._published}
        out = []
        for t in fs.tags:
            sid = self.book.assign(t.survivor_id)
            status = pub[sid].tag.status if sid in pub else t.status
            out.append(tag_result(replace(t, survivor_id=sid, status=status)))
        return out

    # --- internals ----------------------------------------------------------------------------------------
    def _accept(self, obs: SurvivorObservation, lk, capture_pose_ns: int, n_l1_warnings: int) -> None:
        rec = stored_from(obs, lk, capture_pose_ns, self._batch_no, n_l1_warnings, self.footprint is not None)
        evicted = self.store.add(rec)
        if evicted is not None:
            self.corrections.forget(evicted.observation_id)
            self._keys.pop(evicted.observation_id, None)
            old = self.derived.pop(evicted.observation_id, None)
            if old is not None:
                self._carried[evicted.observation_id] = (evicted.batch, old)
                if len(self._carried) > self.l4.p.max_tracks * self.l4.p.max_obs_per_track:
                    self._prune_carried()
        eff = self.corrections.effective(rec)
        if eff.lookup is not rec.base or eff.flags:  # e.g. captured before a frame correction, resolved after it
            obs = self._rederive(rec, eff)
        self.derived[rec.observation_id] = obs
        self._keys[rec.observation_id] = _eff_key(eff)
        self._batch.append(obs)

    def _flush(self) -> None:
        if self._batch:
            super()._flush()
            self._batch_no += 1
            self._dirty = True

    def _apply_correction(self, c: PoseCorrection, arrival_ns: int) -> CorrectionOutcome:
        bad = validate_correction(c, self.l5.correction_source, self.l2.buffer.clock_domain)
        if bad is not None:
            return CorrectionOutcome(False, c.revision, bad[0], bad[1])
        if c.revision <= self.corrections.baked_revision:
            return CorrectionOutcome(False, c.revision, FailureReason.CORRECTION_STALE,
                                     f"revision {c.revision} <= folded revision {self.corrections.baked_revision}")
        prev = self.corrections.events.get(c.revision)
        if prev is not None and prev.event is not None:
            if prev.event.same_content(c):
                return CorrectionOutcome(False, c.revision, None, "duplicate")
            return CorrectionOutcome(False, c.revision, FailureReason.CORRECTION_CONFLICT,
                                     "same revision received with different content")
        applied = AppliedCorrection(c.revision, c.source, c.stamp_ns, c.source == "history", c, {}, arrival_ns)
        if c.source == "history":
            p2 = self.l2.params
            applied.corrected = corrected_lookups(c, self.l2.buffer, list(self.store), p2.max_interp_gap_s,
                                                  p2.allow_extrapolation_s)
            oldest, newest = self.l2.buffer.oldest_ns, self.l2.buffer.newest_ns
            if oldest is not None:
                for p in c.poses:  # later lookups (pending observations) use the newest revision
                    if oldest <= p.stamp_ns <= newest:
                        self.l2.add_pose(p)
        self.corrections.add(applied)  # replaces a missing-correction marker of the same revision
        self._live_revs.pop(c.revision, None)
        self.corrections.bake_excess(self.store)
        self._rebuild(f"correction r{c.revision}", arrival_ns)
        return CorrectionOutcome(True, c.revision)

    def _check_missing(self, now_ns: int) -> None:
        for rev in sorted(self._live_revs):
            stamp, arrival = self._live_revs[rev]
            if rev <= self.corrections.baked_revision or rev in self.corrections.events:
                del self._live_revs[rev]
                continue
            if now_ns - arrival >= self.l5.missing_timeout_s * NS_PER_S:
                del self._live_revs[rev]
                self.corrections.add(AppliedCorrection(rev, "missing", stamp,
                                                       self.l5.correction_source != "frame_correction",
                                                       arrival_ns=now_ns))
                self.corrections.bake_excess(self.store)
                self._rebuild(f"missing correction r{rev}", now_ns)

    def _rederive(self, rec: StoredObservation, eff: EffectivePose,
                  grid: Optional[GridRegistration] = None) -> SurvivorObservation:
        """Layers 2 and 3 from the stored camera-frame measurement and the effective pose (same calls, same order as
        the live path in _resolve, plus the Layer 5 covariance inflation)."""
        obs = rec.fresh()
        self.l2.apply(obs, eff.lookup)
        factor = 1.0
        for code, f in ((FailureReason.FRAME_CORRECTION_APPLIED, self.l5.frame_inflation),
                        (FailureReason.POSE_NOT_CORRECTED, self.l5.uncorrected_inflation)):
            if code in eff.flags:
                obs.warnings.append(code)
                factor *= f
        if factor != 1.0:
            obs.cov_map_m2 = obs.cov_map_m2 * factor
        if self.footprint is not None:
            refine_observation(obs, pose_matrix(eff.lookup.pose) @ self.l2.extrinsic.T_body_camera, self.footprint)
        apply_grid(obs, self.grid if grid is None else grid, self.max_cells)
        return obs

    def _carry_copy(self, old: SurvivorObservation) -> SurvivorObservation:
        c = copy.copy(old)
        c.warnings = [w for w in old.warnings if w not in LAYER4_OBS_CODES]
        c.warnings.append(FailureReason.OBSERVATION_NOT_REDERIVABLE)
        c.cell_probs = dict(old.cell_probs)
        c.cov_map_m2 = old.cov_map_m2 * self.l5.uncorrected_inflation
        return c

    def _prune_carried(self) -> None:
        tracked = {it.obs.observation_id for t in self.l4.tracks.values() for it in t.items}
        self._carried = {k: v for k, v in self._carried.items() if k in tracked}

    def _rebuild(self, cause: str, arrival_ns: int) -> None:
        self._epoch += 1  # any job still in flight is now stale
        if self.l5.recompute_mode == "affected_tracks" and not cause.startswith("registration"):
            self._rebuild_affected(cause, arrival_ns)  # always inline (approximate mode)
        elif self.job_sink is not None:
            self.job_sink(self._snapshot(cause, arrival_ns))
        else:
            job = self._snapshot(cause, arrival_ns)
            job.run()
            self._install(job, catch_up=False)

    def _snapshot(self, cause: str, arrival_ns: int) -> RebuildJob:
        """Take the inputs of a full recompute (pipeline thread). Costs one effective-pose resolution per stored
        observation plus a copy of each carried observation; no Layer 2-4 work."""
        carried = [(self._carried[oid][0], self._carry_copy(self._carried[oid][1])) for oid in sorted(self._carried)]
        events = self.corrections.sorted_events()
        items = [(rec, self.corrections.effective(rec, events)) for rec in self.store]
        return RebuildJob(cause, arrival_ns, self._epoch, self._batch_no, self.grid, self.l4.p, carried, items,
                          self._rederive)

    def complete_rebuild(self, job: RebuildJob) -> bool:
        """Install the result of job.run() (pipeline thread): the atomic swap of Layer 4, the derived observations and
        the tag book. Returns False, changing nothing, if the job is stale (a newer rebuild was started) or not run.
        Observations accepted after the snapshot are replayed into the new state in their original batches first, so
        the result equals a synchronous recompute followed by those observations (limit: observations evicted from
        the store between snapshot and completion are not replayed)."""
        if job.state is None or job.epoch != self._epoch:
            return False
        self._flush()  # pending observations go to the old state, which is replaced below
        self._install(job, catch_up=True)
        return True

    def _install(self, job: RebuildJob, catch_up: bool) -> None:
        st = job.state
        l4, derived, keys = st.l4, st.derived, st.keys
        if catch_up:
            late: Dict[int, List[SurvivorObservation]] = {}
            events = self.corrections.sorted_events()
            for rec in self.store:
                if rec.batch >= job.batch_cut:
                    eff = self.corrections.effective(rec, events)
                    obs = self._rederive(rec, eff)
                    derived[rec.observation_id], keys[rec.observation_id] = obs, _eff_key(eff)
                    late.setdefault(rec.batch, []).append(obs)
            for b in sorted(late):
                l4.update(late[b])
            for rec, _ in job.items:  # evicted from the store since the snapshot: keep their derivation as carried
                if rec.observation_id not in self.store and rec.observation_id in derived:
                    self._carried[rec.observation_id] = (rec.batch, derived.pop(rec.observation_id))
                    keys.pop(rec.observation_id, None)
        self.book.remap(self.l4.tracks, l4.tracks)
        self.l4, self.derived, self._keys = l4, derived, keys
        self._prune_carried()
        self._dirty, self._cause = True, job.cause
        self.recomputes.append(RecomputeStat(job.arrival_ns, job.cause, len(job.items), st.n_carried, st.n_batches,
                                             st.rederive_ms, st.replay_ms, st.total_ms, "full", -1))

    def _rebuild_affected(self, cause: str, arrival_ns: int) -> None:
        """l5.recompute_mode = affected_tracks (R3 remedy). Observations whose effective pose changed are
        re-derived; the tracks that hold them, plus changed observations outside any track, are replayed in their
        original batches into a fresh Layer 4 (track ids continue the old counter); the other tracks are kept as
        they are, then Layer4.reconcile() merges across the two sets. Approximation: the replay does not see the kept
        tracks (no competition for association with them), so R1 equivalence is guaranteed only in 'full' mode."""
        t0 = time.perf_counter()
        effs = {rec.observation_id: (rec, self.corrections.effective(rec)) for rec in self.store}
        changed = {i for i, (_, e) in effs.items() if _eff_key(e) != self._keys.get(i)}
        if not changed:
            self.recomputes.append(RecomputeStat(arrival_ns, cause, 0, 0, 0, 0.0, 0.0,
                                                 (time.perf_counter() - t0) * 1e3, "affected_tracks", 0))
            return
        old = self.l4
        track_of = {it.obs.observation_id: tid for tid, t in old.tracks.items() for it in t.items}
        hit = {track_of[i] for i in changed if i in track_of}
        ids = set(changed) | {it.obs.observation_id for tid in hit for it in old.tracks[tid].items}
        batches: Dict[int, List[SurvivorObservation]] = {}
        n_carried = 0
        for oid in sorted(i for i in ids if i in self._carried):
            b, o = self._carried[oid]
            batches.setdefault(b, []).append(self._carry_copy(o))
            n_carried += 1
        n_rederived = 0
        for rec in self.store:
            oid = rec.observation_id
            if oid not in ids:
                continue
            eff = effs[oid][1]
            if oid in changed or oid not in self.derived:
                obs = self._rederive(rec, eff)
                n_rederived += 1
            else:  # unchanged derivation: a fresh copy without Layer 4 additions
                obs = copy.copy(self.derived[oid])
                obs.warnings = [w for w in obs.warnings if w not in LAYER4_OBS_CODES]
                obs.cell_probs = dict(obs.cell_probs)
            self.derived[oid], self._keys[oid] = obs, _eff_key(eff)
            batches.setdefault(rec.batch, []).append(obs)
        t1 = time.perf_counter()
        sub = Layer4(old.p, self.grid)
        sub.next_id = old.next_id
        for b in sorted(batches):
            sub.update(batches[b])
        new = Layer4(old.p, self.grid)
        merged = {tid: t for tid, t in old.tracks.items() if tid not in hit}
        merged.update(sub.tracks)
        new.tracks = dict(sorted(merged.items()))
        new.next_id = sub.next_id
        stamps = [x for x in (old.now_ns, sub.now_ns) if x is not None]
        new.now_ns = max(stamps) if stamps else None
        new.events.extend(old.events)
        new.events.extend(sub.events)
        new.dropped.extend(old.dropped)
        new.dropped.extend(sub.dropped)
        new.n_dropped, new.n_evicted = old.n_dropped + sub.n_dropped, old.n_evicted + sub.n_evicted
        new.reconcile()
        t2 = time.perf_counter()
        self.book.remap(old.tracks, new.tracks)
        self.l4 = new
        self._prune_carried()
        self._dirty, self._cause = True, cause
        self.recomputes.append(RecomputeStat(arrival_ns, cause, n_rederived, n_carried, len(batches),
                                             (t1 - t0) * 1e3, (t2 - t1) * 1e3, (time.perf_counter() - t0) * 1e3,
                                             "affected_tracks", len(hit)))

    def _after(self, now_ns: int) -> None:
        if self.telemetry is None:
            self.telemetry = TelemetryScheduler(self.telemetry_params, now_ns)
        self._check_missing(now_ns)
        if self._dirty:
            self._published = self.book.publish(self.l4, self._cause, self.map_revision, item_reasons)
            self.publish_seq += 1
            self._dirty, self._cause = False, "layer4"
        msgs = self.telemetry.step(now_ns, self._published, self.book.pop_removed(), self.map_revision,
                                   self.grid.version, self.grid.valid)
        for m in msgs:
            self.sent.append(m)
            if self.sink is not None:
                self.sink(m)
