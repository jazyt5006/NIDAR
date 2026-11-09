"""Layer 5: SLAM correction handling (spec 9.2, 9.3; substep 3.2). Pure library code: no simulator, scenarios,
baseline or harness imports.

Pieces (the wiring that re-runs Layers 2-4 lives in pipeline.Layer5Pipeline):
- StoredObservation / ObservationStore: for every accepted observation the pose-independent measurement (camera-frame
  point and covariance, capture stamps, Layer 1 quality and warnings, detector ids) plus the capture-time pose lookup
  it was first transformed with ("base"), the Layer 4 batch it belonged to and the extrinsic version. Bounded by
  l5.observation_store_max; eviction is oldest first (acceptance order). An evicted observation cannot be re-derived.
- CorrectionHistory: received correction events by revision (bounded by l5.correction_history_max; beyond that the
  oldest is folded into the stored base poses and later events with a revision at or below it are CORRECTION_STALE).
  effective(rec) folds the events over the base pose IN REVISION ORDER, so the result does not depend on arrival
  order:
    history           the corrected capture-time pose of this observation (computed once, when the event arrived,
                      from the event poses plus live poses of the same revision) replaces the pose if it is newer.
    frame_correction  T_new_old is applied to observations captured in [previous correction stamp, event stamp) whose
                      pose is older than the event (FRAME_CORRECTION_APPLIED, covariance inflated by the caller).
    missing           a pose_revision bump seen on the live stream with no event after
                      l5.missing_correction_timeout_s: nothing to apply, older observations are flagged.
  An observation captured before an event that leaves it at an older revision gets POSE_NOT_CORRECTED.
- TagBook: stable survivor ids across Layer 4 rebuilds (matching rule in remap), tag revisions, bounded change
  history, removal announcements.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from typing import Deque, Dict, FrozenSet, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

from .output import PublishedTag, TagChange
from .tracking import CONFIRMED_STATES, Layer4, Track
from .transforms import NS_PER_S, PoseBuffer, PoseLookup, apply_frame_jump, is_rigid
from .types import (CORRECTION_SOURCES, FAILURE_SEVERITY, FailureReason, PoseCorrection, Severity,
                    SurvivorObservation, Tag, TagStatus)

Cell = Tuple[int, int]
NOT_CORRECTED = FailureReason.POSE_NOT_CORRECTED
FRAME_APPLIED = FailureReason.FRAME_CORRECTION_APPLIED
NOT_REDERIVABLE = FailureReason.OBSERVATION_NOT_REDERIVABLE
LAYER5_OBS_CODES = (NOT_CORRECTED, FRAME_APPLIED, NOT_REDERIVABLE)
LAYER4_OBS_CODES = (FailureReason.ASSOCIATION_AMBIGUOUS,
                    FailureReason.NEAR_DUPLICATE_OBSERVATION)  # appended to observations by Layer 4 itself


# ---------------------------------------------------------------------------
# Observation store
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class StoredObservation:
    observation_id: int
    batch: int  # Layer 4 update call it was part of (replay keeps the original batching)
    detection_id: int
    confidence: float
    capture_stamp_ns: int  # image clock (as in SurvivorObservation)
    capture_pose_ns: int  # pose clock (used for every pose lookup)
    tracking_id: Optional[int]
    p_camera_m: np.ndarray
    cov_camera_m2: Optional[np.ndarray]
    ground_ref_camera_m: Optional[np.ndarray]
    n_valid_px: int
    valid_fraction: float
    depth_quality: float
    floor_removed_px: int
    cluster_points_camera_m: Optional[np.ndarray]  # kept only when footprint refinement is on
    l1_warnings: Tuple[FailureReason, ...]
    extrinsic_version: Optional[int]
    base: PoseLookup  # capture-time pose first used (or the folded pose after the oldest event was baked in)
    base_flags: FrozenSet[FailureReason] = frozenset()

    @property
    def base_revision(self) -> int:
        return _min_rev(self.base)

    def fresh(self) -> SurvivorObservation:
        """A new observation holding only the Layer 1 result (Layer 2-4 fields empty)."""
        return SurvivorObservation(
            self.observation_id, self.detection_id, self.confidence, self.capture_stamp_ns, self.tracking_id,
            p_camera_m=self.p_camera_m, cov_camera_m2=self.cov_camera_m2, ground_ref_camera_m=self.ground_ref_camera_m,
            n_valid_px=self.n_valid_px, valid_fraction=self.valid_fraction, depth_quality=self.depth_quality,
            floor_removed_px=self.floor_removed_px, cluster_points_camera_m=self.cluster_points_camera_m,
            warnings=list(self.l1_warnings))


def stored_from(obs: SurvivorObservation, lk: PoseLookup, capture_pose_ns: int, batch: int, n_l1_warnings: int,
                keep_cluster: bool) -> StoredObservation:
    return StoredObservation(
        obs.observation_id, batch, obs.detection_id, obs.confidence, obs.capture_stamp_ns, capture_pose_ns,
        obs.tracking_id, obs.p_camera_m, obs.cov_camera_m2, obs.ground_ref_camera_m, obs.n_valid_px,
        obs.valid_fraction, obs.depth_quality, obs.floor_removed_px,
        obs.cluster_points_camera_m if keep_cluster else None, tuple(obs.warnings[:n_l1_warnings]),
        obs.extrinsic_version, lk)


class ObservationStore:
    """Bounded, acceptance-ordered store. Eviction policy: oldest accepted observation first."""

    def __init__(self, max_items: int):
        self.max_items = max_items
        self._items: "OrderedDict[int, StoredObservation]" = OrderedDict()
        self.n_evicted = 0

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, obs_id: int) -> bool:
        return obs_id in self._items

    def __iter__(self) -> Iterator[StoredObservation]:
        return iter(list(self._items.values()))

    def get(self, obs_id: int) -> Optional[StoredObservation]:
        return self._items.get(obs_id)

    def add(self, rec: StoredObservation) -> Optional[StoredObservation]:
        """Returns the evicted record, if any."""
        self._items[rec.observation_id] = rec
        if len(self._items) > self.max_items:
            _, old = self._items.popitem(last=False)
            self.n_evicted += 1
            return old
        return None

    def replace(self, rec: StoredObservation) -> None:
        if rec.observation_id in self._items:
            self._items[rec.observation_id] = rec


# ---------------------------------------------------------------------------
# Correction events
# ---------------------------------------------------------------------------


def _min_rev(lk: PoseLookup) -> int:
    if lk.min_revision is not None:
        return lk.min_revision
    return lk.revision if lk.revision is not None else 0


@dataclass(eq=False)
class AppliedCorrection:
    revision: int
    source: str  # "history", "frame_correction" or "missing"
    stamp_ns: int
    affects_all_before: bool  # False: only observations captured since the previous correction (frame semantics)
    event: Optional[PoseCorrection] = None  # None for a missing-correction marker
    corrected: Dict[int, PoseLookup] = field(default_factory=dict)  # history: obs id -> corrected lookup
    arrival_ns: Optional[int] = None


@dataclass(frozen=True, eq=False)
class EffectivePose:
    lookup: PoseLookup
    revision: int
    flags: FrozenSet[FailureReason]


@dataclass(frozen=True)
class CorrectionOutcome:
    """What process_correction / process_registration did. applied=False with reason None = duplicate (no-op)."""

    applied: bool
    revision: int
    reason: Optional[FailureReason] = None
    detail: str = ""


def validate_correction(c: PoseCorrection, configured_source: Optional[str],
                        clock_domain: Optional[str]) -> Optional[Tuple[FailureReason, str]]:
    if configured_source in (None, "none"):
        return FailureReason.CORRECTION_SOURCE_UNAVAILABLE, f"l5.correction_source is {configured_source!r}"
    if c.source not in CORRECTION_SOURCES:
        return FailureReason.CORRECTION_INVALID, f"unknown source {c.source!r}"
    if c.source != configured_source:
        return FailureReason.CORRECTION_SOURCE_UNAVAILABLE, f"source {c.source!r} but configured {configured_source!r}"
    if isinstance(c.revision, bool) or not isinstance(c.revision, int) or c.revision < 1:
        return FailureReason.CORRECTION_INVALID, f"revision {c.revision!r} must be an int >= 1"
    if clock_domain is not None and c.clock_domain != clock_domain:
        return FailureReason.CLOCK_DOMAIN_UNKNOWN, f"event clock {c.clock_domain!r}, poses {clock_domain!r}"
    if c.source == "history":
        if not c.poses:
            return FailureReason.CORRECTION_INVALID, "history event without poses"
        stamps = [p.stamp_ns for p in c.poses]
        if any(b <= a for a, b in zip(stamps, stamps[1:])):
            return FailureReason.CORRECTION_INVALID, "history poses not strictly increasing in time"
        for p in c.poses:
            if p.failure() is not None or p.pose_revision != c.revision or p.clock_domain != c.clock_domain:
                return FailureReason.CORRECTION_INVALID, f"pose at {p.stamp_ns}: invalid, other revision or clock"
    else:
        if c.T_new_old is None or not is_rigid(c.T_new_old):
            return FailureReason.CORRECTION_INVALID, "frame_correction needs a rigid 4x4 T_new_old"
    return None


class CorrectionHistory:
    """Received corrections by revision (see module docstring)."""

    def __init__(self, max_events: int):
        self.max_events = max_events
        self.events: Dict[int, AppliedCorrection] = {}
        self.baked_revision = 0  # events at or below this revision were folded into the stored base poses
        self.baked_stamp_ns: Optional[int] = None  # stamp of the newest baked event (frame window start)

    def sorted_events(self) -> List[AppliedCorrection]:
        return [self.events[r] for r in sorted(self.events)]

    @property
    def map_revision(self) -> int:
        """Newest correction revision applied (markers excluded); the snapshot map_rev."""
        real = [r for r, e in self.events.items() if e.source != "missing"]
        return max(real + [self.baked_revision])

    def forget(self, obs_id: int) -> None:
        for e in self.events.values():
            e.corrected.pop(obs_id, None)

    def effective(self, rec: StoredObservation, events: Optional[Sequence[AppliedCorrection]] = None,
                  prev_stamp_ns: Optional[int] = None) -> EffectivePose:
        events = self.sorted_events() if events is None else events
        prev = self.baked_stamp_ns if prev_stamp_ns is None else prev_stamp_ns
        lk, rev, flags = rec.base, rec.base_revision, set(rec.base_flags)
        t = rec.capture_pose_ns
        for e in events:
            affected = t < e.stamp_ns and (e.affects_all_before or prev is None or t >= prev)
            if e.source == "history":
                c = e.corrected.get(rec.observation_id)
                if c is not None and rev < e.revision:
                    lk, rev = c, e.revision
                    flags.discard(NOT_CORRECTED)
                    flags.discard(FRAME_APPLIED)
            elif e.source == "frame_correction":
                if affected and rev < e.revision:
                    pose = apply_frame_jump(lk.pose, e.event.T_new_old, e.revision)
                    lk = PoseLookup(pose, None, lk.stamps_ns, lk.alpha, e.revision, lk.warnings, e.revision)
                    rev = e.revision
                    flags.add(FRAME_APPLIED)
                    flags.discard(NOT_CORRECTED)
            if affected and rev < e.revision:
                flags.add(NOT_CORRECTED)
            prev = e.stamp_ns
        return EffectivePose(lk, rev, frozenset(flags))

    def add(self, applied: AppliedCorrection) -> None:
        self.events[applied.revision] = applied

    def bake_excess(self, store: ObservationStore) -> int:
        """Fold the oldest events into the stored base poses while more than max_events are kept. Returns the number
        of events baked."""
        n = 0
        while len(self.events) > self.max_events:
            oldest = self.events[min(self.events)]
            for rec in store:
                eff = self.effective(rec, [oldest])
                store.replace(replace(rec, base=eff.lookup, base_flags=eff.flags))
            del self.events[oldest.revision]
            self.baked_revision = max(self.baked_revision, oldest.revision)
            self.baked_stamp_ns = oldest.stamp_ns
            n += 1
        return n


def corrected_lookups(c: PoseCorrection, live: PoseBuffer, records: Sequence[StoredObservation], max_gap_s: float,
                      allow_extrapolation_s: float) -> Dict[int, PoseLookup]:
    """History event: capture-time pose of every stored observation captured before the event, interpolated from
    the event poses plus live poses of the SAME revision (so the bracket at the end of the corrected range can use
    the first live pose after the closure). Observations whose bracket is not entirely of this revision are left
    out (they stay uncorrected)."""
    span_s = (max(c.poses[-1].stamp_ns, live.newest_ns or 0) - min(c.poses[0].stamp_ns, live.oldest_ns or 0)) / \
        NS_PER_S
    buf = PoseBuffer(span_s + 1.0, len(c.poses) + len(live) + 1)
    for p in c.poses:
        buf.add(p)
    for p in live.poses():
        if p.pose_revision == c.revision:
            buf.add(p)
    out: Dict[int, PoseLookup] = {}
    for rec in records:
        if rec.capture_pose_ns >= c.stamp_ns:
            continue
        lk = buf.lookup(rec.capture_pose_ns, max_gap_s, allow_extrapolation_s)
        if lk.failure is None and _min_rev(lk) == c.revision:
            out[rec.observation_id] = lk
    return out


# ---------------------------------------------------------------------------
# Survivor identity and tag revisions
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class TagState:
    survivor_id: int
    history_max: int
    tag_revision: int = 0
    confirmed_cell: Optional[Cell] = None
    updated: bool = False
    history: Deque[TagChange] = field(default_factory=deque)
    last: Optional[Tag] = None

    def __post_init__(self) -> None:
        self.history = deque(self.history, maxlen=self.history_max)


def track_obs_ids(t: Track) -> Set[int]:
    return {it.obs.observation_id for it in t.items}


class TagBook:
    """Stable survivor ids and tag revisions on top of Layer 4.

    Survivor id: assigned when a track is first published (status above CANDIDATE), from a counter that is never
    reused. After a rebuild, remap() matches old survivors to new tracks by the number of shared observation ids:
    pairs in order of (most shared observations, lower survivor id, lower new track id); each survivor and each new
    track is matched at most once. So a track that merges two old (split) survivors keeps the id of the one it
    shares more observations with; the other id is announced as removed. Between rebuilds a Layer 4 track keeps its
    id; a track Layer 4 drops or merges away is announced as removed.
    Tag revision: incremented whenever the confirmed cell of a survivor changes (live Layer 4 update or rebuild); from
    then on a confirmed tag is published as UPDATED. The last history_max changes are kept per tag."""

    def __init__(self, history_max: int, removed_max: int):
        self.history_max = history_max
        self.by_track: Dict[int, int] = {}
        self.states: Dict[int, TagState] = {}
        self.next_sid = 0
        self.removed: Deque[PublishedTag] = deque(maxlen=removed_max)  # announcements not yet consumed
        self.n_removed = 0
        self.n_cell_changes = 0  # tag revisions issued (UPDATED events), all survivors

    def survivor_id(self, track_id: int) -> Optional[int]:
        return self.by_track.get(track_id)

    def assign(self, track_id: int) -> int:
        """Survivor id of a track, assigning a new one if it has none yet (e.g. committed but never published)."""
        if track_id not in self.by_track:
            self.by_track[track_id] = self.next_sid
            self.states[self.next_sid] = TagState(self.next_sid, self.history_max)
            self.next_sid += 1
        return self.by_track[track_id]

    def remap(self, old_tracks: Dict[int, Track], new_tracks: Dict[int, Track]) -> Dict[int, int]:
        old = [(sid, track_obs_ids(old_tracks[tid])) for tid, sid in self.by_track.items() if tid in old_tracks]
        new = [(tid, track_obs_ids(t)) for tid, t in new_tracks.items()]
        pairs = sorted(((len(o & n), sid, tid) for sid, o in old for tid, n in new if o & n),
                       key=lambda p: (-p[0], p[1], p[2]))
        mapping: Dict[int, int] = {}
        used: Set[int] = set()
        for _, sid, tid in pairs:
            if sid in used or tid in mapping:
                continue
            mapping[tid] = sid
            used.add(sid)
        self.by_track = mapping
        return mapping

    def publish(self, l4: Layer4, cause: str, map_rev: int, extra_reasons) -> List[PublishedTag]:
        """Current tags with survivor ids and Layer 5 status. extra_reasons(track) -> tuple of item-level codes."""
        tags = {t.survivor_id: t for t in l4.tags()}  # Layer 4 Tag.survivor_id is its track id
        for tid in list(self.by_track):
            if tid not in tags:
                del self.by_track[tid]
        for tid in sorted(tags):
            if tid not in self.by_track and tags[tid].status is not TagStatus.CANDIDATE:
                self.by_track[tid] = self.next_sid
                self.states[self.next_sid] = TagState(self.next_sid, self.history_max)
                self.next_sid += 1
        active = set(self.by_track.values())
        for sid in sorted(self.states):
            if sid not in active:
                st = self.states.pop(sid)
                if st.last is not None:
                    self.removed.append(PublishedTag(sid, st.tag_revision, st.last, True, tuple(st.history)))
                    self.n_removed += 1
        out = []
        for tid, sid in sorted(self.by_track.items(), key=lambda kv: kv[1]):
            tag, st = tags[tid], self.states[sid]
            status = tag.status
            if status in CONFIRMED_STATES and tag.cell is not None:
                if st.confirmed_cell is not None and tag.cell != st.confirmed_cell:
                    st.tag_revision += 1
                    st.updated = True
                    self.n_cell_changes += 1
                    st.history.append(TagChange(tag.update_stamp_ns, st.confirmed_cell, tag.cell, cause, map_rev,
                                                tag.registration_version, st.tag_revision))
                st.confirmed_cell = tag.cell
                if st.updated:
                    status = TagStatus.UPDATED
            reasons = tuple(tag.reasons) + tuple(r for r in extra_reasons(l4.tracks[tid]) if r not in tag.reasons)
            pub = replace(tag, survivor_id=sid, status=status, reasons=reasons)
            st.last = pub
            out.append(PublishedTag(sid, st.tag_revision, pub, False, tuple(st.history)))
        return out

    def pop_removed(self) -> List[PublishedTag]:
        out = list(self.removed)
        self.removed.clear()
        return out


def item_reasons(t: Track) -> Tuple[FailureReason, ...]:
    """Observation-level codes that the tag should carry: Layer 5 codes and BLOCK-severity codes."""
    seen: List[FailureReason] = []
    for it in t.items:
        for w in it.obs.warnings:
            if (w in LAYER5_OBS_CODES or FAILURE_SEVERITY[w] is Severity.BLOCK) and w not in seen:
                seen.append(w)
    return tuple(seen)
