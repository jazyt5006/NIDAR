"""Layer 4: temporal consistency and uncertainty (spec Part 8; architecture.md "Layer 4").

Pure library code: no simulator, scenarios, baseline or harness imports.

Per frame batch (observations sharing one capture stamp, already through Layers 1-3):
1. Association: Mahalanobis gate in the map plane (innovation S = P_track + R_obs + ray bias), one observation per
   track per frame, optimal assignment (Hungarian). The detector tracking_id only lowers the cost (a hint).
   An observation whose best track has an unassigned rival within l4.assoc_ambiguity_margin is ASSOCIATION_AMBIGUOUS
   and is kept out of fusion. Ungated observations start new tracks.
   Each observation's covariance gets an isotropic model-error term l4.obs_model_sigma_m (frame-to-frame variation of
   where on the body the point lands, which the Layer 1 measurement covariance does not describe).
2. Fusion (de-correlated): observations are grouped into view bins (bearing from survivor to camera, relative to the
   track's first bearing, l4.view_bin_deg wide; a silence longer than l4.view_bin_max_gap_s starts a new bin).
   Within a bin: information-weighted mean, but the noise is reduced by at most l4.max_effective_obs_per_bin
   observations (correlated frames), plus a common-mode bias sigma l4.ray_bias_sigma_m along the mean camera ray.
   Bins are then fused by information weighting. Robustness: a wild observation fails the association gate, so it
   never enters an existing track's fusion; it starts a candidate track that cannot confirm and times out.
3. Cell probabilities from the fused Gaussian (grid.cell_probabilities, includes the out-of-grid mass).
4. Status state machine (LEGAL_TRANSITIONS), hysteresis for cell changes, UNRELIABLE on contradiction/ambiguity.
5. Merges (consistent, never seen in one frame), proximity flags, candidate timeouts, bounded memory with eviction.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .config import Config
from .grid import CellProbabilities, GridRegistration, cell_probabilities
from .types import FAILURE_SEVERITY, FailureReason, Severity, SurvivorObservation, Tag, TagStatus

S = TagStatus
Cell = Tuple[int, int]
CONFIRMED_STATES = (S.CONFIRMED, S.UPDATED)
LEGAL_TRANSITIONS: Dict[TagStatus, Set[TagStatus]] = {
    S.CANDIDATE: {S.PROVISIONAL, S.UNRELIABLE},
    S.PROVISIONAL: {S.CONFIRMED, S.UNRELIABLE},
    S.CONFIRMED: {S.UPDATED, S.PROVISIONAL, S.UNRELIABLE},
    S.UPDATED: {S.UPDATED, S.PROVISIONAL, S.UNRELIABLE},
    S.UNRELIABLE: {S.PROVISIONAL},
}
STATUS_RANK = {S.CANDIDATE: 0, S.UNRELIABLE: 1, S.PROVISIONAL: 2, S.CONFIRMED: 3, S.UPDATED: 3}
BIG = 1e12
COV_EPS = 1e-8  # m^2, keeps 2x2 covariances invertible
NS_PER_S = 1_000_000_000


class IllegalTransition(RuntimeError):
    pass


PENALTY_UNKNOWN_NOTE = ("score.wrong_tag_penalty_assumed is unset (O2 unknown): expected_value threshold 0.5 used; "
                        "conservative if the real penalty is below the correct-tag points, over-commits if above")


def expected_value_threshold(wrong_penalty: Optional[float], correct_points: float) -> Tuple[float, str]:
    """Commit a tag iff its expected points are non-negative: p * C - (1 - p) * P >= 0  <=>  p >= P / (C + P), with C
    the points for a correct tag and P (> = 0) the penalty for a wrong or extra tag. P unknown -> 0.5 (the value for
    P = C) and a note. For P < C, 0.5 is conservative (commits fewer than optimal); for P > C, 0.5 over-commits."""
    if wrong_penalty is None:
        return 0.5, PENALTY_UNKNOWN_NOTE
    if wrong_penalty < 0 or correct_points <= 0:
        raise ValueError("wrong_penalty must be >= 0 and correct_points > 0")
    return wrong_penalty / (correct_points + wrong_penalty), ""


@dataclass(frozen=True)
class FinalTagSet:
    tags: list  # List[Tag]
    policy: str
    threshold: Optional[float]  # expected_value only
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Layer4Params:
    gate_chi2: float
    max_assoc_dist_m: float
    ambiguity_margin: float
    id_hint_bonus: float
    merge_gate_chi2: float
    min_separation_m: float
    view_bin_deg: float
    view_bin_max_gap_s: float
    obs_model_sigma_m: float
    ray_bias_sigma_m: float
    max_eff_obs_per_bin: int
    provisional_min_obs: int
    confirm_min_obs: int
    confirm_min_view_bins: int
    confirm_min_span_s: float
    confirm_prob: float
    cell_change_margin: float
    cell_change_updates: int
    unreliable_nis: float
    nis_window: int
    unreliable_ambiguous_streak: int
    candidate_timeout_s: float
    max_obs_per_track: int
    max_tracks: int
    max_candidate_cells: int
    offset_sigma_m: float = 0.0  # l1.body_depth_offset_sigma_m (substep 3.1c)
    commit_min_obs: int = 3  # l5.commit_min_obs
    correct_points: float = 40.0  # score.correct_tag_points
    wrong_penalty: Optional[float] = None  # score.wrong_tag_penalty_assumed (None = unknown, O2)

    @property
    def ray_sigma_m(self) -> float:
        """Per-view-bin ray-bias sigma actually used: sqrt(ray_bias_sigma_m^2 + offset_sigma_m^2). ray_bias_sigma_m
        is the floor WITHOUT the body-offset uncertainty; offset_sigma_m is only the uncertainty of the configured
        body-depth offset. Each error source is counted once."""
        return math.hypot(self.ray_bias_sigma_m, self.offset_sigma_m)

    @classmethod
    def from_config(cls, cfg: Config, **overrides) -> "Layer4Params":
        v = cfg.value
        vals = dict(
            gate_chi2=v("l4.assoc_gate_mahalanobis"), max_assoc_dist_m=v("l4.assoc_max_distance_m"),
            ambiguity_margin=v("l4.assoc_ambiguity_margin"), id_hint_bonus=v("l4.track_id_hint_bonus"),
            merge_gate_chi2=v("l4.merge_gate_mahalanobis"), min_separation_m=v("l4.min_separate_survivor_distance_m"),
            view_bin_deg=v("l4.view_bin_deg"), view_bin_max_gap_s=v("l4.view_bin_max_gap_s"),
            obs_model_sigma_m=v("l4.obs_model_sigma_m"), ray_bias_sigma_m=v("l4.ray_bias_sigma_m"), max_eff_obs_per_bin=v("l4.max_effective_obs_per_bin"),
            provisional_min_obs=v("l4.provisional_min_obs"), confirm_min_obs=v("l4.confirm_min_obs"),
            confirm_min_view_bins=v("l4.confirm_min_view_bins"), confirm_min_span_s=v("l4.confirm_min_span_s"),
            confirm_prob=v("l4.confirm_prob"), cell_change_margin=v("l4.cell_change_margin"),
            cell_change_updates=v("l4.cell_change_updates"), unreliable_nis=v("l4.unreliable_nis"),
            nis_window=v("l4.nis_window"), unreliable_ambiguous_streak=v("l4.unreliable_ambiguous_streak"),
            candidate_timeout_s=v("l4.candidate_timeout_s"), max_obs_per_track=v("l4.track_history_max"),
            max_tracks=v("l4.max_tracks"), max_candidate_cells=v("grid.max_candidate_cells"),
            offset_sigma_m=v("l1.body_depth_offset_sigma_m"), commit_min_obs=v("l5.commit_min_obs"),
            correct_points=v("score.correct_tag_points"), wrong_penalty=v("score.wrong_tag_penalty_assumed"),
        )
        vals.update(overrides)
        return cls(**vals)


# ---------------------------------------------------------------------------
# Optimal assignment
# ---------------------------------------------------------------------------


def hungarian(cost: np.ndarray) -> List[Tuple[int, int]]:
    """Minimum-cost assignment for a rectangular matrix (Kuhn-Munkres with row/column potentials, O(n^2 m) for
    n <= m; the matrix is transposed if needed). Returns (row, col) pairs, one per row of the smaller side. Entries
    of BIG mean "not allowed"; callers drop pairs whose cost is BIG."""
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return []
    transpose = cost.shape[0] > cost.shape[1]
    a = cost.T if transpose else cost
    n, m = a.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p, way = np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = np.full(m + 1, np.inf), np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    pairs = [(int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]
    return [(c, r) for r, c in pairs] if transpose else pairs


# ---------------------------------------------------------------------------
# Track data and fusion
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class TrackObs:
    obs: SurvivorObservation  # reference kept for Layer 5 (pose_revision, camera-frame measurement, ...)
    xy: np.ndarray  # (2,) map
    R: np.ndarray  # (2, 2) map covariance
    ray: np.ndarray  # (2,) unit vector camera -> observation, map plane
    bearing_deg: float  # survivor -> camera, map frame
    stamp_ns: int
    nis: Optional[float] = None  # normalised innovation when it was associated


def _bias(ray: np.ndarray, sigma: float) -> np.ndarray:
    return sigma * sigma * np.outer(ray, ray)


def view_bins(items: Sequence[TrackObs], bin_deg: float, max_gap_s: float) -> Dict[Tuple[int, int], List[TrackObs]]:
    """Bins relative to the first observation's bearing (so the first view sits in the centre of bin 0); a silence
    longer than max_gap_s within one bearing bin starts a new visit (bin key = (bearing bin, visit))."""
    out: Dict[Tuple[int, int], List[TrackObs]] = {}
    if not items:
        return out
    items = sorted(items, key=lambda it: it.stamp_ns)
    ref = items[0].bearing_deg
    last: Dict[int, int] = {}
    visit: Dict[int, int] = {}
    for it in items:
        b = int(math.floor(((it.bearing_deg - ref + bin_deg / 2) % 360.0) / bin_deg))
        if b in last and (it.stamp_ns - last[b]) > max_gap_s * NS_PER_S:
            visit[b] = visit.get(b, 0) + 1
        last[b] = it.stamp_ns
        out.setdefault((b, visit.get(b, 0)), []).append(it)
    return out


def fuse(items: Sequence[TrackObs], p: Layer4Params) -> Tuple[np.ndarray, np.ndarray, int]:
    """Two-stage fusion (module docstring). Returns (x, P, number of view bins)."""
    bins = view_bins(items, p.view_bin_deg, p.view_bin_max_gap_s)
    info, info_vec = np.zeros((2, 2)), np.zeros(2)
    for members in bins.values():
        r_inv = [np.linalg.inv(m.R) for m in members]
        c_avg = np.linalg.inv(sum(r_inv))
        z_b = c_avg @ sum(ri @ m.xy for ri, m in zip(r_inv, members))
        n = len(members)
        c_noise = c_avg * n / min(n, p.max_eff_obs_per_bin)  # correlated frames: floor at R / max_eff
        ray = np.mean([m.ray for m in members], axis=0)
        ray = ray / max(np.linalg.norm(ray), 1e-12)
        c_bin_inv = np.linalg.inv(c_noise + _bias(ray, p.ray_sigma_m))
        info += c_bin_inv
        info_vec += c_bin_inv @ z_b
    cov = np.linalg.inv(info)
    return cov @ info_vec, 0.5 * (cov + cov.T), len(bins)


@dataclass(eq=False)
class Track:
    track_id: int
    created_ns: int
    status: TagStatus = S.CANDIDATE
    items: List[TrackObs] = field(default_factory=list)
    x: Optional[np.ndarray] = None
    P: Optional[np.ndarray] = None
    z_m: float = 0.0
    n_bins: int = 0
    probs: Optional[CellProbabilities] = None
    first_seen_ns: int = 0
    last_seen_ns: int = 0
    n_frames: int = 0  # observations fused over the track's life (evicted ones included)
    max_confidence: float = 0.0
    confirmed_cell: Optional[Cell] = None
    confirmed_ns: Optional[int] = None
    ever_confirmed: bool = False
    change_streak: int = 0
    cell_changes: int = 0
    revision: int = 0
    ambiguous_streak: int = 0
    last_tracking_id: Optional[int] = None
    warnings: List[FailureReason] = field(default_factory=list)
    history: List[Tuple[int, str, str]] = field(default_factory=list)  # (stamp_ns, event, detail)

    def stamps(self) -> Set[int]:
        return {it.stamp_ns for it in self.items}

    def top(self) -> Tuple[Optional[Cell], float]:
        if self.probs is None or not self.probs.cells:
            return None, 0.0
        return self.probs.cells[0]

    def prob_of(self, cell: Optional[Cell]) -> float:
        return 0.0 if self.probs is None or cell is None else self.probs.as_dict().get(cell, 0.0)

    def warn(self, w: FailureReason) -> None:
        if w not in self.warnings:
            self.warnings.append(w)


# ---------------------------------------------------------------------------
# Layer 4
# ---------------------------------------------------------------------------


class Layer4:
    def __init__(self, params: Layer4Params, registration: GridRegistration):
        self.p = params
        self.reg = registration
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.now_ns: Optional[int] = None
        self.events: Deque[Tuple[int, str, int, str]] = deque(maxlen=1000)  # (stamp, code, track id, detail)
        self.dropped: Deque[Track] = deque(maxlen=100)  # most recent dropped/merged tracks (bounded)
        self.n_dropped = 0
        self.n_evicted = 0

    # --- public -------------------------------------------------------------------------------------------
    def update(self, batch: Sequence[SurvivorObservation]) -> None:
        """Process observations (any order); grouped by capture stamp, oldest frame first."""
        usable = [o for o in batch if o.failure is None and o.p_map_m is not None and o.cov_map_m2 is not None
                  and o.camera_map_m is not None]
        for stamp in sorted({o.capture_stamp_ns for o in usable}):
            self._update_frame([o for o in usable if o.capture_stamp_ns == stamp], stamp)
        if usable:
            self._timeouts()
            self._evict_tracks()

    def advance(self, now_ns: int) -> None:
        """Move time forward without observations (applies timeouts)."""
        self.now_ns = now_ns if self.now_ns is None else max(self.now_ns, now_ns)
        self._timeouts()

    def tags(self) -> List[Tag]:
        return [self._tag(t) for t in sorted(self.tracks.values(), key=lambda t: t.track_id) if t.x is not None]

    def final_tags(self, policy: str, threshold: Optional[float] = None) -> List[Tag]:
        return self.final_tag_set(policy, threshold).tags

    def final_tag_set(self, policy: str, threshold: Optional[float] = None) -> "FinalTagSet":
        """End-of-mission commit set. "confirmed_only": CONFIRMED/UPDATED. "confirmed_plus_provisional_top1": also
        PROVISIONAL tracks with their top cell. "expected_value": every track that is not UNRELIABLE and has at
        least commit_min_obs observations, committed to its TOP cell if p_top >= threshold (default
        expected_value_threshold(wrong_penalty, correct_points)). Which policy scores better depends on O2."""
        notes: List[str] = []
        if policy == "confirmed_only":
            keep = CONFIRMED_STATES
        elif policy == "confirmed_plus_provisional_top1":
            keep = CONFIRMED_STATES + (S.PROVISIONAL,)
        elif policy == "expected_value":
            if threshold is None:
                threshold, note = expected_value_threshold(self.p.wrong_penalty, self.p.correct_points)
                if note:
                    notes.append(note)
            out = []
            for t in sorted(self.tracks.values(), key=lambda t: t.track_id):
                cell, prob = t.top()
                if t.x is None or t.status is S.UNRELIABLE or t.n_frames < self.p.commit_min_obs or cell is None:
                    continue
                if prob >= threshold:
                    tag = self._tag(t)
                    out.append(replace(tag, cell=cell, localisation_quality=prob,
                                       cell_label=self.reg.cell_label(cell) if self.reg.numbering != "alnum"
                                       else None))
            return FinalTagSet(out, policy, threshold, tuple(notes))
        else:
            raise ValueError(f"unknown final-tag policy {policy!r}")
        return FinalTagSet([t for t in self.tags() if t.status in keep and t.cell is not None], policy, None, ())

    # --- per frame ----------------------------------------------------------------------------------------
    def _make_item(self, o: SurvivorObservation) -> TrackObs:
        xy = np.asarray(o.p_map_m, dtype=float)[:2]
        cam = np.asarray(o.camera_map_m, dtype=float)[:2]
        ray = xy - cam
        ray = ray / max(np.linalg.norm(ray), 1e-12)
        bearing = math.degrees(math.atan2(cam[1] - xy[1], cam[0] - xy[0])) % 360.0
        r = np.asarray(o.cov_map_m2, dtype=float)[:2, :2] + (self.p.obs_model_sigma_m ** 2 + COV_EPS) * np.eye(2)
        return TrackObs(o, xy, 0.5 * (r + r.T), ray, bearing, o.capture_stamp_ns)

    def _innovation(self, t: Track, it: TrackObs) -> float:
        s = t.P + it.R + _bias(it.ray, self.p.ray_sigma_m)
        d = it.xy - t.x
        return float(d @ np.linalg.solve(s, d))

    def _update_frame(self, obs: List[SurvivorObservation], stamp: int) -> None:
        self.now_ns = stamp if self.now_ns is None else max(self.now_ns, stamp)
        items = [self._make_item(o) for o in obs]
        tracks = [t for t in self.tracks.values() if t.x is not None]
        nt = len(tracks)
        d2 = np.full((len(items), nt), BIG)
        cost = np.full((len(items), nt), BIG)
        for i, it in enumerate(items):
            for j, t in enumerate(tracks):
                if stamp in t.stamps() or np.linalg.norm(it.xy - t.x) > self.p.max_assoc_dist_m:
                    continue
                v = self._innovation(t, it)
                if v <= self.p.gate_chi2:
                    d2[i, j] = v
                    hint = it.obs.tracking_id is not None and it.obs.tracking_id == t.last_tracking_id
                    cost[i, j] = v - (self.p.id_hint_bonus if hint else 0.0)
        assigned = {i: j for i, j in hungarian(cost) if cost[i, j] < BIG / 2}
        taken = set(assigned.values())
        touched: Dict[int, Track] = {}
        for i, it in enumerate(items):
            if i in assigned:
                j = assigned[i]
                rivals = [k for k in range(nt) if k != j and k not in taken and cost[i, k] < BIG / 2
                          and cost[i, k] - cost[i, j] < self.p.ambiguity_margin]
                if rivals:
                    it.obs.warnings.append(FailureReason.ASSOCIATION_AMBIGUOUS)
                    for k in [j] + rivals:
                        tr = tracks[k]
                        tr.ambiguous_streak += 1
                        tr.history.append((stamp, "ambiguous_observation", f"obs {it.obs.observation_id}"))
                        touched[tr.track_id] = tr
                    continue
                t = tracks[j]
                it.nis = float(d2[i, j])
                self._add(t, it)
                t.ambiguous_streak = 0
                touched[t.track_id] = t
            else:
                t = self._new_track(it)
                touched[t.track_id] = t
        for t in touched.values():
            self._refuse(t)
        self._flag_proximity([t for t in touched.values() if stamp in t.stamps()], stamp)
        for t in self._merge(stamp):
            touched[t.track_id] = t
        for t in touched.values():
            if t.track_id in self.tracks:
                self._update_status(t)

    def _new_track(self, it: TrackObs) -> Track:
        t = Track(self.next_id, it.stamp_ns, first_seen_ns=it.stamp_ns)
        self.next_id += 1
        tid = it.obs.tracking_id
        for other in self.tracks.values():
            if tid is not None and other.last_tracking_id == tid:
                t.history.append((it.stamp_ns, "split_from", f"track {other.track_id} (same detector id {tid})"))
                other.history.append((it.stamp_ns, "split_to", f"track {t.track_id} (same detector id {tid})"))
        t.history.append((it.stamp_ns, "created", f"obs {it.obs.observation_id}"))
        self.tracks[t.track_id] = t
        self._add(t, it)
        return t

    def _add(self, t: Track, it: TrackObs) -> None:
        tid = it.obs.tracking_id
        if t.last_tracking_id is not None and tid is not None and tid != t.last_tracking_id:
            t.history.append((it.stamp_ns, "detector_id_changed", f"{t.last_tracking_id} -> {tid}"))
        if tid is not None:
            t.last_tracking_id = tid
        t.items.append(it)
        t.n_frames += 1
        t.first_seen_ns = min(t.first_seen_ns, it.stamp_ns)
        t.last_seen_ns = max(t.last_seen_ns, it.stamp_ns)
        t.max_confidence = max(t.max_confidence, it.obs.confidence)
        if len(t.items) > self.p.max_obs_per_track:
            self._evict_observation(t)

    def _evict_observation(self, t: Track) -> None:
        """Lowest value first: the oldest observation of the most populated view bin (keeps view diversity)."""
        bins = view_bins(t.items, self.p.view_bin_deg, self.p.view_bin_max_gap_s)
        biggest = max(bins.values(), key=len)
        victim = min(biggest, key=lambda it: it.stamp_ns)
        t.items.remove(victim)
        t.warn(FailureReason.CAPACITY_EXCEEDED)
        t.history.append((self.now_ns, "observation_evicted", f"obs {victim.obs.observation_id}"))
        self.events.append((self.now_ns, FailureReason.CAPACITY_EXCEEDED.value, t.track_id, "observation"))

    def _refuse(self, t: Track) -> None:
        t.x, t.P, t.n_bins = fuse(t.items, self.p)
        t.z_m = float(np.median([np.asarray(it.obs.p_map_m)[2] for it in t.items]))
        t.probs = cell_probabilities(self.reg, t.x, t.P, self.p.max_candidate_cells)

    def _flag_proximity(self, tracks: List[Track], stamp: int) -> None:
        for a in range(len(tracks)):
            for b in range(a + 1, len(tracks)):
                ta, tb = tracks[a], tracks[b]
                if np.linalg.norm(ta.x - tb.x) < self.p.min_separation_m:
                    for t, o in ((ta, tb), (tb, ta)):
                        t.warn(FailureReason.ASSOCIATION_AMBIGUOUS)
                        t.history.append((stamp, "too_close_same_frame", f"track {o.track_id}"))

    def _merge(self, stamp: int) -> List[Track]:
        """Merge statistically consistent tracks that were never observed in the same frame. The survivor keeps the
        higher status (ties: the older track)."""
        changed: List[Track] = []
        merged = True
        while merged:
            merged = False
            ts = sorted((t for t in self.tracks.values() if t.x is not None), key=lambda t: t.track_id)
            for a in range(len(ts)):
                for b in range(a + 1, len(ts)):
                    ta, tb = ts[a], ts[b]
                    if ta.stamps() & tb.stamps():
                        continue
                    d = ta.x - tb.x
                    if np.linalg.norm(d) > self.p.max_assoc_dist_m:
                        continue
                    if float(d @ np.linalg.solve(ta.P + tb.P, d)) > self.p.merge_gate_chi2:
                        continue
                    keep, gone = (ta, tb) if STATUS_RANK[ta.status] >= STATUS_RANK[tb.status] else (tb, ta)
                    for it in gone.items:
                        self._add_merged(keep, it)
                    keep.n_frames += gone.n_frames - len(gone.items)
                    keep.history.append((stamp, "merged", f"track {gone.track_id}"))
                    gone.history.append((stamp, "merged_into", f"track {keep.track_id}"))
                    keep.warn(FailureReason.TRACK_DUPLICATE_MERGED)
                    self.events.append((stamp, FailureReason.TRACK_DUPLICATE_MERGED.value, keep.track_id,
                                        f"absorbed {gone.track_id}"))
                    del self.tracks[gone.track_id]
                    self.dropped.append(gone)
                    self._refuse(keep)
                    changed.append(keep)
                    merged = True
                    break
                if merged:
                    break
        return changed

    def _add_merged(self, t: Track, it: TrackObs) -> None:
        t.items.append(it)
        t.n_frames += 1
        t.first_seen_ns = min(t.first_seen_ns, it.stamp_ns)
        t.last_seen_ns = max(t.last_seen_ns, it.stamp_ns)
        t.max_confidence = max(t.max_confidence, it.obs.confidence)
        if len(t.items) > self.p.max_obs_per_track:
            self._evict_observation(t)

    # --- status -------------------------------------------------------------------------------------------
    def transition(self, t: Track, new: TagStatus, why: str) -> None:
        if new not in LEGAL_TRANSITIONS[t.status]:
            raise IllegalTransition(f"track {t.track_id}: {t.status.name} -> {new.name} ({why})")
        t.history.append((self.now_ns, f"{t.status.name}->{new.name}", why))
        t.status = new

    def blocked(self, t: Track) -> bool:
        if not self.reg.valid:
            return True
        return any(FAILURE_SEVERITY[w] is Severity.BLOCK for w in t.warnings) or any(
            FAILURE_SEVERITY[w] is Severity.BLOCK for it in t.items for w in it.obs.warnings)

    def unreliable_reason(self, t: Track) -> Optional[FailureReason]:
        nis = [it.nis for it in sorted(t.items, key=lambda it: it.stamp_ns) if it.nis is not None]
        nis = nis[-self.p.nis_window:]
        if len(nis) >= self.p.nis_window and float(np.mean(nis)) > self.p.unreliable_nis:
            return FailureReason.TRACK_INCONSISTENT
        if t.ambiguous_streak >= self.p.unreliable_ambiguous_streak:
            return FailureReason.ASSOCIATION_AMBIGUOUS
        return None

    def confirmable(self, t: Track) -> bool:
        cell, prob = t.top()
        return (t.n_frames >= self.p.confirm_min_obs and t.n_bins >= self.p.confirm_min_view_bins
                and (t.last_seen_ns - t.first_seen_ns) >= self.p.confirm_min_span_s * NS_PER_S
                and cell is not None and prob >= self.p.confirm_prob and not self.blocked(t))

    def _update_status(self, t: Track) -> None:
        reason = self.unreliable_reason(t)
        if reason is not None:
            t.warn(reason)
            if t.status is not S.UNRELIABLE:
                self.transition(t, S.UNRELIABLE, reason.value)
            return
        if t.status is S.UNRELIABLE:
            self.transition(t, S.PROVISIONAL, "consistent again")
        if t.status is S.CANDIDATE and t.n_frames >= self.p.provisional_min_obs:
            self.transition(t, S.PROVISIONAL, f"{t.n_frames} observations")
        if t.status is S.PROVISIONAL:
            if self.confirmable(t):
                cell, prob = t.top()
                if t.ever_confirmed and cell != t.confirmed_cell:
                    t.cell_changes += 1
                t.confirmed_cell, t.confirmed_ns, t.ever_confirmed = cell, self.now_ns, True
                t.change_streak = 0
                self.transition(t, S.CONFIRMED, f"cell {cell} p={prob:.3f}")
            return
        if t.status in CONFIRMED_STATES:
            if self.blocked(t):
                self.transition(t, S.PROVISIONAL, "blocking warning")
                return
            cell, prob = t.top()
            if cell is not None and cell != t.confirmed_cell and prob > t.prob_of(t.confirmed_cell) + \
                    self.p.cell_change_margin:
                t.change_streak += 1
            else:
                t.change_streak = 0
            if t.change_streak >= self.p.cell_change_updates:
                old = t.confirmed_cell
                t.confirmed_cell, t.change_streak = cell, 0
                t.cell_changes += 1
                t.revision += 1
                self.transition(t, S.UPDATED, f"cell {old} -> {cell} p={prob:.3f}")

    # --- housekeeping ---------------------------------------------------------------------------------------
    def _timeouts(self) -> None:
        """Never-confirmed tracks with fewer than confirm_min_obs frames are dropped after candidate_timeout_s of
        silence (false-positive suppression)."""
        if self.now_ns is None:
            return
        for t in list(self.tracks.values()):
            if (not t.ever_confirmed and t.n_frames < self.p.confirm_min_obs
                    and self.now_ns - t.last_seen_ns > self.p.candidate_timeout_s * NS_PER_S):
                t.history.append((self.now_ns, "dropped", "timeout"))
                del self.tracks[t.track_id]
                self.dropped.append(t)
                self.n_dropped += 1

    def _evict_tracks(self) -> None:
        """Lowest value first: status rank, then fewest frames, then oldest last observation."""
        while len(self.tracks) > self.p.max_tracks:
            victim = min(self.tracks.values(), key=lambda t: (STATUS_RANK[t.status], t.n_frames, t.last_seen_ns))
            victim.history.append((self.now_ns, "evicted", "max_tracks"))
            self.events.append((self.now_ns, FailureReason.CAPACITY_EXCEEDED.value, victim.track_id, "track"))
            del self.tracks[victim.track_id]
            self.dropped.append(victim)
            self.n_evicted += 1

    # --- output ---------------------------------------------------------------------------------------------
    def _tag(self, t: Track) -> Tag:
        confirmed = t.status in CONFIRMED_STATES
        cell = t.confirmed_cell if confirmed else t.top()[0]
        g = self.reg.map_to_grid(t.x)
        cov3 = np.zeros((3, 3))
        cov3[:2, :2] = t.P
        rev = max((it.obs.pose_revision or 0) for it in t.items) if t.items else 0
        ext = [it.obs.extrinsic_version for it in t.items if it.obs.extrinsic_version is not None]
        return Tag(
            survivor_id=t.track_id, status=t.status, cell=cell,
            cell_label=self.reg.cell_label(cell) if cell is not None and self.reg.numbering != "alnum" else None,
            cell_probs=t.probs.cells if t.probs else (), p_map_m=np.array([t.x[0], t.x[1], t.z_m]),
            cov_map_m2=cov3, p_grid_m=g, sigma_horizontal_m=float(math.sqrt(max(np.linalg.eigvalsh(t.P)))),
            boundary_distance_m=self.reg.boundary_distance_m(g), out_of_grid=bool(self.reg.is_out_of_grid(g)),
            max_detection_confidence=t.max_confidence, localisation_quality=t.prob_of(cell),
            observation_count=t.n_frames, effective_independent_count=float(t.n_bins),
            first_seen_ns=t.first_seen_ns, last_seen_ns=t.last_seen_ns, capture_stamp_ns=t.last_seen_ns,
            update_stamp_ns=self.now_ns or t.last_seen_ns, map_revision=rev, registration_version=self.reg.version,
            extrinsic_version=max(ext) if ext else None, reasons=tuple(t.warnings),
            confirmed_stamp_ns=t.confirmed_ns if confirmed else None, cell_changes=t.cell_changes,
        )
