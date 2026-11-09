"""Scenario runner and scoring (docs/design/harness.md).

A pipeline receives poses and frames in arrival order through the Pipeline protocol and returns its tags from
finalize(). Optional methods: process_correction(PoseCorrection, arrival_ns) and
process_registration_event(sim.RegistrationUpdate, arrival_ns) (substep 3.2); pipelines without them ignore those
events. Scoring compares tags with simulator ground truth. A point estimator (Layer 1 candidate) can also be
scored on its own in the camera frame, independently of poses, transforms and the grid.
All results are SIMULATION tested on synthetic data.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from .sim import NS_PER_S, Event, Frame, Scenario, SurvivorTruth, build_events, place_survivors
from .types import (CameraIntrinsics, DepthMeasurement, Detection, FailureReason, PoseSample,
                    SurvivorObservation, TagResult, TagStatus)

MATCH_RADIUS_M = 1.0
FALSE_ASSOC_DIST_M = 0.5  # camera-frame metric: estimate farther than this from the survivor's box = wrong surface
HORIZONTAL_THRESHOLDS_M = (0.15, 0.25, 0.5)  # camera-frame metric: share of horizontal errors below each


class Pipeline(Protocol):
    def process_pose(self, pose: PoseSample, arrival_ns: int) -> None: ...

    def process_frame(self, detections: Sequence[Detection], depth: DepthMeasurement,
                      intrinsics: CameraIntrinsics, arrival_ns: int) -> None: ...

    def finalize(self) -> List[TagResult]: ...


@dataclass(frozen=True, eq=False)
class PointEstimate:
    p_camera_m: Optional[np.ndarray]  # (3,) optical frame, None = no estimate
    failure: Optional[FailureReason] = None


PointEstimator = Callable[[Detection, DepthMeasurement, CameraIntrinsics], PointEstimate]
PipelineFactory = Callable[[Scenario], Pipeline]
EstimatorFactory = Callable[[Scenario], PointEstimator]


# ---------------------------------------------------------------------------
# End-to-end scoring
# ---------------------------------------------------------------------------


@dataclass
class Score:
    n_truth: int = 0
    n_correct: int = 0
    n_missed: int = 0
    n_false: int = 0
    n_duplicate: int = 0
    errors_m: List[float] = field(default_factory=list)
    confirm_times_s: List[float] = field(default_factory=list)
    # Status-aware metrics (substep 3.1)
    n_correct_confirmed: int = 0  # matched tag is CONFIRMED/UPDATED with the true cell
    n_false_confirmed: int = 0  # CONFIRMED/UPDATED tags that are unmatched or in the wrong cell
    n_false_provisional: int = 0  # PROVISIONAL tags that are unmatched or in the wrong cell
    n_flicker: int = 0  # cell changes after confirmation, summed over tags
    confirm_from_first_obs_s: List[float] = field(default_factory=list)
    reliability: List[Tuple[float, bool]] = field(default_factory=list)  # (top-cell probability, correct)


CONFIRMED_STATUSES = (TagStatus.CONFIRMED, TagStatus.UPDATED)


def score_tags(truths: Sequence[SurvivorTruth], tags: Sequence[TagResult], t0_ns: int = 0,
               first_seen_ns: Optional[Dict[int, int]] = None) -> Score:
    """Each scored tag is assigned to its nearest truth within MATCH_RADIUS_M (unassigned = false tag).
    For each truth, the nearest assigned tag is its match; further assigned tags are duplicates.
    Correct = matched tag's cell equals the true cell (any status); n_correct_confirmed also requires CONFIRMED or
    UPDATED. A tag is "false" (counted per status) if it is unmatched or its cell differs from the truth it is
    assigned to. UNRELIABLE tags are not scored; tags without a position cannot be matched and count as false.
    first_seen_ns (survivor id -> first frame the survivor was detected) gives time-to-confirm from first observation.
    Reliability pairs: every scored tag with a top_prob, correct = assigned and in the true cell."""
    sc = Score(n_truth=len(truths))
    centres = np.array([t.centre_map_xy for t in truths], dtype=float).reshape(-1, 2)
    assigned: Dict[int, List[Tuple[float, TagResult]]] = {i: [] for i in range(len(truths))}
    owner: Dict[int, Optional[int]] = {}
    for tag in tags:
        if tag.status == TagStatus.UNRELIABLE:
            continue
        sc.n_flicker += tag.cell_changes
        if tag.p_map_m is None or not len(truths):
            sc.n_false += 1
            owner[id(tag)] = None
            continue
        d = np.linalg.norm(centres - np.asarray(tag.p_map_m, dtype=float)[:2], axis=1)
        i = int(np.argmin(d))
        if d[i] <= MATCH_RADIUS_M:
            assigned[i].append((float(d[i]), tag))
            owner[id(tag)] = i
        else:
            sc.n_false += 1
            owner[id(tag)] = None
    for tag in tags:
        if tag.status == TagStatus.UNRELIABLE:
            continue
        i = owner[id(tag)]
        correct = i is not None and tag.cell is not None and tuple(tag.cell) == tuple(truths[i].cell)
        if tag.top_prob is not None:
            sc.reliability.append((float(tag.top_prob), correct))
        if not correct:
            if tag.status in CONFIRMED_STATUSES:
                sc.n_false_confirmed += 1
            elif tag.status == TagStatus.PROVISIONAL:
                sc.n_false_provisional += 1
    for i, truth in enumerate(truths):
        lst = sorted(assigned[i], key=lambda x: x[0])
        if not lst:
            sc.n_missed += 1
            continue
        dist, best = lst[0]
        sc.n_duplicate += len(lst) - 1
        sc.errors_m.append(dist)
        if best.cell is not None and tuple(best.cell) == tuple(truth.cell):
            sc.n_correct += 1
            if best.status in CONFIRMED_STATUSES:
                sc.n_correct_confirmed += 1
        confirms = [t.confirm_stamp_ns for _, t in lst if t.status in CONFIRMED_STATUSES
                    and t.confirm_stamp_ns is not None]
        if confirms:
            sc.confirm_times_s.append((min(confirms) - t0_ns) / NS_PER_S)
            if first_seen_ns is not None and truth.survivor_id in first_seen_ns:
                sc.confirm_from_first_obs_s.append((min(confirms) - first_seen_ns[truth.survivor_id]) / NS_PER_S)
    return sc


@dataclass
class ScenarioResult:
    scenario: str
    seed: int
    score: Score
    tags: List[TagResult]
    runtime_s: float
    truths: List[SurvivorTruth] = field(default_factory=list)
    pipeline: object = None  # kept so that final_tags(policy) can be evaluated afterwards


def first_detection_ns(frames: Iterable[Frame]) -> Dict[int, int]:
    """Survivor id -> capture stamp of the first frame in which it was detected."""
    out: Dict[int, int] = {}
    for fr in frames:
        for sid in fr.det_survivor_ids:
            if sid is not None and sid not in out:
                out[sid] = fr.stamp_ns
    return out


def run_events(placed: Scenario, events: Sequence[Event], pipeline: Pipeline, seed: int) -> ScenarioResult:
    t_start = time.perf_counter()
    for ev in events:
        if ev.kind == "pose":
            pipeline.process_pose(ev.payload, ev.arrival_ns)
        elif ev.kind == "frame":
            fr: Frame = ev.payload
            pipeline.process_frame(fr.detections, fr.depth, fr.intrinsics, ev.arrival_ns)
        elif ev.kind == "correction":  # pipelines without Layer 5 ignore corrections (the "before Layer 5" case)
            if hasattr(pipeline, "process_correction"):
                pipeline.process_correction(ev.payload, ev.arrival_ns)
        elif ev.kind == "registration":
            if hasattr(pipeline, "process_registration_event"):
                pipeline.process_registration_event(ev.payload, ev.arrival_ns)
        else:
            raise ValueError(f"unknown event kind {ev.kind!r}")
    tags = pipeline.finalize()
    first = first_detection_ns(e.payload for e in events if e.kind == "frame")
    return ScenarioResult(placed.name, seed, score_tags(placed.truths(), tags, first_seen_ns=first), tags,
                          time.perf_counter() - t_start, placed.truths(), pipeline)


def run_scenario(scenario: Scenario, pipeline: Pipeline, seed: int) -> ScenarioResult:
    placed = place_survivors(scenario, seed)
    return run_events(placed, build_events(placed, seed), pipeline, seed)


# ---------------------------------------------------------------------------
# Camera-frame scoring of a point estimator
# ---------------------------------------------------------------------------


@dataclass
class CameraScore:
    n_det: int = 0  # detections of real survivors (false positives excluded)
    n_valid: int = 0
    n_ambiguous: int = 0
    n_false_assoc: int = 0
    errors_m: List[float] = field(default_factory=list)  # 3D
    h_errors_m: List[float] = field(default_factory=list)  # horizontal (grid-plane) error
    times_ms: List[float] = field(default_factory=list)

    def merge(self, other: "CameraScore") -> None:
        self.n_det += other.n_det
        self.n_valid += other.n_valid
        self.n_ambiguous += other.n_ambiguous
        self.n_false_assoc += other.n_false_assoc
        self.errors_m += other.errors_m
        self.h_errors_m += other.h_errors_m
        self.times_ms += other.times_ms


def _distance_to_box(p: np.ndarray, lo, hi) -> float:
    return float(np.linalg.norm(np.maximum(np.maximum(np.asarray(lo) - p, 0.0), p - np.asarray(hi))))


def score_camera_frame(placed: Scenario, frames: Iterable[Frame], estimator: PointEstimator) -> CameraScore:
    """3D error vs the true box centre in the camera frame; horizontal error = that error vector rotated into the
    grid frame with the TRUE camera attitude, vertical component dropped (the cell depends on it only); false
    association = estimate (mapped to the grid with the TRUE camera pose) farther than FALSE_ASSOC_DIST_M from the
    survivor's box."""
    boxes = {s.survivor_id: s.box() for s in placed.survivors}
    cs = CameraScore()
    for fr in frames:
        truth = dict(fr.truth_camera_m)
        for det, sid in zip(fr.detections, fr.det_survivor_ids):
            if sid is None:
                continue
            cs.n_det += 1
            t0 = time.perf_counter()
            est = estimator(det, fr.depth, fr.intrinsics)
            cs.times_ms.append((time.perf_counter() - t0) * 1e3)
            if est.failure == FailureReason.DEPTH_AMBIGUOUS_CLUSTERS:
                cs.n_ambiguous += 1
            if est.p_camera_m is None:
                continue
            cs.n_valid += 1
            p = np.asarray(est.p_camera_m, dtype=float)
            diff = p - np.asarray(truth[sid])
            cs.errors_m.append(float(np.linalg.norm(diff)))
            cs.h_errors_m.append(float(np.linalg.norm((fr.T_grid_camera[:3, :3] @ diff)[:2])))
            p_grid = fr.T_grid_camera[:3, :3] @ p + fr.T_grid_camera[:3, 3]
            if _distance_to_box(p_grid, boxes[sid].lo, boxes[sid].hi) > FALSE_ASSOC_DIST_M:
                cs.n_false_assoc += 1
    return cs


# ---------------------------------------------------------------------------
# Evaluation and tables
# ---------------------------------------------------------------------------


@dataclass
class SummaryRow:
    scenario: str
    runs: int
    n_truth: int
    correct_rate: float
    missed_per_run: float
    false_per_run: float
    duplicate_per_run: float
    err_median_m: float
    err_p90_m: float
    confirm_median_s: float
    correct_confirmed_rate: float = math.nan
    false_confirmed_per_run: float = math.nan
    false_provisional_per_run: float = math.nan
    flicker_per_run: float = math.nan
    confirm_from_first_median_s: float = math.nan


def _pct(xs: Sequence[float], q: float) -> float:
    return float(np.percentile(xs, q)) if len(xs) else math.nan


def summarise(name: str, results: Sequence[ScenarioResult]) -> SummaryRow:
    n = len(results)
    truth = sum(r.score.n_truth for r in results)
    errs = [e for r in results for e in r.score.errors_m]
    conf = [c for r in results for c in r.score.confirm_times_s]
    return SummaryRow(
        name, n, truth,
        sum(r.score.n_correct for r in results) / truth if truth else math.nan,
        sum(r.score.n_missed for r in results) / n,
        sum(r.score.n_false for r in results) / n,
        sum(r.score.n_duplicate for r in results) / n,
        _pct(errs, 50), _pct(errs, 90), _pct(conf, 50),
        sum(r.score.n_correct_confirmed for r in results) / truth if truth else math.nan,
        sum(r.score.n_false_confirmed for r in results) / n,
        sum(r.score.n_false_provisional for r in results) / n,
        sum(r.score.n_flicker for r in results) / n,
        _pct([c for r in results for c in r.score.confirm_from_first_obs_s], 50),
    )


@dataclass
class CameraRow:
    scenario: str
    n_det: int
    valid_rate: float
    ambiguous_rate: float
    false_assoc_rate: float  # among valid estimates
    err_median_m: float
    err_p90_m: float
    time_median_ms: float
    time_p95_ms: float
    h_err_median_m: float = math.nan
    h_err_p90_m: float = math.nan
    h_below: Tuple[float, ...] = ()  # share of horizontal errors below each HORIZONTAL_THRESHOLDS_M


def camera_row(name: str, cs: CameraScore) -> CameraRow:
    h = np.asarray(cs.h_errors_m)
    below = tuple(float((h < t).mean()) if h.size else math.nan for t in HORIZONTAL_THRESHOLDS_M)
    return CameraRow(name, cs.n_det,
                     cs.n_valid / cs.n_det if cs.n_det else math.nan,
                     cs.n_ambiguous / cs.n_det if cs.n_det else math.nan,
                     cs.n_false_assoc / cs.n_valid if cs.n_valid else math.nan,
                     _pct(cs.errors_m, 50), _pct(cs.errors_m, 90), _pct(cs.times_ms, 50), _pct(cs.times_ms, 95),
                     _pct(cs.h_errors_m, 50), _pct(cs.h_errors_m, 90), below)


# ---------------------------------------------------------------------------
# Observation-level scoring (every processed detection, not only the first per tag)
# ---------------------------------------------------------------------------


@dataclass
class ObsScore:
    n_obs: int = 0  # observations of real survivors
    n_accepted: int = 0  # with a cell and no failure
    n_correct: int = 0  # accepted with the true cell
    h_errors_m: List[float] = field(default_factory=list)  # horizontal map-frame error of accepted observations
    reasons: Counter = field(default_factory=Counter)  # rejection reasons
    n_fp_accepted: int = 0  # accepted observations of false-positive detections

    def merge(self, other: "ObsScore") -> None:
        self.n_obs += other.n_obs
        self.n_accepted += other.n_accepted
        self.n_correct += other.n_correct
        self.h_errors_m += other.h_errors_m
        self.reasons.update(other.reasons)
        self.n_fp_accepted += other.n_fp_accepted


def score_observations(placed: Scenario, frames: Iterable[Frame],
                       observations: Iterable[SurvivorObservation]) -> ObsScore:
    """Links each observation to its detection by (capture stamp, detection_id). Correct = observation cell equals
    the true cell (both 0-based in the simulator). Horizontal error is measured in the map frame, which equals the
    grid-frame error because the registration is rigid."""
    sid_of = {(d.image_stamp_ns, d.detection_id): sid
              for fr in frames for d, sid in zip(fr.detections, fr.det_survivor_ids)}
    truths = {t.survivor_id: t for t in placed.truths()}
    sc = ObsScore()
    for o in observations:
        sid = sid_of.get((o.capture_stamp_ns, o.detection_id))
        accepted = o.failure is None and o.cell is not None
        if sid is None:
            sc.n_fp_accepted += int(accepted)
            continue
        sc.n_obs += 1
        if not accepted:
            sc.reasons[o.failure.value if o.failure is not None else "NO_CELL"] += 1
            continue
        sc.n_accepted += 1
        truth = truths[sid]
        sc.n_correct += int(tuple(o.cell) == tuple(truth.cell))
        sc.h_errors_m.append(float(np.linalg.norm(np.asarray(o.p_map_m)[:2] - np.asarray(truth.centre_map_xy))))
    return sc


@dataclass
class ObsRow:
    scenario: str
    n_obs: int
    accept_rate: float
    correct_rate: float  # among accepted observations
    h_err_median_m: float
    h_err_p90_m: float
    fp_accepted: int
    reasons: str


def obs_row(name: str, sc: ObsScore) -> ObsRow:
    reasons = ", ".join(f"{k}:{v}" for k, v in sc.reasons.most_common())
    return ObsRow(name, sc.n_obs, sc.n_accepted / sc.n_obs if sc.n_obs else math.nan,
                  sc.n_correct / sc.n_accepted if sc.n_accepted else math.nan,
                  _pct(sc.h_errors_m, 50), _pct(sc.h_errors_m, 90), sc.n_fp_accepted, reasons)


def format_obs_table(rows: Sequence[ObsRow]) -> str:
    head = (f"{'scenario':<29}{'n_obs':>6}{'accept':>7}{'correct':>8}{'h_med':>7}{'h_p90':>7}{'fp_acc':>7}"
            f"  rejections")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r.scenario:<29}{r.n_obs:>6}{r.accept_rate:>7.2f}{r.correct_rate:>8.2f}"
                     f"{r.h_err_median_m:>7.3f}{r.h_err_p90_m:>7.3f}{r.fp_accepted:>7}  {r.reasons}")
    return "\n".join(lines)


@dataclass
class Evaluation:
    """results[pipeline][scenario] -> per-seed results; camera[estimator][scenario] -> merged CameraScore."""

    scenarios: List[str]
    results: Dict[str, Dict[str, List[ScenarioResult]]]
    camera: Dict[str, Dict[str, CameraScore]]
    obs: Dict[str, Dict[str, ObsScore]] = field(default_factory=dict)  # pipelines exposing observations()
    timing_ms: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)  # pipeline -> {"l2": [...], ...}

    def obs_rows(self, pipeline: str, with_total: bool = False) -> List[ObsRow]:
        r = [obs_row(s, self.obs[pipeline][s]) for s in self.scenarios]
        if with_total:
            total = ObsScore()
            for s in self.scenarios:
                total.merge(self.obs[pipeline][s])
            r.append(obs_row("ALL", total))
        return r

    def rows(self, pipeline: str, with_total: bool = False) -> List[SummaryRow]:
        r = [summarise(s, self.results[pipeline][s]) for s in self.scenarios]
        if with_total:
            r.append(summarise("ALL", [x for s in self.scenarios for x in self.results[pipeline][s]]))
        return r

    def camera_rows(self, estimator: str, with_total: bool = False) -> List[CameraRow]:
        r = [camera_row(s, self.camera[estimator][s]) for s in self.scenarios]
        if with_total:
            total = CameraScore()
            for s in self.scenarios:
                total.merge(self.camera[estimator][s])
            r.append(camera_row("ALL", total))
        return r


def evaluate_many(pipelines: Dict[str, PipelineFactory], scenarios: Iterable[Scenario], seeds: Sequence[int],
                  estimators: Optional[Dict[str, EstimatorFactory]] = None,
                  hooks: Sequence[Callable] = ()) -> Evaluation:
    """Generates each (scenario, seed) event stream once and runs every pipeline and estimator on it.
    Each hook is called as hook(pipeline_name, placed_scenario, frames, pipeline, scenario_result) after a run."""
    estimators = estimators or {}
    ev = Evaluation([], {k: {} for k in pipelines}, {k: {} for k in estimators})
    for scn in scenarios:
        ev.scenarios.append(scn.name)
        for k in pipelines:
            ev.results[k][scn.name] = []
        for k in estimators:
            ev.camera[k][scn.name] = CameraScore()
        for seed in seeds:
            placed = place_survivors(scn, seed)
            events = build_events(placed, seed)
            frames = [e.payload for e in events if e.kind == "frame"]
            for k, factory in pipelines.items():
                pl = factory(placed)
                res = run_events(placed, events, pl, seed)
                ev.results[k][scn.name].append(res)
                for hook in hooks:
                    hook(k, placed, frames, pl, res)
                if hasattr(pl, "observations"):
                    ev.obs.setdefault(k, {}).setdefault(scn.name, ObsScore()).merge(
                        score_observations(placed, frames, pl.observations()))
                for layer in ("l2", "l3"):
                    if hasattr(pl, f"{layer}_ms"):
                        ev.timing_ms.setdefault(k, {}).setdefault(layer, []).extend(getattr(pl, f"{layer}_ms"))
            for k, factory in estimators.items():
                ev.camera[k][scn.name].merge(score_camera_frame(placed, frames, factory(placed)))
    return ev


def evaluate(pipeline_factory: PipelineFactory, scenarios: Iterable[Scenario],
             seeds: Sequence[int]) -> List[SummaryRow]:
    return evaluate_many({"p": pipeline_factory}, scenarios, seeds).rows("p")


def format_table(rows: Sequence[SummaryRow]) -> str:
    head = (f"{'scenario':<29}{'runs':>5}{'correct':>9}{'missed':>8}{'false':>7}{'dup':>6}"
            f"{'err_med_m':>11}{'err_p90_m':>11}{'t_conf_s':>10}")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r.scenario:<29}{r.runs:>5}{r.correct_rate:>9.2f}{r.missed_per_run:>8.2f}"
                     f"{r.false_per_run:>7.2f}{r.duplicate_per_run:>6.2f}{r.err_median_m:>11.3f}"
                     f"{r.err_p90_m:>11.3f}{r.confirm_median_s:>10.2f}")
    return "\n".join(lines)


def format_camera_table(rows: Sequence[CameraRow]) -> str:
    """h_* = horizontal error (primary); 3d_* = 3D error; <0.15 etc. = share of horizontal errors below."""
    head = (f"{'scenario':<29}{'n_det':>6}{'valid':>6}{'ambig':>6}{'f_ass':>6}{'h_med':>7}{'h_p90':>7}"
            f"{'<0.15':>6}{'<0.25':>6}{'<0.5':>6}{'3d_med':>7}{'3d_p90':>7}{'t_med':>7}{'t_p95':>7}")
    lines = [head, "-" * len(head)]
    for r in rows:
        b = r.h_below if len(r.h_below) == 3 else (math.nan,) * 3
        lines.append(f"{r.scenario:<29}{r.n_det:>6}{r.valid_rate:>6.2f}{r.ambiguous_rate:>6.2f}"
                     f"{r.false_assoc_rate:>6.2f}{r.h_err_median_m:>7.3f}{r.h_err_p90_m:>7.3f}"
                     f"{b[0]:>6.2f}{b[1]:>6.2f}{b[2]:>6.2f}{r.err_median_m:>7.3f}{r.err_p90_m:>7.3f}"
                     f"{r.time_median_ms:>7.3f}{r.time_p95_ms:>7.3f}")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    from .baseline import naive_factory
    from .scenarios import pre_32_scenarios

    t = time.perf_counter()
    rows = evaluate(naive_factory, pre_32_scenarios().values(), seeds=range(5))
    print("NaivePipeline, SIMULATION tested on synthetic data, 5 seeds per scenario")
    print(format_table(rows))
    print(f"evaluation time: {time.perf_counter() - t:.1f} s")


if __name__ == "__main__":  # pragma: no cover
    main()


# ---------------------------------------------------------------------------
# Layer 4 tables: status-aware summary, reliability, assumed mission points
# ---------------------------------------------------------------------------


def format_status_table(rows: Sequence[SummaryRow]) -> str:
    """corr_conf = survivors with a CONFIRMED tag in the true cell; f_conf / f_prov = false CONFIRMED / PROVISIONAL
    tags per run; t_conf = median time from the survivor's first detection to confirmation."""
    head = (f"{'scenario':<29}{'corr_any':>9}{'corr_conf':>10}{'missed':>7}{'f_conf':>7}{'f_prov':>7}{'dup':>6}"
            f"{'flick':>6}{'err_med':>8}{'t_conf':>7}")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r.scenario:<29}{r.correct_rate:>9.2f}{r.correct_confirmed_rate:>10.2f}{r.missed_per_run:>7.2f}"
                     f"{r.false_confirmed_per_run:>7.2f}{r.false_provisional_per_run:>7.2f}"
                     f"{r.duplicate_per_run:>6.2f}{r.flicker_per_run:>6.2f}{r.err_median_m:>8.3f}"
                     f"{r.confirm_from_first_median_s:>7.2f}")
    return "\n".join(lines)


RELIABILITY_BINS = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0000001)


def reliability_table(pairs: Sequence[Tuple[float, bool]], bins: Sequence[float] = RELIABILITY_BINS) -> List[tuple]:
    """Rows (lo, hi, count, mean predicted probability, observed fraction correct)."""
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [(p, c) for p, c in pairs if lo <= p < hi]
        if sel:
            rows.append((lo, min(hi, 1.0), len(sel), float(np.mean([p for p, _ in sel])),
                         float(np.mean([c for _, c in sel]))))
        else:
            rows.append((lo, min(hi, 1.0), 0, math.nan, math.nan))
    return rows


def mission_points(truths: Sequence[SurvivorTruth], committed: Sequence[TagResult], correct_points: float,
                   wrong_penalty: float) -> float:
    """ASSUMED scoring (O2 unknown): each survivor whose nearest committed tag (within MATCH_RADIUS_M) has the true
    cell earns correct_points; every other committed tag (wrong cell, unmatched, or a second tag for a survivor)
    adds wrong_penalty."""
    sc = score_tags(truths, committed)
    n_committed = sum(1 for t in committed if t.status != TagStatus.UNRELIABLE)
    return correct_points * sc.n_correct + wrong_penalty * (n_committed - sc.n_correct)
