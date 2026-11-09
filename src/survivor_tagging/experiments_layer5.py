"""Substep 3.2 evaluation: Layer 5 correction handling, revisions and live output.

Run: PYTHONPATH=src python -m survivor_tagging.experiments_layer5 [output file]
Prints the summary tables; the full output is also written to the output file (default docs/reports/3.2_eval.txt).
All results are SIMULATION tested on synthetic data (box-shaped survivors, simulated SLAM corrections).

Pre-registered decision rules (substep 3.2, followed as written):
R1 for observations with full corrected history, the tags after a correction event equal (same cell, position within
   1e-9 m) the tags of an offline run that used the corrected poses from the start.
R2 loop_closure: duplicates per run <= 0.1, and the median error of pre-closure observations <= the post-closure median
   + 0.03 m.
R3 a full recompute with the maximum stored observations (default limits) takes <= 250 ms on this laptop (ASSUMPTION).
   If it fails: report it, implement an incremental recompute restricted to affected tracks, re-measure.
"""

from __future__ import annotations

import copy
import io
import math
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .harness import MATCH_RADIUS_M, _pct, run_events, score_tags
from .output import MSG_SNAPSHOT, decode_snapshot, decode_tag_record, snapshot_size
from .scenarios import (CORRECTION_SCENARIOS, all_scenarios, make_scenario, with_correction_mode)
from .sim import Closure, Event, Scenario, SurvivorSpec, build_events, place_survivors
from .sim_pipelines import layer1234_factory, layer5_factory
from .types import PoseCorrection, TagStatus

SEEDS = range(5)
F1_SCENARIOS = ("loop_closure", "loop_closure_late", "repeated_corrections", "partial_history", "registration_update")
R1_SCENARIOS = ("loop_closure", "loop_closure_late", "repeated_corrections")  # corrected history covers every obs
R1_TOL_M = 1e-9
R2_DUP_MAX = 0.1
R2_MARGIN_M = 0.03
R3_BUDGET_MS = 250.0
R3_SIZES = (250, 500, 1000, 2000)
R3_REPEATS = 3
NS = 1_000_000_000


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def feed(pl, events: Sequence[Event]) -> None:
    for ev in events:
        if ev.kind == "pose":
            pl.process_pose(ev.payload, ev.arrival_ns)
        elif ev.kind == "frame":
            pl.process_frame(ev.payload.detections, ev.payload.depth, ev.payload.intrinsics, ev.arrival_ns)
        elif ev.kind == "correction" and hasattr(pl, "process_correction"):
            pl.process_correction(ev.payload, ev.arrival_ns)
        elif ev.kind == "registration" and hasattr(pl, "process_registration_event"):
            pl.process_registration_event(ev.payload, ev.arrival_ns)


def gcs_fix_times(placed: Scenario, events: Sequence[Event], sent) -> Tuple[List[float], int]:
    """Replays the telemetry the GCS received (decoded) and, for every correction / registration event and every
    survivor whose GCS tag was NOT in the true cell just before the event, the time until the GCS holds a tag
    (not removed, not CANDIDATE) within MATCH_RADIUS_M of the survivor (grid frame) in the true cell.
    Returns (delays in s, number of such cases never fixed before the end)."""
    msgs = sorted(sent, key=lambda m: m.stamp_ns)
    truths = [(t.cell, placed.registration.grid_from_map(t.centre_map_xy)) for t in placed.truths()]

    def ok(view, cell, g):
        for rec in view.values():
            if rec.grid_xy_m is None or rec.status == TagStatus.CANDIDATE or rec.flags & 0x10:
                continue
            if np.hypot(rec.grid_xy_m[0] - g[0], rec.grid_xy_m[1] - g[1]) <= MATCH_RADIUS_M and rec.cell == cell:
                return True
        return False

    def apply(view, m):
        if m.payload[0] == MSG_SNAPSHOT:
            view.clear()
            for r in decode_snapshot(m.payload).records:
                view[r.survivor_id] = r
        else:
            r = decode_tag_record(m.payload)
            if r.flags & 0x10:
                view.pop(r.survivor_id, None)
            else:
                view[r.survivor_id] = r

    delays, never = [], 0
    for ev in events:
        if ev.kind not in ("correction", "registration"):
            continue
        view: Dict[int, object] = {}
        k = 0
        while k < len(msgs) and msgs[k].stamp_ns < ev.arrival_ns:
            apply(view, msgs[k])
            k += 1
        for cell, g in truths:
            if ok(view, cell, g):
                continue
            v2, fixed = dict(view), None
            for m in msgs[k:]:
                apply(v2, m)
                if ok(v2, cell, g):
                    fixed = (m.stamp_ns - ev.arrival_ns) / NS
                    break
            if fixed is None:
                never += 1
            else:
                delays.append(fixed)
    return delays, never


def obs_errors(placed: Scenario, events: Sequence[Event], pl, split_s: float) -> Tuple[List[float], List[float]]:
    """Horizontal map error of accepted observations of real survivors, captured before / after split_s."""
    frames = [e.payload for e in events if e.kind == "frame"]
    sid_of = {(d.image_stamp_ns, d.detection_id): sid for fr in frames
              for d, sid in zip(fr.detections, fr.det_survivor_ids)}
    truths = {t.survivor_id: t for t in placed.truths()}
    pre, post = [], []
    for o in pl.observations():
        sid = sid_of.get((o.capture_stamp_ns, o.detection_id))
        if sid is None or o.failure is not None or o.p_map_m is None:
            continue
        e = float(np.linalg.norm(np.asarray(o.p_map_m)[:2] - np.asarray(truths[sid].centre_map_xy)))
        (pre if o.capture_stamp_ns < split_s * NS else post).append(e)
    return pre, post


def run_pair(scn: Scenario, seed: int):
    placed = place_survivors(scn, seed)
    events = build_events(placed, seed)
    before = run_events(placed, events, layer1234_factory()(placed), seed)
    after = run_events(placed, events, layer5_factory()(placed), seed)
    return placed, events, before, after


# ---------------------------------------------------------------------------
# F1 / R2: before vs after Layer 5
# ---------------------------------------------------------------------------


def f1_table(runs: Dict[str, list]) -> str:
    head = (f"{'scenario':<22}{'pipe':>5}{'correct':>8}{'dup/run':>8}{'err_med':>8}{'UPDATED':>8}"
            f"{'t_fix_med_s':>12}{'fixed':>7}{'unfixed':>8}{'recomp_ms':>10}")
    lines = [head, "-" * len(head)]
    for name in F1_SCENARIOS:
        for label in ("L1-4", "L1-5"):
            res = [r[2] if label == "L1-4" else r[3] for r in runs[name]]
            truth = sum(r.score.n_truth for r in res)
            correct = sum(r.score.n_correct for r in res) / truth
            dup = sum(r.score.n_duplicate for r in res) / len(res)
            err = _pct([e for r in res for e in r.score.errors_m], 50)
            if label == "L1-5":
                upd = sum(r.pipeline.book.n_cell_changes for r in res) / len(res)
                fixes = [gcs_fix_times(r[0], r[1], r[3].pipeline.sent) for r in runs[name]]
                d = [x for f in fixes for x in f[0]]
                never = sum(f[1] for f in fixes)
                rc = _pct([s.total_ms for r in res for s in r.pipeline.recomputes], 50)
                tail = f"{upd:>8.2f}{_pct(d, 50):>12.2f}{len(d):>7}{never:>8}{rc:>10.1f}"
            else:
                tail = f"{'-':>8}{'-':>12}{'-':>7}{'-':>8}{'-':>10}"
            lines.append(f"{name:<22}{label:>5}{correct:>8.2f}{dup:>8.2f}{err:>8.3f}" + tail)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# R1: equivalence with an offline run on the corrected poses
# ---------------------------------------------------------------------------


def offline_events(events: Sequence[Event], upto_revision: Optional[int] = None) -> List[Event]:
    """Same stream, every live pose replaced by its newest corrected pose from the correction events with revision
    <= upto_revision (all if None), and no correction events: a run that had those corrected poses from the start."""
    best = {}
    for e in events:
        if e.kind == "correction" and (upto_revision is None or e.payload.revision <= upto_revision):
            for p in e.payload.poses:
                if p.stamp_ns not in best or p.pose_revision >= best[p.stamp_ns].pose_revision:
                    best[p.stamp_ns] = p
    return [Event(e.arrival_ns, "pose", best.get(e.payload.stamp_ns, e.payload)) if e.kind == "pose" else e
            for e in events if e.kind != "correction"]


def r1_compare(placed: Scenario, ev: Sequence[Event], mode: str = "full") -> List[Tuple[bool, float]]:
    """R1 for one run: right after each correction event k, the online state is compared with an offline run on
    the corrected poses known at that moment (events <= k) up to the same arrival time; at the end with all."""
    f = layer5_factory({"l5.recompute_mode": mode})
    on_pl = f(placed)
    out = []
    for e in ev:
        feed(on_pl, [e])
        if e.kind == "correction":
            off_pl = f(placed)
            feed(off_pl, [x for x in offline_events(ev, e.payload.revision) if x.arrival_ns <= e.arrival_ns])
            out.append(tags_equal(on_pl, off_pl))
    off_pl = f(placed)
    feed(off_pl, offline_events(ev))
    on_pl.finalize()
    off_pl.finalize()
    out.append(tags_equal(on_pl, off_pl))
    return out


def tags_equal(a, b, tol: float = R1_TOL_M) -> Tuple[bool, float]:
    ta = sorted(a.l4.tags(), key=lambda t: (t.first_seen_ns, t.p_map_m[0]))
    tb = sorted(b.l4.tags(), key=lambda t: (t.first_seen_ns, t.p_map_m[0]))
    if len(ta) != len(tb):
        return False, math.inf
    worst = max([float(np.max(np.abs(x.p_map_m - y.p_map_m))) for x, y in zip(ta, tb)] + [0.0])
    return all(x.cell == y.cell for x, y in zip(ta, tb)) and worst <= tol, worst


def r1_check(mode: str = "full") -> Tuple[int, int, float]:
    n_ok, n, worst = 0, 0, 0.0
    for name in R1_SCENARIOS:
        for seed in SEEDS:
            placed = place_survivors(all_scenarios()[name], seed)
            for eq, w in r1_compare(placed, build_events(placed, seed), mode):
                n, n_ok, worst = n + 1, n_ok + int(eq), max(worst, w)
    return n_ok, n, worst


# ---------------------------------------------------------------------------
# R3: recompute cost with the maximum stored observations
# ---------------------------------------------------------------------------


BENCH_DURATION_S = 250.0
BENCH_CLOSURE_S = 248.0


def bench_scenario() -> Scenario:
    """SIM: 6 standing survivors 2 m apart along a 13 m corridor (grid y); the drone flies along it 2.5 m away
    (back and forth, sees ~2-3 survivors at a time) with a slow pose drift; a loop closure at 248 s corrects the
    whole history. About 2600 detections, so the default store (2000) is full and evicts."""
    surv = tuple(SurvivorSpec(k + 1, (5.5, 1.5 + 2 * k), "standing") for k in range(6))
    path = ((3.0, 0.5), (3.0, 12.5)) * 5
    return make_scenario("bench_r3", "R3 recompute cost", surv, path_grid=path, n_cells=13,
                         duration_s=BENCH_DURATION_S, placement="nominal",
                         trajectory_kw=dict(drift_m_per_s=(0.0, 0.002, 0.0), closures=(Closure(BENCH_CLOSURE_S),)))


def frame_events_last(seconds: float) -> List[PoseCorrection]:
    """Two frame corrections (identity rotation, 2 cm shift): revision 1 at closure - seconds, revision 2 at the
    closure. The second one only affects observations captured in that last window (C4 frame semantics)."""
    t = np.eye(4)
    t[1, 3] = 0.02
    return [PoseCorrection(1, "frame_correction", int((BENCH_CLOSURE_S - seconds) * NS), "sim", (), t),
            PoseCorrection(2, "frame_correction", int(BENCH_CLOSURE_S * NS), "sim", (), t)]


def _measure(pl, mode: str, events: Sequence[PoseCorrection], arrival_ns: int, source: str):
    """Median over R3_REPEATS deep copies of the time of the LAST event's recompute."""
    times, stats = [], None
    for _ in range(R3_REPEATS):
        cp = copy.deepcopy(pl)
        cp.l5 = replace(cp.l5, recompute_mode=mode, correction_source=source)
        for c in events:
            out = cp.process_correction(c, arrival_ns)
            assert out.applied, out
        stats = cp.recomputes[-1]
        times.append(stats.total_ms)
    return statistics.median(times), stats


def r3_bench(out) -> Dict[Tuple[int, str, str], float]:
    """Recompute time (ms) per (store size, mode, correction): the stream is processed once per store size up to the
    correction; each measurement runs on deep copies of that state. 'whole' = the history correction of the whole
    flight; 'frame_last_4s' = a frame correction whose window is the last 4 s (only the tracks seen then)."""
    scn = bench_scenario()
    t0 = time.perf_counter()
    ev = build_events(scn, 0)
    corr_ev = [e for e in ev if e.kind == "correction"][0]
    pre = [e for e in ev if e.kind != "correction" and e.arrival_ns <= corr_ev.arrival_ns]
    print(f"R3 bench stream: {sum(len(e.payload.detections) for e in ev if e.kind == 'frame')} detections over "
          f"{BENCH_DURATION_S:.0f} s, generated in {time.perf_counter() - t0:.0f} s", file=out)
    res: Dict[Tuple[int, str, str], float] = {}
    for n in R3_SIZES:
        pl = layer5_factory({"l5.observation_store_max": n})(scn)
        feed(pl, pre)
        info = f"store {len(pl.store)}, evicted {pl.store.n_evicted}, tracks {len(pl.l4.tracks)}"
        cases = [("full", "whole")] + ([("affected_tracks", "whole"), ("full", "frame_last_4s"),
                                        ("affected_tracks", "frame_last_4s")] if n == R3_SIZES[-1] else [])
        for mode, cname in cases:
            if cname == "whole":
                ms, st = _measure(pl, mode, [corr_ev.payload], corr_ev.arrival_ns, "history")
            else:
                ms, st = _measure(pl, mode, frame_events_last(4.0), corr_ev.arrival_ns, "frame_correction")
            res[(n, mode, cname)] = ms
            print(f"  N={n:<5} {mode:<16} {cname:<14} median {ms:8.1f} ms  ({info}; re-derived {st.n_rederived}, "
                  f"carried {st.n_carried}, tracks replayed "
                  f"{'all' if st.n_tracks_replayed < 0 else st.n_tracks_replayed}, re-derive {st.rederive_ms:.0f} ms, "
                  f"Layer 4 {st.replay_ms:.0f} ms)", file=out)
    return res


# ---------------------------------------------------------------------------
# F3: history vs frame_correction
# ---------------------------------------------------------------------------


def f3_rows() -> List[tuple]:
    rows = []
    sc = all_scenarios()
    for name in CORRECTION_SCENARIOS:
        for mode in ("history", "frame_correction"):
            scn = with_correction_mode(sc[name], mode)
            closure_s = min(c.at_s for c in scn.trajectory.closures)
            res, pre = [], []
            for seed in SEEDS:
                placed = place_survivors(scn, seed)
                ev = build_events(placed, seed)
                r = run_events(placed, ev, layer5_factory()(placed), seed)
                res.append(r)
                pre += obs_errors(placed, ev, r.pipeline, closure_s)[0]
            truth = sum(r.score.n_truth for r in res)
            rows.append((name, mode, sum(r.score.n_correct for r in res) / truth,
                         sum(r.score.n_duplicate for r in res) / len(res),
                         _pct([e for r in res for e in r.score.errors_m], 50), _pct(pre, 50), _pct(pre, 90),
                         len(pre)))
    return rows


def format_f3(rows) -> str:
    head = (f"{'scenario':<22}{'source':>17}{'correct':>8}{'dup/run':>8}{'tag_err_med':>12}{'pre_obs_med':>12}"
            f"{'pre_obs_p90':>12}{'n_pre':>6}")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r[0]:<22}{r[1]:>17}{r[2]:>8.2f}{r[3]:>8.2f}{r[4]:>12.3f}{r[5]:>12.3f}{r[6]:>12.3f}{r[7]:>6}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# F4: telemetry bytes
# ---------------------------------------------------------------------------


def f4_table(runs: Dict[str, list], budget_bps: float) -> str:
    snaps, n_upd, total_bytes, duration = [], 0, 0, 0.0
    ntags = []
    for name in F1_SCENARIOS:
        for placed, ev, _, after in runs[name]:
            pl = after.pipeline
            st = pl.telemetry.stats
            snaps += [len(m.payload) for m in pl.sent if m.kind == "snapshot"]
            ntags += [len(m.survivor_ids) for m in pl.sent if m.kind == "snapshot"]
            n_upd += st.n_updates
            total_bytes += st.total_bytes
            duration += placed.duration_s
    lines = [f"{'item':<52}{'value':>12}", "-" * 64,
             f"{'tag update record (bytes)':<52}{25:>12}",
             f"{'snapshot = 14 + 25 N bytes; N = 0 / 1 / 6 / 32':<52}{'14/39/164/814':>12}",
             f"{'observed snapshots: count':<52}{len(snaps):>12}",
             f"{'observed snapshot bytes: median / max':<52}{f'{_pct(snaps, 50):.0f}/{max(snaps)}':>12}",
             f"{'observed tags per snapshot: max':<52}{max(ntags):>12}",
             f"{'observed updates: count':<52}{n_upd:>12}",
             f"{'observed mean rate over all F1 runs (bit/s)':<52}{8 * total_bytes / duration:>12.1f}",
             f"{'  as share of the assumed link budget':<52}{8 * total_bytes / duration / budget_bps:>12.2%}",
             f"{'worst steady case: 32-tag snapshot every 5 s (bit/s)':<52}{8 * snapshot_size(32) / 5:>12.1f}",
             f"{'  + 6 tags updating at the 1/s limit (bit/s)':<52}{8 * (snapshot_size(32) / 5 + 6 * 25):>12.1f}",
             f"{'  as share of the assumed link budget':<52}{8 * (snapshot_size(32) / 5 + 6 * 25) / budget_bps:>12.2%}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(path: str = "docs/reports/3.2_eval.txt") -> None:  # pragma: no cover
    out = io.StringIO()

    def show(text: str = "") -> None:
        print(text)
        print(text, file=out)

    t0 = time.perf_counter()
    show("Substep 3.2 evaluation. SIMULATION tested on synthetic data; seeds 0-4.")
    sc = all_scenarios()
    runs = {name: [run_pair(sc[name], s) for s in SEEDS] for name in F1_SCENARIOS}
    show("\nF1. Before (Layers 1-4) vs after (Layers 1-5). correct = survivors whose matched tag has the true cell; "
         "UPDATED = tag revisions per run;\n    t_fix = time from the correction / registration event to the GCS "
         "(decoded telemetry) holding the true cell, for survivors it did not hold before the event.")
    show(f1_table(runs))

    show("\nR1. Offline equivalence (scenarios with fully corrected history, every correction event and the end).")
    ok, n, worst = r1_check("full")
    r1 = ok == n
    show(f"    full mode: {ok}/{n} comparisons equal, worst position difference {worst:.2e} m -> "
         f"{'PASS' if r1 else 'FAIL'}")
    ok_a, n_a, worst_a = r1_check("affected_tracks")
    show(f"    (affected_tracks mode, information only: {ok_a}/{n_a} equal, worst {worst_a:.2e} m)")

    pre, post, dups = [], [], []
    for placed, ev, _, after in runs["loop_closure"]:
        a, b = obs_errors(placed, ev, after.pipeline, 1.5)
        pre += a
        post += b
        dups.append(after.score.n_duplicate)
    pre_b = []
    for placed, ev, before, _ in runs["loop_closure"]:
        pre_b += obs_errors(placed, ev, before.pipeline, 1.5)[0]
    dup = float(np.mean(dups))
    r2 = dup <= R2_DUP_MAX and _pct(pre, 50) <= _pct(post, 50) + R2_MARGIN_M
    show(f"\nR2. loop_closure: duplicates/run {dup:.2f} (<= {R2_DUP_MAX}); pre-closure observation median "
         f"{_pct(pre, 50):.3f} m (before Layer 5 {_pct(pre_b, 50):.3f} m) vs post-closure {_pct(post, 50):.3f} m "
         f"+ {R2_MARGIN_M} -> {'PASS' if r2 else 'FAIL'}  (n_pre {len(pre)}, n_post {len(post)})")

    show("\nR3. Recompute time vs stored observations (laptop wall clock, median of 3 deep copies; default limits "
         "except the store size).")
    detail = io.StringIO()
    res = r3_bench(detail)
    show(detail.getvalue().rstrip())
    head = f"{'stored N (max)':>15}{'full, whole history (ms)':>27}"
    show(head)
    show("-" * len(head))
    for n_ in R3_SIZES:
        show(f"{n_:>15}{res[(n_, 'full', 'whole')]:>27.1f}")
    nmax = R3_SIZES[-1]
    full_max = res[(nmax, "full", "whole")]
    show(f"    R3 (full, N={nmax}, whole history): {full_max:.0f} ms vs {R3_BUDGET_MS:.0f} ms -> "
         f"{'PASS' if full_max <= R3_BUDGET_MS else 'FAIL'}")
    show(f"    re-measure N={nmax}: affected_tracks whole history {res[(nmax, 'affected_tracks', 'whole')]:.0f} ms; "
         f"frame correction of the last 4 s: full {res[(nmax, 'full', 'frame_last_4s')]:.0f} ms, affected_tracks "
         f"{res[(nmax, 'affected_tracks', 'frame_last_4s')]:.0f} ms")

    show("\nF3. History vs frame_correction (Layers 1-5). pre_obs = horizontal error of observations captured "
         "before the first closure, after re-derivation.")
    show(format_f3(f3_rows()))

    budget = layer5_factory()(sc["loop_closure"]).telemetry_params.link_budget_bps
    show(f"\nF4. Telemetry bytes (assumed link budget {budget:.0f} bit/s, before link framing overhead).")
    show(f4_table(runs, budget))
    show(f"\nevaluation time: {time.perf_counter() - t0:.0f} s")
    Path(path).write_text(out.getvalue(), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main(*sys.argv[1:])
