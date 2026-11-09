"""Substep 3.4: timing of the Layer 4 replay / full recompute and of the live path (harness side, SIMULATION stream).

Run: PYTHONPATH=src python -m survivor_tagging.experiments_34 time <tag>      (tag: before | after; saves
                                                                              docs/reports/3.4_timing_<tag>.json)
     PYTHONPATH=src python -m survivor_tagging.experiments_34 profile        (cProfile top 15: recompute and live)
     PYTHONPATH=src python -m survivor_tagging.experiments_34 report          (before / after table)
Uses the 3.2 R3 bench stream (6 standing survivors, 250 s, ~2600 detections, loop closure at 248 s). Timings are laptop
wall clock (best of 5 deep copies), not Pi numbers. Behaviour is gated separately by survivor_tagging.golden.
"""

from __future__ import annotations

import cProfile
import copy
import gc
import io
import json
import pstats
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np

from .experiments_layer5 import bench_scenario, feed, frame_events_last
from .scenarios import all_scenarios
from .sim import build_events
from .sim_pipelines import layer5_factory

REPORT_DIR = Path("docs/reports")
SIZES = (500, 1000, 2000)
REPEATS = 5
NS = 1_000_000_000


def _stream():
    scn = bench_scenario()
    ev = build_events(scn, 0)
    corr = [e for e in ev if e.kind == "correction"][0]
    pre = [e for e in ev if e.kind != "correction" and e.arrival_ns <= corr.arrival_ns]
    return scn, pre, corr


def _recompute_ms(pl, mode: str, events, arrival_ns: int, source: str) -> float:
    times = []
    for _ in range(REPEATS):
        cp = copy.deepcopy(pl)
        cp.l5 = replace(cp.l5, recompute_mode=mode, correction_source=source)
        gc.collect()
        gc.freeze()  # the deep copy (test set-up) must not be scanned by the collector during the measurement
        try:
            for c in events:
                assert cp.process_correction(c, arrival_ns).applied
        finally:
            gc.unfreeze()
        times.append(cp.recomputes[-1].total_ms)
    return min(times)  # best of REPEATS: the laptop shows 2x run-to-run noise, the minimum is the stable statistic


def measure(sizes=SIZES) -> Dict[str, float]:
    scn, pre, corr = _stream()
    out: Dict[str, float] = {}
    for n in sizes:
        pl = layer5_factory({"l5.observation_store_max": n})(scn)
        t0 = time.perf_counter()
        feed(pl, pre)
        out[f"feed_s_{n}"] = time.perf_counter() - t0
        out[f"full_whole_{n}_ms"] = _recompute_ms(pl, "full", [corr.payload], corr.arrival_ns, "history")
        if n == max(SIZES):
            out[f"affected_whole_{n}_ms"] = _recompute_ms(pl, "affected_tracks", [corr.payload], corr.arrival_ns,
                                                          "history")
            fe = frame_events_last(4.0)
            out[f"full_frame4s_{n}_ms"] = _recompute_ms(pl, "full", fe, corr.arrival_ns, "frame_correction")
            out[f"affected_frame4s_{n}_ms"] = _recompute_ms(pl, "affected_tracks", fe, corr.arrival_ns,
                                                            "frame_correction")
            for layer in ("l2", "l3", "l4"):
                xs = getattr(pl, f"{layer}_ms")
                out[f"live_{layer}_ms_median"] = float(np.median(xs))
                out[f"live_{layer}_ms_mean"] = float(np.mean(xs))
            out["live_n_obs"] = len(pl.l4_ms)
        print(n, {k: round(v, 1) for k, v in out.items() if k.endswith(f"_{n}_ms") or k == f"feed_s_{n}"}, flush=True)
    return out


def profile() -> str:
    scn, pre, corr = _stream()
    pl = layer5_factory({"l5.observation_store_max": 2000})(scn)
    prof_live = cProfile.Profile()
    prof_live.enable()
    feed(pl, pre)
    prof_live.disable()
    cp = copy.deepcopy(pl)
    cp.l5 = replace(cp.l5, correction_source="history")
    prof_rec = cProfile.Profile()
    prof_rec.enable()
    cp.process_correction(corr.payload, corr.arrival_ns)
    prof_rec.disable()
    text = []
    for title, pr in (("FULL RECOMPUTE (2000 stored)", prof_rec), ("LIVE FEED (whole stream, Layers 1-5)", prof_live)):
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(15)
        text.append(f"== {title} ==\n" + s.getvalue())
    return "\n".join(text)


def report() -> str:
    b = json.loads((REPORT_DIR / "3.4_timing_before.json").read_text())
    a = json.loads((REPORT_DIR / "3.4_timing_after.json").read_text())
    rows = [("full recompute, whole history, 500 stored", "full_whole_500_ms"),
            ("full recompute, whole history, 1000 stored", "full_whole_1000_ms"),
            ("full recompute, whole history, 2000 stored", "full_whole_2000_ms"),
            ("affected_tracks, whole history, 2000", "affected_whole_2000_ms"),
            ("full recompute, last-4 s frame correction, 2000", "full_frame4s_2000_ms"),
            ("affected_tracks, last-4 s frame correction, 2000", "affected_frame4s_2000_ms"),
            ("live Layer 2 per observation (median)", "live_l2_ms_median"),
            ("live Layer 3 per observation (median)", "live_l3_ms_median"),
            ("live Layer 4 per observation (median)", "live_l4_ms_median"),
            ("live Layer 4 per observation (mean)", "live_l4_ms_mean")]
    lines = [f"{'measurement':<52}{'before ms':>12}{'after ms':>12}{'speed-up':>10}"]
    for label, k in rows:
        lines.append(f"{label:<52}{b[k]:>12.2f}{a[k]:>12.2f}{b[k] / a[k]:>9.1f}x")
    return "\n".join(lines)


def main(argv: List[str]) -> None:  # pragma: no cover
    if argv and argv[0] == "time":
        res = measure(tuple(int(x) for x in argv[2:]) or SIZES)
        (REPORT_DIR / f"3.4_timing_{argv[1]}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    elif argv and argv[0] == "profile":
        print(profile())
    elif argv and argv[0] == "report":
        print(report())
    else:
        print(__doc__)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
