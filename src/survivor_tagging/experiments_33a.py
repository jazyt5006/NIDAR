"""Substep 3.3a evaluation: near-duplicate observations (Layer 4) and honest pose covariance (simulator).

Run: PYTHONPATH=src python -m survivor_tagging.experiments_33a run <tag>      (tag: before | after)
     PYTHONPATH=src python -m survivor_tagging.experiments_33a report         (tables from the two saved runs)
`run` evaluates every scenario (5 seeds, body offsets 0.0 and 0.2 m, Layers 1-4 and Layers 1-5) and saves the raw
numbers to docs/reports/3.3a_<tag>.json (cached: the "before" run is made with the code as it was before the fixes).
All results are SIMULATION tested on synthetic data.

Pre-registered decision rules (substep 3.3a, followed as written):
A1 double_detection: survivor tagged correctly in >= 0.8 of instances, never lost to a permanent ambiguity streak;
   two_adjacent, two_same_cell, id_merge: missed, duplicates and correct CONFIRMED each within +-0.02 of the before
   values, per scenario, at both body offsets.
A2 pose covariance honesty: mean normalised error squared (2D) of simulated corrected-history and post-correction
   poses in [1.0, 3.0] (expected 2.0); repeated_corrections duplicates per run <= 0.2.
A3 every evaluation number outside the targeted scenarios (double_detection, repeated_corrections, loop_closure*, any
   scenario whose poses carry correction events) unchanged to the printed precision versus the before run.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .harness import evaluate_many, summarise
from .scenarios import all_scenarios
from .sim_pipelines import layer1234_factory, layer5_factory
from .types import FailureReason

SEEDS = range(5)
OFFSETS = (0.0, 0.2)
REPORT_DIR = Path("docs/reports")
TARGETED = ("double_detection", "repeated_corrections", "loop_closure", "loop_closure_late", "partial_history",
            "frame_correction")
A1_REAL = ("two_adjacent", "two_same_cell", "id_merge")
BEFORE_AFTER = ("double_detection", "repeated_corrections", "loop_closure", "loop_closure_late", "two_adjacent",
                "two_same_cell", "id_merge")
FIELDS = ("correct_rate", "missed_per_run", "false_per_run", "duplicate_per_run", "err_median_m", "err_p90_m",
          "confirm_median_s", "correct_confirmed_rate", "false_confirmed_per_run", "false_provisional_per_run",
          "flicker_per_run", "confirm_from_first_median_s")


def pipelines() -> dict:
    out = {}
    for off in OFFSETS:
        ov = {"l1.body_depth_offset_m": off}
        out[f"L1-4@{off}"] = layer1234_factory(ov)
        out[f"L1-5@{off}"] = layer5_factory(ov)
    return out


def ambiguity_lost(pl) -> Dict[str, int]:
    """Layer 4 tracks (alive or dropped) that went UNRELIABLE because of an ambiguity streak, and the alive tracks
    that end the run in that state."""
    l4 = pl.l4
    everything = list(l4.tracks.values()) + list(l4.dropped)
    went = sum(1 for t in everything
               if any(ev.endswith("->UNRELIABLE") and why == FailureReason.ASSOCIATION_AMBIGUOUS.value
                      for _, ev, why in t.history))
    final = sum(1 for t in l4.tracks.values()
                if t.status.name == "UNRELIABLE" and t.ambiguous_streak >= l4.p.unreliable_ambiguous_streak)
    return {"went_unreliable": went, "final_unreliable": final}


def run(tag: str) -> Path:
    extras: Dict[str, dict] = {}

    def hook(name, placed, frames, pl, res):
        d = ambiguity_lost(pl)
        d["n_correct"] = res.score.n_correct
        d["n_truth"] = res.score.n_truth
        extras[f"{name}|{placed.name}|{len(extras)}"] = d

    ev = evaluate_many(pipelines(), list(all_scenarios().values()), list(SEEDS), hooks=(hook,))
    out: Dict[str, dict] = {"rows": {}, "runs": {}}
    for pname in ev.results:
        for scn in ev.scenarios:
            row = asdict(summarise(scn, ev.results[pname][scn]))
            out["rows"][f"{pname}|{scn}"] = {k: (None if isinstance(v, float) and math.isnan(v) else v)
                                              for k, v in row.items() if k in FIELDS or k == "n_truth"}
    for key, d in extras.items():
        pname, scn, _ = key.split("|")
        out["runs"].setdefault(f"{pname}|{scn}", []).append(d)
    path = REPORT_DIR / f"3.3a_{tag}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    return path


def load(tag: str) -> dict:
    return json.loads((REPORT_DIR / f"3.3a_{tag}.json").read_text(encoding="utf-8"))


def _fmt(v) -> str:
    return "nan" if v is None else f"{v:.3f}"


def report() -> str:
    b, a = load("before"), load("after")
    lines: List[str] = []
    scns = sorted({k.split("|")[1] for k in a["rows"]})
    # before / after for the targeted scenarios
    head = (f"{'scenario':<22}{'pipe':<10}{'':>7}{'correct':>8}{'corr_conf':>10}{'missed':>8}{'dup/run':>8}"
            f"{'err_med':>8}{'false':>7}")
    lines += ["Before / after, targeted scenarios (5 seeds)", head, "-" * len(head)]
    for s in BEFORE_AFTER:
        for p in a["rows"]:
            pn, sn = p.split("|")
            if sn != s:
                continue
            for lab, src in (("before", b), ("after", a)):
                r = src["rows"].get(p)
                if r is None:
                    lines.append(f"{s:<22}{pn:<10}{lab:>7}{'(scenario did not exist)':>40}")
                    continue
                lines.append(f"{s:<22}{pn:<10}{lab:>7}{_fmt(r['correct_rate']):>8}{_fmt(r['correct_confirmed_rate']):>10}"
                             f"{_fmt(r['missed_per_run']):>8}{_fmt(r['duplicate_per_run']):>8}"
                             f"{_fmt(r['err_median_m']):>8}{_fmt(r['false_per_run']):>7}")
    # A1 real survivors
    lines += ["", "A1 real-survivor non-regression (|after - before| <= 0.02)"]
    worst = 0.0
    for s in A1_REAL:
        for p in a["rows"]:
            pn, sn = p.split("|")
            if sn != s:
                continue
            for f in ("missed_per_run", "duplicate_per_run", "correct_confirmed_rate"):
                d = abs(a["rows"][p][f] - b["rows"][p][f])
                worst = max(worst, d)
                lines.append(f"  {s:<14}{pn:<10}{f:<24}{b['rows'][p][f]:.3f} -> {a['rows'][p][f]:.3f}  diff {d:.3f}")
    lines.append(f"  worst difference {worst:.3f}: {'PASS' if worst <= 0.02 + 1e-9 else 'FAIL'}")
    # A3
    lines += ["", "A3 regression gate (all rows outside the targeted scenarios, to 3 decimals)"]
    diffs = []
    n = 0
    for p, ra in a["rows"].items():
        pn, sn = p.split("|")
        if sn in TARGETED or sn not in {k.split("|")[1] for k in b["rows"]}:
            continue
        rb = b["rows"][p]
        for f in ra:
            n += 1
            if _fmt(ra[f]) != _fmt(rb[f]):
                diffs.append((p, f, _fmt(rb[f]), _fmt(ra[f])))
    lines.append(f"  {n} numbers compared, {len(diffs)} differ: {'PASS' if not diffs else 'FAIL'}")
    for d in diffs:
        lines.append(f"  DIFF {d[0]} {d[1]}: {d[2]} -> {d[3]}")
    # A1 double detection
    lines += ["", "A1 double_detection (per body offset; 5 instances per pipeline)"]
    for p in a["rows"]:
        pn, sn = p.split("|")
        if sn != "double_detection":
            continue
        runs = a["runs"][p]
        ok = sum(r["n_correct"] for r in runs) / sum(r["n_truth"] for r in runs)
        lost = sum(r["final_unreliable"] for r in runs)
        went = sum(r["went_unreliable"] for r in runs)
        lines.append(f"  {pn:<10}correct {ok:.2f} (>= 0.80), tracks ending UNRELIABLE by ambiguity {lost}, "
                     f"tracks that went UNRELIABLE by ambiguity {went}: "
                     f"{'PASS' if ok >= 0.8 and lost == 0 else 'FAIL'}")
    # A2 duplicates
    lines += ["", "A2 repeated_corrections duplicates per run (<= 0.2)"]
    for p in a["rows"]:
        pn, sn = p.split("|")
        if sn == "repeated_corrections":
            lines.append(f"  {pn:<10}before {_fmt(b['rows'][p]['duplicate_per_run'])} -> after "
                         f"{_fmt(a['rows'][p]['duplicate_per_run'])}: "
                         f"{'PASS' if a['rows'][p]['duplicate_per_run'] <= 0.2 + 1e-9 else 'FAIL'}")
    return "\n".join(lines)


def main(argv: List[str]) -> None:  # pragma: no cover
    if argv and argv[0] == "run":
        print("saved", run(argv[1]))
    elif argv and argv[0] == "report":
        text = report()
        (REPORT_DIR / "3.3a_eval.txt").write_text(text + "\n", encoding="utf-8")
        print(text)
    else:
        print(__doc__)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
