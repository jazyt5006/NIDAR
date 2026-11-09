"""Substep 3.3b evaluation: per-scenario calibration breakdown (A), A3 quantification (R-A), merge distance (R-B),
interim final-tag policy (R-C), full ablation (C). Harness side (uses the simulator). SIMULATION tested only.

Run: PYTHONPATH=src python -m survivor_tagging.experiments_33b run      (computes every stage, caches raw rows)
     PYTHONPATH=src python -m survivor_tagging.experiments_33b report   (tables from the cache; writes 3.3b_eval.txt)
Raw per-run rows are cached in docs/reports/3.3b_cache_<stage>.json; `run` skips a stage whose cache file exists
(delete the file to recompute). Everything is seeded.

Pre-registered rules (substep 3.3b, followed as written):
R-A  per-run mission points (+40 per correct committed tag, P in {0, 40} per wrong/extra committed tag, ASSUMED) for
     revisit_after_gap, opposite_views_boundary, thin_person_in_front_of_wall: before (l4.min_separate_survivor_
     distance_m = 0, the 3.2 behaviour) vs after (0.5). Report. Flag loudly if the net change at P = 40 is below -1.0
     point per run pooled over the three scenarios.
R-B  scenario family two_close_survivors (centres 0.30-0.60 m apart, adjacent and same cell, standing/sitting). Sweep
     l4.min_separate_survivor_distance_m in {0, 0.3, 0.4, 0.5}; choose the LARGEST value whose missed-survivor rate is
     <= the value at 0 + 0.02 at BOTH body offsets (0.0, 0.2; 5 seeds). Change the default only if it differs.
R-C  final-tag policy (INTERIM): candidates confirmed_only and expected_value with threshold 0.5, 0.6, 0.667; the four
     3.1c calibration combinations (0.1 uniform_0.2_0.5, 0.2 fixed, 0.2 uniform_0.2_0.5, 0.3 fixed) x P in
     {0, 20, 40, 60, 80}; choose the candidate with the highest WORST-CASE points per run over all those cells.
Points per run = 40 * correct committed tags - P * (other committed tags); UNRELIABLE tags are never committed.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .baseline import naive_factory
from .calibration import ece, reliability_table
from .experiments_calib import OUTPUT_STATUSES, cache_run, replay, rule_combos
from .harness import evaluate_many, score_tags
from .pipeline import tag_result
from .scenarios import TWO_CLOSE_PREFIX, all_scenarios, pre_32_scenarios, two_close_names
from .sim_pipelines import (layer123_factory, layer1234_factory, layer5_factory, naive_l1_factory, scenario_config)
from .tracking import Layer4Params
from .types import TagStatus

REPORT_DIR = Path("docs/reports")
SEEDS = (0, 1, 2, 3, 4)
OFFSETS = (0.0, 0.2)
CORRECT_POINTS = 40.0  # rulebook p.19
PENALTIES = (0.0, 20.0, 40.0, 60.0, 80.0)
MIN_SEPS = (0.0, 0.3, 0.4, 0.5)
RA_SCENARIOS = ("revisit_after_gap", "opposite_views_boundary", "thin_person_in_front_of_wall")
CANDIDATES = {"confirmed_only": ("confirmed_only", None), "ev_0.500": ("expected_value", 0.5),
              "ev_0.600": ("expected_value", 0.6), "ev_0.667": ("expected_value", 0.667)}
CALIB_EXCLUDED = ("loop_closure", "prone")  # as in 3.1c
SHOW_ROWS = ("prone", "tilt_45", "cell_boundary", "loop_closure", "repeated_corrections", "double_detection")


def _cache_path(stage: str) -> Path:
    return REPORT_DIR / f"3.3b_cache_{stage}.json"


def _workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def _scenario(name: str, model: str = "fixed"):
    return all_scenarios(body_depth=model)[name]


# ---------------------------------------------------------------------------
# Stage A: per-scenario calibration samples (Layer 1-3 cache + Layer 4 replay with the current defaults)
# ---------------------------------------------------------------------------


def _calib_task(args):
    off, model, name, seeds = args
    scn = _scenario(name, model)
    out = []
    for s in seeds:
        run = cache_run(scn, s, off, model)
        base = scenario_config(scn, {"l1.body_depth_offset_m": off})
        samples: list = []
        replay(run, Layer4Params.from_config(base), samples)
        out.extend([[p, bool(c)] for p, c in samples])
    return off, model, name, out


def stage_calib() -> dict:
    names = [n for n in pre_32_scenarios() if n not in CALIB_EXCLUDED]
    tasks = [(o, m, n, tuple(range(8))) for o in OFFSETS for m in ("fixed", "uniform_0.2_0.5") for n in names]
    res: Dict[str, list] = {}
    with ProcessPoolExecutor(_workers()) as ex:
        for off, model, name, samples in ex.map(_calib_task, tasks):
            res[f"{off}|{model}|{name}"] = samples
    return res


# ---------------------------------------------------------------------------
# Stage B: merge-distance sweep on the two_close_survivors family (Layer 1-3 cache + Layer 4 replay)
# ---------------------------------------------------------------------------


def _sweep_task(args):
    off, name, seeds = args
    scn = _scenario(name)
    rows = []
    for s in seeds:
        run = cache_run(scn, s, off, "fixed")
        base = scenario_config(scn, {"l1.body_depth_offset_m": off})
        for sep in MIN_SEPS:
            l4 = replay(run, Layer4Params.from_config(base, min_separation_m=sep))
            sc = score_tags(run.truths, [tag_result(t) for t in l4.tags() if t.status in OUTPUT_STATUSES])
            rows.append(dict(off=off, scenario=name, seed=s, sep=sep, n_truth=sc.n_truth, n_missed=sc.n_missed,
                             n_correct=sc.n_correct, n_dup=sc.n_duplicate, n_false=sc.n_false))
    return rows


def stage_sweep() -> list:
    tasks = [(o, n, SEEDS) for o in OFFSETS for n in two_close_names()]
    with ProcessPoolExecutor(_workers()) as ex:
        return [r for rows in ex.map(_sweep_task, tasks) for r in rows]


# ---------------------------------------------------------------------------
# Stage runs: end-to-end pipeline rows (R-A, R-C, C)
# ---------------------------------------------------------------------------

PIPELINES = ("naive", "l1", "l123", "l1234", "l15")


def _factory(kind: str, off: float, l4: dict):
    ov = {"l1.body_depth_offset_m": off}
    if kind == "naive":
        return naive_factory
    if kind == "l1":
        return naive_l1_factory(ov)
    if kind == "l123":
        return layer123_factory(ov)
    if kind == "l1234":
        return layer1234_factory(ov, **l4)
    if kind == "l15":
        return layer5_factory(ov, **l4)
    raise ValueError(kind)


def _stats(truths, committed) -> List[float]:
    sc = score_tags(truths, committed)
    n = sum(1 for t in committed if t.status != TagStatus.UNRELIABLE)
    return [sc.n_correct, n]


def _run_task(args):
    """(kind, offset, model, scenario, seeds, l4 overrides) -> per-run rows."""
    kind, off, model, name, seeds, l4 = args
    scn = _scenario(name, model)
    ev = evaluate_many({"p": _factory(kind, off, l4)}, [scn], list(seeds))
    rows = []
    for seed, r in zip(seeds, ev.results["p"][name]):
        sc = r.score
        pol = {"output": _stats(r.truths, r.tags)}
        if kind in ("l1234", "l15"):
            for key, (policy, th) in CANDIDATES.items():
                pol[key] = _stats(r.truths, r.pipeline.final_tags(policy, th))
        rows.append(dict(kind=kind, off=off, model=model, scenario=name, seed=seed, l4=l4, n_truth=sc.n_truth,
                         n_correct=sc.n_correct, n_missed=sc.n_missed, n_false=sc.n_false, n_dup=sc.n_duplicate,
                         cc=sc.n_correct_confirmed, fc=sc.n_false_confirmed, fp=sc.n_false_provisional,
                         flicker=sc.n_flicker, err=sc.errors_m, conf=sc.confirm_times_s,
                         conf_first=sc.confirm_from_first_obs_s, pol=pol, runtime_s=r.runtime_s))
    return rows


def _run_all(tasks: list) -> list:
    out = []
    with ProcessPoolExecutor(_workers()) as ex:
        for rows in ex.map(_run_task, tasks, chunksize=1):
            out.extend(rows)
    return out


def stage_rc() -> list:
    """Layers 1-5, all scenarios, the four calibration combinations, seeds 0-4."""
    names = list(all_scenarios())
    tasks = [("l15", o, m, n, SEEDS, {}) for o, m in rule_combos() for n in names]
    return _run_all(tasks)


def stage_ablation() -> list:
    names = list(all_scenarios())
    tasks = [("naive", 0.0, "fixed", n, SEEDS, {}) for n in names]
    for off in OFFSETS:
        for kind in ("l1", "l123", "l1234", "l15"):
            if kind == "l15" and (off, "fixed") in rule_combos():
                continue  # identical run is in stage rc (offset 0.2 fixed)
            tasks += [(kind, off, "fixed", n, SEEDS, {}) for n in names]
    return _run_all(tasks)


def stage_ra() -> list:
    tasks = []
    for off in OFFSETS:
        for sep in (0.0, 0.5):
            tasks += [("l15", off, "fixed", n, SEEDS, {"min_separation_m": sep}) for n in RA_SCENARIOS]
    return _run_all(tasks)


STAGES = (("calib", stage_calib), ("sweep", stage_sweep), ("rc", stage_rc), ("ablation", stage_ablation),
          ("ra", stage_ra))


def run_all() -> None:  # pragma: no cover
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in STAGES:
        path = _cache_path(name)
        if path.exists():
            print(f"stage {name}: cached ({path})")
            continue
        t0 = time.perf_counter()
        data = fn()
        path.write_text(json.dumps(data), encoding="utf-8")
        print(f"stage {name}: {time.perf_counter() - t0:.0f} s -> {path}", flush=True)


def load(stage: str):
    return json.loads(_cache_path(stage).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def points(c: float, n: float, penalty: float) -> float:
    return CORRECT_POINTS * c - penalty * (n - c)


def _mean_points(rows: Sequence[dict], key: str, penalty: float) -> float:
    return float(np.mean([points(r["pol"][key][0], r["pol"][key][1], penalty) for r in rows])) if rows else math.nan


def report_calib() -> List[str]:
    d = load("calib")
    lines = ["== A. Per-scenario calibration (Layer 4 current defaults, seeds 0-7, scenarios as in 3.1c: "
             "loop_closure and prone excluded) =="]
    lines.append("ECE over samples with p >= 0.5; worst = the most over-confident bin (10 bins) with >= 20 samples "
                 "(mean predicted - observed); '-' = no bin with 20 samples.")
    contrib: Dict[str, Dict[Tuple[float, str], Tuple[int, float]]] = {}
    for model in ("fixed", "uniform_0.2_0.5"):
        for off in OFFSETS:
            lines.append(f"-- offset {off:.1f}, true model {model}")
            lines.append(f"{'scenario':<30}{'n':>6}{'n(p>=.5)':>9}{'ECE':>7}{'worst bin':>11}{'over':>8}"
                         f"{'bin 0.6-0.7: n':>15}{'pred':>7}{'obs':>7}")
            allp, allc = [], []
            for key in sorted(d):
                o, m, name = key.split("|")
                if float(o) != off or m != model:
                    continue
                s = d[key]
                p = [x for x, _ in s]
                c = [y for _, y in s]
                allp += p
                allc += c
                rows = reliability_table(p, c) if s else []
                big = [r for r in rows if r.count >= 20]
                w = max(big, key=lambda r: r.over_confidence) if big else None
                b67 = rows[6] if rows else None
                n5 = sum(1 for x in p if x >= 0.5)
                e = ece(p, c, 10, 0.5) if s else math.nan
                lines.append(f"{name:<30}{len(s):>6}{n5:>9}{e:>7.3f}"
                             f"{(f'{w.lo:.1f}-{w.hi:.1f}' if w else '-'):>11}"
                             f"{(f'{w.over_confidence:+.3f}' if w else '-'):>8}"
                             f"{(b67.count if b67 else 0):>15}"
                             f"{(f'{b67.mean_pred:.3f}' if b67 and b67.count else '-'):>7}"
                             f"{(f'{b67.frac_correct:.3f}' if b67 and b67.count else '-'):>7}")
                if b67 is not None and b67.count:
                    contrib.setdefault(name, {})[(off, model)] = (b67.count, b67.over_confidence)
            rows = reliability_table(allp, allc)
            b67 = rows[6]
            lines.append(f"{'POOLED':<30}{len(allp):>6}{sum(1 for x in allp if x >= 0.5):>9}"
                         f"{ece(allp, allc, 10, 0.5):>7.3f}{'':>19}{b67.count:>15}{b67.mean_pred:>7.3f}"
                         f"{b67.frac_correct:>7.3f}")
    lines.append("")
    lines.append("Scenarios behind the 0.6-0.7 bin over-confidence (offset 0.2, fixed): share of the bin's total "
                 "over-confidence mass sum(n x over), scenarios with a positive contribution only")
    tot = {}
    for name, dd in contrib.items():
        if (0.2, "fixed") in dd:
            n, ov = dd[(0.2, "fixed")]
            tot[name] = n * ov
    pos = {k: v for k, v in tot.items() if v > 0}
    z = sum(pos.values())
    for k, v in sorted(pos.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k:<30} n {contrib[k][(0.2, 'fixed')][0]:>4}  over {contrib[k][(0.2, 'fixed')][1]:+.3f}  "
                     f"share {v / z:.2f}")
    return lines


def report_sweep() -> Tuple[List[str], Optional[float]]:
    rows = load("sweep")
    lines = ["== R-B. two_close_survivors: merge-distance sweep (Layers 1-4, 8 pair types x 5 seeds, 2 survivors each) =="]
    lines.append(f"{'min_sep':>8}{'offset':>8}{'survivors':>10}{'missed':>8}{'missed rate':>12}{'correct':>9}"
                 f"{'dup/run':>9}{'false/run':>10}")
    rate: Dict[Tuple[float, float], float] = {}
    for sep in MIN_SEPS:
        for off in OFFSETS:
            sel = [r for r in rows if r["sep"] == sep and r["off"] == off]
            nt = sum(r["n_truth"] for r in sel)
            miss = sum(r["n_missed"] for r in sel)
            rate[(sep, off)] = miss / nt
            lines.append(f"{sep:>8.2f}{off:>8.1f}{nt:>10}{miss:>8}{miss / nt:>12.3f}"
                         f"{sum(r['n_correct'] for r in sel) / nt:>9.3f}{sum(r['n_dup'] for r in sel) / len(sel):>9.2f}"
                         f"{sum(r['n_false'] for r in sel) / len(sel):>10.2f}")
    chosen = None
    lines.append("rule: largest min_sep with missed rate <= (rate at 0) + 0.02 at BOTH offsets")
    for sep in MIN_SEPS:
        ok = all(rate[(sep, off)] <= rate[(0.0, off)] + 0.02 + 1e-12 for off in OFFSETS)
        lines.append(f"  min_sep {sep:.1f}: " + ", ".join(
            f"off {off:.1f} {rate[(sep, off)]:.3f} vs limit {rate[(0.0, off)] + 0.02:.3f}" for off in OFFSETS)
            + f" -> {'OK' if ok else 'no'}")
        if ok:
            chosen = sep
    lines.append(f"CHOSEN: {chosen}")
    lines.append("-- per pair type, missed rate at offsets 0.0 / 0.2 for min_sep " + " / ".join(f"{s:.1f}" for s in MIN_SEPS))
    for n in two_close_names():
        cells = []
        for off in OFFSETS:
            cells.append(" ".join(
                f"{sum(r['n_missed'] for r in rows if r['scenario'] == n and r['sep'] == s and r['off'] == off) / sum(r['n_truth'] for r in rows if r['scenario'] == n and r['sep'] == s and r['off'] == off):.2f}"
                for s in MIN_SEPS))
        lines.append(f"  {n[len(TWO_CLOSE_PREFIX):]:<16} off 0.0: {cells[0]}   off 0.2: {cells[1]}")
    return lines, chosen


def report_rc() -> Tuple[List[str], str]:
    rows = load("rc")
    lines = ["== R-C. Interim final-tag policy (Layers 1-5, all %d scenarios, 5 seeds; points per run) ==" %
             len({r['scenario'] for r in rows})]
    worst = {k: math.inf for k in CANDIDATES}
    where = {}
    combos = rule_combos()
    for off, model in combos:
        sel = [r for r in rows if r["off"] == off and r["model"] == model]
        lines.append(f"-- offset {off:.1f}, true model {model} ({len(sel)} runs); points per run at P = "
                     f"{', '.join(f'{p:.0f}' for p in PENALTIES)}")
        for k in CANDIDATES:
            pts = [_mean_points(sel, k, P) for P in PENALTIES]
            lines.append(f"{k:<16}" + "".join(f"{x:>8.2f}" for x in pts))
            for P, x in zip(PENALTIES, pts):
                if x < worst[k]:
                    worst[k], where[k] = x, (off, model, P)
    lines.append("worst case over the 4 combinations x 5 penalties:")
    for k in CANDIDATES:
        off, model, P = where[k]
        lines.append(f"  {k:<16}{worst[k]:>8.2f}   (at offset {off:.1f} {model}, P = {P:.0f})")
    best = max(worst, key=worst.get)
    lines.append(f"CHOSEN (INTERIM, per-P maximin): {best}")
    return lines, best


def report_ra(policy: str) -> List[str]:
    rows = load("ra")
    lines = [f"== R-A. A3 quantification, Layers 1-5 final_tags({policy}), 5 seeds, mission points per run =="]
    lines.append("before = l4.min_separate_survivor_distance_m 0 (3.2 behaviour), after = 0.5. ASSUMED scoring: +40 per "
                 "correct committed tag, -P per other committed tag.")
    head = f"{'scenario':<30}{'offset':>7}{'P':>4}{'before':>8}{'after':>8}{'change':>8}"
    lines += [head, "-" * len(head)]
    flag = False
    pooled = {}
    for P in (0.0, 40.0):
        for off in OFFSETS + (None,):
            for scn in RA_SCENARIOS + ("POOLED",):
                def sel(sep):
                    return [r for r in rows if r["l4"]["min_separation_m"] == sep and (off is None or r["off"] == off)
                            and (scn == "POOLED" or r["scenario"] == scn)]
                b, a = _mean_points(sel(0.0), policy_key(policy), P), _mean_points(sel(0.5), policy_key(policy), P)
                if off is None and scn == "POOLED":
                    pooled[P] = (b, a)
                lines.append(f"{scn:<30}{('both' if off is None else f'{off:.1f}'):>7}{P:>4.0f}{b:>8.2f}{a:>8.2f}"
                             f"{a - b:>+8.2f}")
    chg = pooled[40.0][1] - pooled[40.0][0]
    if chg < -1.0:
        flag = True
    lines.append(f"pooled net change at P = 40: {chg:+.2f} points per run (flag threshold -1.0): "
                 f"{'*** FLAG: NEGATIVE BY MORE THAN 1.0 ***' if flag else 'not flagged'}; at P = 0: "
                 f"{pooled[0.0][1] - pooled[0.0][0]:+.2f}")
    return lines


def policy_key(policy: str) -> str:
    return policy


def _pool(rows: Sequence[dict], policy: Optional[str]) -> dict:
    n = len(rows)
    nt = sum(r["n_truth"] for r in rows)
    errs = [e for r in rows for e in r["err"]]
    cf = [c for r in rows for c in r["conf_first"]]
    key = policy if (policy is not None and "pol" in rows[0] and policy in rows[0]["pol"]) else "output"
    return dict(runs=n, correct=sum(r["n_correct"] for r in rows) / nt, cc=sum(r["cc"] for r in rows) / nt,
                missed=sum(r["n_missed"] for r in rows) / n, fc=sum(r["fc"] for r in rows) / n,
                fp=sum(r["fp"] for r in rows) / n, dup=sum(r["n_dup"] for r in rows) / n,
                err=float(np.median(errs)) if errs else math.nan, tconf=float(np.median(cf)) if cf else math.nan,
                p0=_mean_points(rows, key, 0.0), p40=_mean_points(rows, key, 40.0))


def report_ablation(policy: str) -> List[str]:
    abl = load("ablation")
    rc = load("rc")
    rows = abl + [r for r in rc if r["off"] == 0.2 and r["model"] == "fixed"]
    labels = {"naive": "naive", "l1": "Layer 1", "l123": "Layers 1-3 (1 tag/id)", "l1234": "Layers 1-4",
              "l15": "Layers 1-5"}
    lines = [f"== C. Ablation (5 seeds, random placement, true body model fixed; L1-4 and L1-5 points use final_tags"
             f"({policy}) [INTERIM]; other rows commit every output tag) =="]
    head = (f"{'pipeline':<24}{'off':>5}{'runs':>5}{'corr_any':>9}{'corr_conf':>10}{'missed':>7}{'f_conf':>7}"
            f"{'f_prov':>7}{'dup':>6}{'err_med':>8}{'t_conf':>7}{'pts P=0':>9}{'pts P=40':>9}")

    def block(title: str, scn: Optional[Sequence[str]]) -> None:
        lines.append(f"-- {title}")
        lines.append(head)
        for kind in PIPELINES:
            for off in ((0.0,) if kind == "naive" else OFFSETS):
                sel = [r for r in rows if r["kind"] == kind and r["off"] == off and r["model"] == "fixed"
                       and (scn is None or r["scenario"] in scn)]
                if not sel:
                    continue
                p = _pool(sel, policy)
                lines.append(f"{labels[kind]:<24}{off:>5.1f}{p['runs']:>5}{p['correct']:>9.3f}{p['cc']:>10.3f}"
                             f"{p['missed']:>7.2f}{p['fc']:>7.2f}{p['fp']:>7.2f}{p['dup']:>6.2f}{p['err']:>8.3f}"
                             f"{p['tconf']:>7.2f}{p['p0']:>9.2f}{p['p40']:>9.2f}")

    n_scn = len({r["scenario"] for r in rows})
    block(f"pooled, all {n_scn} scenarios (the two_close_survivors family counts as its 8 members)", None)
    for s in SHOW_ROWS:
        block(s, [s])
    block("two_close_survivors (8 pair types pooled)", list(two_close_names()))
    return lines


def report() -> str:
    lines: List[str] = ["SIMULATION tested on synthetic data (3.3b). Points are ASSUMED-scoring (O2 unknown).", ""]
    lines += report_calib()
    sweep, chosen = report_sweep()
    rc, best = report_rc()
    lines += [""] + sweep + [""] + rc + [""] + report_ra(best) + [""] + report_ablation(best)
    return "\n".join(lines)


def main(argv: List[str]) -> None:  # pragma: no cover
    if argv and argv[0] == "run":
        run_all()
    elif argv and argv[0] == "report":
        text = report()
        (REPORT_DIR / "3.3b_eval.txt").write_text(text + "\n", encoding="utf-8")
        print(text)
    else:
        print(__doc__)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
