"""Substep 3.1c evaluation: Layer 4 calibration sweep, expected-value final-tag policy, confirm_prob sweep.

Run: PYTHONPATH=src python -m survivor_tagging.experiments_calib   (harness side; uses the simulator)
All results are SIMULATION tested on synthetic data and must be redone on recorded data in Step 5.

Method (fixed before running):
- Layers 1-3 depend only on (configured body offset, true body-depth model, scenario, seed). Their output batches are
  cached once (CachePipeline) and only Layer 4 is replayed per parameter setting. A test checks that a replay gives
  the same tags as Layer1234Pipeline.
- Calibration samples: after every Layer 4 update, each track that received an observation in that update and whose
  fused position lies within MATCH_RADIUS_M of a true survivor contributes (top-cell probability, top cell == true
  cell). Pool: all scenarios except loop_closure (needs Layer 5) and prone (reported separately).
- True mean bias: fixed model 0.25 m (0.5 m deep box); uniform 0.2-0.5 m depth -> 0.175 m.
- DECISION RULE (pre-registered by the user): a setting is ACCEPTABLE if for every (configured offset, true model)
  combination with |offset - true mean bias| <= 0.1 m: ECE over samples with p >= 0.5 is <= 0.05 AND no bin with >= 20
  samples is over-confident by more than 0.10. Select the acceptable setting with the smallest effective ray-bias
  sigma sqrt(floor^2 + offset_sigma^2) (ties: smaller obs_model_sigma, then smaller floor); select on seeds 0-2, report
  on held-out seeds 3-7.
- confirm_prob decision (fixed before running): 0.9 stays unless one alternative gives more confirmed_only points per
  run at EVERY penalty P in {20, 40, 60, 80}, at both offsets 0.0 and 0.2 and for both true body-depth models (P = 0
  excluded: there more is always better).
"""

from __future__ import annotations

import copy
import itertools
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .calibration import ece, format_reliability, reliability_table, worst_over_confidence
from .grid import GridRegistration
from .harness import MATCH_RADIUS_M, score_tags
from .pipeline import Layer1234Pipeline, tag_result
from .scenarios import all_scenarios
from .sim import SurvivorTruth, build_events, place_survivors
from .sim_pipelines import scenario_config
from .tracking import Layer4, Layer4Params, expected_value_threshold
from .types import TagStatus

OFFSETS = (0.0, 0.1, 0.2, 0.3)
MODELS = ("fixed", "uniform_0.2_0.5")
TRUE_MEAN_BIAS_M = {"fixed": 0.25, "uniform_0.2_0.5": 0.175}
FLOORS = (0.05, 0.10, 0.15, 0.20)
OFFSET_SIGMAS = (0.0, 0.05, 0.10, 0.15, 0.20)
OBS_MODELS = (0.03, 0.05, 0.08)
SELECT_SEEDS = (0, 1, 2)
HELDOUT_SEEDS = (3, 4, 5, 6, 7)
EXCLUDED = ("loop_closure", "prone")
PENALTIES = (0.0, 20.0, 40.0, 60.0, 80.0)
CORRECT_POINTS = 40.0  # rulebook p.19
CONFIRM_PROBS = (0.7, 0.8, 0.9, 0.95)
OUTPUT_STATUSES = (TagStatus.PROVISIONAL, TagStatus.CONFIRMED, TagStatus.UPDATED)
OLD = dict(ray_bias_sigma_m=0.10, offset_sigma_m=0.0, obs_model_sigma_m=0.05)  # Layer 4 before 3.1c


def rule_combos() -> List[Tuple[float, str]]:
    return [(o, m) for o in OFFSETS for m in MODELS if abs(o - TRUE_MEAN_BIAS_M[m]) <= 0.1 + 1e-9]


# ---------------------------------------------------------------------------
# Layer 1-3 cache and Layer 4 replay
# ---------------------------------------------------------------------------


class CachePipeline(Layer1234Pipeline):
    """Records the observation batches Layer1234Pipeline would hand to Layer 4, without running Layer 4."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.batches: List[list] = []

    def _flush(self) -> None:
        if self._batch:
            self.batches.append(self._batch)
            self._batch = []


@dataclass
class CachedRun:
    scenario: str
    seed: int
    offset: float
    model: str
    truths: List[SurvivorTruth]
    reg: GridRegistration
    batches: List[list]


def cache_run(scn, seed: int, offset: float, model: str) -> CachedRun:
    placed = place_survivors(scn, seed)
    pl = CachePipeline(scenario_config(placed, {"l1.body_depth_offset_m": offset}))
    for ev in build_events(placed, seed):
        if ev.kind == "pose":
            pl.process_pose(ev.payload, ev.arrival_ns)
        else:
            fr = ev.payload
            pl.process_frame(fr.detections, fr.depth, fr.intrinsics, ev.arrival_ns)
    pl.finalize()
    return CachedRun(scn.name, seed, offset, model, placed.truths(), pl.grid, pl.batches)


def _cache_task(args):
    offset, model, name, seeds = args
    scn = all_scenarios(body_depth=model)[name]
    return [cache_run(scn, s, offset, model) for s in seeds]


def _fresh(o):
    c = copy.copy(o)
    c.warnings = list(o.warnings)
    return c


def match_truth(xy: np.ndarray, truths: Sequence[SurvivorTruth]) -> Optional[SurvivorTruth]:
    if not truths:
        return None
    d = [float(np.linalg.norm(np.asarray(t.centre_map_xy) - xy)) for t in truths]
    i = int(np.argmin(d))
    return truths[i] if d[i] <= MATCH_RADIUS_M else None


def replay(run: CachedRun, params: Layer4Params, samples: Optional[list] = None) -> Layer4:
    """Runs Layer 4 on the cached batches. If samples is a list, appends (p_top, correct) per update (see module
    docstring)."""
    l4 = Layer4(params, run.reg)
    for batch in run.batches:
        b = [_fresh(o) for o in batch]
        l4.update(b)
        if samples is None:
            continue
        stamps = {o.capture_stamp_ns for o in b}
        for t in l4.tracks.values():
            if t.x is None or not (t.stamps() & stamps):
                continue
            truth = match_truth(t.x, run.truths)
            cell, prob = t.top()
            if truth is not None and cell is not None:
                samples.append((float(prob), tuple(cell) == tuple(truth.cell)))
    return l4


def l4_params(base_cfg, floor, offset_sigma, obs_model, **extra) -> Layer4Params:
    return Layer4Params.from_config(base_cfg, ray_bias_sigma_m=floor, offset_sigma_m=offset_sigma,
                                    obs_model_sigma_m=obs_model, **extra)


# ---------------------------------------------------------------------------
# Workers (the cache is sent once per worker process)
# ---------------------------------------------------------------------------

_CACHE: Dict[Tuple[float, str], List[CachedRun]] = {}
_BASE = None


def _init(cache):
    global _CACHE, _BASE
    _CACHE = cache
    _BASE = scenario_config(all_scenarios()["standing"])


@dataclass
class ComboResult:
    samples: list  # pooled (p, correct)
    prone: list
    n_runs: int = 0
    n_truth: int = 0
    correct_confirmed: int = 0
    false_confirmed: int = 0


def _evaluate_setting(args):
    """(floor, offset_sigma, obs_model, seeds, combos, extra) -> {combo: ComboResult}."""
    floor, osig, om, seeds, combos, extra = args
    p = l4_params(_BASE, floor, osig, om, **extra)
    out = {}
    for combo in combos:
        cr = ComboResult([], [])
        for run in _CACHE[combo]:
            if run.seed not in seeds or run.scenario == "loop_closure":
                continue
            prone = run.scenario == "prone"
            l4 = replay(run, p, cr.prone if prone else cr.samples)
            if prone:
                continue
            sc = score_tags(run.truths, [tag_result(t) for t in l4.tags() if t.status in OUTPUT_STATUSES])
            cr.n_runs += 1
            cr.n_truth += sc.n_truth
            cr.correct_confirmed += sc.n_correct_confirmed
            cr.false_confirmed += sc.n_false_confirmed
        out[combo] = cr
    return (floor, osig, om), out


def _commit_stats(truths, committed) -> Tuple[int, int, int]:
    """(correct, wrong = matched but wrong cell or duplicate, false positive = unmatched)."""
    sc = score_tags(truths, committed)
    n = sum(1 for t in committed if t.status != TagStatus.UNRELIABLE)
    return sc.n_correct, n - sc.n_correct - sc.n_false, sc.n_false


def _policy_task(args):
    """Final-tag policies for one combo with the chosen setting -> per policy: summed (correct, wrong, fp), runs."""
    floor, osig, om, combo, seeds, confirm_prob = args
    extra = {} if confirm_prob is None else {"confirm_prob": confirm_prob}
    p = l4_params(_BASE, floor, osig, om, **extra)
    thresholds = sorted({0.5} | {expected_value_threshold(P, CORRECT_POINTS)[0] for P in PENALTIES})
    pols = [("confirmed_only", None), ("confirmed_plus_provisional_top1", None)] + \
           [("expected_value", th) for th in thresholds]
    acc = {k: np.zeros(3) for k in pols}
    runs = 0
    for run in _CACHE[combo]:
        if run.seed not in seeds:
            continue
        runs += 1
        l4 = replay(run, p)
        for pol, th in pols:
            committed = [tag_result(t) for t in l4.final_tags(pol, th)]
            acc[(pol, th)] += _commit_stats(run.truths, committed)
    return combo, confirm_prob, acc, runs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _metrics(cr: ComboResult):
    p = [x for x, _ in cr.samples]
    c = [y for _, y in cr.samples]
    return ece(p, c, 10, p_min=0.5), worst_over_confidence(p, c, 10, 20)


def _summarise(results, combos):
    """Worst case over combos: ECE(p>=0.5), worst over-confidence, min correct CONFIRMED rate, max false CONF/run."""
    e, w, cc, fc = [], [], [], []
    for k in combos:
        cr = results[k]
        a, b = _metrics(cr)
        e.append(a)
        w.append(b)
        cc.append(cr.correct_confirmed / cr.n_truth)
        fc.append(cr.false_confirmed / cr.n_runs)
    nanmax = lambda xs: max((x for x in xs if not math.isnan(x)), default=math.nan)  # noqa: E731
    return nanmax(e), nanmax(w), min(cc), max(fc)


def acceptable(results, combos) -> bool:
    for k in combos:
        e, w = _metrics(results[k])
        if math.isnan(e) or e > 0.05 or (not math.isnan(w) and w > 0.10):
            return False
    return True


def eff_sigma(s) -> float:
    return math.hypot(s[0], s[1])


def main() -> None:  # pragma: no cover
    t0 = time.perf_counter()
    workers = max(1, (os.cpu_count() or 2) - 1)
    names = list(all_scenarios().keys())
    seeds = SELECT_SEEDS + HELDOUT_SEEDS
    tasks = [(o, m, n, seeds) for o in OFFSETS for m in MODELS for n in names]
    cache: Dict[Tuple[float, str], List[CachedRun]] = {(o, m): [] for o in OFFSETS for m in MODELS}
    with ProcessPoolExecutor(workers) as ex:
        for runs in ex.map(_cache_task, tasks):
            for r in runs:
                cache[(r.offset, r.model)].append(r)
    t_cache = time.perf_counter() - t0
    print(f"SIMULATION tested on synthetic data. Layers 1-3 cached: {sum(len(v) for v in cache.values())} runs "
          f"({len(names)} scenarios x {len(seeds)} seeds x {len(OFFSETS)} offsets x {len(MODELS)} models) in "
          f"{t_cache:.0f} s with {workers} worker processes\n")
    combos_all = list(cache.keys())
    combos_rule = rule_combos()
    print(f"Decision-rule combinations (|offset - true mean bias| <= 0.1 m): {combos_rule}\n")

    with ProcessPoolExecutor(workers, initializer=_init, initargs=(cache,)) as ex:
        # C. Sweep on seeds 0-2
        settings = list(itertools.product(FLOORS, OFFSET_SIGMAS, OBS_MODELS))
        grid = dict(ex.map(_evaluate_setting, [(f, s, m, SELECT_SEEDS, combos_all, {}) for f, s, m in settings]))
        t_sweep = time.perf_counter() - t0 - t_cache
        print(f"== C. Calibration sweep, seeds 0-2 ({len(settings)} settings x {len(combos_all)} combos; "
              f"{t_sweep:.0f} s) ==")
        print("rule = worst over rule combos; all = worst over all 8 combos (incl. offsets far from the truth)")
        print(f"{'floor':>6}{'osig':>6}{'obs':>6}{'eff':>7}{'ECE_rule':>9}{'over_rule':>10}{'cc_rule':>8}"
              f"{'fc_rule':>8}{'ECE_all':>8}{'over_all':>9}{'cc_all':>7}{'fc_all':>7}  acceptable")
        ok = []
        for s in sorted(settings, key=lambda s: (eff_sigma(s), s[2], s[0])):
            r = grid[s]
            a = _summarise(r, combos_rule)
            b = _summarise(r, combos_all)
            acc = acceptable(r, combos_rule)
            if acc:
                ok.append(s)
            print(f"{s[0]:>6.2f}{s[1]:>6.2f}{s[2]:>6.2f}{eff_sigma(s):>7.3f}{a[0]:>9.3f}{a[1]:>+10.3f}{a[2]:>8.3f}"
                  f"{a[3]:>8.2f}{b[0]:>8.3f}{b[1]:>+9.3f}{b[2]:>7.3f}{b[3]:>7.2f}  {'YES' if acc else 'no'}")
        if ok:
            chosen = ok[0]
            print(f"\nACCEPTABLE settings: {len(ok)}. CHOSEN (smallest effective sigma): floor {chosen[0]}, "
                  f"offset_sigma {chosen[1]}, obs_model {chosen[2]} (effective {eff_sigma(chosen):.3f} m)")
        else:
            def shortfall(s):
                m = [_metrics(grid[s][k]) for k in combos_rule]
                return max(max(e - 0.05, 0) + max((w if not math.isnan(w) else 0) - 0.10, 0) for e, w in m)
            chosen = min(settings, key=shortfall)
            print(f"\nNO ACCEPTABLE SETTING. Best (smallest summed shortfall): {chosen}, shortfall "
                  f"{shortfall(chosen):.3f}")
        print("Per rule combo for the chosen setting (seeds 0-2): " + "; ".join(
            f"off {k[0]:.1f} {k[1]}: ECE {_metrics(grid[chosen][k])[0]:.3f} over {_metrics(grid[chosen][k])[1]:+.3f}"
            for k in combos_rule))

        # Held-out seeds 3-7: chosen and old
        held = dict(ex.map(_evaluate_setting, [(*chosen, HELDOUT_SEEDS, combos_all, {}),
                                               (OLD["ray_bias_sigma_m"], OLD["offset_sigma_m"],
                                                OLD["obs_model_sigma_m"], HELDOUT_SEEDS, combos_all, {})]))
        old_key = (OLD["ray_bias_sigma_m"], OLD["offset_sigma_m"], OLD["obs_model_sigma_m"])
        hc = held[tuple(chosen)]
        passed = acceptable(hc, combos_rule)
        print(f"\n== C4. Held-out seeds 3-7, chosen setting: {'PASS' if passed else 'FAIL'} ==")
        for k in combos_all:
            e, w = _metrics(hc[k])
            cr = hc[k]
            mark = "*" if k in combos_rule else " "
            print(f"{mark} offset {k[0]:.1f} {k[1]:<16} ECE(p>=0.5) {e:.3f}  worst bin over {w:+.3f}  n {len(cr.samples)}"
                  f"  correct CONF {cr.correct_confirmed / cr.n_truth:.3f}  false CONF/run "
                  f"{cr.false_confirmed / cr.n_runs:.2f}  | prone ECE "
                  f"{ece([x for x, _ in cr.prone], [y for _, y in cr.prone], 10, 0.5):.3f}")
        print("(* = decision-rule combination)")

        print("\n== Reliability tables, held-out seeds 3-7, true model fixed (before = Layer 4 before 3.1c) ==")
        for off in (0.0, 0.2):
            for label, res in (("before", held[old_key]), ("after", hc)):
                cr = res[(off, "fixed")]
                p = [x for x, _ in cr.samples]
                c = [y for _, y in cr.samples]
                print(f"-- offset {off:.1f}, {label}: ECE(p>=0.5) {ece(p, c, 10, 0.5):.3f}, ECE(all) "
                      f"{ece(p, c):.3f}, n {len(p)}")
                print(format_reliability(reliability_table(p, c)))
        print("-- prone (excluded from the pool), offset 0.2 fixed, after:")
        pr = hc[(0.2, "fixed")].prone
        print(format_reliability(reliability_table([x for x, _ in pr], [y for _, y in pr])))

        # D. Policies and E. confirm_prob, all 20 scenarios, seeds 0-7
        ptasks = [(*chosen, (off, m), seeds, cp) for off in (0.0, 0.2) for m in MODELS
                  for cp in (None,) + CONFIRM_PROBS]
        pres = list(ex.map(_policy_task, ptasks))
    print(f"\n== D. Final-tag policies (chosen setting, all 20 scenarios, seeds 0-7; +{CORRECT_POINTS:.0f} per correct "
          "tag and -P per wrong/extra tag, both ASSUMED for scoring O2) ==")
    for combo, cp, acc, runs in pres:
        if cp is not None:
            continue
        print(f"-- offset {combo[0]:.1f}, true model {combo[1]} ({runs} runs); points per run at P = "
              f"{', '.join(f'{P:.0f}' for P in PENALTIES)}; correct / wrong / false-positive tags per run")
        for (pol, th), v in acc.items():
            pts = [(CORRECT_POINTS * v[0] - P * (v[1] + v[2])) / runs for P in PENALTIES]
            name = pol if th is None else f"expected_value th={th:.3f}"
            print(f"{name:<34}" + "".join(f"{x:>8.2f}" for x in pts) +
                  f"   {v[0] / runs:.2f} / {v[1] / runs:.2f} / {v[2] / runs:.2f}")
        best = {}
        for P in PENALTIES:
            evs = {th: (CORRECT_POINTS * v[0] - P * (v[1] + v[2])) / runs
                   for (pol, th), v in acc.items() if pol == "expected_value"}
            best[P] = max(evs, key=evs.get)
        print("best expected_value threshold per P: " + ", ".join(
            f"P={P:.0f}: {best[P]:.3f} (formula {expected_value_threshold(P, CORRECT_POINTS)[0]:.3f})"
            for P in PENALTIES))

    print("\n== E. confirm_prob sweep (chosen setting, confirmed_only commit, all 20 scenarios, seeds 0-7) ==")
    table = {}
    for combo, cp, acc, runs in pres:
        if cp is None:
            continue
        v = acc[("confirmed_only", None)]
        table[(combo, cp)] = [(CORRECT_POINTS * v[0] - P * (v[1] + v[2])) / runs for P in PENALTIES]
        print(f"offset {combo[0]:.1f} {combo[1]:<16} confirm_prob {cp:.2f}: correct {v[0] / runs:.2f} wrong "
              f"{v[1] / runs:.2f} fp {v[2] / runs:.2f} per run; points at P=0..80: "
              + " ".join(f"{x:.2f}" for x in table[(combo, cp)]))
    change = None
    for cp in CONFIRM_PROBS:
        if cp == 0.9:
            continue
        if all(table[((off, m), cp)][i] > table[((off, m), 0.9)][i] for off in (0.0, 0.2) for m in MODELS
               for i in range(1, len(PENALTIES))):
            change = cp
    print(f"confirm_prob decision: {'CHANGE to ' + str(change) if change else 'keep 0.9'} (rule in module docstring)")
    print(f"\ntotal {time.perf_counter() - t0:.0f} s (Python {sys.version.split()[0]})")


if __name__ == "__main__":  # pragma: no cover
    main()
