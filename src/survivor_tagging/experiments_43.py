"""Substep 4.3 evaluation: R1 (wall-alignment accuracy and basin), R2 (registration uncertainty helps), R3 (update
equivalence). SIMULATION evidence only. `PYTHONPATH=src python -m survivor_tagging.experiments_43` writes
docs/reports/4.3_eval.txt (and prints it)."""

from __future__ import annotations

import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .calibration import ece
from .experiments_layer5 import feed
from .harness import run_scenario
from .registration import WallAlignParams, estimate_wall_alignment
from .scenarios import (NEW_IN_43, REG_ERROR_ORIGIN_SIGMA_M, REG_ERROR_YAW_SIGMA_RAD, all_scenarios)
from .sim import Registration, build_events, place_survivors
from .sim_pipelines import layer5_factory, library_registration
from .sim_walls import grid_to_map_points, maze_arena, sample_wall_points_grid, scenario_wall_points

SEEDS = (0, 1, 2, 3, 4)
R1_LAYOUTS_PER_SEED = 10
BASIN_YAW_DEG = (0, 2, 5, 10, 15, 20, 30, 40)
BASIN_OFFSET_M = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
BASIN_LAYOUTS_PER_SEED = 3
SUCCESS_YAW_DEG, SUCCESS_OFFSET_M = 1.5, 0.10  # the p95 limits of R1 define "success"
REPORT = Path("docs/reports/4.3_eval.txt")


def _wrap_deg(a_rad: float) -> float:
    return abs((a_rad + math.pi) % (2 * math.pi) - math.pi) * 180.0 / math.pi


def make_trial(seed: int, k: int, yaw_err_deg: float, off_err_m: float, params: WallAlignParams = None,
               rng_mode: str = "uniform"):
    """One synthetic-maze trial: layout k of seed `seed`, random orientation / origin, 0.03 m noise, 10-30 % outliers,
    50-90 % wall coverage; coarse registration = truth + (yaw_err_deg, off_err_m) in a random direction.
    Returns (yaw error deg, origin error m, WallAlignResult)."""
    rng = np.random.default_rng([seed, k, 43])
    n = int(rng.choice([5, 6, 8, 10, 12, 15]))
    arena = maze_arena(n, seed * 1000 + k)
    true_sim = Registration((float(rng.uniform(-3, 3)), float(rng.uniform(-3, 3))), float(rng.uniform(-math.pi, math.pi)))
    gp = sample_wall_points_grid(arena.all_walls(1.0), seed * 1000 + k, noise_sigma_m=0.03,
                                 outlier_fraction=float(rng.uniform(0.1, 0.3)), coverage=float(rng.uniform(0.5, 0.9)),
                                 extent_m=(-0.5, n + 0.5, -0.5, n + 0.5))
    pts = grid_to_map_points(true_sim, gp)
    true = library_registration(true_sim, n)
    dyaw = math.radians(yaw_err_deg) * float(rng.choice([-1.0, 1.0]))
    ang = float(rng.uniform(0, 2 * math.pi))
    off = off_err_m * np.array([math.cos(ang), math.sin(ang)])
    coarse = replace(true, yaw_rad=true.yaw_rad + dyaw,
                     origin_in_map_m=tuple(np.asarray(true.origin_in_map_m) + off))
    res = estimate_wall_alignment(pts, coarse, params)
    if res.registration is None:
        return math.inf, math.inf, res
    return (_wrap_deg(res.registration.yaw_rad - true.yaw_rad),
            float(np.linalg.norm(np.asarray(res.registration.origin_in_map_m) - np.asarray(true.origin_in_map_m))), res)


def r1_main(seeds: Sequence[int] = SEEDS, layouts: int = R1_LAYOUTS_PER_SEED):
    rows = []
    for s in seeds:
        for k in range(layouts):
            rng = np.random.default_rng([s, k, 7])
            ye, oe, res = make_trial(s, k, float(rng.uniform(0, 10)), float(rng.uniform(0, 0.4)))
            rows.append((ye, oe, res.quality, res.reason))
    return rows


def r1_basin(seeds: Sequence[int] = SEEDS, layouts: int = BASIN_LAYOUTS_PER_SEED):
    table = {}
    for yd in BASIN_YAW_DEG:
        for om in BASIN_OFFSET_M:
            ok, n = 0, 0
            for s in seeds:
                for k in range(layouts):
                    ye, oe, _ = make_trial(s, 100 + k, float(yd), float(om))
                    n += 1
                    ok += int(ye <= SUCCESS_YAW_DEG and oe <= SUCCESS_OFFSET_M)
            table[(yd, om)] = ok / n
    return table


def format_r1(rows, basin) -> List[str]:
    ye = np.array([r[0] for r in rows])
    oe = np.array([r[1] for r in rows])
    n_fail = int(np.sum(~np.isfinite(ye)))
    out = [f"R1. Wall-alignment accuracy: {len(rows)} synthetic mazes (5 seeds x {len(rows) // 5} layouts; 5-15 cells, random "
           "orientation, noise 0.03 m, 10-30 % outliers, 50-90 % coverage); coarse error yaw U(0,10) deg, offset U(0,0.4) m",
           f"  estimator returned no update in {n_fail} trial(s) (counted as failures below)",
           f"{'':<18}{'median':>9}{'p95':>9}{'max':>9}{'limit med/p95':>16}"]
    for name, v, lim in (("yaw error (deg)", ye, (0.5, 1.5)), ("origin error (m)", oe, (0.05, 0.10))):
        out.append(f"  {name:<16}{np.median(v):>9.3f}{np.percentile(v, 95):>9.3f}{np.max(v):>9.3f}"
                   f"{lim[0]:>10.2f}/{lim[1]:.2f}")
    ok = np.median(ye) <= 0.5 and np.percentile(ye, 95) <= 1.5 and np.median(oe) <= 0.05 and np.percentile(oe, 95) <= 0.10
    out.append(f"  R1 {'PASS' if ok else 'FAIL'}")
    out.append("")
    out.append(f"  Basin of convergence: success rate (yaw error <= {SUCCESS_YAW_DEG} deg and origin error <= "
               f"{SUCCESS_OFFSET_M} m versus the TRUTH, so a wrong whole cell counts as failure); "
               f"{5 * BASIN_LAYOUTS_PER_SEED} trials per cell")
    out.append(f"  {'initial yaw error':<20}" + "".join(f"{f'off {o:.1f} m':>10}" for o in BASIN_OFFSET_M))
    for yd in BASIN_YAW_DEG:
        out.append(f"  {f'{yd} deg':<20}" + "".join(f"{basin[(yd, o)]:>10.2f}" for o in BASIN_OFFSET_M))
    return out, ok


# ---------------------------------------------------------------------------
# R2
# ---------------------------------------------------------------------------


def r2_runs(seeds: Sequence[int] = SEEDS) -> Dict[Tuple[str, bool], list]:
    scn = all_scenarios(with_registration_error=True)
    out: Dict[Tuple[str, bool], list] = {}
    for name in NEW_IN_43:
        for prop in (False, True):
            ov = ({"grid.origin_sigma_m": REG_ERROR_ORIGIN_SIGMA_M, "grid.yaw_sigma_rad": REG_ERROR_YAW_SIGMA_RAD}
                  if prop else {})
            out[(name, prop)] = [run_scenario(scn[name], layer5_factory(ov)(scn[name]), s).score for s in seeds]
    return out


def format_r2(runs) -> Tuple[List[str], bool]:
    out = ["R2. Registration uncertainty (configured registration off by 3 deg / 0.2 m; sigma 0.141 m per axis, 3 deg when "
           "propagated). ECE over top-cell probabilities p >= 0.5 (calibration.ece, 10 bins), 5 seeds",
           f"  {'scenario':<28}{'prop':>5}{'n tags':>8}{'n(p>=.5)':>9}{'ECE':>7}{'mean p':>8}{'acc':>6}{'correct':>9}{'false CONF':>11}"]
    pooled = {False: ([], [], 0), True: ([], [], 0)}
    ok_conf = True
    for name in NEW_IN_43:
        fc = {}
        for prop in (False, True):
            sc = runs[(name, prop)]
            p = [x for s in sc for x, _ in s.reliability]
            c = [y for s in sc for _, y in s.reliability]
            fc[prop] = sum(s.n_false_confirmed for s in sc)
            pp, cc, ff = pooled[prop]
            pooled[prop] = (pp + p, cc + c, ff + fc[prop])
            sel = [(a, b) for a, b in zip(p, c) if a >= 0.5]
            e = ece(p, c, 10, 0.5) if sel else math.nan
            out.append(f"  {name:<28}{('yes' if prop else 'no'):>5}{len(p):>8}{len(sel):>9}{e:>7.3f}"
                       f"{np.mean(p):>8.3f}{np.mean(c):>6.2f}{sum(s.n_correct for s in sc):>9}{fc[prop]:>11}")
        ok_conf &= fc[True] <= fc[False]
    e_no = ece(pooled[False][0], pooled[False][1], 10, 0.5)
    e_yes = ece(pooled[True][0], pooled[True][1], 10, 0.5)
    for prop in (False, True):
        p, c, f = pooled[prop]
        out.append(f"  {'POOLED':<28}{('yes' if prop else 'no'):>5}{len(p):>8}{sum(1 for x in p if x >= 0.5):>9}"
                   f"{(e_yes if prop else e_no):>7.3f}{np.mean(p):>8.3f}{np.mean(c):>6.2f}{'':>9}{f:>11}")
    ok = e_yes < e_no and pooled[True][2] <= pooled[False][2] and ok_conf
    out.append(f"  R2 {'PASS' if ok else 'FAIL'}: ECE {e_no:.3f} -> {e_yes:.3f} (strictly lower required); false CONFIRMED "
               f"{pooled[False][2]} -> {pooled[True][2]} (per scenario not increased: {ok_conf})")
    return out, ok


# ---------------------------------------------------------------------------
# R3
# ---------------------------------------------------------------------------


def tag_signature(pl) -> list:
    """Everything a tag publishes except ids that depend on the version number: status, cell, position, probabilities."""
    tags = sorted(pl.l4.tags(), key=lambda t: (t.first_seen_ns, float(t.p_map_m[0])))
    return [(t.status, tuple(t.cell) if t.cell is not None else None, t.p_map_m.tolist(), t.cov_map_m2.tolist(),
             [(c, p) for c, p in t.cell_probs], t.observation_count) for t in tags]


def signatures_equal(a: list, b: list, tol: float = 1e-9) -> Tuple[bool, float]:
    if len(a) != len(b):
        return False, math.inf
    worst = 0.0
    for x, y in zip(a, b):
        if x[0] != y[0] or x[1] != y[1] or x[5] != y[5] or [c for c, _ in x[4]] != [c for c, _ in y[4]]:
            return False, math.inf
        worst = max(worst, float(np.max(np.abs(np.array(x[2]) - np.array(y[2])))),
                    max([abs(p - q) for (_, p), (_, q) in zip(x[4], y[4])] + [0.0]))
    return worst <= tol, worst


def r3_run(name: str, seed: int, update_at_fraction: float = 0.5, coverage: float = 0.8):
    """Online: pipeline with the WRONG registration, wall alignment enabled, wall points at the middle of the run.
    Offline: the refined registration (origin, yaw, sigmas, version 1) from the start. Returns (equal, worst, applied)."""
    scn = place_survivors(all_scenarios(with_registration_error=True)[name], seed)
    ev = build_events(scn, seed)
    pts = scenario_wall_points(scn, seed, coverage=coverage, outlier_fraction=0.2)
    ov = {"grid.origin_sigma_m": REG_ERROR_ORIGIN_SIGMA_M, "grid.yaw_sigma_rad": REG_ERROR_YAW_SIGMA_RAD,
          "grid.wall_alignment_enabled": True}
    on = layer5_factory(ov)(scn)
    t_upd = int(ev[-1].arrival_ns * update_at_fraction)
    feed(on, [e for e in ev if e.arrival_ns <= t_upd])
    out = on.process_wall_points(pts, t_upd)
    feed(on, [e for e in ev if e.arrival_ns > t_upd])
    on.finalize()
    if not out.applied:
        return False, math.inf, out
    reg = on.grid
    ov_off = dict(ov, **{"grid.origin_in_map_m": list(reg.origin_in_map_m), "grid.yaw_map_to_grid_rad": reg.yaw_rad,
                         "grid.origin_sigma_m": reg.origin_sigma_m, "grid.yaw_sigma_rad": reg.yaw_sigma_rad,
                         "grid.registration_version": reg.version, "grid.wall_alignment_enabled": False})
    off = layer5_factory(ov_off)(scn)
    feed(off, ev)
    off.finalize()
    eq, worst = signatures_equal(tag_signature(on), tag_signature(off))
    return eq, worst, out


def r3_all(seeds: Sequence[int] = SEEDS):
    return [(n, s) + r3_run(n, s) for n in NEW_IN_43 for s in seeds]


def format_r3(rows) -> Tuple[List[str], bool]:
    n_ok = sum(1 for r in rows if r[2] and r[4].applied)
    worst = max([r[3] for r in rows if math.isfinite(r[3])] + [0.0])
    ok = n_ok == len(rows)
    out = ["R3. Update equivalence: online (wrong registration, wall points at mid-run, Layer 5 recompute) versus offline "
           "(refined registration from the start); compared: status, cell, position, covariance, cell probabilities, "
           "observation count, tolerance 1e-9",
           f"  runs {len(rows)} ({len(NEW_IN_43)} scenarios x 5 seeds); updates applied {sum(r[4].applied for r in rows)}; "
           f"identical {n_ok}; worst difference {worst:.2e}",
           f"  R3 {'PASS' if ok else 'FAIL'}"]
    return out, ok


def main(path: Path = REPORT) -> int:  # pragma: no cover
    t0 = time.perf_counter()
    lines: List[str] = []
    rows = r1_main()
    basin = r1_basin()
    l1, ok1 = format_r1(rows, basin)
    lines += l1 + [""]
    print("\n".join(l1), flush=True)
    l2, ok2 = format_r2(r2_runs())
    lines += l2 + [""]
    print("\n".join(l2), flush=True)
    l3, ok3 = format_r3(r3_all())
    lines += l3
    print("\n".join(l3))
    lines += ["", f"(run time {time.perf_counter() - t0:.0f} s; SIMULATION evidence only)"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if ok1 and ok2 and ok3 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
