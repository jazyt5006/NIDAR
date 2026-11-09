"""Substep 3.1b evaluation: non-confirmation diagnostics, floor-aware Layer 1, footprint refinement.

Run: PYTHONPATH=src python -m survivor_tagging.experiments
All results are SIMULATION tested on synthetic data (box-shaped survivors).

Decision rules (fixed before running; each must PASS at BOTH body offsets 0.0 and 0.2; "does not rise" = no increase):
- floor removal default on iff: tilt_45 acceptance rises; the wrong-cell rate among accepted observations does not
  rise in any scenario; horizontal p90 does not worsen by more than 0.02 m in any scenario (floor vs current).
- footprint default on iff: prone horizontal median < 0.30 m; prone correct CONFIRMED > 0; horizontal p90 does not
  worsen by more than 0.02 m in any non-extended scenario (floor+footprint vs floor).
"""

from __future__ import annotations

import math
import sys
import time
from collections import Counter

import numpy as np

from .diagnostics import LENGTH_ARTIFACTS, DiagSummary, diagnose_run, format_diag
from .footprint import FootprintParams, analyse_footprint
from .geometry import weighted_median
from .harness import _pct, evaluate_many
from .scenarios import all_scenarios
from .sim_pipelines import layer1234_factory

SEEDS = range(5)
OFFSETS = (0.0, 0.2)
VARIANTS = {
    "current": {},
    "floor": {"l1.floor_removal": True},
    "floor+fp": {"l1.floor_removal": True, "l1b.footprint_refinement": True},
    "floor+fp_aw": {"l1.floor_removal": True, "l1b.footprint_refinement": True, "l1b.method": "area_weighted"},
}
# Layer 4 values used in substep 3.1b, pinned so these tables stay reproducible after the 3.1c defaults change.
PRE_31C_LAYER4 = {"l1.body_depth_offset_sigma_m": 0.0, "l4.ray_bias_sigma_m": 0.1, "l4.obs_model_sigma_m": 0.05}
SHOW = ("prone", "tilt_0", "tilt_30", "tilt_45", "standing", "sitting", "low_occluder", "thin_person_in_front_of_wall",
        "occluded")


def _name(variant: str, off: float) -> str:
    return f"{variant} off={off:.1f}"


def extended_object_check() -> str:
    """Generated check without the simulator: a 1.7 x 0.5 m top surface at 0.25 m height, sampled by pinhole pixels
    from a camera 1.5 m up, tilted 20 degrees down, 2 m from the near end (so pixel density falls along the body)."""
    fx = fy = 212.0
    cx, cy = 211.5, 119.5
    tilt = math.radians(20)
    u, v = np.meshgrid(np.arange(424.0), np.arange(240.0))
    d_opt = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1).reshape(-1, 3)
    r_bc = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    c, s = math.cos(tilt), math.sin(tilt)
    r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ r_bc  # camera -> map rotation
    d_map = d_opt @ r.T
    cam = np.array([0.0, 0.0, 1.5])
    t = (0.25 - cam[2]) / d_map[:, 2]
    hit = cam + t[:, None] * d_map
    inside = (t > 0) & (hit[:, 0] >= 2.0) & (hit[:, 0] <= 3.7) & (np.abs(hit[:, 1]) <= 0.25)
    pts_map, depth = hit[inside], t[inside]  # z_cam = t because the optical ray has unit z
    truth = np.array([2.85, 0.0])
    med = np.array([np.median(pts_map[:, 0]), np.median(pts_map[:, 1])])
    aw = np.array([weighted_median(pts_map[:, 0], depth ** 2), weighted_median(pts_map[:, 1], depth ** 2)])
    fp = analyse_footprint(pts_map[:, :2], FootprintParams("midpoint", 5, 95, 0.9, 0.25, 0.0)).centre
    return ("Generated extended-object check (1.7 m body, true centre x = 2.85 m, {} pixels): along-body error "
            "median {:+.3f} m, area-weighted {:+.3f} m, footprint midpoint {:+.3f} m".format(
                len(pts_map), med[0] - truth[0], aw[0] - truth[0], fp[0] - truth[0]))


def main() -> None:  # pragma: no cover
    t0 = time.perf_counter()
    pipelines = {_name(v, off): layer1234_factory({**PRE_31C_LAYER4, **ov, "l1.body_depth_offset_m": off})
                 for v, ov in VARIANTS.items() for off in OFFSETS}
    diags = {off: DiagSummary() for off in OFFSETS}

    def hook(name, placed, frames, pl, res):
        for off in OFFSETS:
            if name == _name("current", off):
                diagnose_run(placed, frames, pl, diags[off])

    ev = evaluate_many(pipelines, all_scenarios().values(), SEEDS, hooks=[hook])
    elapsed = time.perf_counter() - t0
    print(f"SIMULATION tested on synthetic data; random_in_cell placement; 5 seeds; {len(ev.scenarios)} scenarios\n")

    for off in OFFSETS:
        d = diags[off]
        print(f"== A. Why survivors are not CONFIRMED in the true cell (current defaults, offset {off:.1f}, "
              f"confirm_prob 0.9) ==")
        print(format_diag(d, ev.scenarios))
        c = Counter(x.label for x in d.survivors if not x.confirmed_correct)
        n = sum(c.values())
        near = sum(1 for x in d.survivors if x.label == "prob_below" and x.edge_distance_m < 0.15)
        print(f"not confirmed correctly: {n}/{len(d.survivors)}; scenario-length artifacts (min_obs, span): "
              f"{sum(c[k] for k in LENGTH_ARTIFACTS)}; prob_below with fused estimate < 0.15 m from a cell edge: "
              f"{near}/{c['prob_below']}")
        print(f"final track statuses over all runs: {dict(d.statuses)}\n")

    print("== E1. Observation level per variant (accept / correct among accepted / h_med / h_p90) ==")
    for off in OFFSETS:
        print(f"-- offset {off:.1f}")
        print(f"{'scenario':<29}" + "".join(f"{v:>30}" for v in VARIANTS))
        rows = {v: {r.scenario: r for r in ev.obs_rows(_name(v, off), with_total=True)} for v in VARIANTS}
        for s in SHOW + ("ALL",):
            print(f"{s:<29}" + "".join(f"{rows[v][s].accept_rate:>8.2f}{rows[v][s].correct_rate:>7.2f}"
                                       f"{rows[v][s].h_err_median_m:>7.3f}{rows[v][s].h_err_p90_m:>8.3f}"
                                       for v in VARIANTS))

    print("\n== E2. End to end, Layers 1-4 (all scenarios pooled; prone and tilt_45 separately) ==")
    head = (f"{'pipeline':<22}{'corr_any':>9}{'corr_conf':>10}{'missed':>7}{'f_conf':>7}{'f_prov':>7}{'dup':>6}"
            f"{'err_med':>8}{'t_conf':>7}{'prone conf/err':>16}{'tilt45 conf/err':>17}")
    print(head)
    print("-" * len(head))
    for off in OFFSETS:
        for v in VARIANTS:
            rows = {r.scenario: r for r in ev.rows(_name(v, off), with_total=True)}
            a, pr, tl = rows["ALL"], rows["prone"], rows["tilt_45"]
            print(f"{_name(v, off):<22}{a.correct_rate:>9.3f}{a.correct_confirmed_rate:>10.3f}{a.missed_per_run:>7.2f}"
                  f"{a.false_confirmed_per_run:>7.2f}{a.false_provisional_per_run:>7.2f}{a.duplicate_per_run:>6.2f}"
                  f"{a.err_median_m:>8.3f}{a.confirm_from_first_median_s:>7.2f}"
                  f"{pr.correct_confirmed_rate:>10.2f}/{pr.err_median_m:.3f}{tl.correct_confirmed_rate:>11.2f}/"
                  f"{tl.err_median_m:.3f}")

    print("\n== E3. Decision rules (each must pass at offsets 0.0 and 0.2) ==")
    floor_ok, fp_ok = True, True
    for off in OFFSETS:
        cur = {r.scenario: r for r in ev.obs_rows(_name("current", off))}
        flo = {r.scenario: r for r in ev.obs_rows(_name("floor", off))}
        fpo = {r.scenario: r for r in ev.obs_rows(_name("floor+fp", off))}
        e2e = {r.scenario: r for r in ev.rows(_name("floor+fp", off))}
        r1 = flo["tilt_45"].accept_rate > cur["tilt_45"].accept_rate
        wrong_up = {s: round((1 - flo[s].correct_rate) - (1 - cur[s].correct_rate), 3) for s in cur
                    if (1 - flo[s].correct_rate) > (1 - cur[s].correct_rate) + 1e-12}
        p90_worse = {s: round(flo[s].h_err_p90_m - cur[s].h_err_p90_m, 3) for s in cur
                     if flo[s].h_err_p90_m - cur[s].h_err_p90_m > 0.02}
        ok = r1 and not wrong_up and not p90_worse
        floor_ok &= ok
        print(f"[floor, offset {off:.1f}] tilt_45 acceptance {cur['tilt_45'].accept_rate:.2f} -> "
              f"{flo['tilt_45'].accept_rate:.2f} ({'rises' if r1 else 'does not rise'}); wrong-cell rate rose in: "
              f"{wrong_up or 'none'}; p90 worse by > 0.02 m in: {p90_worse or 'none'} -> {'PASS' if ok else 'FAIL'}")
        pr_med = fpo["prone"].h_err_median_m
        pr_conf = e2e["prone"].correct_confirmed_rate
        p90_fp = {s: round(fpo[s].h_err_p90_m - flo[s].h_err_p90_m, 3) for s in flo
                  if s != "prone" and fpo[s].h_err_p90_m - flo[s].h_err_p90_m > 0.02}
        ok2 = pr_med < 0.30 and pr_conf > 0 and not p90_fp
        fp_ok &= ok2
        print(f"[footprint, offset {off:.1f}] prone h_med {flo['prone'].h_err_median_m:.3f} -> {pr_med:.3f} "
              f"(< 0.30: {pr_med < 0.30}); prone correct CONFIRMED {pr_conf:.2f} (> 0: {pr_conf > 0}); p90 worse by "
              f"> 0.02 m in non-extended scenarios: {p90_fp or 'none'} -> {'PASS' if ok2 else 'FAIL'}")
    print(f"DECISION: floor removal default {'ON' if floor_ok else 'OFF'}; footprint refinement default "
          f"{'ON' if fp_ok else 'OFF'}")

    print("\n== C5. Prone variants (observation level, floor removal on) ==")
    for off in OFFSETS:
        line = f"offset {off:.1f}: "
        for v, label in (("floor", "Layer 1 median pixel"), ("floor+fp_aw", "map-frame area-weighted"),
                         ("floor+fp", "footprint midpoint")):
            r = {x.scenario: x for x in ev.obs_rows(_name(v, off))}["prone"]
            line += f"{label} h_med {r.h_err_median_m:.3f} / p90 {r.h_err_p90_m:.3f} / correct {r.correct_rate:.2f};  "
        print(line)
    print(extended_object_check())

    print(f"\n== E4. Timing (this laptop, Python {sys.version.split()[0]}) ==")
    tm = {}
    for v in ("current", "floor", "floor+fp"):
        tm[v] = [x for s in ev.scenarios for r in ev.results[_name(v, 0.0)][s] for x in r.pipeline.l1_ms]
    fp_ms = [x for s in ev.scenarios for r in ev.results[_name("floor+fp", 0.0)][s] for x in r.pipeline.footprint_ms]
    print(f"Layer 1 per observation: current median {_pct(tm['current'], 50):.3f} / p95 {_pct(tm['current'], 95):.3f}"
          f" ms; with floor removal {_pct(tm['floor'], 50):.3f} / {_pct(tm['floor'], 95):.3f} ms; with floor removal "
          f"+ kept points {_pct(tm['floor+fp'], 50):.3f} / {_pct(tm['floor+fp'], 95):.3f} ms")
    print(f"Footprint refinement per observation: median {_pct(fp_ms, 50):.3f} ms, p95 {_pct(fp_ms, 95):.3f} ms "
          f"(n={len(fp_ms)})")
    print(f"\nevaluation {elapsed:.1f} s, total {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":  # pragma: no cover
    main()
