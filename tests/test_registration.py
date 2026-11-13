"""Substep 4.3: registration uncertainty, wall-alignment estimator, hysteresis, update equivalence. Unit + SIMULATION tests."""

import math
from dataclasses import replace

import numpy as np
import pytest

from survivor_tagging.config import default_config
from survivor_tagging.experiments_43 import make_trial, r3_run
from survivor_tagging.grid import GridRegistration, cell_probabilities
from survivor_tagging.registration import (WallAlignParams, estimate_wall_alignment, registration_change,
                                           update_needed)
from survivor_tagging.scenarios import (NEW_IN_43, REG_ERROR_ORIGIN_SIGMA_M, REG_ERROR_YAW_SIGMA_RAD,
                                        SIM_REGISTRATION, all_scenarios)
from survivor_tagging.sim import Box, Registration, build_events, place_survivors, wall
from survivor_tagging.sim_pipelines import layer5_factory, library_registration
from survivor_tagging.sim_walls import (grid_to_map_points, maze_arena, random_maze_walls, sample_wall_points_grid,
                                        scenario_wall_points)
from survivor_tagging.types import FAILURE_SEVERITY, FailureReason, Severity, TagStatus

F = FailureReason


def reg(**kw):
    base = dict(origin_in_map_m=(1.3, -0.7), yaw_rad=0.6, flip_x=False, flip_y=False, cell_size_m=1.0, n_cells_x=10,
                n_cells_y=10, valid=True)
    base.update(kw)
    return GridRegistration(**base)


# ------------------------------------------------------------------ registration uncertainty
@pytest.mark.parametrize("flips", [(False, False), (True, False), (False, True), (True, True)])
def test_jacobians_match_finite_differences(flips):
    r = reg(flip_x=flips[0], flip_y=flips[1])
    p = np.array([4.2, 3.1])
    h = 1e-6
    d_yaw = (replace(r, yaw_rad=r.yaw_rad + h).map_to_grid(p) - replace(r, yaw_rad=r.yaw_rad - h).map_to_grid(p)) / (2 * h)
    assert np.allclose(r.jacobian_yaw(p), d_yaw, atol=1e-7)
    jo = r.jacobian_origin()
    for a in range(2):
        e = np.zeros(2)
        e[a] = h
        o = np.array(r.origin_in_map_m)
        d_o = (replace(r, origin_in_map_m=tuple(o + e)).map_to_grid(p)
               - replace(r, origin_in_map_m=tuple(o - e)).map_to_grid(p)) / (2 * h)
        assert np.allclose(jo[:, a], d_o, atol=1e-7)


def test_position_covariance_analytic_and_monte_carlo():
    r = reg(origin_sigma_m=0.2, yaw_sigma_rad=0.05)
    p = np.array([6.0, 5.0])
    cov = r.position_covariance(p)
    jo, gy = r.jacobian(), r.jacobian_yaw(p)
    assert np.allclose(cov, 0.2 ** 2 * jo @ jo.T + 0.05 ** 2 * np.outer(gy, gy))
    rng = np.random.default_rng(0)
    n = 200000
    dyaw, do = rng.normal(0, 0.05, n), rng.normal(0, 0.2, (n, 2))
    pts = []
    c, s = np.cos(r.yaw_rad + dyaw), np.sin(r.yaw_rad + dyaw)
    d = p[None, :] - (np.array(r.origin_in_map_m)[None, :] + do)
    g = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)
    mc = np.cov(g.T)
    # second-order terms of the exact rotation make the MC covariance differ slightly from the linearisation
    assert np.allclose(mc, cov, atol=0.01 * np.max(np.abs(cov)) + 2e-3)


def test_origin_covariance_overrides_sigma_and_unknown_is_zero():
    r = reg(origin_cov_m2=((0.04, 0.0), (0.0, 0.01)), yaw_sigma_rad=0.0)
    assert np.allclose(r.origin_covariance(), [[0.04, 0], [0, 0.01]])
    assert np.allclose(r.position_covariance((3, 3)), r.jacobian() @ r.origin_covariance() @ r.jacobian().T)
    assert not reg().uncertainty_known and np.allclose(reg().position_covariance((3, 3)), 0)
    assert reg(origin_sigma_m=0.1, yaw_sigma_rad=0.0).uncertainty_known  # 0.0 is a known value


def test_registration_error_is_added_at_the_cell_probability_step_only():
    cov = np.eye(2) * 0.01
    p = (6.2, 5.1)
    base = cell_probabilities(reg(), p, cov)
    same = cell_probabilities(reg(origin_sigma_m=0.0, yaw_sigma_rad=0.0), p, cov)
    assert base == same
    wide = cell_probabilities(reg(origin_sigma_m=0.3, yaw_sigma_rad=0.09), p, cov)
    assert wide.cells[0][1] < base.cells[0][1]
    skipped = cell_probabilities(reg(origin_sigma_m=0.3, yaw_sigma_rad=0.09), p, cov, include_registration=False)
    assert skipped == base
    # equals a plain call with the covariance added by hand (map-frame equivalent)
    r = reg(origin_sigma_m=0.3, yaw_sigma_rad=0.09)
    j = r.jacobian()
    by_hand = cell_probabilities(reg(), p, cov + j.T @ r.position_covariance(p) @ j)
    for (c1, p1), (c2, p2) in zip(wide.cells, by_hand.cells):
        assert c1 == c2 and abs(p1 - p2) < 1e-9


def test_cell_probabilities_with_registration_match_monte_carlo():
    r = reg(origin_sigma_m=0.25, yaw_sigma_rad=0.08)
    p = np.array([6.3, 4.4])
    cov = np.diag([0.02, 0.03])
    cp = cell_probabilities(r, p, cov).as_dict()
    rng = np.random.default_rng(1)
    n = 200000
    pm = rng.multivariate_normal(p, cov, n)
    dyaw, do = rng.normal(0, 0.08, n), rng.normal(0, 0.25, (n, 2))
    c, s = np.cos(r.yaw_rad + dyaw), np.sin(r.yaw_rad + dyaw)
    d = pm - (np.array(r.origin_in_map_m) + do)
    g = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)
    cells = np.floor(g).astype(int)
    for cell, prob in list(cp.items())[:3]:
        emp = np.mean((cells[:, 0] == cell[0]) & (cells[:, 1] == cell[1]))
        assert abs(emp - prob) < 0.01, (cell, emp, prob)


def test_tags_warn_when_uncertainty_unset_and_not_blocking():
    assert FAILURE_SEVERITY[F.REG_UNCERTAINTY_UNKNOWN] is Severity.DEGRADE
    scn = place_survivors(all_scenarios(with_registration_error=True)["registration_error"], 1)
    ev = build_events(scn, 1)
    seen = {}
    for label, ov in (("unset", {"grid.origin_sigma_m": None, "grid.yaw_sigma_rad": None}),
                      ("set", {"grid.origin_sigma_m": 0.3, "grid.yaw_sigma_rad": 0.09})):
        pl = layer5_factory(ov)(scn)
        from survivor_tagging.experiments_layer5 import feed
        feed(pl, ev)
        pl.finalize()
        seen[label] = pl.l4.tags()
    assert seen["unset"] and all(F.REG_UNCERTAINTY_UNKNOWN in t.reasons for t in seen["unset"])
    assert all(F.REG_UNCERTAINTY_UNKNOWN not in t.reasons for t in seen["set"])
    assert any(t.status in (TagStatus.CONFIRMED, TagStatus.UPDATED) for t in seen["unset"])  # not blocking


def test_per_observation_probabilities_do_not_carry_registration_error():
    scn = place_survivors(all_scenarios(with_registration_error=True)["registration_error_far"], 0)
    ev = build_events(scn, 0)
    from survivor_tagging.experiments_layer5 import feed
    probs = []
    for ov in ({"grid.origin_sigma_m": 0.0, "grid.yaw_sigma_rad": 0.0}, {"grid.origin_sigma_m": 0.3, "grid.yaw_sigma_rad": 0.09}):
        pl = layer5_factory(ov)(scn)
        feed(pl, ev)
        probs.append([dict(o.cell_probs) for o in pl.observations() if o.p_map_m is not None])
    assert probs[0] == probs[1]  # per-observation covariances unchanged: a common-mode error is not averaged away


def test_all_tags_share_the_same_registration_error_covariance_structure():
    """Two points 0.5 m apart get nearly identical registration covariance (common mode), unlike independent noise."""
    r = reg(origin_sigma_m=0.2, yaw_sigma_rad=0.05)
    a, b = r.position_covariance((6.0, 5.0)), r.position_covariance((6.5, 5.0))
    assert np.allclose(a, b, atol=0.15 * np.max(a))


# ------------------------------------------------------------------ estimator
def test_estimator_accuracy_on_synthetic_mazes():
    for seed, k in ((0, 0), (1, 3), (2, 5), (3, 7)):
        ye, oe, res = make_trial(seed, k, 8.0, 0.35)
        assert res.registration is not None and ye < 0.5 and oe < 0.05, (seed, k, ye, oe, res.detail)
        assert res.quality > 0.8 and res.yaw_sigma_rad > 0 and res.origin_sigma_m > 0
        assert res.registration.version == 1 and res.registration.uncertainty_known


def test_estimator_flip_and_cell_size_independent_of_axes():
    n = 6
    arena = maze_arena(n, 3)
    sim = Registration((0.4, -1.1), 2.2)
    pts = grid_to_map_points(sim, sample_wall_points_grid(arena.all_walls(1.0), 3, extent_m=(-.5, n + .5, -.5, n + .5)))
    true = library_registration(sim, n)
    for fx, fy in ((True, False), (False, True)):
        # the same physical lines: flips do not move grid lines, so a flipped coarse registration refines the same way
        coarse = replace(true, flip_x=fx, flip_y=fy, yaw_rad=true.yaw_rad + 0.1,
                         origin_in_map_m=(true.origin_in_map_m[0] + 0.2, true.origin_in_map_m[1] - 0.2))
        res = estimate_wall_alignment(pts, coarse)
        assert res.registration is not None
        assert abs(res.registration.yaw_rad - true.yaw_rad) < math.radians(0.5)
        assert (res.registration.flip_x, res.registration.flip_y) == (fx, fy)


def _walls_pts(boxes, seed=0, n=6, sim=Registration((0.3, 0.2), 0.7), **kw):
    kw.setdefault("extent_m", (-.5, n + .5, -.5, n + .5))
    return grid_to_map_points(sim, sample_wall_points_grid(boxes, seed, **kw)), library_registration(sim, n)


def test_too_few_wall_points():
    pts, true = _walls_pts([wall("x", 2.0, 0.0, 1.0)], outlier_fraction=0.0)
    res = estimate_wall_alignment(pts, true)
    assert res.registration is None and res.reason in (F.WALL_POINTS_INSUFFICIENT, F.WALL_DIRECTION_AMBIGUOUS)
    res = estimate_wall_alignment(np.zeros((5, 2)), true)
    assert res.registration is None and res.reason is F.WALL_POINTS_INSUFFICIENT
    assert estimate_wall_alignment(np.zeros((5, 3, 1)).reshape(5, 3), true).registration is None


def test_one_wall_direction_gives_no_update_and_names_it():
    boxes = [wall("x", float(i), 0.0, 6.0) for i in range(1, 6)]  # five parallel walls only
    pts, true = _walls_pts(boxes, outlier_fraction=0.1)
    res = estimate_wall_alignment(pts, true)
    assert res.registration is None and res.reason is F.WALL_DIRECTION_AMBIGUOUS
    assert res.n_axis_xy[0] > 100 and res.n_axis_xy[1] < 40


def test_heavy_outliers_degrade_gracefully_or_refuse():
    arena = maze_arena(7, 2)
    sim = Registration((0.3, 0.2), 0.7)
    true = library_registration(sim, 7)
    coarse = replace(true, yaw_rad=true.yaw_rad + 0.1, origin_in_map_m=(true.origin_in_map_m[0] + 0.2, true.origin_in_map_m[1]))
    for frac, expect_ok in ((0.3, True), (0.95, False)):
        pts = grid_to_map_points(sim, sample_wall_points_grid(arena.all_walls(1.0), 2, outlier_fraction=frac,
                                                              extent_m=(-.5, 7.5, -.5, 7.5)))
        res = estimate_wall_alignment(pts, coarse)
        if expect_ok:
            assert res.registration is not None and abs(res.registration.yaw_rad - true.yaw_rad) < math.radians(1.0)
        else:  # never a silent wrong answer: either refused with a reason, or still accurate
            assert res.registration is None and res.reason is not None or \
                abs(res.registration.yaw_rad - true.yaw_rad) < math.radians(1.5)


def test_walls_not_on_grid_lines_give_low_quality_no_update():
    boxes = [wall("x", 0.5 + 0.37 * i, 0.0, 6.0) for i in range(1, 12)] + [wall("y", 0.5 + 0.41 * i, 0.0, 6.0) for i in range(1, 12)]
    pts, true = _walls_pts(boxes, outlier_fraction=0.0, noise_sigma_m=0.02)
    res = estimate_wall_alignment(pts, true)
    assert res.registration is None and res.reason is F.WALL_ALIGNMENT_LOW_QUALITY and res.quality < 0.5


def test_coarse_uncertainty_too_large_gives_no_update():
    pts, true = _walls_pts(list(maze_arena(6, 1).all_walls(1.0)))
    for bad in (replace(true, origin_sigma_m=0.3, yaw_sigma_rad=0.05), replace(true, origin_sigma_m=0.05, yaw_sigma_rad=0.3)):
        res = estimate_wall_alignment(pts, bad)
        assert res.registration is None and res.reason is F.REG_COARSE_AMBIGUOUS
    ok = estimate_wall_alignment(pts, replace(true, origin_sigma_m=0.1, yaw_sigma_rad=0.05))
    assert ok.registration is not None


def test_estimator_is_deterministic():
    a = make_trial(2, 4, 6.0, 0.3)
    b = make_trial(2, 4, 6.0, 0.3)
    assert a[:2] == b[:2] and a[2].registration == b[2].registration and a[2].quality == b[2].quality


def test_maze_walls_lie_on_grid_lines_with_one_metre_corridors():
    for w in random_maze_walls(8, 5):
        lo, hi = w.lo, w.hi
        thin = 0 if hi[0] - lo[0] < hi[1] - lo[1] else 1
        assert abs(0.5 * (lo[thin] + hi[thin]) - round(0.5 * (lo[thin] + hi[thin]))) < 1e-9
    assert random_maze_walls(8, 5) == random_maze_walls(8, 5)


# ------------------------------------------------------------------ hysteresis and update flow
def test_hysteresis_thresholds():
    a = reg()
    assert not update_needed(a, replace(a, yaw_rad=a.yaw_rad + 0.001, origin_in_map_m=(a.origin_in_map_m[0] + 0.01, a.origin_in_map_m[1])), 0.002, 0.02)
    assert update_needed(a, replace(a, yaw_rad=a.yaw_rad + 0.003), 0.002, 0.02)
    assert update_needed(a, replace(a, origin_in_map_m=(a.origin_in_map_m[0], a.origin_in_map_m[1] + 0.03)), 0.002, 0.02)
    dyaw, doff = registration_change(replace(a, yaw_rad=3.1), replace(a, yaw_rad=-3.1))
    assert abs(dyaw - (2 * math.pi - 6.2)) < 1e-9 and doff == 0.0  # wrapped


def _online(name="registration_error", seed=0, **ov):
    scn = place_survivors(all_scenarios(with_registration_error=True)[name], seed)
    ev = build_events(scn, seed)
    ovr = {"grid.origin_sigma_m": REG_ERROR_ORIGIN_SIGMA_M, "grid.yaw_sigma_rad": REG_ERROR_YAW_SIGMA_RAD}
    ovr.update(ov)
    return scn, ev, layer5_factory(ovr)(scn)


def test_wall_alignment_is_off_by_default_and_reports_why():
    scn, ev, pl = _online()
    out = pl.process_wall_points(scenario_wall_points(scn, 0), 5)
    assert not out.applied and out.reason is F.WALL_ALIGNMENT_DISABLED and pl.grid.version == 0 and not pl.recomputes


def test_small_estimated_change_triggers_no_recompute():
    scn, ev, pl = _online(**{"grid.wall_alignment_enabled": True})
    true = library_registration(scn.registration, scn.arena.n_cells, origin_sigma_m=0.14, yaw_sigma_rad=0.05)
    pl.grid = true  # the live registration is already right: the estimate differs by less than the thresholds
    pl.l4.reg = true
    n0 = len(pl.recomputes)
    out = pl.process_wall_points(scenario_wall_points(scn, 0, coverage=0.9), 5)
    assert not out.applied and out.reason is F.REG_UPDATE_BELOW_THRESHOLD
    assert pl.grid.version == 0 and len(pl.recomputes) == n0


def test_failed_estimate_is_reported_not_hidden():
    scn, ev, pl = _online(**{"grid.wall_alignment_enabled": True})
    out = pl.process_wall_points(np.zeros((10, 2)), 5)
    assert not out.applied and out.reason is F.WALL_POINTS_INSUFFICIENT
    assert pl.wall_alignments[-1].reason is F.WALL_POINTS_INSUFFICIENT and pl.grid.version == 0


@pytest.mark.parametrize("name,seed", [(n, s) for n in NEW_IN_43 for s in (0, 3)])
def test_r3_update_equals_offline_run_with_refined_registration(name, seed):
    eq, worst, out = r3_run(name, seed)
    assert out.applied and out.revision == 1
    assert eq and worst <= 1e-9


def test_estimator_update_improves_the_registration_error():
    scn, ev, pl = _online(**{"grid.wall_alignment_enabled": True})
    true = library_registration(scn.registration, scn.arena.n_cells)
    before = abs(pl.grid.yaw_rad - true.yaw_rad)
    out = pl.process_wall_points(scenario_wall_points(scn, 0, coverage=0.9), 5)
    assert out.applied and abs(pl.grid.yaw_rad - true.yaw_rad) < 0.1 * before
    assert pl.grid.version == 1 and pl.grid.uncertainty_known


def test_default_config_has_the_new_parameters_unset_or_off():
    c = default_config()
    assert c.value("grid.origin_sigma_m") is None and c.value("grid.yaw_sigma_rad") is None
    assert c.value("grid.wall_alignment_enabled") is False
    assert WallAlignParams.from_config(c).min_quality == 0.5
    assert GridRegistration.from_config(c).uncertainty_known is False
