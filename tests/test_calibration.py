"""Substep 3.1c: calibration metrics, derived ray-bias sigma, expected-value policy, simulator body depth."""

import itertools
import math

import numpy as np
import pytest

from survivor_tagging.calibration import ece, over_confidence_flags, reliability_table, worst_over_confidence
from survivor_tagging.config import default_config
from survivor_tagging.grid import GridRegistration
from survivor_tagging.tracking import (PENALTY_UNKNOWN_NOTE, Layer4, Layer4Params, TrackObs, expected_value_threshold,
                                       fuse)
from survivor_tagging.types import SurvivorObservation, TagStatus

S = 1_000_000_000
REG = GridRegistration((0.0, 0.0), 0.0, n_cells_x=6, n_cells_y=6, valid=True)
_ids = itertools.count()


def params(**kw):
    return Layer4Params.from_config(default_config(), **kw)


def obs(t_s, xy, cam=None, sd=0.02):
    x, y = xy
    cam = cam if cam is not None else (x - 2.5, y)
    return SurvivorObservation(next(_ids), 0, 0.9, int(round(t_s * S)), None, p_map_m=np.array([x, y, 0.85]),
                               cov_map_m2=np.eye(3) * sd * sd, camera_map_m=np.array([cam[0], cam[1], 1.5]),
                               pose_revision=0, extrinsic_version=1)


# --- calibration.py ------------------------------------------------------------------------------------------------

def test_reliability_table_and_ece_hand_built():
    pred = [0.15] * 4 + [0.75] * 6
    correct = [False] * 4 + [True, True, True, False, False, False]
    rows = reliability_table(pred, correct)
    assert len(rows) == 10 and [r.count for r in rows] == [0, 4, 0, 0, 0, 0, 0, 6, 0, 0]
    assert rows[1].mean_pred == pytest.approx(0.15) and rows[1].frac_correct == 0.0
    assert rows[7].frac_correct == pytest.approx(0.5) and rows[7].over_confidence == pytest.approx(0.25)
    assert math.isnan(rows[0].mean_pred) and math.isnan(rows[0].over_confidence)  # empty bin
    assert ece(pred, correct) == pytest.approx(0.4 * 0.15 + 0.6 * 0.25)
    assert ece(pred, correct, p_min=0.5) == pytest.approx(0.25)  # only the 0.75 samples count


def test_ece_edge_cases():
    assert math.isnan(ece([], []))
    assert ece([1.0] * 5, [True] * 5) == 0.0  # p = 1 falls in the last bin
    assert reliability_table([1.0], [True])[9].count == 1
    assert ece([0.0] * 5, [False] * 5) == 0.0
    assert ece([1.0] * 4, [False] * 4) == pytest.approx(1.0)
    assert math.isnan(ece([0.2, 0.3], [True, False], p_min=0.5))
    with pytest.raises(ValueError):
        ece([1.2], [True])
    with pytest.raises(ValueError):
        ece([0.5, 0.5], [True])
    with pytest.raises(ValueError):
        reliability_table([0.5], [True], bins=0)


def test_over_confidence_flags_respect_min_count():
    pred, correct = [0.95] * 25, [True] * 20 + [False] * 5  # observed 0.8 -> over 0.15
    flags = over_confidence_flags(pred, correct)
    assert len(flags) == 1 and flags[0].lo == pytest.approx(0.9)
    assert over_confidence_flags(pred[:19], correct[:19]) == []  # fewer than 20 samples: not judged
    assert worst_over_confidence(pred, correct) == pytest.approx(0.15)
    assert math.isnan(worst_over_confidence(pred[:19], correct[:19]))
    assert over_confidence_flags([0.55] * 30, [True] * 30) == []  # under-confidence is never flagged


# --- derived ray-bias sigma ------------------------------------------------------------------------------------------

def test_ray_sigma_formula_and_defaults():
    p = params()
    assert p.offset_sigma_m == 0.10 and p.ray_bias_sigma_m == 0.1  # ASSUMPTION defaults
    assert p.ray_sigma_m == pytest.approx(math.sqrt(0.1 ** 2 + 0.1 ** 2))
    assert params(ray_bias_sigma_m=0.05, offset_sigma_m=0.12).ray_sigma_m == pytest.approx(0.13)
    assert params(offset_sigma_m=0.0).ray_sigma_m == 0.1  # exactly the pre-3.1c value


def _items(n=12):
    out = []
    for k in range(n):
        cam = (1.0, 3.5) if k < n // 2 else (3.5, 1.0)
        o = obs(0.2 * k, (3.5, 3.5), cam=cam)
        xy = np.array([3.5, 3.5])
        ray = (xy - np.array(cam)) / np.linalg.norm(xy - np.array(cam))
        bearing = math.degrees(math.atan2(cam[1] - 3.5, cam[0] - 3.5)) % 360
        out.append(TrackObs(o, xy, np.eye(2) * 0.003, ray, bearing, o.capture_stamp_ns))
    return out


def test_zero_offset_sigma_reproduces_earlier_fusion_exactly_and_terms_combine():
    items = _items()
    x0, p0, _ = fuse(items, params(offset_sigma_m=0.0))
    xs, ps, _ = fuse(items, params(ray_bias_sigma_m=0.1, offset_sigma_m=0.0))
    assert np.array_equal(p0, ps) and np.array_equal(x0, xs)
    # floor 0.06 + offset 0.08 is the same model as floor 0.1 + offset 0 (each counted once)
    _, pa, _ = fuse(items, params(ray_bias_sigma_m=0.06, offset_sigma_m=0.08))
    assert np.allclose(pa, p0, rtol=1e-12, atol=1e-15)
    _, pb, _ = fuse(items, params(offset_sigma_m=0.2))
    assert np.all(np.linalg.eigvalsh(pb - p0) > 0)  # more offset uncertainty -> strictly larger covariance


# --- expected-value policy -------------------------------------------------------------------------------------------

def test_expected_value_threshold_formula():
    assert expected_value_threshold(40.0, 40.0) == (0.5, "")
    assert expected_value_threshold(20.0, 40.0)[0] == pytest.approx(1 / 3)
    assert expected_value_threshold(80.0, 40.0)[0] == pytest.approx(2 / 3)
    assert expected_value_threshold(0.0, 40.0)[0] == 0.0
    th, note = expected_value_threshold(None, 40.0)
    assert th == 0.5 and note == PENALTY_UNKNOWN_NOTE
    with pytest.raises(ValueError):
        expected_value_threshold(-1.0, 40.0)


def _policy_layer4(**kw):
    l4 = Layer4(params(**kw), REG)
    for k in range(6):  # A: centre of (3,3), confirms; B: on the x = 2 edge, stays near 0.5
        l4.update([obs(0.2 * k, (3.5, 3.5)), obs(0.2 * k, (2.0, 1.5), cam=(-0.5, 1.5))])
    l4.update([obs(1.2, (5.5, 5.5)), obs(1.2, (0.5, 4.5))])  # C, D: one observation each
    l4.update([obs(1.4, (0.5, 4.5))])  # D: two observations
    return l4


def test_expected_value_policy_threshold_min_obs_and_unreliable():
    l4 = _policy_layer4()
    by_x = {round(float(t.x[0]), 1): t for t in l4.tracks.values()}
    a, b, d = by_x[3.5], by_x[2.0], by_x[0.5]
    assert a.status is TagStatus.CONFIRMED and 0.3 < b.top()[1] < 0.7 and d.n_frames == 2
    res = l4.final_tag_set("expected_value")  # penalty unset -> 0.5 and a note
    assert res.threshold == 0.5 and res.notes == (PENALTY_UNKNOWN_NOTE,)
    committed = {t.survivor_id for t in res.tags}
    assert a.track_id in committed and d.track_id not in committed  # D: below commit_min_obs (3)
    assert (b.track_id in committed) == (b.top()[1] >= 0.5)
    every = {t.survivor_id for t in l4.final_tags("expected_value", 0.0)}
    assert every == {a.track_id, b.track_id}  # min obs still applies at threshold 0
    tag_b = next(t for t in l4.final_tags("expected_value", 0.0) if t.survivor_id == b.track_id)
    assert tag_b.cell == b.top()[0] and tag_b.localisation_quality == pytest.approx(b.top()[1])
    b.status = TagStatus.UNRELIABLE
    assert b.track_id not in {t.survivor_id for t in l4.final_tags("expected_value", 0.0)}
    l4b = _policy_layer4(commit_min_obs=2)
    assert len(l4b.final_tags("expected_value", 0.0)) == 3  # D now qualifies


def test_expected_value_policy_uses_configured_penalty():
    l4 = _policy_layer4(wrong_penalty=80.0)
    res = l4.final_tag_set("expected_value")
    assert res.threshold == pytest.approx(2 / 3) and res.notes == ()
    assert all(t.localisation_quality >= 2 / 3 for t in res.tags)


# --- simulator body depth, determinism, references -------------------------------------------------------------------

def test_body_depth_option_default_unchanged_and_seeded():
    from survivor_tagging.scenarios import all_scenarios
    from survivor_tagging.sim import POSTURE_DIMS_M, place_survivors

    fixed = all_scenarios()["two_adjacent"]
    uni = all_scenarios(body_depth="uniform_0.2_0.5")["two_adjacent"]
    pf, pu = place_survivors(fixed, 3), place_survivors(uni, 3)
    assert all(s.depth_m is None for s in pf.survivors)
    assert [s.grid_xy for s in pf.survivors] == [s.grid_xy for s in pu.survivors]  # placement stream unchanged
    depths = [s.depth_m for s in pu.survivors]
    assert all(0.2 <= d <= 0.5 for d in depths) and depths == [s.depth_m for s in place_survivors(uni, 3).survivors]
    assert place_survivors(pu, 3).survivors == pu.survivors  # idempotent
    box = pu.survivors[0].box()
    assert box.hi[0] - box.lo[0] == pytest.approx(depths[0])
    assert box.hi[1] - box.lo[1] == pytest.approx(POSTURE_DIMS_M["standing"][1])
    prone = place_survivors(all_scenarios(body_depth="uniform_0.2_0.5")["prone"], 0).survivors[0].box()
    assert prone.hi[0] - prone.lo[0] == pytest.approx(POSTURE_DIMS_M["prone"][0])  # prone keeps its box


def test_layer4_replay_matches_pipeline_deterministic_and_finite():
    from survivor_tagging.experiments_calib import OUTPUT_STATUSES, cache_run, replay
    from survivor_tagging.harness import run_scenario
    from survivor_tagging.scenarios import all_scenarios
    from survivor_tagging.sim import place_survivors
    from survivor_tagging.sim_pipelines import layer1234_factory, scenario_config

    scn = all_scenarios(body_depth="uniform_0.2_0.5")["id_merge"]
    placed = place_survivors(scn, 1)
    direct = run_scenario(scn, layer1234_factory({"l1.body_depth_offset_m": 0.2})(placed), 1)
    run = cache_run(scn, 1, 0.2, "uniform_0.2_0.5")
    p = Layer4Params.from_config(scenario_config(placed))
    s1, s2 = [], []
    l4 = replay(run, p, s1)
    replay(run, p, s2)
    assert [(t.cell, t.status) for t in direct.tags] == \
           [(t.cell, t.status) for t in l4.tags() if t.status in OUTPUT_STATUSES]
    assert s1 == s2 and s1 and all(0.0 <= x <= 1.0 and not math.isnan(x) for x, _ in s1)


def test_naive_reference_unchanged():
    """Naive baseline over the original 15 scenarios x 5 seeds: 0.612 correct, 0.52 false and 0.40 duplicates per
    run, 0.245 m median error (the reference since substep 2.3)."""
    from survivor_tagging.baseline import naive_factory
    from survivor_tagging.harness import evaluate_many, summarise
    from survivor_tagging.scenarios import all_scenarios

    ev = evaluate_many({"n": naive_factory}, list(all_scenarios().values())[:15], range(5))
    r = summarise("ALL", [x for s in ev.scenarios for x in ev.results["n"][s]])
    assert (round(r.correct_rate, 3), round(r.false_per_run, 2), round(r.duplicate_per_run, 2),
            round(r.err_median_m, 3)) == (0.612, 0.52, 0.40, 0.245)
