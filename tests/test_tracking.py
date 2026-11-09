import itertools
import math

import numpy as np
import pytest

from survivor_tagging.config import default_config
from survivor_tagging.grid import GridRegistration
from survivor_tagging.tracking import (
    LEGAL_TRANSITIONS,
    IllegalTransition,
    Layer4,
    Layer4Params,
    Track,
    TrackObs,
    fuse,
    hungarian,
)
from survivor_tagging.types import FailureReason, SurvivorObservation, TagStatus

S = 1_000_000_000
REG = GridRegistration((0.0, 0.0), 0.0, n_cells_x=6, n_cells_y=6, valid=True)
_ids = itertools.count()


def params(**kw):
    return Layer4Params.from_config(default_config(), **kw)


def layer4(reg=REG, **kw):
    return Layer4(params(**kw), reg)


def obs(t_s, xy, cam=None, sd=0.02, tid=None, conf=0.9):
    x, y = xy
    cam = cam if cam is not None else (x - 2.5, y)
    return SurvivorObservation(next(_ids), 0, conf, int(round(t_s * S)), tid,
                               p_map_m=np.array([x, y, 0.85]), cov_map_m2=np.eye(3) * sd * sd,
                               camera_map_m=np.array([cam[0], cam[1], 1.5]), pose_revision=0, extrinsic_version=1)


def feed(l4, frames):
    for f in frames:
        l4.update(f)


def only(l4):
    assert len(l4.tracks) == 1, [(t.track_id, t.x) for t in l4.tracks.values()]
    return next(iter(l4.tracks.values()))


# --- spec 12.4 cases -------------------------------------------------------------------------------------------

def test_124_repeated_detections_of_one_person():
    l4 = layer4()
    rng = np.random.default_rng(0)
    feed(l4, [[obs(0.2 * k, (3.5, 3.5) + rng.normal(0, 0.02, 2), tid=7)] for k in range(10)])
    t = only(l4)
    assert t.status is TagStatus.CONFIRMED and t.confirmed_cell == (3, 3) and t.n_frames == 10


def test_124_different_tracking_ids_for_the_same_person():
    l4 = layer4()
    feed(l4, [[obs(0.2 * k, (3.5, 3.5), tid=100 + k)] for k in range(8)])
    t = only(l4)
    assert t.status is TagStatus.CONFIRMED
    assert sum(1 for h in t.history if h[1] == "detector_id_changed") == 7


def test_124_noisy_depth_measurements():
    rng = np.random.default_rng(1)
    l4 = layer4()
    truth = np.array([3.4, 2.6])
    frames = []
    for k in range(20):  # depth noise 0.1 m along the ray, from two viewpoints
        cam = (truth[0] - 2.5, truth[1]) if k % 2 else (truth[0], truth[1] - 2.5)
        ray = (truth - cam) / np.linalg.norm(truth - cam)
        frames.append([obs(0.2 * k, truth + ray * rng.normal(0, 0.1), cam=cam, sd=0.1)])
    feed(l4, frames)
    t = only(l4)
    err = t.x - truth
    assert float(err @ np.linalg.solve(t.P, err)) < 13.8  # within the 99.9 % ellipse of the stated covariance
    assert t.n_bins == 2


def test_124_inconsistent_observations_become_unreliable():
    l4 = layer4(gate_chi2=50.0)  # a loose gate lets contradicting observations associate instead of splitting
    feed(l4, [[obs(0.2 * k, (3.5, 3.5))] for k in range(6)])
    assert only(l4).status is TagStatus.CONFIRMED
    feed(l4, [[obs(1.2 + 0.2 * k, (3.5, 3.5 + (0.17 if k % 2 else -0.17)))] for k in range(6)])
    t = only(l4)
    assert t.status is TagStatus.UNRELIABLE and FailureReason.TRACK_INCONSISTENT in t.warnings


def test_124_independent_viewpoints_reduce_uncertainty():
    one, two = layer4(), layer4()
    feed(one, [[obs(0.2 * k, (3.5, 3.5), cam=(1.0, 3.5))] for k in range(10)])
    feed(two, [[obs(0.2 * k, (3.5, 3.5), cam=(1.0, 3.5) if k < 5 else (3.5, 1.0))] for k in range(10)])
    t1, t2 = only(one), only(two)
    assert (t1.n_bins, t2.n_bins) == (1, 2)
    assert np.linalg.det(t2.P) < 0.5 * np.linalg.det(t1.P)
    assert t2.P[0, 0] < 0.5 * t1.P[0, 0]  # the first view's ray (map x) is constrained by the second view


def test_124_correlated_consecutive_frames_do_not_shrink_below_floor():
    p = params()
    l4 = Layer4(p, REG)
    feed(l4, [[obs(0.05 * k, (3.5, 3.5), cam=(1.0, 3.5), sd=0.02)] for k in range(100)])
    t = only(l4)
    r = 0.02 ** 2 + p.obs_model_sigma_m ** 2 + 1e-8
    floor_ray = r / p.max_eff_obs_per_bin + p.ray_bias_sigma_m ** 2 + p.offset_sigma_m ** 2  # 3.1c: derived sigma
    floor_lat = r / p.max_eff_obs_per_bin
    assert t.P[0, 0] == pytest.approx(floor_ray, rel=1e-6)  # ray along map x
    assert t.P[1, 1] == pytest.approx(floor_lat, rel=1e-6)
    assert t.n_bins == 1


def test_124_cell_boundary_uncertainty_stays_provisional():
    l4 = layer4()
    feed(l4, [[obs(0.2 * k, (3.0, 3.5), cam=(0.5, 3.5))] for k in range(20)])
    t = only(l4)
    assert t.status is TagStatus.PROVISIONAL and not t.ever_confirmed
    probs = t.probs.as_dict()
    assert probs[(2, 3)] == pytest.approx(0.5, abs=0.05) and probs[(3, 3)] == pytest.approx(0.5, abs=0.05)


def test_124_two_nearby_survivors_stay_two_tracks():
    l4 = layer4()
    feed(l4, [[obs(0.2 * k, (3.5, 2.6), tid=1), obs(0.2 * k, (3.5, 3.4), tid=2)] for k in range(8)])
    assert len(l4.tracks) == 2
    cells = sorted(t.confirmed_cell for t in l4.tracks.values())
    assert cells == [(3, 2), (3, 3)]
    assert all(t.status is TagStatus.CONFIRMED for t in l4.tracks.values())


def test_124_temporary_occlusion_keeps_the_track():
    l4 = layer4()
    feed(l4, [[obs(0.2 * k, (3.5, 3.5))] for k in range(6)])
    feed(l4, [[obs(4.0 + 0.2 * k, (3.52, 3.48))] for k in range(4)])  # 2.8 s unseen, then back
    t = only(l4)
    assert t.n_frames == 10 and t.status is TagStatus.CONFIRMED


# --- association ---------------------------------------------------------------------------------------------

def test_hungarian_matches_brute_force():
    rng = np.random.default_rng(2)
    for shape in ((3, 3), (2, 4), (4, 2), (5, 5), (1, 3)):
        c = rng.uniform(0, 10, size=shape)
        pairs = hungarian(c)
        n = min(shape)
        best = min(
            (sum(c[i, j] if shape[0] <= shape[1] else c[j, i] for i, j in enumerate(perm)), perm)
            for perm in itertools.permutations(range(max(shape)), n)
        )[0]
        assert len(pairs) == n and sum(c[i, j] for i, j in pairs) == pytest.approx(best)


def test_gate_rejects_far_observation():
    l4 = layer4()
    feed(l4, [[obs(0.2 * k, (3.5, 3.5))] for k in range(5)])
    l4.update([obs(1.0, (3.5, 4.3))])  # 0.8 m sideways: outside the gate
    assert len(l4.tracks) == 2


def test_observation_between_two_tracks_is_ambiguous():
    l4 = layer4()
    feed(l4, [[obs(0.2 * k, (3.5, 3.35)), obs(0.2 * k, (3.5, 3.65))] for k in range(4)])
    ids = sorted(l4.tracks)
    before = {i: l4.tracks[i].n_frames for i in ids}
    o = obs(1.0, (3.5, 3.5))
    l4.update([o])
    assert FailureReason.ASSOCIATION_AMBIGUOUS in o.warnings
    assert {i: l4.tracks[i].n_frames for i in ids} == before  # not fused into either
    assert all(l4.tracks[i].ambiguous_streak == 1 for i in ids)


def test_id_merge_two_people_one_detector_id_become_two_tracks():
    l4 = layer4()
    feed(l4, [[obs(0.2 * k, (3.5, 2.6), tid=5), obs(0.2 * k, (3.5, 3.4), tid=5)] for k in range(6)])
    assert len(l4.tracks) == 2
    assert any(h[1] in ("split_from", "split_to") for t in l4.tracks.values() for h in t.history)


def test_tracking_id_is_only_a_hint():
    l4 = layer4()
    feed(l4, [[obs(0.2 * k, (3.5, 2.0), tid=1), obs(0.2 * k, (3.5, 4.0), tid=2)] for k in range(4)])
    a = next(t for t in l4.tracks.values() if t.last_tracking_id == 1)
    b = next(t for t in l4.tracks.values() if t.last_tracking_id == 2)
    l4.update([obs(1.0, (3.5, 4.0), tid=1)])  # detector says id 1, but the position is track b's
    assert (a.n_frames, b.n_frames) == (4, 5)


# --- fusion ----------------------------------------------------------------------------------------------------

def _items(cams, xy=(3.5, 3.5), sd=0.02):
    l4 = layer4()
    out = []
    for k, cam in enumerate(cams):
        out.append(l4._make_item(obs(0.2 * k, xy, cam=cam, sd=sd)))
    return out


def test_opposite_views_reduce_ray_bias():
    p = params()
    _, p_one, _ = fuse(_items([(1.0, 3.5)] * 10), p)
    _, p_opp, n = fuse(_items([(1.0, 3.5)] * 5 + [(6.0, 3.5)] * 5), p)
    assert n == 2
    assert p_opp[0, 0] == pytest.approx(p_one[0, 0] / 2, rel=1e-6)  # two independent biases along one axis


def test_wild_observation_is_not_fused_and_its_candidate_times_out():
    l4 = layer4()
    feed(l4, [[obs(0.2 * k, (3.5, 3.5))] for k in range(5)])
    main = only(l4)
    x_before = main.x.copy()
    l4.update([obs(1.0, (4.2, 3.5))])  # 0.7 m along the ray: outside the gate
    assert np.allclose(main.x, x_before) and len(l4.tracks) == 2
    feed(l4, [[obs(1.2 + 0.2 * k, (3.5, 3.5))] for k in range(15)])
    assert list(l4.tracks) == [main.track_id] and l4.n_dropped == 1


def test_fusion_is_deterministic_and_order_independent():
    rng = np.random.default_rng(3)
    frames = [[obs(0.2 * k, (3.5, 2.6) + rng.normal(0, 0.03, 2)), obs(0.2 * k, (3.5, 3.4) + rng.normal(0, 0.03, 2))]
              for k in range(8)]
    a, b = layer4(), layer4()
    feed(a, frames)
    feed(b, [list(reversed(f)) for f in frames])
    xa = sorted((tuple(t.x), tuple(t.P.ravel())) for t in a.tracks.values())
    xb = sorted((tuple(t.x), tuple(t.P.ravel())) for t in b.tracks.values())
    assert len(xa) == len(xb) and np.allclose(np.array([x for x, _ in xa]), np.array([x for x, _ in xb]))
    assert np.allclose(np.array([q for _, q in xa]), np.array([q for _, q in xb]))
    c = layer4()
    feed(c, frames)
    assert sorted(tuple(t.x) for t in c.tracks.values()) == sorted(tuple(t.x) for t in a.tracks.values())


# --- state machine -----------------------------------------------------------------------------------------------

def test_every_legal_and_illegal_transition():
    l4 = layer4()
    for a, b in itertools.product(TagStatus, TagStatus):
        t = Track(0, 0, status=a)
        if b in LEGAL_TRANSITIONS[a]:
            l4.transition(t, b, "test")
            assert t.status is b
        else:
            with pytest.raises(IllegalTransition):
                l4.transition(t, b, "test")
    assert all(TagStatus.CANDIDATE not in v for v in LEGAL_TRANSITIONS.values())


def _confirmed_track(l4):
    feed(l4, [[obs(0.2 * k, (3.5, 3.5))] for k in range(8)])
    t = only(l4)
    assert t.status is TagStatus.CONFIRMED and t.confirmed_cell == (3, 3)
    return t


def test_hysteresis_changes_cell_only_after_n_updates():
    from survivor_tagging.grid import CellProbabilities

    l4 = layer4(cell_change_updates=3, cell_change_margin=0.2)
    t = _confirmed_track(l4)
    t.probs = CellProbabilities((((4, 3), 0.55), ((3, 3), 0.45)), 0.0)  # new cell ahead, but within the margin
    for _ in range(5):
        l4._update_status(t)
    assert t.status is TagStatus.CONFIRMED and t.change_streak == 0
    t.probs = CellProbabilities((((4, 3), 0.8), ((3, 3), 0.2)), 0.0)
    l4._update_status(t)
    l4._update_status(t)
    assert t.status is TagStatus.CONFIRMED and t.change_streak == 2  # not yet
    l4._update_status(t)
    assert t.status is TagStatus.UPDATED and t.confirmed_cell == (4, 3) and t.cell_changes == 1 and t.revision == 1


def test_no_flicker_on_alternating_evidence():
    from survivor_tagging.grid import CellProbabilities

    l4 = layer4(cell_change_updates=3, cell_change_margin=0.2)
    t = _confirmed_track(l4)
    a = CellProbabilities((((4, 3), 0.8), ((3, 3), 0.2)), 0.0)
    b = CellProbabilities((((3, 3), 0.9), ((4, 3), 0.1)), 0.0)
    for k in range(20):
        t.probs = a if k % 2 == 0 else b
        l4._update_status(t)
    assert t.cell_changes == 0 and t.confirmed_cell == (3, 3)


def test_moving_target_does_not_drag_a_confirmed_track():
    """Survivors are static: steadily moving evidence starts new tracks instead of moving a confirmed one."""
    l4 = layer4()
    t = _confirmed_track(l4)
    for k in range(30):
        l4.update([obs(2.0 + 0.2 * k, (3.5 + min(0.05 * (k + 1), 1.0), 3.5), cam=(3.5, 1.0))])
    assert t.confirmed_cell == (3, 3) and len(l4.tracks) > 1


def test_grid_unregistered_blocks_confirmation():
    reg = GridRegistration((0.0, 0.0), 0.0, n_cells_x=6, n_cells_y=6, valid=False)
    l4 = layer4(reg)
    feed(l4, [[obs(0.2 * k, (3.5, 3.5))] for k in range(20)])
    t = only(l4)
    assert t.status is TagStatus.PROVISIONAL and l4.blocked(t)


def test_single_frame_false_positive_never_confirms_and_is_dropped():
    l4 = layer4()
    feed(l4, [[obs(0.0, (1.5, 4.5))]] + [[obs(0.2 * k, (3.5, 2.5))] for k in range(1, 20)])
    assert all(t.x[0] > 3 for t in l4.tracks.values()) and l4.n_dropped == 1
    assert not any(t.ever_confirmed for t in l4.dropped)


# --- memory bounds -----------------------------------------------------------------------------------------------

def test_track_limit_evicts_lowest_value_with_event():
    l4 = layer4(max_tracks=3)
    feed(l4, [[obs(0.2 * k, (3.5, 3.5))] for k in range(6)])
    l4.update([obs(1.4, (0.5 + i, 0.5)) for i in range(5)])
    assert len(l4.tracks) == 3 and l4.n_evicted == 3
    assert any(e[1] == FailureReason.CAPACITY_EXCEEDED.value and e[3] == "track" for e in l4.events)
    assert any(t.status is TagStatus.CONFIRMED for t in l4.tracks.values())  # the valuable track survives


def test_observation_limit_per_track_with_warning():
    l4 = layer4(max_obs_per_track=10)
    feed(l4, [[obs(0.1 * k, (3.5, 3.5))] for k in range(30)])
    t = only(l4)
    assert len(t.items) == 10 and t.n_frames == 30 and FailureReason.CAPACITY_EXCEEDED in t.warnings
    assert all(it.obs.pose_revision is not None for it in t.items)  # references kept for Layer 5


# --- outputs and properties --------------------------------------------------------------------------------------

def test_final_tag_policies():
    l4 = layer4()
    feed(l4, [[obs(0.2 * k, (3.5, 3.5)), obs(0.2 * k, (1.0, 4.0))] for k in range(6)]
         + [[obs(1.2 + 0.2 * k, (3.5, 3.5))] for k in range(1)])
    l4.tracks[max(l4.tracks)].status = TagStatus.PROVISIONAL  # force one provisional
    conf = l4.final_tags("confirmed_only")
    both = l4.final_tags("confirmed_plus_provisional_top1")
    assert len(conf) == 1 and len(both) == 2
    with pytest.raises(ValueError):
        l4.final_tags("anything")


def test_property_random_streams_legal_finite_bounded():
    rng = np.random.default_rng(4)
    for trial in range(10):
        l4 = layer4(max_tracks=8, max_obs_per_track=15)
        people = rng.uniform(0.5, 5.5, size=(rng.integers(1, 4), 2))
        for k in range(30):
            batch = []
            for i, pxy in enumerate(people):
                if rng.random() < 0.8:
                    cam = pxy + 2.5 * np.array([math.cos(k / 10 + i), math.sin(k / 10 + i)])
                    batch.append(obs(0.2 * k, pxy + rng.normal(0, 0.05, 2), cam=cam, sd=0.03,
                                     tid=int(rng.integers(0, 4))))
            if rng.random() < 0.3:
                batch.append(obs(0.2 * k, rng.uniform(0, 6, 2), sd=0.03))
            l4.update(batch)
            for t in l4.tracks.values():
                assert np.all(np.isfinite(t.x)) and np.all(np.isfinite(t.P))
                assert sum(p for _, p in t.probs.cells) + t.probs.out_of_grid <= 1 + 1e-9
                assert len(t.items) <= 15
            assert len(l4.tracks) <= 8
        for t in list(l4.tracks.values()) + list(l4.dropped):
            for h in t.history:
                if "->" in h[1]:
                    a, b = h[1].split("->")
                    assert TagStatus[b] in LEGAL_TRANSITIONS[TagStatus[a]]
