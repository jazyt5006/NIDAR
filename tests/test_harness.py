import math

import pytest

import numpy as np

from survivor_tagging.baseline import NaivePipeline, naive_factory
from survivor_tagging.harness import evaluate, format_table, run_scenario, score_tags
from survivor_tagging.scenarios import SIM_MOUNT_XYZ_M, all_scenarios, make_ideal, make_scenario
from survivor_tagging.sim import SurvivorSpec, SurvivorTruth, camera_mount
from survivor_tagging.types import TagResult, TagStatus


def test_naive_baseline_is_perfect_in_ideal_conditions():
    scn = make_ideal(all_scenarios(placement="nominal")["standing"])  # survivor centred in its cell
    for seed in range(3):
        s = run_scenario(scn, naive_factory(scn), seed).score
        assert (s.n_truth, s.n_correct, s.n_missed, s.n_false, s.n_duplicate) == (1, 1, 0, 0, 0)


def test_harness_detects_wrong_camera_calibration():
    tilt = 30.0
    # Survivor 3 m ahead of the camera (camera x = drone x + 0.1).
    scn = make_ideal(make_scenario("cal", "cal", (SurvivorSpec(1, (3.4, 3.5), "standing"),),
                                   path_grid=((0.3, 3.5),), tilt_deg=tilt, placement="nominal"))
    reg = scn.registration

    def run(tilt_deg):
        T = camera_mount(math.radians(tilt_deg), SIM_MOUNT_XYZ_M)
        return run_scenario(scn, NaivePipeline(T, reg.offset_m, reg.yaw_rad, reg.cell_size_m), 0).score

    good, bad = run(tilt), run(tilt + 10.0)
    assert good.n_correct / good.n_truth == 1.0
    assert bad.n_correct / bad.n_truth < 1.0
    assert bad.errors_m[0] > good.errors_m[0] + 0.1


def _tag(xy, cell, status=TagStatus.CONFIRMED, confirm=2_000_000_000):
    return TagResult(0, cell, status, None if xy is None else np.array(xy, dtype=float), 1_000_000_000, confirm)


TRUTHS = [SurvivorTruth(1, (0.5, 0.5), (0, 0), "standing"), SurvivorTruth(2, (5.5, 5.5), (5, 5), "standing")]


def test_score_correct_and_missed():
    s = score_tags(TRUTHS, [_tag((0.6, 0.4), (0, 0))])
    assert (s.n_correct, s.n_missed, s.n_false, s.n_duplicate) == (1, 1, 0, 0)
    assert len(s.errors_m) == 1 and abs(s.errors_m[0] - math.hypot(0.1, 0.1)) < 1e-12
    assert s.confirm_times_s == [2.0]


def test_score_wrong_cell_false_and_duplicate():
    tags = [
        _tag((0.9, 0.5), (1, 0)),  # nearest to truth 1, wrong cell
        _tag((0.4, 0.5), (0, 0)),  # nearer to truth 1 -> becomes the match (correct); the other is a duplicate
        _tag((3.0, 3.0), (3, 3)),  # > 1 m from both -> false
        _tag(None, (5, 5)),  # no position -> false
        _tag((5.5, 5.4), (5, 4), status=TagStatus.UNRELIABLE),  # not scored
    ]
    s = score_tags(TRUTHS, tags)
    assert (s.n_correct, s.n_missed, s.n_false, s.n_duplicate) == (1, 1, 2, 1)


def test_score_tag_shared_by_two_truths_counts_once():
    truths = [SurvivorTruth(1, (3.1, 3.5), (3, 3), "standing"), SurvivorTruth(2, (3.9, 3.5), (3, 3), "standing")]
    s = score_tags(truths, [_tag((3.45, 3.5), (3, 3))])
    assert (s.n_correct, s.n_missed, s.n_false, s.n_duplicate) == (1, 1, 0, 0)


def test_score_unconfirmed_has_no_confirm_time():
    s = score_tags(TRUTHS[:1], [_tag((0.5, 0.5), (0, 0), status=TagStatus.PROVISIONAL, confirm=None)])
    assert s.n_correct == 1 and s.confirm_times_s == []


def test_evaluate_and_table():
    scn = all_scenarios()["standing"]
    rows = evaluate(naive_factory, [scn], seeds=[0, 1])
    assert rows[0].runs == 2 and rows[0].n_truth == 2
    table = format_table(rows)
    assert "standing" in table and "correct" in table


# --- camera-frame metric and placement --------------------------------------

from survivor_tagging.harness import FALSE_ASSOC_DIST_M, PointEstimate, camera_row, score_camera_frame  # noqa: E402
from survivor_tagging.sim import Frame, place_survivors  # noqa: E402
from survivor_tagging.types import Detection, FailureReason  # noqa: E402


def test_camera_frame_metric_on_hand_built_estimates():
    scn = make_scenario("m", "m", (SurvivorSpec(1, (3.5, 3.5), "standing"),), placement="nominal")
    # Standing box: x [3.25, 3.75], y [3.35, 3.65], z [0, 1.7]; identity camera pose -> camera = grid coordinates.
    planned = {
        0: PointEstimate(np.array([3.5, 3.5, 1.25])),  # inside the box, error 0.4
        1: PointEstimate(np.array([3.5, 3.5, 2.75])),  # 1.05 m above the box: false association, error 1.9
        2: PointEstimate(None, FailureReason.DEPTH_AMBIGUOUS_CLUSTERS),
        3: PointEstimate(np.array([0.0, 0.0, 0.0])),  # false positive detection: not scored
    }
    dets = tuple(Detection(i, 0.9, (0, 0, 10, 10), 0, "sim", "c", (424, 240)) for i in planned)
    frame = Frame(0, dets, None, None, (1,), (1, 1, 1, None), ((1, (3.5, 3.5, 0.85)),), np.eye(4))
    cs = score_camera_frame(scn, [frame], lambda det, depth, intr: planned[det.detection_id])
    assert (cs.n_det, cs.n_valid, cs.n_ambiguous, cs.n_false_assoc) == (3, 2, 1, 1)
    assert cs.errors_m == pytest.approx([0.4, 1.9])
    row = camera_row("m", cs)
    assert row.valid_rate == pytest.approx(2 / 3) and row.ambiguous_rate == pytest.approx(1 / 3)
    assert row.false_assoc_rate == pytest.approx(0.5) and row.err_median_m == pytest.approx(1.15)
    assert len(cs.times_ms) == 3 and FALSE_ASSOC_DIST_M == 0.5


def test_random_in_cell_placement_reproducible_and_inside_cell():
    for scn in all_scenarios().values():
        nominal = {t.survivor_id: t.cell for t in scn.truths()}
        for seed in range(10):
            a, b = place_survivors(scn, seed), place_survivors(scn, seed)
            assert [s.grid_xy for s in a.survivors] == [s.grid_xy for s in b.survivors]
            for s, s0 in zip(a.survivors, scn.survivors):
                (xl, xh), (yl, yh) = s0.centre_region(1.0)
                assert xl <= s.grid_xy[0] <= xh and yl <= s.grid_xy[1] <= yh
            assert {t.survivor_id: t.cell for t in a.truths()} == nominal
            if len(a.survivors) == 2:
                d = np.subtract(a.survivors[0].grid_xy, a.survivors[1].grid_xy)
                assert np.linalg.norm(d) == pytest.approx(0.8)
        assert len({place_survivors(scn, s).survivors[0].grid_xy for s in range(5)}) == 5  # seeds differ
    assert all(place_survivors(all_scenarios()["cell_boundary"], s).survivors[0].grid_xy[0] == 3.0 for s in range(5))


def test_nominal_placement_flag_keeps_authored_centres():
    scn = all_scenarios(placement="nominal")["standing"]
    assert place_survivors(scn, 3).survivors[0].grid_xy == (3.5, 3.5)


def test_horizontal_error_metric_on_hand_built_estimates():
    # Level camera at grid (0, 0, 1.5) looking along grid +x: optical z -> grid x, x -> -grid y, y -> -grid z.
    T = np.eye(4)
    T[:3, :3] = camera_mount(0.0)[:3, :3]
    T[:3, 3] = (0.0, 0.0, 1.5)
    scn = make_scenario("h", "h", (SurvivorSpec(1, (3.5, 0.0), "standing"),), placement="nominal")
    truth = np.array([0.0, 1.5 - 0.85, 3.5])  # box centre (3.5, 0, 0.85) in the optical frame
    planned = {
        0: PointEstimate(truth + np.array([0.0, 0.4, 0.3])),  # 0.4 down (vertical only) + 0.3 forward -> h 0.3
        1: PointEstimate(truth + np.array([0.1, 0.0, 0.0])),  # 0.1 right -> h 0.1
        2: PointEstimate(truth + np.array([0.0, 0.0, -0.2])),  # 0.2 short -> h 0.2
    }
    dets = tuple(Detection(i, 0.9, (0, 0, 10, 10), 0, "sim", "c", (424, 240)) for i in planned)
    frame = Frame(0, dets, None, None, (1,), (1, 1, 1), ((1, tuple(truth)),), T)
    cs = score_camera_frame(scn, [frame], lambda det, depth, intr: planned[det.detection_id])
    assert cs.h_errors_m == pytest.approx([0.3, 0.1, 0.2])
    assert cs.errors_m == pytest.approx([0.5, 0.1, 0.2])
    row = camera_row("h", cs)
    assert row.h_err_median_m == pytest.approx(0.2)
    assert row.h_below == pytest.approx((1 / 3, 2 / 3, 1.0))


# --- substep 3.1 metrics -------------------------------------------------------------------------------------------

def test_status_aware_scores_reliability_and_points():
    from survivor_tagging.harness import mission_points, reliability_table

    truths = [SurvivorTruth(1, (0.5, 0.5), (0, 0), "standing"), SurvivorTruth(2, (3.5, 3.5), (3, 3), "standing")]
    tags = [
        TagResult(0, (0, 0), TagStatus.CONFIRMED, np.array([0.5, 0.5]), 0, 2 * 10**9, 0.95, 1),  # correct
        TagResult(1, (2, 3), TagStatus.CONFIRMED, np.array([3.0, 3.5]), 0, 3 * 10**9, 0.9),  # wrong cell
        TagResult(2, (5, 5), TagStatus.PROVISIONAL, np.array([5.5, 5.5]), 0, None, 0.6),  # unmatched
    ]
    s = score_tags(truths, tags, first_seen_ns={1: 10**9, 2: 10**9})
    assert (s.n_correct_confirmed, s.n_false_confirmed, s.n_false_provisional, s.n_flicker) == (1, 1, 1, 1)
    assert s.confirm_from_first_obs_s == pytest.approx([1.0, 2.0])
    assert sorted(s.reliability) == [(0.6, False), (0.9, False), (0.95, True)]
    rows = {r[0]: r for r in reliability_table(s.reliability)}
    assert rows[0.95][2:] == (1, 0.95, 1.0) and rows[0.9][2] == 1
    # ASSUMED scoring: +40 for the correct survivor, two wrong committed tags
    assert mission_points(truths, tags[:2], 40, -20) == 40 - 20
    assert mission_points(truths, tags, 40, -20) == 40 - 40
    assert mission_points(truths, tags, 40, 0) == 40


def test_layer1234_confirms_ideal_standing_survivor():
    from survivor_tagging.sim_pipelines import layer1234_factory

    scn = make_ideal(all_scenarios(placement="nominal")["standing"])
    r = run_scenario(scn, layer1234_factory()(scn), 0)
    assert [t.status for t in r.tags] == [TagStatus.CONFIRMED]
    assert r.score.n_correct_confirmed == 1 and r.score.n_false_confirmed == 0
