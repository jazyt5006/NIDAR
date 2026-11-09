"""Substep 3.3a: near-duplicate observations end to end (SIMULATION tests on synthetic data)."""

from collections import defaultdict
from dataclasses import replace

import numpy as np
import pytest

from survivor_tagging.harness import run_scenario
from survivor_tagging.scenarios import SIM_DETECTOR, all_scenarios, make_scenario, SIM_DOUBLE_DETECTION_PROB
from survivor_tagging.sim import Closure, SurvivorSpec
from survivor_tagging.sim_pipelines import layer123_factory, layer1234_factory, layer5_factory
from survivor_tagging.types import FailureReason, TagStatus

DUP = FailureReason.NEAR_DUPLICATE_OBSERVATION
AMB = FailureReason.ASSOCIATION_AMBIGUOUS
SEEDS = range(5)


def _l4_tracks(res):
    l4 = res.pipeline.l4
    return list(l4.tracks.values()) + list(l4.dropped)


def test_double_detection_survivor_is_tagged_and_never_lost_to_an_ambiguity_streak():
    scn = all_scenarios()["double_detection"]
    correct = 0
    for seed in SEEDS:
        res = run_scenario(scn, layer1234_factory()(scn), seed)
        correct += res.score.n_correct
        l4 = res.pipeline.l4
        assert not any(t.status is TagStatus.UNRELIABLE for t in l4.tracks.values())
        assert not any(ev.endswith("->UNRELIABLE") and why == AMB.value
                       for t in _l4_tracks(res) for _, ev, why in t.history)
        assert res.score.n_duplicate == 0 and res.score.n_false == 0
        assert any(DUP in t.warnings for t in l4.tracks.values())  # the duplicates really occurred and were noted
    assert correct / len(SEEDS) >= 0.8


def test_double_detection_no_observation_is_counted_as_ambiguous():
    scn = all_scenarios()["double_detection"]
    res = run_scenario(scn, layer1234_factory()(scn), 0)
    obs = res.pipeline.observations()
    assert any(DUP in o.warnings for o in obs) and not any(AMB in o.warnings for o in obs)


def test_realised_estimate_separation_of_the_two_boxes_is_small_and_on_the_body():
    """Documented limitation: the two boxes' ESTIMATES land on the same body (a flat-faced body gives the same surface
    point), so they differ by noise (cm), not by the 0.1-0.4 m box offset."""
    scn = all_scenarios()["double_detection"]
    seps = []
    for seed in SEEDS:
        res = run_scenario(scn, layer123_factory()(scn), seed)
        by = defaultdict(list)
        for o in res.pipeline.observations():
            if o.failure is None and o.p_map_m is not None:
                by[o.capture_stamp_ns].append(np.asarray(o.p_map_m[:2]))
        seps += [float(np.linalg.norm(v[0] - v[1])) for v in by.values() if len(v) == 2]
    assert len(seps) >= 20 and np.percentile(seps, 90) < 0.25 and np.median(seps) < 0.1


@pytest.mark.parametrize("name", ["two_adjacent", "two_same_cell", "id_merge"])
def test_real_neighbouring_survivors_are_never_collapsed(name):
    """0.8 m apart (>= l4.min_separate_survivor_distance_m = 0.5): both survivors keep their own well-supported track.
    A false-positive box on a person, or a rare frame in which one body's estimate lands on the other, may be dropped
    as a near-duplicate, but only a few observations (< 5 %)."""
    scn = all_scenarios()[name]
    n_obs = n_dup = 0
    for seed in SEEDS:
        res = run_scenario(scn, layer1234_factory()(scn), seed)
        obs = [o for o in res.pipeline.observations() if o.failure is None and o.p_map_m is not None]
        n_obs += len(obs)
        n_dup += sum(DUP in o.warnings for o in obs)
        assert sum(1 for t in res.pipeline.l4.tracks.values() if t.n_frames >= 5) >= 2
        assert res.score.n_missed == 0
    assert n_dup / n_obs < 0.05


def test_near_duplicate_warning_is_not_doubled_by_a_layer5_recompute():
    scn = make_scenario("dup_closure", "double detection with a loop closure", (SurvivorSpec(1, (3.5, 3.5), "standing"),),
                        detector=replace(SIM_DETECTOR, dup_prob=SIM_DOUBLE_DETECTION_PROB),
                        trajectory_kw=dict(loop_closure_s=1.5, loop_closure_jump_m=(0.0, 0.3, 0.0),
                                           closures=(Closure(1.5),)))
    res = run_scenario(scn, layer5_factory()(scn), 1)
    obs = res.pipeline.observations()
    assert any(DUP in o.warnings for o in obs)
    assert all(o.warnings.count(DUP) <= 1 for o in obs)
    assert sum(1 for t in res.pipeline.l4.tracks.values() if t.n_frames >= 5) == 1
