"""Substep 3.3b: two_close_survivors scenario family (simulation values) and the experiment helpers."""
import math

import numpy as np

from survivor_tagging.scenarios import TWO_CLOSE_PAIRS, TWO_CLOSE_PREFIX, all_scenarios, pre_32_scenarios, two_close_names
from survivor_tagging.sim import place_survivors


def test_family_is_registered_and_excluded_from_pre_32():
    names = two_close_names()
    assert len(names) == len(TWO_CLOSE_PAIRS) == 8
    assert all(n.startswith(TWO_CLOSE_PREFIX) for n in names)
    assert set(names) <= set(all_scenarios(with_two_close=True))
    assert not set(names) & set(all_scenarios())  # opt-in: other tests assume a 0.8 m spacing of every pair
    assert not set(names) & set(pre_32_scenarios())
    assert len(pre_32_scenarios()) == 20


def test_separation_cells_and_no_overlap_for_every_seed():
    scns = all_scenarios(with_two_close=True)
    for tag, _p1, _p2, sep, adj in TWO_CLOSE_PAIRS:
        for seed in range(12):
            placed = place_survivors(scns[TWO_CLOSE_PREFIX + tag], seed)
            a, b = placed.survivors
            d = math.dist(a.grid_xy, b.grid_xy)
            assert abs(d - sep) < 1e-9 and 0.30 - 1e-9 <= d <= 0.60 + 1e-9
            ba, bb = a.box(), b.box()
            assert ba.hi[1] <= bb.lo[1] + 1e-9, "boxes must not overlap (side by side along grid y)"
            cells = [t.cell for t in placed.truths()]
            assert (cells[0][1] != cells[1][1]) == adj
            assert cells[0][0] == cells[1][0]


def test_placement_is_deterministic_and_varies_with_seed():
    scn = all_scenarios(with_two_close=True)[TWO_CLOSE_PREFIX + "stand_045_same"]
    a = place_survivors(scn, 3).survivors[0].grid_xy
    assert a == place_survivors(scn, 3).survivors[0].grid_xy
    assert len({place_survivors(scn, s).survivors[0].grid_xy for s in range(6)}) > 1


def test_postures_cover_standing_and_sitting():
    postures = {p for t in TWO_CLOSE_PAIRS for p in t[1:3]}
    assert postures == {"standing", "sitting"}
    assert any(t[4] for t in TWO_CLOSE_PAIRS) and any(not t[4] for t in TWO_CLOSE_PAIRS)
