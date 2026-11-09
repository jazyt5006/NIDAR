"""Gate P1 of substep 3.4: the Layers 1-5 outputs must not change while the code is optimised.

tests/golden/layer5_golden.json.gz was recorded BEFORE the optimisation (survivor_tagging.golden record): every
scenario (incl. two_close_survivors), 3 seeds, body offsets 0.0 and 0.2, final tags, published tags with revisions and
cell probabilities, final-tag sets, counts and the score fields, plus the correction scenarios in affected_tracks mode.
Decisions (cells, statuses, counts, ids, revisions) must be equal; floats within 1e-9.
This test runs seed 0 (fast, in parallel); the full 3-seed comparison is `python -m survivor_tagging.golden check`.
SIMULATION data.
"""
import json

from survivor_tagging import golden


def test_golden_file_is_small_and_complete():
    assert golden.GOLDEN_PATH.stat().st_size <= 200 * 1024
    gold = golden.load()
    assert len(gold["runs"]) == 234 and set(gold) == {"runs", "pooled"}
    assert {k.split("|")[3] for k in gold["runs"]} == {"0", "1", "2"}


def test_diff_detects_changes_and_tolerates_rounding():
    a = {"cell": [3, 3], "p": 0.5, "s": "CONFIRMED"}
    assert golden.diff(a, {"cell": [3, 3], "p": 0.5 + 5e-10, "s": "CONFIRMED"}) == []
    assert golden.diff(a, {"cell": [3, 3], "p": 0.5 + 5e-9, "s": "CONFIRMED"})
    assert golden.diff(a, {"cell": [3, 4], "p": 0.5, "s": "CONFIRMED"})
    assert golden.diff(a, {"cell": [3, 3], "p": 0.5, "s": "UPDATED"})


def test_layer5_outputs_equal_the_golden_file_seed0():
    gold = golden.load()
    new = json.loads(json.dumps(golden.capture(seeds=(0,))))
    ref = {k: v for k, v in gold["runs"].items() if k.endswith("|0")}
    assert len(new["runs"]) == len(ref) == 78
    d = golden.diff(ref, new["runs"])
    assert not d, d[:5]
