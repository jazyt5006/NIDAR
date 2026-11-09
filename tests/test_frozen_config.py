"""configs/frozen_synthetic_v1.json lists every registry parameter with its value, source and evidence level."""
import json
from pathlib import Path

from survivor_tagging.config import default_config

PATH = Path(__file__).resolve().parents[1] / "configs" / "frozen_synthetic_v1.json"
SOURCES = {"rulebook", "ASSUMPTION", "tuned on simulation", "interim", "unset: needs measurement or an answer"}
EVIDENCE = {"none", "unit", "simulation", "Gazebo", "recorded-data", "hardware", "rulebook"}


def _doc():
    return json.loads(PATH.read_text(encoding="utf-8"))


def test_every_registry_parameter_is_listed_exactly():
    reg = {p.name: p for p in default_config().params()}
    listed = _doc()["parameters"]
    assert set(listed) == set(reg)
    for name, p in reg.items():
        assert listed[name]["value"] == p.value, name  # the freeze documents the current defaults


def test_labels_are_valid_and_unset_values_are_not_guessed():
    for name, e in _doc()["parameters"].items():
        assert e["source"] in SOURCES and e["evidence"] in EVIDENCE, name
        if e["value"] is None:
            assert e["source"].startswith("unset") and e["evidence"] == "none", name
        if e["source"] == "rulebook":
            assert name in ("grid.cell_size_m", "score.correct_tag_points")


def test_known_unknowns_stay_unset_or_zero():
    p = _doc()["parameters"]
    assert p["l1.body_depth_offset_m"]["value"] == 0.0
    for name in ("extr.T_body_camera", "grid.origin_in_map_m", "score.wrong_tag_penalty_assumed",
                 "l5.end_of_mission_commit_policy", "cam.fx_px", "clock.camera_to_pose_offset_ns"):
        assert p[name]["value"] is None, name
    assert _doc()["interim_choices"]["final_tag_policy"]["source"] == "interim"
    assert len(_doc()["unfrozen_unknowns"]) >= 10
