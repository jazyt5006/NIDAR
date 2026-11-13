"""Substep 4.3 R4: preflight checklist on the frozen config (unit tests)."""

import json
from pathlib import Path

from survivor_tagging.config import default_config
from survivor_tagging.preflight import UNSET_TABLE, main, run_preflight

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "configs" / "frozen_synthetic_v1.json"


def test_every_unset_parameter_of_the_frozen_config_is_listed_with_a_reason():
    listed = json.loads(FROZEN.read_text(encoding="utf-8"))["parameters"]
    unset = {k for k, e in listed.items() if e["value"] is None}
    rep = run_preflight(FROZEN)
    names = {i.name for i in rep.unset}
    assert names == unset and len(unset) >= 30
    assert all(i.reason and "no preflight classification" not in i.reason for i in rep.unset)
    assert unset <= set(UNSET_TABLE)  # explicit table entry for each, none fell through to the default
    required_unset = {k for k, e in listed.items() if e["value"] is None and e["required"]}
    assert required_unset <= names


def test_classification_of_known_cases():
    rep = {i.name: i for i in run_preflight(FROZEN).unset}
    for name in ("extr.T_body_camera", "grid.origin_in_map_m", "grid.registration_valid", "clock.camera_to_pose_offset_ns",
                 "l2.pose_mode", "cam.intrinsics_source", "l5.correction_source"):
        assert rep[name].blocking, name
    for name in ("grid.origin_sigma_m", "grid.yaw_sigma_rad", "map.floor_z_m", "score.wrong_tag_penalty_assumed",
                 "l5.end_of_mission_commit_policy", "cam.depth_scale_m_per_unit"):
        assert not rep[name].blocking, name


def test_report_text_and_checks_on_frozen_config():
    rep = run_preflight(FROZEN)
    text = rep.format()
    assert "BLOCKING unset parameters" in text and "NON-BLOCKING unset parameters" in text
    assert "VERDICT: NOT READY" in text and "FROZEN FOR SYNTHETIC DATA, v1" in text and "sha256" in text
    topics = {c.topic: c for c in rep.checks}
    assert topics["registration"].status == "BLOCK" and topics["extrinsics"].status == "BLOCK"
    assert topics["registration uncertainty"].status == "WARN" and topics["floor height"].status == "WARN"
    assert topics["pose covariance policy"].status == "NOT CHECKED"
    assert not rep.ready


def test_main_exit_codes(capsys, tmp_path):
    assert main([str(FROZEN)]) == 1
    assert "VERDICT: NOT READY" in capsys.readouterr().out
    assert main([str(tmp_path / "missing.json")]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert main([str(bad)]) == 2
    assert main([]) == 2


def test_ready_config_and_registry_coverage(tmp_path):
    """A config with every blocking item set is READY; invented-value guard: the registry's unset names are all covered."""
    cfg = default_config()
    assert {p.name for p in cfg.params() if p.value is None} <= set(UNSET_TABLE)
    ov = {"cam.intrinsics_source": "device", "cam.depth_frame_id": "d", "cam.color_frame_id": "c",
          "cam.detections_use_depth_aligned_frame": True, "extr.T_body_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
          "extr.T_body_camera_cov": [[0.0] * 6 for _ in range(6)], "extr.version": 1,
          "clock.camera_to_pose_offset_ns": 0, "clock.offset_uncertainty_ns": 0, "l1.depth_min_m": 0.1, "l1.depth_max_m": 5.0,
          "l1.depth_noise_model": [0.0, 0.0, 0.002], "l2.max_interp_gap_s": 0.1, "l2.max_rgb_depth_skew_s": 0.01,
          "l2.pose_mode": "full_3d", "grid.origin_in_map_m": [0.0, 0.0], "grid.yaw_map_to_grid_rad": 0.0,
          "grid.axis_flip_x": False, "grid.axis_flip_y": False, "grid.index_base": 0, "grid.numbering": "ij",
          "grid.registration_valid": True, "l5.correction_source": "none"}
    p = tmp_path / "ready.json"
    p.write_text(json.dumps(ov), encoding="utf-8")
    rep = run_preflight(p)
    assert rep.ready and not rep.blocking and rep.fmt == "override file"
    assert main([str(p)]) == 0
