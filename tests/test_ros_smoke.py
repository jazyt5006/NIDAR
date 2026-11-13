"""AC4: the smoke-test tool runs to PASS in dry-run mode here (in-process fake publisher/subscriber layer, no rclpy)
and writes smoke_report.json; a failing run exits non-zero."""
import json

import ros_fakes as rf  # noqa: F401
from survivor_tagging_ros import smoke_test as S


def test_ac4_dry_run_passes_and_writes_the_report(tmp_path, capsys):
    out = tmp_path / "smoke_report.json"
    rc = S.main(["--dry-run", "--out", str(out)])
    assert rc == 0 and "PASS" in capsys.readouterr().out
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["pass"] is True and rep["mode"] == "dry-run" and rep["evidence"] == "simulation" and not rep["failures"]
    assert set(rep["scenarios"]) == set(S.DEFAULT_SCENARIOS) and len(rep["runs"]) == 10
    for r in rep["runs"]:
        assert r["equal_to_direct_pipeline"] and r["n_correct"] >= 0 and r["messages"]["tags"] > 0
        assert {"n_truth", "n_correct", "n_correct_confirmed", "n_missed", "n_false", "rebuilds"} <= set(r)
    for kind in ("pose", "depth", "detections", "camera_info", "correction", "registration"):
        assert rep["latency_ms"][kind]["n"] > 0 and rep["latency_ms"][kind]["p95"] >= rep["latency_ms"][kind]["median"]
    assert rep["rejections"] == {}
    v = rep["versions"]
    assert {"python", "numpy", "survivor_tagging", "platform", "ROS_DISTRO"} <= set(v)
    for scn in S.DEFAULT_SCENARIOS:
        assert sum(r["n_correct"] for r in rep["runs"] if r["scenario"] == scn) >= 1


def test_failing_run_exits_non_zero(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(S.R, "TaggerRuntime", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = tmp_path / "r.json"
    assert S.main(["--dry-run", "--scenarios", "standing", "--seeds", "0", "--out", str(out)]) == 1
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["pass"] is False and "boom" in rep["failures"][0]
    assert "FAIL" in capsys.readouterr().out


def test_a_difference_from_the_direct_pipeline_is_a_failure():
    res = {"runs": [{"scenario": "standing", "seed": 0, "equal_to_direct_pipeline": False, "diffs": ["x"], "n_correct": 1}]}
    assert S.evaluate(res, True)["pass"] is False
    res["runs"][0].update(equal_to_direct_pipeline=True, n_correct=0)
    assert S.evaluate(res, True)["pass"] is False  # nobody tagged in the correct cell
    res["runs"][0]["n_correct"] = 1
    assert S.evaluate(res, True)["pass"] is True
    assert S.evaluate({**res, "rejections": {"REBUILD_FAILED": 1}}, True)["pass"] is False


def test_written_sim_overrides_build_a_working_runtime_from_node_parameters(tmp_path):
    """The file the runbook tells the colleague to generate works with build_runtime (the node's own construction path)."""
    import ros_runtime_helpers as H
    from survivor_tagging_ros import runtime as R
    from survivor_tagging_ros import sim_bridge as B
    f = tmp_path / "sim_overrides.json"
    assert S.main(["--write-sim-overrides", str(f)]) == 0
    doc = json.loads(f.read_text(encoding="utf-8"))
    assert "SIMULATION" in doc["_note"] and "extr.T_body_camera" in doc and "grid.origin_in_map_m" in doc
    params = dict(config_file=str(rf.ROOT / "configs" / "frozen_synthetic_v1.json"), overrides_file=str(f),
                  pose_frame="map", body_frame="body", body_frame_convention="FLU", clock_domain="sim",
                  depth_aligned_to_color="true", rebuild_in_thread=False,
                  **{"pose.default_position_sigma_m": "UNSET", "pose.default_yaw_sigma_rad": "UNSET"})
    clock, bus = [0], H.Bus()
    rt = R.build_runtime(params, lambda: clock[0], bus)
    from survivor_tagging.scenarios import all_scenarios
    from survivor_tagging.sim import build_events, place_survivors
    placed = place_survivors(all_scenarios()["standing"], 0)
    events = build_events(placed, 0)
    B.feed_runtime(rt, clock, B.events_to_messages(placed, events, H.T0_NS))
    rt.finish()
    H.compare_published(H.run_direct(placed, events, 0), rt.latest_tags)
