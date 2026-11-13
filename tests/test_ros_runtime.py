"""Substep 4.1b: TaggerRuntime tested without ROS (simulator -> messages -> runtime). Evidence level: unit / simulation.
Nothing here proves the rclpy node, the launch files or the message build work (they are "not run")."""
import math
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

import ros_fakes as rf
import ros_runtime_helpers as H
from ros_fakes import adapters as A
from survivor_tagging import output
from survivor_tagging.config import default_config
from survivor_tagging.scenarios import all_scenarios
from survivor_tagging.sim import build_events, place_survivors
from survivor_tagging.sim_pipelines import scenario_config
from survivor_tagging.types import FailureReason as F
from survivor_tagging.types import TagStatus
from survivor_tagging_ros import runtime as R
from survivor_tagging_ros import sim_bridge as B

AC1_SCENARIOS = ("standing", "sitting", "loop_closure", "registration_update", "double_detection")


# --------------------------------------------------------------------------- AC1
@pytest.mark.parametrize("name", AC1_SCENARIOS)
@pytest.mark.parametrize("seed", [0, 1])
def test_ac1_runtime_equals_direct_pipeline(name, seed):
    placed, events, rt, bus = H.run_via_messages(name, seed)
    direct = H.run_direct(placed, events, seed)
    n = H.compare_published(direct, rt.latest_tags)
    assert n == len(placed.truths()) == 1
    assert not rt.rejections, dict(rt.rejections)
    if name in ("loop_closure", "registration_update"):
        assert rt.counts["rebuilds"] == 1  # the correction / registration went through the rebuild and swap path
    assert bus.of("tags") and bus.of("markers") and bus.of("telemetry")


@pytest.mark.parametrize("name", ["loop_closure", "registration_update"])
def test_ac1_also_holds_with_the_rebuild_on_the_worker_thread(name):
    placed, events, rt, bus = H.run_via_messages(name, 0, thread=True)
    H.compare_published(H.run_direct(placed, events, 0), rt.latest_tags)
    assert rt.counts["rebuilds"] == 1


# --------------------------------------------------------------------------- rejections with explicit reasons
def _frame_msgs(name="standing", seed=0):
    placed = place_survivors(all_scenarios()[name], seed)
    events = build_events(placed, seed)
    msgs = B.events_to_messages(placed, events, H.T0_NS)
    return placed, msgs


def _first(msgs, kind):
    return next(m for m in msgs if m.kind == kind)


def _with_stamp_shift(m, delta_ns):
    d = {**m.msg, "header": {**m.msg["header"], "stamp": A.ns_to_stamp(A.stamp_to_ns(NS(**m.msg["header"]["stamp"])) + delta_ns)}}
    return d


def test_depth_matching_reasons_and_skew():
    placed, msgs = _frame_msgs()
    clock, bus = [0], H.Bus()
    rt = H.make_runtime(placed, clock, bus)
    ci, dm, dt, pose = (_first(msgs, k) for k in ("camera_info", "depth", "detections", "pose"))
    clock[0] = ci.arrival_ns
    assert rt.on_camera_info(B.ns(ci.msg)) is None
    assert rt.on_detections(B.ns(dt.msg)) == "RGB_DEPTH_TIME_MISMATCH:no_depth_buffered"
    assert rt.on_depth(B.ns(dm.msg)) is None
    skew_s = float(scenario_config(placed).value("l2.max_rgb_depth_skew_s"))
    late = _with_stamp_shift(dt, int((skew_s + 0.01) * 1e9))
    assert rt.on_detections(B.ns(late)) == "RGB_DEPTH_TIME_MISMATCH:skew_exceeded"
    ok = _with_stamp_shift(dt, int(skew_s * 0.5e9))  # inside the tolerance: matched
    assert rt.on_detections(B.ns(ok)) is None
    assert rt.rejections["RGB_DEPTH_TIME_MISMATCH:no_depth_buffered"] == 1
    assert rt.rejections["RGB_DEPTH_TIME_MISMATCH:skew_exceeded"] == 1
    assert rt.counts["frames"] == 1
    # nearest depth image wins when several are buffered
    clock[0] = _first(msgs, "detections").arrival_ns
    assert rt.diagnostics(clock[0])["queues"]["depth_buffer"] == 1


def test_unset_skew_and_unset_parameters_are_configuration_errors():
    placed = place_survivors(all_scenarios()["standing"], 0)
    cfg = scenario_config(placed)
    cfg["l2.max_rgb_depth_skew_s"].value = None
    with pytest.raises(A.ConfigurationError) as e:
        R.TaggerRuntime(H.adapter_cfg(), cfg, lambda: 0, H.Bus())
    assert "l2.max_rgb_depth_skew_s" in e.value.names
    with pytest.raises(A.ConfigurationError) as e:
        R.TaggerRuntime(H.adapter_cfg(), default_config(), lambda: 0, H.Bus())  # nothing configured at all
    assert {"extr.T_body_camera", "clock.camera_to_pose_offset_ns"} <= set(e.value.names)
    for field in ("map_frame_id", "body_frame_id", "body_convention", "clock_domain"):
        with pytest.raises(A.ConfigurationError) as e:
            R.TaggerRuntime(H.adapter_cfg(**{field: None}), scenario_config(placed), lambda: 0, H.Bus())
        assert field in e.value.names
    cfg2 = scenario_config(placed)
    cfg2["cam.detections_use_depth_aligned_frame"].value = None
    with pytest.raises(A.ConfigurationError) as e:  # neither the adapter nor the library config says it
        R.TaggerRuntime(H.adapter_cfg(depth_aligned_to_color=None), cfg2, lambda: 0, H.Bus())
    assert "depth_aligned_to_color" in e.value.names


def test_zero_stamps_frame_mismatch_and_missing_camera_info():
    placed, msgs = _frame_msgs()
    clock, bus = [0], H.Bus()
    rt = H.make_runtime(placed, clock, bus)
    ci, dm, dt, pose = (_first(msgs, k) for k in ("camera_info", "depth", "detections", "pose"))
    clock[0] = ci.arrival_ns
    # no CameraInfo yet
    assert rt.on_depth(B.ns(dm.msg)) == "INTRINSICS_MISSING:no_camera_info"
    assert rt.on_detections(B.ns(dt.msg)) == "INTRINSICS_MISSING:no_camera_info"
    assert rt.on_camera_info(B.ns({**ci.msg, "k": [0.0] * 9})) == "INTRINSICS_MISSING"  # uncalibrated info
    assert rt.on_camera_info(B.ns(ci.msg)) is None
    assert rt.on_depth(B.ns(dm.msg)) is None
    # zero stamps
    zero = {"stamp": {"sec": 0, "nanosec": 0}}
    assert rt.on_pose(B.ns({**pose.msg, "header": {**pose.msg["header"], **zero}})) == "TIMESTAMP_INVALID"
    assert rt.on_detections(B.ns({**dt.msg, "header": {**dt.msg["header"], **zero}})) == "TIMESTAMP_INVALID"
    assert rt.on_depth(B.ns({**dm.msg, "header": {**dm.msg["header"], **zero}})) == "TIMESTAMP_INVALID"
    # frame mismatch
    assert rt.on_pose(B.ns({**pose.msg, "header": {**pose.msg["header"], "frame_id": "odom"}})) == "FRAME_MISMATCH"
    assert rt.on_detections(B.ns({**dt.msg, "header": {**dt.msg["header"], "frame_id": "other_cam"}})) == \
        "DEPTH_FRAME_MISMATCH"
    assert rt.on_pose(B.ns(pose.msg)) is None
    assert rt.counts["poses"] == 1
    assert rt.rejections["TIMESTAMP_INVALID"] == 3 and rt.rejections["FRAME_MISMATCH"] == 1
    # everything shows up in the diagnostics
    assert rt.diagnostics(clock[0])["rejections"]["DEPTH_FRAME_MISMATCH"] == 1


def test_non_monotonic_pose_stamp_is_rejected():
    placed, msgs = _frame_msgs()
    clock, bus = [0], H.Bus()
    rt = H.make_runtime(placed, clock, bus)
    poses = [m for m in msgs if m.kind == "pose"][:3]
    for m in poses:
        assert rt.on_pose(B.ns(m.msg)) is None
    assert rt.on_pose(B.ns(poses[0].msg)) == "TIMESTAMP_INVALID"  # goes backwards


# --------------------------------------------------------------------------- telemetry rate limiting and freeze parameters
def _telemetry_events(bus):
    out = []
    for p in bus.of("telemetry"):
        data = bytes(p["data"])
        stamp = p["header"]["stamp"]["sec"] * 1e9 + p["header"]["stamp"]["nanosec"]
        if data[0] == output.MSG_TAG_UPDATE:
            out.append(("update", stamp, output.decode_tag_record(data).survivor_id))
        else:
            out.append(("snapshot", stamp, None))
    return out


def test_telemetry_snapshots_every_5_s_and_updates_rate_limited():
    placed, events, rt, bus = H.run_via_messages("revisit_after_gap", 0)
    ev = _telemetry_events(bus)
    snaps = [s for k, s, _ in ev if k == "snapshot"]
    ups = [(s, i) for k, s, i in ev if k == "update"]
    assert len(snaps) >= 2 and ups
    p = rt.pipeline.telemetry_params
    assert (p.snapshot_period_s, p.update_min_interval_s) == (5.0, 1.0)
    gaps = np.diff(snaps) / 1e9
    assert gaps.min() >= 5.0 - 1e-6, gaps  # no corrections or registration changes in this scenario: strictly periodic
    for sid in {i for _, i in ups}:
        t = [s for s, i in ups if i == sid]
        assert all(b - a >= 1.0e9 - 1 for a, b in zip(t, t[1:])), "updates of one tag closer than the interval"
    assert rt.pipeline.telemetry.stats.n_updates >= len(ups) - 1


def test_runtime_parameters_come_from_the_freeze_file(tmp_path):
    frozen = H.rf.ROOT / "configs" / "frozen_synthetic_v1.json"
    base = dict(config_file=str(frozen), overrides_file=str(rf.ROOT / "configs" / "example_synthetic.json"),
                pose_frame="map", body_frame="base_link", body_frame_convention="FLU", clock_domain="ros_time",
                depth_aligned_to_color="true", rebuild_in_thread=False)
    rt = R.build_runtime(base, lambda: 0, H.Bus())
    p = rt.pipeline.telemetry_params
    assert p.snapshot_period_s == 5.0 and p.update_min_interval_s == 1.0 and p.confirm_prob == 0.9
    assert rt.cfg.pose_default_position_sigma_m == 0.10 and rt.cfg.pose_default_yaw_sigma_rad == 0.05  # ASSUMPTION
    assert rt.cfg.map_frame_id == "map" and rt.cfg.depth_aligned_to_color is True and rt.cfg.body_convention == "FLU"
    for missing in ("pose_frame", "body_frame", "body_frame_convention"):
        bad = dict(base, **{missing: "UNSET"})
        with pytest.raises(A.ConfigurationError):
            R.build_runtime(bad, lambda: 0, H.Bus())
    import json  # alignment unset in the node AND in the overrides file -> error (the example file sets it to true)
    ov = json.loads((rf.ROOT / "configs" / "example_synthetic.json").read_text(encoding="utf-8"))
    ov.pop("cam.detections_use_depth_aligned_frame")
    f = tmp_path / "ov.json"
    f.write_text(json.dumps(ov), encoding="utf-8")
    with pytest.raises(A.ConfigurationError) as e:
        R.build_runtime(dict(base, overrides_file=str(f), depth_aligned_to_color="UNSET"), lambda: 0, H.Bus())
    assert "depth_aligned_to_color" in str(e.value) or "cam.detections_use_depth_aligned_frame" in str(e.value)
    with pytest.raises(A.ConfigurationError):  # no overrides file: required library values are unset
        R.build_runtime(dict(base, overrides_file=""), lambda: 0, H.Bus())
    t = np.eye(4)
    t[0, 3] = 0.25
    rt2 = R.build_runtime(dict(base, T_body_camera=[float(v) for v in t.reshape(16)], **{
        "pose.default_position_sigma_m": 0.2, "pose.default_yaw_sigma_rad": "UNSET"}), lambda: 0, H.Bus())
    assert rt2.pipeline.l2.extrinsic.T_body_camera[0, 3] == 0.25
    assert rt2.cfg.pose_default_position_sigma_m == 0.2 and rt2.cfg.pose_default_yaw_sigma_rad is None


# --------------------------------------------------------------------------- services
def test_set_registration_service():
    placed, events, rt, bus = H.run_via_messages("standing", 0)
    req = B.registration_request(NS(version=3, registration=placed.registration), placed.arena.n_cells)
    ok = rt.call_set_registration(B.ns(req))
    assert ok == {"accepted": True, "reason": "", "current_version": 3}
    assert rt.pipeline.grid.version == 3 and rt.latest_tags["reg_rev"] == 3
    again = rt.call_set_registration(B.ns(req))  # same version, same content: duplicate
    assert not again["accepted"] and again["reason"] == "duplicate" and again["current_version"] == 3
    stale = rt.call_set_registration(B.ns({**req, "version": 1}))
    assert not stale["accepted"] and "CORRECTION_STALE" in stale["reason"] and stale["current_version"] == 3
    bad = rt.call_set_registration(B.ns({**req, "version": 4, "numbering": "bogus"}))
    assert not bad["accepted"] and "CORRECTION_INVALID" in bad["reason"] and rt.pipeline.grid.version == 3
    conflict = rt.call_set_registration(B.ns({**req, "origin_in_map_m": [9.0, 9.0]}))  # version 3 again, new content
    assert not conflict["accepted"] and "CORRECTION_CONFLICT" in conflict["reason"]


def test_commit_final_tags_service():
    placed, events, rt, bus = H.run_via_messages("standing", 0)
    r = rt.call_commit_final_tags(NS(policy="confirmed_only", threshold=float("nan")))
    assert r["success"] and r["policy_used"] == "confirmed_only" and math.isnan(r["threshold_used"])
    ev = rt.call_commit_final_tags(NS(policy="expected_value", threshold=0.667))
    assert ev["success"] and ev["threshold_used"] == 0.667 and ev["tags"][0]["has_cell"]
    t = ev["tags"][0]
    assert (t["cell_i"], t["cell_j"]) == placed.truths()[0].cell and t["has_position"]
    default = rt.call_commit_final_tags(NS(policy="expected_value", threshold=float("nan")))
    assert default["success"] and math.isnan(default["threshold_used"])  # library default (P unset: 0.5)
    for bad in (NS(policy="nonsense", threshold=float("nan")), NS(policy="confirmed_only", threshold=0.5),
                NS(policy="expected_value", threshold=2.0)):
        out = rt.call_commit_final_tags(bad)
        assert out["success"] is False and out["reason"] and out["tags"] == []


# --------------------------------------------------------------------------- markers and diagnostics
def test_marker_payload_structure():
    placed, events, rt, bus = H.run_via_messages("standing", 0)
    mk = bus.of("markers")[-1]["markers"]
    cube = next(m for m in mk if m["type"] == R.MARKER_CUBE)
    text = next(m for m in mk if m["type"] == R.MARKER_TEXT)
    pt = rt.pipeline.published()[0]
    cell = pt.tag.cell
    centre = rt.pipeline.grid.grid_to_map([cell[0] + 0.5, cell[1] + 0.5])  # index_base 0, cell size 1 m
    assert np.allclose([cube["pose"]["position"]["x"], cube["pose"]["position"]["y"]], centre)
    assert cube["scale"] == {"x": 1.0, "y": 1.0, "z": 0.02} and cube["header"]["frame_id"] == "map"
    yaw = rt.pipeline.grid.yaw_rad
    assert np.isclose(cube["pose"]["orientation"]["z"], math.sin(yaw / 2)) and cube["action"] == R.ACTION_ADD
    r, g, b, a = R.STATUS_COLOR[pt.tag.status]
    assert (cube["color"]["r"], cube["color"]["g"], cube["color"]["b"], cube["color"]["a"]) == (r, g, b, a)
    assert text["text"].startswith(f"#{pt.survivor_id} {cell[0]},{cell[1]} {pt.tag.status.name}")
    assert cube["id"] != text["id"] and cube["ns"] == "survivor_cells" and text["ns"] == "survivor_labels"
    # removed tag deletes; invalid registration draws no square; colour follows the status
    removed = NS(survivor_id=pt.survivor_id, removed=True, tag=pt.tag)
    assert all(m["action"] == R.ACTION_DELETE for m in R.build_markers([removed], rt.pipeline.grid, "map", 5)["markers"])
    from dataclasses import replace
    unreg = R.build_markers([pt], replace(rt.pipeline.grid, valid=False), "map", 5)["markers"]
    assert [m["type"] for m in unreg] == [R.MARKER_TEXT]
    assert len({R.STATUS_COLOR[s] for s in TagStatus}) == len(TagStatus)


def test_diagnostics_payload_and_one_hz_timer():
    placed, events, rt, bus = H.run_via_messages("loop_closure", 0)
    d = rt.diagnostics(10 * 10**9)
    assert set(d) == {"stamp", "rejections", "counts", "pose_lag_s", "queues", "last_recompute_ms",
                      "last_recompute_cause", "revisions", "registration_valid", "tags"}
    assert set(d["queues"]) == {"depth_buffer", "pending_detections", "pose_buffer", "stored_observations",
                                "rebuilds_in_flight"}
    assert set(d["revisions"]) == {"pose", "map", "registration"} and d["revisions"]["pose"] == 1
    assert d["last_recompute_ms"] > 0 and d["last_recompute_cause"].startswith("correction") and d["registration_valid"]
    assert d["pose_lag_s"] is not None and d["counts"]["corrections"] == 1
    bus.items.clear()
    clock = [rt.now_ns_fn()]
    rt._last_diag_ns = None
    for dt in (0, 0.4, 0.9, 1.1, 1.5, 2.2):
        clock[0] = int(20e9 + dt * 1e9)
        rt.tick(clock[0])
    assert len(bus.of("diagnostics")) == 3  # at 0 s, 1.1 s and 2.2 s


# --------------------------------------------------------------------------- clock and time
def test_capture_time_never_comes_from_the_clock():
    """Shifting the injected arrival clock by 0.2 s (inside every limit) changes nothing: positions, cells and covariance
    come from header stamps. (The pipeline's arrival time only drives staleness limits.)"""
    placed = place_survivors(all_scenarios()["standing"], 0)
    events = build_events(placed, 0)
    msgs = B.events_to_messages(placed, events, H.T0_NS)
    results = []
    for shift in (0, 200_000_000):
        clock, bus = [0], H.Bus()
        rt = H.make_runtime(placed, clock, bus)
        shifted = [B.BridgeMsg(m.arrival_ns + shift, m.kind, m.msg) for m in msgs]
        B.feed_runtime(rt, clock, shifted)
        rt.finish()
        results.append(rt.latest_tags["tags"])
    a, b = results
    assert len(a) == len(b) == 1
    for key in ("position_map", "covariance_map_xy_m2", "cell_i", "cell_j", "status", "observation_count"):
        assert a[0][key] == b[0][key], key
    assert a[0]["first_seen_time"] == b[0]["first_seen_time"]  # first_seen is the capture stamp of the first detection


def test_no_wall_clock_in_adapters_and_runtime():
    import ast
    banned_attr = {"time", "time_ns", "perf_counter", "now", "utcnow", "today"}
    for name in ("adapters.py", "runtime.py"):
        src = (rf.ROS_PKG / "survivor_tagging_ros" / name).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in banned_attr and isinstance(node.value, ast.Name) and \
                    node.value.id in ("time", "_t", "datetime"):
                raise AssertionError(f"{name}:{node.lineno}: wall-clock call {node.value.id}.{node.attr}")
            if isinstance(node, ast.Import):
                assert not any(a.name == "datetime" for a in node.names), name
        if name == "runtime.py":  # the only time module use is waiting (monotonic / sleep) in flush_rebuilds
            used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                    and n.value.id == "_t"}
            assert used <= {"monotonic", "sleep"}, used


# --------------------------------------------------------------------------- threading
def test_rebuild_swap_is_atomic_and_does_not_block_live_updates():
    placed = place_survivors(all_scenarios()["repeated_corrections"], 0)
    events = build_events(placed, 0)
    msgs = B.events_to_messages(placed, events, H.T0_NS)
    first_corr = next(i for i, m in enumerate(msgs) if m.kind == "correction")
    clock, bus = [0], H.Bus()
    rt = H.make_runtime(placed, clock, bus, thread=True)
    gate = threading.Event()
    rt.worker.before_run = lambda job: gate.wait(20)
    B.feed_runtime(rt, clock, msgs[:first_corr + 1])
    assert rt.worker.n_inflight == 1 and rt.counts["rebuilds"] == 0
    old_l4 = rt.pipeline.l4
    n_before = rt.counts["frames"]
    B.feed_runtime(rt, clock, msgs[first_corr + 1:first_corr + 80])  # live path keeps running while the job waits
    assert rt.counts["poses"] > 0 and rt.pipeline.l4 is old_l4 and rt.counts["rebuilds"] == 0
    assert rt.counts["frames"] >= n_before
    gate.set()
    B.feed_runtime(rt, clock, msgs[first_corr + 80:])
    rt.finish()
    rt.close()
    direct = H.run_direct(placed, events, 0)
    H.compare_published(direct, rt.latest_tags)
    assert rt.pipeline.l4 is not old_l4 and rt.counts["rebuilds"] + rt.counts["rebuilds_discarded_stale"] == 2


def test_concurrent_tick_and_input_threads_are_safe():
    placed = place_survivors(all_scenarios()["loop_closure"], 1)
    events = build_events(placed, 1)
    msgs = B.events_to_messages(placed, events, H.T0_NS)
    clock, bus = [0], H.Bus()
    rt = H.make_runtime(placed, clock, bus, thread=True)
    stop, errors = threading.Event(), []

    def ticker():
        try:
            while not stop.is_set():
                rt.tick()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    th = threading.Thread(target=ticker, daemon=True)
    th.start()
    try:
        B.feed_runtime(rt, clock, msgs)
    finally:
        stop.set()
        th.join(10)
    rt.finish()
    rt.close()
    assert not errors, errors
    H.compare_published(H.run_direct(placed, events, 1), rt.latest_tags)


# --------------------------------------------------------------------------- POSE_COV_ASSUMED
def test_default_pose_covariance_only_when_unknown():
    unknown = rf.pose_msg()
    p = A.pose_from_msg(unknown, H.adapter_cfg(pose_default_position_sigma_m=0.1, pose_default_yaw_sigma_rad=0.05))
    assert p.covariance_assumed and np.allclose(np.diag(p.covariance), [0.01] * 3 + [0.0025] * 3)
    assert np.count_nonzero(p.covariance - np.diag(np.diag(p.covariance))) == 0
    cov = [0.0] * 36
    for i in range(6):
        cov[i * 6 + i] = 0.5
    known = A.pose_from_msg(rf.pose_msg(cov=cov), H.adapter_cfg(pose_default_position_sigma_m=0.1,
                                                                pose_default_yaw_sigma_rad=0.05))
    assert not known.covariance_assumed and known.covariance[0, 0] == 0.5
    none = A.pose_from_msg(unknown, H.adapter_cfg())  # explicitly disabled (None, None): no covariance, no flag
    assert none.covariance is None and not none.covariance_assumed
    assert A.pose_from_msg(unknown, A.AdapterConfig(map_frame_id="map", body_frame_id="b", body_convention="FLU",
                                                    clock_domain="x")).covariance_assumed  # the defaults 0.10 / 0.05
    with pytest.raises(A.ConfigurationError):
        A.pose_from_msg(unknown, H.adapter_cfg(pose_default_position_sigma_m=-1.0, pose_default_yaw_sigma_rad=0.05))


def test_pose_cov_assumed_warning_reaches_the_tag_and_is_not_blocking():
    on = dict(pose_default_position_sigma_m=0.10, pose_default_yaw_sigma_rad=0.05)
    placed, events, rt_on, _ = H.run_via_messages("standing", 0, **on)
    placed, events, rt_off, _ = H.run_via_messages("standing", 0)
    t_on = A.tag_from_msg_dict(rt_on.latest_tags["tags"][0])[0]
    t_off = A.tag_from_msg_dict(rt_off.latest_tags["tags"][0])[0]
    assert F.POSE_COV_ASSUMED in t_on.reasons and F.POSE_COV_ASSUMED not in t_off.reasons
    from survivor_tagging.types import FAILURE_SEVERITY, Severity
    assert FAILURE_SEVERITY[F.POSE_COV_ASSUMED] is Severity.DEGRADE  # not BLOCK: it does not forbid CONFIRMED
    assert not rt_on.pipeline.l4.blocked(next(iter(rt_on.pipeline.l4.tracks.values())))
    # the default widens the position covariance (more uncertainty than "pose is exact")
    assert t_on.cov_map_m2[0, 0] + t_on.cov_map_m2[1, 1] > t_off.cov_map_m2[0, 0] + t_off.cov_map_m2[1, 1]
    # the warning survives a rebuild (loop_closure re-derives every observation)
    _, _, rt_lc, _ = H.run_via_messages("loop_closure", 0, **on)
    assert F.POSE_COV_ASSUMED in A.tag_from_msg_dict(rt_lc.latest_tags["tags"][0])[0].reasons
    assert rt_lc.counts["rebuilds"] == 1


def test_assumed_flag_propagates_through_interpolation_and_lookup():
    from survivor_tagging.transforms import PoseBuffer, interpolate_pose
    from survivor_tagging.types import PoseSample
    a = PoseSample(10, "d", "map", "b", (0, 0, 0), (1, 0, 0, 0), True, 0, "s", np.eye(6) * 0.01, True)
    b = PoseSample(20, "d", "map", "b", (1, 0, 0), (1, 0, 0, 0), True, 0, "s", np.eye(6) * 0.01, False)
    c = PoseSample(30, "d", "map", "b", (2, 0, 0), (1, 0, 0, 0), True, 0, "s", np.eye(6) * 0.01, False)
    assert interpolate_pose(a, b, 15)[0].covariance_assumed and not interpolate_pose(b, c, 25)[0].covariance_assumed
    buf = PoseBuffer(10.0, 10)
    for p in (a, b, c):
        buf.add(p)
    assert buf.lookup(10, 1.0).warnings == (F.POSE_COV_ASSUMED,)  # exact stamp
    assert F.POSE_COV_ASSUMED in buf.lookup(15, 1.0).warnings  # interpolated across an assumed pose
    assert buf.lookup(25, 1.0).warnings == ()
