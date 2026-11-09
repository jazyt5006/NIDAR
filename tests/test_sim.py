import math

import numpy as np
import pytest

from survivor_tagging.scenarios import all_scenarios, make_ideal, make_scenario
from survivor_tagging.sim import (
    HIT_FLOOR,
    SIM_DEPTH_SCALE_M,
    Arena,
    DetectorSim,
    Registration,
    Scenario,
    SimCamera,
    SurvivorSpec,
    Trajectory,
    build_events,
    camera_mount,
    camera_pose_in_grid,
    cast_rays,
    project_points,
    render_depth_m,
    wall,
)

CAM_XYZ = (0.5, 2.5, 1.2)


def _level_scene(walls=(), survivors=()):
    """Identity registration, level camera at CAM_XYZ looking along grid +x, no wobble, no outer walls."""
    cam = SimCamera(T_body_camera=camera_mount(0.0))
    traj = Trajectory(waypoints_map=(CAM_XYZ,))
    return Scenario("t", "t", Arena(6, tuple(walls), outer_walls=False), tuple(survivors), cam, DetectorSim(),
                    traj, Registration(), duration_s=0.4)


def test_wall_at_known_distance_gives_expected_depth():
    scn = _level_scene(walls=[wall("x", 3.025, 0.0, 6.0)])  # face at x = 3.0, so depth Z = 2.5 m
    cam = scn.camera
    T = camera_pose_in_grid(scn, 0.0)
    z = render_depth_m(T, cam, scn.boxes())
    assert z[int(cam.cy), int(cam.cx)] == pytest.approx(2.5, abs=1e-9)
    _, hit = cast_rays(T[:3, 3], cam.pixel_rays() @ T[:3, :3].T, scn.boxes())
    on_wall = (hit == 0).reshape(z.shape)
    assert on_wall.sum() > 1000
    assert np.allclose(z[on_wall], 2.5, atol=1e-9)
    # Bottom row hits the floor first: Z = h / ((v - cy) / fy).
    v = cam.height - 1
    assert (hit.reshape(z.shape)[v, int(cam.cx)]) == HIT_FLOOR
    assert z[v, int(cam.cx)] == pytest.approx(CAM_XYZ[2] / ((v - cam.cy) / cam.fy), abs=1e-9)


def test_known_point_projects_to_expected_pixel():
    scn = _level_scene()
    cam = scn.camera
    T = camera_pose_in_grid(scn, 0.0)
    # 2 m ahead, 0.5 m to the right (grid -y), 0.25 m down -> optical (0.5, 0.25, 2.0)
    p = np.array([[CAM_XYZ[0] + 2.0, CAM_XYZ[1] - 0.5, CAM_XYZ[2] - 0.25]])
    u, v, z = project_points(T, cam, p)[0]
    assert z == pytest.approx(2.0)
    assert u == pytest.approx(cam.cx + cam.fx * 0.25)
    assert v == pytest.approx(cam.cy + cam.fy * 0.125)


def test_camera_mount_tilt_points_optical_axis_down():
    r = camera_mount(math.radians(30))[:3, :3]
    z_axis_in_body = r[:, 2]
    assert z_axis_in_body == pytest.approx([math.cos(math.radians(30)), 0.0, -math.sin(math.radians(30))])
    assert r[:, 0] == pytest.approx([0.0, -1.0, 0.0])  # optical x = body right


def test_uint16_depth_and_scale():
    scn = make_ideal(all_scenarios()["standing"])
    frame = next(e.payload for e in build_events(scn, 0) if e.kind == "frame")
    assert frame.depth.depth_raw.dtype == np.uint16
    assert frame.depth.depth_scale_m_per_unit == SIM_DEPTH_SCALE_M
    assert frame.depth.depth_raw.shape == (scn.camera.height, scn.camera.width)


def _frames(scn, seed=0):
    return [e.payload for e in build_events(scn, seed) if e.kind == "frame"]


def test_fully_occluded_survivor_is_not_detected():
    surv = (SurvivorSpec(1, (3.5, 3.5), "standing"),)
    hidden = make_ideal(make_scenario("h", "h", surv, walls=[wall("x", 2.0, 2.0, 5.0)]))
    frames = _frames(hidden)
    assert frames and all(f.truth_visible == () and f.detections == () for f in frames)
    shown = make_ideal(make_scenario("s", "s", surv))  # positive control
    assert all(len(f.detections) == 1 for f in _frames(shown))


def test_partially_occluded_survivor_is_detected():
    frames = _frames(make_ideal(all_scenarios()["occluded"]))
    assert all(f.truth_visible == (1,) for f in frames)


def test_seeds_reproduce_identical_output():
    scn = all_scenarios()["two_adjacent"]
    a, b, c = build_events(scn, 3), build_events(scn, 3), build_events(scn, 4)
    assert [(e.arrival_ns, e.kind) for e in a] == [(e.arrival_ns, e.kind) for e in b]
    fa, fb, fc = ([e.payload for e in ev if e.kind == "frame"] for ev in (a, b, c))
    for x, y in zip(fa, fb):
        assert np.array_equal(x.depth.depth_raw, y.depth.depth_raw)
        assert [(d.bbox_px, d.tracking_id, d.confidence) for d in x.detections] == \
               [(d.bbox_px, d.tracking_id, d.confidence) for d in y.detections]
    assert any(not np.array_equal(x.depth.depth_raw, z.depth.depth_raw) for x, z in zip(fa, fc))


def test_pose_latency_and_loop_closure():
    scn = all_scenarios()["latency_fast_yaw"]
    poses = [e for e in build_events(scn, 0) if e.kind == "pose"]
    assert all(e.arrival_ns - e.payload.stamp_ns == 100_000_000 for e in poses)
    lc = all_scenarios()["loop_closure"]
    poses = [e.payload for e in build_events(lc, 0) if e.kind == "pose"]
    assert {p.pose_revision for p in poses if p.stamp_ns < 1_500_000_000} == {0}
    assert {p.pose_revision for p in poses if p.stamp_ns >= 1_500_000_000} == {1}


def test_events_are_time_ordered_and_single_clock():
    ev = build_events(all_scenarios()["standing"], 1)
    assert [e.arrival_ns for e in ev] == sorted(e.arrival_ns for e in ev)
    for e in ev:
        if e.kind == "pose":
            assert e.payload.clock_domain == "sim"
        else:
            assert e.payload.depth.clock_domain == "sim"
            assert all(d.clock_domain == "sim" and d.image_stamp_ns == e.payload.stamp_ns
                       for d in e.payload.detections)


def test_truth_cell_uses_floor_and_registration():
    scn = all_scenarios()["cell_boundary"]
    (truth,) = scn.truths()
    assert truth.cell == (3, 3)
    assert scn.registration.grid_from_map(truth.centre_map_xy) == pytest.approx([3.0, 3.5])


def test_arena_size_limits():
    Arena(15)
    with pytest.raises(ValueError):
        Arena(16)


# --- substep 3.1 simulator additions -----------------------------------------------------------------------------

def test_pose_covariance_is_honest_by_construction():
    """The reported pose error must be distributed as the reported covariance (normalised error ~ chi2(2))."""
    from survivor_tagging.sim import _pose_events

    scn = all_scenarios()["revisit_after_gap"]
    nees = []
    for seed in range(300):
        rng = np.random.default_rng(seed)
        ev = _pose_events(scn, rng)
        for e in ev[::100]:
            p = e.payload
            var = p.covariance[0, 0]
            if var <= 0:
                continue
            true_pos, _ = scn.trajectory.true_pose(p.stamp_ns / 1e9, scn.duration_s)
            err = p.position_m[:2] - true_pos[:2]
            nees.append(float(err @ err) / var)
    assert np.mean(nees) == pytest.approx(2.0, rel=0.1)


def test_loop_closure_reduces_variance_and_error_consistently():
    from dataclasses import replace

    from survivor_tagging.sim import _pose_events

    base = all_scenarios()["revisit_after_gap"]
    scn = replace(base, trajectory=replace(base.trajectory, loop_closure_s=6.0, loop_closure_variance_factor=0.25))
    before, after, nees = [], [], []
    for seed in range(300):
        ev = _pose_events(scn, np.random.default_rng(seed))
        pre = [e.payload for e in ev if e.payload.stamp_ns < 6e9][-1]
        post = [e.payload for e in ev if e.payload.stamp_ns >= 6e9][0]
        before.append(pre.covariance[0, 0])
        after.append(post.covariance[0, 0])
        true_pos, _ = scn.trajectory.true_pose(post.stamp_ns / 1e9, scn.duration_s)
        err = post.position_m[:2] - true_pos[:2]
        nees.append(float(err @ err) / post.covariance[0, 0])
        assert post.pose_revision == 1
    assert np.mean(after) == pytest.approx(0.25 * np.mean(before), rel=0.05)
    assert np.mean(nees) == pytest.approx(2.0, rel=0.15)


def test_earlier_scenarios_have_no_pose_covariance():
    ev = build_events(all_scenarios()["standing"], 0)
    assert all(e.payload.covariance is None for e in ev if e.kind == "pose")


def test_id_merge_and_phantom_detections():
    frames = _frames(all_scenarios()["id_merge"])  # make_ideal would replace the merging detector
    for f in frames:
        ids = [d.tracking_id for d, sid in zip(f.detections, f.det_survivor_ids) if sid is not None]
        if len(ids) == 2:
            assert ids[0] == ids[1]
    assert any(sum(s is not None for s in f.det_survivor_ids) == 2 for f in frames)
    scn = all_scenarios()["intermittent_false_positive"]
    fr = [e.payload for s in range(5) for e in build_events(scn, s) if e.kind == "frame"]
    phantom = [d for f in fr for d, sid in zip(f.detections, f.det_survivor_ids) if sid is None]
    assert 0 < len(phantom) < len(fr)  # intermittent, never in every frame


# --- substep 3.2 simulator additions -----------------------------------------------------------------------------

import hashlib  # noqa: E402

from survivor_tagging.sim import place_survivors  # noqa: E402
from survivor_tagging.scenarios import (LAYER5_SCENARIOS, SIM_REGISTRATION, pre_32_scenarios,  # noqa: E402
                                        with_correction_mode, without_correction_events)

# Rounded digests of the pose and frame events, seed 1, computed with the code BEFORE substep 3.2 (dev machine).
PRE_32_DIGESTS = {"standing": "64bee97aaf4e1974", "latency_fast_yaw": "b54d3ad16f36b7fc",
                  "loop_closure": "994c44c3c8f70d98", "revisit_after_gap": "de804630033b36c0",
                  "opposite_views_boundary": "f6f00d41b9278bd5"}


def _digest(events):
    h = hashlib.sha256()
    for e in events:
        if e.kind not in ("pose", "frame"):
            continue
        p = e.payload
        if e.kind == "pose":
            v = (e.arrival_ns, p.stamp_ns, p.pose_revision, tuple(np.round(p.position_m, 9)),
                 tuple(np.round(p.orientation_wxyz, 9)),
                 None if p.covariance is None else tuple(np.round(np.diag(p.covariance), 12)))
            h.update(repr(v).encode())
        else:
            h.update(repr((e.arrival_ns, p.stamp_ns)).encode())
            h.update(np.ascontiguousarray(p.depth.depth_raw).tobytes())
            for d in p.detections:
                h.update(repr((tuple(round(x, 6) for x in d.bbox_px), round(d.confidence, 9), d.tracking_id)).encode())
    return h.hexdigest()[:16]


@pytest.mark.parametrize("name", sorted(PRE_32_DIGESTS))
def test_earlier_scenarios_reproduce_pre_32_streams(name):
    assert _digest(build_events(all_scenarios()[name], 1)) == PRE_32_DIGESTS[name]


def test_pre_32_scenario_set_is_unchanged():
    assert len(pre_32_scenarios()) == 20 and set(LAYER5_SCENARIOS) - set(pre_32_scenarios()) == {
        "loop_closure_late", "repeated_corrections", "partial_history", "registration_update", "frame_correction"}


def _split(events):
    return [(e.arrival_ns, e.kind) for e in events if e.kind in ("pose", "frame")], \
        [e for e in events if e.kind not in ("pose", "frame")]


def test_correction_events_only_add_events():
    for name in ("loop_closure", "repeated_corrections"):
        scn = all_scenarios()[name]
        a, ca = _split(build_events(scn, 2))
        b, cb = _split(build_events(without_correction_events(scn), 2))
        c, cc = _split(build_events(with_correction_mode(scn, "frame_correction"), 2))
        assert a == b == c and len(ca) == len(cc) == len(scn.trajectory.closures) and cb == []


def test_history_correction_is_near_truth_and_better_than_live():
    scn = place_survivors(all_scenarios()["repeated_corrections"], 0)
    ev = build_events(scn, 0)
    live = {e.payload.stamp_ns: e.payload for e in ev if e.kind == "pose"}
    corr = [e for e in ev if e.kind == "correction"]
    assert [c.payload.revision for c in corr] == [1, 2]
    assert [c.arrival_ns for c in corr] == [2_200_000_000, 4_200_000_000]
    for c in corr:
        assert c.payload.source == "history" and c.payload.stamp_ns == c.arrival_ns - 200_000_000
        assert c.payload.range_ns == (0, c.payload.stamp_ns - 20_000_000)
        live_err, corr_err = [], []
        for p in c.payload.poses:
            true_pos, _ = scn.trajectory.true_pose(p.stamp_ns / 1e9, scn.duration_s)
            assert p.pose_revision == c.payload.revision
            assert np.array_equal(p.orientation_wxyz, live[p.stamp_ns].orientation_wxyz)
            corr_err.append(np.linalg.norm(p.position_m - true_pos))
            live_err.append(np.linalg.norm(live[p.stamp_ns].position_m - true_pos))
        assert max(corr_err) < 4 * 0.02 * np.sqrt(2) and np.std(corr_err) < 1e-12  # constant residual (SIM)
        assert np.mean(corr_err) < np.mean(live_err)


def test_live_revision_and_drift_restart_at_each_closure():
    scn = all_scenarios()["repeated_corrections"]
    poses = [e.payload for e in build_events(scn, 0) if e.kind == "pose"]
    revs = {p.pose_revision for p in poses if p.stamp_ns < 2e9}, {p.pose_revision for p in poses if 2e9 <= p.stamp_ns < 4e9}, \
        {p.pose_revision for p in poses if p.stamp_ns >= 4e9}
    assert revs == ({0}, {1}, {2})
    at = {p.stamp_ns: p for p in poses}
    for t_s in (2.0, 4.0):
        true_pos, _ = scn.trajectory.true_pose(t_s, scn.duration_s)
        assert np.allclose(at[int(t_s * 1e9)].position_m, true_pos, atol=1e-12)  # error restarts from zero


def test_frame_correction_event_removes_the_error_at_the_closure():
    scn = place_survivors(all_scenarios()["frame_correction"], 0)
    ev = build_events(scn, 0)
    (c,) = [e.payload for e in ev if e.kind == "correction"]
    assert c.source == "frame_correction" and c.poses == () and np.allclose(c.T_new_old[:3, :3], np.eye(3))
    last = [e.payload for e in ev if e.kind == "pose" and e.payload.stamp_ns < c.stamp_ns][-1]
    true_pos, _ = scn.trajectory.true_pose(last.stamp_ns / 1e9, scn.duration_s)
    fixed = last.position_m + c.T_new_old[:3, 3]
    assert np.linalg.norm(fixed - true_pos) < 4 * 0.02 * np.sqrt(2)
    assert np.linalg.norm(last.position_m - true_pos) == pytest.approx(0.15 * 2.48, abs=1e-9)


def test_partial_history_and_registration_update_events():
    ev = build_events(all_scenarios()["partial_history"], 0)
    (c,) = [e.payload for e in ev if e.kind == "correction"]
    assert c.range_ns == (900_000_000, 1_480_000_000)
    scn = all_scenarios()["registration_update"]
    regs = [e for e in build_events(scn, 0) if e.kind == "registration"]
    assert len(regs) == 1 and regs[0].arrival_ns == 1_500_000_000 and regs[0].payload.version == 1
    assert regs[0].payload.registration == SIM_REGISTRATION == scn.registration != scn.initial_registration


# --- substep 3.3a simulator additions -----------------------------------------------------------------------------

from survivor_tagging.sim import _correction_events, _pose_events, _rngs  # noqa: E402


def _true_xy(scn, pose):
    return scn.trajectory.true_pose(pose.stamp_ns / 1e9, scn.duration_s)[0][:2]


def _nis_samples(name, seeds=range(300)):
    scn = all_scenarios()[name]
    live, corrected = [], []
    for seed in seeds:
        r = _rngs(scn, seed)
        poses = _pose_events(scn, r[3], r[6])
        for e in poses[::5]:
            p = e.payload
            if p.pose_revision >= 1 and p.covariance is not None and p.covariance[0, 0] > 0:
                err = p.position_m[:2] - _true_xy(scn, p)
                live.append(float(err @ err) / p.covariance[0, 0])
        for ce in _correction_events(scn, poses, r[5]):
            for p in ce.payload.poses[::10]:
                assert p.covariance is not None and p.covariance[0, 0] > 0
                err = p.position_m[:2] - _true_xy(scn, p)
                corrected.append(float(err @ err) / p.covariance[0, 0])
    return live, corrected


@pytest.mark.parametrize("name", ["repeated_corrections", "frame_correction", "loop_closure_late"])
def test_33a_corrected_and_post_correction_pose_covariance_is_honest(name):
    """Normalised error squared (2D) must average 2.0 (chi-squared, 2 DoF); the pre-registered window is [1, 3]."""
    live, corrected = _nis_samples(name)
    if name != "loop_closure_late":  # no drift there: live poses after the closure are exact and carry no covariance
        assert 1.0 <= np.mean(live) <= 3.0 and np.mean(live) == pytest.approx(2.0, rel=0.15)
    else:
        assert live == []
    if name != "frame_correction":  # a frame_correction event carries a transform, not poses
        assert 1.0 <= np.mean(corrected) <= 3.0 and np.mean(corrected) == pytest.approx(2.0, rel=0.15)


def test_33a_post_closure_error_restarts_from_zero_with_zero_variance():
    scn = all_scenarios()["repeated_corrections"]
    r = _rngs(scn, 3)
    poses = {e.payload.stamp_ns: e.payload for e in _pose_events(scn, r[3], r[6])}
    for t_s in (2.0, 4.0):
        p = poses[int(t_s * 1e9)]
        assert np.allclose(p.position_m[:2], _true_xy(scn, p), atol=1e-12) and p.covariance[0, 0] == 0.0
    before_first = [p for s, p in poses.items() if s < 2e9]
    assert all(p.covariance is None for p in before_first)  # scripted legacy drift before the first closure


def test_33a_scenarios_without_drift_keep_their_live_poses_and_get_covariance_on_corrected_history():
    scn = all_scenarios()["loop_closure"]
    ev = build_events(scn, 0)
    assert all(e.payload.covariance is None for e in ev if e.kind == "pose")
    (c,) = [e.payload for e in ev if e.kind == "correction"]
    assert all(p.covariance[0, 0] == pytest.approx(0.02 ** 2) for p in c.poses)


def test_33a_pose_stream_is_deterministic_and_drift_stream_is_separate():
    scn = all_scenarios()["repeated_corrections"]
    a, b = build_events(scn, 4), build_events(scn, 4)
    assert _digest(a) == _digest(b)
    assert len(_rngs(scn, 4)) == 7
    # the first six streams are unchanged by the added seventh
    ref = np.random.SeedSequence([4, __import__("zlib").crc32(scn.name.encode())]).spawn(6)
    assert [np.random.default_rng(s).random() for s in ref] == [g.random() for g in _rngs(scn, 4)[:6]]


def test_33a_double_detection_scenario():
    scn = all_scenarios()["double_detection"]
    n_frames = n_pairs = 0
    for seed in range(3):
        for e in build_events(scn, seed):
            if e.kind != "frame":
                continue
            f = e.payload
            on_person = [d for d, sid in zip(f.detections, f.det_survivor_ids) if sid == 1]
            n_frames += bool(on_person)
            if len(on_person) == 2:
                n_pairs += 1
                a, b = on_person
                assert a.tracking_id != b.tracking_id
                assert b.bbox_px[0] <= a.bbox_px[0] and b.bbox_px[2] >= a.bbox_px[2]  # the second box covers the first
                assert b.bbox_px[1] == a.bbox_px[1] and b.bbox_px[3] == a.bbox_px[3]
    assert n_frames > 20 and 0.2 < n_pairs / n_frames < 0.8
    assert _digest(build_events(scn, 1)) == _digest(build_events(scn, 1))


def test_33a_double_detection_is_not_in_the_pre_32_set_and_other_detectors_make_no_extra_draws():
    assert "double_detection" in all_scenarios() and "double_detection" not in pre_32_scenarios()
    assert all_scenarios()["standing"].detector.dup_prob == 0.0
