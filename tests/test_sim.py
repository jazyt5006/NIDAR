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
