import math

import numpy as np
import pytest

from survivor_tagging.config import default_config
from survivor_tagging.footprint import FootprintParams, analyse_footprint, principal_axis, refine_observation
from survivor_tagging.geometry import FloorPlane, Layer1Params, estimate_survivor_point, subsample_points
from survivor_tagging.transforms import Extrinsic, floor_plane_in_camera, make_transform, rotvec_to_matrix
from survivor_tagging.types import (CameraIntrinsics, DepthMeasurement, Detection, FailureReason, PoseSample,
                                    SurvivorObservation)

R_LEVEL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # optical axes in the body frame


def _pose(z=1.5, rotvec=(0, 0, 0), cov_z=None):
    r = rotvec_to_matrix(rotvec)
    from survivor_tagging.transforms import matrix_to_quat
    cov = None
    if cov_z is not None:
        cov = np.zeros((6, 6))
        cov[2, 2] = cov_z
    return PoseSample(0, "ros", "map", "body", (0.0, 0.0, z), matrix_to_quat(r), True, 0, "t", cov)


EXTR = Extrinsic(make_transform(R_LEVEL, (0.0, 0.0, 0.0)), 1)


# --- floor plane ---------------------------------------------------------------------------------------------

def test_floor_plane_level_camera():
    f = floor_plane_in_camera(_pose(1.5), EXTR, 0.0, 0.06)
    assert f.normal == pytest.approx([0, -1, 0])  # map up = optical -y
    assert f.offset_m == pytest.approx(-1.5)
    assert f.heights(np.array([[0.3, 1.5, 4.0], [0.0, 0.0, 2.0]])) == pytest.approx([0.0, 1.5])
    f2 = floor_plane_in_camera(_pose(1.5), EXTR, 0.2, 0.06)
    assert f2.offset_m == pytest.approx(-1.3)


def test_floor_plane_pitched_and_rolled():
    a = math.radians(30)
    pitched = floor_plane_in_camera(_pose(1.5, (0, a, 0)), EXTR, 0.0, 0.06)  # nose down 30 degrees
    assert pitched.normal == pytest.approx([0.0, -math.cos(a), -math.sin(a)])
    # a point straight ahead at depth d is at height 1.5 - d sin(30)
    assert pitched.heights(np.array([0.0, 0.0, 2.0])) == pytest.approx(1.5 - 2.0 * math.sin(a))
    b = math.radians(20)
    rolled = floor_plane_in_camera(_pose(1.0, (b, 0, 0)), EXTR, 0.0, 0.06)
    assert rolled.normal == pytest.approx([-math.sin(b), -math.cos(b), 0.0])
    tilted_mount = Extrinsic(make_transform(rotvec_to_matrix((0, a, 0)) @ R_LEVEL, (0.1, 0.0, 0.05)), 1)
    m = floor_plane_in_camera(_pose(1.5), tilted_mount, 0.0, 0.06)
    assert m.normal == pytest.approx(pitched.normal) and m.offset_m == pytest.approx(-1.55)


def test_floor_margin_grows_with_pose_z_uncertainty():
    f = floor_plane_in_camera(_pose(1.5, cov_z=0.0004), EXTR, 0.0, 0.06, pose_sigma_k=2.0)
    assert f.margin_m == pytest.approx(0.06 + 2 * 0.02)


INTR = CameraIntrinsics(100.0, 100.0, 49.5, 39.5, 100, 80, "none")


def _params(**kw):
    cfg = default_config()
    vals = {"l1.depth_min_m": 0.1, "l1.depth_max_m": 10.0, "l1.depth_noise_model": [0.0, 0.0, 0.002]}
    vals.update(kw)
    cfg.apply_overrides(vals, "test")
    return Layer1Params.from_config(cfg)


def _run(img, floor, box=(20.0, 10.0, 80.0, 70.0), params=None):
    det = Detection(0, 0.9, box, 0, "ros", "cam", (100, 80))
    depth = DepthMeasurement(0, "ros", "cam", img, 1.0, INTR, True, (0.1, 10.0))
    return estimate_survivor_point(det, depth, INTR, params or _params(), floor=floor)


def test_points_just_above_and_below_the_margin():
    img = np.full((80, 100), 2.0)  # a plane at Z = 2: Y = (v - 39.5) * 0.02, height = -Y - offset
    for offset, kept in ((-(61 - 39.5) * 0.02 - 0.059, False), (-(61 - 39.5) * 0.02 - 0.061, True)):
        # offset chosen so that the last sampled row (v = 61) sits 0.059 m / 0.061 m above the floor
        f = FloorPlane(np.array([0.0, -1.0, 0.0]), offset, 0.06)
        r = _run(img, f)
        row61 = 22  # sampled columns 29..71 step 2 -> 22 per row
        assert (r.floor_removed_px >= row61) is (not kept)


def test_floor_removal_on_standing_person():
    """Level camera 1.5 m up; person (a plane at Z = 3) in columns 40-60, standing on the floor; floor elsewhere
    below the horizon, a far wall above it. Head and torso are kept, the floor is removed, and the person's lowest
    ~6 cm (the feet, within the margin) are removed too (documented behaviour)."""
    fy, cy = 100.0, 39.5
    img = np.full((80, 100), 6.0)
    for v in range(80):
        if v > cy:
            img[v, :] = 1.5 * fy / (v - cy)  # floor
        y_person = (v - cy) * 3.0 / fy
        if y_person <= 1.5:
            img[v, 40:61] = 3.0
    f = floor_plane_in_camera(_pose(1.5), EXTR, 0.0, 0.06)
    without = _run(img, None, box=(30.0, 0.0, 70.0, 79.0))
    with_floor = _run(img, f, box=(30.0, 0.0, 70.0, 79.0))
    assert with_floor.failure is None and with_floor.floor_removed_px > 0
    assert with_floor.depth_m == pytest.approx(3.0)
    assert with_floor.p_camera_m[1] < 1.5 - 0.06  # the estimate is above the floor band
    assert without.floor_removed_px == 0


def test_floor_unknown_warning_when_floor_height_missing():
    from test_pipeline import INTR as PINTR, _cfg, _frame, _pose as ppose
    from survivor_tagging.pipeline import Layer123Pipeline

    S = 1_000_000_000
    p = Layer123Pipeline(_cfg(**{"l1.floor_removal": True}))  # map.floor_z_m unset
    p.process_pose(ppose(0.98), 0)
    p.process_pose(ppose(1.02), 0)
    dets, depth = _frame(1.0)
    p.process_frame(dets, depth, PINTR, int(1.05 * S))
    (obs,) = p.observations()
    assert FailureReason.FLOOR_UNKNOWN in obs.warnings and obs.failure is None
    p2 = Layer123Pipeline(_cfg(**{"l1.floor_removal": True, "map.floor_z_m": 0.0}))
    p2.process_pose(ppose(0.98), 0)
    p2.process_pose(ppose(1.02), 0)
    p2.process_frame(dets, depth, PINTR, int(1.05 * S))
    assert FailureReason.FLOOR_UNKNOWN not in p2.observations()[0].warnings


def test_missing_pose_fails_explicitly_with_floor_removal_on():
    from test_pipeline import INTR as PINTR, _cfg, _frame, _pose as ppose
    from survivor_tagging.pipeline import Layer123Pipeline

    S = 1_000_000_000
    p = Layer123Pipeline(_cfg(**{"l1.floor_removal": True, "map.floor_z_m": 0.0, "l2.max_pose_wait_s": 0.2}))
    p.process_pose(ppose(0.90), 0)  # only an older pose: must not be used
    dets, depth = _frame(1.0)
    p.process_frame(dets, depth, PINTR, int(1.05 * S))
    p.process_pose(ppose(0.95), int(1.4 * S))
    (obs,) = p.observations()
    assert obs.failure is FailureReason.POSE_UNAVAILABLE_AT_CAPTURE and obs.p_map_m is None


# --- footprint -----------------------------------------------------------------------------------------------

FP = FootprintParams("midpoint", 5.0, 95.0, 0.9, 0.25, 0.0)


def _line(angle_deg, length=1.7, n=300, centre=(3.0, 2.0), seed=0):
    rng = np.random.default_rng(seed)
    a = math.radians(angle_deg)
    s = np.linspace(-length / 2, length / 2, n)
    xy = np.stack([centre[0] + s * math.cos(a), centre[1] + s * math.sin(a)], axis=1)
    return xy + rng.normal(0, 0.01, xy.shape)


def test_principal_axis_and_midpoint_of_a_line():
    xy = _line(30)
    _, axis, _ = principal_axis(xy)
    assert math.degrees(math.atan2(axis[1], axis[0])) == pytest.approx(30, abs=1.0)
    fp = analyse_footprint(xy, FP)
    assert fp.extended and fp.extent_along_m == pytest.approx(0.9 * 1.7, abs=0.03)
    assert fp.centre == pytest.approx([3.0, 2.0], abs=0.02)


def test_extent_percentiles_reject_outliers():
    xy = np.vstack([_line(0), [[9.0, 2.0]] * 5])  # 5 wild points (< 5 %)
    fp = analyse_footprint(xy, FP)
    assert fp.extent_along_m == pytest.approx(0.9 * 1.7, abs=0.05) and fp.centre[0] == pytest.approx(3.0, abs=0.05)


def _obs_from_map_points(xy, t_mc, z=0.2):
    pm = np.column_stack([xy, np.full(len(xy), z)])
    pc = (pm - t_mc[:3, 3]) @ t_mc[:3, :3]  # inverse transform
    o = SurvivorObservation(0, 0, 0.9, 0, cluster_points_camera_m=pc,
                            p_map_m=np.array([xy[:, 0].mean(), xy[:, 1].mean(), z]), cov_map_m2=np.eye(3) * 0.01)
    return o


T_MC = make_transform(R_LEVEL, (0.0, 2.0, 1.5))  # level camera at map (0, 2, 1.5) looking along +x


def test_refine_extended_inflates_covariance_along_axis_and_flags():
    xy = _line(90, centre=(3.0, 2.0))  # body lying across the view (along map y)
    o = _obs_from_map_points(xy, T_MC)
    cov0 = o.cov_map_m2.copy()
    assert refine_observation(o, T_MC, FP)
    assert FailureReason.EXTENDED_CLUSTER in o.warnings
    d = o.cov_map_m2[:2, :2] - cov0[:2, :2]
    ext = 0.9 * 1.7
    assert d == pytest.approx((0.25 * ext) ** 2 * np.array([[0, 0], [0, 1]]), abs=2e-3)  # along map y only
    assert o.p_map_m[:2] == pytest.approx([3.0, 2.0], abs=0.02)


def test_body_offset_applied_across_the_axis_only():
    off = FootprintParams("midpoint", 5.0, 95.0, 0.9, 0.25, 0.2)
    side = _obs_from_map_points(_line(90, centre=(3.0, 2.0)), T_MC)  # seen from the side: shift 0.2 away
    refine_observation(side, T_MC, off)
    assert side.p_map_m[:2] == pytest.approx([3.2, 2.0], abs=0.02)
    end_on = _obs_from_map_points(_line(0, centre=(3.0, 2.0)), T_MC)  # seen end-on: no across shift
    refine_observation(end_on, T_MC, off)
    assert end_on.p_map_m[1] == pytest.approx(2.0, abs=0.02)


def test_non_extended_cluster_is_bit_identical():
    o = _obs_from_map_points(_line(30, length=0.4), T_MC)
    p0, c0, w0 = o.p_map_m.copy(), o.cov_map_m2.copy(), list(o.warnings)
    assert not refine_observation(o, T_MC, FP)
    assert np.array_equal(o.p_map_m, p0) and np.array_equal(o.cov_map_m2, c0) and o.warnings == w0


def test_deterministic_subsample_and_kept_points():
    pts = np.arange(3000, dtype=float).reshape(1000, 3)
    a, b = subsample_points(pts, 300), subsample_points(pts, 300)
    assert len(a) == 300 and np.array_equal(a, b) and np.all(np.diff(a[:, 0]) > 0)
    small = subsample_points(pts[:10], 300)
    assert np.array_equal(small, pts[:10])
    r = _run(np.full((80, 100), 2.0), None, params=_params(**{"l1b.footprint_refinement": True, "l1b.max_points": 50}))
    assert r.cluster_points_camera_m.shape == (50, 3)
    assert _run(np.full((80, 100), 2.0), None).cluster_points_camera_m is None  # off by default


def test_area_weighted_variant_runs():
    aw = FootprintParams("area_weighted", 5.0, 95.0, 0.9, 0.25, 0.0)
    xy = _line(0)
    fp = analyse_footprint(xy, aw, weights=np.ones(len(xy)))
    assert fp.extended and fp.centre[0] == pytest.approx(3.0, abs=0.02)
