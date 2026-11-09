import math

import numpy as np
import pytest

from survivor_tagging.config import default_config
from survivor_tagging.transforms import (
    Extrinsic,
    Layer2,
    Layer2Params,
    PendingItem,
    PendingQueue,
    PoseBuffer,
    apply_transform,
    camera_to_map,
    capture_in_pose_clock,
    check_timestamps,
    compose,
    extrinsic_from_config,
    invert_transform,
    make_transform,
    matrix_to_quat,
    quat_multiply,
    quat_slerp,
    quat_to_matrix,
    rotvec_to_matrix,
    skew,
)
from survivor_tagging.types import CameraIntrinsics, DepthMeasurement, Detection, FailureReason, PoseSample

S = 1_000_000_000


def qaxis(axis, deg):
    a = math.radians(deg) / 2
    v = np.zeros(3)
    v["xyz".index(axis)] = math.sin(a)
    return np.array([math.cos(a), *v])


def pose(t_s, xyz=(0, 0, 0), q=(1, 0, 0, 0), valid=True, rev=0, domain="ros", cov=None):
    return PoseSample(int(round(t_s * S)), domain, "map", "body", xyz, q, valid, rev, "test", cov)


# --- quaternions and transforms ------------------------------------------------

def test_basic_rotations_match_hand_calculation():
    assert quat_to_matrix(qaxis("z", 90)) @ [1, 0, 0] == pytest.approx([0, 1, 0])  # yaw 90
    assert quat_to_matrix(qaxis("x", 90)) @ [0, 1, 0] == pytest.approx([0, 0, 1])  # roll 90
    assert quat_to_matrix(qaxis("y", 90)) @ [0, 0, 1] == pytest.approx([1, 0, 0])  # pitch 90


def test_quat_multiply_matches_matrix_product():
    a, b = qaxis("z", 90), qaxis("x", 90)
    assert quat_to_matrix(quat_multiply(a, b)) == pytest.approx(quat_to_matrix(a) @ quat_to_matrix(b))


def test_matrix_quat_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q = q if q[0] >= 0 else -q
        assert matrix_to_quat(quat_to_matrix(q)) == pytest.approx(q, abs=1e-9)


def test_slerp_midpoint_and_alphas():
    q0, q1 = np.array([1.0, 0, 0, 0]), qaxis("z", 90)
    for alpha, deg in ((0.0, 0), (0.25, 22.5), (0.5, 45), (0.75, 67.5), (1.0, 90)):
        assert quat_slerp(q0, q1, alpha) == pytest.approx(qaxis("z", deg))
    assert quat_slerp(q0, -q1, 0.5) == pytest.approx(qaxis("z", 45))  # shortest arc despite sign


def test_compose_invert_apply():
    t = make_transform(quat_to_matrix(qaxis("z", 90)), (1, 2, 3))
    assert apply_transform(t, [1, 0, 0]) == pytest.approx([1, 3, 3])
    assert compose(t, invert_transform(t)) == pytest.approx(np.eye(4))
    assert apply_transform(invert_transform(t), [1, 3, 3]) == pytest.approx([1, 0, 0])


# Forward camera (optical z = body x, optical x = -body y, optical y = -body z), 0.1 m ahead of the body origin.
R_BC = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
EXTR = Extrinsic(make_transform(R_BC, (0.1, 0.0, 0.0)), 3)


def test_camera_to_map_with_offset_and_yaw():
    p, _ = camera_to_map(np.array([0.0, 0.0, 2.0]), None, pose(0, (0, 0, 1), qaxis("z", 90)), EXTR)
    assert p == pytest.approx([0.0, 2.1, 1.0])  # 2 m ahead + 0.1 m mount, rotated to map +y, at height 1
    p, _ = camera_to_map(np.array([0.5, 0.0, 2.0]), None, pose(0), EXTR)  # 0.5 m right
    assert p == pytest.approx([2.1, -0.5, 0.0])
    # pitched down 90 degrees: forward camera now looks at the floor (map -z)
    p, _ = camera_to_map(np.array([0.0, 0.0, 1.0]), None, pose(0, (0, 0, 2), qaxis("y", 90)), EXTR)
    assert p == pytest.approx([0.0, 0.0, 2.0 - 1.1])


# --- covariance Jacobians vs finite differences ----------------------------------

def _num_jac(f, x0, eps=1e-6):
    x0 = np.asarray(x0, dtype=float)
    cols = []
    for i in range(x0.size):
        d = np.zeros_like(x0)
        d[i] = eps
        cols.append((f(x0 + d) - f(x0 - d)) / (2 * eps))
    return np.stack(cols, axis=1)


def test_covariance_jacobians_match_finite_differences():
    rng = np.random.default_rng(1)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    pz = pose(0, rng.normal(size=3), q)
    r_bc = rotvec_to_matrix(rng.normal(size=3) * 0.5)
    t_bc = rng.normal(size=3) * 0.2
    p_c = np.array([0.3, -0.2, 2.5])
    r_mb, t_mb = quat_to_matrix(q), pz.position_m

    def f_pose(d):  # map-frame perturbation of the pose
        return rotvec_to_matrix(d[3:]) @ r_mb @ (r_bc @ p_c + t_bc) + t_mb + d[:3]

    def f_ext(d):  # body-frame perturbation of the extrinsic
        return r_mb @ (rotvec_to_matrix(d[3:]) @ r_bc @ p_c + t_bc + d[:3]) + t_mb

    def f_pc(x):
        return r_mb @ (r_bc @ x + t_bc) + t_mb

    j_pose, j_ext, j_pc = _num_jac(f_pose, np.zeros(6)), _num_jac(f_ext, np.zeros(6)), _num_jac(f_pc, p_c)

    def spd(n):
        a = rng.normal(size=(n, n)) * 0.05
        return a @ a.T

    c_pose, c_ext, c_cam = spd(6), spd(6), spd(3)
    expected = j_pc @ c_cam @ j_pc.T + j_pose @ c_pose @ j_pose.T + j_ext @ c_ext @ j_ext.T
    pz_cov = PoseSample(0, "ros", "map", "body", t_mb, q, True, 0, "t", c_pose)
    p, cov = camera_to_map(p_c, c_cam, pz_cov, Extrinsic(make_transform(r_bc, t_bc), 1, c_ext))
    assert p == pytest.approx(f_pc(p_c))
    assert cov == pytest.approx(expected, abs=1e-9)
    assert skew([1, 2, 3]) @ [4, 5, 6] == pytest.approx(np.cross([1, 2, 3], [4, 5, 6]))


# --- extrinsic from config --------------------------------------------------------

def test_extrinsic_from_config():
    cfg = default_config()
    assert extrinsic_from_config(cfg) == (None, FailureReason.EXTRINSIC_MISSING)
    cfg.set("extr.T_body_camera", np.eye(4).tolist())
    cfg.set("extr.version", 2)
    e, f = extrinsic_from_config(cfg)
    assert f is None and e.version == 2 and e.cov6 is None
    cfg["extr.T_body_camera"].value = np.diag([2.0, 1.0, 1.0, 1.0])  # bypasses load-time validation on purpose
    assert extrinsic_from_config(cfg) == (None, FailureReason.CALIBRATION_INVALID)


# --- pose buffer ---------------------------------------------------------------------

def test_out_of_order_insertion_and_interpolation():
    b = PoseBuffer(10.0, 100)
    for t in (0.3, 0.1, 0.2):
        assert b.add(pose(t, (t * 10, 0, 0), quat_slerp([1, 0, 0, 0], qaxis("z", 90), t / 0.3))) is None
    assert [round(x / S, 1) for x in b._stamps] == [0.1, 0.2, 0.3]
    lk = b.lookup(int(0.15 * S), 0.2)
    assert lk.failure is None and lk.alpha == pytest.approx(0.5) and lk.stamps_ns == (int(0.1 * S), int(0.2 * S))
    assert lk.pose.position_m == pytest.approx([1.5, 0, 0])
    assert lk.pose.orientation_wxyz == pytest.approx(qaxis("z", 45))
    for alpha in (0.25, 0.75):
        lk = b.lookup(int((0.2 + 0.1 * alpha) * S), 0.2)
        assert lk.alpha == pytest.approx(alpha) and lk.pose.position_m[0] == pytest.approx(2 + alpha)
    assert b.lookup(int(0.2 * S), 0.2).alpha == 0.0  # exact stamp


def test_buffer_bounds_and_invalid_pose():
    b = PoseBuffer(1.0, 100)
    for i in range(31):
        b.add(pose(i * 0.1))
    assert b.oldest_ns == int(round(2.0 * S)) and b.newest_ns == int(round(3.0 * S))
    b2 = PoseBuffer(100.0, 3)
    for i in range(10):
        b2.add(pose(i * 0.1))
    assert len(b2) == 3 and b2.oldest_ns == int(round(0.7 * S))
    assert b.add(pose(5.0, valid=False)) is FailureReason.POSE_INVALID
    assert b.add(pose(5.0, domain="other")) is FailureReason.CLOCK_DOMAIN_UNKNOWN


def test_gap_too_large_extrapolation_refused_and_revisions():
    b = PoseBuffer(10.0, 100)
    b.add(pose(0.0))
    b.add(pose(1.0, (1, 0, 0)))
    assert b.lookup(int(0.5 * S), 0.1).failure is FailureReason.POSE_GAP_TOO_LARGE
    assert b.lookup(int(1.05 * S), 2.0).failure is FailureReason.POSE_UNAVAILABLE_AT_CAPTURE  # never latest pose
    assert b.lookup(int(-0.1 * S), 2.0).failure is FailureReason.POSE_UNAVAILABLE_AT_CAPTURE
    lk = b.lookup(int(1.05 * S), 2.0, allow_extrapolation_s=0.1)
    assert lk.failure is None and lk.pose.position_m[0] == pytest.approx(1.05) and lk.alpha == pytest.approx(1.05)
    b.add(pose(1.1, rev=1))
    assert FailureReason.SLAM_CORRECTION_APPLIED in b.lookup(int(1.05 * S), 2.0).warnings


# --- pending queue and timestamps ----------------------------------------------------------

def test_pending_queue_ready_expired_bounded():
    q = PendingQueue(2)
    assert q.push(PendingItem(10, 100, "a")) is None
    assert q.push(PendingItem(20, 200, "b")) is None
    assert q.push(PendingItem(30, 300, "c")).payload == "a"  # evicted
    assert [i.payload for i in q.pop_ready(25)] == ["b"]
    assert [i.payload for i in q.pop_expired(301)] == ["c"]
    assert len(q) == 0


def _params(**kw):
    vals = dict(pose_buffer_duration_s=10.0, pose_buffer_max_samples=100, max_interp_gap_s=0.1,
                allow_extrapolation_s=0.0, max_detection_age_s=1.0, max_rgb_depth_skew_s=0.01, max_pose_wait_s=0.5,
                pending_queue_max=8, camera_to_pose_offset_ns=None)
    vals.update(kw)
    return Layer2Params(**vals)


INTR = CameraIntrinsics(100.0, 100.0, 49.5, 39.5, 100, 80, "none")


def _det_depth(t_s=1.0, depth_t_s=None, domain="ros"):
    det = Detection(0, 0.9, (20.0, 10.0, 80.0, 70.0), int(t_s * S), domain, "cam", (100, 80), 7)
    depth = DepthMeasurement(int((depth_t_s if depth_t_s is not None else t_s) * S), domain, "cam",
                             np.full((80, 100), 2000, dtype=np.uint16), 0.001, INTR, True, (0.1, 10.0))
    return det, depth


def test_timestamp_checks():
    det, depth = _det_depth()
    assert check_timestamps(det, depth, int(1.1 * S), _params(), "ros") == (int(1.0 * S), None)
    assert check_timestamps(det, depth, int(1.1 * S), _params(max_rgb_depth_skew_s=None), "ros")[1] is \
        FailureReason.CONFIG_MISSING
    det2, depth2 = _det_depth(1.0, 1.05)
    assert check_timestamps(det2, depth2, int(1.1 * S), _params(), "ros")[1] is FailureReason.RGB_DEPTH_TIME_MISMATCH
    assert check_timestamps(det, depth, int(2.5 * S), _params(), "ros")[1] is FailureReason.DETECTION_STALE
    det3, depth3 = _det_depth(domain="camera_hw")
    assert check_timestamps(det3, depth3, int(1.1 * S), _params(), "ros")[1] is FailureReason.CLOCK_DOMAIN_UNKNOWN
    assert check_timestamps(det3, depth3, int(1.1 * S), _params(camera_to_pose_offset_ns=5), "ros") == \
        (int(1.0 * S) + 5, None)
    assert capture_in_pose_clock(10, "a", None, None) == (10, None)


def test_layer2_waits_for_pose_then_transforms():
    l2 = Layer2(_params(), EXTR)
    l2.add_pose(pose(0.98))
    assert l2.waiting_for_pose(int(1.0 * S))
    l2.add_pose(pose(1.02, (0.4, 0, 0)))
    assert not l2.waiting_for_pose(int(1.0 * S))
    from survivor_tagging.types import SurvivorObservation
    obs = SurvivorObservation(0, 0, 0.9, int(1.0 * S), p_camera_m=np.array([0.0, 0.0, 2.0]))
    assert l2.transform(obs, int(1.0 * S)) is None
    assert obs.p_map_m == pytest.approx([2.3, 0, 0]) and obs.interp_alpha == pytest.approx(0.5)
    assert obs.extrinsic_version == 3 and obs.pose_revision == 0
    missing = Layer2(_params(), None)
    obs2 = SurvivorObservation(1, 0, 0.9, 0, p_camera_m=np.zeros(3))
    assert missing.transform(obs2, 0) is FailureReason.EXTRINSIC_MISSING
