"""Substep 4.1a: the ROS-free adapters, tested with fake (duck-typed) messages on a machine without ROS.
Evidence level: unit. Nothing here proves the real ROS messages or nodes work (no ROS on this machine)."""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

import ros_fakes as rf
from ros_fakes import adapters as A
from survivor_tagging.corrections import validate_correction
from survivor_tagging.grid import GridRegistration
from survivor_tagging.output import (Snapshot, SnapshotFlags, TagFlags, TagRecord, encode_snapshot, encode_tag_record,
                                     TelemetryDecodeError)
from survivor_tagging.pipeline import tag_result
from survivor_tagging.transforms import quat_to_matrix
from survivor_tagging.types import FailureReason, Tag, TagResult, TagStatus

F = FailureReason


def cfg(**kw):
    base = dict(map_frame_id="map", body_frame_id="base_link", body_convention="FLU", clock_domain="ros_time",
                depth_aligned_to_color=True, depth_min_m=0.1, depth_max_m=10.0)
    base.update(kw)
    return A.AdapterConfig(**base)


def reason(excinfo):
    return excinfo.value.reason


# --------------------------------------------------------------------------- stamps
@pytest.mark.parametrize("ns", [1, 999_999_999, 1_000_000_000, 1_700_000_123_456_789_012, 2**62])
def test_stamp_round_trip(ns):
    s = A.ns_to_stamp(ns)
    assert 0 <= s["nanosec"] < 10**9
    assert A.stamp_to_ns(NS(**s)) == ns


@pytest.mark.parametrize("stamp,why", [
    (NS(sec=0, nanosec=0), "zero"), (NS(sec=-1, nanosec=5), "negative"), (NS(sec=1, nanosec=10**9), "nanosec"),
    (NS(sec=1, nanosec=-1), "nanosec"), (NS(sec=1.5, nanosec=0), "integers"), (NS(sec=True, nanosec=0), "integers"),
    (NS(), "sec"),
])
def test_bad_stamps_are_rejected_with_a_reason(stamp, why):
    with pytest.raises(A.AdapterError) as e:
        A.stamp_to_ns(stamp)
    assert reason(e) is F.TIMESTAMP_INVALID and why in e.value.detail


def test_stamp_monitor_rejects_backwards_and_jumps_but_not_repeats():
    m = A.StampMonitor(max_backward_s=0.0, max_jump_s=5.0)
    assert m.check(10_000_000_000) == 10_000_000_000
    assert m.check(10_000_000_000) == 10_000_000_000  # same stamp (two messages of one frame) is fine
    with pytest.raises(A.AdapterError) as e:
        m.check(9_999_999_999)
    assert reason(e) is F.TIMESTAMP_INVALID
    with pytest.raises(A.AdapterError):
        m.check(20_000_000_000)  # 10 s forward jump
    assert A.StampMonitor(max_backward_s=0.1).check(5) == 5
    tol = A.StampMonitor(max_backward_s=0.1)
    tol.check(1_000_000_000)
    tol.check(950_000_000)  # 50 ms backwards within the tolerance
    with pytest.raises(A.AdapterError):
        tol.check(800_000_000)


def test_ns_to_stamp_rejects_negative():
    with pytest.raises(A.AdapterError):
        A.ns_to_stamp(-1)


# --------------------------------------------------------------------------- depth
def test_depth_16uc1_millimetres_zero_invalid():
    arr = np.array([[0, 1000, 1500], [65535, 2, 300]], dtype=np.uint16)
    m = A.depth_image_to_metres(rf.depth_msg(arr, "16UC1"))
    assert m.shape == (2, 3) and m.dtype == np.float64
    assert np.isnan(m[0, 0]) and m[0, 1] == pytest.approx(1.0) and m[0, 2] == pytest.approx(1.5)
    assert m[1, 0] == pytest.approx(65.535) and m[1, 1] == pytest.approx(0.002)
    raw, scale = A.depth_image_to_raw(rf.depth_msg(arr, "16UC1"))
    assert raw.dtype == np.uint16 and scale == 0.001 and raw[0, 0] == 0


def test_depth_32fc1_metres_nan_inf_nonpositive_invalid():
    arr = np.array([[np.nan, np.inf, -np.inf], [0.0, -1.0, 2.5]], dtype=np.float32)
    msg = rf.depth_msg(arr, "32FC1")
    m = A.depth_image_to_metres(msg)
    assert np.isnan(m[:, :2]).all() and np.isnan(m[0, 2]) and np.isnan(m[1, 0]) and np.isnan(m[1, 1])
    assert m[1, 2] == pytest.approx(2.5)
    raw, scale = A.depth_image_to_raw(msg)
    assert scale == 1.0 and raw.dtype == np.float32 and raw[1, 2] == np.float32(2.5) and raw[:, :2].sum() == 0


def test_depth_row_stride_and_big_endian():
    arr = (np.arange(12, dtype=np.uint16).reshape(3, 4) + 1) * 100
    for pad in (0, 6):
        for big in (False, True):
            m = A.depth_image_to_metres(rf.depth_msg(arr, "16UC1", step_pad=pad, big=big))
            assert np.allclose(m, arr * 0.001)
    f = np.linspace(0.5, 3.0, 12, dtype=np.float32).reshape(3, 4)
    assert np.allclose(A.depth_image_to_metres(rf.depth_msg(f, "32FC1", step_pad=8, big=True)), f)


def test_depth_shape_and_format_checks():
    arr = np.ones((3, 4), dtype=np.uint16)
    ok = rf.depth_msg(arr, "16UC1")
    bad_len = NS(**{**ok.__dict__, "data": ok.data[:-2]})
    bad_step = NS(**{**ok.__dict__, "step": 6})
    bad_enc = NS(**{**ok.__dict__, "encoding": "mono8"})
    zero_h = NS(**{**ok.__dict__, "height": 0})
    for m in (bad_len, bad_step, bad_enc, zero_h):
        with pytest.raises(A.AdapterError) as e:
            A.depth_image_to_metres(m)
        assert reason(e) is F.IMAGE_FORMAT_INVALID


def test_depth_measurement_checks_size_against_camera_info_and_needs_config():
    arr = np.full((3, 4), 2000, dtype=np.uint16)
    d = A.depth_measurement_from_msgs(rf.depth_msg(arr, "16UC1"), rf.camera_info(4, 3), cfg())
    assert d.depth_raw.shape == (3, 4) and d.depth_scale_m_per_unit == 0.001 and d.aligned_to_color
    assert d.intrinsics.fx_px == 100.0 and d.intrinsics.cy_px == 1.5 and d.depth_stamp_ns == 1_000_000_000
    assert d.frame_id == "cam_color_optical" and d.valid_range_m == (0.1, 10.0) and d.clock_domain == "ros_time"
    with pytest.raises(A.AdapterError) as e:
        A.depth_measurement_from_msgs(rf.depth_msg(arr, "16UC1"), rf.camera_info(8, 6), cfg())
    assert reason(e) is F.INTRINSICS_MISSING


def test_missing_parameters_raise_configuration_errors_never_defaults():
    arr = np.full((3, 4), 2000, dtype=np.uint16)
    with pytest.raises(A.ConfigurationError) as e:
        A.depth_measurement_from_msgs(rf.depth_msg(arr, "16UC1"), rf.camera_info(4, 3), cfg(depth_aligned_to_color=None))
    assert "depth_aligned_to_color" in str(e.value) and e.value.reason is F.CONFIG_MISSING
    with pytest.raises(A.ConfigurationError) as e:
        A.pose_from_msg(rf.pose_msg(), A.AdapterConfig())
    assert {"map_frame_id", "body_frame_id", "body_convention", "clock_domain"} <= set(e.value.names)
    with pytest.raises(A.ConfigurationError):
        A.pose_from_msg(rf.pose_msg(), cfg(body_convention="NED"))  # not guessed, not accepted
    with pytest.raises(A.ConfigurationError):
        A.correction_from_msg(NS(), cfg(map_frame_id=None))
    with pytest.raises(A.ConfigurationError):
        A.scale_detection_to_depth(_det(), 640, 480, cfg(depth_aligned_to_color=None))


def test_intrinsics_from_camera_info():
    i = A.intrinsics_from_camera_info(rf.camera_info(640, 480, 600.0, 601.0, 320.0, 240.0))
    assert (i.fx_px, i.fy_px, i.cx_px, i.cy_px, i.width_px, i.height_px) == (600.0, 601.0, 320.0, 240.0, 640, 480)
    assert i.distortion_model == "plumb_bob" and len(i.distortion_coeffs) == 5
    for k in ([0.0] * 9, [600, 0.5, 320, 0, 600, 240, 0, 0, 1], [0, 0, 320, 0, 600, 240, 0, 0, 1]):
        with pytest.raises(A.AdapterError) as e:
            A.intrinsics_from_camera_info(NS(k=k, width=640, height=480, distortion_model="", d=[]))
        assert reason(e) is F.INTRINSICS_MISSING
    with pytest.raises(A.AdapterError):
        A.intrinsics_from_camera_info(NS(k=[1.0] * 3, width=1, height=1))


# --------------------------------------------------------------------------- detections
def _detmsg(box=(10, 20, 110, 220), conf=0.8, tid=-1, w=640, h=480, name="person", idx=7, cid=0):
    return NS(frame_index=idx, bbox_xyxy=list(box), confidence=conf, class_id=cid, class_name=name, tracking_id=tid,
              source_image_width=w, source_image_height=h)


def _det(**kw):
    return A.detection_from_msg(_detmsg(**kw), 0, 5, "cam", cfg())


def test_detection_message_to_library_detection():
    d = A.detection_from_msg(_detmsg(tid=-1), 3, 123, "cam", cfg())
    assert d.detection_id == 3 and d.tracking_id is None and d.confidence == pytest.approx(0.8)
    assert d.bbox_px == (10.0, 20.0, 110.0, 220.0) and d.image_stamp_ns == 123 and d.image_frame_id == "cam"
    assert d.image_size_px == (640, 480) and d.clock_domain == "ros_time" and d.class_label == "person"
    assert A.detection_from_msg(_detmsg(tid=12), 0, 1, "cam", cfg()).tracking_id == 12
    assert A.detection_from_msg(_detmsg(name=""), 0, 1, "cam", cfg()).class_label is None


def test_detection_validation_reasons():
    for kw, why in (({"box": (10, 20, 5, 220)}, F.BBOX_INVALID), ({"box": (10, 20, 700, 220)}, F.BBOX_OUTSIDE_IMAGE),
                    ({"box": (float("nan"), 0, 5, 5)}, F.BBOX_INVALID), ({"conf": 1.5}, F.BBOX_INVALID),
                    ({"w": 0}, F.IMAGE_FORMAT_INVALID)):
        with pytest.raises(A.AdapterError) as e:
            _det(**kw)
        assert reason(e) is why


def test_detection_array_keeps_good_detections_and_reports_rejected_ones():
    arr = NS(header=rf.header(3_000_000_000, "cam_color_optical"),
             detections=[_detmsg(), _detmsg(box=(5, 5, 1, 1)), _detmsg(box=(100, 100, 200, 300), tid=4)])
    b = A.detections_from_array(arr, cfg())
    assert b.stamp_ns == 3_000_000_000 and b.frame_id == "cam_color_optical"
    assert [d.detection_id for d in b.detections] == [0, 2]
    assert [(i, r) for i, r, _ in b.rejected] == [(1, F.BBOX_INVALID)]
    for bad in (NS(header=rf.header(0), detections=[]), NS(header=rf.header(5, ""), detections=[])):
        with pytest.raises(A.AdapterError):
            A.detections_from_array(bad, cfg())
    assert A.detections_from_array(NS(header=rf.header(7, "c"), detections=[]), cfg()).detections == []


def test_detection_scaling_rule():
    d = _det(box=(100, 50, 300, 250), w=640, h=480)
    assert A.scale_detection_to_depth(d, 640, 480, cfg(depth_aligned_to_color=False)) is d  # equal size: no flag needed
    s = A.scale_detection_to_depth(d, 320, 240, cfg())
    assert s.bbox_px == (50.0, 25.0, 150.0, 125.0) and s.image_size_px == (320, 240)
    with pytest.raises(A.AdapterError) as e:  # not aligned: different size means different sensor geometry
        A.scale_detection_to_depth(d, 320, 240, cfg(depth_aligned_to_color=False))
    assert reason(e) is F.DEPTH_FRAME_MISMATCH
    with pytest.raises(A.AdapterError) as e:  # other aspect ratio: a crop, not a scaling
        A.scale_detection_to_depth(d, 424, 240, cfg())
    assert reason(e) is F.DEPTH_FRAME_MISMATCH
    arr = NS(header=rf.header(3_000_000_000, "c"), detections=[_detmsg(box=(100, 50, 300, 250))])
    assert A.detections_from_array(arr, cfg(), depth_size=(320, 240)).detections[0].image_size_px == (320, 240)


# --------------------------------------------------------------------------- poses
def test_frd_to_flu_hand_computed_quaternions():
    s = math.sqrt(0.5)
    cases = {  # q_frd (w, x, y, z) -> q_flu
        (1, 0, 0, 0): (0, 1, 0, 0),
        (s, 0, 0, s): (0, s, s, 0),  # FRD yawed +90 deg about its z
        (0, 1, 0, 0): (-1, 0, 0, 0),  # FRD rotated 180 deg about x = FLU identity (q and -q are the same rotation)
        (s, s, 0, 0): (-s, s, 0, 0),
        (s, 0, s, 0): (0, s, 0, -s),
    }
    for q_frd, expected in cases.items():
        assert np.allclose(A.frd_to_flu_quaternion(q_frd), expected), q_frd


def test_frd_to_flu_rotation_matrix_and_vectors_consistent():
    rng = np.random.default_rng(0)
    c = np.diag([1.0, -1.0, -1.0])
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        r_frd = quat_to_matrix(q)
        r_flu = quat_to_matrix(A.frd_to_flu_quaternion(q))
        assert np.allclose(r_flu, r_frd @ c)
        v_flu = np.array([1.0, 2.0, 3.0])  # a vector given in FLU body axes
        v_frd = c @ v_flu  # the same vector in FRD axes: (1, -2, -3)
        assert np.allclose(v_frd, [1.0, -2.0, -3.0]) and np.allclose(r_flu @ v_flu, r_frd @ v_frd)


def test_pose_from_msg_flu_and_frd_conversion():
    s = math.sqrt(0.5)
    flu = A.pose_from_msg(rf.pose_msg(q_wxyz=(s, 0, 0, s)), cfg(), pose_revision=4, source="slam")
    assert flu.stamp_ns == 2_000_000_000 and flu.frame_id == "map" and flu.child_frame_id == "base_link"
    assert np.allclose(flu.position_m, [1, 2, 3]) and np.allclose(flu.orientation_wxyz, [s, 0, 0, s])
    assert flu.pose_revision == 4 and flu.source == "slam" and flu.valid and flu.clock_domain == "ros_time"
    frd = A.pose_from_msg(rf.pose_msg(q_wxyz=(s, 0, 0, s)), cfg(body_convention="FRD"))
    assert np.allclose(frd.orientation_wxyz, [0, s, s, 0]) and np.allclose(frd.position_m, [1, 2, 3])
    # an un-normalised but close quaternion is normalised, a bad one is rejected
    assert np.isclose(np.linalg.norm(A.pose_from_msg(rf.pose_msg(q_wxyz=(1.0005, 0, 0, 0)), cfg()).orientation_wxyz), 1)
    for q in ((2.0, 0, 0, 0), (float("nan"), 0, 0, 0)):
        with pytest.raises(A.AdapterError) as e:
            A.pose_from_msg(rf.pose_msg(q_wxyz=q), cfg())
        assert reason(e) is F.POSE_INVALID


def test_pose_frame_mismatch_and_odometry_child_frame():
    with pytest.raises(A.AdapterError) as e:
        A.pose_from_msg(rf.pose_msg(frame="odom"), cfg())
    assert reason(e) is F.FRAME_MISMATCH
    assert A.pose_from_msg(rf.pose_msg(frame="/map"), cfg()).frame_id == "map"  # leading slash is not a different frame
    assert A.pose_from_msg(rf.pose_msg(child="base_link"), cfg()).child_frame_id == "base_link"  # Odometry
    with pytest.raises(A.AdapterError) as e:
        A.pose_from_msg(rf.pose_msg(child="imu_link"), cfg())
    assert reason(e) is F.FRAME_MISMATCH
    with pytest.raises(A.AdapterError) as e:
        A.pose_from_msg(rf.pose_msg(ns=0), cfg())
    assert reason(e) is F.TIMESTAMP_INVALID


def test_covariance_reorder_with_a_non_diagonal_example():
    """ROS order x, y, z, rot-x, rot-y, rot-z (fixed axes) == library order: row i, column j stays at [i, j]."""
    c = np.zeros((6, 6))
    np.fill_diagonal(c, [0.01, 0.02, 0.03, 0.001, 0.002, 0.003])
    c[0, 4] = c[4, 0] = 0.0004  # x - rotation about y (a typical lever-arm term)
    c[1, 3] = c[3, 1] = -0.0002  # y - rotation about x
    c[2, 5] = c[5, 2] = 0.0001
    c[0, 1] = c[1, 0] = 0.005
    flat = [float(v) for v in c.reshape(36)]
    assert flat[0 * 6 + 4] == 0.0004 and flat[1 * 6 + 3] == -0.0002  # row-major
    lib = A.ros_covariance_to_library(flat)
    assert np.array_equal(lib, c) and lib[0, 4] == 0.0004 and lib[4, 0] == 0.0004 and lib[1, 3] == -0.0002
    assert A.library_covariance_to_ros(lib) == flat
    # the body convention does not touch it (fixed-frame perturbation), and it reaches the PoseSample
    for conv in ("FLU", "FRD"):
        p = A.pose_from_msg(rf.pose_msg(cov=flat), cfg(body_convention=conv))
        assert np.array_equal(p.covariance, c)
    # unknown / invalid covariances
    assert A.ros_covariance_to_library([0.0] * 36) is None
    assert A.ros_covariance_to_library([-1.0] + [0.0] * 35) is None
    for bad in (flat[:35], [float("nan")] + flat[1:], list(np.diag([0.01, 0.01, -0.01, 0.01, 0.01, 0.01]).reshape(36)),
                [v + (1e-3 if i == 5 else 0.0) for i, v in enumerate(flat)]):
        with pytest.raises(A.AdapterError) as e:
            A.ros_covariance_to_library(bad)
        assert reason(e) is F.POSE_INVALID


# --------------------------------------------------------------------------- tags
def _tag(**over):
    kw = dict(survivor_id=5, status=TagStatus.CONFIRMED, cell=(3, 4), cell_label="3,4",
              cell_probs=(((3, 4), 0.93), ((3, 3), 0.06), ((2, 4), 0.01)),
              p_map_m=np.array([4.25, 5.5, 0.8]), cov_map_m2=np.array([[0.04, 0.01, 0], [0.01, 0.09, 0], [0, 0, 0]]),
              p_grid_m=np.array([3.4, 4.6]), sigma_horizontal_m=0.3, boundary_distance_m=0.2, out_of_grid=False,
              max_detection_confidence=0.87, localisation_quality=0.93, observation_count=12,
              effective_independent_count=2.0, first_seen_ns=1_000_000_000, last_seen_ns=9_500_000_000,
              capture_stamp_ns=9_400_000_000, update_stamp_ns=9_600_000_123, map_revision=2, registration_version=1,
              extrinsic_version=3, reasons=(F.CELL_BOUNDARY_AMBIGUOUS, F.OUT_OF_GRID),
              confirmed_stamp_ns=4_000_000_000, cell_changes=1)
    kw.update(over)
    return Tag(**kw)


def _assert_tags_equal(a: Tag, b: Tag):
    assert a.survivor_id == b.survivor_id and a.status == b.status and a.cell == b.cell
    assert a.cell_label == b.cell_label and a.cell_probs == b.cell_probs and a.out_of_grid == b.out_of_grid
    for x, y in ((a.p_map_m, b.p_map_m), (a.cov_map_m2, b.cov_map_m2), (a.p_grid_m, b.p_grid_m)):
        assert (x is None) == (y is None) and (x is None or np.array_equal(x, y))
    for n in ("sigma_horizontal_m", "boundary_distance_m", "max_detection_confidence", "localisation_quality",
              "observation_count", "effective_independent_count", "first_seen_ns", "last_seen_ns", "capture_stamp_ns",
              "update_stamp_ns", "map_revision", "registration_version", "extrinsic_version", "reasons",
              "confirmed_stamp_ns", "cell_changes"):
        assert getattr(a, n) == getattr(b, n), n


def test_tag_round_trip_full_and_minimal():
    t = _tag()
    d = A.tag_to_msg_dict(t, "map", tag_revision=2, removed=False)
    assert d["status"] == 2 and d["has_cell"] and (d["cell_i"], d["cell_j"]) == (3, 4) and d["tag_revision"] == 2
    assert d["covariance_map_xy_m2"] == [0.04, 0.01, 0.01, 0.09] and d["warning_codes"] == [
        "CELL_BOUNDARY_AMBIGUOUS", "OUT_OF_GRID"]
    assert d["header"]["stamp"] == {"sec": 9, "nanosec": 600_000_123} and d["header"]["frame_id"] == "map"
    back, rev, removed = A.tag_from_msg_dict(d)
    _assert_tags_equal(t, back)
    assert rev == 2 and removed is False
    # a tag with every optional value unknown stays unknown (no fake zeros)
    m = _tag(cell=None, cell_label=None, cell_probs=(), p_map_m=None, cov_map_m2=None, p_grid_m=None,
             sigma_horizontal_m=None, boundary_distance_m=None, extrinsic_version=None, reasons=(),
             confirmed_stamp_ns=None, status=TagStatus.CANDIDATE)
    dm = A.tag_to_msg_dict(m, "map", 0, True)
    assert not dm["has_cell"] and not dm["has_position"] and not dm["has_grid_position"] and dm["removed"]
    assert math.isnan(dm["sigma_horizontal_m"]) and dm["extrinsic_version"] == -1 and dm["cell_label"] == ""
    assert not dm["has_confirmed_time"] and all(math.isnan(v) for v in dm["covariance_map_xy_m2"])
    bm, _, rm = A.tag_from_msg_dict(dm)
    _assert_tags_equal(m, bm)
    assert rm is True


def test_published_tag_array_and_tag_result():
    pt = NS(survivor_id=9, tag_revision=3, tag=_tag(survivor_id=1), removed=False)
    arr = A.tag_array_to_msg_dict([pt], "map", 10_000_000_000, 4, 2, True, True)
    assert arr["tags"][0]["survivor_id"] == 9 and arr["tags"][0]["tag_revision"] == 3 and arr["is_snapshot"]
    assert arr["map_rev"] == 4 and arr["reg_rev"] == 2 and arr["registration_valid"]
    tr = tag_result(_tag())
    d = A.tag_result_to_msg_dict(tr, "map", 10_000_000_000)
    assert d["has_cell"] and d["cell_i"] == 3 and d["localisation_quality"] == pytest.approx(0.93)
    assert not d["has_grid_position"] and math.isnan(d["sigma_horizontal_m"]) and d["has_confirmed_time"]
    nopos = A.tag_result_to_msg_dict(TagResult(1, None, TagStatus.PROVISIONAL, None, 1_000_000_000, None), "map", 5)
    assert not nopos["has_position"] and not nopos["has_cell"] and not nopos["has_confirmed_time"]
    assert math.isnan(nopos["position_map"]["z"]) or not nopos["has_position"]


def test_message_size_estimate_for_six_tags():
    reg = rf.msgdef.load_package(rf.MSGS_PKG)
    pts = [NS(survivor_id=i + 1, tag_revision=i, tag=_tag(survivor_id=i + 1), removed=False) for i in range(6)]
    arr = A.tag_array_to_msg_dict(pts, "map", 10_000_000_000, 2, 1, True, True)
    one = rf.msgdef.cdr_size(reg, "survivor_tagging_msgs", "SurvivorTag", arr["tags"][0])
    total = rf.msgdef.cdr_size(reg, "survivor_tagging_msgs", "SurvivorTagArray", arr)
    print(f"SurvivorTag {one} B, SurvivorTagArray(6 tags) {total} B")  # reported in docs/handoff/ros_interfaces.md
    assert 250 < one < 700 and abs(total - 6 * one) < 60  # header + counters, minus alignment padding
    assert total < 4096  # one DDS datagram budget on a local link; the telemetry packet stays separate
    # a record packet is tiny by comparison
    pkt = A.telemetry_packet_to_msg_dict(_record_bytes(), 5)
    assert rf.msgdef.cdr_size(reg, "survivor_tagging_msgs", "TelemetryPacket", pkt) < 120


# --------------------------------------------------------------------------- telemetry
def _record_bytes():
    return encode_tag_record(TagRecord(1, 5, TagStatus.CONFIRMED, TagFlags.REGISTRATION_VALID, (3, 4), 0.93, 0.87, 0.3,
                                       (3.4, 4.6), 1234, 2, 1))


def test_telemetry_packet_round_trip_and_corruption():
    rec = _record_bytes()
    d = A.telemetry_packet_to_msg_dict(rec, 7_000_000_000, "map")
    assert d["kind"] == 1 and bytes(d["data"]) == rec and d["header"]["stamp"]["sec"] == 7
    assert A.telemetry_bytes_from_msg(NS(kind=d["kind"], data=bytes(d["data"]))) == rec
    snap = encode_snapshot(Snapshot(1, 2, 2, 1, SnapshotFlags.REGISTRATION_VALID, 99,
                                    (TagRecord(1, 5, TagStatus.CONFIRMED, TagFlags.NONE, (3, 4), 0.9, 0.9, 0.3,
                                               (3.4, 4.6), 1234, 2, 1),)))
    assert A.telemetry_packet_to_msg_dict(snap, 1)["kind"] == 2
    for bad in (b"", rec[:-1], rec[:-2] + b"\x00\x00", b"\x7f" + rec[1:]):
        with pytest.raises(A.AdapterError) as e:
            A.telemetry_packet_to_msg_dict(bad, 1)
        assert reason(e) is F.IMAGE_FORMAT_INVALID
    with pytest.raises(A.AdapterError):
        A.telemetry_bytes_from_msg(NS(kind=2, data=rec))  # kind does not match the first byte


# --------------------------------------------------------------------------- corrections / registration / commit
def _corr_msg(mode="history", rev=1, n=3, frame="map", start=None, end=None, with_poses=True):
    poses = [rf.pose_msg(ns=(10 + k) * 10**8, frame=frame, pos=(float(k), 0.0, 1.0), q_wxyz=(1, 0, 0, 0),
                         cov=[0.01 if i % 7 == 0 else 0.0 for i in range(36)]) for k in range(n)] if with_poses else []
    ang = math.radians(90)
    return NS(header=rf.header(30 * 10**8, frame), revision=rev, mode=mode,
              range_start=rf.stamp(10**9) if start is None else start,
              range_end=rf.stamp((10 + n - 1) * 10**8) if end is None else end, corrected_poses=poses,
              frame_transform=NS(translation=NS(x=0.1, y=0.2, z=0.0),
                                 rotation=NS(x=0.0, y=0.0, z=math.sin(ang / 2), w=math.cos(ang / 2))))


def test_correction_history_and_frame_modes():
    h = A.correction_from_msg(_corr_msg(), cfg(body_convention="FRD"))
    assert h.source == "history" and h.revision == 1 and h.stamp_ns == 3 * 10**9 and len(h.poses) == 3
    assert all(p.pose_revision == 1 and p.valid for p in h.poses)
    assert np.allclose(h.poses[0].orientation_wxyz, [0, 1, 0, 0])  # FRD identity -> FLU 180 deg about x
    assert h.range_ns == (10**9, 12 * 10**8)
    assert validate_correction(h, "history", "ros_time") is None  # the library accepts it
    f = A.correction_from_msg(_corr_msg("frame_correction", 2, with_poses=False), cfg())
    assert f.source == "frame_correction" and f.poses == () and f.revision == 2
    expected = np.array([[0, -1, 0, 0.1], [1, 0, 0, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    assert np.allclose(f.T_new_old, expected, atol=1e-12)
    assert validate_correction(f, "frame_correction", "ros_time") is None


@pytest.mark.parametrize("kw,why", [
    (dict(rev=0), F.CORRECTION_INVALID), (dict(mode="bogus"), F.CORRECTION_INVALID),
    (dict(frame="odom"), F.FRAME_MISMATCH), (dict(with_poses=False), F.CORRECTION_INVALID),
    (dict(start=rf.stamp(5 * 10**8)), F.CORRECTION_INVALID), (dict(end=rf.stamp(99 * 10**9)), F.CORRECTION_INVALID),
])
def test_correction_rejections(kw, why):
    with pytest.raises(A.AdapterError) as e:
        A.correction_from_msg(_corr_msg(**kw), cfg())
    assert reason(e) is why


def test_correction_other_rejections():
    m = _corr_msg()
    m.corrected_poses = list(reversed(m.corrected_poses))
    with pytest.raises(A.AdapterError) as e:
        A.correction_from_msg(m, cfg())
    assert reason(e) is F.CORRECTION_INVALID
    with pytest.raises(A.AdapterError):  # frame correction must not carry poses
        A.correction_from_msg(_corr_msg("frame_correction", 2, with_poses=True), cfg())
    bad = _corr_msg("frame_correction", 2, with_poses=False)
    bad.frame_transform.rotation.w = 3.0
    with pytest.raises(A.AdapterError):
        A.correction_from_msg(bad, cfg())
    zero_range = _corr_msg(start=rf.ZERO_STAMP, end=rf.ZERO_STAMP)  # unset range is not checked
    assert A.correction_from_msg(zero_range, cfg()).revision == 1


def _reg_req(**kw):
    base = dict(origin_in_map_m=[1.2, 0.8], yaw_map_to_grid_rad=0.25, axis_flip_x=False, axis_flip_y=True,
                index_base=1, numbering="ij", cell_size_m=1.0, n_cells_x=15, n_cells_y=0, version=3, valid=True)
    base.update(kw)
    return NS(**base)


def test_registration_request_rules():
    cur = GridRegistration((0.0, 0.0), 0.0, version=2, boundary_rule="floor")
    r = A.registration_from_request(_reg_req(), cur)
    assert r.origin_in_map_m == (1.2, 0.8) and r.yaw_rad == 0.25 and r.flip_y and not r.flip_x and r.index_base == 1
    assert r.n_cells_x == 15 and r.n_cells_y is None and r.version == 3 and r.valid and r.numbering == "ij"
    assert r.boundary_rule == "floor"
    assert A.registration_from_request(_reg_req(version=2), cur).version == 2  # equal: the pipeline decides
    for kw, why in ((dict(version=1), F.CORRECTION_STALE), (dict(version=-1), F.CORRECTION_INVALID),
                    (dict(numbering="abc"), F.CORRECTION_INVALID), (dict(index_base=2), F.CORRECTION_INVALID),
                    (dict(cell_size_m=0.0), F.CORRECTION_INVALID), (dict(yaw_map_to_grid_rad=float("nan")),
                                                                    F.CORRECTION_INVALID),
                    (dict(origin_in_map_m=[1.0]), F.CORRECTION_INVALID)):
        with pytest.raises(A.AdapterError) as e:
            A.registration_from_request(_reg_req(**kw), cur)
        assert reason(e) is why, kw


def test_registration_response_and_commit_args():
    from survivor_tagging.corrections import CorrectionOutcome
    assert A.registration_response_dict(CorrectionOutcome(True, 3), 3) == {"accepted": True, "reason": "",
                                                                          "current_version": 3}
    stale = A.registration_response_dict(CorrectionOutcome(False, 1, F.CORRECTION_STALE, "old"), 3)
    assert not stale["accepted"] and stale["reason"].startswith("CORRECTION_STALE")
    assert A.registration_response_dict(CorrectionOutcome(False, 3, None, "duplicate"), 3)["reason"] == "duplicate"
    assert A.commit_request_to_args(NS(policy="confirmed_only", threshold=float("nan"))) == ("confirmed_only", None)
    assert A.commit_request_to_args(NS(policy="expected_value", threshold=0.667)) == ("expected_value", 0.667)
    assert A.commit_request_to_args(NS(policy="expected_value", threshold=float("nan"))) == ("expected_value", None)
    for p, t in (("nonsense", 0.5), ("confirmed_only", 0.5), ("expected_value", 0.0), ("expected_value", 1.5)):
        with pytest.raises(A.AdapterError):
            A.commit_request_to_args(NS(policy=p, threshold=t))
    r = A.commit_response_dict([TagResult(1, (3, 4), TagStatus.CONFIRMED, np.array([1.0, 2.0]), 10**9, 2 * 10**9, 0.9)],
                               "expected_value", 0.667, "map", 5 * 10**9)
    assert r["success"] and r["threshold_used"] == 0.667 and r["tags"][0]["cell_i"] == 3
    assert math.isnan(A.commit_response_dict([], "confirmed_only", None, "map", 1)["threshold_used"])
