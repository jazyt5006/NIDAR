"""ROS-free adapters between ROS 2 messages and the survivor_tagging library types (substep 4.1a).

Pure Python + numpy. No rclpy, no ROS message imports: messages are DUCK-TYPED (any object with the same attributes
works, so the unit tests use fake objects), and messages this module produces are plain dicts with exactly the field
names of the .msg files (nested messages are dicts, arrays are lists). A node (substep 4.1b) copies those dicts into
real message objects. Every failure raises AdapterError with an explicit `reason` (a library FailureReason); a missing
parameter raises ConfigurationError (reason CONFIG_MISSING). Nothing is guessed.

Conventions (docs/handoff/ros_interfaces.md):
- Time: ROS builtin_interfaces/Time (sec, nanosec) <-> integer nanoseconds in the configured clock domain. A zero stamp
  is "unset" and is rejected. Detection stamps must be the IMAGE CAPTURE time from the camera image header.
- Body frame: the library uses FLU (x forward, y left, z up). A source in FRD (x forward, y right, z down) is
  converted with R_flu = R_frd * diag(1, -1, -1), i.e. q_flu = q_frd (x) (w=0, x=1, y=0, z=0). Positions are the
  body origin and do not change.
- Covariance: the ROS 6x6 pose covariance (row-major, order x, y, z, rotation about FIXED x, y, z, in the message
  frame) is the library layout (docs/design/conventions.md: (dt, dtheta) in the MAP frame, order x, y, z, rx, ry, rz),
  so the mapping is a reshape. It is NOT changed by the FRD/FLU choice: the body axes enter as R' = R * C, and a
  fixed-frame perturbation Exp(dtheta) R C has the same dtheta.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from survivor_tagging.grid import GridRegistration
from survivor_tagging.output import MSG_SNAPSHOT, MSG_TAG_UPDATE, TelemetryDecodeError, decode_snapshot, \
    decode_tag_record
from survivor_tagging.types import (CameraIntrinsics, DepthMeasurement, Detection, FailureReason, PoseCorrection,
                                    PoseSample, Tag, TagResult, TagStatus)

NS_PER_S = 1_000_000_000
BODY_CONVENTIONS = ("FLU", "FRD")
COMMIT_POLICIES = ("confirmed_only", "confirmed_plus_provisional_top1", "expected_value")
NUMBERINGS = ("ij", "row_major_label", "alnum")
QUAT_NORM_TOL = 1e-3  # ASSUMPTION: message quaternions within this of unit norm are normalised, others rejected
COV_SYMMETRY_TOL = 1e-9  # relative, ASSUMPTION


class AdapterError(ValueError):
    """A message was rejected. `reason` is a library FailureReason; `detail` says what exactly."""

    def __init__(self, reason: FailureReason, detail: str):
        super().__init__(f"{reason.value}: {detail}")
        self.reason, self.detail = reason, detail


class ConfigurationError(ValueError):
    """A required adapter parameter is unset (never guessed). reason is always CONFIG_MISSING."""

    reason = FailureReason.CONFIG_MISSING

    def __init__(self, names: Sequence[str]):
        super().__init__(f"{FailureReason.CONFIG_MISSING.value}: unset parameter(s) {', '.join(names)}")
        self.names = tuple(names)


@dataclass(frozen=True)
class AdapterConfig:
    """Adapter parameters. Every field defaults to None = unset; a function that needs one raises ConfigurationError.
    (ROS parameter names in the handoff document.) image_scale_tolerance is the only tuning default (ASSUMPTION)."""

    map_frame_id: Optional[str] = None  # frame of poses and tags, e.g. "map"
    body_frame_id: Optional[str] = None  # child frame of the pose, e.g. "base_link"
    body_convention: Optional[str] = None  # "FLU" or "FRD": axes of the body frame the pose refers to
    clock_domain: Optional[str] = None  # clock domain label of all stamps, e.g. "ros_time"
    depth_aligned_to_color: Optional[bool] = None  # = library cam.detections_use_depth_aligned_frame (Q D2)
    depth_min_m: Optional[float] = None  # = l1.depth_min_m
    depth_max_m: Optional[float] = None  # = l1.depth_max_m
    image_scale_tolerance: float = 0.01  # ASSUMPTION: |sx / sy - 1| allowed when scaling a box to the depth image

    def require(self, *names: str) -> None:
        missing = [n for n in names if getattr(self, n) is None]
        if missing:
            raise ConfigurationError(missing)
        if self.body_convention is not None and self.body_convention not in BODY_CONVENTIONS:
            raise ConfigurationError([f"body_convention (value {self.body_convention!r} must be one of "
                                      f"{BODY_CONVENTIONS})"])


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def stamp_to_ns(stamp: Any) -> int:
    """builtin_interfaces/Time (attributes sec, nanosec) -> integer nanoseconds. Rejects a zero stamp (unset), negative
    seconds, nanosec outside [0, 1e9) and non-integers, with TIMESTAMP_INVALID."""
    try:
        sec, nsec = stamp.sec, stamp.nanosec
    except AttributeError:
        raise AdapterError(FailureReason.TIMESTAMP_INVALID, "stamp has no sec / nanosec") from None
    if isinstance(sec, bool) or isinstance(nsec, bool) or not isinstance(sec, (int, np.integer)) or \
            not isinstance(nsec, (int, np.integer)):
        raise AdapterError(FailureReason.TIMESTAMP_INVALID, f"stamp sec={sec!r} nanosec={nsec!r} must be integers")
    sec, nsec = int(sec), int(nsec)
    if sec < 0:
        raise AdapterError(FailureReason.TIMESTAMP_INVALID, f"negative stamp sec={sec}")
    if not 0 <= nsec < NS_PER_S:
        raise AdapterError(FailureReason.TIMESTAMP_INVALID, f"nanosec={nsec} outside [0, 1e9)")
    ns = sec * NS_PER_S + nsec
    if ns == 0:
        raise AdapterError(FailureReason.TIMESTAMP_INVALID, "zero stamp (unset)")
    return ns


def ns_to_stamp(ns: int) -> Dict[str, int]:
    """Integer nanoseconds (>= 0) -> {"sec", "nanosec"} (a builtin_interfaces/Time as a dict)."""
    if isinstance(ns, bool) or not isinstance(ns, (int, np.integer)) or ns < 0:
        raise AdapterError(FailureReason.TIMESTAMP_INVALID, f"cannot convert {ns!r} to a stamp")
    return {"sec": int(ns) // NS_PER_S, "nanosec": int(ns) % NS_PER_S}


class StampMonitor:
    """Rejects stamps that go backwards (beyond max_backward_s) or jump forwards by more than max_jump_s (None = no
    limit): a detector or driver that stamps with wall-clock time after inference shows up as an irregular sequence."""

    def __init__(self, max_backward_s: float = 0.0, max_jump_s: Optional[float] = None):
        self.max_backward_ns = int(max_backward_s * NS_PER_S)
        self.max_jump_ns = None if max_jump_s is None else int(max_jump_s * NS_PER_S)
        self.last_ns: Optional[int] = None

    def check(self, ns: int) -> int:
        if self.last_ns is not None:
            if ns < self.last_ns - self.max_backward_ns:
                raise AdapterError(FailureReason.TIMESTAMP_INVALID,
                                   f"stamp {ns} ns is {(self.last_ns - ns) / 1e9:.6f} s before the previous one")
            if self.max_jump_ns is not None and ns - self.last_ns > self.max_jump_ns:
                raise AdapterError(FailureReason.TIMESTAMP_INVALID,
                                   f"stamp jumps {(ns - self.last_ns) / 1e9:.3f} s forward")
        self.last_ns = ns if self.last_ns is None else max(self.last_ns, ns)
        return ns


def header_stamp_ns(header: Any, monitor: Optional[StampMonitor] = None) -> int:
    ns = stamp_to_ns(header.stamp)
    return ns if monitor is None else monitor.check(ns)


def _frame_id(header: Any) -> str:
    return str(header.frame_id).lstrip("/")


# ---------------------------------------------------------------------------
# Depth image and camera intrinsics
# ---------------------------------------------------------------------------

_DEPTH_ENCODINGS = {"16UC1": (2, "u2"), "32FC1": (4, "f4")}


def _image_array(msg: Any) -> np.ndarray:
    """Pixel array (height, width) of a depth image message, honouring step (row stride) and endianness."""
    enc = str(msg.encoding)
    if enc not in _DEPTH_ENCODINGS:
        raise AdapterError(FailureReason.IMAGE_FORMAT_INVALID,
                           f"depth encoding {enc!r} not supported (16UC1 or 32FC1)")
    isz, kind = _DEPTH_ENCODINGS[enc]
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    if h <= 0 or w <= 0:
        raise AdapterError(FailureReason.IMAGE_FORMAT_INVALID, f"image size {w}x{h}")
    if step < w * isz:
        raise AdapterError(FailureReason.IMAGE_FORMAT_INVALID, f"step {step} < width*{isz} = {w * isz}")
    data = msg.data
    buf = np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, bytearray, memoryview)) \
        else np.asarray(data, dtype=np.uint8).reshape(-1)
    if buf.size != step * h:
        raise AdapterError(FailureReason.IMAGE_FORMAT_INVALID, f"data length {buf.size} != step*height = {step * h}")
    rows = np.ascontiguousarray(buf.reshape(h, step)[:, :w * isz])
    order = ">" if getattr(msg, "is_bigendian", 0) else "<"
    return rows.view(np.dtype(order + kind)).reshape(h, w)


def depth_image_to_metres(msg: Any) -> np.ndarray:
    """Depth in metres as float64 (h, w); invalid pixels are NaN. 16UC1: millimetres, 0 = invalid (REP 118).
    32FC1: metres, NaN, +-inf and <= 0 are invalid."""
    arr = _image_array(msg)
    if str(msg.encoding) == "16UC1":
        out = arr.astype(np.float64) * 0.001
        out[arr == 0] = np.nan
        return out
    out = arr.astype(np.float64)
    out[~np.isfinite(out) | (out <= 0)] = np.nan
    return out


def depth_image_to_raw(msg: Any) -> Tuple[np.ndarray, float]:
    """(raw array, metres per unit) in the form the library expects (0 = no depth): 16UC1 stays uint16 with scale
    0.001; 32FC1 becomes float32 with invalid pixels set to 0 and scale 1.0."""
    arr = _image_array(msg)
    if str(msg.encoding) == "16UC1":
        return arr.astype(np.uint16, copy=False), 0.001
    out = arr.astype(np.float32)
    out[~np.isfinite(out) | (out <= 0)] = 0.0
    return out, 1.0


def intrinsics_from_camera_info(info: Any) -> CameraIntrinsics:
    """CameraInfo (k row-major 3x3, width, height, distortion_model, d) -> CameraIntrinsics. An all-zero k
    (uncalibrated), zero skew violation or non-positive focal length gives INTRINSICS_MISSING."""
    k = [float(v) for v in info.k]
    if len(k) != 9 or not all(math.isfinite(v) for v in k):
        raise AdapterError(FailureReason.INTRINSICS_MISSING, "CameraInfo.k must hold 9 finite values")
    if all(v == 0.0 for v in k):
        raise AdapterError(FailureReason.INTRINSICS_MISSING, "CameraInfo.k is all zero (uncalibrated)")
    if k[1] != 0.0 or k[3] != 0.0 or k[6] != 0.0 or k[7] != 0.0 or k[8] != 1.0:
        raise AdapterError(FailureReason.INTRINSICS_MISSING, f"CameraInfo.k is not a zero-skew pinhole matrix: {k}")
    intr = CameraIntrinsics(k[0], k[4], k[2], k[5], int(info.width), int(info.height),
                            str(getattr(info, "distortion_model", "")), tuple(float(v) for v in getattr(info, "d", ())))
    if intr.failure() is not None:
        raise AdapterError(FailureReason.INTRINSICS_MISSING, f"invalid intrinsics {intr}")
    return intr


def depth_measurement_from_msgs(img: Any, info: Any, cfg: AdapterConfig,
                                monitor: Optional[StampMonitor] = None) -> DepthMeasurement:
    """Depth Image + CameraInfo -> DepthMeasurement. The image size must equal the CameraInfo size. The alignment flag
    comes ONLY from cfg.depth_aligned_to_color (unset -> ConfigurationError)."""
    cfg.require("clock_domain", "depth_aligned_to_color", "depth_min_m", "depth_max_m")
    stamp = header_stamp_ns(img.header, monitor)
    raw, scale = depth_image_to_raw(img)
    intr = intrinsics_from_camera_info(info)
    if raw.shape != (intr.height_px, intr.width_px):
        raise AdapterError(FailureReason.INTRINSICS_MISSING,
                           f"depth image {raw.shape[1]}x{raw.shape[0]} differs from CameraInfo "
                           f"{intr.width_px}x{intr.height_px}")
    return DepthMeasurement(stamp, cfg.clock_domain, _frame_id(img.header), raw, scale, intr,
                            bool(cfg.depth_aligned_to_color), (float(cfg.depth_min_m), float(cfg.depth_max_m)))


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------


@dataclass
class DetectionBatch:
    stamp_ns: int
    frame_id: str
    detections: List[Detection]
    rejected: List[Tuple[int, FailureReason, str]] = field(default_factory=list)  # (index in the message, reason, detail)


def detection_from_msg(det: Any, index: int, stamp_ns: int, frame_id: str, cfg: AdapterConfig) -> Detection:
    """One Detection message -> library Detection. detection_id = index in the array (the message has no id).
    tracking_id < 0 means none. Raises AdapterError (BBOX_INVALID / IMAGE_FORMAT_INVALID)."""
    cfg.require("clock_domain")
    w, h = int(det.source_image_width), int(det.source_image_height)
    if w <= 0 or h <= 0:
        raise AdapterError(FailureReason.IMAGE_FORMAT_INVALID, f"source image size {w}x{h}")
    box = tuple(float(v) for v in det.bbox_xyxy)
    if len(box) != 4:
        raise AdapterError(FailureReason.BBOX_INVALID, f"bbox_xyxy has {len(box)} values")
    conf = float(det.confidence)
    if not math.isfinite(conf) or not 0.0 <= conf <= 1.0:
        raise AdapterError(FailureReason.BBOX_INVALID, f"confidence {conf!r} outside [0, 1]")
    tid = int(det.tracking_id)
    out = Detection(index, conf, box, stamp_ns, cfg.clock_domain, frame_id, (w, h), None if tid < 0 else tid,
                    str(det.class_name) or None)
    bad = out.bbox_failure()
    if bad is not None:
        raise AdapterError(bad, f"bbox {box} in a {w}x{h} image")
    return out


def scale_detection_to_depth(det: Detection, depth_w: int, depth_h: int, cfg: AdapterConfig) -> Detection:
    """The library needs the box in the pixels of the depth image. Equal size: unchanged. Different size: allowed only
    if the depth image is aligned to the detection image (cfg.depth_aligned_to_color True) and the two scale factors
    agree within cfg.image_scale_tolerance (same field of view; a crop or other aspect ratio is not a scaling);
    then the box is multiplied by (depth_w / w, depth_h / h) and image_size_px becomes the depth size. Otherwise
    DEPTH_FRAME_MISMATCH."""
    cfg.require("depth_aligned_to_color")
    w, h = det.image_size_px
    if (w, h) == (depth_w, depth_h):
        return det
    if not cfg.depth_aligned_to_color:
        raise AdapterError(FailureReason.DEPTH_FRAME_MISMATCH,
                           f"detection image {w}x{h} != depth image {depth_w}x{depth_h} and depth is not aligned")
    sx, sy = depth_w / w, depth_h / h
    if abs(sx / sy - 1.0) > cfg.image_scale_tolerance:
        raise AdapterError(FailureReason.DEPTH_FRAME_MISMATCH,
                           f"scale factors differ (x {sx:.4f}, y {sy:.4f}): not the same field of view")
    x0, y0, x1, y1 = det.bbox_px
    scaled = replace(det, bbox_px=(x0 * sx, y0 * sy, x1 * sx, y1 * sy), image_size_px=(depth_w, depth_h))
    bad = scaled.bbox_failure()
    if bad is not None:
        raise AdapterError(bad, f"scaled bbox {scaled.bbox_px} in {depth_w}x{depth_h}")
    return scaled


def detections_from_array(msg: Any, cfg: AdapterConfig, monitor: Optional[StampMonitor] = None,
                          depth_size: Optional[Tuple[int, int]] = None) -> DetectionBatch:
    """DetectionArray -> DetectionBatch. Array-level problems (bad stamp, empty frame_id) raise AdapterError;
    per-detection problems are collected in `rejected` and the other detections are kept. depth_size = (width, height)
    of the depth image: boxes are scaled to it by the rule in scale_detection_to_depth."""
    cfg.require("clock_domain")
    stamp = header_stamp_ns(msg.header, monitor)
    frame = _frame_id(msg.header)
    if not frame:
        raise AdapterError(FailureReason.FRAME_MISMATCH, "DetectionArray.header.frame_id is empty")
    batch = DetectionBatch(stamp, frame, [])
    for i, d in enumerate(msg.detections):
        try:
            det = detection_from_msg(d, i, stamp, frame, cfg)
            if depth_size is not None:
                det = scale_detection_to_depth(det, depth_size[0], depth_size[1], cfg)
            batch.detections.append(det)
        except AdapterError as exc:
            batch.rejected.append((i, exc.reason, exc.detail))
    return batch


# ---------------------------------------------------------------------------
# Poses
# ---------------------------------------------------------------------------

_C_FRD_FLU = np.diag([1.0, -1.0, -1.0])


def quat_multiply_wxyz(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def frd_to_flu_quaternion(q_frd_wxyz: Sequence[float]) -> np.ndarray:
    """Orientation of a FRD body frame -> orientation of the FLU body frame with the same origin:
    q_flu = q_frd (x) (0, 1, 0, 0) = (-x, w, z, -y) for q_frd = (w, x, y, z). Equivalent to R_flu = R_frd diag(1, -1, -1)."""
    w, x, y, z = (float(v) for v in q_frd_wxyz)
    return np.array([-x, w, z, -y])


def ros_covariance_to_library(cov36: Sequence[float]) -> Optional[np.ndarray]:
    """ROS pose covariance (36 values, row-major, order x, y, z, rot-x, rot-y, rot-z about FIXED axes of the message
    frame) -> library 6x6 (same order, same frame): a reshape plus symmetrisation. Unknown is None: all zeros (what
    many drivers publish when they have none; a zero covariance would claim a perfect pose) or cov[0] == -1 (ROS
    convention for 'not available'). NaN/inf, a negative diagonal or an asymmetry above COV_SYMMETRY_TOL: POSE_INVALID."""
    c = [float(v) for v in cov36]
    if len(c) != 36:
        raise AdapterError(FailureReason.POSE_INVALID, f"covariance has {len(c)} values, expected 36")
    if c[0] == -1.0 or all(v == 0.0 for v in c):
        return None
    m = np.array(c).reshape(6, 6)
    if not np.all(np.isfinite(m)):
        raise AdapterError(FailureReason.POSE_INVALID, "covariance contains NaN or inf")
    scale = max(float(np.max(np.abs(m))), 1e-300)
    if np.max(np.abs(m - m.T)) > COV_SYMMETRY_TOL * scale:
        raise AdapterError(FailureReason.POSE_INVALID, "covariance is not symmetric")
    if np.any(np.diag(m) < 0):
        raise AdapterError(FailureReason.POSE_INVALID, "covariance has a negative diagonal")
    return 0.5 * (m + m.T)


def library_covariance_to_ros(cov: Optional[np.ndarray]) -> List[float]:
    """Inverse of ros_covariance_to_library (None -> all zeros)."""
    return [0.0] * 36 if cov is None else [float(v) for v in np.asarray(cov, dtype=float).reshape(36)]


def pose_from_msg(msg: Any, cfg: AdapterConfig, pose_revision: int = 0, source: str = "ros",
                  monitor: Optional[StampMonitor] = None) -> PoseSample:
    """PoseWithCovarianceStamped or nav_msgs/Odometry (both: header, pose.pose, pose.covariance; Odometry also
    child_frame_id) -> PoseSample in the library convention (FLU body, covariance as above).
    header.frame_id must equal cfg.map_frame_id (FRAME_MISMATCH); an Odometry child_frame_id must equal
    cfg.body_frame_id. pose_revision is not in ROS messages: the caller supplies it (SLAM correction counter)."""
    cfg.require("map_frame_id", "body_frame_id", "body_convention", "clock_domain")
    stamp = header_stamp_ns(msg.header, monitor)
    frame = _frame_id(msg.header)
    if frame != str(cfg.map_frame_id).lstrip("/"):
        raise AdapterError(FailureReason.FRAME_MISMATCH, f"pose frame {frame!r}, configured map frame "
                                                         f"{cfg.map_frame_id!r}")
    child = getattr(msg, "child_frame_id", None)
    if child not in (None, "") and str(child).lstrip("/") != str(cfg.body_frame_id).lstrip("/"):
        raise AdapterError(FailureReason.FRAME_MISMATCH, f"child frame {child!r}, configured body frame "
                                                         f"{cfg.body_frame_id!r}")
    p, o = msg.pose.pose.position, msg.pose.pose.orientation
    pos = np.array([float(p.x), float(p.y), float(p.z)])
    q = np.array([float(o.w), float(o.x), float(o.y), float(o.z)])
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(q))):
        raise AdapterError(FailureReason.POSE_INVALID, "pose contains NaN or inf")
    n = float(np.linalg.norm(q))
    if abs(n - 1.0) > QUAT_NORM_TOL:
        raise AdapterError(FailureReason.POSE_INVALID, f"quaternion norm {n:.6f}")
    q = q / n
    if cfg.body_convention == "FRD":
        q = frd_to_flu_quaternion(q)
    cov = ros_covariance_to_library(msg.pose.covariance)
    return PoseSample(stamp, cfg.clock_domain, frame, str(cfg.body_frame_id).lstrip("/"), pos, q, True,
                      int(pose_revision), source, cov)


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

NAN = float("nan")

# Library Tag field -> SurvivorTag message field(s). None = not mapped; the reason is in TAG_OMITTED.
TAG_FIELD_MAP: Dict[str, Tuple[str, ...]] = {
    "survivor_id": ("survivor_id",),
    "status": ("status",),
    "cell": ("has_cell", "cell_i", "cell_j"),
    "cell_label": ("cell_label",),
    "cell_probs": ("candidate_cells",),
    "p_map_m": ("has_position", "position_map"),
    "cov_map_m2": ("covariance_map_xy_m2",),
    "p_grid_m": ("has_grid_position", "position_grid_xy_m"),
    "sigma_horizontal_m": ("sigma_horizontal_m",),
    "boundary_distance_m": ("boundary_distance_m",),
    "out_of_grid": ("out_of_grid",),
    "max_detection_confidence": ("max_detection_confidence",),
    "localisation_quality": ("localisation_quality",),
    "observation_count": ("observation_count",),
    "effective_independent_count": ("effective_independent_count",),
    "first_seen_ns": ("first_seen_time",),
    "last_seen_ns": ("last_seen_time",),
    "capture_stamp_ns": ("capture_time",),
    "update_stamp_ns": ("header",),  # header.stamp
    "map_revision": ("map_rev",),
    "registration_version": ("reg_rev",),
    "extrinsic_version": ("extrinsic_version",),
    "reasons": ("warning_codes",),
    "confirmed_stamp_ns": ("has_confirmed_time", "confirmed_time"),
    "cell_changes": ("cell_changes",),
}
# Parts of mapped fields that are deliberately not transmitted, and message fields that do not come from Tag.
TAG_OMITTED: Dict[str, str] = {
    "cov_map_m2[z rows/columns]": "the library fills them with zeros (Layer 4 tracks x, y only); the message carries "
                                  "the 2x2 xy block",
}
TAG_MSG_ONLY: Dict[str, str] = {
    "tag_revision": "Layer 5 publication field (output.PublishedTag.tag_revision)",
    "removed": "Layer 5 publication field (output.PublishedTag.removed)",
}
DETECTION_FIELD_MAP: Dict[str, Tuple[str, ...]] = {
    "detection_id": (),  # assigned by the adapter = index in the array
    "confidence": ("confidence",),
    "bbox_px": ("bbox_xyxy",),
    "image_stamp_ns": ("header.stamp",),  # of the array
    "clock_domain": (),  # adapter parameter
    "image_frame_id": ("header.frame_id",),
    "image_size_px": ("source_image_width", "source_image_height"),
    "tracking_id": ("tracking_id",),
    "class_label": ("class_name",),
}
DETECTION_OMITTED: Dict[str, str] = {
    "detection_id": "the message has none; the adapter uses the index in the array",
    "clock_domain": "not a message field; AdapterConfig.clock_domain",
    "frame_index": "message field with no library counterpart (diagnostics only)",
    "class_id": "message field with no library counterpart (the library keeps class_label for logs)",
}
CORRECTION_FIELD_MAP: Dict[str, Tuple[str, ...]] = {
    "revision": ("revision",),
    "source": ("mode",),
    "stamp_ns": ("header.stamp",),
    "clock_domain": (),  # adapter parameter
    "poses": ("corrected_poses",),
    "T_new_old": ("frame_transform",),
}
CORRECTION_OMITTED: Dict[str, str] = {
    "clock_domain": "not a message field; AdapterConfig.clock_domain",
    "range_start/range_end": "message fields only used to cross-check the first / last corrected pose stamp "
                             "(library PoseCorrection.range_ns is derived from the poses)",
}


def _ns_stamp_or_zero(ns: Optional[int]) -> Dict[str, int]:
    return ns_to_stamp(0 if ns is None else ns)


def tag_to_msg_dict(tag: Tag, frame_id: str, tag_revision: int = 0, removed: bool = False,
                    survivor_id: Optional[int] = None) -> Dict[str, Any]:
    """Library Tag -> SurvivorTag as a dict with exactly the message fields. Unknown optional values are flagged by
    has_* (False) or NaN / -1 / empty string, never zeros that look like data. Floats stay Python floats here; the
    real message stores float32 for the fields declared so."""
    d: Dict[str, Any] = {}
    d["header"] = {"stamp": ns_to_stamp(tag.update_stamp_ns), "frame_id": frame_id}
    d["survivor_id"] = int(tag.survivor_id if survivor_id is None else survivor_id)
    d["tag_revision"], d["map_rev"], d["reg_rev"] = int(tag_revision), int(tag.map_revision), \
        int(tag.registration_version)
    d["status"], d["removed"] = int(tag.status), bool(removed)
    d["has_position"] = tag.p_map_m is not None
    p = np.zeros(3) if tag.p_map_m is None else np.asarray(tag.p_map_m, dtype=float)
    d["position_map"] = {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
    c = tag.cov_map_m2
    d["covariance_map_xy_m2"] = [NAN] * 4 if c is None else [float(c[0, 0]), float(c[0, 1]), float(c[1, 0]),
                                                              float(c[1, 1])]
    d["has_grid_position"] = tag.p_grid_m is not None
    g = np.zeros(2) if tag.p_grid_m is None else np.asarray(tag.p_grid_m, dtype=float)
    d["position_grid_xy_m"] = [float(g[0]), float(g[1])]
    d["has_cell"] = tag.cell is not None
    d["cell_i"], d["cell_j"] = (0, 0) if tag.cell is None else (int(tag.cell[0]), int(tag.cell[1]))
    d["cell_label"] = tag.cell_label or ""
    d["candidate_cells"] = [{"i": int(c_[0]), "j": int(c_[1]), "probability": float(pr)} for c_, pr in tag.cell_probs]
    d["out_of_grid"] = bool(tag.out_of_grid)
    d["sigma_horizontal_m"] = NAN if tag.sigma_horizontal_m is None else float(tag.sigma_horizontal_m)
    d["boundary_distance_m"] = NAN if tag.boundary_distance_m is None else float(tag.boundary_distance_m)
    d["max_detection_confidence"] = float(tag.max_detection_confidence)
    d["localisation_quality"] = float(tag.localisation_quality)
    d["observation_count"] = int(tag.observation_count)
    d["effective_independent_count"] = float(tag.effective_independent_count)
    d["extrinsic_version"] = -1 if tag.extrinsic_version is None else int(tag.extrinsic_version)
    d["cell_changes"] = int(tag.cell_changes)
    d["first_seen_time"] = ns_to_stamp(tag.first_seen_ns)
    d["last_seen_time"] = ns_to_stamp(tag.last_seen_ns)
    d["capture_time"] = ns_to_stamp(tag.capture_stamp_ns)
    d["has_confirmed_time"] = tag.confirmed_stamp_ns is not None
    d["confirmed_time"] = _ns_stamp_or_zero(tag.confirmed_stamp_ns)
    d["warning_codes"] = [r.value for r in tag.reasons]
    return d


def _stamp_dict_to_ns(s: Dict[str, int]) -> int:
    return int(s["sec"]) * NS_PER_S + int(s["nanosec"])


def tag_from_msg_dict(d: Dict[str, Any]) -> Tuple[Tag, int, bool]:
    """Reverse of tag_to_msg_dict (for tests and for nodes that read tags): (Tag, tag_revision, removed). The 3x3
    covariance is rebuilt with zero z rows/columns, as the library produces it."""
    cov = None
    c4 = d["covariance_map_xy_m2"]
    if not any(math.isnan(v) for v in c4):
        cov = np.zeros((3, 3))
        cov[:2, :2] = np.array(c4).reshape(2, 2)
    p = d["position_map"]
    nan_to_none = lambda v: None if math.isnan(v) else float(v)  # noqa: E731
    tag = Tag(
        survivor_id=d["survivor_id"], status=TagStatus(d["status"]),
        cell=(d["cell_i"], d["cell_j"]) if d["has_cell"] else None, cell_label=d["cell_label"] or None,
        cell_probs=tuple(((c["i"], c["j"]), c["probability"]) for c in d["candidate_cells"]),
        p_map_m=np.array([p["x"], p["y"], p["z"]]) if d["has_position"] else None, cov_map_m2=cov,
        p_grid_m=np.array(d["position_grid_xy_m"]) if d["has_grid_position"] else None,
        sigma_horizontal_m=nan_to_none(d["sigma_horizontal_m"]),
        boundary_distance_m=nan_to_none(d["boundary_distance_m"]), out_of_grid=d["out_of_grid"],
        max_detection_confidence=d["max_detection_confidence"], localisation_quality=d["localisation_quality"],
        observation_count=d["observation_count"], effective_independent_count=d["effective_independent_count"],
        first_seen_ns=_stamp_dict_to_ns(d["first_seen_time"]), last_seen_ns=_stamp_dict_to_ns(d["last_seen_time"]),
        capture_stamp_ns=_stamp_dict_to_ns(d["capture_time"]),
        update_stamp_ns=_stamp_dict_to_ns(d["header"]["stamp"]), map_revision=d["map_rev"],
        registration_version=d["reg_rev"],
        extrinsic_version=None if d["extrinsic_version"] < 0 else d["extrinsic_version"],
        reasons=tuple(FailureReason(r) for r in d["warning_codes"]),
        confirmed_stamp_ns=_stamp_dict_to_ns(d["confirmed_time"]) if d["has_confirmed_time"] else None,
        cell_changes=d["cell_changes"])
    return tag, d["tag_revision"], d["removed"]


def published_tag_to_msg_dict(pt: Any, frame_id: str) -> Dict[str, Any]:
    """output.PublishedTag (tag, survivor_id, tag_revision, removed) -> SurvivorTag dict."""
    return tag_to_msg_dict(pt.tag, frame_id, pt.tag_revision, pt.removed, pt.survivor_id)


def tag_array_to_msg_dict(published: Iterable[Any], frame_id: str, stamp_ns: int, map_rev: int, reg_rev: int,
                          registration_valid: bool, is_snapshot: bool) -> Dict[str, Any]:
    return {"header": {"stamp": ns_to_stamp(stamp_ns), "frame_id": frame_id},
            "tags": [published_tag_to_msg_dict(p, frame_id) for p in published], "map_rev": int(map_rev),
            "reg_rev": int(reg_rev), "registration_valid": bool(registration_valid), "is_snapshot": bool(is_snapshot)}


def tag_result_to_msg_dict(tr: TagResult, frame_id: str, stamp_ns: int, map_rev: int = 0, reg_rev: int = 0,
                           tag_revision: int = 0) -> Dict[str, Any]:
    """TagResult (what final_tags returns: fewer fields than Tag) -> SurvivorTag dict. Fields a TagResult does not
    have are flagged unknown (has_* False, NaN, -1) or zero counts; localisation_quality is its top_prob."""
    p = None if tr.p_map_m is None else np.asarray(tr.p_map_m, dtype=float)
    d = tag_to_msg_dict(
        Tag(tr.survivor_id, tr.status, tr.cell, None, (), None if p is None else np.array([p[0], p[1], p[2] if len(p) > 2
                                                                                           else 0.0]), None, None, None,
            None, False, 0.0, 0.0 if tr.top_prob is None else float(tr.top_prob), 0, 0.0, tr.first_stamp_ns,
            tr.first_stamp_ns, tr.first_stamp_ns, stamp_ns, map_rev, reg_rev, None, (), tr.confirm_stamp_ns,
            tr.cell_changes), frame_id, tag_revision)
    if p is None or len(p) < 3:
        d["position_map"]["z"] = NAN
    return d


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

KIND_RECORD, KIND_SNAPSHOT = 1, 2  # = TelemetryPacket.KIND_*, = output.MSG_TAG_UPDATE / MSG_SNAPSHOT


def telemetry_packet_to_msg_dict(data: bytes, stamp_ns: int, frame_id: str = "") -> Dict[str, Any]:
    """Codec bytes -> TelemetryPacket dict. The bytes are decoded once to check length, type and CRC (the wire format
    itself stays in survivor_tagging.output); a corrupt packet raises AdapterError(IMAGE_FORMAT_INVALID)."""
    if not data:
        raise AdapterError(FailureReason.IMAGE_FORMAT_INVALID, "empty telemetry packet")
    kind = data[0]
    try:
        if kind == MSG_TAG_UPDATE:
            decode_tag_record(bytes(data))
        elif kind == MSG_SNAPSHOT:
            decode_snapshot(bytes(data))
        else:
            raise AdapterError(FailureReason.IMAGE_FORMAT_INVALID, f"unknown telemetry message type {kind:#04x}")
    except TelemetryDecodeError as exc:
        raise AdapterError(FailureReason.IMAGE_FORMAT_INVALID, f"telemetry decode failed: {exc}") from None
    return {"header": {"stamp": ns_to_stamp(stamp_ns), "frame_id": frame_id}, "kind": kind, "data": list(bytes(data))}


def telemetry_bytes_from_msg(msg: Any) -> bytes:
    """TelemetryPacket (kind, data) -> codec bytes, checking that kind matches the first byte."""
    data = bytes(msg.data)
    if not data or data[0] != int(msg.kind) or int(msg.kind) not in (KIND_RECORD, KIND_SNAPSHOT):
        raise AdapterError(FailureReason.IMAGE_FORMAT_INVALID, f"kind {msg.kind} does not match the data")
    return data


# ---------------------------------------------------------------------------
# Corrections and registration
# ---------------------------------------------------------------------------


def _transform_msg_to_matrix(t: Any) -> np.ndarray:
    tr, ro = t.translation, t.rotation
    q = np.array([float(ro.w), float(ro.x), float(ro.y), float(ro.z)])
    v = np.array([float(tr.x), float(tr.y), float(tr.z)])
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(v))):
        raise AdapterError(FailureReason.CORRECTION_INVALID, "frame_transform contains NaN or inf")
    n = float(np.linalg.norm(q))
    if abs(n - 1.0) > QUAT_NORM_TOL:
        raise AdapterError(FailureReason.CORRECTION_INVALID, f"frame_transform quaternion norm {n:.6f}")
    w, x, y, z = q / n
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, v
    return m


def correction_from_msg(msg: Any, cfg: AdapterConfig) -> PoseCorrection:
    """CorrectionEvent -> library PoseCorrection, both modes. history: the corrected poses are converted like live poses
    (frame check, FRD -> FLU, covariance) with pose_revision = msg.revision; stamps must increase strictly; if
    range_start / range_end are set they must equal the first / last pose stamp. frame_correction: no poses allowed,
    frame_transform -> 4x4 T_new_old. Violations raise AdapterError(CORRECTION_INVALID / FRAME_MISMATCH / ...)."""
    cfg.require("map_frame_id", "body_frame_id", "body_convention", "clock_domain")
    stamp = header_stamp_ns(msg.header)
    rev = msg.revision
    if isinstance(rev, bool) or not isinstance(rev, (int, np.integer)) or rev < 1:
        raise AdapterError(FailureReason.CORRECTION_INVALID, f"revision {rev!r} must be an int >= 1")
    mode = str(msg.mode)
    if mode not in ("history", "frame_correction"):
        raise AdapterError(FailureReason.CORRECTION_INVALID, f"mode {mode!r} must be history or frame_correction")
    if _frame_id(msg.header) != str(cfg.map_frame_id).lstrip("/"):
        raise AdapterError(FailureReason.FRAME_MISMATCH, f"correction frame {_frame_id(msg.header)!r}, configured "
                                                         f"{cfg.map_frame_id!r}")
    if mode == "history":
        if not msg.corrected_poses:
            raise AdapterError(FailureReason.CORRECTION_INVALID, "history correction without corrected_poses")
        poses = tuple(pose_from_msg(p, cfg, int(rev), "slam_correction") for p in msg.corrected_poses)
        if any(b.stamp_ns <= a.stamp_ns for a, b in zip(poses, poses[1:])):
            raise AdapterError(FailureReason.CORRECTION_INVALID, "corrected poses not strictly increasing in time")
        for name, expected in (("range_start", poses[0].stamp_ns), ("range_end", poses[-1].stamp_ns)):
            s = getattr(msg, name)
            if not (s.sec == 0 and s.nanosec == 0) and stamp_to_ns(s) != expected:
                raise AdapterError(FailureReason.CORRECTION_INVALID, f"{name} differs from the corrected poses")
        return PoseCorrection(int(rev), "history", stamp, cfg.clock_domain, poses, None)
    if msg.corrected_poses:
        raise AdapterError(FailureReason.CORRECTION_INVALID, "frame_correction must not carry corrected_poses")
    return PoseCorrection(int(rev), "frame_correction", stamp, cfg.clock_domain, (),
                          _transform_msg_to_matrix(msg.frame_transform))


def registration_from_request(req: Any, current: GridRegistration) -> GridRegistration:
    """SetGridRegistration request -> new GridRegistration (a copy of `current` with every request field applied;
    boundary_rule is kept). Rules: finite origin / yaw, cell_size_m > 0, index_base 0 or 1, numbering known, extent 0 =
    unknown (None); version < current.version -> CORRECTION_STALE (as Layer5Pipeline.process_registration); an equal
    version is passed through (the pipeline decides duplicate or conflict). Nothing is taken from defaults."""
    origin = [float(v) for v in req.origin_in_map_m]
    if len(origin) != 2 or not all(math.isfinite(v) for v in origin) or not math.isfinite(float(req.yaw_map_to_grid_rad)):
        raise AdapterError(FailureReason.CORRECTION_INVALID, "origin / yaw must be finite (2 values for the origin)")
    if not (math.isfinite(float(req.cell_size_m)) and float(req.cell_size_m) > 0):
        raise AdapterError(FailureReason.CORRECTION_INVALID, f"cell_size_m {req.cell_size_m!r} must be > 0")
    if int(req.index_base) not in (0, 1):
        raise AdapterError(FailureReason.CORRECTION_INVALID, f"index_base {req.index_base!r} must be 0 or 1")
    if str(req.numbering) not in NUMBERINGS:
        raise AdapterError(FailureReason.CORRECTION_INVALID, f"numbering {req.numbering!r} not in {NUMBERINGS}")
    version = req.version
    if isinstance(version, bool) or not isinstance(version, (int, np.integer)) or version < 0:
        raise AdapterError(FailureReason.CORRECTION_INVALID, f"version {version!r} must be an int >= 0")
    if version < current.version:
        raise AdapterError(FailureReason.CORRECTION_STALE, f"registration version {version} < current "
                                                           f"{current.version}")
    nx, ny = int(req.n_cells_x), int(req.n_cells_y)
    return replace(current, origin_in_map_m=(origin[0], origin[1]), yaw_rad=float(req.yaw_map_to_grid_rad),
                   flip_x=bool(req.axis_flip_x), flip_y=bool(req.axis_flip_y), cell_size_m=float(req.cell_size_m),
                   n_cells_x=nx or None, n_cells_y=ny or None, index_base=int(req.index_base),
                   numbering=str(req.numbering), version=int(version), valid=bool(req.valid), missing=())


def registration_response_dict(outcome: Any, current_version: int) -> Dict[str, Any]:
    """CorrectionOutcome of process_registration -> SetGridRegistration response. A duplicate (applied False, reason None)
    is accepted=False with reason 'duplicate'."""
    reason = "" if outcome.applied else (f"{outcome.reason.value}: {outcome.detail}" if outcome.reason is not None
                                         else outcome.detail or "duplicate")
    return {"accepted": bool(outcome.applied), "reason": reason, "current_version": int(current_version)}


def commit_request_to_args(req: Any) -> Tuple[str, Optional[float]]:
    """CommitFinalTags request -> (policy, threshold or None). NaN threshold = library default. A threshold must lie in
    (0, 1] and only belongs to expected_value."""
    policy = str(req.policy)
    if policy not in COMMIT_POLICIES:
        raise AdapterError(FailureReason.CONFIG_MISSING, f"policy {policy!r} not in {COMMIT_POLICIES}")
    th = float(req.threshold)
    if math.isnan(th):
        return policy, None
    if policy != "expected_value":
        raise AdapterError(FailureReason.CONFIG_MISSING, f"a threshold is only valid for expected_value, not {policy}")
    if not 0.0 < th <= 1.0:
        raise AdapterError(FailureReason.CONFIG_MISSING, f"threshold {th} outside (0, 1]")
    return policy, th


def commit_response_dict(tags: Sequence[TagResult], policy: str, threshold: Optional[float], frame_id: str,
                         stamp_ns: int, map_rev: int = 0, reg_rev: int = 0) -> Dict[str, Any]:
    return {"success": True, "reason": "", "policy_used": policy, "threshold_used": NAN if threshold is None else
            float(threshold), "tags": [tag_result_to_msg_dict(t, frame_id, stamp_ns, map_rev, reg_rev) for t in tags]}
