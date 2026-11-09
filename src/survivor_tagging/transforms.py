"""Layer 2: time-aligned coordinate transformation (spec Part 6; architecture.md "Layer 2").

Pure library code: no simulator, baseline or ROS imports.

Conventions (docs/design/conventions.md):
- Quaternions are (w, x, y, z), unit norm, Hamilton product, active rotation of vectors (v' = q v q*).
  quat_to_matrix(q) is the body-to-parent rotation R, so p_parent = R p_body + t.
- Rigid transforms are 4x4 matrices T_<to>_<from>; compose(A, B) = A @ B.
- Covariance perturbations: a pose (parent = map) is perturbed as t' = t + dt and R' = Exp(dtheta) R with dt and
  dtheta in the MAP frame, order (x, y, z, rx, ry, rz) (the ROS PoseWithCovariance fixed-axis convention). The
  extrinsic T_body_camera is perturbed the same way in the BODY frame.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config, ConfigError
from .geometry import FloorPlane
from .types import DepthMeasurement, Detection, FailureReason, PoseSample, SurvivorObservation

NS_PER_S = 1_000_000_000
RIGID_TOL = 1e-6


# ---------------------------------------------------------------------------
# Quaternions and rigid transforms
# ---------------------------------------------------------------------------


def quat_normalize(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if not math.isfinite(n) or n == 0.0:
        raise ValueError("quaternion has zero or non-finite norm")
    return q / n


def quat_multiply(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Hamilton product a * b (apply b first, then a)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion with w >= 0 (Shepperd's method)."""
    r = np.asarray(r, dtype=float)
    tr = float(np.trace(r))
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = quat_normalize(q)
    return -q if q[0] < 0 else q


def quat_slerp(q0: Sequence[float], q1: Sequence[float], alpha: float) -> np.ndarray:
    """Spherical linear interpolation along the shortest arc; alpha outside [0, 1] extrapolates."""
    a, b = quat_normalize(q0), quat_normalize(q1)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b, dot = -b, -dot
    if dot > 0.9995:  # nearly parallel: normalised lerp is accurate and stable
        return quat_normalize(a + alpha * (b - a))
    theta = math.acos(min(dot, 1.0))
    s = math.sin(theta)
    return quat_normalize((math.sin((1 - alpha) * theta) * a + math.sin(alpha * theta) * b) / s)


def skew(v: Sequence[float]) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotvec_to_matrix(v: Sequence[float]) -> np.ndarray:
    """Exp map (Rodrigues)."""
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3) + skew(v)
    k = skew(v / th)
    return np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * (k @ k)


def make_transform(r: np.ndarray, t: Sequence[float]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = t
    return m


def invert_transform(t: np.ndarray) -> np.ndarray:
    r = t[:3, :3]
    return make_transform(r.T, -r.T @ t[:3, 3])


def compose(*ts: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    for t in ts:
        out = out @ t
    return out


def apply_transform(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply to a point (3,) or points (N, 3)."""
    p = np.asarray(p, dtype=float)
    return p @ t[:3, :3].T + t[:3, 3]


def is_rigid(t: np.ndarray, tol: float = RIGID_TOL) -> bool:
    t = np.asarray(t, dtype=float)
    if t.shape != (4, 4) or not np.all(np.isfinite(t)):
        return False
    r = t[:3, :3]
    return (np.allclose(t[3], [0, 0, 0, 1], atol=tol) and np.allclose(r.T @ r, np.eye(3), atol=tol)
            and np.linalg.det(r) > 0)


def pose_matrix(pose: PoseSample) -> np.ndarray:
    return make_transform(quat_to_matrix(pose.orientation_wxyz), pose.position_m)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Extrinsic:
    T_body_camera: np.ndarray
    version: int
    cov6: Optional[np.ndarray] = None  # (dt_body, dphi_body)


def extrinsic_from_config(cfg: Config) -> Tuple[Optional[Extrinsic], Optional[FailureReason]]:
    """EXTRINSIC_MISSING if the transform or its version is unset; CALIBRATION_INVALID if not a proper rigid
    transform or the covariance is not symmetric with a non-negative diagonal."""
    t, ver = cfg.value("extr.T_body_camera"), cfg.value("extr.version")
    if t is None or ver is None:
        return None, FailureReason.EXTRINSIC_MISSING
    t = np.asarray(t, dtype=float)
    if not is_rigid(t):
        return None, FailureReason.CALIBRATION_INVALID
    cov = cfg.value("extr.T_body_camera_cov")
    if cov is not None:
        cov = np.asarray(cov, dtype=float)
        if cov.shape != (6, 6) or not np.allclose(cov, cov.T) or np.any(np.diag(cov) < 0):
            return None, FailureReason.CALIBRATION_INVALID
    return Extrinsic(t, int(ver), cov), None


@dataclass(frozen=True)
class Layer2Params:
    pose_buffer_duration_s: float
    pose_buffer_max_samples: int
    max_interp_gap_s: float
    allow_extrapolation_s: float
    max_detection_age_s: float
    max_rgb_depth_skew_s: Optional[float]  # None: the skew check reports CONFIG_MISSING
    max_pose_wait_s: float
    pending_queue_max: int
    camera_to_pose_offset_ns: Optional[int]  # None: different clock domains -> CLOCK_DOMAIN_UNKNOWN

    @classmethod
    def from_config(cls, cfg: Config) -> "Layer2Params":
        v = cfg.value
        if v("l2.max_interp_gap_s") is None:
            raise ConfigError(f"{FailureReason.CONFIG_MISSING.value}: l2.max_interp_gap_s")
        return cls(v("l2.pose_buffer_duration_s"), v("l2.pose_buffer_max_samples"), v("l2.max_interp_gap_s"),
                   v("l2.allow_extrapolation_s"), v("l2.max_detection_age_s"), v("l2.max_rgb_depth_skew_s"),
                   v("l2.max_pose_wait_s"), v("l2.pending_queue_max"), v("clock.camera_to_pose_offset_ns"))


# ---------------------------------------------------------------------------
# Pose buffer and capture-time lookup
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PoseLookup:
    pose: Optional[PoseSample]  # pose at the capture time (interpolated)
    failure: Optional[FailureReason] = None
    stamps_ns: Optional[Tuple[int, int]] = None
    alpha: Optional[float] = None
    revision: Optional[int] = None
    warnings: Tuple[FailureReason, ...] = ()


def interpolate_pose(p0: PoseSample, p1: PoseSample, t_ns: int) -> Tuple[PoseSample, float]:
    """Linear position, slerp orientation, linear covariance (if both have one). alpha > 1 extrapolates."""
    alpha = 0.0 if p1.stamp_ns == p0.stamp_ns else (t_ns - p0.stamp_ns) / (p1.stamp_ns - p0.stamp_ns)
    pos = p0.position_m + alpha * (p1.position_m - p0.position_m)
    q = quat_slerp(p0.orientation_wxyz, p1.orientation_wxyz, alpha)
    cov = None
    if p0.covariance is not None and p1.covariance is not None:
        a = min(max(alpha, 0.0), 1.0)
        cov = (1 - a) * p0.covariance + a * p1.covariance
    pose = PoseSample(t_ns, p0.clock_domain, p0.frame_id, p0.child_frame_id, pos, q, True,
                      max(p0.pose_revision, p1.pose_revision), "interpolated", cov)
    return pose, alpha


class PoseBuffer:
    """Time-sorted, bounded pose history. Accepts out-of-order poses; a pose with an existing stamp replaces it
    only if its pose_revision is not older."""

    def __init__(self, duration_s: float, max_samples: int):
        self.duration_ns = int(round(duration_s * NS_PER_S))
        self.max_samples = max_samples
        self._stamps: List[int] = []
        self._poses: List[PoseSample] = []

    def __len__(self) -> int:
        return len(self._poses)

    @property
    def newest_ns(self) -> Optional[int]:
        return self._stamps[-1] if self._stamps else None

    @property
    def oldest_ns(self) -> Optional[int]:
        return self._stamps[0] if self._stamps else None

    @property
    def clock_domain(self) -> Optional[str]:
        return self._poses[-1].clock_domain if self._poses else None

    def add(self, pose: PoseSample) -> Optional[FailureReason]:
        if pose.failure() is not None:
            return FailureReason.POSE_INVALID
        if self._poses and pose.clock_domain != self.clock_domain:
            return FailureReason.CLOCK_DOMAIN_UNKNOWN
        i = bisect.bisect_left(self._stamps, pose.stamp_ns)
        if i < len(self._stamps) and self._stamps[i] == pose.stamp_ns:
            if pose.pose_revision >= self._poses[i].pose_revision:
                self._poses[i] = pose
        else:
            self._stamps.insert(i, pose.stamp_ns)
            self._poses.insert(i, pose)
        self._prune()
        return None

    def _prune(self) -> None:
        cutoff = self._stamps[-1] - self.duration_ns
        k = bisect.bisect_left(self._stamps, cutoff)
        k = max(k, len(self._stamps) - self.max_samples)
        if k > 0:
            del self._stamps[:k]
            del self._poses[:k]

    def lookup(self, t_ns: int, max_gap_s: float, allow_extrapolation_s: float = 0.0) -> PoseLookup:
        """Pose at t_ns from the two bracketing poses. Never returns the latest pose as a substitute:
        t after the newest pose (beyond allow_extrapolation_s) or before the oldest -> POSE_UNAVAILABLE_AT_CAPTURE;
        bracketing gap > max_gap_s -> POSE_GAP_TOO_LARGE."""
        if not self._stamps or t_ns < self._stamps[0]:
            return PoseLookup(None, FailureReason.POSE_UNAVAILABLE_AT_CAPTURE)
        max_gap_ns = max_gap_s * NS_PER_S
        if t_ns > self._stamps[-1]:
            if t_ns - self._stamps[-1] > allow_extrapolation_s * NS_PER_S or len(self._stamps) < 2:
                return PoseLookup(None, FailureReason.POSE_UNAVAILABLE_AT_CAPTURE)
            p0, p1 = self._poses[-2], self._poses[-1]
            if p1.stamp_ns - p0.stamp_ns > max_gap_ns:
                return PoseLookup(None, FailureReason.POSE_GAP_TOO_LARGE)
        else:
            i = bisect.bisect_left(self._stamps, t_ns)
            if self._stamps[i] == t_ns:
                p = self._poses[i]
                return PoseLookup(p, None, (p.stamp_ns, p.stamp_ns), 0.0, p.pose_revision)
            p0, p1 = self._poses[i - 1], self._poses[i]
            if p1.stamp_ns - p0.stamp_ns > max_gap_ns:
                return PoseLookup(None, FailureReason.POSE_GAP_TOO_LARGE)
        pose, alpha = interpolate_pose(p0, p1, t_ns)
        warn = (FailureReason.SLAM_CORRECTION_APPLIED,) if p0.pose_revision != p1.pose_revision else ()
        return PoseLookup(pose, None, (p0.stamp_ns, p1.stamp_ns), alpha, pose.pose_revision, warn)


# ---------------------------------------------------------------------------
# Pending queue (observations waiting for a newer pose)
# ---------------------------------------------------------------------------


@dataclass
class PendingItem:
    capture_pose_ns: int  # capture time in the pose clock
    deadline_ns: int  # give up after this (arrival clock)
    payload: Any


class PendingQueue:
    """Bounded FIFO. When full, the oldest item is evicted and returned to the caller."""

    def __init__(self, max_items: int):
        self.max_items = max_items
        self.items: List[PendingItem] = []

    def __len__(self) -> int:
        return len(self.items)

    def push(self, item: PendingItem) -> Optional[PendingItem]:
        self.items.append(item)
        return self.items.pop(0) if len(self.items) > self.max_items else None

    def pop_ready(self, newest_pose_ns: Optional[int]) -> List[PendingItem]:
        if newest_pose_ns is None:
            return []
        ready = [it for it in self.items if it.capture_pose_ns <= newest_pose_ns]
        self.items = [it for it in self.items if it.capture_pose_ns > newest_pose_ns]
        return ready

    def pop_expired(self, now_ns: int) -> List[PendingItem]:
        expired = [it for it in self.items if now_ns > it.deadline_ns]
        self.items = [it for it in self.items if now_ns <= it.deadline_ns]
        return expired

    def drain(self) -> List[PendingItem]:
        out, self.items = self.items, []
        return out


# ---------------------------------------------------------------------------
# Timestamp checks
# ---------------------------------------------------------------------------


def capture_in_pose_clock(stamp_ns: int, image_domain: str, pose_domain: Optional[str],
                          offset_ns: Optional[int]) -> Tuple[Optional[int], Optional[FailureReason]]:
    """Same domain: unchanged. Different domain: stamp + clock.camera_to_pose_offset_ns if configured, else
    CLOCK_DOMAIN_UNKNOWN. With no pose yet, the image domain is assumed to be the pose domain (checked later)."""
    if pose_domain is None or image_domain == pose_domain:
        return stamp_ns, None
    if offset_ns is None:
        return None, FailureReason.CLOCK_DOMAIN_UNKNOWN
    return stamp_ns + offset_ns, None


def check_timestamps(det: Detection, depth: DepthMeasurement, arrival_ns: int, params: Layer2Params,
                     pose_domain: Optional[str]) -> Tuple[Optional[int], Optional[FailureReason]]:
    """Returns (capture time in the pose clock, failure). Order: RGB/depth skew (CONFIG_MISSING if its tolerance is
    unset), clock domain, staleness. ASSUMPTION: arrival_ns is in the pose clock domain."""
    if params.max_rgb_depth_skew_s is None:
        return None, FailureReason.CONFIG_MISSING
    if det.clock_domain != depth.clock_domain:
        return None, FailureReason.CLOCK_DOMAIN_UNKNOWN
    if abs(det.image_stamp_ns - depth.depth_stamp_ns) > params.max_rgb_depth_skew_s * NS_PER_S:
        return None, FailureReason.RGB_DEPTH_TIME_MISMATCH
    t, fail = capture_in_pose_clock(det.image_stamp_ns, det.clock_domain, pose_domain,
                                    params.camera_to_pose_offset_ns)
    if fail is not None:
        return None, fail
    if arrival_ns - t > params.max_detection_age_s * NS_PER_S:
        return None, FailureReason.DETECTION_STALE
    return t, None


# ---------------------------------------------------------------------------
# Camera -> map with covariance
# ---------------------------------------------------------------------------


def camera_to_map(p_cam: np.ndarray, cov_cam: Optional[np.ndarray], pose: PoseSample,
                  extr: Extrinsic) -> Tuple[np.ndarray, np.ndarray]:
    """P_map = R_mb (R_bc p_c + t_bc) + t_mb, i.e. T_map_body(t) * T_body_camera * P_camera.

    Covariance (independent sources, first order):
      measurement   J = R_mb R_bc                       C += J C_cam J^T
      pose (6x6)    J = [I, -[R_mb q]x], q = R_bc p_c + t_bc   C += J C_pose J^T  (map-frame perturbation)
      extrinsic     J = [R_mb, -R_mb [R_bc p_c]x]       C += J C_ext J^T   (body-frame perturbation)
    A missing pose or extrinsic covariance contributes nothing.
    """
    r_mb, t_mb = quat_to_matrix(pose.orientation_wxyz), pose.position_m
    r_bc, t_bc = extr.T_body_camera[:3, :3], extr.T_body_camera[:3, 3]
    c_b = r_bc @ p_cam
    q = c_b + t_bc
    p_map = r_mb @ q + t_mb
    cov = np.zeros((3, 3))
    if cov_cam is not None:
        j = r_mb @ r_bc
        cov += j @ cov_cam @ j.T
    if pose.covariance is not None:
        j = np.hstack([np.eye(3), -skew(r_mb @ q)])
        cov += j @ pose.covariance @ j.T
    if extr.cov6 is not None:
        j = np.hstack([r_mb, -r_mb @ skew(c_b)])
        cov += j @ extr.cov6 @ j.T
    return p_map, 0.5 * (cov + cov.T)


# ---------------------------------------------------------------------------
# Floor plane in the camera frame
# ---------------------------------------------------------------------------


def floor_plane_in_camera(pose: PoseSample, extr: Extrinsic, floor_z_m: float, margin_m: float,
                          pose_sigma_k: float = 0.0) -> FloorPlane:
    """Floor z = floor_z_m (map frame, map +z up) expressed in the camera frame at the capture-time pose.

    With T_map_camera = T_map_body(t) T_body_camera = (R, t):  z_map(p_c) = (R^T e_z) . p_c + t_z, so
    height_above_floor(p_c) = n . p_c - (floor_z_m - t_z) with n = R^T e_z (third row of R).
    Margin: margin_m + pose_sigma_k * sigma_z, sigma_z = sqrt(pose covariance[2, 2]) (0 if no covariance): a pose
    that is uncertain in height widens the band that is treated as floor. Attitude uncertainty is not included.
    """
    t_mc = pose_matrix(pose) @ extr.T_body_camera
    r, t = t_mc[:3, :3], t_mc[:3, 3]
    sigma_z = math.sqrt(max(float(pose.covariance[2, 2]), 0.0)) if pose.covariance is not None else 0.0
    return FloorPlane(np.array(r[2, :], dtype=float), float(floor_z_m - t[2]), margin_m + pose_sigma_k * sigma_z)


# ---------------------------------------------------------------------------
# Layer 2 state
# ---------------------------------------------------------------------------


class Layer2:
    """Pose buffer + pending queue + transform. The caller owns the observations; Layer 2 fills their fields."""

    def __init__(self, params: Layer2Params, extrinsic: Optional[Extrinsic],
                 extrinsic_failure: Optional[FailureReason] = None):
        self.params = params
        self.extrinsic = extrinsic
        self.extrinsic_failure = extrinsic_failure if extrinsic is None else None
        if extrinsic is None and self.extrinsic_failure is None:
            self.extrinsic_failure = FailureReason.EXTRINSIC_MISSING
        self.buffer = PoseBuffer(params.pose_buffer_duration_s, params.pose_buffer_max_samples)
        self.queue = PendingQueue(params.pending_queue_max)

    def add_pose(self, pose: PoseSample) -> Optional[FailureReason]:
        return self.buffer.add(pose)

    def check(self, det: Detection, depth: DepthMeasurement,
              arrival_ns: int) -> Tuple[Optional[int], Optional[FailureReason]]:
        return check_timestamps(det, depth, arrival_ns, self.params, self.buffer.clock_domain)

    def waiting_for_pose(self, capture_pose_ns: int) -> bool:
        """True if no pose at or after the capture time has arrived yet (so a later pose may still come)."""
        newest = self.buffer.newest_ns
        return newest is None or capture_pose_ns > newest + self.params.allow_extrapolation_s * NS_PER_S

    def lookup(self, capture_pose_ns: int) -> PoseLookup:
        """Capture-time pose (never the latest pose as a substitute); EXTRINSIC_MISSING/CALIBRATION_INVALID first."""
        if self.extrinsic is None:
            return PoseLookup(None, self.extrinsic_failure)
        return self.buffer.lookup(capture_pose_ns, self.params.max_interp_gap_s, self.params.allow_extrapolation_s)

    def transform(self, obs: SurvivorObservation, capture_pose_ns: int) -> Optional[FailureReason]:
        """Fill obs Layer 2 fields from the pose at capture_pose_ns. Returns the failure (also stored in obs)."""
        return self.apply(obs, self.lookup(capture_pose_ns))

    def apply(self, obs: SurvivorObservation, lk: PoseLookup) -> Optional[FailureReason]:
        """Fill obs Layer 2 fields from an already looked-up capture-time pose."""
        if lk.failure is not None:
            obs.failure = lk.failure
            return lk.failure
        obs.p_map_m, obs.cov_map_m2 = camera_to_map(obs.p_camera_m, obs.cov_camera_m2, lk.pose, self.extrinsic)
        obs.camera_map_m = apply_transform(pose_matrix(lk.pose), self.extrinsic.T_body_camera[:3, 3])
        obs.pose_stamps_ns, obs.interp_alpha, obs.pose_revision = lk.stamps_ns, lk.alpha, lk.revision
        obs.extrinsic_version = self.extrinsic.version
        obs.warnings.extend(lk.warnings)
        return None
