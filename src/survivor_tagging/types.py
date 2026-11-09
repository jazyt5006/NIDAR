"""Data types, status enums and failure-reason codes (architecture.md sections b and d).

No layer logic lives here. Units: metres, radians, integer nanoseconds for timestamps.
Frame and time conventions: docs/design/conventions.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum, IntFlag
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

Cell = Tuple[int, int]


class Severity(str, Enum):
    REJECT = "REJECT"  # observation discarded
    BLOCK = "BLOCK"  # processing continues, CONFIRMED is forbidden
    DEGRADE = "DEGRADE"  # kept with reduced quality / provisional
    EVENT = "EVENT"  # informational, not a failure


class TagStatus(IntEnum):
    # Integer values are the telemetry status codes (architecture.md section e).
    CANDIDATE = 0
    PROVISIONAL = 1
    CONFIRMED = 2
    UPDATED = 3
    UNRELIABLE = 4


class FailureReason(str, Enum):
    # Spec Part 10 rows
    DEPTH_NO_VALID = "DEPTH_NO_VALID"
    DEPTH_TOO_FEW_PIXELS = "DEPTH_TOO_FEW_PIXELS"
    DEPTH_BACKGROUND_DOMINANT = "DEPTH_BACKGROUND_DOMINANT"
    DEPTH_AMBIGUOUS_CLUSTERS = "DEPTH_AMBIGUOUS_CLUSTERS"
    DETECTION_STALE = "DETECTION_STALE"
    RGB_DEPTH_TIME_MISMATCH = "RGB_DEPTH_TIME_MISMATCH"
    POSE_UNAVAILABLE_AT_CAPTURE = "POSE_UNAVAILABLE_AT_CAPTURE"
    EXTRINSIC_MISSING = "EXTRINSIC_MISSING"
    CAMERA_ORIENTATION_UNKNOWN = "CAMERA_ORIENTATION_UNKNOWN"
    GRID_UNREGISTERED = "GRID_UNREGISTERED"
    CELL_BOUNDARY_AMBIGUOUS = "CELL_BOUNDARY_AMBIGUOUS"
    TRACK_DUPLICATE_MERGED = "TRACK_DUPLICATE_MERGED"
    TRACK_ID_REASSOCIATED = "TRACK_ID_REASSOCIATED"
    SLAM_CORRECTION_APPLIED = "SLAM_CORRECTION_APPLIED"
    TRACK_NOT_OBSERVED = "TRACK_NOT_OBSERVED"
    CALIBRATION_INVALID = "CALIBRATION_INVALID"
    OUT_OF_GRID = "OUT_OF_GRID"
    # Additional diagnostic codes
    BBOX_INVALID = "BBOX_INVALID"
    BBOX_OUTSIDE_IMAGE = "BBOX_OUTSIDE_IMAGE"
    BBOX_TRUNCATED = "BBOX_TRUNCATED"  # box touches the image border; person may be cut off
    DEPTH_SCALE_INVALID = "DEPTH_SCALE_INVALID"
    INTRINSICS_MISSING = "INTRINSICS_MISSING"
    DEPTH_FRAME_MISMATCH = "DEPTH_FRAME_MISMATCH"
    CLOCK_DOMAIN_UNKNOWN = "CLOCK_DOMAIN_UNKNOWN"
    POSE_INVALID = "POSE_INVALID"
    POSE_GAP_TOO_LARGE = "POSE_GAP_TOO_LARGE"
    GROUND_REF_UNAVAILABLE = "GROUND_REF_UNAVAILABLE"
    OBSERVATION_OUTLIER = "OBSERVATION_OUTLIER"
    TRACK_INCONSISTENT = "TRACK_INCONSISTENT"
    ASSOCIATION_AMBIGUOUS = "ASSOCIATION_AMBIGUOUS"  # observation fits two tracks / two tracks too close
    FLOOR_UNKNOWN = "FLOOR_UNKNOWN"  # floor removal requested but no floor height configured
    EXTENDED_CLUSTER = "EXTENDED_CLUSTER"  # footprint refinement applied (partially visible extended body)
    CORRECTION_SOURCE_UNAVAILABLE = "CORRECTION_SOURCE_UNAVAILABLE"
    CONFIG_MISSING = "CONFIG_MISSING"
    CAPACITY_EXCEEDED = "CAPACITY_EXCEEDED"


_R, _B, _D, _E = Severity.REJECT, Severity.BLOCK, Severity.DEGRADE, Severity.EVENT

FAILURE_SEVERITY: Dict[FailureReason, Severity] = {
    FailureReason.DEPTH_NO_VALID: _R,
    FailureReason.DEPTH_TOO_FEW_PIXELS: _D,
    FailureReason.DEPTH_BACKGROUND_DOMINANT: _R,
    FailureReason.DEPTH_AMBIGUOUS_CLUSTERS: _R,
    FailureReason.DETECTION_STALE: _R,
    FailureReason.RGB_DEPTH_TIME_MISMATCH: _R,
    FailureReason.POSE_UNAVAILABLE_AT_CAPTURE: _B,
    FailureReason.EXTRINSIC_MISSING: _B,
    FailureReason.CAMERA_ORIENTATION_UNKNOWN: _D,
    FailureReason.GRID_UNREGISTERED: _B,
    FailureReason.CELL_BOUNDARY_AMBIGUOUS: _D,
    FailureReason.TRACK_DUPLICATE_MERGED: _E,
    FailureReason.TRACK_ID_REASSOCIATED: _E,
    FailureReason.SLAM_CORRECTION_APPLIED: _E,
    FailureReason.TRACK_NOT_OBSERVED: _E,
    FailureReason.CALIBRATION_INVALID: _B,
    FailureReason.OUT_OF_GRID: _D,
    FailureReason.BBOX_INVALID: _R,
    FailureReason.BBOX_OUTSIDE_IMAGE: _R,
    FailureReason.BBOX_TRUNCATED: _D,
    FailureReason.DEPTH_SCALE_INVALID: _R,
    FailureReason.INTRINSICS_MISSING: _R,
    FailureReason.DEPTH_FRAME_MISMATCH: _R,
    FailureReason.CLOCK_DOMAIN_UNKNOWN: _R,
    FailureReason.POSE_INVALID: _B,
    FailureReason.POSE_GAP_TOO_LARGE: _B,
    FailureReason.GROUND_REF_UNAVAILABLE: _D,
    FailureReason.OBSERVATION_OUTLIER: _R,
    FailureReason.TRACK_INCONSISTENT: _D,
    FailureReason.ASSOCIATION_AMBIGUOUS: _D,
    FailureReason.FLOOR_UNKNOWN: _D,
    FailureReason.EXTENDED_CLUSTER: _D,
    FailureReason.CORRECTION_SOURCE_UNAVAILABLE: _D,
    FailureReason.CONFIG_MISSING: _B,
    FailureReason.CAPACITY_EXCEEDED: _E,
}

# Spec Part 10 "Failure" column text -> the code that handles it.
SPEC_PART10_CODES: Dict[str, FailureReason] = {
    "No valid depth": FailureReason.DEPTH_NO_VALID,
    "Too few valid depth pixels": FailureReason.DEPTH_TOO_FEW_PIXELS,
    "Background dominates bounding box": FailureReason.DEPTH_BACKGROUND_DOMINANT,
    "Multiple plausible depth clusters": FailureReason.DEPTH_AMBIGUOUS_CLUSTERS,
    "Detection timestamp stale": FailureReason.DETECTION_STALE,
    "Depth and RGB timestamps mismatch": FailureReason.RGB_DEPTH_TIME_MISMATCH,
    "Historical pose unavailable": FailureReason.POSE_UNAVAILABLE_AT_CAPTURE,
    "Camera extrinsic missing": FailureReason.EXTRINSIC_MISSING,
    "Camera orientation unknown": FailureReason.CAMERA_ORIENTATION_UNKNOWN,
    "Grid registration unavailable": FailureReason.GRID_UNREGISTERED,
    "Position near cell boundary": FailureReason.CELL_BOUNDARY_AMBIGUOUS,
    "Duplicate detection": FailureReason.TRACK_DUPLICATE_MERGED,
    "Detector tracking ID changes": FailureReason.TRACK_ID_REASSOCIATED,
    "SLAM correction": FailureReason.SLAM_CORRECTION_APPLIED,
    "Survivor leaves camera view": FailureReason.TRACK_NOT_OBSERVED,
    "Invalid calibration": FailureReason.CALIBRATION_INVALID,
    "Position outside official grid": FailureReason.OUT_OF_GRID,
}


def _vec(value, shape: Tuple[int, ...], name: str) -> np.ndarray:
    """Read-only float64 array of the given shape; raises ValueError on a shape mismatch."""
    arr = np.array(value, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {arr.shape}")
    arr.setflags(write=False)
    return arr


def _opt_vec(value, shape: Tuple[int, ...], name: str) -> Optional[np.ndarray]:
    return None if value is None else _vec(value, shape, name)


# ---------------------------------------------------------------------------
# Inputs (frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Detection:
    detection_id: int
    confidence: float  # detector confidence [0, 1], NOT localisation quality
    bbox_px: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max), pixels of image_frame_id
    image_stamp_ns: int  # image capture time, not callback time
    clock_domain: str
    image_frame_id: str
    image_size_px: Tuple[int, int]  # (width, height) from the stream
    tracking_id: Optional[int] = None
    class_label: Optional[str] = None

    def bbox_failure(self) -> Optional[FailureReason]:
        """Structural bbox check only (finite, ordered, inside the image). Size thresholds belong to Layer 1."""
        x0, y0, x1, y1 = self.bbox_px
        if not all(math.isfinite(v) for v in self.bbox_px) or x1 <= x0 or y1 <= y0:
            return FailureReason.BBOX_INVALID
        w, h = self.image_size_px
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return FailureReason.BBOX_OUTSIDE_IMAGE
        return None


@dataclass(frozen=True, eq=False)
class CameraIntrinsics:
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    width_px: int
    height_px: int
    distortion_model: str
    distortion_coeffs: Tuple[float, ...] = ()

    def failure(self) -> Optional[FailureReason]:
        vals = (self.fx_px, self.fy_px, self.cx_px, self.cy_px)
        if not all(math.isfinite(v) for v in vals) or self.fx_px <= 0 or self.fy_px <= 0:
            return FailureReason.INTRINSICS_MISSING
        if self.width_px <= 0 or self.height_px <= 0:
            return FailureReason.INTRINSICS_MISSING
        return None


@dataclass(frozen=True, eq=False)
class DepthMeasurement:
    depth_stamp_ns: int
    clock_domain: str
    frame_id: str
    depth_raw: np.ndarray  # 2D raw units; kept as a view, never copied here
    depth_scale_m_per_unit: float
    intrinsics: CameraIntrinsics
    aligned_to_color: bool
    valid_range_m: Tuple[float, float]  # from config l1.depth_min_m / l1.depth_max_m

    def __post_init__(self) -> None:
        arr = np.asarray(self.depth_raw)  # no copy for ndarray input
        if arr.ndim != 2:
            raise ValueError(f"depth_raw: expected 2D array, got ndim={arr.ndim}")
        object.__setattr__(self, "depth_raw", arr)

    def scale_failure(self) -> Optional[FailureReason]:
        s = self.depth_scale_m_per_unit
        if not math.isfinite(s) or s <= 0:
            return FailureReason.DEPTH_SCALE_INVALID
        return None


@dataclass(frozen=True, eq=False)
class PoseSample:
    stamp_ns: int
    clock_domain: str
    frame_id: str  # e.g. map
    child_frame_id: str  # e.g. body
    position_m: np.ndarray  # (3,)
    orientation_wxyz: np.ndarray  # (4,) unit quaternion
    valid: bool
    pose_revision: int
    source: str  # provenance, e.g. "slam_2d+fc_attitude+rangefinder"
    covariance: Optional[np.ndarray] = None  # (6, 6) m^2 / rad^2

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", _vec(self.position_m, (3,), "position_m"))
        object.__setattr__(self, "orientation_wxyz", _vec(self.orientation_wxyz, (4,), "orientation_wxyz"))
        object.__setattr__(self, "covariance", _opt_vec(self.covariance, (6, 6), "covariance"))

    def failure(self, quat_norm_tol: float = 1e-6) -> Optional[FailureReason]:
        if not self.valid:
            return FailureReason.POSE_INVALID
        if not (np.all(np.isfinite(self.position_m)) and np.all(np.isfinite(self.orientation_wxyz))):
            return FailureReason.POSE_INVALID
        if abs(float(np.linalg.norm(self.orientation_wxyz)) - 1.0) > quat_norm_tol:
            return FailureReason.POSE_INVALID
        return None


# ---------------------------------------------------------------------------
# Pipeline state (mutable)
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class SurvivorObservation:
    observation_id: int
    detection_id: int
    confidence: float
    capture_stamp_ns: int
    tracking_id: Optional[int] = None
    # Layer 1
    p_camera_m: Optional[np.ndarray] = None  # (3,)
    cov_camera_m2: Optional[np.ndarray] = None  # (3, 3)
    ground_ref_camera_m: Optional[np.ndarray] = None  # (3,), None = not reliably estimable
    n_valid_px: int = 0
    valid_fraction: float = 0.0
    depth_quality: float = 0.0
    floor_removed_px: int = 0
    cluster_points_camera_m: Optional[np.ndarray] = None  # (N, 3) bounded subsample, kept for the footprint/Layer 5
    # Layer 2 provenance, kept for re-derivation (spec 9.2)
    extrinsic_version: Optional[int] = None
    pose_stamps_ns: Optional[Tuple[int, int]] = None
    interp_alpha: Optional[float] = None
    pose_revision: Optional[int] = None
    camera_map_m: Optional[np.ndarray] = None  # (3,) camera position in the map at capture (view direction)
    p_map_m: Optional[np.ndarray] = None  # (3,)
    cov_map_m2: Optional[np.ndarray] = None  # (3, 3)
    # Layer 3
    p_grid_m: Optional[np.ndarray] = None  # (2,)
    cell: Optional[Cell] = None
    cell_probs: Dict[Cell, float] = field(default_factory=dict)
    registration_version: Optional[int] = None
    boundary_distance_m: Optional[float] = None
    # Diagnostics
    failure: Optional[FailureReason] = None
    warnings: List[FailureReason] = field(default_factory=list)


@dataclass(eq=False)
class SurvivorTrack:
    survivor_id: int
    first_seen_ns: int
    last_seen_ns: int
    status: TagStatus = TagStatus.CANDIDATE
    detector_tracking_ids: Set[int] = field(default_factory=set)
    p_map_m: Optional[np.ndarray] = None  # (3,)
    cov_map_m2: Optional[np.ndarray] = None  # (3, 3)
    p_grid_m: Optional[np.ndarray] = None  # (2,)
    cell: Optional[Cell] = None
    cell_probs: Dict[Cell, float] = field(default_factory=dict)
    max_detection_confidence: float = 0.0
    localisation_quality: float = 0.0  # from covariance, never from detector confidence
    observation_count: int = 0
    effective_independent_count: float = 0.0
    observation_ids: List[int] = field(default_factory=list)  # bounded by l4.track_history_max in Layer 4
    last_pose_revision: Optional[int] = None
    registration_version: Optional[int] = None
    reasons: List[FailureReason] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Output (frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Tag:
    """Full tag (architecture.md section e). The compact radio form is output.TagRecord."""

    survivor_id: int
    status: TagStatus
    cell: Optional[Cell]
    cell_label: Optional[str]  # official label once the numbering convention is known
    cell_probs: Tuple[Tuple[Cell, float], ...]  # top cells, highest first
    p_map_m: Optional[np.ndarray]
    cov_map_m2: Optional[np.ndarray]
    p_grid_m: Optional[np.ndarray]
    sigma_horizontal_m: Optional[float]
    boundary_distance_m: Optional[float]
    out_of_grid: bool
    max_detection_confidence: float
    localisation_quality: float
    observation_count: int
    effective_independent_count: float
    first_seen_ns: int
    last_seen_ns: int
    capture_stamp_ns: int  # latest contributing capture
    update_stamp_ns: int
    map_revision: int
    registration_version: int
    extrinsic_version: Optional[int]
    reasons: Tuple[FailureReason, ...] = ()
    confirmed_stamp_ns: Optional[int] = None  # when the tag was (last) confirmed
    cell_changes: int = 0  # cell changes after the first confirmation


@dataclass(frozen=True, eq=False)
class TagResult:
    """Pipeline output scored by the harness (survivor_id is the pipeline's own tag id)."""

    survivor_id: int
    cell: Optional[Cell]
    status: TagStatus
    p_map_m: Optional[np.ndarray]  # (3,) or (2,); only x, y are scored
    first_stamp_ns: int
    confirm_stamp_ns: Optional[int]
    top_prob: Optional[float] = None  # probability of `cell` (Layer 4); None if not computed
    cell_changes: int = 0  # cell changes after first confirmation (flicker)
