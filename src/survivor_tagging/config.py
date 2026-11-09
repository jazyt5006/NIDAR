"""Configuration registry (architecture.md section c).

Hardware and competition parameters ship unset (value None, required). Only tuning parameters have defaults,
each labelled ASSUMPTION. Overrides come from a flat JSON file: {"_note": "...", "<param name>": <value>, ...}.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .types import FailureReason

ARCH_REF = "architecture.md (c)"
IGNORED_KEYS = ("_note",)
RIGID_TOL = 1e-6  # numerical tolerance for rotation orthonormality, not a tuning threshold


class ParamSource(str, Enum):
    NEEDS_MEASUREMENT = "needs measurement"
    ASSUMPTION = "ASSUMPTION"
    RULEBOOK = "rulebook"


class ConfigError(ValueError):
    pass


@dataclass
class ConfigParam:
    name: str
    unit: str
    value: Any
    required: bool
    source: ParamSource
    reference: str
    valid_range: Optional[Tuple[float, float]] = None
    kind: str = "float"  # float, int, bool, str, enum, float_list, rigid_transform, covariance
    meaning: str = ""
    choices: Optional[Tuple[str, ...]] = None
    shape: Optional[Tuple[int, ...]] = None
    required_when: Optional[Tuple[str, Any]] = None  # required only if another param has this value

    @property
    def stage(self) -> str:
        return _PREFIX_STAGE[self.name.split(".", 1)[0]]


_PREFIX_STAGE = {
    "cam": "camera",
    "extr": "extrinsics",
    "clock": "clock",
    "l1": "l1",
    "l1b": "l1",
    "map": "venue",
    "l2": "l2",
    "grid": "l3",
    "l4": "l4",
    "l5": "l5",
    "score": "l5",
}

STAGES = ("camera", "extrinsics", "clock", "venue", "l1", "l2", "l3", "l4", "l5")

# A stage cannot run unless its dependencies are configured too.
STAGE_DEPENDENCIES: Dict[str, Tuple[str, ...]] = {
    "camera": (),
    "extrinsics": (),
    "clock": (),
    "venue": (),
    "l1": ("camera",),
    "l2": ("extrinsics", "clock"),
    "l3": (),
    "l4": (),
    "l5": (),
}


def _req(name, unit, kind, meaning, *, rng=None, choices=None, shape=None, when=None, ref=ARCH_REF) -> ConfigParam:
    """Required parameter, unset, needs measurement."""
    return ConfigParam(name, unit, None, when is None, ParamSource.NEEDS_MEASUREMENT, ref, rng, kind, meaning,
                       choices, shape, when)


def _opt(name, unit, kind, meaning, *, rng=None, ref=ARCH_REF) -> ConfigParam:
    """Optional parameter, unset, needs measurement."""
    return ConfigParam(name, unit, None, False, ParamSource.NEEDS_MEASUREMENT, ref, rng, kind, meaning)


def _tune(name, unit, kind, default, meaning, *, rng=None, choices=None, ref=ARCH_REF) -> ConfigParam:
    """Tuning parameter with an ASSUMPTION default, to be validated experimentally."""
    return ConfigParam(name, unit, default, False, ParamSource.ASSUMPTION, ref, rng, kind, meaning, choices)


_FILE_INTRINSICS = ("cam.intrinsics_source", "file")


def _build_params() -> List[ConfigParam]:
    return [
        # Sensor / calibration
        _req("cam.intrinsics_source", "-", "enum", "intrinsics from device (CameraInfo/librealsense) or from file",
             choices=("device", "file")),
        _req("cam.fx_px", "px", "float", "focal length x (file intrinsics only)", rng=(1e-6, 1e5), when=_FILE_INTRINSICS),
        _req("cam.fy_px", "px", "float", "focal length y (file intrinsics only)", rng=(1e-6, 1e5), when=_FILE_INTRINSICS),
        _req("cam.cx_px", "px", "float", "principal point x (file intrinsics only)", rng=(0, 1e5), when=_FILE_INTRINSICS),
        _req("cam.cy_px", "px", "float", "principal point y (file intrinsics only)", rng=(0, 1e5), when=_FILE_INTRINSICS),
        _req("cam.width_px", "px", "int", "image width (file intrinsics only)", rng=(1, 100000), when=_FILE_INTRINSICS),
        _req("cam.height_px", "px", "int", "image height (file intrinsics only)", rng=(1, 100000), when=_FILE_INTRINSICS),
        _opt("cam.depth_scale_m_per_unit", "m/unit", "float",
             "read from device at runtime; config value is only a cross-check", rng=(1e-9, 1.0)),
        _req("cam.depth_frame_id", "-", "str", "frame of the depth image"),
        _req("cam.color_frame_id", "-", "str", "frame of the image detections are made on"),
        _req("cam.detections_use_depth_aligned_frame", "-", "bool",
             "true only if detections come from D435i colour with depth aligned to it (Q D2)"),
        _req("extr.T_body_camera", "m, rad", "rigid_transform", "camera-to-body 4x4 transform (Q H4)", shape=(4, 4)),
        _req("extr.T_body_camera_cov", "m^2, rad^2", "covariance", "extrinsic uncertainty", shape=(6, 6)),
        _req("extr.version", "-", "int", "bumped on any calibration change", rng=(0, 2**31 - 1)),
        _req("clock.camera_to_pose_offset_ns", "ns", "int", "offset between camera and pose clock domains (Q S3)",
             rng=(-(2**62), 2**62)),
        _req("clock.offset_uncertainty_ns", "ns", "int", "uncertainty of that offset", rng=(0, 2**62)),
        # Layer 1
        _req("l1.depth_min_m", "m", "float", "minimum accepted depth", rng=(0.0, 100.0)),
        _req("l1.depth_max_m", "m", "float", "maximum accepted depth", rng=(0.0, 100.0)),
        _tune("l1.bbox_inset_fraction", "ratio", "float", 0.15, "bbox inset per side before sampling", rng=(0.0, 0.49)),
        _tune("l1.min_bbox_size_px", "px", "int", 8, "reject boxes smaller than this", rng=(1, 10000)),
        _tune("l1.sample_stride_px", "px", "int", 2, "subsample stride to bound CPU", rng=(1, 64)),
        _tune("l1.min_valid_pixels", "count", "int", 50, "below this: DEPTH_TOO_FEW_PIXELS", rng=(1, 10**7)),
        _tune("l1.min_valid_fraction", "ratio", "float", 0.2, "valid / sampled pixels", rng=(0.0, 1.0)),
        _tune("l1.cluster_gap_m", "m", "float", 0.3, "depth gap separating clusters", rng=(0.01, 10.0)),
        _tune("l1.min_density_fraction", "ratio", "float", 0.05,
              "1D density clustering: a depth sample is kept only if at least this share of the valid samples lies "
              "within +/- cluster_gap_m / 2 of it (removes sparse continua such as floor and flying pixels; 0 = "
              "plain gap clustering)", rng=(0.0, 1.0), ref="substep 2.3b"),
        _tune("l1.min_cluster_fraction", "ratio", "float", 0.3, "cluster must hold this share of valid pixels",
              rng=(0.0, 1.0)),
        _tune("l1.ambiguity_ratio", "ratio", "float", 0.6, "2nd/1st cluster size above this = ambiguous",
              rng=(0.0, 1.0)),
        _tune("l1.background_fraction_max", "ratio", "float", 0.7,
              "far-cluster share above this = background dominant", rng=(0.0, 1.0)),
        _tune("l1.background_near_min_fraction", "ratio", "float", 0.05,
              "share of valid pixels nearer than the chosen cluster needed before it can be called background",
              rng=(0.0, 1.0), ref="substep 2.3 (geometry.select_cluster)"),
        _req("l1.depth_noise_model", "m", "float_list",
             "depth 1-sigma polynomial coefficients c_k: sigma_m = sum c_k z^k (polynomial form is ASSUMPTION)"),
        _tune("l1.point_method", "-", "enum", "median_pixel",
              "camera-frame point: median_pixel (median depth at the median pixel), cluster_centroid "
              "(per-axis median of all back-projected cluster points) or cluster_area_weighted (per-axis "
              "median weighted by depth^2, i.e. by each pixel's surface footprint)",
              choices=("median_pixel", "cluster_centroid", "cluster_area_weighted"), ref="substeps 2.3b, 2.4"),
        _tune("l1.cluster_rule", "-", "enum", "spatial",
              "background rule: spatial (occluder band / thin strip layout checks) or fraction_only (2.3 rule)",
              choices=("spatial", "fraction_only"), ref="substep 2.3b"),
        _tune("l1.occluder_edge_span_min", "ratio", "float", 0.6,
              "nearer pixels are an occluder if they are the outermost valid pixels along at least this share of "
              "one box edge", rng=(0.0, 1.0), ref="substep 2.3b"),
        _tune("l1.strip_rows_fraction_min", "ratio", "float", 0.6,
              "nearer pixels are a thin object in front of background if at least this share of their rows has "
              "chosen-cluster pixels on both sides", rng=(0.0, 1.0), ref="substep 2.3b"),
        _tune("l1.truncation_margin_px", "px", "float", 1.0,
              "box within this distance of the image border is BBOX_TRUNCATED", rng=(0.0, 100.0), ref="substep 2.3b"),
        _tune("l1.truncated_cov_scale", "ratio", "float", 4.0,
              "covariance multiplier for BBOX_TRUNCATED estimates (4 = doubled sigma)", rng=(1.0, 100.0),
              ref="substep 2.3b"),
        _tune("l1.floor_removal", "-", "bool", False,
              "remove pixels within the floor margin before clustering (needs map.floor_z_m and the capture pose)",
              ref="substep 3.1b"),
        _tune("l1.floor_margin_m", "m", "float", 0.06, "pixels lower than this above the floor plane are floor",
              rng=(0.0, 1.0), ref="substep 3.1b"),
        _tune("l1.floor_margin_pose_sigma_k", "-", "float", 2.0,
              "margin += k * pose z standard deviation (pose covariance)", rng=(0.0, 10.0), ref="substep 3.1b"),
        _tune("l1b.footprint_refinement", "-", "bool", False,
              "map-frame footprint refinement for extended clusters (prone survivors)", ref="substep 3.1b"),
        _tune("l1b.method", "-", "enum", "midpoint",
              "extended-cluster centre along the principal axis: midpoint of the robust extent, or the depth^2 "
              "(area) weighted median", choices=("midpoint", "area_weighted"), ref="substep 3.1b"),
        _tune("l1b.max_points", "count", "int", 300, "cluster points kept per observation (deterministic subsample)",
              rng=(10, 100000), ref="substep 3.1b"),
        _tune("l1b.extent_percentile_lo", "%", "float", 5.0, "robust extent lower percentile", rng=(0.0, 50.0),
              ref="substep 3.1b"),
        _tune("l1b.extent_percentile_hi", "%", "float", 95.0, "robust extent upper percentile", rng=(50.0, 100.0),
              ref="substep 3.1b"),
        _tune("l1b.extended_min_m", "m", "float", 0.9, "extent along the principal axis that makes a cluster EXTENDED",
              rng=(0.0, 10.0), ref="substep 3.1b"),
        _tune("l1b.extended_axis_sigma_fraction", "ratio", "float", 0.25,
              "extra sigma along the axis = fraction x extent (partial visibility)", rng=(0.0, 2.0), ref="substep 3.1b"),
        _tune("l1.ground_ref_method", "-", "enum", "none", "ground reference estimator (none = provisional only)",
              choices=("none", "bbox_bottom_ray", "keypoints")),
        _tune("l1.body_depth_offset_m", "m", "float", 0.0,
              "offset from visible body surface to body centre; no correction until measured", rng=(0.0, 0.5),
              ref="substep 2.1 instruction"),
        _tune("l1.body_depth_offset_sigma_m", "m", "float", 0.10,
              "1-sigma uncertainty of l1.body_depth_offset_m (true front-surface bias minus the configured offset, "
              "along the camera ray). Used only by Layer 4's per-view-bin ray-bias term. Shrink it only after the "
              "offset is measured on real bodies (Step 5)", rng=(0.0, 1.0), ref="substep 3.1c"),
        # Venue
        _opt("map.floor_z_m", "m", "float", "floor height in the SLAM map frame (venue/hardware config; unset = floor "
             "removal disabled, observations carry FLOOR_UNKNOWN)", rng=(-100.0, 100.0)),
        # Layer 2
        _tune("l2.pose_buffer_duration_s", "s", "float", 10.0, "pose history kept", rng=(0.1, 600.0)),
        _tune("l2.pose_buffer_max_samples", "count", "int", 2000, "hard memory bound on pose buffer", rng=(2, 10**6)),
        _req("l2.max_interp_gap_s", "s", "float", "max gap between bracketing poses (depends on pose rate, Q S2)",
             rng=(0.0, 10.0)),
        _tune("l2.allow_extrapolation_s", "s", "float", 0.0, "extrapolation beyond newest pose (0 = never)",
              rng=(0.0, 1.0)),
        _tune("l2.max_detection_age_s", "s", "float", 1.0, "detection older than this = stale", rng=(0.0, 60.0)),
        _tune("l2.max_pose_wait_s", "s", "float", 0.5,
              "hold an observation this long (after its arrival) waiting for a pose newer than its capture time",
              rng=(0.0, 10.0), ref="substep 2.4"),
        _tune("l2.pending_queue_max", "count", "int", 64, "bound on observations waiting for a pose",
              rng=(1, 100000), ref="substep 2.4"),
        _req("l2.max_rgb_depth_skew_s", "s", "float", "RGB vs depth stamp tolerance", rng=(0.0, 1.0)),
        _req("l2.pose_mode", "-", "enum", "pose source structure (Q S1)",
             choices=("full_3d", "planar_plus_attitude_altitude")),
        # Layer 3 (Q O1)
        ConfigParam("grid.cell_size_m", "m", 1.0, True, ParamSource.RULEBOOK, "NIDAR 26-27 rulebook p.19",
                    (0.01, 100.0), "float", "cell edge length (1 m x 1 m reference grid)"),
        _req("grid.n_cells_x", "count", "int", "grid extent along grid x", rng=(1, 10000)),
        _req("grid.n_cells_y", "count", "int", "grid extent along grid y", rng=(1, 10000)),
        _req("grid.origin_in_map_m", "m", "float_list",
             "map-frame coordinates (x, y) of the official grid origin (conventions.md)", shape=(2,)),
        _req("grid.yaw_map_to_grid_rad", "rad", "float",
             "angle from the map +x axis to the grid +x axis, counter-clockwise about map +z (conventions.md)",
             rng=(-math.pi, math.pi)),
        _req("grid.axis_flip_x", "-", "bool", "invert grid x axis"),
        _req("grid.axis_flip_y", "-", "bool", "invert grid y axis"),
        _req("grid.index_base", "-", "int", "first cell index", rng=(0, 1)),
        _req("grid.numbering", "-", "enum", "cell label convention", choices=("ij", "row_major_label", "alnum")),
        _tune("grid.boundary_rule", "-", "enum", "floor", "owner of points exactly on an edge (lower-left-closed)",
              choices=("floor",)),
        _tune("grid.registration_version", "-", "int", 0, "bumped on any registration change (starts at 0)",
              rng=(0, 2**31 - 1)),
        _req("grid.registration_valid", "-", "bool", "false = never publish CONFIRMED official cells"),
        _tune("grid.max_candidate_cells", "count", "int", 4, "candidate cells kept with probabilities",
              rng=(1, 100), ref="substep 2.4"),
        _tune("grid.boundary_margin_m", "m", "float", 0.1, "floor of covariance-derived edge margin",
              rng=(0.0, 1.0)),
        # Layer 4 (substep 3.1; replaces the 2.x placeholders decorrelation_time_s, min_viewpoint_baseline_m,
        # confirm_min_independent_obs, confirm_min_cell_probability, confirm_max_sigma_m, outlier_mahalanobis,
        # unreliable_after_inconsistent_obs)
        _tune("l4.assoc_gate_mahalanobis", "chi^2", "float", 9.21, "association gate, 2-DoF 99 %", rng=(0.0, 1000.0)),
        _tune("l4.assoc_max_distance_m", "m", "float", 1.0, "hard cap on association distance", rng=(0.0, 20.0)),
        _tune("l4.assoc_ambiguity_margin", "chi^2", "float", 2.0,
              "observation is ambiguous if a second track's cost is within this of the best", rng=(0.0, 100.0)),
        _tune("l4.track_id_hint_bonus", "chi^2", "float", 1.0,
              "cost reduction when the detector tracking_id matches the track (a hint only)", rng=(0.0, 100.0)),
        _tune("l4.merge_gate_mahalanobis", "chi^2", "float", 9.21, "two tracks are consistent below this",
              rng=(0.0, 1000.0)),
        _tune("l4.min_separate_survivor_distance_m", "m", "float", 0.5,
              "two tracks seen in one frame closer than this are flagged, never silently merged; two OBSERVATIONS of one frame closer than this that fit the same track (or no track) are one body seen twice: the better one is kept (NEAR_DUPLICATE_OBSERVATION, substep 3.3a). 0 disables", rng=(0.0, 20.0)),
        _tune("l4.view_bin_deg", "deg", "float", 30.0, "bearing bin width: same-bin views share bias",
              rng=(1.0, 180.0)),
        _tune("l4.view_bin_max_gap_s", "s", "float", 5.0, "a gap longer than this starts a new view bin (revisit)",
              rng=(0.0, 3600.0)),
        _tune("l4.obs_model_sigma_m", "m", "float", 0.05,
              "isotropic per-observation model error added to the Layer 1-3 covariance (where on the body the point "
              "lands varies frame to frame; not part of the measurement covariance)", rng=(0.0, 5.0)),
        _tune("l4.ray_bias_sigma_m", "m", "float", 0.1,
              "floor of the common-mode error along the camera ray per view bin, EXCLUDING the body-offset "
              "uncertainty (pose/extrinsic/depth-scale/surface effects). Layer 4 uses sqrt(this^2 + "
              "l1.body_depth_offset_sigma_m^2)", rng=(0.0, 5.0), ref="substeps 3.1, 3.1c"),
        _tune("l4.max_effective_obs_per_bin", "count", "int", 3,
              "averaging within one view bin reduces the noise by at most this many observations", rng=(1, 1000)),
        _tune("l4.provisional_min_obs", "count", "int", 2, "CANDIDATE -> PROVISIONAL after this many frames",
              rng=(1, 1000)),
        _tune("l4.confirm_min_obs", "count", "int", 5, "frames needed to confirm", rng=(1, 1000)),
        _tune("l4.confirm_min_view_bins", "count", "int", 1, "distinct view bins needed to confirm", rng=(1, 100)),
        _tune("l4.confirm_min_span_s", "s", "float", 0.5, "first-to-last observation span needed to confirm",
              rng=(0.0, 3600.0)),
        _tune("l4.confirm_prob", "prob", "float", 0.9, "top-cell probability needed to confirm", rng=(0.0, 1.0)),
        _tune("l4.cell_change_margin", "prob", "float", 0.2,
              "a confirmed tag moves only if the new cell beats the old by this margin", rng=(0.0, 1.0)),
        _tune("l4.cell_change_updates", "count", "int", 3, "... for this many consecutive updates", rng=(1, 1000)),
        _tune("l4.unreliable_nis", "chi^2", "float", 6.0,
              "mean normalised innovation over the window above this = observations contradict", rng=(0.0, 1000.0)),
        _tune("l4.nis_window", "count", "int", 5, "observations in the consistency window", rng=(1, 1000)),
        _tune("l4.unreliable_ambiguous_streak", "count", "int", 3,
              "consecutive ambiguous frames that make a track UNRELIABLE", rng=(1, 1000)),
        _tune("l4.candidate_timeout_s", "s", "float", 2.0,
              "never-confirmed tracks with fewer than confirm_min_obs frames are dropped after this silence",
              rng=(0.0, 3600.0)),
        _tune("l4.track_history_max", "count", "int", 50, "observations kept per track (eviction beyond)",
              rng=(1, 100000)),
        _tune("l4.max_tracks", "count", "int", 64, "tracks kept (lowest-value evicted beyond)", rng=(1, 100000)),
        # Layer 5
        _req("l5.correction_source", "-", "enum", "SLAM correction mechanism (Q S4): 'history' = corrected poses per "
             "timestamp, 'frame_correction' = one map->odom style jump, 'none' = no correction events (substep 3.2)",
             choices=("none", "history", "frame_correction")),
        _tune("l5.observation_store_max", "count", "int", 2000, "bounded re-derivation store (oldest evicted first)",
              rng=(1, 10**7)),
        _tune("l5.frame_correction_cov_inflation", "factor", "float", 4.0,
              "map covariance multiplier for frame_correction observations (the jump ignores drift inside the "
              "window)", rng=(1.0, 1e6), ref="substep 3.2"),
        _tune("l5.uncorrected_cov_inflation", "factor", "float", 4.0,
              "map covariance multiplier for POSE_NOT_CORRECTED / OBSERVATION_NOT_REDERIVABLE observations",
              rng=(1.0, 1e6), ref="substep 3.2"),
        _tune("l5.correction_history_max", "count", "int", 8,
              "correction events kept (the oldest is folded into the stored poses beyond this)", rng=(1, 1000),
              ref="substep 3.2"),
        _tune("l5.missing_correction_timeout_s", "s", "float", 2.0,
              "a pose_revision bump with no correction event after this long marks older observations "
              "POSE_NOT_CORRECTED", rng=(0.0, 3600.0), ref="substep 3.2"),
        _tune("l5.recompute_mode", "-", "enum", "full",
              "'full' = re-derive every stored observation and replay Layer 4 from scratch (exact: equals a run with "
              "the corrected poses from the start); 'affected_tracks' = re-derive changed observations and replay "
              "only the tracks that hold them (approximation, added after R3 failed)",
              choices=("full", "affected_tracks"), ref="substep 3.2"),
        _tune("l5.tag_history_max", "count", "int", 8, "cell changes kept per tag", rng=(1, 1000), ref="substep 3.2"),
        _tune("l5.removed_tags_max", "count", "int", 64, "removed survivor ids remembered (to announce removal)",
              rng=(1, 100000), ref="substep 3.2"),
        _tune("l5.telemetry_update_min_interval_s", "s", "float", 1.0,
              "per-tag rate limit for event updates (architecture send rule 1)", rng=(0.0, 3600.0),
              ref="substep 3.2"),
        _tune("l5.telemetry_log_max", "count", "int", 4096, "sent telemetry messages kept in memory (diagnostics)",
              rng=(0, 10**7), ref="substep 3.2"),
        _tune("l5.telemetry_snapshot_period_s", "s", "float", 5.0, "full snapshot cadence", rng=(0.1, 600.0)),
        _tune("l5.telemetry_max_tags_per_snapshot", "count", "int", 32, "tags per snapshot (u8 n_tags)",
              rng=(0, 255)),
        _tune("l5.telemetry_send_candidates", "-", "bool", False, "include CANDIDATE tags on the radio"),
        _tune("l5.telemetry_link_budget_bps", "bit/s", "float", 20000.0, "link budget we design against (Q H5)",
              rng=(1.0, 1e9)),
        _req("l5.end_of_mission_commit_policy", "-", "enum", "which tags to commit at mission end (Q O2)",
             choices=("confirmed_only", "confirmed_plus_provisional_top1", "expected_value")),
        _tune("l5.commit_min_obs", "count", "int", 3,
              "expected_value policy: a track needs at least this many observations to be committed", rng=(1, 1000),
              ref="substep 3.1c"),
        # Scoring (substep 3.1c)
        ConfigParam("score.correct_tag_points", "points", 40.0, True, ParamSource.RULEBOOK,
                    "NIDAR 26-27 rulebook p.19 (40 marks per survivor detected and correctly tagged)", (0.0, 1000.0),
                    "float", "points for a correctly tagged survivor"),
        _opt("score.wrong_tag_penalty_assumed", "points", "float",
             "ASSUMED penalty (positive number) for a wrong or extra tag; UNKNOWN until organisers answer O2. Unset = "
             "expected_value threshold 0.5 with a 'penalty unknown' note", rng=(0.0, 1000.0)),
    ]


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_range(p: ConfigParam, v: float) -> None:
    if p.valid_range is not None:
        lo, hi = p.valid_range
        if not (lo <= v <= hi):
            raise ConfigError(f"{p.name}: {v} outside valid range [{lo}, {hi}]")


def _matrix(p: ConfigParam, raw: Any) -> np.ndarray:
    try:
        m = np.array(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{p.name}: not a numeric matrix ({exc})") from None
    if m.shape != p.shape:
        raise ConfigError(f"{p.name}: expected shape {p.shape}, got {m.shape}")
    if not np.all(np.isfinite(m)):
        raise ConfigError(f"{p.name}: non-finite entries")
    return m


def coerce_value(p: ConfigParam, raw: Any) -> Any:
    """Validate a raw JSON value for parameter p and return the stored form. Raises ConfigError."""
    if raw is None:
        if p.required:
            raise ConfigError(f"{p.name}: required parameter cannot be set to null")
        return None
    k = p.kind
    if k == "float":
        if not _is_number(raw) or not math.isfinite(raw):
            raise ConfigError(f"{p.name}: expected a finite number, got {raw!r}")
        _check_range(p, float(raw))
        return float(raw)
    if k == "int":
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ConfigError(f"{p.name}: expected an integer, got {raw!r}")
        _check_range(p, raw)
        return raw
    if k == "bool":
        if not isinstance(raw, bool):
            raise ConfigError(f"{p.name}: expected true/false, got {raw!r}")
        return raw
    if k == "str":
        if not isinstance(raw, str) or not raw:
            raise ConfigError(f"{p.name}: expected a non-empty string, got {raw!r}")
        return raw
    if k == "enum":
        if raw not in (p.choices or ()):
            raise ConfigError(f"{p.name}: {raw!r} not one of {p.choices}")
        return raw
    if k == "float_list":
        if not isinstance(raw, list) or not raw or not all(_is_number(x) and math.isfinite(x) for x in raw):
            raise ConfigError(f"{p.name}: expected a non-empty list of finite numbers, got {raw!r}")
        if p.shape is not None and (len(raw),) != p.shape:
            raise ConfigError(f"{p.name}: expected {p.shape[0]} values, got {len(raw)}")
        return tuple(float(x) for x in raw)
    if k == "rigid_transform":
        m = _matrix(p, raw)
        r = m[:3, :3]
        if not np.allclose(m[3], [0.0, 0.0, 0.0, 1.0], atol=RIGID_TOL):
            raise ConfigError(f"{p.name}: last row must be [0, 0, 0, 1]")
        if not np.allclose(r.T @ r, np.eye(3), atol=RIGID_TOL) or np.linalg.det(r) <= 0:
            raise ConfigError(f"{p.name}: rotation block is not a proper rotation")
        m.setflags(write=False)
        return m
    if k == "covariance":
        m = _matrix(p, raw)
        if not np.allclose(m, m.T, atol=RIGID_TOL) or np.any(np.diag(m) < 0):
            raise ConfigError(f"{p.name}: covariance must be symmetric with non-negative diagonal")
        m.setflags(write=False)
        return m
    raise ConfigError(f"{p.name}: unknown kind {k!r}")  # pragma: no cover


class Config:
    """Parameter registry. Use default_config() for a fresh instance."""

    def __init__(self, params: List[ConfigParam]):
        self._params: Dict[str, ConfigParam] = {p.name: p for p in params}

    def __contains__(self, name: str) -> bool:
        return name in self._params

    def __getitem__(self, name: str) -> ConfigParam:
        try:
            return self._params[name]
        except KeyError:
            raise ConfigError(f"unknown parameter {name!r}") from None

    def value(self, name: str) -> Any:
        return self[name].value

    def params(self) -> List[ConfigParam]:
        return list(self._params.values())

    def set(self, name: str, raw: Any, reference: Optional[str] = None) -> None:
        p = self[name]
        p.value = coerce_value(p, raw)
        if reference is not None:
            p.reference = reference

    def is_required(self, p: ConfigParam) -> bool:
        if p.required:
            return True
        if p.required_when is not None:
            other, wanted = p.required_when
            return self._params[other].value == wanted
        return False

    def missing_required(self, stage: str, include_dependencies: bool = True) -> List[str]:
        """Names of unset required parameters for a stage (and, by default, the stages it depends on)."""
        if stage not in STAGES:
            raise ConfigError(f"unknown stage {stage!r}; expected one of {STAGES}")
        stages = [stage] + (list(STAGE_DEPENDENCIES[stage]) if include_dependencies else [])
        return [p.name for p in self._params.values()
                if p.stage in stages and p.value is None and self.is_required(p)]

    def is_stage_ready(self, stage: str) -> bool:
        return not self.missing_required(stage)

    def stage_failure(self, stage: str) -> Optional[FailureReason]:
        """CONFIG_MISSING if the stage has unset required parameters, else None."""
        return FailureReason.CONFIG_MISSING if self.missing_required(stage) else None

    def apply_overrides(self, overrides: Dict[str, Any], reference: str) -> None:
        """Validate every key first, then apply all; nothing is changed if any key is invalid."""
        if not isinstance(overrides, dict):
            raise ConfigError("override file must contain a JSON object")
        unknown = sorted(k for k in overrides if k not in IGNORED_KEYS and k not in self._params)
        if unknown:
            raise ConfigError(f"unknown parameter name(s): {', '.join(unknown)}")
        staged = {k: coerce_value(self._params[k], v) for k, v in overrides.items() if k not in IGNORED_KEYS}
        lo = staged.get("l1.depth_min_m", self._params["l1.depth_min_m"].value)
        hi = staged.get("l1.depth_max_m", self._params["l1.depth_max_m"].value)
        if lo is not None and hi is not None and lo >= hi:
            raise ConfigError(f"l1.depth_min_m ({lo}) must be less than l1.depth_max_m ({hi})")
        for k, v in staged.items():
            self._params[k].value = v
            self._params[k].reference = reference


def default_config() -> Config:
    return Config(copy.deepcopy(_build_params()))


def load_config(path: Union[str, Path], base: Optional[Config] = None) -> Config:
    """Load a JSON override file onto `base` (or a fresh default config). Raises ConfigError on any problem."""
    path = Path(path)
    cfg = base if base is not None else default_config()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path.name}: invalid JSON ({exc})") from None
    cfg.apply_overrides(data, reference=f"override: {path.name}")
    return cfg
