"""Layer 1: robust survivor geometry (spec Part 5; architecture.md "Layer 1").

estimate_survivor_point() is a pure function of (detection, depth frame, intrinsics, Layer1Params). It never
copies the full depth frame: it slices a strided view of the inset box. No simulator, baseline or ROS imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

from .config import Config, ConfigError
from .types import CameraIntrinsics, DepthMeasurement, Detection, FailureReason

MAD_TO_SIGMA = 1.4826  # MAD -> standard deviation for a normal distribution


@dataclass(frozen=True)
class Layer1Params:
    """Values come only from the config registry (from_config); no defaults are repeated here."""

    depth_min_m: float
    depth_max_m: float
    depth_noise_coeffs: Tuple[float, ...]  # sigma_m(z) = sum c_k z^k
    bbox_inset_fraction: float
    min_bbox_size_px: int
    sample_stride_px: int
    min_valid_pixels: int
    min_valid_fraction: float
    cluster_gap_m: float
    min_density_fraction: float
    min_cluster_fraction: float
    ambiguity_ratio: float
    background_fraction_max: float
    background_near_min_fraction: float
    ground_ref_method: str
    body_depth_offset_m: float
    point_method: str
    cluster_rule: str
    occluder_edge_span_min: float
    strip_rows_fraction_min: float
    truncation_margin_px: float
    truncated_cov_scale: float
    keep_points: int = 0  # >0: keep a deterministic subsample of this many cluster points (footprint refinement)

    @classmethod
    def from_config(cls, cfg: Config) -> "Layer1Params":
        """Raises ConfigError (CONFIG_MISSING) if a required l1 parameter is unset."""
        missing = cfg.missing_required("l1", include_dependencies=False)
        if missing:
            raise ConfigError(f"{FailureReason.CONFIG_MISSING.value}: {', '.join(missing)}")
        v = cfg.value
        return cls(
            depth_min_m=v("l1.depth_min_m"), depth_max_m=v("l1.depth_max_m"),
            depth_noise_coeffs=tuple(v("l1.depth_noise_model")),
            bbox_inset_fraction=v("l1.bbox_inset_fraction"), min_bbox_size_px=v("l1.min_bbox_size_px"),
            sample_stride_px=v("l1.sample_stride_px"), min_valid_pixels=v("l1.min_valid_pixels"),
            min_valid_fraction=v("l1.min_valid_fraction"), cluster_gap_m=v("l1.cluster_gap_m"),
            min_density_fraction=v("l1.min_density_fraction"),
            min_cluster_fraction=v("l1.min_cluster_fraction"), ambiguity_ratio=v("l1.ambiguity_ratio"),
            background_fraction_max=v("l1.background_fraction_max"),
            background_near_min_fraction=v("l1.background_near_min_fraction"),
            ground_ref_method=v("l1.ground_ref_method"), body_depth_offset_m=v("l1.body_depth_offset_m"),
            point_method=v("l1.point_method"), cluster_rule=v("l1.cluster_rule"),
            occluder_edge_span_min=v("l1.occluder_edge_span_min"),
            strip_rows_fraction_min=v("l1.strip_rows_fraction_min"),
            truncation_margin_px=v("l1.truncation_margin_px"), truncated_cov_scale=v("l1.truncated_cov_scale"),
            keep_points=v("l1b.max_points") if v("l1b.footprint_refinement") else 0,
        )


@dataclass
class Layer1Result:
    """The Layer 1 fields of SurvivorObservation plus diagnostics."""

    p_camera_m: Optional[np.ndarray] = None  # (3,) optical frame
    cov_camera_m2: Optional[np.ndarray] = None  # (3, 3)
    ground_ref_camera_m: Optional[np.ndarray] = None
    n_valid_px: int = 0
    valid_fraction: float = 0.0
    depth_quality: float = 0.0
    failure: Optional[FailureReason] = None
    warnings: List[FailureReason] = field(default_factory=list)
    # diagnostics
    depth_m: Optional[float] = None  # robust depth of the chosen cluster (before the body offset)
    pixel_uv: Optional[Tuple[float, float]] = None
    n_clusters: int = 0
    floor_removed_px: int = 0
    cluster_points_camera_m: Optional[np.ndarray] = None  # (N, 3), before the body offset


@dataclass(frozen=True, eq=False)
class FloorPlane:
    """Floor in the CAMERA frame: height_above_floor(p) = normal . p - offset (normal is the map +z axis expressed in
    the camera frame). Pixels with height < margin_m are floor (or below it) and are removed before clustering."""

    normal: np.ndarray  # (3,) unit
    offset_m: float
    margin_m: float

    def heights(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.normal - self.offset_m


def subsample_points(pts: np.ndarray, n_max: int) -> np.ndarray:
    """Deterministic, order-preserving subsample to at most n_max points (evenly spaced indices)."""
    if len(pts) <= n_max:
        return pts.copy()
    return pts[np.linspace(0, len(pts) - 1, n_max).round().astype(int)]


# ---------------------------------------------------------------------------
# 1. Input and bounding-box validation
# ---------------------------------------------------------------------------


def validate_inputs(det: Detection, depth: DepthMeasurement, intr: CameraIntrinsics,
                    params: Layer1Params) -> Optional[FailureReason]:
    if det.clock_domain != depth.clock_domain:
        return FailureReason.CLOCK_DOMAIN_UNKNOWN
    h, w = depth.depth_raw.shape
    if det.image_frame_id != depth.frame_id or not depth.aligned_to_color or tuple(det.image_size_px) != (w, h):
        return FailureReason.DEPTH_FRAME_MISMATCH
    if intr.failure() is not None or (intr.width_px, intr.height_px) != (w, h):
        return FailureReason.INTRINSICS_MISSING
    if depth.scale_failure() is not None:
        return FailureReason.DEPTH_SCALE_INVALID
    return validate_bbox(det, params)


def validate_bbox(det: Detection, params: Layer1Params) -> Optional[FailureReason]:
    reason = det.bbox_failure()  # finite, ordered, inside the image
    if reason is not None:
        return reason
    x0, y0, x1, y1 = det.bbox_px
    if x1 - x0 < params.min_bbox_size_px or y1 - y0 < params.min_bbox_size_px:
        return FailureReason.BBOX_INVALID
    return None


def bbox_truncated(bbox_px, width: int, height: int, margin_px: float) -> bool:
    """True if the box touches the image border (within margin_px): the person may be cut off."""
    x0, y0, x1, y1 = bbox_px
    return x0 <= margin_px or y0 <= margin_px or x1 >= width - 1 - margin_px or y1 >= height - 1 - margin_px


# ---------------------------------------------------------------------------
# 2. Region extraction (replaceable: a mask or keypoint extractor can implement the same interface)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DepthRegion:
    raw: np.ndarray  # strided VIEW into the depth frame
    u0: int
    v0: int
    stride: int

    def pixel_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        rows, cols = self.raw.shape
        u = self.u0 + self.stride * np.arange(cols)
        v = self.v0 + self.stride * np.arange(rows)
        uu, vv = np.meshgrid(u, v)
        return uu, vv


class RegionExtractor(Protocol):
    def __call__(self, bbox_px: Tuple[float, float, float, float], depth_raw: np.ndarray,
                 params: Layer1Params) -> Optional[DepthRegion]: ...


def inset_box_region(bbox_px, depth_raw: np.ndarray, params: Layer1Params) -> Optional[DepthRegion]:
    """Inset the box by bbox_inset_fraction per side and sample every sample_stride_px pixels (a view, no copy)."""
    x0, y0, x1, y1 = bbox_px
    f = params.bbox_inset_fraction
    dx, dy = f * (x1 - x0), f * (y1 - y0)
    h, w = depth_raw.shape
    u_lo, u_hi = max(math.ceil(x0 + dx), 0), min(math.floor(x1 - dx), w - 1)
    v_lo, v_hi = max(math.ceil(y0 + dy), 0), min(math.floor(y1 - dy), h - 1)
    if u_hi < u_lo or v_hi < v_lo:
        return None
    s = params.sample_stride_px
    return DepthRegion(depth_raw[v_lo:v_hi + 1:s, u_lo:u_hi + 1:s], u_lo, v_lo, s)


# ---------------------------------------------------------------------------
# 3. Invalid-depth rejection
# ---------------------------------------------------------------------------


def region_depths_m(region: DepthRegion, scale: float, params: Layer1Params) -> Tuple[np.ndarray, np.ndarray]:
    """Depth in metres for the sampled region (small copy) and the validity mask: finite, > 0, inside
    [depth_min_m, depth_max_m]. Zero raw values are the sensor's 'no depth' marker."""
    z = region.raw.astype(np.float64) * scale
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(z) & (z > 0) & (z >= params.depth_min_m) & (z <= params.depth_max_m)
    return z, valid


# ---------------------------------------------------------------------------
# 4. 1D depth clustering and cluster selection
# ---------------------------------------------------------------------------


def dense_mask(z_sorted: np.ndarray, gap_m: float, min_count: float) -> np.ndarray:
    """1D density test (DBSCAN-style core points): sample i is dense if at least min_count samples (itself
    included) lie within +/- gap_m / 2 of it. A sloped floor or flying pixels spread thinly over depth and can
    otherwise bridge the gap between a person and a wall behind them."""
    if min_count <= 1:
        return np.ones(z_sorted.size, dtype=bool)
    hi = np.searchsorted(z_sorted, z_sorted + gap_m / 2, side="right")
    lo = np.searchsorted(z_sorted, z_sorted - gap_m / 2, side="left")
    return (hi - lo) >= min_count


def cluster_sorted_depths(z_sorted: np.ndarray, gap_m: float) -> List[Tuple[int, int]]:
    """Split sorted depths at gaps larger than gap_m. Returns [start, end) index ranges, nearest first."""
    if z_sorted.size == 0:
        return []
    breaks = (np.nonzero(np.diff(z_sorted) > gap_m)[0] + 1).tolist()
    bounds = [0] + breaks + [int(z_sorted.size)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def edge_span(near: np.ndarray, valid: np.ndarray) -> float:
    """Largest, over the four box edges, share of lines (columns for top/bottom, rows for left/right) whose
    outermost valid pixel on that edge belongs to `near`. 1.0 = the nearer pixels form a band along that edge."""
    rows, cols = valid.shape
    best = 0.0
    for axis, from_end in ((0, True), (0, False), (1, False), (1, True)):  # bottom, top, left, right
        has = valid.any(axis=axis)
        if not has.any():
            continue
        if from_end:
            v = valid[::-1, :] if axis == 0 else valid[:, ::-1]
        else:
            v = valid
        first = np.argmax(v, axis=axis)
        n_along = rows if axis == 0 else cols
        idx = n_along - 1 - first if from_end else first
        lines = np.nonzero(has)[0]
        hit = near[idx[lines], lines] if axis == 0 else near[lines, idx[lines]]
        best = max(best, float(hit.mean()))
    return best


def strip_fraction(near: np.ndarray, far: np.ndarray) -> float:
    """Share of the rows containing `near` pixels that also have `far` pixels both left and right of them."""
    rows = np.nonzero(near.any(axis=1))[0]
    if rows.size == 0:
        return 0.0
    cols = near.shape[1]
    n = near[rows]
    cum = np.cumsum(far[rows], axis=1)
    r = np.arange(rows.size)
    lo = np.argmax(n, axis=1)
    hi = cols - 1 - np.argmax(n[:, ::-1], axis=1)
    left = (lo > 0) & (cum[r, np.maximum(lo - 1, 0)] > 0)
    right = (cum[:, -1] - cum[r, hi]) > 0
    return float((left & right).mean())


def select_cluster(sizes: Sequence[int], params: Layer1Params,
                   labels: Optional[np.ndarray] = None) -> Tuple[Optional[int], Optional[FailureReason]]:
    """Choose the survivor's depth cluster. `sizes` are cluster sizes ordered nearest first; `labels` is the
    sampled region with each pixel's cluster index (-1 = invalid), needed for the spatial rule.

    1. Clusters smaller than min_cluster_fraction of the valid pixels are not candidates (flying pixels, edges,
       small objects). If none qualifies, the depth is scattered: AMBIGUOUS.
    2. Two or more candidates with second/first size > ambiguity_ratio: no surface dominates the inset box and
       picking one (in particular the nearest) would be a guess: AMBIGUOUS, no point (spec 5.5).
    3. Otherwise the largest candidate D is chosen. If fewer than background_near_min_fraction of the valid pixels
       are nearer than D, D is the front-most surface: accept it.
    4. Something is in front of D. With cluster_rule = "spatial" the image layout of the nearer pixels N decides:
       a. OCCLUDER: N are the outermost valid pixels along >= occluder_edge_span_min of one box edge (typically a
          band across the bottom: a low wall in front of the person). D is the person behind it: accept D.
       b. THIN OBJECT IN FRONT: in >= strip_rows_fraction_min of the rows containing N, D pixels lie both left and
          right of N (a narrow strip with background on both sides). The detection is the thin object, so choose
          the largest nearer cluster if it has >= min_valid_pixels, else BACKGROUND_DOMINANT.
       The band test runs first: a band along an edge cannot have background on both sides in most rows.
    5. Neither layout (or cluster_rule = "fraction_only", the 2.3 behaviour): if D holds more than
       background_fraction_max of the valid pixels, the box mostly shows what is behind the nearer object and the
       two cases (wall behind a thin person / person behind an occluder) cannot be separated: BACKGROUND_DOMINANT
       (the safe failure). Otherwise accept D.
    """
    n = float(sum(sizes))
    if n <= 0:
        return None, FailureReason.DEPTH_NO_VALID
    cands = [i for i, s in enumerate(sizes) if s >= params.min_cluster_fraction * n]
    if not cands:
        return None, FailureReason.DEPTH_AMBIGUOUS_CLUSTERS
    ranked = sorted(cands, key=lambda i: sizes[i], reverse=True)
    if len(ranked) >= 2 and sizes[ranked[1]] / sizes[ranked[0]] > params.ambiguity_ratio:
        return None, FailureReason.DEPTH_AMBIGUOUS_CLUSTERS
    chosen = ranked[0]
    if sum(sizes[:chosen]) / n < params.background_near_min_fraction:
        return chosen, None
    if params.cluster_rule == "spatial" and labels is not None:
        valid = labels >= 0
        near = valid & (labels < chosen)
        if edge_span(near, valid) >= params.occluder_edge_span_min:
            return chosen, None
        if strip_fraction(near, labels == chosen) >= params.strip_rows_fraction_min:
            k = max(range(chosen), key=lambda i: sizes[i])
            if sizes[k] >= params.min_valid_pixels:
                return k, None
            return None, FailureReason.DEPTH_BACKGROUND_DOMINANT
    if sizes[chosen] / n > params.background_fraction_max:
        return None, FailureReason.DEPTH_BACKGROUND_DOMINANT
    return chosen, None


# ---------------------------------------------------------------------------
# 5-7. Robust depth, back-projection, body offset
# ---------------------------------------------------------------------------


def robust_depth(z: np.ndarray) -> Tuple[float, float]:
    """Median and MAD-based standard deviation."""
    med = float(np.median(z))
    return med, MAD_TO_SIGMA * float(np.median(np.abs(z - med)))


def back_project(u: float, v: float, z: float, intr: CameraIntrinsics) -> np.ndarray:
    """Pinhole: X = (u - cx) Z / fx, Y = (v - cy) Z / fy, Z (optical frame: x right, y down, z forward).
    No undistortion (see architecture.md Layer 1 deviations)."""
    return np.array([(u - intr.cx_px) * z / intr.fx_px, (v - intr.cy_px) * z / intr.fy_px, z])


def apply_body_offset(p: np.ndarray, offset_m: float) -> np.ndarray:
    """Move p further along its camera ray by offset_m (front surface -> body centre)."""
    if offset_m == 0:
        return p
    return p + offset_m * p / np.linalg.norm(p)


# ---------------------------------------------------------------------------
# 8. Ground reference (interface only)
# ---------------------------------------------------------------------------


def ground_reference(method: str, p_camera_m: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[FailureReason]]:
    """Only "none" is implemented: no ground reference, warning GROUND_REF_UNAVAILABLE. Other configured
    methods ("bbox_bottom_ray", "keypoints") behave the same until implemented."""
    return None, FailureReason.GROUND_REF_UNAVAILABLE


# ---------------------------------------------------------------------------
# 9. Covariance and quality
# ---------------------------------------------------------------------------


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Lower weighted median: the smallest value whose cumulative weight reaches half of the total."""
    order = np.argsort(values, kind="stable")
    cum = np.cumsum(weights[order])
    return float(values[order][int(np.searchsorted(cum, 0.5 * cum[-1]))])


def depth_noise_sigma(z: float, coeffs: Sequence[float]) -> float:
    return float(sum(c * z ** k for k, c in enumerate(coeffs)))


def measurement_covariance(u: float, v: float, z: float, sigma_z: float, sigma_u_px: float, sigma_v_px: float,
                           intr: CameraIntrinsics) -> np.ndarray:
    """First-order propagation through the pinhole back-projection: cov = J diag(su^2, sv^2, sz^2) J^T with
    J = d(X, Y, Z)/d(u, v, z) = [[z/fx, 0, (u-cx)/fx], [0, z/fy, (v-cy)/fy], [0, 0, 1]]."""
    j = np.array([[z / intr.fx_px, 0.0, (u - intr.cx_px) / intr.fx_px],
                  [0.0, z / intr.fy_px, (v - intr.cy_px) / intr.fy_px],
                  [0.0, 0.0, 1.0]])
    return j @ np.diag([sigma_u_px ** 2, sigma_v_px ** 2, sigma_z ** 2]) @ j.T


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def estimate_survivor_point(det: Detection, depth: DepthMeasurement, intr: CameraIntrinsics,
                            params: Layer1Params, extractor: RegionExtractor = inset_box_region,
                            floor: Optional[FloorPlane] = None) -> Layer1Result:
    """Camera-frame survivor point for one detection.

    point_method "median_pixel": depth = median of the chosen cluster; pixel = median (u, v) of that cluster's own
      pixels (same surface as the depth, e.g. when the lower part of the box is a nearer occluder).
      Covariance: sigma_z^2 = sigma_noise(z)^2 + (1.4826 MAD_z)^2 (pixels are correlated, no 1/n);
      sigma_u^2 = stride^2/12 + (1.4826 MAD_u)^2 / n_cluster, same for v; propagated by measurement_covariance().
    point_method "cluster_area_weighted": as cluster_centroid, but each point is weighted by depth^2 (its surface
      footprint), using a weighted per-axis median, so the near part of an extended surface (which covers more
      pixels) does not dominate; covariance uses the same weights.
    point_method "cluster_centroid": back-project every sampled pixel of the cluster and take the per-axis median
      of the 3D points. Per-axis median rather than a trimmed mean: no trim fraction to tune, and robust to the
      tails a cluster still contains (flying pixels within the gap tolerance, a strip of floor merged with a prone
      body). Covariance: C = Cov(cluster points) + sigma_noise(z)^2 r r^T, r = unit ray to the point. It is the
      spread of the visible points, not its standard error: pixels are correlated and the body centre can lie
      anywhere within the visible extent.
    Both: the point is then moved along its ray by body_depth_offset_m. A box touching the image border adds the
    warning BBOX_TRUNCATED and multiplies the covariance by truncated_cov_scale.
    depth_quality = valid_fraction * cluster_share * 1 / (1 + 1.4826 MAD_z / cluster_gap_m), in [0, 1].
    """
    res = Layer1Result()
    reason = validate_inputs(det, depth, intr, params)
    if reason is not None:
        res.failure = reason
        return res
    region = extractor(det.bbox_px, depth.depth_raw, params)
    if region is None or region.raw.size == 0:
        res.failure = FailureReason.BBOX_INVALID
        return res
    z, valid = region_depths_m(region, depth.depth_scale_m_per_unit, params)
    n_valid = int(valid.sum())
    res.n_valid_px = n_valid
    res.valid_fraction = n_valid / z.size
    if n_valid == 0:
        res.failure = FailureReason.DEPTH_NO_VALID
        return res
    if n_valid < params.min_valid_pixels or res.valid_fraction < params.min_valid_fraction:
        res.failure = FailureReason.DEPTH_TOO_FEW_PIXELS
        return res
    uu, vv = region.pixel_coords()
    if floor is not None:  # remove floor pixels BEFORE clustering (sensor validity above is unchanged)
        pts = np.stack([(uu - intr.cx_px) * z / intr.fx_px, (vv - intr.cy_px) * z / intr.fy_px, z], axis=-1)
        is_floor = valid & (floor.heights(pts) < floor.margin_m)
        res.floor_removed_px = int(is_floor.sum())
        valid = valid & ~is_floor
        n_valid = int(valid.sum())
        if n_valid == 0:
            res.failure = FailureReason.DEPTH_NO_VALID
            return res
        if n_valid < params.min_valid_pixels:
            res.failure = FailureReason.DEPTH_TOO_FEW_PIXELS
            return res
    zv, uv, vv_ = z[valid], uu[valid], vv[valid]
    order = np.argsort(zv, kind="stable")
    zs = zv[order]
    dense = dense_mask(zs, params.cluster_gap_m, params.min_density_fraction * n_valid)
    order, zs = order[dense], zs[dense]  # sparse samples are excluded from every cluster
    clusters = cluster_sorted_depths(zs, params.cluster_gap_m)
    res.n_clusters = len(clusters)
    if not clusters:  # valid depth exists but nothing is dense: scattered
        res.failure = FailureReason.DEPTH_AMBIGUOUS_CLUSTERS
        return res
    lab_valid = np.full(n_valid, -2, dtype=np.int32)  # -2 = valid but sparse (no cluster)
    for ci, (s0, e0) in enumerate(clusters):
        lab_valid[order[s0:e0]] = ci
    labels = np.full(z.shape, -1, dtype=np.int32)
    labels[valid] = lab_valid
    chosen, reason = select_cluster([e - s for s, e in clusters], params, labels)
    if reason is not None:
        res.failure = reason
        return res
    s, e = clusters[chosen]
    idx = order[s:e]
    zc, uc, vc = zs[s:e], uv[idx].astype(np.float64), vv_[idx].astype(np.float64)
    if params.keep_points > 0:
        rc = np.lexsort((uc, vc))  # row-major pixel order: deterministic, independent of depth ties
        cpts = np.stack([(uc[rc] - intr.cx_px) * zc[rc] / intr.fx_px, (vc[rc] - intr.cy_px) * zc[rc] / intr.fy_px,
                         zc[rc]], axis=1)
        res.cluster_points_camera_m = subsample_points(cpts, params.keep_points)
    z_med, sigma_spread = robust_depth(zc)
    u_med, v_med = float(np.median(uc)), float(np.median(vc))
    n_c = e - s

    if params.point_method in ("cluster_centroid", "cluster_area_weighted"):
        pts = np.stack([(uc - intr.cx_px) * zc / intr.fx_px, (vc - intr.cy_px) * zc / intr.fy_px, zc], axis=1)
        if params.point_method == "cluster_centroid":
            centre = np.median(pts, axis=0)
            w = None
        else:
            w = zc ** 2  # pixel footprint area grows with depth^2
            centre = np.array([weighted_median(pts[:, a], w) for a in range(3)])
        p = apply_body_offset(centre, params.body_depth_offset_m)
        ray = p / np.linalg.norm(p)
        spread = np.cov(pts.T, aweights=w) if n_c >= 2 else np.zeros((3, 3))
        cov = spread + depth_noise_sigma(float(centre[2]), params.depth_noise_coeffs) ** 2 * np.outer(ray, ray)
    else:
        p = apply_body_offset(back_project(u_med, v_med, z_med, intr), params.body_depth_offset_m)
        sigma_z = math.hypot(depth_noise_sigma(z_med, params.depth_noise_coeffs), sigma_spread)
        q = params.sample_stride_px ** 2 / 12.0
        sigma_u = math.sqrt(q + (MAD_TO_SIGMA * float(np.median(np.abs(uc - u_med)))) ** 2 / n_c)
        sigma_v = math.sqrt(q + (MAD_TO_SIGMA * float(np.median(np.abs(vc - v_med)))) ** 2 / n_c)
        cov = measurement_covariance(u_med, v_med, float(p[2]), sigma_z, sigma_u, sigma_v, intr)

    h, w = depth.depth_raw.shape
    if bbox_truncated(det.bbox_px, w, h, params.truncation_margin_px):
        res.warnings.append(FailureReason.BBOX_TRUNCATED)
        cov = cov * params.truncated_cov_scale
    res.p_camera_m = p
    res.cov_camera_m2 = cov
    res.depth_m = z_med
    res.pixel_uv = (u_med, v_med)
    res.depth_quality = float(res.valid_fraction * (n_c / n_valid) / (1.0 + sigma_spread / params.cluster_gap_m))
    res.ground_ref_camera_m, warn = ground_reference(params.ground_ref_method, p)
    if warn is not None:
        res.warnings.append(warn)
    return res
