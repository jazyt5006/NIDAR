"""Map-frame footprint refinement for extended clusters (substep 3.1b; architecture.md "Layer 1 refinement").

Layer 1 measures the visible surface. For an extended body (a prone person, 1.7 m long) the visible pixels crowd
toward its near end, so a pixel median falls short of the body centre. Here the kept cluster points are moved to
the map frame with the capture-time pose, projected onto the floor plane (x, y), and analysed along their principal
axis. Only EXTENDED clusters (robust extent along the axis >= l1b.extended_min_m) are changed; every other
observation is left bit-identical. Pure library code: no simulator imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import Config
from .geometry import weighted_median
from .types import FailureReason, SurvivorObservation


@dataclass(frozen=True)
class FootprintParams:
    method: str  # "midpoint" or "area_weighted"
    extent_lo_pct: float
    extent_hi_pct: float
    extended_min_m: float
    axis_sigma_fraction: float
    body_offset_m: float

    @classmethod
    def from_config(cls, cfg: Config) -> "FootprintParams":
        v = cfg.value
        return cls(v("l1b.method"), v("l1b.extent_percentile_lo"), v("l1b.extent_percentile_hi"),
                   v("l1b.extended_min_m"), v("l1b.extended_axis_sigma_fraction"), v("l1.body_depth_offset_m"))


@dataclass(frozen=True, eq=False)
class Footprint:
    mean: np.ndarray  # (2,) mean of the projected points
    axis: np.ndarray  # (2,) unit principal axis (largest spread)
    across: np.ndarray  # (2,) unit, perpendicular (axis rotated +90 degrees)
    extent_along_m: float  # robust (percentile) extent along the axis
    extent_across_m: float
    extended: bool
    centre: np.ndarray  # (2,) refined centre (before the body offset)


def principal_axis(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean, axis, across) from the 2x2 covariance eigen-decomposition. The axis sign is fixed (x > 0, or y > 0
    when x == 0) so the result is deterministic."""
    mean = xy.mean(axis=0)
    w, v = np.linalg.eigh(np.cov((xy - mean).T))
    axis = v[:, int(np.argmax(w))]
    if axis[0] < 0 or (axis[0] == 0 and axis[1] < 0):
        axis = -axis
    return mean, axis, np.array([-axis[1], axis[0]])


def analyse_footprint(xy: np.ndarray, p: FootprintParams, weights: Optional[np.ndarray] = None) -> Footprint:
    """Along the axis: percentile extent; centre = midpoint of [p_lo, p_hi] ("midpoint") or the weighted median
    ("area_weighted", weights = depth^2). Across the axis: median (the visible side of a body is one-sided either way)."""
    mean, axis, across = principal_axis(xy)
    s, r = (xy - mean) @ axis, (xy - mean) @ across
    lo, hi = np.percentile(s, [p.extent_lo_pct, p.extent_hi_pct])
    r_lo, r_hi = np.percentile(r, [p.extent_lo_pct, p.extent_hi_pct])
    along = 0.5 * (lo + hi) if p.method == "midpoint" else weighted_median(s, weights)
    centre = mean + along * axis + float(np.median(r)) * across
    extent = float(hi - lo)
    return Footprint(mean, axis, across, extent, float(r_hi - r_lo), extent >= p.extended_min_m, centre)


def refine_observation(obs: SurvivorObservation, t_map_camera: np.ndarray, p: FootprintParams) -> bool:
    """Refine obs.p_map_m / cov_map_m2 in place if its cluster is EXTENDED; returns True if refined.

    Body offset: Layer 1 shifted its point along the camera ray by the body-depth offset. For an extended body the
    along-axis position now comes from the extent, so only the ACROSS-axis component of that shift is kept:
    centre += offset * (v . across) * across, with v the unit horizontal viewing direction (camera -> centre). Seen
    from the side, the visible near side is moved toward the body's mid-line; seen end-on, nothing moves.
    Covariance: the Layer 2 covariance plus (axis_sigma_fraction * extent)^2 along the axis (the far end may be
    hidden, so the extent itself is uncertain).
    """
    pts = obs.cluster_points_camera_m
    if pts is None or len(pts) < 3 or obs.p_map_m is None or obs.cov_map_m2 is None:
        return False
    pm = pts @ t_map_camera[:3, :3].T + t_map_camera[:3, 3]
    fp = analyse_footprint(pm[:, :2], p, weights=pts[:, 2] ** 2)
    if not fp.extended:
        return False
    centre = fp.centre.copy()
    v = centre - t_map_camera[:2, 3]
    n = float(np.linalg.norm(v))
    if n > 0 and p.body_offset_m:
        centre += p.body_offset_m * float(v @ fp.across) / n * fp.across
    cov = np.array(obs.cov_map_m2, dtype=float)
    cov[:2, :2] += (p.axis_sigma_fraction * fp.extent_along_m) ** 2 * np.outer(fp.axis, fp.axis)
    obs.p_map_m = np.array([centre[0], centre[1], obs.p_map_m[2]])
    obs.cov_map_m2 = cov
    obs.warnings.append(FailureReason.EXTENDED_CLUSTER)
    return True
