"""Venue grid registration from wall alignment (substep 4.3). Library: numpy only, no simulator imports.

ASSUMPTION (organiser question O1, NOT confirmed, gated by `grid.wall_alignment_enabled`, OFF by default): the walls of
the arena lie on the lines of the 1 m grid, and the measured wall points (scan hits or occupied cells, map frame) lie
on the wall centre line, i.e. a wall of thickness t is not seen t/2 off its line. If that is false the estimate is
biased by up to t/2 and the quality score cannot detect it.

Method (all in the "h frame" h = R(-yaw) (p_map - origin), the grid frame before axis flips; grid lines are at integer
multiples of the cell size on both axes, flips do not change that):
 1. Local direction of every wall point from a PCA of its neighbours within `radius_m` (linearity filter removes
    isolated outliers, corners and clutter).
 2. Dominant orientation modulo 90 degrees: circular mean of exp(4 i phi) (stage 1, coarse).
 3. Yaw refinement: scan the yaw correction around stage 1 and keep the angle that makes the coordinates of the
    wall points of each direction class concentrate on integer lines (resultant length of exp(2 pi i x/cell)).
 4. Offset modulo one cell on each axis: circular mean, then a Tukey-weighted refinement.
 5. The whole-cell ambiguity is resolved by the coarse registration: the offset is the representative in
    [-cell/2, cell/2) of the fitted phase. ASSUMPTION: the coarse origin is within half a cell of the truth after the
    yaw correction, and the coarse yaw is within 45 degrees. If the coarse uncertainty says otherwise
    (3 sigma >= half a cell / 45 degrees) no update is made (REG_COARSE_AMBIGUOUS).
 6. Uncertainty by a deterministic block bootstrap (1 m blocks) of steps 3-4. It is a STATISTICAL uncertainty only; it
    does not include the assumption (walls on lines, wall thickness) being wrong.
A failed estimate is never hidden: `registration` is None and `reason` says why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .config import Config
from .grid import GridRegistration
from .types import FailureReason

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class WallAlignParams:
    """All values are ASSUMPTION (tuned on synthetic mazes in substep 4.3, simulation evidence only)."""

    radius_m: float = 0.25  # PCA neighbourhood
    min_neighbours: int = 6
    min_linearity: float = 0.8
    max_points: int = 4000  # deterministic subsample bound
    yaw_search_rad: float = math.radians(3.0)  # refinement range around stage 1
    yaw_step_rad: float = math.radians(0.05)
    axis_tolerance_rad: float = math.radians(25.0)  # direction class membership
    min_points_per_axis: int = 40  # linear points needed in each direction class
    inlier_tol_m: float = 0.10  # quality = fraction of class points within this of a grid line
    min_quality: float = 0.5
    tukey_scale_m: float = 0.08
    bootstrap: int = 16
    block_m: float = 1.0
    seed: int = 0
    coarse_k_sigma: float = 3.0  # coarse-uncertainty check

    @classmethod
    def from_config(cls, cfg: Config, **overrides) -> "WallAlignParams":
        kw = dict(min_quality=cfg.value("grid.wall_align_min_quality"))
        kw.update(overrides)
        return cls(**kw)


@dataclass(frozen=True)
class WallAlignResult:
    registration: Optional[GridRegistration]  # None = no update (see reason)
    reason: Optional[FailureReason]
    detail: str
    quality: float = 0.0  # min over the two axes of the inlier fraction, 0..1
    inlier_fraction_xy: Tuple[float, float] = (0.0, 0.0)
    n_points: int = 0
    n_linear: int = 0
    n_axis_xy: Tuple[int, int] = (0, 0)
    yaw_correction_rad: float = 0.0
    origin_shift_m: float = 0.0
    yaw_sigma_rad: Optional[float] = None
    origin_sigma_m: Optional[float] = None
    notes: Tuple[str, ...] = ()


def _rot(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _wrap(a, period):
    """Representative of a in [-period/2, period/2)."""
    return (a + period / 2.0) % period - period / 2.0


def local_directions(h: np.ndarray, radius_m: float, min_neighbours: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per point: line direction phi (rad, modulo pi), linearity in [0, 1] and a validity mask (enough neighbours).
    Brute-force neighbour sums in row chunks (N <= params.max_points)."""
    n = len(h)
    feats = np.stack([np.ones(n), h[:, 0], h[:, 1], h[:, 0] ** 2, h[:, 0] * h[:, 1], h[:, 1] ** 2], axis=1)
    phi, lin, ok = np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool)
    r2 = radius_m ** 2
    for a in range(0, n, 500):
        d = h[a:a + 500, None, :] - h[None, :, :]
        m = ((d ** 2).sum(axis=2) <= r2).astype(float)
        s = m @ feats
        cnt = s[:, 0]
        good = cnt >= min_neighbours
        cnt = np.maximum(cnt, 1.0)
        mx, my = s[:, 1] / cnt, s[:, 2] / cnt
        cxx, cxy, cyy = s[:, 3] / cnt - mx * mx, s[:, 4] / cnt - mx * my, s[:, 5] / cnt - my * my
        tr = cxx + cyy
        disc = np.sqrt((cxx - cyy) ** 2 + 4 * cxy ** 2)
        phi[a:a + 500] = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
        lin[a:a + 500] = np.where(tr > 1e-12, disc / np.maximum(tr, 1e-12), 0.0)
        ok[a:a + 500] = good
    return phi, lin, ok


def _axis_stats(coord: np.ndarray, w: np.ndarray, cell: float, tukey: float) -> Tuple[float, float]:
    """(epsilon, resultant length R): grid-line phase of weighted coordinates, circular mean then Tukey refinement.
    epsilon in [-cell/2, cell/2): coord - epsilon lies on the integer lines."""
    z = np.sum(w * np.exp(1j * TWO_PI * coord / cell))
    tot = float(np.sum(w))
    if tot <= 0:
        return 0.0, 0.0
    eps = float(np.angle(z) / TWO_PI * cell)
    r_len = float(abs(z) / tot)
    for _ in range(6):
        res = _wrap(coord - eps, cell)
        u = res / tukey
        wt = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0) * w
        sw = float(np.sum(wt))
        if sw <= 0:
            break
        eps += float(np.sum(wt * res) / sw)
    return float(_wrap(eps, cell)), r_len


def _fit(h: np.ndarray, in_x: np.ndarray, in_y: np.ndarray, w: np.ndarray, alphas: np.ndarray, cell: float,
         tukey: float) -> Tuple[float, float, float, float, float]:
    """Best yaw correction alpha (h' = R(-alpha) h), offsets (eps_x, eps_y) and the two resultant lengths."""
    hx, hy = h[in_x], h[in_y]
    wx, wy = w[in_x], w[in_y]
    best = np.zeros(len(alphas))
    for k, a in enumerate(alphas):
        c, s = math.cos(a), math.sin(a)
        zx = np.sum(wx * np.exp(1j * TWO_PI * (c * hx[:, 0] + s * hx[:, 1]) / cell))
        zy = np.sum(wy * np.exp(1j * TWO_PI * (-s * hy[:, 0] + c * hy[:, 1]) / cell))
        best[k] = abs(zx) / max(np.sum(wx), 1e-12) + abs(zy) / max(np.sum(wy), 1e-12)
    i = int(np.argmax(best))
    a = float(alphas[i])
    if 0 < i < len(alphas) - 1:  # parabolic peak refinement
        y0, y1, y2 = best[i - 1], best[i], best[i + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            a += float(0.5 * (y0 - y2) / den * (alphas[1] - alphas[0]))
    c, s = math.cos(a), math.sin(a)
    ex, rx = _axis_stats(c * hx[:, 0] + s * hx[:, 1], wx, cell, tukey)
    ey, ry = _axis_stats(-s * hy[:, 0] + c * hy[:, 1], wy, cell, tukey)
    return a, ex, ey, rx, ry


def _registration_from(coarse: GridRegistration, alpha: float, eps: np.ndarray) -> Tuple[float, np.ndarray]:
    """theta' = theta + alpha, o' = o + R(theta') eps (derivation in the module docstring)."""
    theta = coarse.yaw_rad + alpha
    o = np.asarray(coarse.origin_in_map_m) + _rot(theta) @ eps
    return theta, o


def estimate_wall_alignment(points_map_xy: np.ndarray, coarse: GridRegistration,
                            params: Optional[WallAlignParams] = None) -> WallAlignResult:
    """Refine `coarse` (origin, yaw, with its uncertainty) from 2D wall points in the map frame. See the module
    docstring for the method and the assumption. Deterministic (seeded subsampling and bootstrap)."""
    p = params or WallAlignParams()
    cell = coarse.cell_size_m
    pts = np.asarray(points_map_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return WallAlignResult(None, FailureReason.WALL_POINTS_INSUFFICIENT, "points must have shape (N, 2)")
    pts = pts[:, :2][np.all(np.isfinite(pts[:, :2]), axis=1)]
    n_all = len(pts)

    def fail(reason, detail, **kw):
        return WallAlignResult(None, reason, detail, n_points=n_all, **kw)

    if coarse.yaw_sigma_rad is not None and p.coarse_k_sigma * coarse.yaw_sigma_rad >= math.pi / 4:
        return fail(FailureReason.REG_COARSE_AMBIGUOUS,
                    f"{p.coarse_k_sigma:g} x yaw sigma >= 45 deg: orientation modulo 90 deg is ambiguous")
    sig_o = coarse.origin_sigma_m
    if coarse.origin_cov_m2 is not None:
        sig_o = float(math.sqrt(max(np.linalg.eigvalsh(np.array(coarse.origin_cov_m2)))))
    if sig_o is not None and p.coarse_k_sigma * sig_o >= cell / 2:
        return fail(FailureReason.REG_COARSE_AMBIGUOUS,
                    f"{p.coarse_k_sigma:g} x origin sigma >= half a cell: whole-cell ambiguity cannot be resolved")
    if n_all < p.min_points_per_axis * 2:
        return fail(FailureReason.WALL_POINTS_INSUFFICIENT, f"{n_all} wall points")
    if n_all > p.max_points:
        pts = pts[np.linspace(0, n_all - 1, p.max_points).astype(int)]
    h = (pts - np.asarray(coarse.origin_in_map_m)) @ _rot(-coarse.yaw_rad).T

    phi, lin, ok = local_directions(h, p.radius_m, p.min_neighbours)
    keep = ok & (lin >= p.min_linearity)
    n_lin = int(np.sum(keep))
    if n_lin < 2 * p.min_points_per_axis:
        return fail(FailureReason.WALL_POINTS_INSUFFICIENT, f"{n_lin} points on straight wall segments", n_linear=n_lin)
    h, phi, lin = h[keep], phi[keep], lin[keep]
    z4 = np.sum(lin * np.exp(4j * phi))
    delta = float(np.angle(z4) / 4.0)  # direction of the walls relative to the h axes, modulo 90 deg
    rel = phi - delta
    horiz = np.abs(_wrap(rel, math.pi)) < p.axis_tolerance_rad  # line along h x: coordinate y is on a grid line
    vert = np.abs(_wrap(rel - math.pi / 2, math.pi)) < p.axis_tolerance_rad
    n_x, n_y = int(np.sum(vert)), int(np.sum(horiz))  # x-class = vertical lines (their x coordinate is a grid line)
    if min(n_x, n_y) < p.min_points_per_axis:
        if max(n_x, n_y) >= p.min_points_per_axis:
            return fail(FailureReason.WALL_DIRECTION_AMBIGUOUS,
                        f"walls in one direction only ({n_x} / {n_y} points per direction): one offset axis is "
                        "unobservable", n_linear=n_lin, n_axis_xy=(n_x, n_y))
        return fail(FailureReason.WALL_POINTS_INSUFFICIENT, f"{n_x} / {n_y} points per wall direction",
                    n_linear=n_lin, n_axis_xy=(n_x, n_y))

    alphas = delta + np.arange(-p.yaw_search_rad, p.yaw_search_rad + 1e-12, p.yaw_step_rad)
    w1 = np.ones(len(h))
    a, ex, ey, rx, ry = _fit(h, vert, horiz, w1, alphas, cell, p.tukey_scale_m)
    c, s = math.cos(a), math.sin(a)
    res_x = _wrap(c * h[vert, 0] + s * h[vert, 1] - ex, cell)
    res_y = _wrap(-s * h[horiz, 0] + c * h[horiz, 1] - ey, cell)
    inl_x = float(np.mean(np.abs(res_x) <= p.inlier_tol_m))
    inl_y = float(np.mean(np.abs(res_y) <= p.inlier_tol_m))
    quality = min(inl_x, inl_y)
    info = dict(quality=quality, inlier_fraction_xy=(inl_x, inl_y), n_points=n_all, n_linear=n_lin,
                n_axis_xy=(n_x, n_y))
    if quality < p.min_quality:
        return WallAlignResult(None, FailureReason.WALL_ALIGNMENT_LOW_QUALITY,
                               f"quality {quality:.2f} < {p.min_quality:.2f}: walls do not fit grid lines", **info)

    # Bootstrap over 1 m blocks of the aligned points (statistical uncertainty).
    blocks = np.floor((c * h[:, 0] + s * h[:, 1]) / p.block_m).astype(np.int64) * 100003 + \
        np.floor((-s * h[:, 0] + c * h[:, 1]) / p.block_m).astype(np.int64)
    ub, inv = np.unique(blocks, return_inverse=True)
    rng = np.random.default_rng(p.seed)
    fine = a + np.arange(-math.radians(0.5), math.radians(0.5) + 1e-12, p.yaw_step_rad)
    yaws, origins = [], []
    for _ in range(p.bootstrap):
        counts = np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub)).astype(float)
        w = counts[inv]
        if np.sum(w[vert]) < 1 or np.sum(w[horiz]) < 1:
            continue
        ab, exb, eyb, _, _ = _fit(h, vert, horiz, w, fine, cell, p.tukey_scale_m)
        th, o = _registration_from(coarse, ab, np.array([exb, eyb]))
        yaws.append(th)
        origins.append(o)
    yaw_sigma = float(np.std(yaws, ddof=1)) if len(yaws) >= 3 else None
    origin_sigma = None
    if len(origins) >= 3:
        origin_sigma = float(math.sqrt(0.5 * np.sum(np.var(np.array(origins), axis=0, ddof=1))))
    theta, origin = _registration_from(coarse, a, np.array([ex, ey]))
    reg = coarse.updated(origin_in_map_m=(float(origin[0]), float(origin[1])), yaw_rad=float(theta),
                         origin_sigma_m=origin_sigma, yaw_sigma_rad=yaw_sigma, origin_cov_m2=None)
    notes = ("assumes walls on grid lines and wall points on the wall centre line (Q O1)",
             "whole-cell offset taken from the coarse registration (|error| < cell/2 assumed)",
             "uncertainty is statistical (block bootstrap) only")
    return WallAlignResult(reg, None, "ok", yaw_correction_rad=float(a),
                           origin_shift_m=float(np.linalg.norm(origin - np.asarray(coarse.origin_in_map_m))),
                           yaw_sigma_rad=yaw_sigma, origin_sigma_m=origin_sigma, notes=notes, **info)


# ---------------------------------------------------------------------------
# Update flow: hysteresis
# ---------------------------------------------------------------------------


def registration_change(old: GridRegistration, new: GridRegistration) -> Tuple[float, float]:
    """(|yaw change| in rad wrapped to [-pi, pi), |origin change| in m)."""
    dyaw = abs(float(_wrap(new.yaw_rad - old.yaw_rad, TWO_PI)))
    doff = float(np.linalg.norm(np.asarray(new.origin_in_map_m) - np.asarray(old.origin_in_map_m)))
    return dyaw, doff


def update_needed(old: GridRegistration, new: GridRegistration, min_yaw_rad: float, min_offset_m: float) -> bool:
    """Hysteresis: a new registration is worth a Layer 5 recompute only if yaw changed by >= min_yaw_rad or the origin
    by >= min_offset_m (both thresholds are ASSUMPTIONS, grid.reg_update_min_*)."""
    dyaw, doff = registration_change(old, new)
    return dyaw >= min_yaw_rad or doff >= min_offset_m


__all__ = ["WallAlignParams", "WallAlignResult", "estimate_wall_alignment", "local_directions", "registration_change",
           "update_needed"]
