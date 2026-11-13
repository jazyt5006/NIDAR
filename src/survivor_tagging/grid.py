"""Layer 3: NIDAR grid registration (spec Part 7, 8.7; architecture.md "Layer 3").

Pure library code: no simulator, baseline or ROS imports. Map -> grid definition (docs/design/conventions.md):

    d      = p_map_xy - origin_in_map_m
    g_rot  = R(-yaw) d          yaw = angle from map +x to grid +x, counter-clockwise about map +z
    g      = (s_x g_rot_x, s_y g_rot_y),  s = -1 where grid.axis_flip_* is true, else +1
    cell   = (floor(g_x / cell_size), floor(g_y / cell_size)) + index_base      (boundary rule "floor")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config, ConfigError
from .types import FailureReason, SurvivorObservation

Cell = Tuple[int, int]
PROB_WINDOW_SIGMA = 5.0  # candidate cells cover mean +/- 5 sigma per grid axis (tail mass < 6e-7 per side)
GL_NODES = 24  # Gauss-Legendre nodes per x-cell for correlated covariances
_GL_RULE = np.polynomial.legendre.leggauss(GL_NODES)  # constant; computed once (substep 3.2, identical results)

_REQUIRED = ("grid.origin_in_map_m", "grid.yaw_map_to_grid_rad", "grid.axis_flip_x", "grid.axis_flip_y",
             "grid.index_base", "grid.numbering", "grid.registration_valid")


def _rot(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


@dataclass(frozen=True)
class GridRegistration:
    origin_in_map_m: Tuple[float, float]
    yaw_rad: float
    flip_x: bool = False
    flip_y: bool = False
    cell_size_m: float = 1.0
    n_cells_x: Optional[int] = None  # None: extent unknown, out-of-grid cannot be decided
    n_cells_y: Optional[int] = None
    index_base: int = 0
    numbering: str = "ij"
    boundary_rule: str = "floor"
    version: int = 0
    valid: bool = False
    missing: Tuple[str, ...] = ()  # unset parameters replaced by provisional values (diagnostics only)
    # Registration uncertainty (substep 4.3). None = unknown (REG_UNCERTAINTY_UNKNOWN; treated as no extra error in the
    # probabilities, never as a made-up value). origin_cov_m2 (2x2, map frame, m^2) overrides origin_sigma_m.
    origin_sigma_m: Optional[float] = None
    yaw_sigma_rad: Optional[float] = None
    origin_cov_m2: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None

    @classmethod
    def from_config(cls, cfg: Config) -> "GridRegistration":
        """Unset parameters do not raise: a provisional registration (origin 0, yaw 0, no flips, base 0, "ij") is
        built for diagnostics only, valid = False, so every cell it produces carries GRID_UNREGISTERED."""
        v = cfg.value
        missing = tuple(n for n in _REQUIRED if v(n) is None)
        origin = v("grid.origin_in_map_m")
        return cls(
            origin_in_map_m=tuple(origin) if origin is not None else (0.0, 0.0),
            yaw_rad=v("grid.yaw_map_to_grid_rad") or 0.0,
            flip_x=bool(v("grid.axis_flip_x")), flip_y=bool(v("grid.axis_flip_y")),
            cell_size_m=v("grid.cell_size_m"), n_cells_x=v("grid.n_cells_x"), n_cells_y=v("grid.n_cells_y"),
            index_base=v("grid.index_base") or 0, numbering=v("grid.numbering") or "ij",
            boundary_rule=v("grid.boundary_rule"), version=v("grid.registration_version"),
            valid=bool(v("grid.registration_valid")) and not missing, missing=missing,
            origin_sigma_m=v("grid.origin_sigma_m"), yaw_sigma_rad=v("grid.yaw_sigma_rad"),
        )

    def updated(self, **changes) -> "GridRegistration":
        """New registration with the changes applied and version + 1."""
        return replace(self, version=self.version + 1, **changes)

    # --- frames -------------------------------------------------------------------------------------------
    def _signs(self) -> np.ndarray:
        return np.array([-1.0 if self.flip_x else 1.0, -1.0 if self.flip_y else 1.0])

    def jacobian(self) -> np.ndarray:
        """d(grid xy)/d(map xy) = diag(s) R(-yaw)."""
        return np.diag(self._signs()) @ _rot(-self.yaw_rad)

    # --- registration uncertainty (substep 4.3) --------------------------------------------------------------
    @property
    def uncertainty_known(self) -> bool:
        """True if both the origin and the yaw uncertainty are configured."""
        return (self.origin_sigma_m is not None or self.origin_cov_m2 is not None) and self.yaw_sigma_rad is not None

    def origin_covariance(self) -> np.ndarray:
        """2x2 covariance of the origin in the MAP frame (m^2); zeros if unknown."""
        if self.origin_cov_m2 is not None:
            return np.array(self.origin_cov_m2, dtype=float)
        return np.eye(2) * (self.origin_sigma_m or 0.0) ** 2

    def jacobian_yaw(self, p_map_xy: Sequence[float]) -> np.ndarray:
        """d(grid xy)/d(yaw) at map point p: diag(s) R'(-yaw) (p - o) with R'(-yaw) = d R(-yaw)/d yaw
        = [[-sin, cos], [-cos, -sin]]. (The task's R'(theta)(p - o) in the grid-to-map orientation; sign fixed by the
        finite-difference test.)"""
        d = np.asarray(p_map_xy, dtype=float)[:2] - np.asarray(self.origin_in_map_m)
        c, s_ = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return self._signs() * (np.array([[-s_, c], [-c, -s_]]) @ d)

    def jacobian_origin(self) -> np.ndarray:
        """d(grid xy)/d(origin xy) = -diag(s) R(-yaw) = -J (constant)."""
        return -self.jacobian()

    def position_covariance(self, p_map_xy: Sequence[float]) -> np.ndarray:
        """Grid-frame covariance (2x2, m^2) that the registration error adds to a point at map position p:
        J_o Sigma_o J_o^T + (dg/dyaw)(dg/dyaw)^T sigma_yaw^2 (origin and yaw errors independent: ASSUMPTION; an
        unknown sigma contributes 0). It is the SAME error for every tag (common mode), so it belongs on the fused
        position at the cell-probability step, never on the per-observation covariances."""
        jo = self.jacobian_origin()
        cov = jo @ self.origin_covariance() @ jo.T
        if self.yaw_sigma_rad:
            gy = self.jacobian_yaw(p_map_xy)
            cov = cov + np.outer(gy, gy) * self.yaw_sigma_rad ** 2
        return cov

    def map_to_grid(self, p_map_xy: Sequence[float]) -> np.ndarray:
        d = np.asarray(p_map_xy, dtype=float)[..., :2] - np.asarray(self.origin_in_map_m)
        return d @ self.jacobian().T

    def grid_to_map(self, g_xy: Sequence[float]) -> np.ndarray:
        g = np.asarray(g_xy, dtype=float)[..., :2]
        return g @ self.jacobian() + np.asarray(self.origin_in_map_m)  # J is orthogonal: J^-1 = J^T

    # --- cells --------------------------------------------------------------------------------------------
    def zero_based_cell(self, g_xy: Sequence[float]) -> Cell:
        if self.boundary_rule != "floor":
            raise ConfigError(f"grid.boundary_rule {self.boundary_rule!r} not implemented")
        return (int(math.floor(g_xy[0] / self.cell_size_m)), int(math.floor(g_xy[1] / self.cell_size_m)))

    def cell_index(self, g_xy: Sequence[float]) -> Cell:
        i, j = self.zero_based_cell(g_xy)
        return (i + self.index_base, j + self.index_base)

    def in_grid_zero_based(self, i0: int, j0: int) -> Optional[bool]:
        if self.n_cells_x is None or self.n_cells_y is None:
            return None
        return 0 <= i0 < self.n_cells_x and 0 <= j0 < self.n_cells_y

    def is_out_of_grid(self, g_xy: Sequence[float]) -> Optional[bool]:
        inside = self.in_grid_zero_based(*self.zero_based_cell(g_xy))
        return None if inside is None else not inside

    def cell_label(self, cell: Cell) -> str:
        """'ij': "i,j" (grid x index, grid y index). 'row_major_label': row * n_cells_x + col + index_base, with
        row = grid-y index and col = grid-x index (zero-based). 'alnum': not yet known (organiser question O1)."""
        if self.numbering == "ij":
            return f"{cell[0]},{cell[1]}"
        if self.numbering == "row_major_label":
            if self.n_cells_x is None:
                raise ConfigError("grid.numbering 'row_major_label' needs grid.n_cells_x")
            col, row = cell[0] - self.index_base, cell[1] - self.index_base
            return str(row * self.n_cells_x + col + self.index_base)
        raise ConfigError("grid.numbering 'alnum': the official alphanumeric cell convention is not yet known "
                          "(organiser question O1); configure 'ij' or 'row_major_label'")

    def boundary_distance_m(self, g_xy: Sequence[float]) -> float:
        """Distance from the point to the nearest cell edge."""
        c = self.cell_size_m
        best = math.inf
        for x in (g_xy[0], g_xy[1]):
            f = x / c - math.floor(x / c)
            best = min(best, min(f, 1.0 - f) * c)
        return best


# ---------------------------------------------------------------------------
# Cell probabilities
# ---------------------------------------------------------------------------


def _phi(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF, elementwise (math.erf)."""
    flat = np.asarray(x, dtype=float).ravel()
    return np.array([0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in flat]).reshape(np.shape(x))


_SQRT2 = math.sqrt(2.0)


def _phi_s(v: float) -> float:
    """Standard normal CDF of a scalar (same expression as _phi, element by element)."""
    return 0.5 * (1.0 + math.erf(v / _SQRT2))


def _interval_prob(lo, hi, mean, sd):
    if sd <= 0:
        return ((np.asarray(lo) <= mean) & (mean < np.asarray(hi))).astype(float)
    return _phi((np.asarray(hi) - mean) / sd) - _phi((np.asarray(lo) - mean) / sd)


@dataclass(frozen=True)
class CellProbabilities:
    cells: Tuple[Tuple[Cell, float], ...]  # top candidate in-grid cells (index_base applied), highest first
    out_of_grid: float  # probability mass outside the grid (0 if the extent is unknown)

    def as_dict(self) -> Dict[Cell, float]:
        return dict(self.cells)


def cell_probabilities(reg: GridRegistration, mean_map_xy: Sequence[float], cov_map_xy: np.ndarray,
                       max_cells: int = 4, include_registration: bool = True) -> CellProbabilities:
    """Integrate the 2D Gaussian position (map covariance rotated into the grid frame) over each cell.

    Diagonal grid-frame covariance: exact, product of per-axis normal CDF differences (math.erf).
    Correlated covariance: exact inner integral over y given x (conditional normal CDF) and a fixed 24-node
    Gauss-Legendre rule over x within each cell (restricted to mean +/- 5 sigma). The integrand is smooth, so the
    error is far below 1e-4 for any covariance (tests compare with a seeded Monte Carlo).
    Candidate cells span mean +/- 5 sigma per axis; mass beyond that is < 1.2e-6 and is dropped, so the returned
    probabilities plus out_of_grid sum to at most 1.
    include_registration (substep 4.3): add reg.position_covariance(mean) (registration error, common to all tags) in
    the grid frame. Use it for the FUSED track position only; per-observation probabilities pass False.
    """
    j = reg.jacobian()
    mu = reg.map_to_grid(mean_map_xy)
    cov = j @ np.asarray(cov_map_xy, dtype=float)[:2, :2] @ j.T
    if include_registration and (reg.origin_sigma_m or reg.origin_cov_m2 is not None or reg.yaw_sigma_rad):
        cov = cov + reg.position_covariance(mean_map_xy)
    mu0, mu1 = float(mu[0]), float(mu[1])
    sx, sy = math.sqrt(max(float(cov[0, 0]), 0.0)), math.sqrt(max(float(cov[1, 1]), 0.0))
    c = reg.cell_size_m
    k = PROB_WINDOW_SIGMA
    i_range = range(math.floor((mu0 - k * sx) / c), math.floor((mu0 + k * sx) / c) + 1)
    j_range = range(math.floor((mu1 - k * sy) / c), math.floor((mu1 + k * sy) / c) + 1)
    # Scalar math.erf throughout (substep 3.4): same formulas as before, no per-call array allocation.
    y_lo = [jj * c for jj in j_range]
    y_hi = [lo + c for lo in y_lo]
    probs: Dict[Cell, float] = {}
    cov01 = float(cov[0, 1])
    diagonal = abs(cov01) <= 1e-12 * max(sx * sy, 1e-300) or sx == 0 or sy == 0
    if diagonal:
        if sy <= 0:
            py = [1.0 if (lo <= mu1 < hi) else 0.0 for lo, hi in zip(y_lo, y_hi)]
        else:
            py = [_phi_s((hi - mu1) / sy) - _phi_s((lo - mu1) / sy) for lo, hi in zip(y_lo, y_hi)]
        for ii in i_range:
            lo_x, hi_x = ii * c, (ii + 1) * c
            if sx <= 0:
                px = 1.0 if (lo_x <= mu0 < hi_x) else 0.0
            else:
                px = _phi_s((hi_x - mu0) / sx) - _phi_s((lo_x - mu0) / sx)
            for jj, p in zip(j_range, py):
                probs[(ii, jj)] = px * p
    else:
        rho = cov01 / (sx * sy)
        s_cond = sy * math.sqrt(max(1.0 - rho * rho, 0.0))
        nodes, weights = _GL_RULE
        for ii in i_range:
            a, b = max(ii * c, mu0 - k * sx), min((ii + 1) * c, mu0 + k * sx)
            if b <= a:
                continue
            x = 0.5 * (b - a) * nodes + 0.5 * (a + b)
            wx = 0.5 * (b - a) * weights * np.exp(-0.5 * ((x - mu0) / sx) ** 2) / (sx * math.sqrt(2 * math.pi))
            m = mu1 + rho * sy / sx * (x - mu0)  # conditional mean of y given x
            if s_cond > 0:
                # Phi at each distinct cell edge, shared by the two cells that meet there (edges are j * c)
                edges = {}
                for lo, hi in zip(y_lo, y_hi):
                    for e in (lo, hi):
                        if e not in edges:
                            edges[e] = np.array([_phi_s((e - mv) / s_cond) for mv in m.tolist()])
                for jj, lo, hi in zip(j_range, y_lo, y_hi):
                    probs[(ii, jj)] = float(np.sum(wx * (edges[hi] - edges[lo])))
            else:
                for jj, lo, hi in zip(j_range, y_lo, y_hi):
                    probs[(ii, jj)] = float(np.sum(wx * ((lo <= m) & (m < hi)).astype(float)))
    in_grid = {cell: p for cell, p in probs.items() if reg.in_grid_zero_based(*cell) is not False}
    unknown_extent = reg.n_cells_x is None or reg.n_cells_y is None
    out = 0.0 if unknown_extent else max(0.0, 1.0 - sum(in_grid.values()))
    top = sorted(in_grid.items(), key=lambda kv: kv[1], reverse=True)[:max_cells]
    based = tuple(((i + reg.index_base, jj + reg.index_base), p) for (i, jj), p in top if p > 0)
    return CellProbabilities(based, out)


# ---------------------------------------------------------------------------
# Observation update and registration fit
# ---------------------------------------------------------------------------


def _deferred_probs(reg: "GridRegistration", mean: np.ndarray, cov: np.ndarray, max_cells: int) -> Dict[Cell, float]:
    return cell_probabilities(reg, mean, cov, max_cells, include_registration=False).as_dict()


def apply_grid(obs: SurvivorObservation, reg: GridRegistration, max_cells: int = 4) -> None:
    """Fill the Layer 3 fields of an observation that has p_map_m. Warnings: GRID_UNREGISTERED (BLOCK) when the
    registration is not valid (the cell is provisional, for diagnostics only); OUT_OF_GRID when outside."""
    g = reg.map_to_grid(obs.p_map_m[:2])
    obs.p_grid_m = g
    obs.cell = reg.cell_index(g)
    obs.boundary_distance_m = reg.boundary_distance_m(g)
    obs.registration_version = reg.version
    if obs.cov_map_m2 is not None:
        # Deferred (substep 3.4): computed on first access of obs.cell_probs, from copies of the inputs.
        obs.cell_probs = {}
        obs._cell_probs_thunk = partial(_deferred_probs, reg, np.array(obs.p_map_m[:2], dtype=float),
                                        np.array(obs.cov_map_m2[:2, :2], dtype=float), max_cells)
    if not reg.valid:
        obs.warnings.append(FailureReason.GRID_UNREGISTERED)
    if reg.is_out_of_grid(g):
        obs.warnings.append(FailureReason.OUT_OF_GRID)


def fit_registration_from_points(map_points: np.ndarray, grid_points: np.ndarray,
                                 template: GridRegistration) -> Tuple[GridRegistration, float]:
    """Closed-form 2D rigid least squares (rotation + translation, no scale) from >= 2 point pairs.

    The template supplies flips, cell size, extent, numbering and version; grid points are un-flipped first, then
    g = R(phi) m + t is fitted (centred cross-covariance), giving yaw = -phi and origin = -R(phi)^T t.
    Returns template.updated(origin, yaw) and the RMS residual in metres (grid frame)."""
    m = np.asarray(map_points, dtype=float)[:, :2]
    g = np.asarray(grid_points, dtype=float)[:, :2] * template._signs()
    if len(m) < 2 or len(m) != len(g):
        raise ValueError("need >= 2 matching point pairs")
    mc, gc = m - m.mean(axis=0), g - g.mean(axis=0)
    phi = math.atan2(float(np.sum(mc[:, 0] * gc[:, 1] - mc[:, 1] * gc[:, 0])),
                     float(np.sum(mc[:, 0] * gc[:, 0] + mc[:, 1] * gc[:, 1])))
    r = _rot(phi)
    t = g.mean(axis=0) - r @ m.mean(axis=0)
    origin = -r.T @ t
    reg = template.updated(origin_in_map_m=(float(origin[0]), float(origin[1])), yaw_rad=-phi)
    resid = reg.map_to_grid(m) - np.asarray(grid_points, dtype=float)[:, :2]
    return reg, float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))
