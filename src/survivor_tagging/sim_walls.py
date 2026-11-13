"""Synthetic wall points and mazes for the wall-alignment estimator tests (substep 4.3). HARNESS side (simulation only).

All numbers are SIMULATION values. Wall points are drawn on the wall CENTRE line (the estimator assumes walls on grid
lines, so thickness is not modelled here), with isotropic Gaussian noise, a fraction of uniform outlier points, and
partial coverage (a random share of the unit wall segments is never seen).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .sim import Arena, Box, Registration, Scenario, wall

Segment = Tuple[str, float, float, float]  # (axis "x"|"y", line coordinate, start, end) in the grid frame


def random_maze_walls(n_cells: int, seed: int, extra_open_fraction: float = 0.25) -> Tuple[Box, ...]:
    """Interior walls of an n x n maze with 1 m cells: a random spanning tree (depth-first, seeded) opens a perfect
    maze (1 m corridors), then `extra_open_fraction` of the remaining interior walls are removed so that rooms and
    loops appear. Walls lie on grid lines. The outer boundary is NOT included (Arena.outer_walls adds it)."""
    rng = np.random.default_rng(seed)
    n = n_cells
    opened = set()  # pairs of adjacent cells that are connected
    seen = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        c = stack[-1]
        nb = [(c[0] + dx, c[1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
              if 0 <= c[0] + dx < n and 0 <= c[1] + dy < n and (c[0] + dx, c[1] + dy) not in seen]
        if not nb:
            stack.pop()
            continue
        d = nb[int(rng.integers(len(nb)))]
        opened.add(frozenset((c, d)))
        seen.add(d)
        stack.append(d)
    walls: List[Box] = []
    for i in range(n):
        for j in range(n):
            for di, dj in ((1, 0), (0, 1)):
                a, b = (i, j), (i + di, j + dj)
                if b[0] >= n or b[1] >= n or frozenset((a, b)) in opened:
                    continue
                if rng.random() < extra_open_fraction:
                    continue
                if di == 1:  # wall between (i, j) and (i + 1, j): line x = i + 1, spans y in [j, j + 1]
                    walls.append(wall("x", float(i + 1), float(j), float(j + 1)))
                else:
                    walls.append(wall("y", float(j + 1), float(i), float(i + 1)))
    return tuple(walls)


def box_segments(boxes: Sequence[Box]) -> List[Segment]:
    """Centre-line segments of thin wall boxes (the thin direction is the line coordinate)."""
    out: List[Segment] = []
    for b in boxes:
        dx, dy = b.hi[0] - b.lo[0], b.hi[1] - b.lo[1]
        if dx < dy:
            out.append(("x", 0.5 * (b.lo[0] + b.hi[0]), b.lo[1], b.hi[1]))
        else:
            out.append(("y", 0.5 * (b.lo[1] + b.hi[1]), b.lo[0], b.hi[0]))
    return out


def sample_wall_points_grid(boxes: Sequence[Box], seed: int, spacing_m: float = 0.05, noise_sigma_m: float = 0.03,
                            outlier_fraction: float = 0.2, coverage: float = 1.0,
                            extent_m: Optional[Tuple[float, float, float, float]] = None,
                            max_segment_m: float = 1.0) -> np.ndarray:
    """(N, 2) wall points in the GRID frame. Each wall box is cut into pieces of at most `max_segment_m`; a piece is
    seen with probability `coverage`. outlier_fraction is the share of ALL returned points that are uniform outliers
    inside `extent_m` = (x0, x1, y0, y1) (default: the bounding box of the walls plus 0.5 m)."""
    rng = np.random.default_rng(seed)
    pts: List[np.ndarray] = []
    for axis, at, s0, s1 in box_segments(boxes):
        n_piece = max(1, int(np.ceil((s1 - s0) / max_segment_m - 1e-9)))
        edges = np.linspace(s0, s1, n_piece + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            if rng.random() > coverage:
                continue
            t = np.arange(a, b, spacing_m) + 0.5 * spacing_m * (b > a)
            pts.append(np.stack([np.full_like(t, at), t], axis=1) if axis == "x"
                       else np.stack([t, np.full_like(t, at)], axis=1))
    wall_pts = np.concatenate(pts) if pts else np.zeros((0, 2))
    wall_pts = wall_pts + rng.normal(0.0, noise_sigma_m, wall_pts.shape)
    if extent_m is None:
        e = np.array([[b.lo[0], b.lo[1], b.hi[0], b.hi[1]] for b in boxes]) if len(boxes) else np.zeros((1, 4))
        extent_m = (e[:, 0].min() - 0.5, e[:, 2].max() + 0.5, e[:, 1].min() - 0.5, e[:, 3].max() + 0.5)
    n_out = int(round(len(wall_pts) * outlier_fraction / max(1.0 - outlier_fraction, 1e-9)))
    outliers = np.stack([rng.uniform(extent_m[0], extent_m[1], n_out), rng.uniform(extent_m[2], extent_m[3], n_out)],
                        axis=1)
    out = np.concatenate([wall_pts, outliers])
    return out[rng.permutation(len(out))]


def grid_to_map_points(reg: Registration, pts_grid: np.ndarray) -> np.ndarray:
    """Grid-frame points -> map frame through the simulator's public map_from_grid."""
    return np.array([reg.map_from_grid(p) for p in pts_grid]).reshape(-1, 2)


def scenario_wall_points(scn: Scenario, seed: int, **kw) -> np.ndarray:
    """Wall points of the scenario's arena (walls plus outer walls) in the MAP frame (TRUE registration)."""
    boxes = scn.arena.all_walls(scn.registration.cell_size_m)
    ext = (-0.5, scn.arena.n_cells * scn.registration.cell_size_m + 0.5) * 2
    kw.setdefault("extent_m", (ext[0], ext[1], ext[0], ext[1]))
    return grid_to_map_points(scn.registration, sample_wall_points_grid(boxes, seed, **kw))


def maze_arena(n_cells: int, seed: int) -> Arena:
    return Arena(n_cells, random_maze_walls(n_cells, seed), outer_walls=True)


__all__ = ["box_segments", "grid_to_map_points", "maze_arena", "random_maze_walls", "sample_wall_points_grid",
           "scenario_wall_points"]
