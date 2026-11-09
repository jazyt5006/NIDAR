"""Synthetic test world: arena, survivors, virtual RGB-D camera, virtual detector, pose stream.

Every numeric value in this module is a SIMULATION value, not a hardware or competition value.
Geometry is implemented independently of the real pipeline layers: the world is projected INTO the image and depth
comes from ray casting, so sign or frame errors in the layers under test cannot cancel out.

Frames follow docs/design/conventions.md. The world geometry (walls, survivors, floor z = 0) is axis-aligned in the
official `grid` frame. The drone trajectory is authored in the `map` frame; `Registration` relates the two.
Camera optical frame: x right, y down, z forward. Body: x forward, y left, z up. One clock domain: "sim".
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .types import CameraIntrinsics, DepthMeasurement, Detection, PoseCorrection, PoseSample

CLOCK = "sim"
CAMERA_FRAME = "sim_camera"
NS_PER_S = 1_000_000_000
SIM_DEPTH_SCALE_M = 0.001  # SIMULATION value: uint16 depth in millimetres
WALL_HEIGHT_M = 2.4
WALL_THICKNESS_M = 0.05
POSTURE_DIMS_M: Dict[str, Tuple[float, float, float]] = {  # (x, y, z) extents in the grid frame
    "standing": (0.5, 0.3, 1.7),
    "sitting": (0.5, 0.5, 1.0),
    "prone": (1.7, 0.5, 0.25),
}
# Substep 3.1c: an optional per-survivor true body depth replaces the grid-x extent of these postures (prone keeps
# its fixed box: its front-surface bias comes from its length, not its depth).
DEPTH_VARIABLE_POSTURES = ("standing", "sitting")
HIT_FLOOR = -1
HIT_NONE = -2


# ---------------------------------------------------------------------------
# Rotations and transforms (simulator's own implementation)
# ---------------------------------------------------------------------------


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body-to-parent rotation, ZYX convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def matrix_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    tr = float(np.trace(r))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def make_transform(r: np.ndarray, t: Sequence[float]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = t
    return m


# Optical axes expressed in the body frame: x_c = -y_b, y_c = -z_b, z_c = +x_b (camera looking forward, level).
_R_BODY_OPTICAL_LEVEL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def camera_mount(tilt_down_rad: float, translation_m: Sequence[float] = (0.0, 0.0, 0.0),
                 yaw_rad: float = 0.0) -> np.ndarray:
    """T_body_camera for a forward camera pitched down by tilt_down_rad (positive = down) and yawed by yaw_rad."""
    return make_transform(rot_z(yaw_rad) @ rot_y(tilt_down_rad) @ _R_BODY_OPTICAL_LEVEL, translation_m)


@dataclass(frozen=True)
class Registration:
    """Simulated map -> grid relation: p_grid = Rz(yaw) p_map + offset (z unchanged)."""

    offset_m: Tuple[float, float] = (0.0, 0.0)  # grid coordinates of the map origin
    yaw_rad: float = 0.0
    cell_size_m: float = 1.0

    def _r2(self) -> np.ndarray:
        return rot_z(self.yaw_rad)[:2, :2]

    def grid_from_map(self, xy: Sequence[float]) -> np.ndarray:
        return self._r2() @ np.asarray(xy, dtype=float) + np.asarray(self.offset_m)

    def map_from_grid(self, xy: Sequence[float]) -> np.ndarray:
        return self._r2().T @ (np.asarray(xy, dtype=float) - np.asarray(self.offset_m))

    def T_grid_map(self) -> np.ndarray:
        return make_transform(rot_z(self.yaw_rad), (self.offset_m[0], self.offset_m[1], 0.0))

    def cell_of_grid(self, xy: Sequence[float]) -> Tuple[int, int]:
        return (int(math.floor(xy[0] / self.cell_size_m)), int(math.floor(xy[1] / self.cell_size_m)))


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    lo: Tuple[float, float, float]
    hi: Tuple[float, float, float]
    survivor_id: Optional[int] = None  # None = wall / obstacle


def wall(axis: str, at: float, start: float, end: float, height: float = WALL_HEIGHT_M,
         thickness: float = WALL_THICKNESS_M) -> Box:
    """Wall centred on the line axis=at (grid frame), spanning [start, end] along the other axis."""
    h = thickness / 2
    if axis == "x":
        return Box((at - h, start, 0.0), (at + h, end, height))
    if axis == "y":
        return Box((start, at - h, 0.0), (end, at + h, height))
    raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


Range2 = Tuple[Tuple[float, float], Tuple[float, float]]


@dataclass(frozen=True)
class SurvivorSpec:
    survivor_id: int
    grid_xy: Tuple[float, float]  # nominal box centre in the grid frame
    posture: str
    rotated: bool = False  # swap x/y extents
    region: Optional[Range2] = None  # allowed centre ((x_lo, x_hi), (y_lo, y_hi)); None = the nominal cell
    depth_m: Optional[float] = None  # true body depth (substep 3.1c); None = POSTURE_DIMS_M

    def centre_region(self, cell_size_m: float) -> Range2:
        if self.region is not None:
            return self.region
        i, j = (math.floor(c / cell_size_m) for c in self.grid_xy)
        return ((i * cell_size_m, (i + 1) * cell_size_m), (j * cell_size_m, (j + 1) * cell_size_m))

    def box(self) -> Box:
        dx, dy, dz = POSTURE_DIMS_M[self.posture]
        if self.depth_m is not None and self.posture in DEPTH_VARIABLE_POSTURES:
            dx = self.depth_m  # grid-x extent = front-to-back depth for the scenarios' +x viewing direction
        if self.rotated:
            dx, dy = dy, dx
        x, y = self.grid_xy
        return Box((x - dx / 2, y - dy / 2, 0.0), (x + dx / 2, y + dy / 2, dz), self.survivor_id)


@dataclass(frozen=True)
class SurvivorTruth:
    survivor_id: int
    centre_map_xy: Tuple[float, float]
    cell: Tuple[int, int]  # floor convention, 0-based, in the grid frame
    posture: str


@dataclass(frozen=True)
class Arena:
    n_cells: int = 6
    walls: Tuple[Box, ...] = ()
    outer_walls: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.n_cells <= 15:
            raise ValueError("n_cells must be in [1, 15]")

    def all_walls(self, cell_size_m: float) -> List[Box]:
        out = list(self.walls)
        if self.outer_walls:
            n = self.n_cells * cell_size_m
            out += [wall("x", 0.0, 0.0, n), wall("x", n, 0.0, n), wall("y", 0.0, 0.0, n), wall("y", n, 0.0, n)]
        return out


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DepthNoise:
    sigma_a_per_m: float = 0.0  # gaussian sigma = a * z^2
    dropout: float = 0.0  # fraction of pixels set to 0
    invalid_left_band_px: int = 0  # zero-valued band on the left edge
    flying_prob: float = 0.0  # probability an edge pixel gets a mixed depth
    flying_edge_m: float = 0.3  # depth jump that defines an edge


@dataclass(frozen=True, eq=False)
class SimCamera:
    width: int = 424
    height: int = 240
    fx: float = 212.0
    fy: float = 212.0
    cx: float = 211.5
    cy: float = 119.5
    fps: float = 5.0
    T_body_camera: np.ndarray = field(default_factory=lambda: camera_mount(math.radians(20), (0.10, 0.0, 0.0)))
    min_range_m: float = 0.1
    max_range_m: float = 10.0
    noise: DepthNoise = DepthNoise()

    def intrinsics(self) -> CameraIntrinsics:
        return CameraIntrinsics(self.fx, self.fy, self.cx, self.cy, self.width, self.height, "none", ())

    def pixel_rays(self) -> np.ndarray:
        """(H*W, 3) optical-frame directions with z = 1, so the ray parameter equals depth Z."""
        u, v = np.meshgrid(np.arange(self.width, dtype=float), np.arange(self.height, dtype=float))
        x = (u.ravel() - self.cx) / self.fx
        y = (v.ravel() - self.cy) / self.fy
        return np.stack([x, y, np.ones_like(x)], axis=1)


@dataclass(frozen=True)
class DetectorSim:
    bbox_jitter_px: float = 0.0
    miss_prob: float = 0.0
    fp_per_frame: float = 0.0  # Poisson mean of false-positive boxes per frame
    id_switch_prob: float = 0.0  # per survivor per frame
    conf_range: Tuple[float, float] = (0.6, 0.95)
    fp_conf_range: Tuple[float, float] = (0.3, 0.6)
    latency_s: float = 0.0  # frame + detections arrive this long after capture
    bbox_pad_frac: float = 0.0  # loose boxes: widen each side by this fraction of the box width (then clip)
    id_merge_dist_m: float = 0.0  # >0: two detected survivors closer than this share the first one's tracking id
    phantom_box: Optional["Box"] = None  # a non-survivor object (also present as an arena wall/obstacle box)
    phantom_prob: float = 0.0  # per-frame probability the detector fires on the phantom (fresh id each time)
    # Substep 3.3a: with this probability per detected survivor and frame the detector fires a SECOND box on the same
    # person (fresh tracking id): the first box extended sideways by a world offset at the person's depth drawn
    # uniformly from dup_offset_m (random side), so it still covers the whole body. A box that only partly covers a
    # flat-faced body would land on the wall behind it instead (a wrong surface, not a duplicate). The position
    # ESTIMATES of the two boxes then differ only by depth noise and sampling (about 0.01-0.05 m, measured in the
    # 3.3a tests), not by dup_offset_m. Random draws are made only if dup_prob > 0.
    dup_prob: float = 0.0
    dup_offset_m: Tuple[float, float] = (0.1, 0.4)


@dataclass(frozen=True)
class Closure:
    """SIM loop closure (substep 3.2). From at_s on, live poses carry revision k + 1 (k = index in time order) and
    the drift error restarts from zero; latency_s later a correction event arrives:
    - "history": corrected poses (true pose + a residual error, constant per closure, N(0, residual_sigma_m) per
      horizontal axis) for every live pose stamp in range_s ([lo, hi), default [0, at_s)).
    - "frame_correction": one translation that removes the live error of the last pose before at_s, plus the same
      residual; it is exact only for observations whose pose error equals that one.
    emit=False: the event is never delivered (missing correction). All values are SIMULATION values."""

    at_s: float
    mode: str = "history"
    range_s: Optional[Tuple[float, float]] = None
    residual_sigma_m: float = 0.02  # SIM: much smaller than the drift it removes
    latency_s: float = 0.2  # SIM
    emit: bool = True


@dataclass(frozen=True)
class Trajectory:
    """Body trajectory in the map frame."""

    waypoints_map: Tuple[Tuple[float, float, float], ...]  # visited at constant speed over the scenario
    yaw0_rad: float = 0.0
    yaw_rate_rad_s: float = 0.0
    yaw_osc_amp_rad: float = 0.0
    yaw_osc_hz: float = 0.0
    wobble_amp_rad: float = 0.0  # roll and pitch wobble amplitude
    wobble_hz: float = 1.0
    rate_hz: float = 50.0
    pose_latency_s: float = 0.0
    drift_m_per_s: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    loop_closure_s: Optional[float] = None  # poses stamped before this carry the jump error; revision +1 after
    loop_closure_jump_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    look_at_map_xy: Optional[Tuple[float, float]] = None  # yaw points at this map point (overrides yaw terms)
    yaw_schedule: Tuple[Tuple[float, float], ...] = ()  # (t_s, yaw_rad) keyframes, linear; overrides yaw terms
    # Honest pose uncertainty (0 = off): the reported horizontal position error is a random walk with variance
    # pose_drift_sigma_m_per_sqrt_m^2 per metre travelled; the reported covariance is exactly that variance. At
    # loop_closure_s both error and variance are scaled (error by sqrt(factor)), so the error stays distributed as
    # the stated covariance.
    pose_drift_sigma_m_per_sqrt_m: float = 0.0
    loop_closure_variance_factor: float = 1.0
    # Substep 3.2: loop closures with correction events. Non-empty: the revision counts the closures passed and
    # drift_m_per_s restarts at each closure (the legacy jump / random-walk terms are unchanged). Empty: unchanged.
    closures: Tuple[Closure, ...] = ()
    # Substep 3.3a (closures only): after closure k the live horizontal error is a drift with a RANDOM rate a_k ~
    # N(0, sigma_a^2) per axis, sigma_a = |drift_m_per_s| / sqrt(2) (so the rms drift speed equals |drift_m_per_s|),
    # from zero at the closure, and the reported covariance is exactly its variance (sigma_a * dt)^2 per axis, so
    # the error is distributed as the stated covariance. Before the first closure the scripted drift_m_per_s (no
    # covariance) is unchanged.

    def revision_at(self, t_s: float) -> int:
        if self.closures:
            return sum(1 for c in self.closures if t_s >= c.at_s)
        return int(self.loop_closure_s is not None and t_s >= self.loop_closure_s)

    def drift_start_s(self, t_s: float) -> float:
        """Time since which drift_m_per_s accumulates (0, or the last closure when closures are set)."""
        return max([0.0] + [c.at_s for c in self.closures if t_s >= c.at_s])

    def true_pose(self, t_s: float, duration_s: float) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        pts = np.asarray(self.waypoints_map, dtype=float)
        if len(pts) == 1 or duration_s <= 0:
            pos = pts[0].copy()
        else:
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            s = min(max(t_s / duration_s, 0.0), 1.0) * cum[-1]
            pos = np.array([np.interp(s, cum, pts[:, k]) for k in range(3)])
        w = 2 * math.pi * self.wobble_hz * t_s
        roll = self.wobble_amp_rad * math.sin(w)
        pitch = self.wobble_amp_rad * math.sin(1.3 * w + 0.7)
        yaw = (self.yaw0_rad + self.yaw_rate_rad_s * t_s
               + self.yaw_osc_amp_rad * math.sin(2 * math.pi * self.yaw_osc_hz * t_s))
        if self.look_at_map_xy is not None:
            yaw = math.atan2(self.look_at_map_xy[1] - pos[1], self.look_at_map_xy[0] - pos[0])
        elif self.yaw_schedule:
            ts, ys = zip(*self.yaw_schedule)
            yaw = float(np.interp(t_s, ts, ys))
        return pos, (roll, pitch, yaw)


@dataclass(frozen=True, eq=False)
class Scenario:
    name: str
    tests: str  # what this scenario exercises
    arena: Arena
    survivors: Tuple[SurvivorSpec, ...]
    camera: SimCamera
    detector: DetectorSim
    trajectory: Trajectory
    registration: Registration
    duration_s: float = 3.0
    first_frame_s: float = 0.2
    placement: str = "random_in_cell"  # or "nominal" (use grid_xy as given)
    # Substep 3.1c: None = fixed POSTURE_DIMS_M depth (unchanged default); (lo, hi) = per-survivor true depth drawn
    # uniformly per seed from its own random stream. The pipeline never sees it.
    body_depth_range_m: Optional[Tuple[float, float]] = None
    # Substep 3.2: the registration the pipeline is configured with at start (None = the true one) and later
    # registration updates (arrival time s, registration). Truth (scoring) always uses `registration`.
    initial_registration: Optional[Registration] = None
    registration_updates: Tuple[Tuple[float, Registration], ...] = ()

    def truths(self) -> List[SurvivorTruth]:
        return [SurvivorTruth(s.survivor_id, tuple(self.registration.map_from_grid(s.grid_xy)),
                              self.registration.cell_of_grid(s.grid_xy), s.posture) for s in self.survivors]

    def boxes(self) -> List[Box]:
        return self.arena.all_walls(self.registration.cell_size_m) + [s.box() for s in self.survivors]


# ---------------------------------------------------------------------------
# Ray casting and projection
# ---------------------------------------------------------------------------


def cast_rays(origin: np.ndarray, dirs: np.ndarray, boxes: Sequence[Box]) -> Tuple[np.ndarray, np.ndarray]:
    """First hit of rays origin + t*dirs (t > 0) against the floor z = 0 and axis-aligned boxes.

    Returns (t, hit) with t = inf and hit = HIT_NONE where nothing is hit, hit = HIT_FLOOR for the floor,
    otherwise the box index. A ray starting inside a box ignores that box.
    """
    d = np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
    inv = [np.ascontiguousarray(1.0 / d[:, a]) for a in range(3)]  # per-axis 1D arrays are much faster
    n = dirs.shape[0]
    t_best = np.full(n, np.inf)
    hit = np.full(n, HIT_NONE, dtype=np.int64)
    t_floor = -origin[2] * inv[2]
    floor_ok = (dirs[:, 2] < 0) & (t_floor > 0)
    t_best[floor_ok] = t_floor[floor_ok]
    hit[floor_ok] = HIT_FLOOR
    for k, b in enumerate(boxes):
        t_near = np.full(n, -np.inf)
        t_far = np.full(n, np.inf)
        for a in range(3):
            t1 = (b.lo[a] - origin[a]) * inv[a]
            t2 = (b.hi[a] - origin[a]) * inv[a]
            np.maximum(t_near, np.minimum(t1, t2), out=t_near)
            np.minimum(t_far, np.maximum(t1, t2), out=t_far)
        ok = (t_near <= t_far) & (t_near > 0) & (t_near < t_best)
        t_best[ok] = t_near[ok]
        hit[ok] = k
    return t_best, hit


def camera_pose_in_grid(scn: Scenario, t_s: float) -> np.ndarray:
    """True T_grid_camera at time t_s."""
    pos, (roll, pitch, yaw) = scn.trajectory.true_pose(t_s, scn.duration_s)
    t_map_body = make_transform(rpy_to_matrix(roll, pitch, yaw), pos)
    return scn.registration.T_grid_map() @ t_map_body @ scn.camera.T_body_camera


def project_points(T_grid_camera: np.ndarray, cam: SimCamera, pts_grid: np.ndarray) -> np.ndarray:
    """(N, 3) grid points -> (N, 3) array of (u, v, Z). Z <= 0 means behind the camera."""
    r, t = T_grid_camera[:3, :3], T_grid_camera[:3, 3]
    pc = (np.asarray(pts_grid, dtype=float) - t) @ r  # = R^T (p - t) for each row
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam.fx * pc[:, 0] / z + cam.cx
        v = cam.fy * pc[:, 1] / z + cam.cy
    return np.stack([u, v, z], axis=1)


def render_depth_m(T_grid_camera: np.ndarray, cam: SimCamera, boxes: Sequence[Box],
                   rays: Optional[np.ndarray] = None) -> np.ndarray:
    """Noise-free depth image in metres (0 = no return)."""
    rays = cam.pixel_rays() if rays is None else rays
    dirs = rays @ T_grid_camera[:3, :3].T
    t, hit = cast_rays(T_grid_camera[:3, 3], dirs, boxes)
    z = np.where((hit != HIT_NONE) & (t >= cam.min_range_m) & (t <= cam.max_range_m), t, 0.0)
    return z.reshape(cam.height, cam.width)


def apply_depth_noise(z: np.ndarray, noise: DepthNoise, rng: np.random.Generator) -> np.ndarray:
    z = z.copy()
    valid = z > 0
    if noise.flying_prob > 0:
        right = np.zeros_like(z)
        right[:, :-1] = z[:, 1:]
        edge = valid & (right > 0) & (np.abs(right - z) > noise.flying_edge_m)
        fly = edge & (rng.random(z.shape) < noise.flying_prob)
        mix = rng.random(z.shape)
        z[fly] = z[fly] + mix[fly] * (right[fly] - z[fly])
    if noise.sigma_a_per_m > 0:
        z[valid] += rng.normal(0.0, 1.0, size=int(valid.sum())) * noise.sigma_a_per_m * z[valid] ** 2
    if noise.dropout > 0:
        z[rng.random(z.shape) < noise.dropout] = 0.0
    if noise.invalid_left_band_px > 0:
        z[:, :noise.invalid_left_band_px] = 0.0
    return np.clip(z, 0.0, None)


def to_uint16_mm(z_m: np.ndarray) -> np.ndarray:
    return np.clip(np.round(z_m / SIM_DEPTH_SCALE_M), 0, 65535).astype(np.uint16)


def survivor_visible(T_grid_camera: np.ndarray, cam: SimCamera, boxes: Sequence[Box], k: int) -> bool:
    """Visible if the box centre is in front, inside the image, and the first hit along its ray is box k."""
    centre = _box_centre(boxes[k])
    u, v, z = project_points(T_grid_camera, cam, centre[None, :])[0]
    if not (z > cam.min_range_m and 0 <= u <= cam.width - 1 and 0 <= v <= cam.height - 1):
        return False
    origin = T_grid_camera[:3, 3]
    _, hit = cast_rays(origin, (centre - origin)[None, :], boxes)
    return int(hit[0]) == k


def projected_bbox(T_grid_camera: np.ndarray, cam: SimCamera, box: Box) -> Optional[Tuple[float, float, float, float]]:
    """Bounding box of the projected box corners in front of the camera, clipped to the image."""
    xs = (box.lo[0], box.hi[0])
    ys = (box.lo[1], box.hi[1])
    zs = (box.lo[2], box.hi[2])
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    uvz = project_points(T_grid_camera, cam, corners)
    front = uvz[:, 2] > cam.min_range_m
    if not front.any():
        return None
    u, v = uvz[front, 0], uvz[front, 1]
    x0, x1 = np.clip([u.min(), u.max()], 0, cam.width - 1)
    y0, y1 = np.clip([v.min(), v.max()], 0, cam.height - 1)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return float(x0), float(y0), float(x1), float(y1)


# ---------------------------------------------------------------------------
# Event stream
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Frame:
    stamp_ns: int
    detections: Tuple[Detection, ...]
    depth: DepthMeasurement
    intrinsics: CameraIntrinsics
    # Ground truth, for scoring only; a pipeline must never read these.
    truth_visible: Tuple[int, ...]  # survivor ids visible at capture
    det_survivor_ids: Tuple[Optional[int], ...] = ()  # per detection: survivor id, None = false positive
    truth_camera_m: Tuple[Tuple[int, Tuple[float, float, float]], ...] = ()  # (id, true box centre in camera)
    T_grid_camera: Optional[np.ndarray] = None  # true camera pose at capture


@dataclass(frozen=True)
class RegistrationUpdate:
    """SIM registration change (substep 3.2): version 1, 2, ... (the initial registration is version 0)."""

    stamp_ns: int
    version: int
    registration: Registration


@dataclass(frozen=True)
class Event:
    arrival_ns: int
    kind: str  # "pose", "frame", "correction" (PoseCorrection) or "registration" (RegistrationUpdate)
    payload: Union[PoseSample, Frame, PoseCorrection, RegistrationUpdate]


def _ns(t_s: float) -> int:
    return int(round(t_s * NS_PER_S))


def _pose_events(scn: Scenario, rng_pose: Optional[np.random.Generator] = None,
                 rng_drift: Optional[np.random.Generator] = None) -> List[Event]:
    tr = scn.trajectory
    n = int(math.floor(scn.duration_s * tr.rate_hz)) + 1
    out = []
    err, var, prev, closed = np.zeros(2), 0.0, None, False
    k2 = tr.pose_drift_sigma_m_per_sqrt_m ** 2
    closures = sorted(tr.closures, key=lambda c: c.at_s)
    sigma_a = float(np.linalg.norm(tr.drift_m_per_s)) / math.sqrt(2.0)
    rates = np.zeros((len(closures), 2))
    if closures:
        if rng_drift is None:
            raise ValueError("a trajectory with closures needs rng_drift")
        rates = rng_drift.normal(0.0, 1.0, size=(len(closures), 2)) * sigma_a  # always drawn: stream independent of sigma
    for i in range(n):
        t = i / tr.rate_hz
        pos, (roll, pitch, yaw) = tr.true_pose(t, scn.duration_s)
        revision = tr.revision_at(t)
        closure_var = 0.0
        if closures and revision > 0:  # after a closure: random-rate drift, honest covariance (substep 3.3a)
            dt = t - closures[revision - 1].at_s
            rep = pos + np.array([rates[revision - 1, 0] * dt, rates[revision - 1, 1] * dt, 0.0])
            closure_var = (sigma_a * dt) ** 2
        else:
            rep = pos + np.asarray(tr.drift_m_per_s) * (t - tr.drift_start_s(t))
        if tr.loop_closure_s is not None and t < tr.loop_closure_s:
            rep = rep + np.asarray(tr.loop_closure_jump_m)
        cov = None
        if k2 > 0:
            step = 0.0 if prev is None else float(np.linalg.norm(pos[:2] - prev[:2]))
            prev = pos
            err = err + rng_pose.normal(0.0, math.sqrt(k2 * step), size=2)
            var += k2 * step
            if tr.loop_closure_s is not None and t >= tr.loop_closure_s and not closed:
                closed = True
                err = err * math.sqrt(tr.loop_closure_variance_factor)
                var *= tr.loop_closure_variance_factor
            rep = rep + np.array([err[0], err[1], 0.0])
            cov = np.zeros((6, 6))
            cov[0, 0] = cov[1, 1] = var + closure_var
        elif closures and revision > 0 and sigma_a > 0:
            cov = np.zeros((6, 6))
            cov[0, 0] = cov[1, 1] = closure_var
        pose = PoseSample(_ns(t), CLOCK, "map", "body", rep, matrix_to_quat_wxyz(rpy_to_matrix(roll, pitch, yaw)),
                          True, revision, "sim", cov)
        out.append(Event(_ns(t + tr.pose_latency_s), "pose", pose))
    return out


def _frame_events(scn: Scenario, rng_depth: np.random.Generator, rng_det: np.random.Generator) -> List[Event]:
    cam, det = scn.camera, scn.detector
    boxes = scn.boxes()
    surv_idx = {b.survivor_id: k for k, b in enumerate(boxes) if b.survivor_id is not None}
    rays = cam.pixel_rays()
    intr = cam.intrinsics()
    track_ids = {sid: 100 + sid for sid in surv_idx}
    next_id = 1000
    phantom_k = boxes.index(det.phantom_box) if det.phantom_box is not None and det.phantom_box in boxes else None
    out = []
    n = int(math.floor((scn.duration_s - scn.first_frame_s) * cam.fps)) + 1
    for i in range(n):
        t = scn.first_frame_s + i / cam.fps
        stamp = _ns(t)
        T = camera_pose_in_grid(scn, t)
        z = apply_depth_noise(render_depth_m(T, cam, boxes, rays), cam.noise, rng_depth)
        depth = DepthMeasurement(stamp, CLOCK, CAMERA_FRAME, to_uint16_mm(z), SIM_DEPTH_SCALE_M, intr, True,
                                 (cam.min_range_m, cam.max_range_m))
        dets: List[Detection] = []
        det_sids: List[Optional[int]] = []
        visible = []
        for sid, k in surv_idx.items():
            if rng_det.random() < det.id_switch_prob:
                track_ids[sid] = next_id
                next_id += 1
            if not survivor_visible(T, cam, boxes, k):
                continue
            visible.append(sid)
            bb = projected_bbox(T, cam, boxes[k])
            if bb is None or rng_det.random() < det.miss_prob:
                continue
            if det.bbox_pad_frac > 0:
                pad = det.bbox_pad_frac * (bb[2] - bb[0])
                bb = (max(bb[0] - pad, 0.0), bb[1], min(bb[2] + pad, cam.width - 1.0), bb[3])
            bb = _jitter(bb, det.bbox_jitter_px, cam, rng_det)
            if bb is None:
                continue
            conf = float(rng_det.uniform(*det.conf_range))
            dets.append(Detection(len(dets), conf, bb, stamp, CLOCK, CAMERA_FRAME, (cam.width, cam.height),
                                  track_ids[sid], "person"))
            det_sids.append(sid)
            if det.dup_prob > 0 and rng_det.random() < det.dup_prob:
                depth_z = float(project_points(T, cam, _box_centre(boxes[k])[None, :])[0, 2])
                shift = float(rng_det.uniform(*det.dup_offset_m)) * (1.0 if rng_det.random() < 0.5 else -1.0)
                shift_px = shift * cam.fx / max(depth_z, 1e-6)
                bb2 = (max(min(bb[0], bb[0] + shift_px), 0.0), bb[1],
                       min(max(bb[2], bb[2] + shift_px), cam.width - 1.0), bb[3])
                if bb2[2] - bb2[0] >= 1:
                    dets.append(Detection(len(dets), float(rng_det.uniform(*det.conf_range)), bb2, stamp, CLOCK,
                                          CAMERA_FRAME, (cam.width, cam.height), next_id, "person"))
                    det_sids.append(sid)
                    next_id += 1
        if det.id_merge_dist_m > 0:  # detector identity merge: nearby people share the first one's id
            for a in range(len(dets)):
                for b in range(a + 1, len(dets)):
                    ca, cb = _box_centre(boxes[surv_idx[det_sids[a]]]), _box_centre(boxes[surv_idx[det_sids[b]]])
                    if np.linalg.norm((ca - cb)[:2]) < det.id_merge_dist_m:
                        dets[b] = replace(dets[b], tracking_id=dets[a].tracking_id)
        if det.phantom_box is not None and phantom_k is not None and survivor_visible(T, cam, boxes, phantom_k):
            if rng_det.random() < det.phantom_prob:
                bb = projected_bbox(T, cam, boxes[phantom_k])
                if bb is not None:
                    dets.append(Detection(len(dets), float(rng_det.uniform(*det.fp_conf_range)), bb, stamp, CLOCK,
                                          CAMERA_FRAME, (cam.width, cam.height), next_id, "person"))
                    det_sids.append(None)
                    next_id += 1
        for _ in range(int(rng_det.poisson(det.fp_per_frame)) if det.fp_per_frame > 0 else 0):
            w, h = rng_det.uniform(15, 60), rng_det.uniform(30, 120)
            x0, y0 = rng_det.uniform(0, cam.width - 1 - w), rng_det.uniform(0, cam.height - 1 - h)
            dets.append(Detection(len(dets), float(rng_det.uniform(*det.fp_conf_range)), (x0, y0, x0 + w, y0 + h),
                                  stamp, CLOCK, CAMERA_FRAME, (cam.width, cam.height), next_id, "person"))
            det_sids.append(None)
            next_id += 1
        r, tc = T[:3, :3], T[:3, 3]
        truth_cam = tuple((sid, tuple(float(c) for c in r.T @ (_box_centre(boxes[k]) - tc)))
                          for sid, k in surv_idx.items())
        frame = Frame(stamp, tuple(dets), depth, intr, tuple(visible), tuple(det_sids), truth_cam, T)
        out.append(Event(_ns(t + det.latency_s), "frame", frame))
    return out


def _correction_events(scn: Scenario, pose_events: Sequence[Event], rng: np.random.Generator) -> List[Event]:
    """One correction event per emitted Closure (see Closure). The residual is drawn for every closure, emitted or
    not, so the stream does not depend on emit flags."""
    poses = [e.payload for e in pose_events]
    out = []
    for k, c in enumerate(sorted(scn.trajectory.closures, key=lambda c: c.at_s)):
        rev = k + 1
        res = rng.normal(0.0, c.residual_sigma_m, size=2)
        if not c.emit:
            continue
        stamp = _ns(c.at_s)
        if c.mode == "history":
            lo, hi = c.range_s if c.range_s is not None else (0.0, c.at_s)
            corrected = []
            for p in poses:
                if not (_ns(lo) <= p.stamp_ns < _ns(hi)):
                    continue
                true_pos, _ = scn.trajectory.true_pose(p.stamp_ns / NS_PER_S, scn.duration_s)
                # Substep 3.3a: the corrected pose's error is exactly `res`, drawn from N(0, residual_sigma_m^2) per
                # axis, so that is its covariance (it was left out when the live pose carried none).
                cov = np.zeros((6, 6))
                cov[0, 0] = cov[1, 1] = c.residual_sigma_m ** 2
                corrected.append(PoseSample(p.stamp_ns, CLOCK, "map", "body",
                                            true_pos + np.array([res[0], res[1], 0.0]), p.orientation_wxyz, True, rev,
                                            "sim_corrected", cov))
            payload = PoseCorrection(rev, "history", stamp, CLOCK, tuple(corrected))
        elif c.mode == "frame_correction":
            last = [p for p in poses if p.stamp_ns < stamp][-1]
            true_pos, _ = scn.trajectory.true_pose(last.stamp_ns / NS_PER_S, scn.duration_s)
            err = last.position_m - true_pos
            payload = PoseCorrection(rev, "frame_correction", stamp, CLOCK, (),
                                     make_transform(np.eye(3), -err + np.array([res[0], res[1], 0.0])))
        else:
            raise ValueError(f"unknown closure mode {c.mode!r}")
        out.append(Event(_ns(c.at_s + c.latency_s), "correction", payload))
    return out


def _registration_events(scn: Scenario) -> List[Event]:
    return [Event(_ns(t), "registration", RegistrationUpdate(_ns(t), k + 1, reg))
            for k, (t, reg) in enumerate(sorted(scn.registration_updates, key=lambda x: x[0]))]


def _jitter(bb, sigma: float, cam: SimCamera, rng: np.random.Generator):
    if sigma <= 0:
        return bb
    x0, y0, x1, y1 = np.asarray(bb) + rng.normal(0.0, sigma, size=4)
    x0, x1 = np.clip(sorted((x0, x1)), 0, cam.width - 1)
    y0, y1 = np.clip(sorted((y0, y1)), 0, cam.height - 1)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return float(x0), float(y0), float(x1), float(y1)


def _rngs(scn: Scenario, seed: int) -> List[np.random.Generator]:
    """Streams: [depth noise, detector, placement, pose error, body depth, correction residual, closure drift]. Depend on (seed,
    scenario name) so scenarios do not share identical random draws. Adding a stream never changes the earlier ones
    (SeedSequence children)."""
    entropy = [seed, zlib.crc32(scn.name.encode("utf-8"))]
    return [np.random.default_rng(s) for s in np.random.SeedSequence(entropy).spawn(7)]


def _box_centre(b: Box) -> np.ndarray:
    return (np.asarray(b.lo) + np.asarray(b.hi)) / 2


def place_survivors(scn: Scenario, seed: int) -> Scenario:
    """Resolve the placement. "random_in_cell": one common shift, uniform over the shifts that keep every
    survivor's centre inside its allowed region (its cell unless the scenario narrows it), so the spacing between
    survivors and the scenario's wall/occlusion layout stay valid. Returns a scenario with placement "nominal".
    If body_depth_range_m is set, each survivor's true depth is drawn (stream 4, one draw per survivor in order) and
    the range is cleared, so placing twice is idempotent."""
    if scn.body_depth_range_m is not None:
        lo_d, hi_d = scn.body_depth_range_m
        rng_d = _rngs(scn, seed)[4]
        scn = replace(scn, body_depth_range_m=None,
                      survivors=tuple(replace(s, depth_m=float(rng_d.uniform(lo_d, hi_d))) for s in scn.survivors))
    if scn.placement == "nominal":
        return scn
    if scn.placement != "random_in_cell":
        raise ValueError(f"unknown placement {scn.placement!r}")
    lo, hi = np.full(2, -np.inf), np.full(2, np.inf)
    for s in scn.survivors:
        reg = s.centre_region(scn.registration.cell_size_m)
        for a in range(2):
            lo[a] = max(lo[a], reg[a][0] - s.grid_xy[a])
            hi[a] = min(hi[a], reg[a][1] - s.grid_xy[a])
    if np.any(lo > hi):
        raise ValueError(f"{scn.name}: survivor regions admit no common placement")
    rng = _rngs(scn, seed)[2]
    shift = [lo[a] if lo[a] == hi[a] else rng.uniform(lo[a], hi[a]) for a in range(2)]
    survivors = tuple(replace(s, grid_xy=(s.grid_xy[0] + shift[0], s.grid_xy[1] + shift[1]))
                      for s in scn.survivors)
    return replace(scn, survivors=survivors, placement="nominal")


def build_events(scn: Scenario, seed: int) -> List[Event]:
    """Time-ordered event stream for place_survivors(scn, seed). Ties at equal arrival: poses first, then generation
    order (frames, corrections, registration updates)."""
    scn = place_survivors(scn, seed)
    rng_depth, rng_det, _, rng_pose, _, rng_corr, rng_drift = _rngs(scn, seed)
    poses = _pose_events(scn, rng_pose, rng_drift)
    events = poses + _frame_events(scn, rng_depth, rng_det) + _correction_events(scn, poses, rng_corr) + \
        _registration_events(scn)
    order = {id(e): i for i, e in enumerate(events)}
    return sorted(events, key=lambda e: (e.arrival_ns, 0 if e.kind == "pose" else 1, order[id(e)]))
