"""NaivePipeline: a deliberately naive CONTROL for the harness. Do not reuse in the real layers.

Per detection: a camera-frame point from a pluggable estimator (default: depth at the bbox centre pixel, only zero
rejected, pinhole back-projection), then camera -> body -> map with the LATEST ARRIVED pose (no time alignment),
map -> grid with the given registration, floor to a cell. One tag per detector tracking_id (first valid
detection), no fusion. Tags are reported CONFIRMED at their first detection so that time-to-confirm is measurable:
the baseline commits immediately.

"NaiveTransform + Layer 1" = NaivePipeline with layer1_estimator(): the ablation that isolates Layer 1.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config
from .geometry import Layer1Params, estimate_survivor_point
from .harness import PipelineFactory, PointEstimate, PointEstimator, TagResult
from .sim import Scenario
from .types import (CameraIntrinsics, DepthMeasurement, Detection, FailureReason, PoseSample,
                    SurvivorObservation, TagStatus)


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def centre_pixel_estimator(det: Detection, depth: DepthMeasurement, intrinsics: CameraIntrinsics) -> PointEstimate:
    """Depth at the bbox centre pixel (only zero rejected), pinhole back-projection."""
    img = depth.depth_raw
    x0, y0, x1, y1 = det.bbox_px
    u = min(max(int(round((x0 + x1) / 2)), 0), img.shape[1] - 1)
    v = min(max(int(round((y0 + y1) / 2)), 0), img.shape[0] - 1)
    raw = int(img[v, u])
    if raw == 0:
        return PointEstimate(None, FailureReason.DEPTH_NO_VALID)
    z = raw * depth.depth_scale_m_per_unit
    return PointEstimate(np.array([(u - intrinsics.cx_px) * z / intrinsics.fx_px,
                                   (v - intrinsics.cy_px) * z / intrinsics.fy_px, z]))


def layer1_estimator(params: Layer1Params) -> PointEstimator:
    def est(det: Detection, depth: DepthMeasurement, intrinsics: CameraIntrinsics) -> PointEstimate:
        r = estimate_survivor_point(det, depth, intrinsics, params)
        return PointEstimate(r.p_camera_m, r.failure)
    return est


class NaivePipeline:
    def __init__(self, T_body_camera: np.ndarray, grid_offset_m: Tuple[float, float], grid_yaw_rad: float,
                 cell_size_m: float, point_estimator: PointEstimator = centre_pixel_estimator):
        self.T_body_camera = np.asarray(T_body_camera, dtype=float)
        self.grid_offset = np.asarray(grid_offset_m, dtype=float)
        c, s = math.cos(grid_yaw_rad), math.sin(grid_yaw_rad)
        self.grid_rot = np.array([[c, -s], [s, c]])
        self.cell_size = cell_size_m
        self.point_estimator = point_estimator
        self.latest_pose: Optional[PoseSample] = None
        self.tags: Dict[object, TagResult] = {}
        self.obs: List[SurvivorObservation] = []  # every detection, for observation-level scoring

    def process_pose(self, pose: PoseSample, arrival_ns: int) -> None:
        self.latest_pose = pose  # latest to ARRIVE, whatever its stamp

    def process_frame(self, detections: Sequence[Detection], depth: DepthMeasurement,
                      intrinsics: CameraIntrinsics, arrival_ns: int) -> None:
        if self.latest_pose is None:
            return
        T_map_body = np.eye(4)
        T_map_body[:3, :3] = _quat_to_matrix(self.latest_pose.orientation_wxyz)
        T_map_body[:3, 3] = self.latest_pose.position_m
        T_map_camera = T_map_body @ self.T_body_camera
        for det in detections:
            obs = SurvivorObservation(len(self.obs), det.detection_id, det.confidence, det.image_stamp_ns,
                                      det.tracking_id)
            self.obs.append(obs)
            est = self.point_estimator(det, depth, intrinsics)
            if est.p_camera_m is None:
                obs.failure = est.failure
                continue
            p_map = T_map_camera[:3, :3] @ est.p_camera_m + T_map_camera[:3, 3]
            p_grid = self.grid_rot @ p_map[:2] + self.grid_offset
            cell = (int(math.floor(p_grid[0] / self.cell_size)), int(math.floor(p_grid[1] / self.cell_size)))
            obs.p_camera_m, obs.p_map_m, obs.p_grid_m, obs.cell = est.p_camera_m, p_map, p_grid, cell
            key = det.tracking_id if det.tracking_id is not None else ("untracked", det.image_stamp_ns,
                                                                       det.detection_id)
            if key not in self.tags:
                self.tags[key] = TagResult(len(self.tags), cell, TagStatus.CONFIRMED, p_map, det.image_stamp_ns,
                                           det.image_stamp_ns)

    def finalize(self) -> List[TagResult]:
        return list(self.tags.values())

    def observations(self) -> List[SurvivorObservation]:
        return self.obs


def naive_factory(scn: Scenario, point_estimator: PointEstimator = centre_pixel_estimator) -> NaivePipeline:
    reg = scn.registration
    return NaivePipeline(scn.camera.T_body_camera, reg.offset_m, reg.yaw_rad, reg.cell_size_m, point_estimator)


def naive_layer1_factory(cfg: Config) -> PipelineFactory:
    """NaiveTransform + Layer 1 with the given config (Layer 1 parameters only are read)."""
    est = layer1_estimator(Layer1Params.from_config(cfg))
    return lambda scn: naive_factory(scn, est)
