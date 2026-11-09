"""Layer123Pipeline: capture-time pose -> floor plane -> Layer 1 -> camera-to-map (Layer 2) -> footprint refinement
-> Layer 3, configured only from a Config. Detections wait in the bounded pending queue until a pose newer than their
capture time has arrived (or fail explicitly); the latest pose is never used as a substitute.

Implements the harness Pipeline protocol. Tag policy is deliberately the SAME as the naive baseline (one tag per
detector tracking_id, from its first fully processed observation, no fusion) so that comparisons isolate the layers;
Layer 4 replaces it later. A tag is PROVISIONAL instead of CONFIRMED if a BLOCK-severity warning is present
(e.g. GRID_UNREGISTERED). No simulator imports.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

from .config import Config
from .footprint import FootprintParams, refine_observation
from .geometry import Layer1Params, estimate_survivor_point
from .grid import GridRegistration, apply_grid
from .tracking import Layer4, Layer4Params
from .transforms import Layer2, Layer2Params, PendingItem, extrinsic_from_config, floor_plane_in_camera, pose_matrix
from .types import (FAILURE_SEVERITY, CameraIntrinsics, DepthMeasurement, Detection, FailureReason, PoseSample,
                    Severity, SurvivorObservation, Tag, TagResult, TagStatus)


class Layer123Pipeline:
    def __init__(self, cfg: Config):
        self.l1 = Layer1Params.from_config(cfg)
        extr, extr_fail = extrinsic_from_config(cfg)
        self.l2 = Layer2(Layer2Params.from_config(cfg), extr, extr_fail)
        self.grid = GridRegistration.from_config(cfg)
        self.max_cells = cfg.value("grid.max_candidate_cells")
        self.floor_removal = bool(cfg.value("l1.floor_removal"))
        self.floor_z_m = cfg.value("map.floor_z_m")
        self.floor_margin_m = cfg.value("l1.floor_margin_m")
        self.floor_sigma_k = cfg.value("l1.floor_margin_pose_sigma_k")
        self.footprint = FootprintParams.from_config(cfg) if cfg.value("l1b.footprint_refinement") else None
        self.l1_ms: List[float] = []
        self.footprint_ms: List[float] = []
        self.obs: List[SurvivorObservation] = []
        self.tags: Dict[object, object] = {}
        self.now_ns: Optional[int] = None
        self.l2_ms: List[float] = []
        self.l3_ms: List[float] = []

    # --- Pipeline protocol --------------------------------------------------------------------------------
    def process_pose(self, pose: PoseSample, arrival_ns: int) -> None:
        self.now_ns = arrival_ns
        self.l2.add_pose(pose)
        for item in self.l2.queue.pop_ready(self.l2.buffer.newest_ns):
            self._resolve(item)
        self._expire()

    def process_frame(self, detections: Sequence[Detection], depth: DepthMeasurement,
                      intrinsics: CameraIntrinsics, arrival_ns: int) -> None:
        self.now_ns = arrival_ns
        self._expire()
        for det in detections:
            obs = SurvivorObservation(len(self.obs), det.detection_id, det.confidence, det.image_stamp_ns,
                                      det.tracking_id)
            self.obs.append(obs)
            capture, fail = self.l2.check(det, depth, arrival_ns)
            if fail is not None:
                obs.failure = fail
                continue
            item = PendingItem(capture, arrival_ns + int(self.l2.params.max_pose_wait_s * 1e9),
                               (obs, det, depth, intrinsics))
            if self.l2.waiting_for_pose(capture):
                evicted = self.l2.queue.push(item)
                if evicted is not None:
                    evicted.payload[0].failure = FailureReason.CAPACITY_EXCEEDED
            else:
                self._resolve(item)

    def finalize(self) -> List[object]:
        for item in self.l2.queue.drain():
            item.payload[0].failure = FailureReason.POSE_UNAVAILABLE_AT_CAPTURE
        return list(self.tags.values())

    def observations(self) -> List[SurvivorObservation]:
        return self.obs

    # --- internals ----------------------------------------------------------------------------------------
    def _expire(self) -> None:
        for item in self.l2.queue.pop_expired(self.now_ns):
            item.payload[0].failure = FailureReason.POSE_UNAVAILABLE_AT_CAPTURE

    def _resolve(self, item: PendingItem) -> None:
        obs, det, depth, intrinsics = item.payload
        t0 = time.perf_counter()
        lk = self.l2.lookup(item.capture_pose_ns)
        if lk.failure is not None:
            obs.failure = lk.failure
            self.l2_ms.append((time.perf_counter() - t0) * 1e3)
            return
        floor = None
        if self.floor_removal:
            if self.floor_z_m is None:
                obs.warnings.append(FailureReason.FLOOR_UNKNOWN)
            else:
                floor = floor_plane_in_camera(lk.pose, self.l2.extrinsic, self.floor_z_m, self.floor_margin_m,
                                              self.floor_sigma_k)
        ta = time.perf_counter()
        r = estimate_survivor_point(det, depth, intrinsics, self.l1, floor=floor)
        tb = time.perf_counter()
        self.l1_ms.append((tb - ta) * 1e3)
        obs.p_camera_m, obs.cov_camera_m2, obs.ground_ref_camera_m = r.p_camera_m, r.cov_camera_m2, \
            r.ground_ref_camera_m
        obs.n_valid_px, obs.valid_fraction, obs.depth_quality = r.n_valid_px, r.valid_fraction, r.depth_quality
        obs.floor_removed_px, obs.cluster_points_camera_m = r.floor_removed_px, r.cluster_points_camera_m
        obs.warnings.extend(r.warnings)
        if r.failure is not None:
            obs.failure = r.failure
            return
        self.l2.apply(obs, lk)
        t1 = time.perf_counter()
        self.l2_ms.append((ta - t0 + t1 - tb) * 1e3)
        if self.footprint is not None:
            refine_observation(obs, pose_matrix(lk.pose) @ self.l2.extrinsic.T_body_camera, self.footprint)
            self.footprint_ms.append((time.perf_counter() - t1) * 1e3)
            t1 = time.perf_counter()
        apply_grid(obs, self.grid, self.max_cells)
        self.l3_ms.append((time.perf_counter() - t1) * 1e3)
        self._tag(obs)

    def _tag(self, obs: SurvivorObservation) -> None:
        key = obs.tracking_id if obs.tracking_id is not None else ("untracked", obs.capture_stamp_ns,
                                                                   obs.detection_id)
        if key in self.tags:
            return
        blocked = any(FAILURE_SEVERITY[w] is Severity.BLOCK for w in obs.warnings)
        status = TagStatus.PROVISIONAL if blocked else TagStatus.CONFIRMED
        self.tags[key] = TagResult(len(self.tags), obs.cell, status, obs.p_map_m, obs.capture_stamp_ns,
                                         None if blocked else obs.capture_stamp_ns)


class Layer1234Pipeline(Layer123Pipeline):
    """Layers 1-4. Observations finished by Layer 3 are batched per call and handed to Layer 4, which owns the tag
    policy. finalize() returns TagResults for the statuses in `output_statuses` (default: PROVISIONAL, CONFIRMED,
    UPDATED; CANDIDATE and UNRELIABLE are not reported); final_tags(policy) gives the end-of-mission commit set."""

    def __init__(self, cfg: Config, output_statuses=(TagStatus.PROVISIONAL, TagStatus.CONFIRMED, TagStatus.UPDATED),
                 **layer4_overrides):
        super().__init__(cfg)
        self.l4 = Layer4(Layer4Params.from_config(cfg, **layer4_overrides), self.grid)
        self.output_statuses = tuple(output_statuses)
        self._batch: List[SurvivorObservation] = []
        self.l4_ms: List[float] = []

    def _tag(self, obs: SurvivorObservation) -> None:
        self._batch.append(obs)

    def _flush(self) -> None:
        if not self._batch:
            return
        t0 = time.perf_counter()
        self.l4.update(self._batch)
        per_obs = (time.perf_counter() - t0) * 1e3 / len(self._batch)
        self.l4_ms.extend([per_obs] * len(self._batch))
        self._batch = []

    def process_pose(self, pose: PoseSample, arrival_ns: int) -> None:
        super().process_pose(pose, arrival_ns)
        self._flush()

    def process_frame(self, detections: Sequence[Detection], depth: DepthMeasurement,
                      intrinsics: CameraIntrinsics, arrival_ns: int) -> None:
        super().process_frame(detections, depth, intrinsics, arrival_ns)
        self._flush()

    def finalize(self) -> List[TagResult]:
        super().finalize()
        self._flush()
        return [tag_result(t) for t in self.l4.tags() if t.status in self.output_statuses]

    def final_tags(self, policy: str, threshold: Optional[float] = None) -> List[TagResult]:
        return [tag_result(t) for t in self.l4.final_tags(policy, threshold)]


def tag_result(t: Tag) -> TagResult:
    prob = dict(t.cell_probs).get(t.cell) if t.cell is not None else None
    return TagResult(t.survivor_id, t.cell, t.status, t.p_map_m, t.first_seen_ns, t.confirmed_stamp_ns,
                     0.0 if prob is None and t.cell is not None else prob, t.cell_changes)
