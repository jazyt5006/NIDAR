"""Named synthetic scenarios. Every number here is a SIMULATION value, not hardware or competition data.

Scenarios are authored in the official grid frame (cells are easy to reason about there) and converted to the map
frame through a non-trivial registration, so the map frame and the grid always differ.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, Optional, Sequence, Tuple

from .config import Config, default_config
import numpy as np

from .sim import (
    Box,
    Arena,
    DepthNoise,
    DetectorSim,
    Registration,
    Scenario,
    SimCamera,
    SurvivorSpec,
    Trajectory,
    camera_mount,
    wall,
)

# SIMULATION defaults
SIM_REGISTRATION = Registration(offset_m=(1.2, 0.8), yaw_rad=0.25)  # map origin at grid (1.2, 0.8), rotated
SIM_ALTITUDE_M = 1.5
SIM_MOUNT_XYZ_M = (0.10, 0.0, 0.0)
SIM_TILT_DEG = 20.0
SIM_NOISE = DepthNoise(sigma_a_per_m=0.002, dropout=0.02, flying_prob=0.5, flying_edge_m=0.3)
SIM_DETECTOR = DetectorSim(bbox_jitter_px=3.0, miss_prob=0.1, fp_per_frame=0.05, id_switch_prob=0.02,
                           latency_s=0.05)
SIM_WOBBLE_RAD = math.radians(2.0)
BASE_PATH_GRID = ((0.8, 3.3), (1.2, 3.7))  # slow sideways drift in front of the survivors


def make_scenario(name: str, tests: str, survivors: Sequence[SurvivorSpec], *,
                  walls: Sequence = (),
                  path_grid: Sequence[Tuple[float, float]] = BASE_PATH_GRID,
                  yaw_grid_rad: float = 0.0,
                  altitude_m: float = SIM_ALTITUDE_M,
                  tilt_deg: float = SIM_TILT_DEG,
                  noise: DepthNoise = SIM_NOISE,
                  detector: DetectorSim = SIM_DETECTOR,
                  registration: Registration = SIM_REGISTRATION,
                  n_cells: int = 6,
                  duration_s: float = 3.0,
                  trajectory_kw: Optional[dict] = None,
                  placement: str = "random_in_cell") -> Scenario:
    reg = registration
    waypoints = tuple((*reg.map_from_grid(p), altitude_m) for p in path_grid)
    kw = dict(wobble_amp_rad=SIM_WOBBLE_RAD)
    kw.update(trajectory_kw or {})
    traj = Trajectory(waypoints_map=waypoints, yaw0_rad=yaw_grid_rad - reg.yaw_rad, **kw)
    cam = SimCamera(T_body_camera=camera_mount(math.radians(tilt_deg), SIM_MOUNT_XYZ_M), noise=noise)
    return Scenario(name, tests, Arena(n_cells, tuple(walls)), tuple(survivors), cam, detector, traj, reg,
                    duration_s, placement=placement)


def make_ideal(scn: Scenario) -> Scenario:
    """Same geometry, no sensor noise, perfect detector, no pose impairments."""
    traj = replace(scn.trajectory, pose_latency_s=0.0, drift_m_per_s=(0.0, 0.0, 0.0), loop_closure_s=None,
                   loop_closure_jump_m=(0.0, 0.0, 0.0))
    return replace(scn, camera=replace(scn.camera, noise=DepthNoise()), detector=DetectorSim(), trajectory=traj)


def _one(posture: str, xy=(3.5, 3.5), rotated: bool = False, region=None):
    return (SurvivorSpec(1, xy, posture, rotated, region),)


BODY_DEPTH_MODELS: Dict[str, Optional[Tuple[float, float]]] = {
    "fixed": None,  # POSTURE_DIMS_M (0.5 m deep standing/sitting box -> front-surface bias 0.25 m), the default
    "uniform_0.2_0.5": (0.2, 0.5),  # SIM: true depth per survivor -> front-surface bias about 0.10-0.25 m
}


def all_scenarios(placement: str = "random_in_cell", body_depth: str = "fixed") -> Dict[str, Scenario]:
    """All named scenarios. With "random_in_cell" (default) each survivor centre is drawn per seed inside its
    cell, or inside the narrower region a scenario states to keep its walls/occlusion valid; "nominal" uses the
    authored centres. body_depth selects a BODY_DEPTH_MODELS entry (substep 3.1c; "fixed" = unchanged)."""
    s = [
        make_scenario("standing", "standing survivor centred in cell (3,3), ~2.4 m ahead; basic geometry",
                      _one("standing")),
        make_scenario("sitting", "sitting survivor (1.0 m tall) centred in cell (3,3)", _one("sitting")),
        make_scenario("prone", "prone survivor (0.25 m tall, 1.7 m long along view) centred in cell (3,3)",
                      _one("prone")),
        make_scenario("near_wall", "standing survivor with a full wall 0.01-0.3 m behind its back face "
                      "(background contamination of depth)", _one("standing", (3.6, 3.5), region=((3.425, 3.715), (3.0, 4.0))),
                      walls=[wall("x", 4.0, 2.5, 4.5)]),
        make_scenario("cell_boundary", "worst case: standing survivor centred exactly on the x = 3 cell edge "
                      "(true cell 3 by floor); y random in the cell",
                      _one("standing", (3.0, 3.5), region=((3.0, 3.0), (3.0, 4.0)))),
        make_scenario("occluded", "standing survivor whose lower half is hidden behind a 1.2 m half-height wall "
                      "at x = 2; centre still visible", _one("standing", region=((3.2, 4.0), (3.0, 4.0))),
                      walls=[wall("x", 2.0, 2.6, 4.4, height=1.2)], path_grid=((0.5, 3.4), (0.5, 3.6)),
                      altitude_m=1.8),
        make_scenario("two_same_cell", "two standing survivors 0.8 m apart in the same cell (3,3)",
                      (SurvivorSpec(1, (3.4, 3.1), "standing"), SurvivorSpec(2, (3.4, 3.9), "standing"))),
        make_scenario("two_adjacent", "two standing survivors 0.8 m apart in adjacent cells (3,2) and (3,3)",
                      (SurvivorSpec(1, (3.5, 2.6), "standing"), SurvivorSpec(2, (3.5, 3.4), "standing"))),
        make_scenario("low_occluder", "standing survivor behind a 0.9 m wall at x = 2.5 (top below the chest); "
                      "the wall is a band across the bottom of the box", _one("standing", region=((3.2, 3.9), (3.0, 4.0))),
                      walls=[wall("x", 2.5, 2.6, 4.4, height=0.9)], path_grid=((1.0, 3.4), (1.0, 3.6))),
        make_scenario("thin_person_in_front_of_wall", "loose detector box (2.5x the person's width each side) "
                      "around a standing survivor 1.0-1.5 m in front of a full wall at x = 4.5: the person is a "
                      "narrow strip, the wall fills most of the box", _one("standing", region=((3.0, 3.5), (3.0, 4.0))),
                      walls=[wall("x", 4.5, 1.5, 5.5)], detector=replace(SIM_DETECTOR, bbox_pad_frac=1.5)),
    ]
    for tilt in (0, 30, 45):
        s.append(make_scenario(f"tilt_{tilt}", f"camera tilted down {tilt} deg, standing survivor 1.5-2.1 m "
                               "ahead of the camera (camera mount handling)",
                               _one("standing", region=((3.0, 3.6), (3.0, 4.0))),
                               path_grid=((1.4, 3.5),), tilt_deg=tilt))
    s.append(make_scenario("latency_fast_yaw", "pose delivered 100 ms late while yawing +/-25 deg at 0.6 Hz "
                           "(peak ~94 deg/s); tests capture-time pose alignment", _one("standing"),
                           path_grid=((1.0, 3.5),),
                           trajectory_kw=dict(pose_latency_s=0.1, yaw_osc_amp_rad=math.radians(25), yaw_osc_hz=0.6)))
    s.append(make_scenario("loop_closure", "poses before t = 1.5 s carry a 0.3 m SLAM error (map +y, mostly "
                           "sideways to the view) that a loop closure removes (pose_revision 0 -> 1); tests "
                           "correction handling", _one("standing"),
                           trajectory_kw=dict(loop_closure_s=1.5, loop_closure_jump_m=(0.0, 0.3, 0.0))))
    s += _layer4_scenarios()
    if placement != "random_in_cell":
        s = [replace(x, placement=placement) for x in s]
    if BODY_DEPTH_MODELS[body_depth] is not None:
        s = [replace(x, body_depth_range_m=BODY_DEPTH_MODELS[body_depth]) for x in s]
    return {scn.name: scn for scn in s}


# Layer 1 configuration matched to the SIMULATED camera. SIMULATION values, never hardware defaults.
SIM_LAYER1_OVERRIDES = {
    "l1.depth_min_m": 0.1,  # SimCamera.min_range_m
    "l1.depth_max_m": 10.0,  # SimCamera.max_range_m
    "l1.depth_noise_model": [0.0, 0.0, 0.002],  # SIM_NOISE: sigma = 0.002 z^2
}


def sim_layer1_config(overrides: Optional[dict] = None) -> Config:
    """Default config + SIM_LAYER1_OVERRIDES + overrides (dotted names, e.g. {"l1.body_depth_offset_m": 0.2})."""
    cfg = default_config()
    vals = dict(SIM_LAYER1_OVERRIDES)
    vals.update(overrides or {})
    cfg.apply_overrides(vals, reference="SIMULATION (scenarios.sim_layer1_config)")
    return cfg


SIM_POSE_DRIFT_SIGMA = 0.05  # SIM ASSUMPTION: horizontal pose error grows as 0.05 m per sqrt(metre) travelled
PHANTOM_BOX = Box((3.3, 1.8, 0.0), (3.7, 2.2, 0.8))  # SIM: a non-survivor object the detector sometimes fires on


def _layer4_scenarios():
    reg = SIM_REGISTRATION

    def to_map(xy):
        return tuple(float(c) for c in reg.map_from_grid(xy))

    drift = dict(pose_drift_sigma_m_per_sqrt_m=SIM_POSE_DRIFT_SIGMA)
    semicircle = tuple((3.05 + 2 * math.cos(a), 3.5 - 2 * math.sin(a)) for a in np.linspace(math.pi, 0, 7))
    return [
        make_scenario("id_switch", "detector re-assigns a new tracking id in ~30 % of frames (one person, many ids)",
                      _one("standing"), detector=replace(SIM_DETECTOR, id_switch_prob=0.3)),
        make_scenario("id_merge", "two survivors 0.8 m apart; the detector gives both the same tracking id",
                      (SurvivorSpec(1, (3.5, 2.6), "standing"), SurvivorSpec(2, (3.5, 3.4), "standing")),
                      detector=replace(SIM_DETECTOR, id_merge_dist_m=1.0)),
        make_scenario("intermittent_false_positive", "a 0.8 m obstacle in cell (3,1) triggers a false detection "
                      "in ~20 % of frames (fresh id each time); the real survivor is in cell (3,3)",
                      _one("standing", region=((3.0, 4.0), (3.2, 4.0))), walls=[PHANTOM_BOX],
                      detector=replace(SIM_DETECTOR, phantom_box=PHANTOM_BOX, phantom_prob=0.2)),
        make_scenario("revisit_after_gap", "survivor seen for ~2 s, camera turned away for ~6 s while the drone "
                      "flies 2.7 m away and back (pose drift 0.05 m/sqrt(m)), then seen again",
                      _one("standing"), path_grid=((1.0, 3.5), (1.0, 0.8), (1.0, 3.5)), duration_s=11.0,
                      trajectory_kw=dict(yaw_schedule=tuple((t, y - reg.yaw_rad) for t, y in
                                                            ((0, 0.0), (2, 0.0), (3, math.pi), (8, math.pi),
                                                             (9, 0.0), (11, 0.0))), **drift)),
        make_scenario("opposite_views_boundary", "survivor 0.03-0.12 m from the x = 3 cell edge, observed while the "
                      "drone flies a half circle (r = 2 m) from the west side to the east side (pose drift)",
                      _one("standing", (3.05, 3.5), region=((3.03, 3.12), (3.3, 3.7))), path_grid=semicircle,
                      duration_s=6.0, trajectory_kw=dict(look_at_map_xy=to_map((3.05, 3.5)), **drift)),
    ]
