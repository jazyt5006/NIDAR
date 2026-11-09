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
    Closure,
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
                   loop_closure_jump_m=(0.0, 0.0, 0.0), closures=())
    return replace(scn, camera=replace(scn.camera, noise=DepthNoise()), detector=DetectorSim(), trajectory=traj,
                   initial_registration=None, registration_updates=())


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
                           trajectory_kw=dict(loop_closure_s=1.5, loop_closure_jump_m=(0.0, 0.3, 0.0),
                                              closures=(Closure(1.5),))))
    s += _layer4_scenarios()
    s += _layer5_scenarios()
    s += _layer33a_scenarios()
    s += _two_close_scenarios()
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


# ---------------------------------------------------------------------------
# Layer 5 scenarios (substep 3.2). SIMULATION values.
# ---------------------------------------------------------------------------

NEW_IN_32 = ("loop_closure_late", "repeated_corrections", "partial_history", "registration_update", "frame_correction")


NEW_IN_33A = ("double_detection",)
# Substep 3.3b: members of the two_close_survivors family (name prefix "two_close_survivors/").
TWO_CLOSE_PREFIX = "two_close_survivors/"


def pre_32_scenarios(**kw) -> Dict[str, Scenario]:
    """The 20 scenarios that existed before substep 3.2 (earlier evaluations stay reproducible)."""
    return {k: v for k, v in all_scenarios(**kw).items()
            if k not in NEW_IN_32 + NEW_IN_33A and not k.startswith(TWO_CLOSE_PREFIX)}


LAYER5_SCENARIOS = ("loop_closure", "loop_closure_late", "repeated_corrections", "partial_history",
                    "registration_update", "frame_correction")
CORRECTION_SCENARIOS = ("loop_closure", "loop_closure_late", "repeated_corrections", "partial_history",
                        "frame_correction")
SIM_WRONG_REGISTRATION = Registration(offset_m=(1.2, 1.25), yaw_rad=0.28)  # SIM: 0.45 m and 0.03 rad off the truth


def with_correction_mode(scn: Scenario, mode: str) -> Scenario:
    """Same scenario (same name, so the same random streams) with every closure delivered as `mode`."""
    tr = scn.trajectory
    return replace(scn, trajectory=replace(tr, closures=tuple(replace(c, mode=mode) for c in tr.closures)))


def without_correction_events(scn: Scenario) -> Scenario:
    """Same live poses and frames; the correction events are never delivered (missing correction)."""
    tr = scn.trajectory
    return replace(scn, trajectory=replace(tr, closures=tuple(replace(c, emit=False) for c in tr.closures)))


def _layer5_scenarios():
    reg = SIM_REGISTRATION
    away = tuple((t, y - reg.yaw_rad) for t, y in ((0.0, 0.0), (2.5, 0.0), (3.5, math.pi), (7.0, math.pi)))
    return [
        make_scenario("loop_closure_late", "survivor seen 0.2-2.5 s (tag CONFIRMED) under a 0.4 m pose error (map "
                      "+y); camera then turned away; loop closure at 5.0 s, history correction arrives 0.2 s later",
                      _one("standing"), duration_s=7.0,
                      trajectory_kw=dict(yaw_schedule=away, loop_closure_s=5.0, loop_closure_jump_m=(0.0, 0.4, 0.0),
                                         closures=(Closure(5.0),))),
        make_scenario("repeated_corrections", "pose drift 0.12 m/s (map +y) that restarts at two loop closures "
                      "(2.0 s, 4.0 s); the second correction re-optimises the whole history [0, 4.0 s)",
                      _one("standing"), duration_s=6.0,
                      trajectory_kw=dict(drift_m_per_s=(0.0, 0.12, 0.0), closures=(Closure(2.0), Closure(4.0)))),
        make_scenario("partial_history", "as loop_closure (0.3 m error before 1.5 s), but the corrected history "
                      "covers only [0.9, 1.5) s: older observations stay uncorrected (POSE_NOT_CORRECTED)",
                      _one("standing"),
                      trajectory_kw=dict(loop_closure_s=1.5, loop_closure_jump_m=(0.0, 0.3, 0.0),
                                         closures=(Closure(1.5, range_s=(0.9, 1.5)),))),
        replace(make_scenario("registration_update", "pipeline starts with a grid registration 0.45 m / 0.03 rad off "
                              "the truth; the true registration arrives at 1.5 s (version 1)", _one("standing")),
                initial_registration=SIM_WRONG_REGISTRATION, registration_updates=((1.5, reg),)),
        make_scenario("frame_correction", "pose drift 0.15 m/s (map +y) removed at 2.5 s by a single map->odom style "
                      "jump (frame_correction source): exact only at the closure, earlier observations keep part "
                      "of their error", _one("standing"), duration_s=4.0,
                      trajectory_kw=dict(drift_m_per_s=(0.0, 0.15, 0.0),
                                         closures=(Closure(2.5, mode="frame_correction"),))),
    ]


# ---------------------------------------------------------------------------
# Substep 3.3a scenarios. SIMULATION values.
# ---------------------------------------------------------------------------

SIM_DOUBLE_DETECTION_PROB = 0.5  # SIM: per detected frame, a second box on the same person
SIM_DOUBLE_DETECTION_OFFSET_M = (0.1, 0.4)  # SIM: sideways world offset of the second box at the person's depth


def _layer33a_scenarios():
    return [
        make_scenario("double_detection", "one standing survivor; in ~50 % of the frames the detector fires a second "
                      "box on the same person (fresh tracking id; the box is widened sideways by 0.1-0.4 m at the "
                      "person's depth, so both estimates land on the body); tests near-duplicate observations in "
                      "one frame. The centre is kept 0.35-1.0 m inside cell (3,3) along x and 0.1 m from its y "
                      "edges so that the known front-surface bias (l1.body_depth_offset_m = 0) cannot push the tag "
                      "into the neighbouring cell: the scenario measures duplicate handling, not that bias",
                      _one("standing", (3.675, 3.5), region=((3.35, 4.0), (3.1, 3.9))),
                      detector=replace(SIM_DETECTOR, dup_prob=SIM_DOUBLE_DETECTION_PROB,
                                       dup_offset_m=SIM_DOUBLE_DETECTION_OFFSET_M)),
    ]


# ---------------------------------------------------------------------------
# Substep 3.3b scenario family: two real survivors close together. SIMULATION values.
# ---------------------------------------------------------------------------

# (tag, posture of the lower-y survivor, posture of the upper-y survivor, centre separation along grid y, adjacent
# cells?). Separations are 0.30-0.60 m, chosen so that the two boxes do not overlap: the y extent is 0.3 m for a
# standing and 0.5 m for a sitting survivor (sim.POSTURE_DIMS_M). Adjacent = the pair straddles the y = 3 cell edge
# (cells (3,2) and (3,3)); otherwise both lie in cell (3,3). The camera looks along +x, so the pair is side by side.
TWO_CLOSE_PAIRS = (
    ("stand_030_same", "standing", "standing", 0.30, False),
    ("stand_030_adj", "standing", "standing", 0.30, True),
    ("stand_045_same", "standing", "standing", 0.45, False),
    ("stand_045_adj", "standing", "standing", 0.45, True),
    ("stand_060_adj", "standing", "standing", 0.60, True),
    ("mix_040_same", "standing", "sitting", 0.40, False),
    ("sit_050_adj", "sitting", "sitting", 0.50, True),
    ("sit_060_same", "sitting", "sitting", 0.60, False),
)


def two_close_names() -> Tuple[str, ...]:
    return tuple(TWO_CLOSE_PREFIX + t[0] for t in TWO_CLOSE_PAIRS)


def _two_close_scenarios():
    out = []
    for tag, p1, p2, sep, adj in TWO_CLOSE_PAIRS:
        yc = 3.0 if adj else 3.5
        y1, y2 = yc - sep / 2, yc + sep / 2
        xr = (3.25, 3.75)  # keeps the centre away from the x = 3 and x = 4 edges (front-surface bias, as double_detection)
        if adj:  # each centre stays in its own cell: the common shift is in [-sep/2, sep/2]
            r1, r2 = (xr, (2.0, 3.0)), (xr, (3.0, 4.0))
        else:
            r1 = r2 = (xr, (3.05, 3.95))
        out.append(make_scenario(
            TWO_CLOSE_PREFIX + tag, f"two survivors ({p1}, {p2}) with centres {sep:.2f} m apart "
            f"({'adjacent cells (3,2) and (3,3)' if adj else 'same cell (3,3)'}), side by side for the camera; "
            "tests whether a close real neighbour is lost as a near-duplicate",
            (SurvivorSpec(1, (3.5, y1), p1, region=r1), SurvivorSpec(2, (3.5, y2), p2, region=r2))))
    return out
