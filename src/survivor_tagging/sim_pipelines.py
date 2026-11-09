"""Simulator-side glue: turn a scenario into a Config the way a deployment would configure the real system, and
build pipeline factories for the harness. Library layers never import this module.

The grid registration is converted through the simulator's PUBLIC map_from_grid() only (grid origin and the grid
+x axis expressed in the map frame), so the library's own map->grid definition is checked against an independent
one. All values here are SIMULATION values.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

import numpy as np

from .baseline import naive_factory, naive_layer1_factory
from .config import Config
from .harness import PipelineFactory
from .pipeline import Layer123Pipeline, Layer5Pipeline
from .scenarios import sim_layer1_config
from .sim import Registration, RegistrationUpdate, Scenario

SIM_MAX_INTERP_GAP_S = 0.05  # SIM: poses at 50 Hz -> 0.02 s spacing
SIM_MAX_RGB_DEPTH_SKEW_S = 0.005  # SIM: detections and depth share the capture stamp


def registration_overrides(reg: Registration, n_cells_x: int, n_cells_y: int) -> dict:
    origin = np.asarray(reg.map_from_grid((0.0, 0.0)))
    x_axis = np.asarray(reg.map_from_grid((1.0, 0.0))) - origin
    return {
        "grid.cell_size_m": reg.cell_size_m,
        "grid.n_cells_x": n_cells_x,
        "grid.n_cells_y": n_cells_y,
        "grid.origin_in_map_m": [float(origin[0]), float(origin[1])],
        "grid.yaw_map_to_grid_rad": math.atan2(float(x_axis[1]), float(x_axis[0])),
        "grid.axis_flip_x": False,
        "grid.axis_flip_y": False,
        "grid.index_base": 0,
        "grid.numbering": "ij",
        "grid.registration_valid": True,
    }


def correction_source(scn: Scenario) -> str:
    """l5.correction_source a deployment would declare for this scenario's SLAM (SIM)."""
    modes = {c.mode for c in scn.trajectory.closures}
    if len(modes) > 1:
        raise ValueError(f"{scn.name}: mixed correction sources {sorted(modes)}")
    return modes.pop() if modes else "none"


def scenario_config(scn: Scenario, overrides: Optional[dict] = None) -> Config:
    """The pipeline is configured with scn.initial_registration when set (registration_update scenario)."""
    vals = {
        "extr.T_body_camera": np.asarray(scn.camera.T_body_camera).tolist(),
        "extr.T_body_camera_cov": np.zeros((6, 6)).tolist(),
        "extr.version": 1,
        "clock.camera_to_pose_offset_ns": 0,
        "clock.offset_uncertainty_ns": 0,
        "l2.max_interp_gap_s": SIM_MAX_INTERP_GAP_S,
        "l2.max_rgb_depth_skew_s": SIM_MAX_RGB_DEPTH_SKEW_S,
        "l2.pose_mode": "full_3d",
        "map.floor_z_m": 0.0,  # SIM: the simulated floor is z = 0 in the map frame (grid z = map z)
    }
    vals.update(registration_overrides(scn.initial_registration or scn.registration, scn.arena.n_cells,
                                       scn.arena.n_cells))
    vals["grid.registration_version"] = 0
    vals["l5.correction_source"] = correction_source(scn)
    vals.update(overrides or {})
    return sim_layer1_config(vals)


def layer123_factory(overrides: Optional[dict] = None) -> PipelineFactory:
    return lambda scn: Layer123Pipeline(scenario_config(scn, overrides))


def naive_l1_factory(overrides: Optional[dict] = None) -> PipelineFactory:
    return naive_layer1_factory(sim_layer1_config(overrides))


__all__ = ["layer123_factory", "layer1234_factory", "layer5_factory", "naive_factory", "naive_l1_factory",
           "registration_overrides", "scenario_config", "SimLayer5Pipeline"]


def layer1234_factory(overrides: Optional[dict] = None, **layer4_overrides) -> PipelineFactory:
    from .pipeline import Layer1234Pipeline

    return lambda scn: Layer1234Pipeline(scenario_config(scn, overrides), **layer4_overrides)


class SimLayer5Pipeline(Layer5Pipeline):
    """Layer5Pipeline plus the simulator adapter for registration updates (sim.RegistrationUpdate -> the library
    GridRegistration, through the simulator's public map_from_grid only, like registration_overrides)."""

    def process_registration_event(self, upd: RegistrationUpdate, arrival_ns: int):
        ov = registration_overrides(upd.registration, self.grid.n_cells_x or 0, self.grid.n_cells_y or 0)
        reg = replace(self.grid, origin_in_map_m=tuple(ov["grid.origin_in_map_m"]),
                      yaw_rad=ov["grid.yaw_map_to_grid_rad"], cell_size_m=ov["grid.cell_size_m"],
                      version=upd.version)
        return self.process_registration(reg, arrival_ns)


def layer5_factory(overrides: Optional[dict] = None, sink=None, **layer4_overrides) -> PipelineFactory:
    return lambda scn: SimLayer5Pipeline(scenario_config(scn, overrides), sink=sink, **layer4_overrides)
