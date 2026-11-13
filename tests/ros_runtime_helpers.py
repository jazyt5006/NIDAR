"""Shared helpers for the runtime tests and the smoke tool tests: build the runtime for a simulator scenario, run the
library pipeline directly on the same events, compare tag lists."""
from typing import Any, Dict, List, Tuple

import numpy as np

import ros_fakes as rf  # noqa: F401  (puts ros2/survivor_tagging_ros on sys.path)
from survivor_tagging_ros import adapters as A
from survivor_tagging_ros import runtime as R
from survivor_tagging_ros import sim_bridge as B
from survivor_tagging.harness import run_events
from survivor_tagging.scenarios import all_scenarios
from survivor_tagging.sim import build_events, place_survivors
from survivor_tagging.sim_pipelines import layer5_factory, scenario_config


T0_NS = 1_000_000_000  # the simulator starts at stamp 0, which ROS treats as unset: shift every stamp by 1 s


def adapter_cfg(**kw) -> A.AdapterConfig:
    base = dict(map_frame_id="map", body_frame_id="body", body_convention="FLU", clock_domain=B.SIM_CLOCK_DOMAIN,
                depth_aligned_to_color=True, pose_default_position_sigma_m=None, pose_default_yaw_sigma_rad=None)
    base.update(kw)
    return A.AdapterConfig(**base)


class Bus:
    """Captures everything the runtime publishes."""

    def __init__(self):
        self.items: List[Tuple[str, Any]] = []

    def __call__(self, key, payload):
        self.items.append((key, payload))

    def of(self, key):
        return [p for k, p in self.items if k == key]


def make_runtime(placed, clock, bus, thread=False, **acfg):
    return R.TaggerRuntime(adapter_cfg(**acfg), scenario_config(placed), lambda: clock[0], bus, rebuild_in_thread=thread)


def run_via_messages(name: str, seed: int, thread: bool = False, offset: float = 0.0, **acfg):
    placed = place_survivors(all_scenarios()[name], seed)
    events = build_events(placed, seed)
    clock, bus = [0], Bus()
    rt = R.TaggerRuntime(adapter_cfg(**acfg), scenario_config(placed, {"l1.body_depth_offset_m": offset}),
                         lambda: clock[0], bus, rebuild_in_thread=thread)
    B.feed_runtime(rt, clock, B.events_to_messages(placed, events, T0_NS))
    rt.finish()
    rt.close()
    return placed, events, rt, bus


def run_direct(placed, events, seed: int, offset: float = 0.0):
    pl = layer5_factory({"l1.body_depth_offset_m": offset})(placed)
    run_events(placed, events, pl, seed)
    return pl


def compare_published(direct_pl, tags_payload, tol=1e-9):
    """Assert equality (see sim_bridge.tag_diffs); returns the number of tags."""
    diffs = B.tag_diffs(direct_pl.published(), tags_payload, tol)
    assert not diffs, diffs
    return len(direct_pl.published())
