"""Smoke test of the tagging runtime / node (substep 4.1b). Entry point `smoke_test`.

  smoke_test --dry-run                 in-process fake publisher/subscriber layer, no rclpy (RUN on the dev machine)
  smoke_test                           against a running tagger_node over ROS 2 (NOT RUN on the dev machine)

Synthetic inputs come from the library simulator (this tool may import it; the library and the node may not). Both modes
write smoke_report.json (correct cells vs ground truth, message counts, latency, rejection reasons, versions, ROS distro if
present) and exit non-zero on failure.

PASS (dry-run): no exception, no REBUILD_FAILED, every published tag list equals the tags of the library pipeline run
directly on the same events (cells, statuses, revisions; positions within 1e-9), and each scenario tags at least one
survivor in the correct cell in at least one seed. PASS (ROS mode): every scenario tags at least one survivor in the
correct cell in at least one seed (transport timing makes exact equality with the direct run meaningless; cell agreement
with it is reported). All results SIMULATION evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from survivor_tagging import __version__ as LIB_VERSION
from survivor_tagging.harness import run_events, score_tags
from survivor_tagging.scenarios import all_scenarios
from survivor_tagging.sim import build_events, place_survivors
from survivor_tagging.sim_pipelines import layer5_factory, scenario_config
from survivor_tagging.types import TagResult, TagStatus

from . import adapters as A
from . import runtime as R
from . import sim_bridge as B

DEFAULT_SCENARIOS = ("standing", "sitting", "loop_closure", "registration_update", "double_detection")
T0_DRY_NS = 1_000_000_000  # the simulator starts at stamp 0, which ROS treats as unset


class FakeBus:
    """In-process stand-in for publishers/subscribers: records every publish per topic key."""

    def __init__(self) -> None:
        self.items: List[tuple] = []

    def __call__(self, key: str, payload: Any) -> None:
        self.items.append((key, payload))

    def count(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for k, _ in self.items:
            out[k] = out.get(k, 0) + 1
        return out


def _tag_results(tag_dicts: Sequence[Dict[str, Any]]) -> List[TagResult]:
    out = []
    for d in tag_dicts:
        tag, _, removed = A.tag_from_msg_dict(d)
        if removed:
            continue
        out.append(TagResult(tag.survivor_id, tag.cell, tag.status, tag.p_map_m, tag.first_seen_ns,
                             tag.confirmed_stamp_ns, tag.localisation_quality, tag.cell_changes))
    return out


def _score(placed, tag_dicts) -> Dict[str, int]:
    sc = score_tags(placed.truths(), _tag_results(tag_dicts))
    return {"n_truth": sc.n_truth, "n_correct": sc.n_correct, "n_correct_confirmed": sc.n_correct_confirmed,
            "n_missed": sc.n_missed, "n_false": sc.n_false}


def _pct(xs: List[float], q: float) -> Optional[float]:
    return float(np.percentile(xs, q)) if xs else None


def run_dry(scenarios: Sequence[str], seeds: Sequence[int]) -> Dict[str, Any]:
    runs, rejections = [], {}
    timings: Dict[str, List[float]] = {}
    for name in scenarios:
        for seed in seeds:
            placed = place_survivors(all_scenarios()[name], seed)
            events = build_events(placed, seed)
            clock, bus = [0], FakeBus()
            acfg = A.AdapterConfig(map_frame_id="map", body_frame_id="body", body_convention="FLU",
                                   clock_domain=B.SIM_CLOCK_DOMAIN, depth_aligned_to_color=True,
                                   pose_default_position_sigma_m=None, pose_default_yaw_sigma_rad=None)
            rt = R.TaggerRuntime(acfg, scenario_config(placed), lambda: clock[0], bus, rebuild_in_thread=True)
            B.feed_runtime(rt, clock, B.events_to_messages(placed, events, T0_DRY_NS), timings)
            rt.tick(clock[0] + 1)
            rt.finish()
            rt.close()
            direct = layer5_factory()(placed)
            run_events(placed, events, direct, seed)
            diffs = B.tag_diffs(direct.published(), rt.latest_tags)
            for k, v in rt.rejections.items():
                rejections[k] = rejections.get(k, 0) + v
            runs.append({"scenario": name, "seed": seed, "equal_to_direct_pipeline": not diffs, "diffs": diffs[:5],
                         "messages": bus.count(), "tags_published": len((rt.latest_tags or {}).get("tags", [])),
                         "rebuilds": rt.counts.get("rebuilds", 0), **_score(placed, (rt.latest_tags or {}).get("tags", []))})
    return {"runs": runs, "rejections": rejections, "latency_ms": {
        k: {"median": statistics.median(v), "p95": _pct(v, 95), "max": max(v), "n": len(v)} for k, v in timings.items()}}


def run_ros(scenarios: Sequence[str], seeds: Sequence[int], speed: float, settle_s: float) -> Dict[str, Any]:  # pragma: no cover
    """NOT RUN on the dev machine. Publishes the simulator data (stamps shifted onto the ROS clock) to a running tagger_node
    at real-time pace and reads /survivor_tagging/tags."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from sensor_msgs.msg import CameraInfo, Image
    from survivor_tagging_msgs.msg import CorrectionEvent, DetectionArray, SurvivorTagArray
    from survivor_tagging_msgs.srv import SetGridRegistration
    from .convert import dict_to_msg

    rclpy.init()
    node = Node("smoke_test")
    sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=5)
    rel = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
    latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
    pubs = {"camera_info": node.create_publisher(CameraInfo, "/camera/aligned_depth_to_color/camera_info", sensor),
            "depth": node.create_publisher(Image, "/camera/aligned_depth_to_color/image_raw", sensor),
            "detections": node.create_publisher(DetectionArray, "/survivor_tagging/detections", rel),
            "pose": node.create_publisher(PoseWithCovarianceStamped, "/slam/pose", sensor),
            "correction": node.create_publisher(CorrectionEvent, "/slam/correction", latched)}
    classes = {"camera_info": CameraInfo, "depth": Image, "detections": DetectionArray,
               "pose": PoseWithCovarianceStamped, "correction": CorrectionEvent}
    latest: Dict[str, Any] = {"tags": None, "n": 0}

    def on_tags(m) -> None:
        latest["tags"], latest["n"] = m, latest["n"] + 1

    node.create_subscription(SurvivorTagArray, "/survivor_tagging/tags", on_tags, latched)
    reg_client = node.create_client(SetGridRegistration, "/survivor_tagging/set_grid_registration")
    if not reg_client.wait_for_service(timeout_sec=10.0):
        raise RuntimeError("tagger_node services not found: is the node running?")
    runs = []
    for name in scenarios:
        for seed in seeds:
            placed = place_survivors(all_scenarios()[name], seed)
            events = build_events(placed, seed)
            t0 = node.get_clock().now().nanoseconds + 2_000_000_000
            msgs = B.events_to_messages(placed, events, t0)
            latest.update(tags=None, n=0)
            wall0, a0 = time.monotonic(), msgs[0].arrival_ns  # pacing only; stamps come from the simulator + t0
            for m in msgs:
                while (time.monotonic() - wall0) * speed < (m.arrival_ns - a0) / 1e9:
                    rclpy.spin_once(node, timeout_sec=0.002)
                if m.kind == "registration":
                    fut = reg_client.call_async(dict_to_msg(SetGridRegistration.Request, m.msg))
                    rclpy.spin_until_future_complete(node, fut, timeout_sec=10.0)
                else:
                    pubs[m.kind].publish(dict_to_msg(classes[m.kind], m.msg))
            end = time.monotonic() + settle_s
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.05)
            tags = []
            if latest["tags"] is not None:
                tags = [{"survivor_id": t.survivor_id, "status": t.status, "has_cell": t.has_cell, "cell_i": t.cell_i,
                         "cell_j": t.cell_j, "has_position": t.has_position,
                         "position_map": {"x": t.position_map.x, "y": t.position_map.y, "z": t.position_map.z}}
                        for t in latest["tags"].tags]
            res = score_tags(placed.truths(), [TagResult(t["survivor_id"], (t["cell_i"], t["cell_j"]) if t["has_cell"]
                                                         else None, TagStatus(t["status"]),
                                                         np.array([t["position_map"][k] for k in "xyz"])
                                                         if t["has_position"] else None, 0, None) for t in tags])
            runs.append({"scenario": name, "seed": seed, "tag_messages": latest["n"], "n_truth": res.n_truth,
                         "n_correct": res.n_correct, "n_correct_confirmed": res.n_correct_confirmed,
                         "n_missed": res.n_missed, "n_false": res.n_false})
    node.destroy_node()
    rclpy.shutdown()
    return {"runs": runs}


def evaluate(result: Dict[str, Any], dry: bool) -> Dict[str, Any]:
    ok = True
    reasons: List[str] = []
    by_scn: Dict[str, int] = {}
    for r in result["runs"]:
        by_scn[r["scenario"]] = by_scn.get(r["scenario"], 0) + r["n_correct"]
        if dry and not r["equal_to_direct_pipeline"]:
            ok = False
            reasons.append(f"{r['scenario']} seed {r['seed']}: differs from the direct pipeline: {r['diffs'][:2]}")
    for scn, n in by_scn.items():
        if n < 1:
            ok = False
            reasons.append(f"{scn}: no survivor tagged in the correct cell in any seed")
    if result.get("rejections", {}).get("REBUILD_FAILED"):
        ok = False
        reasons.append("a rebuild failed")
    return {"pass": ok, "failures": reasons}


def sim_overrides(scenario: str = "standing") -> Dict[str, Any]:
    """Flat library override dict (JSON-able) with every REQUIRED library parameter of the simulator scene (extrinsics,
    clock, depth range, l2 limits, grid registration, ...). SIMULATION values: use only for the smoke test, never on hardware."""
    cfg = scenario_config(place_survivors(all_scenarios()[scenario], 0))
    out = {p.name: (p.value.tolist() if hasattr(p.value, "tolist") else p.value) for p in cfg.params()
           if p.value is not None and cfg.is_required(p)}
    # Optional but part of the simulated scene (substep 4.3): the simulated registration is exact (sigma 0.0), which
    # also avoids REG_UNCERTAINTY_UNKNOWN on every tag in the smoke test. Never a hardware value.
    for name in ("grid.origin_sigma_m", "grid.yaw_sigma_rad"):
        out[name] = cfg.value(name)
    return out


def versions() -> Dict[str, Any]:
    return {"python": sys.version.split()[0], "numpy": np.__version__, "survivor_tagging": LIB_VERSION,
            "platform": platform.platform(), "ROS_DISTRO": os.environ.get("ROS_DISTRO"),
            "ROS_VERSION": os.environ.get("ROS_VERSION")}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="smoke test of the survivor tagging runtime / node")
    ap.add_argument("--dry-run", action="store_true", help="no ROS: in-process fake publisher/subscriber layer")
    ap.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--out", default="smoke_report.json")
    ap.add_argument("--write-sim-overrides", metavar="PATH", help="write the simulator scene's required library "
                    "parameters as an overrides_file for the node (SIMULATION values) and exit")
    ap.add_argument("--speed", type=float, default=1.0, help="ROS mode: replay speed factor")
    ap.add_argument("--settle", type=float, default=3.0, help="ROS mode: seconds to wait after the last message")
    a = ap.parse_args(argv)
    if a.write_sim_overrides:
        with open(a.write_sim_overrides, "w", encoding="utf-8") as f:
            json.dump({"_note": "SIMULATION VALUES for the smoke test only, not hardware", **sim_overrides()}, f, indent=1)
        print(f"wrote {a.write_sim_overrides}")
        return 0
    scenarios, seeds = [s for s in a.scenarios.split(",") if s], [int(s) for s in a.seeds.split(",") if s]
    t = time.monotonic()
    try:
        result = run_dry(scenarios, seeds) if a.dry_run else run_ros(scenarios, seeds, a.speed, a.settle)
        verdict = evaluate(result, a.dry_run)
    except Exception as exc:  # noqa: BLE001 - reported in the file, exit code 2
        result, verdict = {"runs": []}, {"pass": False, "failures": [f"exception: {exc!r}"]}
    report = {"mode": "dry-run" if a.dry_run else "ros", "evidence": "simulation", "scenarios": scenarios,
              "seeds": seeds, "versions": versions(), "duration_s": time.monotonic() - t, **verdict, **result}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    n_ok = sum(1 for r in result["runs"] if r.get("n_correct", 0) >= 1)
    print(f"smoke_test {'PASS' if verdict['pass'] else 'FAIL'} ({report['mode']}): {len(result['runs'])} runs, "
          f"{n_ok} with a correct cell; report {a.out}")
    for line in verdict["failures"]:
        print("  FAIL:", line)
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
