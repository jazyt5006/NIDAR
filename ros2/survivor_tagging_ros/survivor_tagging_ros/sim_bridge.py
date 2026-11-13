"""Simulator -> message-shaped dicts (substep 4.1b). TEST / SMOKE-TEST SIDE ONLY: this module imports the library
simulator (survivor_tagging.sim); the runtime and the node never import it.

Every message is a plain dict with the field names of the .msg file (or the ROS message it stands for): nested messages are
dicts, arrays are lists, image data is bytes. `ns()` turns such a dict into an attribute-style object for the ROS-free
runtime; survivor_tagging_ros.convert.dict_to_msg turns it into a real ROS message.
Stamps are the simulator's (shifted by `t0_ns` if given, so a real ROS clock can be used).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from survivor_tagging.sim import CLOCK, Event, Frame, RegistrationUpdate, Scenario
from survivor_tagging.sim_pipelines import registration_overrides
from survivor_tagging.transforms import matrix_to_quat
from survivor_tagging.types import PoseCorrection, PoseSample

from .adapters import ns_to_stamp

SIM_CLOCK_DOMAIN = CLOCK  # "sim"


@dataclass
class BridgeMsg:
    arrival_ns: int
    kind: str  # camera_info | depth | detections | pose | correction | registration
    msg: Dict[str, Any]


def ns(d: Any) -> Any:
    """dict / list -> SimpleNamespace / list (recursively), everything else unchanged."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [ns(v) for v in d]
    return d


def _header(stamp_ns: int, frame_id: str) -> Dict[str, Any]:
    return {"stamp": ns_to_stamp(stamp_ns), "frame_id": frame_id}


def pose_msg(p: PoseSample, t0_ns: int = 0) -> Dict[str, Any]:
    """PoseWithCovarianceStamped as a dict (no covariance -> zeros = unknown, as ROS drivers publish it)."""
    q = p.orientation_wxyz
    cov = [0.0] * 36 if p.covariance is None else [float(v) for v in np.asarray(p.covariance).reshape(36)]
    return {"header": _header(p.stamp_ns + t0_ns, p.frame_id),
            "pose": {"pose": {"position": {"x": float(p.position_m[0]), "y": float(p.position_m[1]),
                                           "z": float(p.position_m[2])},
                              "orientation": {"x": float(q[1]), "y": float(q[2]), "z": float(q[3]), "w": float(q[0])}},
                     "covariance": cov}}


def camera_info_msg(fr: Frame, t0_ns: int = 0) -> Dict[str, Any]:
    i = fr.intrinsics
    return {"header": _header(fr.stamp_ns + t0_ns, fr.depth.frame_id), "width": i.width_px, "height": i.height_px,
            "distortion_model": i.distortion_model, "d": list(i.distortion_coeffs),
            "k": [i.fx_px, 0.0, i.cx_px, 0.0, i.fy_px, i.cy_px, 0.0, 0.0, 1.0]}


def depth_msg(fr: Frame, t0_ns: int = 0) -> Dict[str, Any]:
    raw = np.asarray(fr.depth.depth_raw)
    if raw.dtype != np.uint16 or fr.depth.depth_scale_m_per_unit != 0.001:
        raise ValueError("the simulator is expected to produce uint16 millimetres")
    h, w = raw.shape
    return {"header": _header(fr.depth.depth_stamp_ns + t0_ns, fr.depth.frame_id), "height": h, "width": w,
            "encoding": "16UC1", "is_bigendian": 0, "step": w * 2, "data": raw.astype("<u2").tobytes()}


def detections_msg(fr: Frame, t0_ns: int = 0) -> Dict[str, Any]:
    dets = []
    for k, d in enumerate(fr.detections):
        dets.append({"frame_index": 0, "bbox_xyxy": [float(v) for v in d.bbox_px], "confidence": float(d.confidence),
                     "class_id": 0, "class_name": d.class_label or "person",
                     "tracking_id": -1 if d.tracking_id is None else int(d.tracking_id),
                     "source_image_width": d.image_size_px[0], "source_image_height": d.image_size_px[1]})
    frame = fr.detections[0].image_frame_id if fr.detections else fr.depth.frame_id
    stamp = fr.detections[0].image_stamp_ns if fr.detections else fr.stamp_ns
    return {"header": _header(stamp + t0_ns, frame), "detections": dets}


def correction_msg(c: PoseCorrection, frame_id: str = "map", t0_ns: int = 0) -> Dict[str, Any]:
    zero = ns_to_stamp(0)
    out = {"header": _header(c.stamp_ns + t0_ns, frame_id), "revision": int(c.revision), "mode": c.source,
           "range_start": zero, "range_end": zero, "corrected_poses": [],
           "frame_transform": {"translation": {"x": 0.0, "y": 0.0, "z": 0.0},
                               "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}}
    if c.source == "history":
        out["corrected_poses"] = [pose_msg(p, t0_ns) for p in c.poses]
        out["range_start"], out["range_end"] = ns_to_stamp(c.poses[0].stamp_ns + t0_ns), \
            ns_to_stamp(c.poses[-1].stamp_ns + t0_ns)
    else:
        t = np.asarray(c.T_new_old)
        q = matrix_to_quat(t[:3, :3])
        out["frame_transform"] = {"translation": {"x": float(t[0, 3]), "y": float(t[1, 3]), "z": float(t[2, 3])},
                                  "rotation": {"x": float(q[1]), "y": float(q[2]), "z": float(q[3]),
                                               "w": float(q[0])}}
    return out


def registration_request(upd: RegistrationUpdate, n_cells: int) -> Dict[str, Any]:
    ov = registration_overrides(upd.registration, n_cells, n_cells)
    return {"origin_in_map_m": list(ov["grid.origin_in_map_m"]), "yaw_map_to_grid_rad": ov["grid.yaw_map_to_grid_rad"],
            "axis_flip_x": ov["grid.axis_flip_x"], "axis_flip_y": ov["grid.axis_flip_y"],
            "index_base": ov["grid.index_base"], "numbering": ov["grid.numbering"],
            "cell_size_m": ov["grid.cell_size_m"], "n_cells_x": n_cells, "n_cells_y": n_cells,
            "version": int(upd.version), "valid": True}


def events_to_messages(placed: Scenario, events: Sequence[Event], t0_ns: int = 0) -> List[BridgeMsg]:
    """The simulator's event stream as messages, in arrival order (a frame becomes camera_info, depth, detections)."""
    out: List[BridgeMsg] = []
    for ev in events:
        a = ev.arrival_ns + t0_ns
        if ev.kind == "pose":
            out.append(BridgeMsg(a, "pose", pose_msg(ev.payload, t0_ns)))
        elif ev.kind == "frame":
            out += [BridgeMsg(a, "camera_info", camera_info_msg(ev.payload, t0_ns)),
                    BridgeMsg(a, "depth", depth_msg(ev.payload, t0_ns)),
                    BridgeMsg(a, "detections", detections_msg(ev.payload, t0_ns))]
        elif ev.kind == "correction":
            out.append(BridgeMsg(a, "correction", correction_msg(ev.payload, "map", t0_ns)))
        elif ev.kind == "registration":
            out.append(BridgeMsg(a, "registration", registration_request(ev.payload, placed.arena.n_cells)))
    return out


def feed_runtime(rt: Any, clock: List[int], msgs: Sequence[BridgeMsg],
                 timings: Optional[Dict[str, List[float]]] = None) -> List[Tuple[str, Any]]:
    """Calls the runtime's input methods in arrival order; `clock` is the one-element list the runtime's injected clock
    reads (set to each message's arrival time). Returns (kind, result) per message: rejection reason or service reply.
    If `timings` is given, the handler wall time (ms) is appended per kind (processing latency, laptop only)."""
    import time as _time  # measuring handler duration only
    results: List[Tuple[str, Any]] = []
    for m in msgs:
        clock[0] = m.arrival_ns
        fn = {"camera_info": rt.on_camera_info, "depth": rt.on_depth, "detections": rt.on_detections,
              "pose": rt.on_pose, "correction": rt.on_correction, "registration": rt.call_set_registration}[m.kind]
        obj = ns(m.msg)
        t = _time.perf_counter()
        results.append((m.kind, fn(obj)))
        if timings is not None:
            timings.setdefault(m.kind, []).append((_time.perf_counter() - t) * 1e3)
    return results


def tag_diffs(direct_published: Sequence[Any], tags_payload: Optional[Dict[str, Any]], tol: float = 1e-9) -> List[str]:
    """Differences between the tags of a directly run library pipeline (PublishedTag list) and a SurvivorTagArray dict:
    survivors, tag revisions, cells, statuses, reasons, counts and revisions equal; positions, covariance and cell
    probabilities within tol. Empty list = equal."""
    from .adapters import tag_from_msg_dict
    if tags_payload is None:
        return ["no tag array was published"]
    want = sorted(direct_published, key=lambda p: p.survivor_id)
    got = sorted((tag_from_msg_dict(d) for d in tags_payload["tags"]), key=lambda t: t[0].survivor_id)
    if len(want) != len(got):
        return [f"{len(want)} tags expected, {len(got)} published"]
    out: List[str] = []
    for pt, (tag, rev, removed) in zip(want, got):
        t, sid = pt.tag, pt.survivor_id
        for name, a, b in (("id", sid, tag.survivor_id), ("tag_revision", pt.tag_revision, rev),
                           ("removed", pt.removed, removed), ("status", t.status, tag.status), ("cell", t.cell, tag.cell),
                           ("out_of_grid", t.out_of_grid, tag.out_of_grid), ("cell_changes", t.cell_changes,
                                                                             tag.cell_changes),
                           ("reasons", t.reasons, tag.reasons), ("observation_count", t.observation_count,
                                                                 tag.observation_count),
                           ("map_rev", t.map_revision, tag.map_revision),
                           ("reg_rev", t.registration_version, tag.registration_version),
                           ("candidate cells", [c for c, _ in t.cell_probs], [c for c, _ in tag.cell_probs])):
            if a != b:
                out.append(f"survivor {sid}: {name} {a!r} != {b!r}")
        if (t.p_map_m is None) != (tag.p_map_m is None):
            out.append(f"survivor {sid}: position presence differs")
        elif t.p_map_m is not None:
            if not np.allclose(t.p_map_m, tag.p_map_m, atol=tol, rtol=0):
                out.append(f"survivor {sid}: position {t.p_map_m} != {tag.p_map_m}")
            if not np.allclose(t.cov_map_m2[:2, :2], tag.cov_map_m2[:2, :2], atol=tol, rtol=0):
                out.append(f"survivor {sid}: covariance differs")
        if not np.allclose([p for _, p in t.cell_probs], [p for _, p in tag.cell_probs], atol=tol, rtol=0):
            out.append(f"survivor {sid}: cell probabilities differ")
    return out
