"""TaggerRuntime: everything the ROS node does except talking to ROS (substep 4.1b). Pure Python, uses the adapters and
the survivor_tagging library only; no rclpy, no ROS message imports. Evidence: unit / simulation tested here.

Inputs are plain method calls with message-like objects (duck-typed, see adapters.py). Outputs go through
publish(topic_key, payload) with payload a plain dict/list structure whose keys are the field names of the matching
message: "tags" (SurvivorTagArray), "markers" ({"markers": [Marker-like dicts]}), "telemetry" (TelemetryPacket),
"diagnostics" (plain dict, published as JSON text by the node).

TIME. `now_ns_fn` is the injected clock. It is used for exactly three things: the pipeline's *arrival* time (staleness
and pose-wait limits), the 1 Hz diagnostics timer and the pose-lag diagnostic. The capture time of every observation
comes only from message header stamps (adapters.stamp_to_ns); a test proves the tags do not depend on the clock value.

THREADING. One RLock (`self.lock`) protects ALL pipeline state; every public method takes it. A full recompute
(Layer5Pipeline snapshot -> RebuildJob) runs on the worker thread: the worker only executes job.run() on the private
snapshot and puts the finished job on a queue.Queue; it never touches the pipeline. The swap (complete_rebuild) is done by
the caller's thread, under the lock, in the next input handler or tick. While a job is in flight the live path keeps
updating the old state; observations accepted meanwhile are replayed into the new state at the swap. A job superseded
by a newer rebuild is discarded (counted in diagnostics). commit_final_tags waits for in-flight rebuilds first.
"""

from __future__ import annotations

import json
import math
import queue
import threading
from collections import Counter, deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from survivor_tagging.config import Config, ConfigError, default_config, load_config
from survivor_tagging.pipeline import Layer5Pipeline, RebuildJob
from survivor_tagging.types import DepthMeasurement, FailureReason, TagStatus

from . import adapters as A

PublishFn = Callable[[str, Any], None]
STAGES_REQUIRED = ("l1", "l2", "extrinsics", "clock")  # without these the pipeline cannot run at all
DIAG_PERIOD_NS = 1_000_000_000

STATUS_COLOR = {  # r, g, b, a (marker colour by tag status)
    TagStatus.CANDIDATE: (0.6, 0.6, 0.6, 0.3), TagStatus.PROVISIONAL: (1.0, 0.85, 0.0, 0.4),
    TagStatus.CONFIRMED: (0.0, 0.8, 0.2, 0.5), TagStatus.UPDATED: (0.0, 0.5, 1.0, 0.5),
    TagStatus.UNRELIABLE: (0.9, 0.0, 0.0, 0.4)}
MARKER_CUBE, MARKER_TEXT, ACTION_ADD, ACTION_DELETE = 1, 9, 0, 2  # visualization_msgs/Marker constants
MARKER_Z_M = 0.0  # ASSUMPTION: cell squares are drawn at map z = 0 (map.floor_z_m may be unset)


class RebuildWorker:
    """One background thread running RebuildJob.run() on private snapshots (see module docstring)."""

    def __init__(self) -> None:
        self._in: "queue.Queue[Optional[RebuildJob]]" = queue.Queue()
        self.done: "queue.Queue[RebuildJob]" = queue.Queue()
        self.before_run: Optional[Callable[[RebuildJob], None]] = None  # test hook (e.g. wait on an Event)
        self._thread: Optional[threading.Thread] = None
        self.n_inflight = 0
        self._count_lock = threading.Lock()

    def submit(self, job: RebuildJob) -> None:
        with self._count_lock:
            self.n_inflight += 1
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._main, name="rebuild-worker", daemon=True)
            self._thread.start()
        self._in.put(job)

    def _main(self) -> None:
        while True:
            job = self._in.get()
            if job is None:
                return
            try:
                if self.before_run is not None:
                    self.before_run(job)
                job.run()
            except BaseException:  # noqa: BLE001 - the error is kept on job.error and reported by the runtime
                pass
            self.done.put(job)

    def finished(self) -> List[RebuildJob]:
        out = []
        while True:
            try:
                out.append(self.done.get_nowait())
            except queue.Empty:
                break
        with self._count_lock:
            self.n_inflight -= len(out)
        return out

    def close(self) -> None:
        self._in.put(None)


class TaggerRuntime:
    def __init__(self, adapter_cfg: A.AdapterConfig, lib_cfg: Config, now_ns_fn: Callable[[], int], publish: PublishFn,
                 *, depth_buffer_size: int = 8, rebuild_in_thread: bool = True, pose_max_backward_s: float = 0.0,
                 log: Optional[Callable[[str], None]] = None):
        """depth_buffer_size 8 and pose_max_backward_s 0.0 are ASSUMPTIONS (parameters of the node)."""
        missing = [n for st in STAGES_REQUIRED for n in lib_cfg.missing_required(st, include_dependencies=False)]
        if missing:
            raise A.ConfigurationError(sorted(set(missing)))
        skew = lib_cfg.value("l2.max_rgb_depth_skew_s")
        if skew is None:
            raise A.ConfigurationError(["l2.max_rgb_depth_skew_s"])
        self.skew_ns = int(round(float(skew) * A.NS_PER_S))
        aligned = adapter_cfg.depth_aligned_to_color
        if aligned is None:
            aligned = lib_cfg.value("cam.detections_use_depth_aligned_frame")
        self.cfg = replace(adapter_cfg, depth_aligned_to_color=aligned,
                           depth_min_m=adapter_cfg.depth_min_m if adapter_cfg.depth_min_m is not None
                           else lib_cfg.value("l1.depth_min_m"),
                           depth_max_m=adapter_cfg.depth_max_m if adapter_cfg.depth_max_m is not None
                           else lib_cfg.value("l1.depth_max_m"))
        self.cfg.require("map_frame_id", "body_frame_id", "body_convention", "clock_domain", "depth_aligned_to_color",
                         "depth_min_m", "depth_max_m")
        self.now_ns_fn, self.publish, self.log = now_ns_fn, publish, log or (lambda s: None)
        self.lock = threading.RLock()
        self._new_jobs: List[RebuildJob] = []
        self.pipeline = Layer5Pipeline(lib_cfg, sink=self._on_telemetry)
        self.pipeline.job_sink = self._on_job
        self.worker = RebuildWorker() if rebuild_in_thread else None
        self.intrinsics = None
        self.depths: Deque[DepthMeasurement] = deque(maxlen=int(depth_buffer_size))
        self.monitors = {k: A.StampMonitor(pose_max_backward_s if k == "pose" else 0.0)
                         for k in ("pose", "depth", "detections")}
        self.pose_revision = 0
        self.rejections: Counter = Counter()
        self.counts: Counter = Counter()
        self.last_pose_stamp_ns: Optional[int] = None
        self._telemetry: List[Any] = []
        self._last_seq = -1
        self._last_diag_ns: Optional[int] = None
        self.latest_tags: Optional[Dict[str, Any]] = None

    # --- inputs ---------------------------------------------------------------------------------------------
    def _reject(self, key: str, detail: str = "") -> str:
        self.rejections[key] += 1
        if detail:
            self.log(f"rejected {key}: {detail}")
        return key

    def on_camera_info(self, msg: Any) -> Optional[str]:
        with self.lock:
            try:
                self.intrinsics = A.intrinsics_from_camera_info(msg)
            except A.AdapterError as exc:
                return self._reject(exc.reason.value, exc.detail)
            self.counts["camera_info"] += 1
        return None

    def on_depth(self, msg: Any) -> Optional[str]:
        with self.lock:
            if self.intrinsics is None:
                return self._reject("INTRINSICS_MISSING:no_camera_info", "depth image before any CameraInfo")
            try:
                d = A.depth_measurement_from_image(msg, self.intrinsics, self.cfg, self.monitors["depth"])
            except A.AdapterError as exc:
                return self._reject(exc.reason.value, exc.detail)
            self.depths.append(d)
            self.counts["depth"] += 1
        return None

    def _match_depth(self, stamp_ns: int) -> Tuple[Optional[DepthMeasurement], str]:
        if not self.depths:
            return None, "no_depth_buffered"
        best = min(self.depths, key=lambda d: abs(d.depth_stamp_ns - stamp_ns))
        if abs(best.depth_stamp_ns - stamp_ns) > self.skew_ns:
            return None, "skew_exceeded"
        return best, ""

    def on_detections(self, msg: Any) -> Optional[str]:
        with self.lock:
            now = self.now_ns_fn()
            try:
                stamp = A.header_stamp_ns(msg.header)
            except A.AdapterError as exc:
                return self._reject(exc.reason.value, exc.detail)
            if self.intrinsics is None:
                return self._reject("INTRINSICS_MISSING:no_camera_info", "detections before any CameraInfo")
            depth, why = self._match_depth(stamp)
            if depth is None:
                return self._reject(f"{FailureReason.RGB_DEPTH_TIME_MISMATCH.value}:{why}",
                                    f"detections at {stamp} ns, {len(self.depths)} depth images buffered")
            if A._frame_id(msg.header) != depth.frame_id:
                return self._reject(FailureReason.DEPTH_FRAME_MISMATCH.value,
                                    f"detection frame {msg.header.frame_id!r}, depth frame {depth.frame_id!r}")
            try:
                h, w = depth.depth_raw.shape
                batch = A.detections_from_array(msg, self.cfg, self.monitors["detections"], (w, h))
            except A.AdapterError as exc:
                return self._reject(exc.reason.value, exc.detail)
            for _, reason, detail in batch.rejected:
                self._reject(reason.value, detail)
            if batch.detections:
                self.pipeline.process_frame(batch.detections, depth, depth.intrinsics, now)
                self.counts["frames"] += 1
                self.counts["detections"] += len(batch.detections)
            self._after_input(now)
        return None

    def on_pose(self, msg: Any) -> Optional[str]:
        with self.lock:
            now = self.now_ns_fn()
            try:
                pose = A.pose_from_msg(msg, self.cfg, self.pose_revision, "ros_pose", self.monitors["pose"])
            except A.AdapterError as exc:
                return self._reject(exc.reason.value, exc.detail)
            self.pipeline.process_pose(pose, now)
            self.last_pose_stamp_ns = max(self.last_pose_stamp_ns or 0, pose.stamp_ns)
            self.counts["poses"] += 1
            self._after_input(now)
        return None

    def on_correction(self, msg: Any) -> Optional[str]:
        with self.lock:
            now = self.now_ns_fn()
            try:
                c = A.correction_from_msg(msg, self.cfg)
            except A.AdapterError as exc:
                return self._reject(exc.reason.value, exc.detail)
            out = self.pipeline.process_correction(c, now)
            if not out.applied:
                if out.reason is not None:
                    return self._reject(out.reason.value, out.detail)
                return None  # duplicate: nothing to do
            # live poses after this event carry its revision (the counter follows the SLAM revision numbers)
            self.pose_revision = max(self.pose_revision + 1, c.revision)
            self.counts["corrections"] += 1
            self._after_input(now)
        return None

    # --- services ---------------------------------------------------------------------------------------------
    def call_set_registration(self, req: Any) -> Dict[str, Any]:
        with self.lock:
            now = self.now_ns_fn()
            try:
                reg = A.registration_from_request(req, self.pipeline.grid)
            except A.AdapterError as exc:
                self._reject(exc.reason.value, exc.detail)
                return {"accepted": False, "reason": str(exc), "current_version": self.pipeline.grid.version}
            out = self.pipeline.process_registration(reg, now)
            if out.reason is not None:
                self._reject(out.reason.value, out.detail)
            self._after_input(now)
            return A.registration_response_dict(out, self.pipeline.grid.version)

    def call_commit_final_tags(self, req: Any, wait_s: float = 30.0) -> Dict[str, Any]:
        try:
            policy, threshold = A.commit_request_to_args(req)
        except A.AdapterError as exc:
            return {"success": False, "reason": str(exc), "policy_used": "", "threshold_used": A.NAN, "tags": []}
        self.flush_rebuilds(wait_s)  # never commit a state that is about to be replaced
        with self.lock:
            now = self.now_ns_fn()
            tags = self.pipeline.final_tags(policy, threshold)
            return A.commit_response_dict(tags, policy, threshold, self.cfg.map_frame_id, now,
                                          self.pipeline.map_revision, self.pipeline.grid.version)

    # --- time and rebuilds --------------------------------------------------------------------------------------
    def tick(self, now_ns: Optional[int] = None) -> None:
        """Timer callback: advances the pipeline clock (pose-wait expiry, telemetry schedule), swaps finished rebuilds,
        publishes 1 Hz diagnostics."""
        with self.lock:
            now = self.now_ns_fn() if now_ns is None else int(now_ns)
            self.pipeline.advance(now)
            self._after_input(now)
            if self._last_diag_ns is None or now - self._last_diag_ns >= DIAG_PERIOD_NS:
                self._last_diag_ns = now
                self.publish("diagnostics", self.diagnostics(now))

    def _on_job(self, job: RebuildJob) -> None:  # called by the pipeline (lock held by the caller's handler)
        self._new_jobs.append(job)

    def _drive_rebuilds(self, now: int) -> None:
        jobs, self._new_jobs = self._new_jobs, []
        for job in jobs:
            if self.worker is not None:
                self.worker.submit(job)
            else:
                try:
                    job.run()
                except Exception as exc:  # noqa: BLE001
                    self._reject("REBUILD_FAILED", repr(exc))
                    continue
                self._swap(job, now)
        if self.worker is not None:
            for job in self.worker.finished():
                if job.error is not None:
                    self._reject("REBUILD_FAILED", repr(job.error))
                else:
                    self._swap(job, now)

    def _swap(self, job: RebuildJob, now: int) -> None:
        if self.pipeline.complete_rebuild(job):
            self.counts["rebuilds"] += 1
            self.pipeline.advance(now)  # publish the new state
        else:
            self.counts["rebuilds_discarded_stale"] += 1

    def flush_rebuilds(self, timeout_s: float = 30.0) -> bool:
        """Wait for in-flight rebuilds and swap them in. True if none is left."""
        import time as _t  # waiting only; never used for capture time
        end = _t.monotonic() + timeout_s
        while True:
            with self.lock:
                self._drive_rebuilds(self.now_ns_fn())
                self._after_input(self.now_ns_fn())
                if self.worker is None or self.worker.n_inflight <= 0:
                    return True
            if _t.monotonic() > end:
                return False
            _t.sleep(0.005)

    def finish(self) -> None:
        """End of a run (mission end, tests): swap in pending rebuilds, drop detections still waiting for a pose, publish."""
        self.flush_rebuilds()
        with self.lock:
            self.pipeline.finalize()
            self._after_input(self.now_ns_fn())

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()

    # --- outputs --------------------------------------------------------------------------------------------------
    def _on_telemetry(self, m: Any) -> None:
        self._telemetry.append(m)

    def _after_input(self, now: int) -> None:
        self._drive_rebuilds(now)
        pl = self.pipeline
        for m in self._telemetry:
            self.publish("telemetry", A.telemetry_packet_to_msg_dict(m.payload, m.stamp_ns, self.cfg.map_frame_id))
        self._telemetry = []
        if pl.publish_seq != self._last_seq:
            self._last_seq = pl.publish_seq
            published = pl.published()
            self.latest_tags = A.tag_array_to_msg_dict(published, self.cfg.map_frame_id, now, pl.map_revision,
                                                       pl.grid.version, pl.grid.valid, True)
            self.publish("tags", self.latest_tags)
            self.publish("markers", build_markers(published, pl.grid, self.cfg.map_frame_id, now))

    def diagnostics(self, now_ns: int) -> Dict[str, Any]:
        pl = self.pipeline
        lag = None if self.last_pose_stamp_ns is None else (now_ns - self.last_pose_stamp_ns) / 1e9
        last = pl.recomputes[-1] if pl.recomputes else None
        return {
            "stamp": A.ns_to_stamp(max(now_ns, 0)),
            "rejections": dict(self.rejections), "counts": dict(self.counts),
            "pose_lag_s": lag,
            "queues": {"depth_buffer": len(self.depths), "pending_detections": len(pl.l2.queue.items),
                       "pose_buffer": len(pl.l2.buffer), "stored_observations": len(pl.store),
                       "rebuilds_in_flight": 0 if self.worker is None else self.worker.n_inflight},
            "last_recompute_ms": None if last is None else last.total_ms,
            "last_recompute_cause": None if last is None else last.cause,
            "revisions": {"pose": self.pose_revision, "map": pl.map_revision, "registration": pl.grid.version},
            "registration_valid": bool(pl.grid.valid), "tags": len(pl.published()),
        }


def build_markers(published: Sequence[Any], grid: Any, frame_id: str, stamp_ns: int) -> Dict[str, Any]:
    """Marker-like dicts: a flat square on the tag's cell (only if the registration is valid) and a text label at the
    tag position, colour by status; a removed tag deletes both."""
    out: List[Dict[str, Any]] = []

    def base(pt: Any, ns: str, mid: int, typ: int, pos: Sequence[float], yaw: float, scale: Sequence[float]):
        r, g, b, a = STATUS_COLOR[pt.tag.status]
        return {"header": {"stamp": A.ns_to_stamp(max(stamp_ns, 0)), "frame_id": frame_id}, "ns": ns,
                "id": int(mid), "type": typ, "action": ACTION_DELETE if pt.removed else ACTION_ADD,
                "pose": {"position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
                         "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(yaw / 2), "w": math.cos(yaw / 2)}},
                "scale": {"x": float(scale[0]), "y": float(scale[1]), "z": float(scale[2])},
                "color": {"r": r, "g": g, "b": b, "a": a}, "text": "", "lifetime": {"sec": 0, "nanosec": 0}}

    for pt in published:
        t = pt.tag
        if t.cell is not None and grid.valid:
            c = grid.cell_size_m
            gx, gy = (t.cell[0] - grid.index_base + 0.5) * c, (t.cell[1] - grid.index_base + 0.5) * c
            m = grid.grid_to_map([gx, gy])
            out.append(base(pt, "survivor_cells", 2 * pt.survivor_id, MARKER_CUBE, (m[0], m[1], MARKER_Z_M),
                            grid.yaw_rad, (c, c, 0.02)))
        if t.p_map_m is not None:
            lab = t.cell_label or (f"{t.cell[0]},{t.cell[1]}" if t.cell is not None else "?")
            m = base(pt, "survivor_labels", 2 * pt.survivor_id + 1, MARKER_TEXT,
                     (t.p_map_m[0], t.p_map_m[1], float(t.p_map_m[2]) + 0.5), 0.0, (0.0, 0.0, 0.3))
            m["text"] = f"#{pt.survivor_id} {lab} {t.status.name} p={t.localisation_quality:.2f}"
            out.append(m)
    return {"markers": out}


# ---------------------------------------------------------------------------
# Construction from node parameters (kept here so it is testable without ROS)
# ---------------------------------------------------------------------------


def _tri_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    return None  # "", "UNSET", None: unset (never guessed)


def build_library_config(params: Dict[str, Any]) -> Config:
    """Library Config = defaults + the frozen file's values (sources other than 'unset') + overrides_file (flat library
    override JSON, e.g. extr.*, clock.*, l1.depth_*, l2.*, grid.*, l5.correction_source) + node parameters."""
    cfg = default_config()
    frozen = params.get("config_file") or ""
    if frozen:
        doc = json.loads(Path(frozen).read_text(encoding="utf-8"))
        vals = {k: e["value"] for k, e in doc["parameters"].items() if e.get("value") is not None}
        cfg.apply_overrides(vals, reference=f"frozen: {Path(frozen).name}")
    if params.get("overrides_file"):
        load_config(params["overrides_file"], base=cfg)
    t = params.get("T_body_camera")
    if t is not None and not isinstance(t, str) and len(t) in (4, 16):
        cfg.apply_overrides({"extr.T_body_camera": np.asarray(t, dtype=float).reshape(4, 4).tolist()},
                            reference="node parameter T_body_camera")
    al = _tri_bool(params.get("depth_aligned_to_color"))
    if al is not None:
        cfg.apply_overrides({"cam.detections_use_depth_aligned_frame": al}, reference="node parameter")
    return cfg


def build_adapter_config(params: Dict[str, Any], lib_cfg: Config) -> A.AdapterConfig:
    def s(key: str) -> Optional[str]:
        v = params.get(key)
        return None if v in (None, "", "UNSET") else str(v)

    kw: Dict[str, Any] = dict(map_frame_id=s("pose_frame"), body_frame_id=s("body_frame"),
                              body_convention=s("body_frame_convention"), clock_domain=s("clock_domain"),
                              depth_aligned_to_color=_tri_bool(params.get("depth_aligned_to_color")))
    for src, dst in (("pose.default_position_sigma_m", "pose_default_position_sigma_m"),
                     ("pose.default_yaw_sigma_rad", "pose_default_yaw_sigma_rad")):
        if src in params:
            kw[dst] = None if params[src] in (None, "", "UNSET") else float(params[src])
    return A.AdapterConfig(**kw)


def build_runtime(params: Dict[str, Any], now_ns_fn: Callable[[], int], publish: PublishFn,
                  log: Optional[Callable[[str], None]] = None) -> TaggerRuntime:
    """Runtime from the flat parameter dict of the node. Missing required values raise ConfigurationError."""
    try:
        lib_cfg = build_library_config(params)
    except ConfigError as exc:
        raise A.ConfigurationError([str(exc)]) from None
    return TaggerRuntime(build_adapter_config(params, lib_cfg), lib_cfg, now_ns_fn, publish,
                         depth_buffer_size=int(params.get("depth_buffer_size", 8)),
                         rebuild_in_thread=bool(params.get("rebuild_in_thread", True)),
                         pose_max_backward_s=float(params.get("pose_max_backward_s", 0.0)), log=log)
