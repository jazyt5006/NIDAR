import numpy as np
import pytest

from survivor_tagging.config import default_config
from survivor_tagging.pipeline import Layer123Pipeline
from survivor_tagging.types import (CameraIntrinsics, DepthMeasurement, Detection, FailureReason, PoseSample,
                                    TagStatus)

S = 1_000_000_000
INTR = CameraIntrinsics(100.0, 100.0, 49.5, 39.5, 100, 80, "none")
R_BC = [[0.0, 0.0, 1.0, 0.1], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
GRID = {"grid.origin_in_map_m": [0.0, 0.0], "grid.yaw_map_to_grid_rad": 0.0, "grid.axis_flip_x": False,
        "grid.axis_flip_y": False, "grid.index_base": 0, "grid.numbering": "ij", "grid.n_cells_x": 6,
        "grid.n_cells_y": 6, "grid.registration_valid": True}


def _cfg(grid=True, **extra):
    cfg = default_config()
    vals = {"l1.depth_min_m": 0.1, "l1.depth_max_m": 10.0, "l1.depth_noise_model": [0.0, 0.0, 0.002],
            "extr.T_body_camera": R_BC, "extr.T_body_camera_cov": np.zeros((6, 6)).tolist(), "extr.version": 1,
            "clock.camera_to_pose_offset_ns": 0, "clock.offset_uncertainty_ns": 0, "l2.max_interp_gap_s": 0.1,
            "l2.max_rgb_depth_skew_s": 0.005, "l2.pose_mode": "full_3d"}
    if grid:
        vals.update(GRID)
    vals.update(extra)
    cfg.apply_overrides(vals, "test")
    return cfg


def _pose(t_s, x=0.5):
    return PoseSample(int(round(t_s * S)), "ros", "map", "body", (x, 2.5, 1.0), (1, 0, 0, 0), True, 0, "t")


def _frame(t_s):
    det = Detection(0, 0.9, (20.0, 10.0, 80.0, 70.0), int(t_s * S), "ros", "cam", (100, 80), 7)
    depth = DepthMeasurement(int(t_s * S), "ros", "cam", np.full((80, 100), 2000, dtype=np.uint16), 0.001, INTR,
                             True, (0.1, 10.0))
    return [det], depth


def test_observation_waits_for_late_pose_then_resolves():
    p = Layer123Pipeline(_cfg())
    p.process_pose(_pose(0.98), int(0.99 * S))
    dets, depth = _frame(1.0)
    p.process_frame(dets, depth, INTR, int(1.05 * S))
    assert len(p.l2.queue) == 1 and p.tags == {}
    p.process_pose(_pose(1.02, 0.7), int(1.10 * S))  # pose after the capture time arrives late
    (obs,) = p.observations()
    assert obs.failure is None and obs.interp_alpha == pytest.approx(0.5)
    assert obs.p_map_m == pytest.approx([0.6 + 2.1, 2.5 - 0.01, 1.0 - 0.01])  # body at x 0.6, 2 m + 0.1 m ahead
    assert obs.cell == (2, 2) and sum(obs.cell_probs.values()) <= 1.0 + 1e-9
    (tag,) = p.finalize()
    assert tag.status is TagStatus.CONFIRMED and tag.cell == (2, 2)


def test_observation_times_out_without_pose():
    p = Layer123Pipeline(_cfg(**{"l2.max_pose_wait_s": 0.2}))
    p.process_pose(_pose(0.98), int(0.99 * S))
    dets, depth = _frame(1.0)
    p.process_frame(dets, depth, INTR, int(1.05 * S))
    p.process_pose(_pose(0.99), int(1.30 * S))  # still older than the capture time; now past the deadline
    (obs,) = p.observations()
    assert obs.failure is FailureReason.POSE_UNAVAILABLE_AT_CAPTURE and obs.p_map_m is None
    assert p.finalize() == []


def test_finalize_fails_pending_explicitly():
    p = Layer123Pipeline(_cfg())
    dets, depth = _frame(1.0)
    p.process_frame(dets, depth, INTR, int(1.05 * S))  # no pose at all yet
    assert p.finalize() == [] and p.observations()[0].failure is FailureReason.POSE_UNAVAILABLE_AT_CAPTURE


def test_unregistered_grid_gives_provisional_tag():
    p = Layer123Pipeline(_cfg(grid=False))
    p.process_pose(_pose(0.98), 0)
    p.process_pose(_pose(1.02), 0)
    dets, depth = _frame(1.0)
    p.process_frame(dets, depth, INTR, int(1.05 * S))
    (tag,) = p.finalize()
    assert tag.status is TagStatus.PROVISIONAL and tag.confirm_stamp_ns is None
    assert FailureReason.GRID_UNREGISTERED in p.observations()[0].warnings


def test_missing_skew_tolerance_reports_config_missing():
    cfg = _cfg()
    cfg["l2.max_rgb_depth_skew_s"].value = None
    p = Layer123Pipeline(cfg)
    p.process_pose(_pose(0.98), 0)
    dets, depth = _frame(1.0)
    p.process_frame(dets, depth, INTR, int(1.05 * S))
    assert p.observations()[0].failure is FailureReason.CONFIG_MISSING


def test_library_layers_do_not_import_simulator_or_baseline():
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "survivor_tagging"
    banned = {"sim", "scenarios", "baseline", "harness", "sim_pipelines", "experiments", "diagnostics",
              "experiments_calib", "experiments_layer5"}
    for name in ("types", "config", "geometry", "transforms", "grid", "pipeline", "output", "tracking", "footprint",
                 "calibration", "corrections"):
        tree = ast.parse((src / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                assert (node.module or "").split(".")[0] not in banned, (name, node.module)
