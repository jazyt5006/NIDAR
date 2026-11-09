import math

import numpy as np
import pytest

from survivor_tagging.config import ConfigError, default_config
from survivor_tagging.grid import GridRegistration, apply_grid, cell_probabilities, fit_registration_from_points
from survivor_tagging.types import FAILURE_SEVERITY, FailureReason, Severity, SurvivorObservation

# conventions.md worked example: origin (2, 1) in the map, grid +x along map +y (yaw +90 deg)
REG = GridRegistration((2.0, 1.0), math.pi / 2, n_cells_x=6, n_cells_y=6, valid=True)


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def test_worked_example_offset_and_rotation():
    assert REG.map_to_grid([2.0, 3.0]) == pytest.approx([2.0, 0.0])
    assert REG.map_to_grid([1.0, 1.0]) == pytest.approx([0.0, 1.0])
    assert REG.map_to_grid([2.5, 1.5]) == pytest.approx([0.5, -0.5])


def test_axis_flips():
    fy = GridRegistration((2.0, 1.0), math.pi / 2, flip_y=True, valid=True)
    assert fy.map_to_grid([1.0, 1.0]) == pytest.approx([0.0, -1.0])
    fx = GridRegistration((0.0, 0.0), 0.0, flip_x=True, valid=True)
    assert fx.map_to_grid([1.5, 2.0]) == pytest.approx([-1.5, 2.0])


def test_round_trips():
    rng = np.random.default_rng(0)
    for flip_x, flip_y in ((False, False), (True, False), (False, True), (True, True)):
        reg = GridRegistration((1.3, -0.7), 0.4, flip_x, flip_y, valid=True)
        pts = rng.uniform(-5, 5, size=(20, 2))
        assert reg.grid_to_map(reg.map_to_grid(pts)) == pytest.approx(pts)
        assert reg.map_to_grid(reg.grid_to_map(pts)) == pytest.approx(pts)


def test_cells_index_base_edges_negative_and_out_of_grid():
    assert REG.cell_index([2.0, 0.0]) == (2, 0)
    assert GridRegistration((0, 0), 0, index_base=1, valid=True).cell_index([2.0, 0.0]) == (3, 1)
    assert REG.cell_index([3.0, 2.9999]) == (3, 2)  # exactly on an edge: floor -> upper/right cell
    assert REG.cell_index([-0.2, -1.0]) == (-1, -1)
    assert REG.is_out_of_grid([-0.2, 1.0]) is True
    assert REG.is_out_of_grid([6.0, 1.0]) is True
    assert REG.is_out_of_grid([5.99, 5.99]) is False
    assert GridRegistration((0, 0), 0).is_out_of_grid([100.0, 0.0]) is None  # extent unknown


def test_labels():
    assert REG.cell_label((3, 4)) == "3,4"
    rm = GridRegistration((0, 0), 0, n_cells_x=6, numbering="row_major_label", valid=True)
    assert rm.cell_label((2, 1)) == "8"
    rm1 = GridRegistration((0, 0), 0, n_cells_x=6, index_base=1, numbering="row_major_label", valid=True)
    assert rm1.cell_label((3, 2)) == "9"
    with pytest.raises(ConfigError, match="O1"):
        GridRegistration((0, 0), 0, numbering="alnum").cell_label((1, 1))


def test_boundary_distance():
    assert REG.boundary_distance_m([3.1, 2.5]) == pytest.approx(0.1)
    assert REG.boundary_distance_m([2.5, 2.5]) == pytest.approx(0.5)
    assert REG.boundary_distance_m([-0.3, 2.45]) == pytest.approx(0.3)


def test_probabilities_diagonal_analytic():
    reg = GridRegistration((0, 0), 0, n_cells_x=6, n_cells_y=6, valid=True)
    s = 0.3
    p = cell_probabilities(reg, [2.5, 2.5], np.diag([s * s, s * s]), max_cells=9).as_dict()
    assert p[(2, 2)] == pytest.approx((phi(0.5 / s) - phi(-0.5 / s)) ** 2, abs=1e-12)
    assert p[(3, 2)] == pytest.approx((phi(1.5 / s) - phi(0.5 / s)) * (phi(0.5 / s) - phi(-0.5 / s)), abs=1e-12)
    edge = cell_probabilities(reg, [3.0, 2.5], np.diag([0.04, 0.04]))
    d = edge.as_dict()
    assert d[(2, 2)] == pytest.approx(d[(3, 2)], abs=1e-12)
    out = cell_probabilities(reg, [0.05, 2.5], np.diag([0.04, 0.0001]))
    assert out.out_of_grid == pytest.approx(phi(-0.05 / 0.2), abs=1e-6)
    assert sum(pp for _, pp in out.cells) + out.out_of_grid <= 1.0 + 1e-12


def test_probabilities_rotated_registration_and_index_base():
    reg = GridRegistration((2.0, 1.0), math.pi / 2, index_base=1, n_cells_x=6, n_cells_y=6, valid=True)
    # map covariance elongated along map y = grid x after the 90 degree registration
    p = cell_probabilities(reg, [2.0, 3.5], np.diag([0.01, 0.25])).as_dict()  # grid (2.5, 0)
    assert p[(3, 1)] == pytest.approx((phi(1.0) - phi(-1.0)) * (phi(10.0) - phi(0.0)), abs=1e-9)


def test_probabilities_correlated_vs_monte_carlo():
    reg = GridRegistration((0, 0), 0, n_cells_x=6, n_cells_y=6, valid=True)
    mean = np.array([2.9, 3.1])
    cov = np.array([[0.09, 0.06], [0.06, 0.09]])
    p = cell_probabilities(reg, mean, cov, max_cells=9).as_dict()
    rng = np.random.default_rng(7)
    x = rng.multivariate_normal(mean, cov, size=400_000)
    cells, counts = np.unique(np.floor(x).astype(int), axis=0, return_counts=True)
    mc = {tuple(c): n / len(x) for c, n in zip(cells, counts)}
    for cell in ((2, 3), (3, 3), (2, 2), (3, 2)):
        assert p[cell] == pytest.approx(mc[cell], abs=0.003)
    assert sum(p.values()) <= 1.0 + 1e-9


def test_fit_registration_recovers_known_transform():
    truth = GridRegistration((1.3, -0.7), 0.4, flip_y=True, n_cells_x=6, n_cells_y=6, version=2, valid=True)
    rng = np.random.default_rng(3)
    m = rng.uniform(-3, 6, size=(12, 2))
    g = truth.map_to_grid(m) + rng.normal(0, 0.01, size=(12, 2))
    template = GridRegistration((0, 0), 0, flip_y=True, n_cells_x=6, n_cells_y=6, version=2, valid=True)
    fit, rms = fit_registration_from_points(m, g, template)
    assert fit.yaw_rad == pytest.approx(0.4, abs=0.01)
    assert fit.origin_in_map_m == pytest.approx((1.3, -0.7), abs=0.02)
    assert 0.003 < rms < 0.02 and fit.version == 3


def test_versioning():
    r2 = REG.updated(yaw_rad=0.1)
    assert (r2.version, REG.version, r2.yaw_rad, REG.yaw_rad) == (1, 0, 0.1, math.pi / 2)


def test_unregistered_grid_blocks_confirmation():
    reg = GridRegistration.from_config(default_config())
    assert not reg.valid and "grid.origin_in_map_m" in reg.missing
    obs = SurvivorObservation(0, 0, 0.9, 0, p_map_m=np.array([1.5, 2.5, 0.0]), cov_map_m2=np.eye(3) * 0.01)
    apply_grid(obs, reg)
    assert obs.cell == (1, 2)  # provisional, diagnostics only
    assert FailureReason.GRID_UNREGISTERED in obs.warnings
    assert FAILURE_SEVERITY[FailureReason.GRID_UNREGISTERED] is Severity.BLOCK
    cfg = default_config()
    cfg.apply_overrides({"grid.origin_in_map_m": [0.0, 0.0], "grid.yaw_map_to_grid_rad": 0.0, "grid.axis_flip_x": False,
                         "grid.axis_flip_y": False, "grid.index_base": 0, "grid.numbering": "ij",
                         "grid.registration_valid": False}, "test")
    assert not GridRegistration.from_config(cfg).valid  # all set but declared not valid


def test_apply_grid_fills_fields_and_out_of_grid():
    obs = SurvivorObservation(0, 0, 0.9, 0, p_map_m=np.array([2.0, 0.5, 0.0]), cov_map_m2=np.eye(3) * 0.01)
    apply_grid(obs, REG)  # grid (-0.5, -1.0): outside
    assert FailureReason.OUT_OF_GRID in obs.warnings and FailureReason.GRID_UNREGISTERED not in obs.warnings
    assert obs.registration_version == 0 and obs.boundary_distance_m == pytest.approx(0.0)


def test_layer3_agrees_with_simulator_registration():
    """Independence: the library's map->grid (from config built via the simulator's PUBLIC map_from_grid) must equal
    the simulator's own grid_from_map for several offsets and yaws."""
    from survivor_tagging.config import default_config as dc
    from survivor_tagging.sim import Registration
    from survivor_tagging.sim_pipelines import registration_overrides

    rng = np.random.default_rng(11)
    for offset, yaw in (((0, 0), 0.0), ((1.2, 0.8), 0.25), ((-3.0, 2.0), -1.3), ((0.4, -0.6), 3.0)):
        sim_reg = Registration(offset, yaw)
        cfg = dc()
        cfg.apply_overrides(registration_overrides(sim_reg, 6, 6), "test")
        lib = GridRegistration.from_config(cfg)
        assert lib.valid
        for p in rng.uniform(-6, 6, size=(10, 2)):
            assert lib.map_to_grid(p) == pytest.approx(sim_reg.grid_from_map(p), abs=1e-9)
            assert lib.cell_index(lib.map_to_grid(p)) == sim_reg.cell_of_grid(sim_reg.grid_from_map(p))
