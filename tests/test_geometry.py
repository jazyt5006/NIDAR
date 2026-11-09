import numpy as np
import pytest

from survivor_tagging.config import ConfigError, default_config
from survivor_tagging.geometry import (
    Layer1Params,
    apply_body_offset,
    back_project,
    cluster_sorted_depths,
    dense_mask,
    edge_span,
    estimate_survivor_point,
    inset_box_region,
    strip_fraction,
)
from survivor_tagging.types import CameraIntrinsics, DepthMeasurement, Detection, FailureReason

INTR = CameraIntrinsics(100.0, 100.0, 49.5, 39.5, 100, 80, "none")
BOX = (20.0, 10.0, 80.0, 70.0)  # inset 0.15 -> u 29..71, v 19..61


def _params(overrides=None) -> Layer1Params:
    cfg = default_config()
    vals = {"l1.depth_min_m": 0.1, "l1.depth_max_m": 10.0, "l1.depth_noise_model": [0.0, 0.0, 0.002]}
    vals.update(overrides or {})
    cfg.apply_overrides(vals, reference="test")
    return Layer1Params.from_config(cfg)


P = _params()


def _depth(arr, scale=0.001, frame="cam", clock="sim", intr=INTR):
    return DepthMeasurement(0, clock, frame, arr, scale, intr, True, (0.1, 10.0))


def _det(bbox=BOX, frame="cam", clock="sim", size=(100, 80)):
    return Detection(0, 0.9, bbox, 0, clock, frame, size)


def _img(mm=2000):
    return np.full((80, 100), mm, dtype=np.uint16)


def _run(img, bbox=BOX, params=P, **kw):
    return estimate_survivor_point(_det(bbox), _depth(img, **kw), INTR, params)


# --- back-projection -------------------------------------------------------

@pytest.mark.parametrize("fx, fy, cx, cy, u, v, z, expected", [
    (100, 100, 49.5, 39.5, 49.5, 39.5, 2.0, (0.0, 0.0, 2.0)),
    (100, 100, 49.5, 39.5, 149.5, 39.5, 2.0, (2.0, 0.0, 2.0)),
    (100, 100, 49.5, 39.5, 49.5, 89.5, 3.0, (0.0, 1.5, 3.0)),
    (600, 550, 320, 240, 20, 400, 1.5, (-0.75, 160 * 1.5 / 550, 1.5)),
    (212, 212, 211.5, 119.5, 0, 0, 4.0, (-211.5 * 4 / 212, -119.5 * 4 / 212, 4.0)),
])
def test_back_projection(fx, fy, cx, cy, u, v, z, expected):
    intr = CameraIntrinsics(fx, fy, cx, cy, 640, 480, "none")
    assert back_project(u, v, z, intr) == pytest.approx(expected)


# --- validation ------------------------------------------------------------

@pytest.mark.parametrize("bbox, reason", [
    ((80.0, 10.0, 20.0, 70.0), FailureReason.BBOX_INVALID),
    ((20.0, 10.0, 20.0 + 5, 70.0), FailureReason.BBOX_INVALID),  # narrower than min_bbox_size_px
    ((-5.0, 10.0, 40.0, 70.0), FailureReason.BBOX_OUTSIDE_IMAGE),
    ((20.0, 10.0, 120.0, 70.0), FailureReason.BBOX_OUTSIDE_IMAGE),
])
def test_bbox_validation(bbox, reason):
    assert _run(_img(), bbox=bbox).failure is reason


def test_frame_clock_scale_and_intrinsics_checks():
    img = _img()
    assert estimate_survivor_point(_det(frame="other"), _depth(img), INTR, P).failure is \
        FailureReason.DEPTH_FRAME_MISMATCH
    assert estimate_survivor_point(_det(size=(640, 480)), _depth(img), INTR, P).failure is \
        FailureReason.DEPTH_FRAME_MISMATCH
    assert estimate_survivor_point(_det(clock="other"), _depth(img), INTR, P).failure is \
        FailureReason.CLOCK_DOMAIN_UNKNOWN
    assert _run(img, scale=0.0).failure is FailureReason.DEPTH_SCALE_INVALID
    bad = CameraIntrinsics(100.0, 100.0, 49.5, 39.5, 640, 480, "none")
    assert estimate_survivor_point(_det(), _depth(img, intr=bad), bad, P).failure is FailureReason.INTRINSICS_MISSING


# --- invalid depth ---------------------------------------------------------

def test_all_zero_depth_is_no_valid():
    r = _run(_img(0))
    assert r.failure is FailureReason.DEPTH_NO_VALID and r.p_camera_m is None and r.n_valid_px == 0


def test_out_of_range_depth_is_no_valid():
    assert _run(_img(20000)).failure is FailureReason.DEPTH_NO_VALID  # 20 m > depth_max_m
    assert _run(_img(50)).failure is FailureReason.DEPTH_NO_VALID  # 0.05 m < depth_min_m


def test_float_depth_nan_and_inf_rejected():
    img = np.full((80, 100), np.nan)
    img[::7, ::7] = np.inf
    assert _run(img, scale=1.0).failure is FailureReason.DEPTH_NO_VALID
    img[30:34, 40:44] = 2.0  # a few valid pixels only
    r = _run(img, scale=1.0)
    assert r.failure is FailureReason.DEPTH_TOO_FEW_PIXELS and r.n_valid_px > 0


def test_too_few_pixels():
    img = _img(0)
    img[40:46, 40:46] = 2000
    r = _run(img)
    assert r.failure is FailureReason.DEPTH_TOO_FEW_PIXELS
    assert r.p_camera_m is None


# --- clustering ------------------------------------------------------------

def test_person_in_front_of_wall_chooses_person():
    img = _img(3000)  # wall at 3 m
    img[:, 35:66] = 2000  # person at 2 m covers ~70 % of the inset columns
    r = _run(img)
    assert r.failure is None
    assert r.depth_m == pytest.approx(2.0)
    assert r.n_clusters == 2


def test_two_equal_clusters_are_ambiguous():
    img = _img(3000)
    img[:, :50] = 2000  # inset columns 29..71 split evenly
    r = _run(img)
    assert r.failure is FailureReason.DEPTH_AMBIGUOUS_CLUSTERS and r.p_camera_m is None


def test_thin_person_in_front_of_wall_is_not_taken_as_background():
    img = _img(3000)
    img[:, 47:52] = 2000  # thin person strip, ~14 % of the sampled pixels, wall on both sides
    r = _run(img)
    assert r.failure is None and r.depth_m == pytest.approx(2.0)
    # The 2.3 rule could not tell this apart and rejected it.
    old = _run(img, params=_params({"l1.cluster_rule": "fraction_only"}))
    assert old.failure is FailureReason.DEPTH_BACKGROUND_DOMINANT


def test_unexplained_near_blob_with_dominant_wall_is_background_dominant():
    img = _img(3000)
    img[:28, :42] = 2000  # nearer blob in the top-left corner: neither an edge band nor a strip
    r = _run(img)
    assert r.failure is FailureReason.DEPTH_BACKGROUND_DOMINANT and r.p_camera_m is None


def test_low_occluder_band_accepts_person_behind():
    img = _img(2500)  # person at 2.5 m
    img[52:, :] = 1500  # low wall at 1.5 m: band across the bottom of the box, ~20 % of the sampled pixels
    r = _run(img)
    assert r.failure is None and r.depth_m == pytest.approx(2.5)
    old = _run(img, params=_params({"l1.cluster_rule": "fraction_only"}))
    assert old.failure is FailureReason.DEPTH_BACKGROUND_DOMINANT  # the 2.3 behaviour


def test_side_occluder_band_accepts_person_behind():
    img = _img(2500)
    img[:, :34] = 1500  # door frame / wall edge along the left side of the box
    r = _run(img)
    assert r.failure is None and r.depth_m == pytest.approx(2.5)


def test_density_clustering_removes_sparse_floor_continuum():
    z = np.sort(np.concatenate([np.full(100, 2.0), np.linspace(2.1, 2.9, 9), np.full(100, 3.0)]))
    assert len(cluster_sorted_depths(z, 0.3)) == 1  # the ramp bridges the gap
    keep = dense_mask(z, 0.3, 0.05 * z.size)
    # interior ramp samples 2.2..2.8 are sparse; 2.1 and 2.9 sit within 0.15 m of the dense surfaces
    assert np.allclose(z[~keep], np.linspace(2.2, 2.8, 7))
    assert len(cluster_sorted_depths(z[keep], 0.3)) == 2
    assert dense_mask(z, 0.3, 0).all()


def test_edge_span_and_strip_fraction():
    valid = np.ones((10, 10), dtype=bool)
    band = np.zeros_like(valid)
    band[8:, :] = True
    assert edge_span(band, valid) == 1.0
    strip = np.zeros_like(valid)
    strip[:, 4:6] = True
    assert strip_fraction(strip, ~strip) == 1.0
    assert edge_span(strip, valid) == pytest.approx(0.2)
    assert strip_fraction(band, ~band) == 0.0


def test_single_far_surface_is_accepted():
    r = _run(_img(3000))
    assert r.failure is None and r.depth_m == pytest.approx(3.0)


def test_fifty_percent_dropout_still_works():
    rng = np.random.default_rng(0)
    img = _img(2000)
    img[rng.random(img.shape) < 0.5] = 0
    r = _run(img)
    assert r.failure is None
    assert r.depth_m == pytest.approx(2.0)
    assert 0.4 < r.valid_fraction < 0.6


def test_noisy_depth_median_is_robust():
    rng = np.random.default_rng(1)
    img = (2000 + rng.normal(0, 10, size=(80, 100))).astype(np.uint16)
    r = _run(img)
    assert r.failure is None and abs(r.depth_m - 2.0) < 0.005


# --- point, covariance, offset, ground reference ---------------------------

def test_point_uses_cluster_pixel_median():
    r = _run(_img(2000))
    u, v = r.pixel_uv
    # sampled columns 29, 31, ..., 71 (22 values) -> median (49 + 51) / 2 = 50; rows 19..61 -> 40
    assert (u, v) == (50.0, 40.0)
    assert r.p_camera_m == pytest.approx(back_project(u, v, 2.0, INTR))


def test_covariance_grows_with_depth_like_noise_model():
    r2, r4 = _run(_img(2000)), _run(_img(4000))
    assert r2.cov_camera_m2[2, 2] > 0
    # flat surface: sigma_z = 0.002 z^2 -> variance ratio (4/2)^4 = 16
    assert r4.cov_camera_m2[2, 2] / r2.cov_camera_m2[2, 2] == pytest.approx(16.0)
    assert r4.cov_camera_m2[0, 0] > r2.cov_camera_m2[0, 0]
    assert np.allclose(r2.cov_camera_m2, r2.cov_camera_m2.T)
    assert 0.0 <= r2.depth_quality <= 1.0


def test_body_offset_moves_point_along_ray():
    base = _run(_img(2000)).p_camera_m
    moved = _run(_img(2000), params=_params({"l1.body_depth_offset_m": 0.2})).p_camera_m
    assert np.linalg.norm(moved - base) == pytest.approx(0.2)
    assert np.cross(moved, base) == pytest.approx(np.zeros(3), abs=1e-12)
    assert np.linalg.norm(moved) > np.linalg.norm(base)
    p = np.array([1.0, 2.0, 2.0])
    assert apply_body_offset(p, 0.3) == pytest.approx(p * (3.3 / 3.0))


def test_ground_reference_none_with_warning():
    r = _run(_img(2000))
    assert r.ground_ref_camera_m is None
    assert r.warnings == [FailureReason.GROUND_REF_UNAVAILABLE]


def test_region_is_strided_view():
    img = _img(2000)
    reg = inset_box_region(BOX, img, P)
    assert np.shares_memory(reg.raw, img)
    assert reg.stride == 2 and (reg.u0, reg.v0) == (29, 19)
    uu, vv = reg.pixel_coords()
    assert uu[0, 1] - uu[0, 0] == 2 and vv[1, 0] - vv[0, 0] == 2 and uu.max() <= 71 and vv.max() <= 61


def test_params_require_l1_config():
    with pytest.raises(ConfigError, match="CONFIG_MISSING"):
        Layer1Params.from_config(default_config())


# --- cluster centroid, truncation ----------------------------------------------

CENTROID = {"l1.point_method": "cluster_centroid"}


def test_centroid_of_flat_patch():
    r = _run(_img(2000), params=_params(CENTROID))
    # sampled columns 29..71, rows 19..61 (step 2); per-axis median of X = (u - cx) Z / fx etc.
    assert r.p_camera_m == pytest.approx([(50 - 49.5) * 2 / 100, (40 - 39.5) * 2 / 100, 2.0])


def test_centroid_of_tilted_prone_like_patch():
    intr = CameraIntrinsics(100.0, 100.0, 50.0, 30.0, 100, 80, "none")
    rows = np.arange(80, dtype=float)[:, None]
    img = np.repeat(2.0 + 0.02 * rows, 100, axis=1)  # depth grows by 2 cm per image row (a surface seen obliquely)
    det = _det()
    r = estimate_survivor_point(det, _depth(img, scale=1.0, intr=intr), intr, _params(CENTROID))
    assert r.failure is None
    v_mid = (39, 41)  # middle of the 22 sampled rows 19..61
    y = np.mean([(v - 30.0) * (2.0 + 0.02 * v) / 100 for v in v_mid])
    z = np.mean([2.0 + 0.02 * v for v in v_mid])
    assert r.p_camera_m == pytest.approx([0.0, y, z])  # columns 29..71 are symmetric about cx = 50


def test_centroid_covariance_and_offset():
    r = _run(_img(2000), params=_params(CENTROID))
    c = r.cov_camera_m2
    assert np.allclose(c, c.T) and np.all(np.linalg.eigvalsh(c) >= -1e-12)
    moved = _run(_img(2000), params=_params({**CENTROID, "l1.body_depth_offset_m": 0.2})).p_camera_m
    assert np.linalg.norm(moved - r.p_camera_m) == pytest.approx(0.2)


def test_bbox_truncated_warning_enlarges_covariance():
    box = (0.0, 10.0, 60.0, 70.0)
    r1 = _run(_img(2000), bbox=box, params=_params({"l1.truncated_cov_scale": 1.0}))
    r4 = _run(_img(2000), bbox=box)
    assert FailureReason.BBOX_TRUNCATED in r4.warnings and r4.failure is None
    assert r4.cov_camera_m2 == pytest.approx(4.0 * r1.cov_camera_m2)
    assert FailureReason.BBOX_TRUNCATED not in _run(_img(2000)).warnings
    from survivor_tagging.types import FAILURE_SEVERITY, Severity
    assert FAILURE_SEVERITY[FailureReason.BBOX_TRUNCATED] is Severity.DEGRADE
