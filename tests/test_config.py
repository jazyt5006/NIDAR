import json
from pathlib import Path

import pytest

from survivor_tagging.config import STAGES, ConfigError, ParamSource, default_config, load_config
from survivor_tagging.types import FailureReason

EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "example_synthetic.json"


def test_required_parameters_start_unset_unless_from_rulebook():
    cfg = default_config()
    required = [p for p in cfg.params() if p.required and p.source is not ParamSource.RULEBOOK]
    assert required
    for p in required:
        assert p.value is None, p.name
        assert p.source is ParamSource.NEEDS_MEASUREMENT, p.name


def test_defaults_only_on_assumption_or_rulebook_parameters():
    for p in default_config().params():
        if p.value is not None:
            assert p.source in (ParamSource.ASSUMPTION, ParamSource.RULEBOOK), p.name
            if p.source is ParamSource.ASSUMPTION:
                assert not p.required, p.name


def test_cell_size_comes_from_rulebook():
    cfg = default_config()
    p = cfg["grid.cell_size_m"]
    assert (p.value, p.source, p.reference, p.required) == \
           (1.0, ParamSource.RULEBOOK, "NIDAR 26-27 rulebook p.19", True)
    rulebook = [q.name for q in cfg.params() if q.source is ParamSource.RULEBOOK]
    # every other grid parameter stays unset; the only other rulebook value is the 40 points per correct tag (p.19)
    assert rulebook == ["grid.cell_size_m", "score.correct_tag_points"]
    assert cfg.value("score.correct_tag_points") == 40.0
    assert cfg.value("score.wrong_tag_penalty_assumed") is None  # O2 unknown


def test_body_depth_offset_param():
    p = default_config()["l1.body_depth_offset_m"]
    assert (p.value, p.unit, p.source, p.valid_range) == (0.0, "m", ParamSource.ASSUMPTION, (0.0, 0.5))


def test_missing_required_lists_unset_params():
    cfg = default_config()
    l3 = cfg.missing_required("l3")
    assert "grid.cell_size_m" not in l3 and "grid.origin_in_map_m" in l3
    assert "extr.T_body_camera" in cfg.missing_required("l2")  # dependency
    assert "extr.T_body_camera" not in cfg.missing_required("l2", include_dependencies=False)
    assert cfg.missing_required("l4") == []  # tuning only
    for stage in ("camera", "extrinsics", "clock", "l1", "l2", "l3", "l5"):
        assert not cfg.is_stage_ready(stage)
        assert cfg.stage_failure(stage) is FailureReason.CONFIG_MISSING
    with pytest.raises(ConfigError):
        cfg.missing_required("nope")


def test_file_intrinsics_are_conditionally_required():
    cfg = default_config()
    assert "cam.fx_px" not in cfg.missing_required("camera")
    cfg.set("cam.intrinsics_source", "file")
    assert "cam.fx_px" in cfg.missing_required("camera")
    cfg.set("cam.intrinsics_source", "device")
    assert "cam.fx_px" not in cfg.missing_required("camera")


def test_example_json_makes_all_stages_ready():
    cfg = load_config(EXAMPLE)
    for stage in STAGES:
        assert cfg.is_stage_ready(stage), (stage, cfg.missing_required(stage))
        assert cfg.stage_failure(stage) is None
    assert cfg.value("grid.cell_size_m") == 1.0
    assert cfg["grid.cell_size_m"].reference == "override: example_synthetic.json"
    assert "TEST VALUES ONLY, NOT HARDWARE" in json.loads(EXAMPLE.read_text(encoding="utf-8"))["_note"]


def _write(tmp_path, data):
    path = tmp_path / "o.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("data, match", [
    ({"l1.no_such_param": 1}, "unknown parameter"),
    ({"_comment": "x"}, "unknown parameter"),
    ({"l1.bbox_inset_fraction": 0.9}, "outside valid range"),
    ({"l1.body_depth_offset_m": 0.6}, "outside valid range"),
    ({"l1.min_valid_pixels": 1.5}, "expected an integer"),
    ({"l1.min_valid_pixels": True}, "expected an integer"),
    ({"grid.axis_flip_x": 1}, "expected true/false"),
    ({"grid.numbering": "spiral"}, "not one of"),
    ({"grid.origin_in_map_m": [0.0]}, "expected 2 values"),
    ({"extr.T_body_camera": [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}, "not a proper rotation"),
    ({"extr.T_body_camera": [[1, 0, 0]]}, "expected shape"),
    ({"grid.cell_size_m": None}, "cannot be set to null"),
    ({"l1.depth_min_m": 5.0, "l1.depth_max_m": 1.0}, "must be less than"),
])
def test_invalid_overrides_rejected(tmp_path, data, match):
    with pytest.raises(ConfigError, match=match):
        load_config(_write(tmp_path, data))


def test_invalid_override_changes_nothing(tmp_path):
    cfg = default_config()
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, {"grid.n_cells_x": 6, "grid.numbering": "spiral"}), base=cfg)
    assert cfg.value("grid.n_cells_x") is None


def test_invalid_json_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid JSON"):
        load_config(path)


def test_default_configs_are_independent():
    a, b = default_config(), default_config()
    a.set("grid.n_cells_x", 6)
    assert b.value("grid.n_cells_x") is None
