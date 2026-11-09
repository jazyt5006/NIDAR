import re
from pathlib import Path

import numpy as np
import pytest

from survivor_tagging.types import (
    FAILURE_SEVERITY,
    SPEC_PART10_CODES,
    Detection,
    FailureReason,
    PoseSample,
    Severity,
    TagStatus,
)

SPEC = Path(__file__).resolve().parents[1] / "docs" / "specs" / "survivor_tagging_master_spec.md"


def _spec_part10_rows():
    text = SPEC.read_text(encoding="utf-8")
    part = text.split("# PART 10", 1)[1].split("# PART 11", 1)[0]
    rows = []
    for line in part.splitlines():
        m = re.match(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$", line)
        if m and m.group(1) != "Failure" and not set(m.group(1)) <= set("-"):
            rows.append(m.group(1))
    return rows


def test_every_failure_reason_has_a_severity():
    assert set(FAILURE_SEVERITY) == set(FailureReason)
    assert all(isinstance(s, Severity) for s in FAILURE_SEVERITY.values())


def test_every_spec_part10_row_is_covered():
    rows = _spec_part10_rows()
    assert len(rows) == 17
    assert set(rows) == set(SPEC_PART10_CODES)
    assert all(isinstance(code, FailureReason) for code in SPEC_PART10_CODES.values())


def test_tag_status_codes_match_telemetry_layout():
    assert [s.name for s in TagStatus] == ["CANDIDATE", "PROVISIONAL", "CONFIRMED", "UPDATED", "UNRELIABLE"]
    assert [int(s) for s in TagStatus] == [0, 1, 2, 3, 4]


def _det(bbox, size=(640, 480)):
    return Detection(1, 0.9, bbox, 0, "camera_hw", "cam", size)


def test_detection_bbox_failure():
    assert _det((10, 10, 50, 80)).bbox_failure() is None
    assert _det((50, 10, 10, 80)).bbox_failure() is FailureReason.BBOX_INVALID
    assert _det((10, 10, float("nan"), 80)).bbox_failure() is FailureReason.BBOX_INVALID
    assert _det((-1, 10, 50, 80)).bbox_failure() is FailureReason.BBOX_OUTSIDE_IMAGE
    assert _det((10, 10, 641, 80)).bbox_failure() is FailureReason.BBOX_OUTSIDE_IMAGE


def test_detection_is_frozen():
    with pytest.raises(Exception):
        _det((10, 10, 50, 80)).confidence = 0.1


def _pose(**kw):
    args = dict(stamp_ns=1, clock_domain="ros", frame_id="map", child_frame_id="body",
                position_m=[0, 0, 1], orientation_wxyz=[1, 0, 0, 0], valid=True, pose_revision=0, source="test")
    args.update(kw)
    return PoseSample(**args)


def test_pose_sample_validation():
    p = _pose()
    assert p.failure() is None
    assert not p.position_m.flags.writeable
    assert _pose(valid=False).failure() is FailureReason.POSE_INVALID
    assert _pose(orientation_wxyz=[1, 1, 0, 0]).failure() is FailureReason.POSE_INVALID
    assert _pose(position_m=[np.inf, 0, 0]).failure() is FailureReason.POSE_INVALID
    with pytest.raises(ValueError):
        _pose(position_m=[0, 0])
    with pytest.raises(ValueError):
        _pose(covariance=np.eye(3))
