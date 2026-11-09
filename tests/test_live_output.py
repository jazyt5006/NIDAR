"""Layer 5 live output (substep 3.2): records from tags, telemetry sizes, CRC / decode round trips with the
revision fields, snapshot triggers, rate limiting, removal and truncation. Unit tests."""

import json

import numpy as np
import pytest

from survivor_tagging.config import default_config
from survivor_tagging.output import (TAG_RECORD_SIZE, PublishedTag, TagChange, TagFlags, TelemetryDecodeError,
                                     TelemetryParams, TelemetryScheduler, decode_snapshot, decode_tag_record,
                                     encode_tag_record, full_tag_dict, snapshot_size, tag_record)
from survivor_tagging.pipeline import telemetry_params
from survivor_tagging.types import FailureReason, Tag, TagStatus

S = 1_000_000_000


def make_tag(sid=0, status=TagStatus.CONFIRMED, cell=(3, 3), prob=0.95, map_rev=1, reg_rev=1, out=False,
             reasons=(), update_s=2.0):
    return Tag(sid, status, cell, None if cell is None else f"{cell[0]},{cell[1]}", ((cell, prob), ((3, 4), 0.04)),
               np.array([3.4, 3.5, 0.8]), np.eye(3) * 0.01, np.array([3.41, 3.52]), 0.1, 0.41, out, 0.87, prob, 10,
               2.0, 0, int(update_s * S), int(update_s * S), int(update_s * S), map_rev, reg_rev, 1, tuple(reasons))


def pub(sid=0, rev=0, removed=False, **kw):
    return PublishedTag(sid, rev, make_tag(sid, **kw), removed, ())


def params(**kw):
    p = telemetry_params(default_config())
    return TelemetryParams(**{**p.__dict__, **kw})


def test_record_from_published_tag_round_trip_with_revisions():
    pt = pub(sid=300, status=TagStatus.UPDATED, map_rev=257, reg_rev=3)
    rec = tag_record(pt, seq=9, t0_ns=0, reg_valid=True, confirm_prob=0.9)
    data = encode_tag_record(rec)
    assert len(data) == TAG_RECORD_SIZE == 25
    back = decode_tag_record(data)
    assert back.survivor_id == 300 & 0xFF and back.map_rev == 1 and back.reg_rev == 3  # low bytes
    assert back.status == TagStatus.UPDATED and back.cell == (3, 3) and back.update_ms == 2000
    assert back.grid_xy_m == pytest.approx((3.41, 3.52)) and back.sigma_m == pytest.approx(0.1)
    assert back.cell_prob == pytest.approx(0.95, abs=1 / 255) and back.det_conf == pytest.approx(0.87, abs=1 / 255)
    assert back.flags == TagFlags.REGISTRATION_VALID


def test_record_flags():
    assert tag_record(pub(prob=0.6), 0, 0, False, 0.9).flags == TagFlags.BOUNDARY_AMBIGUOUS
    f = tag_record(pub(out=True, reasons=(FailureReason.POSE_GAP_TOO_LARGE,)), 0, 0, True, 0.9).flags
    assert f == TagFlags.REGISTRATION_VALID | TagFlags.OUT_OF_GRID | TagFlags.POSE_BLOCKED
    assert tag_record(pub(reasons=(FailureReason.GRID_UNREGISTERED,)), 0, 0, False, 0.9).flags == TagFlags.NONE
    assert tag_record(pub(removed=True), 0, 0, True, 0.9).flags & TagFlags.REMOVED


@pytest.mark.parametrize("index", [0, 5, 21, 22, 24])
def test_crc_detects_corruption_of_a_built_record(index):
    data = bytearray(encode_tag_record(tag_record(pub(map_rev=5, reg_rev=6), 1, 0, True, 0.9)))
    data[index] ^= 0x01
    with pytest.raises(TelemetryDecodeError):
        decode_tag_record(bytes(data))


def test_snapshot_size_and_round_trip_with_revisions():
    sched = TelemetryScheduler(params(), 0)
    tags = [pub(sid=k, cell=(k, 1)) for k in range(6)]
    (msg,) = [m for m in sched.step(0, tags, [], 258, 7, True) if m.kind == "snapshot"]
    assert len(msg.payload) == snapshot_size(6) == 164
    snap = decode_snapshot(msg.payload)
    assert (snap.map_rev, snap.reg_rev, len(snap.records)) == (2, 7, 6)
    assert [r.cell for r in snap.records] == [(k, 1) for k in range(6)]
    bad = bytearray(msg.payload)
    bad[-3] ^= 0xFF
    with pytest.raises(TelemetryDecodeError):
        decode_snapshot(bytes(bad))


def test_snapshot_period_and_immediate_snapshot_on_revision_change():
    sched = TelemetryScheduler(params(snapshot_period_s=5.0), 0)
    kinds = lambda msgs: [m.kind for m in msgs]  # noqa: E731
    assert kinds(sched.step(0, [], [], 0, 0, True)) == ["snapshot"]
    assert kinds(sched.step(4 * S, [], [], 0, 0, True)) == []
    assert kinds(sched.step(5 * S, [], [], 0, 0, True)) == ["snapshot"]
    assert kinds(sched.step(5 * S + 1, [], [], 1, 0, True)) == ["snapshot"]  # map_rev changed
    assert kinds(sched.step(5 * S + 2, [], [], 1, 1, True)) == ["snapshot"]  # reg_rev changed
    assert sched.stats.n_snapshots == 4


def test_updates_are_rate_limited_per_tag_and_send_the_latest_state():
    sched = TelemetryScheduler(params(update_min_interval_s=1.0, snapshot_period_s=100.0), 0)
    sched.step(0, [], [], 0, 0, True)  # initial snapshot (no tags)
    m = sched.step(int(0.1 * S), [pub(status=TagStatus.PROVISIONAL)], [], 0, 0, True)
    assert [x.kind for x in m] == ["update"]
    assert sched.step(int(0.4 * S), [pub(status=TagStatus.CONFIRMED)], [], 0, 0, True) == []
    assert sched.step(int(0.8 * S), [pub(status=TagStatus.CONFIRMED, cell=(3, 4))], [], 0, 0, True) == []
    assert sched.stats.rate_limited == 2
    (late,) = sched.step(int(1.1 * S), [pub(status=TagStatus.CONFIRMED, cell=(3, 4))], [], 0, 0, True)
    r = decode_tag_record(late.payload)
    assert (r.status, r.cell) == (TagStatus.CONFIRMED, (3, 4))  # deferred change, current state
    other = sched.step(int(1.2 * S), [pub(status=TagStatus.CONFIRMED, cell=(3, 4)), pub(sid=1)], [], 0, 0, True)
    assert [decode_tag_record(x.payload).survivor_id for x in other] == [1]  # the limit is per tag


def test_candidates_not_sent_unless_configured_and_removed_announced_once():
    sched = TelemetryScheduler(params(), 0)
    (snap,) = sched.step(0, [pub(0, status=TagStatus.CANDIDATE), pub(1)], [], 0, 0, True)
    assert snap.survivor_ids == (1,)
    sched2 = TelemetryScheduler(params(send_candidates=True), 0)
    assert sched2.step(0, [pub(0, status=TagStatus.CANDIDATE)], [], 0, 0, True)[0].survivor_ids == (0,)
    msgs = sched.step(3 * S, [], [pub(1, removed=True)], 0, 0, True)
    assert [decode_tag_record(m.payload).flags & TagFlags.REMOVED for m in msgs] == [TagFlags.REMOVED]
    assert sched.step(4 * S, [], [], 0, 0, True) == []
    assert sched.step(4 * S, [], [pub(7, removed=True)], 0, 0, True) == []  # never sent -> nothing to retract


def test_snapshot_truncation_keeps_highest_status():
    sched = TelemetryScheduler(params(max_tags_per_snapshot=2), 0)
    tags = [pub(0, status=TagStatus.PROVISIONAL), pub(1), pub(2, status=TagStatus.UNRELIABLE), pub(3)]
    (snap,) = [m for m in sched.step(0, tags, [], 0, 0, True) if m.kind == "snapshot"]
    assert snap.survivor_ids == (1, 3) and sched.stats.truncated_snapshots == 1


def test_bytes_against_link_budget():
    """architecture.md (e): snapshot 14 + 25 N bytes; a 32-tag snapshot every 5 s plus 6 tags at the 1/s update
    limit is 2502.4 bit/s, inside the ASSUMED 20 kbit/s link (12.5 %, before link framing overhead)."""
    p = params()
    worst_bps = 8 * (snapshot_size(p.max_tags_per_snapshot) / p.snapshot_period_s + 6 * TAG_RECORD_SIZE
                     / p.update_min_interval_s)
    assert snapshot_size(32) == 814 and worst_bps == pytest.approx(2502.4) and worst_bps < p.link_budget_bps


def test_full_tag_dict_is_plain_json():
    ch = TagChange(2 * S, (3, 4), (3, 3), "correction r1", 1, 0, 1)
    d = full_tag_dict(PublishedTag(4, 1, make_tag(4, status=TagStatus.UPDATED), False, (ch,)))
    text = json.dumps(d)
    assert '"tag_status": "UPDATED"' in text and d["history"][0]["new_cell"] == [3, 3]
    for key in ("survivor_id", "map_position_m", "grid_position_m", "grid_cell", "position_covariance_m2",
                "detection_confidence", "localisation_quality", "tag_status", "capture_timestamp_ns",
                "update_timestamp_ns", "map_revision", "registration_revision"):  # spec 9.4 field list
        assert key in d
