import pytest

from survivor_tagging.output import (
    I16_NONE,
    TAG_RECORD_SIZE,
    U16_NONE,
    Snapshot,
    SnapshotFlags,
    TagFlags,
    TagRecord,
    TelemetryDecodeError,
    crc16_ccitt_false,
    decode_snapshot,
    decode_tag_record,
    encode_snapshot,
    encode_tag_record,
    snapshot_size,
)
from survivor_tagging.types import TagStatus


def test_crc_check_value():
    assert crc16_ccitt_false(b"123456789") == 0x29B1


def _rec(**kw):
    args = dict(seq=7, survivor_id=3, status=TagStatus.CONFIRMED, flags=TagFlags.REGISTRATION_VALID,
                cell=(4, 11), cell_prob=0.93, det_conf=0.81, sigma_m=0.12, grid_xy_m=(4.37, 11.52),
                update_ms=123456, map_rev=2, reg_rev=1)
    args.update(kw)
    return TagRecord(**args)


def _assert_close(a: TagRecord, b: TagRecord):
    assert (a.seq, a.survivor_id, a.status, a.flags, a.cell, a.update_ms, a.map_rev, a.reg_rev) == \
           (b.seq, b.survivor_id, b.status, b.flags, b.cell, b.update_ms, b.map_rev, b.reg_rev)
    assert abs(a.cell_prob - b.cell_prob) <= 0.5 / 255 + 1e-12
    assert abs(a.det_conf - b.det_conf) <= 0.5 / 255 + 1e-12
    if a.sigma_m is None:
        assert b.sigma_m is None
    else:
        assert abs(a.sigma_m - b.sigma_m) <= 0.005 + 1e-12
    if a.grid_xy_m is None:
        assert b.grid_xy_m is None
    else:
        assert all(abs(x - y) <= 0.005 + 1e-12 for x, y in zip(a.grid_xy_m, b.grid_xy_m))


CASES = [
    _rec(),
    _rec(status=TagStatus.CANDIDATE, flags=TagFlags.NONE, cell=None, grid_xy_m=None, sigma_m=None),
    _rec(status=TagStatus.PROVISIONAL, flags=TagFlags.REGISTRATION_VALID | TagFlags.BOUNDARY_AMBIGUOUS),
    _rec(status=TagStatus.UPDATED, flags=TagFlags.REGISTRATION_VALID | TagFlags.OUT_OF_GRID,
         cell=(-1, 15), grid_xy_m=(-0.4, 15.2)),
    _rec(status=TagStatus.UNRELIABLE, flags=TagFlags.POSE_BLOCKED | TagFlags.REMOVED, cell_prob=0.0, det_conf=1.0),
    _rec(seq=255, survivor_id=255, update_ms=2**32 - 1, map_rev=255, reg_rev=255, cell=(-32768, 32766)),
]


@pytest.mark.parametrize("rec", CASES)
def test_tag_record_round_trip(rec):
    data = encode_tag_record(rec)
    assert len(data) == TAG_RECORD_SIZE == 25
    out = decode_tag_record(data)
    _assert_close(rec, out)
    assert encode_tag_record(out) == data  # decode is the exact inverse of the quantised form


def test_tag_record_sentinels():
    data = encode_tag_record(_rec(cell=None, grid_xy_m=None, sigma_m=None))
    assert int.from_bytes(data[5:7], "little", signed=True) == I16_NONE
    assert int.from_bytes(data[7:9], "little", signed=True) == I16_NONE
    assert int.from_bytes(data[11:13], "little") == U16_NONE
    assert int.from_bytes(data[13:15], "little", signed=True) == I16_NONE
    assert int.from_bytes(data[15:17], "little", signed=True) == I16_NONE
    out = decode_tag_record(data)
    assert out.cell is None and out.grid_xy_m is None and out.sigma_m is None


def test_tag_record_byte_layout():
    data = encode_tag_record(_rec())
    assert data[0] == 0x01 and data[1] == 7 and data[2] == 3 and data[3] == 2 and data[4] == 0x01
    assert int.from_bytes(data[5:7], "little", signed=True) == 4
    assert int.from_bytes(data[7:9], "little", signed=True) == 11
    assert data[9] == round(0.93 * 255) and data[10] == round(0.81 * 255)
    assert int.from_bytes(data[11:13], "little") == 12
    assert int.from_bytes(data[13:15], "little", signed=True) == 437
    assert int.from_bytes(data[15:17], "little", signed=True) == 1152
    assert int.from_bytes(data[17:21], "little") == 123456
    assert data[21] == 2 and data[22] == 1
    assert int.from_bytes(data[23:25], "little") == crc16_ccitt_false(data[:23])


def test_sigma_saturates_below_sentinel():
    out = decode_tag_record(encode_tag_record(_rec(sigma_m=1e6)))
    assert out.sigma_m == (U16_NONE - 1) / 100.0


@pytest.mark.parametrize("index", range(TAG_RECORD_SIZE))
def test_corrupted_byte_detected(index):
    data = bytearray(encode_tag_record(_rec()))
    data[index] ^= 0x10
    with pytest.raises(TelemetryDecodeError):
        decode_tag_record(bytes(data))


@pytest.mark.parametrize("length", [0, 24, 26])
def test_wrong_length_rejected(length):
    data = (encode_tag_record(_rec()) * 2)[:length]
    with pytest.raises(TelemetryDecodeError, match="expected 25 bytes"):
        decode_tag_record(data)


@pytest.mark.parametrize("kw", [
    dict(seq=256), dict(survivor_id=-1), dict(cell=(I16_NONE, 0)), dict(cell_prob=1.01),
    dict(det_conf=float("nan")), dict(sigma_m=-0.1), dict(grid_xy_m=(400.0, 0.0)), dict(update_ms=2**32),
    dict(map_rev=300), dict(flags=TagFlags(0x20)),
])
def test_encode_rejects_out_of_range(kw):
    with pytest.raises(ValueError):
        encode_tag_record(_rec(**kw))


@pytest.mark.parametrize("n", [0, 1, 16])
def test_snapshot_round_trip(n):
    recs = tuple(_rec(survivor_id=i, seq=i, status=TagStatus(i % 5)) for i in range(n))
    snap = Snapshot(seq=9, snapshot_id=513, map_rev=4, reg_rev=1, flags=SnapshotFlags.REGISTRATION_VALID,
                    update_ms=99999, records=recs)
    data = encode_snapshot(snap)
    assert len(data) == snapshot_size(n) == 14 + 25 * n
    out = decode_snapshot(data)
    assert (out.seq, out.snapshot_id, out.map_rev, out.reg_rev, out.flags, out.update_ms) == \
           (9, 513, 4, 1, SnapshotFlags.REGISTRATION_VALID, 99999)
    assert len(out.records) == n
    for a, b in zip(recs, out.records):
        _assert_close(a, b)
    assert encode_snapshot(out) == data


def test_snapshot_rejects_bad_crc_and_length():
    data = encode_snapshot(Snapshot(1, 1, 0, 0, SnapshotFlags.NONE, 0, (_rec(),)))
    bad = bytearray(data)
    bad[3] ^= 0x01
    with pytest.raises(TelemetryDecodeError, match="CRC"):
        decode_snapshot(bytes(bad))
    with pytest.raises(TelemetryDecodeError):
        decode_snapshot(data[:-1])
    with pytest.raises(TelemetryDecodeError, match="shorter"):
        decode_snapshot(data[:10])
    with pytest.raises(TelemetryDecodeError):
        decode_tag_record(data[:25])  # a snapshot header is not a tag record
