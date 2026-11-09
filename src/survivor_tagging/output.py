"""Layer 5 output. Currently only the compact telemetry codec (architecture.md section e).

Tag record: 25 bytes, little-endian, CRC-16/CCITT-FALSE over bytes 0..22.
Snapshot: 12-byte header + N tag records + CRC-16 over everything before it (14 + 25N bytes).
Encoders quantise physical values; decoders return the quantised values. Out-of-range input raises ValueError
instead of being silently clipped; the only saturation is sigma_cm (see _encode_sigma).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from enum import IntFlag
from typing import Optional, Tuple

from .types import TagStatus

MSG_TAG_UPDATE = 0x01
MSG_SNAPSHOT = 0x02

TAG_RECORD_SIZE = 25
SNAPSHOT_HEADER_SIZE = 12
CRC_SIZE = 2

I16_NONE = 0x7FFF  # sentinel for "no cell" / "unknown position" in i16 fields
U16_NONE = 0xFFFF  # sentinel for "unknown sigma"

_TAG_BODY = struct.Struct("<BBBBBhhBBHhhIBB")  # bytes 0..22
_HEADER = struct.Struct("<BBHBBBBI")
_CRC = struct.Struct("<H")
assert _TAG_BODY.size + CRC_SIZE == TAG_RECORD_SIZE
assert _HEADER.size == SNAPSHOT_HEADER_SIZE


class TagFlags(IntFlag):
    NONE = 0
    REGISTRATION_VALID = 1 << 0
    OUT_OF_GRID = 1 << 1
    BOUNDARY_AMBIGUOUS = 1 << 2
    POSE_BLOCKED = 1 << 3
    REMOVED = 1 << 4


class SnapshotFlags(IntFlag):
    # Not defined in architecture.md; only bit0 is assigned here, the rest is reserved.
    NONE = 0
    REGISTRATION_VALID = 1 << 0


class TelemetryDecodeError(ValueError):
    pass


def crc16_ccitt_false(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no final XOR."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


@dataclass(frozen=True)
class TagRecord:
    seq: int  # u8
    survivor_id: int  # u8
    status: TagStatus
    flags: TagFlags
    cell: Optional[Tuple[int, int]]  # None -> 0x7FFF sentinel in both fields
    cell_prob: float  # [0, 1], quantised to 1/255
    det_conf: float  # [0, 1], quantised to 1/255
    sigma_m: Optional[float]  # horizontal 1-sigma, quantised to 1 cm; None -> 0xFFFF
    grid_xy_m: Optional[Tuple[float, float]]  # quantised to 1 cm; None -> 0x7FFF in both fields
    update_ms: int  # u32, ms since library start; caller wraps
    map_rev: int  # u8, low byte of pose_revision
    reg_rev: int  # u8, low byte of registration_version


@dataclass(frozen=True)
class Snapshot:
    seq: int  # u8
    snapshot_id: int  # u16
    map_rev: int  # u8
    reg_rev: int  # u8
    flags: SnapshotFlags
    update_ms: int  # u32
    records: Tuple[TagRecord, ...]


def _check_uint(name: str, v: int, bits: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or not (0 <= v < (1 << bits)):
        raise ValueError(f"{name}={v!r} is not a u{bits}")
    return v


def _check_i16(name: str, v: int) -> int:
    # 0x7FFF is reserved as the sentinel, so the largest encodable value is 0x7FFE.
    if isinstance(v, bool) or not isinstance(v, int) or not (-0x8000 <= v < I16_NONE):
        raise ValueError(f"{name}={v!r} outside i16 range [-32768, 32766]")
    return v


def _encode_unit(name: str, v: float) -> int:
    if not (isinstance(v, (int, float)) and math.isfinite(v) and 0.0 <= v <= 1.0):
        raise ValueError(f"{name}={v!r} must be in [0, 1]")
    return int(round(v * 255))


def _encode_sigma(v: Optional[float]) -> int:
    if v is None:
        return U16_NONE
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"sigma_m={v!r} must be finite and >= 0")
    # Saturates at 0xFFFE (655.34 m): any such value already means "useless", and 0xFFFF must stay "unknown".
    return min(int(round(v * 100)), U16_NONE - 1)


def _encode_cm(name: str, v: float) -> int:
    if not math.isfinite(v):
        raise ValueError(f"{name}={v!r} must be finite")
    return _check_i16(name, int(round(v * 100)))


def encode_tag_record(rec: TagRecord) -> bytes:
    if rec.cell is None:
        ci = cj = I16_NONE
    else:
        ci, cj = _check_i16("cell_i", rec.cell[0]), _check_i16("cell_j", rec.cell[1])
    if rec.grid_xy_m is None:
        gx = gy = I16_NONE
    else:
        gx, gy = _encode_cm("grid_x_m", rec.grid_xy_m[0]), _encode_cm("grid_y_m", rec.grid_xy_m[1])
    flags = int(rec.flags)
    if flags & ~0x1F:
        raise ValueError(f"flags={flags:#x} uses reserved bits 5-7")
    body = _TAG_BODY.pack(
        MSG_TAG_UPDATE,
        _check_uint("seq", rec.seq, 8),
        _check_uint("survivor_id", rec.survivor_id, 8),
        int(TagStatus(rec.status)),
        flags,
        ci,
        cj,
        _encode_unit("cell_prob", rec.cell_prob),
        _encode_unit("det_conf", rec.det_conf),
        _encode_sigma(rec.sigma_m),
        gx,
        gy,
        _check_uint("update_ms", rec.update_ms, 32),
        _check_uint("map_rev", rec.map_rev, 8),
        _check_uint("reg_rev", rec.reg_rev, 8),
    )
    return body + _CRC.pack(crc16_ccitt_false(body))


def _check_crc(data: bytes, what: str) -> None:
    (stored,) = _CRC.unpack_from(data, len(data) - CRC_SIZE)
    computed = crc16_ccitt_false(data[:-CRC_SIZE])
    if stored != computed:
        raise TelemetryDecodeError(f"{what}: CRC mismatch (stored {stored:#06x}, computed {computed:#06x})")


def decode_tag_record(data: bytes) -> TagRecord:
    data = bytes(data)
    if len(data) != TAG_RECORD_SIZE:
        raise TelemetryDecodeError(f"tag record: expected {TAG_RECORD_SIZE} bytes, got {len(data)}")
    _check_crc(data, "tag record")
    (msg_type, seq, sid, status, flags, ci, cj, prob, conf, sigma, gx, gy, ms, map_rev, reg_rev) = \
        _TAG_BODY.unpack_from(data)
    if msg_type != MSG_TAG_UPDATE:
        raise TelemetryDecodeError(f"tag record: msg_type {msg_type:#04x}, expected {MSG_TAG_UPDATE:#04x}")
    try:
        status_enum = TagStatus(status)
    except ValueError:
        raise TelemetryDecodeError(f"tag record: unknown status {status}") from None
    if flags & ~0x1F:
        raise TelemetryDecodeError(f"tag record: reserved flag bits set ({flags:#04x})")
    if (ci == I16_NONE) != (cj == I16_NONE):
        raise TelemetryDecodeError("tag record: only one cell index is the 'no cell' sentinel")
    if (gx == I16_NONE) != (gy == I16_NONE):
        raise TelemetryDecodeError("tag record: only one grid coordinate is the 'unknown' sentinel")
    return TagRecord(
        seq=seq,
        survivor_id=sid,
        status=status_enum,
        flags=TagFlags(flags),
        cell=None if ci == I16_NONE else (ci, cj),
        cell_prob=prob / 255.0,
        det_conf=conf / 255.0,
        sigma_m=None if sigma == U16_NONE else sigma / 100.0,
        grid_xy_m=None if gx == I16_NONE else (gx / 100.0, gy / 100.0),
        update_ms=ms,
        map_rev=map_rev,
        reg_rev=reg_rev,
    )


def snapshot_size(n_tags: int) -> int:
    return SNAPSHOT_HEADER_SIZE + TAG_RECORD_SIZE * n_tags + CRC_SIZE


def encode_snapshot(snap: Snapshot) -> bytes:
    n = len(snap.records)
    flags = int(snap.flags)
    if flags & ~0x01:
        raise ValueError(f"snapshot flags={flags:#x} uses reserved bits")
    header = _HEADER.pack(
        MSG_SNAPSHOT,
        _check_uint("seq", snap.seq, 8),
        _check_uint("snapshot_id", snap.snapshot_id, 16),
        _check_uint("n_tags", n, 8),
        _check_uint("map_rev", snap.map_rev, 8),
        _check_uint("reg_rev", snap.reg_rev, 8),
        flags,
        _check_uint("update_ms", snap.update_ms, 32),
    )
    frame = header + b"".join(encode_tag_record(r) for r in snap.records)
    return frame + _CRC.pack(crc16_ccitt_false(frame))


def decode_snapshot(data: bytes) -> Snapshot:
    data = bytes(data)
    if len(data) < snapshot_size(0):
        raise TelemetryDecodeError(f"snapshot: {len(data)} bytes is shorter than the minimum {snapshot_size(0)}")
    _check_crc(data, "snapshot")
    msg_type, seq, snapshot_id, n, map_rev, reg_rev, flags, ms = _HEADER.unpack_from(data)
    if msg_type != MSG_SNAPSHOT:
        raise TelemetryDecodeError(f"snapshot: msg_type {msg_type:#04x}, expected {MSG_SNAPSHOT:#04x}")
    if len(data) != snapshot_size(n):
        raise TelemetryDecodeError(f"snapshot: n_tags={n} needs {snapshot_size(n)} bytes, got {len(data)}")
    if flags & ~0x01:
        raise TelemetryDecodeError(f"snapshot: reserved flag bits set ({flags:#04x})")
    records = tuple(
        decode_tag_record(data[SNAPSHOT_HEADER_SIZE + i * TAG_RECORD_SIZE:SNAPSHOT_HEADER_SIZE + (i + 1) * TAG_RECORD_SIZE])
        for i in range(n)
    )
    return Snapshot(seq, snapshot_id, map_rev, reg_rev, SnapshotFlags(flags), ms, records)
