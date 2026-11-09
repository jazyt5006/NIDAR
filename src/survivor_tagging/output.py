"""Layer 5 output: the compact telemetry codec (architecture.md section e) and, since substep 3.2, the live
output built on it (PublishedTag, tag_record, full_tag_dict, TelemetryScheduler).

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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .types import FAILURE_SEVERITY, FailureReason, Severity, Tag, TagStatus

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


# ---------------------------------------------------------------------------
# Live output (substep 3.2): published tags, record builder, telemetry scheduler
# ---------------------------------------------------------------------------

NS_PER_MS = 1_000_000
U32 = 1 << 32


@dataclass(frozen=True)
class TagChange:
    """One confirmed-cell change of a published tag."""

    stamp_ns: int
    old_cell: Optional[Tuple[int, int]]
    new_cell: Optional[Tuple[int, int]]
    cause: str  # "layer4", "correction r<N>", "registration v<N>", "missing correction r<N>"
    map_revision: int
    registration_version: int
    tag_revision: int


@dataclass(frozen=True, eq=False)
class PublishedTag:
    """Layer 5 tag for the GCS: the full Tag (survivor_id = stable survivor id, status = published status) plus the
    tag revision, the bounded change history and the removed flag (survivor merged away / dropped)."""

    survivor_id: int
    tag_revision: int
    tag: Tag
    removed: bool = False
    history: Tuple[TagChange, ...] = ()


def record_flags(tag: Tag, reg_valid: bool, confirm_prob: float, removed: bool = False) -> TagFlags:
    """bit0 registration valid; bit1 out of grid; bit2 boundary ambiguous = the tag cell's probability is below
    l4.confirm_prob and another cell has probability; bit3 pose blocked = a BLOCK-severity reason other than
    GRID_UNREGISTERED; bit4 removed."""
    f = TagFlags.NONE
    if reg_valid:
        f |= TagFlags.REGISTRATION_VALID
    if tag.out_of_grid:
        f |= TagFlags.OUT_OF_GRID
    if tag.localisation_quality < confirm_prob and len(tag.cell_probs) > 1:
        f |= TagFlags.BOUNDARY_AMBIGUOUS
    if any(FAILURE_SEVERITY[r] is Severity.BLOCK and r is not FailureReason.GRID_UNREGISTERED for r in tag.reasons):
        f |= TagFlags.POSE_BLOCKED
    if removed:
        f |= TagFlags.REMOVED
    return f


def _unit(v: float) -> float:
    return min(max(float(v), 0.0), 1.0) if math.isfinite(v) else 0.0


def tag_record(pt: PublishedTag, seq: int, t0_ns: int, reg_valid: bool, confirm_prob: float) -> TagRecord:
    """Compact record of a published tag. survivor_id, map_rev and reg_rev are sent as their low byte."""
    t = pt.tag
    g = None if t.p_grid_m is None else (float(t.p_grid_m[0]), float(t.p_grid_m[1]))
    return TagRecord(
        seq=seq & 0xFF, survivor_id=pt.survivor_id & 0xFF, status=t.status,
        flags=record_flags(t, reg_valid, confirm_prob, pt.removed), cell=t.cell,
        cell_prob=_unit(t.localisation_quality), det_conf=_unit(t.max_detection_confidence),
        sigma_m=t.sigma_horizontal_m, grid_xy_m=g,
        update_ms=(max(t.update_stamp_ns - t0_ns, 0) // NS_PER_MS) % U32,
        map_rev=t.map_revision & 0xFF, reg_rev=t.registration_version & 0xFF)


def full_tag_dict(pt: PublishedTag) -> dict:
    """Full GCS / log record (spec 9.4 fields plus revisions and history) with plain Python types."""
    t = pt.tag

    def arr(a):
        return None if a is None else np.asarray(a, dtype=float).tolist()

    return {
        "survivor_id": pt.survivor_id, "tag_revision": pt.tag_revision, "removed": pt.removed,
        "tag_status": t.status.name, "grid_cell": None if t.cell is None else list(t.cell),
        "cell_label": t.cell_label, "cell_probs": [[list(c), float(p)] for c, p in t.cell_probs],
        "map_position_m": arr(t.p_map_m), "grid_position_m": arr(t.p_grid_m),
        "position_covariance_m2": arr(t.cov_map_m2), "sigma_horizontal_m": t.sigma_horizontal_m,
        "detection_confidence": t.max_detection_confidence, "localisation_quality": t.localisation_quality,
        "observation_count": t.observation_count, "capture_timestamp_ns": t.capture_stamp_ns,
        "update_timestamp_ns": t.update_stamp_ns, "map_revision": t.map_revision,
        "registration_revision": t.registration_version, "extrinsic_version": t.extrinsic_version,
        "out_of_grid": t.out_of_grid, "reasons": [r.value for r in t.reasons],
        "history": [{"stamp_ns": h.stamp_ns, "old_cell": None if h.old_cell is None else list(h.old_cell),
                     "new_cell": None if h.new_cell is None else list(h.new_cell), "cause": h.cause,
                     "map_revision": h.map_revision, "registration_version": h.registration_version,
                     "tag_revision": h.tag_revision} for h in pt.history],
    }


@dataclass(frozen=True)
class TelemetryParams:
    snapshot_period_s: float  # l5.telemetry_snapshot_period_s
    max_tags_per_snapshot: int  # l5.telemetry_max_tags_per_snapshot
    send_candidates: bool  # l5.telemetry_send_candidates
    update_min_interval_s: float  # l5.telemetry_update_min_interval_s
    link_budget_bps: float  # l5.telemetry_link_budget_bps
    confirm_prob: float  # l4.confirm_prob (boundary flag)


@dataclass(frozen=True)
class TelemetryMessage:
    stamp_ns: int
    kind: str  # "update" or "snapshot"
    payload: bytes
    survivor_ids: Tuple[int, ...]


@dataclass
class _SentState:
    key: Optional[tuple] = None
    last_ns: Optional[int] = None


@dataclass
class TelemetryStats:
    n_snapshots: int = 0
    n_updates: int = 0
    snapshot_bytes: int = 0
    update_bytes: int = 0
    max_snapshot_bytes: int = 0
    truncated_snapshots: int = 0
    rate_limited: int = 0  # update decisions deferred by the per-tag rate limit

    @property
    def total_bytes(self) -> int:
        return self.snapshot_bytes + self.update_bytes


_RANK = {TagStatus.CONFIRMED: 3, TagStatus.UPDATED: 3, TagStatus.PROVISIONAL: 2, TagStatus.UNRELIABLE: 1,
         TagStatus.CANDIDATE: 0}


class TelemetryScheduler:
    """Send rule (architecture.md e): a full snapshot every snapshot_period_s and immediately after a map_rev or
    reg_rev change; an update record whenever a tag's (status, cell, flags) changes, at most one per tag per
    update_min_interval_s (a deferred change is sent at the first step after the interval, with the then-current
    state). Snapshots carry every non-CANDIDATE tag (CANDIDATE too if send_candidates, or if the tag was sent
    before), highest status first, at most max_tags_per_snapshot. A removed survivor is announced once by an update
    with the REMOVED flag. Memory: one small state per survivor id that was ever sent and not removed."""

    def __init__(self, params: TelemetryParams, t0_ns: int):
        self.p = params
        self.t0_ns = t0_ns
        self.seq = 0
        self.snapshot_id = 0
        self.last_snapshot_ns: Optional[int] = None
        self.last_revs: Optional[Tuple[int, int]] = None
        self.sent: Dict[int, _SentState] = {}
        self.pending_removed: Dict[int, PublishedTag] = {}
        self.stats = TelemetryStats()

    def _next_seq(self) -> int:
        s = self.seq
        self.seq = (self.seq + 1) & 0xFF
        return s

    def _eligible(self, pt: PublishedTag) -> bool:
        return pt.tag.status is not TagStatus.CANDIDATE or self.p.send_candidates or pt.survivor_id in self.sent

    def _key(self, pt: PublishedTag, reg_valid: bool) -> tuple:
        return (int(pt.tag.status), pt.tag.cell, int(record_flags(pt.tag, reg_valid, self.p.confirm_prob,
                                                                   pt.removed)))

    def step(self, now_ns: int, tags: Sequence[PublishedTag], removed: Sequence[PublishedTag], map_rev: int,
             reg_rev: int, reg_valid: bool) -> List[TelemetryMessage]:
        out: List[TelemetryMessage] = []
        for pt in removed:
            if pt.survivor_id in self.sent:
                self.pending_removed[pt.survivor_id] = pt
        live = [pt for pt in tags if self._eligible(pt)]
        revs = (map_rev, reg_rev)
        if (self.last_snapshot_ns is None or now_ns - self.last_snapshot_ns >= self.p.snapshot_period_s * 1e9
                or revs != self.last_revs):
            ranked = sorted(live, key=lambda pt: (-_RANK[pt.tag.status], pt.survivor_id))
            chosen = ranked[:self.p.max_tags_per_snapshot]
            if len(ranked) > len(chosen):
                self.stats.truncated_snapshots += 1
            seq = self._next_seq()
            snap = Snapshot(seq, self.snapshot_id, map_rev & 0xFF, reg_rev & 0xFF,
                            SnapshotFlags.REGISTRATION_VALID if reg_valid else SnapshotFlags.NONE,
                            (max(now_ns - self.t0_ns, 0) // NS_PER_MS) % U32,
                            tuple(tag_record(pt, seq, self.t0_ns, reg_valid, self.p.confirm_prob) for pt in chosen))
            data = encode_snapshot(snap)
            self.snapshot_id = (self.snapshot_id + 1) & 0xFFFF
            self.last_snapshot_ns, self.last_revs = now_ns, revs
            self.stats.n_snapshots += 1
            self.stats.snapshot_bytes += len(data)
            self.stats.max_snapshot_bytes = max(self.stats.max_snapshot_bytes, len(data))
            out.append(TelemetryMessage(now_ns, "snapshot", data, tuple(pt.survivor_id for pt in chosen)))
            for pt in chosen:
                self.sent.setdefault(pt.survivor_id, _SentState()).key = self._key(pt, reg_valid)
        interval_ns = self.p.update_min_interval_s * 1e9
        for pt in list(live) + list(self.pending_removed.values()):
            st = self.sent.setdefault(pt.survivor_id, _SentState())
            key = self._key(pt, reg_valid)
            if key == st.key:
                continue
            if st.last_ns is not None and now_ns - st.last_ns < interval_ns:
                self.stats.rate_limited += 1
                continue
            data = encode_tag_record(tag_record(pt, self._next_seq(), self.t0_ns, reg_valid, self.p.confirm_prob))
            st.key, st.last_ns = key, now_ns
            self.stats.n_updates += 1
            self.stats.update_bytes += len(data)
            out.append(TelemetryMessage(now_ns, "update", data, (pt.survivor_id,)))
            if pt.removed:
                self.pending_removed.pop(pt.survivor_id, None)
                self.sent.pop(pt.survivor_id, None)
        return out
