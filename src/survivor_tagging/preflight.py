"""Mission-day preflight checklist (substep 4.3).

    python -m survivor_tagging.preflight <config.json> [--json]

Accepts either a flat override file (loaded with config.load_config) or a frozen listing such as
configs/frozen_synthetic_v1.json (a "parameters" object whose entries carry "value"). It never invents a value: every
unset parameter is listed with BLOCKING or NON-BLOCKING and the reason. The classification table below is explicit for
every unset parameter of the registry (test_preflight checks that no unset parameter falls through); BLOCKING means no
correct CONFIRMED tag can be produced (or the pipeline refuses to run) without it. Exit code 1 if anything blocks.
Library module: stdlib + config + grid only.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import Config, ConfigError, default_config, load_config
from .grid import GridRegistration

B, NB = True, False

# name -> (blocking, reason). Conditional / optional entries state the condition.
UNSET_TABLE: Dict[str, Tuple[bool, str]] = {
    "cam.intrinsics_source": (B, "Layer 1 cannot back-project without intrinsics (INTRINSICS_MISSING); device or file (Q H3)"),
    "cam.fx_px": (NB, "needed only if cam.intrinsics_source = file; a device source provides them (source itself unset)"),
    "cam.fy_px": (NB, "needed only if cam.intrinsics_source = file"),
    "cam.cx_px": (NB, "needed only if cam.intrinsics_source = file"),
    "cam.cy_px": (NB, "needed only if cam.intrinsics_source = file"),
    "cam.width_px": (NB, "needed only if cam.intrinsics_source = file"),
    "cam.height_px": (NB, "needed only if cam.intrinsics_source = file"),
    "cam.depth_scale_m_per_unit": (NB, "read from the device at runtime; a config value is only a cross-check"),
    "cam.depth_frame_id": (B, "camera stage CONFIG_MISSING: depth frame cannot be checked against the detections"),
    "cam.color_frame_id": (B, "camera stage CONFIG_MISSING: detection image frame unknown"),
    "cam.detections_use_depth_aligned_frame": (B, "bbox pixels may not index the depth image (Q D2); depth would be "
                                                  "sampled on the wrong pixels"),
    "extr.T_body_camera": (B, "camera-to-body transform missing: EXTRINSIC_MISSING, no map position (Q H4)"),
    "extr.T_body_camera_cov": (B, "extrinsic uncertainty missing: the map covariance would be understated (Q H4)"),
    "extr.version": (B, "extrinsics stage CONFIG_MISSING; tags must record which calibration produced them"),
    "clock.camera_to_pose_offset_ns": (B, "pose-at-capture lookup needs a known camera/pose clock relation (Q S3)"),
    "clock.offset_uncertainty_ns": (B, "clock stage CONFIG_MISSING: timing error cannot enter the covariance (Q S3)"),
    "l1.depth_min_m": (B, "Layer 1 valid-depth window unknown (sensor limit, Q H3)"),
    "l1.depth_max_m": (B, "Layer 1 valid-depth window unknown (sensor limit, Q H3)"),
    "l1.depth_noise_model": (B, "depth noise model unknown: measurement covariance cannot be computed (needs measurement)"),
    "map.floor_z_m": (NB, "only floor removal (l1.floor_removal, OFF by default) and the floor cross-check use it"),
    "l2.max_interp_gap_s": (B, "pose interpolation gap limit unknown (depends on the pose rate, Q S1)"),
    "l2.max_rgb_depth_skew_s": (B, "RGB/depth stamp tolerance unknown (Q D2)"),
    "l2.pose_mode": (B, "full 3D or planar+attitude pose unknown (Q S1)"),
    "grid.n_cells_x": (NB, "out-of-grid cannot be decided (no OUT_OF_GRID warning, mass beyond the grid not counted); "
                           "labels 'row_major_label' need it (blocking then)"),
    "grid.n_cells_y": (NB, "out-of-grid cannot be decided (no OUT_OF_GRID warning)"),
    "grid.origin_in_map_m": (B, "grid origin unknown (Q O1): every cell would be GRID_UNREGISTERED, never CONFIRMED"),
    "grid.yaw_map_to_grid_rad": (B, "grid orientation unknown (Q O1): GRID_UNREGISTERED"),
    "grid.axis_flip_x": (B, "grid axis direction unknown (Q O1): GRID_UNREGISTERED"),
    "grid.axis_flip_y": (B, "grid axis direction unknown (Q O1): GRID_UNREGISTERED"),
    "grid.index_base": (B, "first cell index unknown (Q O1): the reported cell would be off by one"),
    "grid.numbering": (B, "official cell label convention unknown (Q O1): labels cannot be produced"),
    "grid.registration_valid": (B, "false by default: the library never publishes CONFIRMED cells without a valid registration"),
    "grid.origin_sigma_m": (NB, "registration origin uncertainty unknown: probabilities ignore it, tags carry "
                                "REG_UNCERTAINTY_UNKNOWN (suggested start 0.3 m, ASSUMPTION)"),
    "grid.yaw_sigma_rad": (NB, "registration yaw uncertainty unknown: same warning (suggested start 0.09 rad, ASSUMPTION)"),
    "l5.correction_source": (B, "SLAM correction mechanism unknown (Q S4): corrections are rejected "
                                "(CORRECTION_SOURCE_UNAVAILABLE); set 'none' to declare there are none"),
    "l5.end_of_mission_commit_policy": (NB, "unset until O2 is answered; final_tags(policy, threshold) must be passed "
                                            "explicitly (interim simulation choice: expected_value, 0.667)"),
    "score.wrong_tag_penalty_assumed": (NB, "wrong-tag penalty unknown (Q O2): expected_value uses threshold 0.5 with a note"),
}


@dataclass(frozen=True)
class Item:
    name: str
    blocking: bool
    required: bool
    reason: str


@dataclass(frozen=True)
class Check:
    topic: str
    status: str  # OK, WARN, BLOCK, NOT CHECKED
    text: str


@dataclass
class PreflightReport:
    path: str
    fmt: str
    version: str
    unset: List[Item]
    checks: List[Check]

    @property
    def blocking(self) -> List[Item]:
        return [i for i in self.unset if i.blocking]

    @property
    def ready(self) -> bool:
        return not self.blocking and not any(c.status == "BLOCK" for c in self.checks)

    def format(self) -> str:
        out = [f"NIDAR mission-day preflight: {self.path}", f"config format: {self.fmt}", f"config version: {self.version}", ""]
        for title, flag in (("BLOCKING unset parameters", True), ("NON-BLOCKING unset parameters", False)):
            rows = [i for i in self.unset if i.blocking is flag]
            out.append(f"{title} ({len(rows)}):")
            out += [f"  {i.name:<40}{'required' if i.required else 'optional':<10}{i.reason}" for i in rows] or ["  none"]
            out.append("")
        out.append("Checks:")
        out += [f"  [{c.status:<11}] {c.topic}: {c.text}" for c in self.checks]
        out.append("")
        nb = sum(c.status == "BLOCK" for c in self.checks)
        out.append("VERDICT: READY" if self.ready else
                   f"VERDICT: NOT READY ({len(self.blocking)} blocking unset parameter(s), {nb} blocking check(s))")
        return "\n".join(out)


def _load(path: Path) -> Tuple[Config, str, str]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    if isinstance(data, dict) and isinstance(data.get("parameters"), dict):
        cfg = default_config()
        flat = {k: e["value"] for k, e in data["parameters"].items()
                if isinstance(e, dict) and e.get("value") is not None}
        cfg.apply_overrides(flat, reference=f"frozen listing: {path.name}")
        note = str(data.get("_note", ""))
        label = note.split(". ")[0] if note else path.name
        return cfg, "frozen listing", f"{label} (sha256 {digest})"
    cfg = load_config(path)
    return cfg, "override file", f"{path.name} (sha256 {digest}); {str(data.get('_note', ''))[:80]}".rstrip("; ")


def run_preflight(path) -> PreflightReport:
    """Build the checklist for a config file. Raises ConfigError / OSError / ValueError on an unreadable file."""
    path = Path(path)
    cfg, fmt, version = _load(path)
    v = cfg.value
    unset: List[Item] = []
    for p in cfg.params():
        if p.value is not None:
            continue
        blocking, reason = UNSET_TABLE.get(p.name, (B, "unset parameter with no preflight classification: treated as blocking"))
        if p.name in ("cam.fx_px", "cam.fy_px", "cam.cx_px", "cam.cy_px", "cam.width_px", "cam.height_px"):
            blocking = cfg.is_required(p)
        if p.name == "grid.n_cells_x" and v("grid.numbering") == "row_major_label":
            blocking = True
        unset.append(Item(p.name, blocking, cfg.is_required(p), reason))
    unset.sort(key=lambda i: (not i.blocking, i.name))
    checks: List[Check] = []
    reg = GridRegistration.from_config(cfg)
    if reg.valid:
        checks.append(Check("registration", "OK", f"valid, version {reg.version}, origin {reg.origin_in_map_m}, yaw "
                                                  f"{reg.yaw_rad:.4f} rad"))
    else:
        checks.append(Check("registration", "BLOCK", "not valid (unset: " + (", ".join(reg.missing) or "none; "
                            "grid.registration_valid false") + "): no CONFIRMED tag possible"))
    if reg.uncertainty_known:
        checks.append(Check("registration uncertainty", "OK", f"origin sigma {reg.origin_sigma_m} m, yaw sigma "
                                                              f"{reg.yaw_sigma_rad} rad (all tags share this error)"))
    else:
        checks.append(Check("registration uncertainty", "WARN", "unknown: probabilities ignore registration error, "
                            "tags carry REG_UNCERTAINTY_UNKNOWN (non-blocking). Suggested start 0.3 m / 0.09 rad (ASSUMPTION)"))
    checks.append(Check("wall alignment", "OK" if not v("grid.wall_alignment_enabled") else "WARN",
                        "OFF (default)" if not v("grid.wall_alignment_enabled") else
                        "ENABLED: assumes walls lie on grid lines (Q O1 unanswered)"))
    if v("extr.T_body_camera") is None:
        checks.append(Check("extrinsics", "BLOCK", "extr.T_body_camera unset (EXTRINSIC_MISSING)"))
    else:
        cov = "with covariance" if v("extr.T_body_camera_cov") is not None else "WITHOUT covariance (blocking)"
        checks.append(Check("extrinsics", "OK" if v("extr.T_body_camera_cov") is not None else "BLOCK",
                            f"present, version {v('extr.version')}, {cov}"))
    fz = v("map.floor_z_m")
    if v("l1.floor_removal") and fz is None:
        checks.append(Check("floor height", "BLOCK", "l1.floor_removal is on but map.floor_z_m is unset (FLOOR_UNKNOWN)"))
    else:
        checks.append(Check("floor height", "OK" if fz is not None else "WARN",
                            f"map.floor_z_m = {fz}" if fz is not None else
                            "unset: fine while floor removal is off (default, failed its rule in 3.1b)"))
    checks.append(Check("pose covariance policy", "NOT CHECKED",
                        "lives in the ROS adapter (pose.default_position_sigma_m / default_yaw_sigma_rad, ASSUMPTION; "
                        "POSE_COV_ASSUMED on every tag that used it), not in the library config; verify the ROS params file"))
    pol = v("l5.end_of_mission_commit_policy")
    checks.append(Check("end-of-mission policy", "OK" if pol else "WARN",
                        pol or "unset (O2): pass final_tags(policy, threshold) explicitly; interim simulation choice "
                               "expected_value 0.667"))
    return PreflightReport(str(path), fmt, version, unset, checks)


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - thin CLI
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args
    args = [a for a in args if a != "--json"]
    if len(args) != 1:
        print("usage: python -m survivor_tagging.preflight <config.json> [--json]", file=sys.stderr)
        return 2
    try:
        rep = run_preflight(args[0])
    except (ConfigError, OSError, ValueError, KeyError) as exc:
        print(f"PREFLIGHT ERROR: cannot read {args[0]}: {exc}", file=sys.stderr)
        return 2
    if as_json:
        print(json.dumps({"path": rep.path, "version": rep.version, "ready": rep.ready,
                          "unset": [i.__dict__ for i in rep.unset], "checks": [c.__dict__ for c in rep.checks]}, indent=1))
    else:
        print(rep.format())
    return 0 if rep.ready else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
