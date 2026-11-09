"""Minimal ROS 2 .msg / .srv parser, validator and CDR size estimator that works WITHOUT ROS (substep 4.1a).

Used by the unit tests to check the interface files on a machine that cannot run rosidl, and to estimate message sizes.
It is not a replacement for rosidl: it checks the common syntax rules only (see validate_text) and knows the handful of
standard ROS types the interfaces use (BUILTIN_TYPES).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PRIMITIVES = {"bool", "byte", "char", "float32", "float64", "int8", "uint8", "int16", "uint16", "int32", "uint32",
              "int64", "uint64", "string"}
PRIM_SIZE = {"bool": 1, "byte": 1, "char": 1, "float32": 4, "float64": 8, "int8": 1, "uint8": 1, "int16": 2,
             "uint16": 2, "int32": 4, "uint32": 4, "int64": 8, "uint64": 8}
FIELD_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
CONST_RE = re.compile(r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$")
TYPE_RE = re.compile(r"^(?P<base>[A-Za-z][A-Za-z0-9_]*(/[A-Za-z][A-Za-z0-9_]*)?)(?P<arr>\[(<=)?\d*\])?$")


@dataclass
class Field:
    type: str  # base type, e.g. "float32", "std_msgs/Header", "Detection"
    name: str
    array: Optional[str] = None  # None, "[]", "[N]" or "[<=N]"
    line: int = 0

    @property
    def fixed_len(self) -> Optional[int]:
        if self.array and re.fullmatch(r"\[\d+\]", self.array):
            return int(self.array[1:-1])
        return None


@dataclass
class MsgDef:
    name: str
    fields: List[Field] = field(default_factory=list)
    constants: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # name -> (type, value text)
    errors: List[str] = field(default_factory=list)

    def field_names(self) -> List[str]:
        return [f.name for f in self.fields]


def _f(typ: str, name: str, arr: Optional[str] = None) -> Field:
    return Field(typ, name, arr)


# Standard ROS types the interfaces may use (field lists as in the ROS 2 common_interfaces packages).
BUILTIN_TYPES: Dict[str, MsgDef] = {
    "builtin_interfaces/Time": MsgDef("Time", [_f("int32", "sec"), _f("uint32", "nanosec")]),
    "std_msgs/Header": MsgDef("Header", [_f("builtin_interfaces/Time", "stamp"), _f("string", "frame_id")]),
    "geometry_msgs/Point": MsgDef("Point", [_f("float64", "x"), _f("float64", "y"), _f("float64", "z")]),
    "geometry_msgs/Vector3": MsgDef("Vector3", [_f("float64", "x"), _f("float64", "y"), _f("float64", "z")]),
    "geometry_msgs/Quaternion": MsgDef("Quaternion", [_f("float64", "x"), _f("float64", "y"), _f("float64", "z"),
                                                      _f("float64", "w")]),
    "geometry_msgs/Pose": MsgDef("Pose", [_f("geometry_msgs/Point", "position"),
                                          _f("geometry_msgs/Quaternion", "orientation")]),
    "geometry_msgs/Transform": MsgDef("Transform", [_f("geometry_msgs/Vector3", "translation"),
                                                    _f("geometry_msgs/Quaternion", "rotation")]),
    "geometry_msgs/PoseWithCovariance": MsgDef("PoseWithCovariance", [_f("geometry_msgs/Pose", "pose"),
                                                                       _f("float64", "covariance", "[36]")]),
    "geometry_msgs/PoseWithCovarianceStamped": MsgDef("PoseWithCovarianceStamped", [
        _f("std_msgs/Header", "header"), _f("geometry_msgs/PoseWithCovariance", "pose")]),
}


def parse_msg_text(text: str, name: str = "") -> MsgDef:
    """Parse one message body (no '---'). Syntax errors are collected in .errors, never raised."""
    out = MsgDef(name)
    seen: Dict[str, int] = {}
    for no, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^(\S+)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:=\s*(.*))?$", line)
        if not m:
            out.errors.append(f"{name}:{no}: cannot parse '{line}'")
            continue
        typ, ident, value = m.group(1), m.group(2), m.group(3)
        tm = TYPE_RE.match(typ)
        if not tm:
            out.errors.append(f"{name}:{no}: bad type '{typ}'")
            continue
        base, arr = tm.group("base"), tm.group("arr")
        if value is not None:  # constant
            if base not in PRIMITIVES or arr:
                out.errors.append(f"{name}:{no}: constant '{ident}' must have a primitive, non-array type")
            if not CONST_RE.match(ident):
                out.errors.append(f"{name}:{no}: constant name '{ident}' must be UPPER_SNAKE_CASE")
            if not value.strip():
                out.errors.append(f"{name}:{no}: constant '{ident}' has no value")
            if ident in seen:
                out.errors.append(f"{name}:{no}: duplicate name '{ident}' (first at line {seen[ident]})")
            seen.setdefault(ident, no)
            out.constants[ident] = (base, (value or "").strip())
            continue
        if not FIELD_RE.match(ident):
            out.errors.append(f"{name}:{no}: field name '{ident}' must be lower_snake_case")
        if ident in seen:
            out.errors.append(f"{name}:{no}: duplicate field name '{ident}' (first at line {seen[ident]})")
        seen.setdefault(ident, no)
        out.fields.append(Field(base, ident, arr, no))
    return out


def load_package(path: Path) -> Dict[str, MsgDef]:
    """All interfaces of an ament_cmake interface package: key 'pkg/Name' for messages and 'pkg/Name_Request' /
    'pkg/Name_Response' for services. Message type names without a package inside a file refer to the same package."""
    path = Path(path)
    pkg = path.name
    reg: Dict[str, MsgDef] = {}
    for f in sorted((path / "msg").glob("*.msg")):
        reg[f"{pkg}/{f.stem}"] = parse_msg_text(f.read_text(encoding="utf-8"), f"{f.name}")
    for f in sorted((path / "srv").glob("*.srv")):
        parts = re.split(r"^---\s*$", f.read_text(encoding="utf-8"), flags=re.M)
        if len(parts) != 2:
            bad = MsgDef(f.name)
            bad.errors.append(f"{f.name}: a service needs exactly one '---' separator, found {len(parts) - 1}")
            reg[f"{pkg}/{f.stem}_Request"] = bad
            continue
        reg[f"{pkg}/{f.stem}_Request"] = parse_msg_text(parts[0], f"{f.name}[request]")
        reg[f"{pkg}/{f.stem}_Response"] = parse_msg_text(parts[1], f"{f.name}[response]")
    return reg


def resolve(typ: str, pkg: str) -> str:
    return typ if "/" in typ else f"{pkg}/{typ}"


def validate_package(path: Path) -> List[str]:
    """Syntax and reference errors of all interface files of a package (empty list = valid)."""
    path = Path(path)
    pkg = path.name
    reg = load_package(path)
    errors: List[str] = []
    known = set(reg) | set(BUILTIN_TYPES)
    for key, d in reg.items():
        errors += d.errors
        for f in d.fields:
            if f.type in PRIMITIVES:
                continue
            full = resolve(f.type, pkg)
            if full not in known or full.endswith(("_Request", "_Response")):
                errors.append(f"{key}: field '{f.name}' has unknown type '{f.type}'")
    # cycles
    def visit(k: str, stack: Tuple[str, ...]) -> None:
        if k in stack:
            errors.append(f"recursive type: {' -> '.join(stack + (k,))}")
            return
        for f in (reg.get(k) or BUILTIN_TYPES.get(k, MsgDef(k))).fields:
            if f.type not in PRIMITIVES:
                visit(resolve(f.type, pkg), stack + (k,))

    for k in reg:
        visit(k, ())
    return errors


# ---------------------------------------------------------------------------
# CDR (XCDR1, little endian) serialised size of an instance given as nested dicts / lists
# ---------------------------------------------------------------------------


def _align(pos: int, n: int) -> int:
    return (pos + n - 1) // n * n


def _prim(pos: int, typ: str, value: Any) -> int:
    if typ == "string":
        return _align(pos, 4) + 4 + len(str(value).encode("utf-8")) + 1
    n = PRIM_SIZE[typ]
    return _align(pos, n) + n


def cdr_size(registry: Dict[str, MsgDef], pkg: str, typ: str, value: Any, pos: int = 4) -> int:
    """Serialised size in bytes of `value` (dict for messages, list for arrays) of message type `typ` (e.g. 'SurvivorTag'
    or 'std_msgs/Header'), including the 4-byte encapsulation header when called with the default pos=4 ... the returned
    value is the end position, i.e. the total size. Alignment rules of XCDR1: a primitive aligns to its own size
    (max 8 in XCDR1) relative to the start of the data after the 4-byte header, which is why pos starts at 4 here and
    alignment is computed on pos - 4."""
    reg = {**BUILTIN_TYPES, **registry}

    def msg(p: int, full: str, v: Dict[str, Any]) -> int:
        for f in reg[full].fields:
            p = fld(p, f, v[f.name])
        return p

    def one(p: int, ftype: str, v: Any) -> int:
        if ftype in PRIMITIVES:
            return _prim(p - 4, ftype, v) + 4
        return msg(p, resolve(ftype, pkg), v)

    def fld(p: int, f: Field, v: Any) -> int:
        if f.array is None:
            return one(p, f.type, v)
        if f.fixed_len is None:  # sequence: uint32 length first
            p = _align(p - 4, 4) + 4 + 4
        for item in v:
            p = one(p, f.type, item)
        return p

    full = typ if "/" in typ else f"{pkg}/{typ}"
    return msg(pos, full, value)
