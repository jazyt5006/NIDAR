"""Plain dict -> ROS message object (substep 4.1b). No rclpy import: it works on any generated ROS 2 Python message class
through `get_fields_and_field_types()` (stable in Humble and Jazzy), and is tested here with fake message classes.

Field type strings look like 'int32', 'string', 'double[4]', 'sequence<uint8>', 'sequence<pkg/Name>', 'pkg/Name'.
"""

from __future__ import annotations

import importlib
import re
from typing import Any, Callable, Dict

PRIMS_INT = {"int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "byte", "octet", "char"}
PRIMS_FLOAT = {"float", "double", "float32", "float64", "long double"}


class ConvertError(ValueError):
    pass


def default_resolver(type_name: str) -> Any:
    """'pkg/Name' -> generated class (pkg.msg.Name)."""
    pkg, name = type_name.split("/")[0], type_name.split("/")[-1]
    return getattr(importlib.import_module(f"{pkg}.msg"), name)


def _scalar(base: str, v: Any, resolve: Callable[[str], Any]) -> Any:
    if base in PRIMS_INT:
        return int(v)
    if base in PRIMS_FLOAT:
        return float(v)
    if base == "boolean" or base == "bool":
        return bool(v)
    if base.startswith("string") or base.startswith("wstring"):
        return str(v)
    if not isinstance(v, dict):
        raise ConvertError(f"expected a dict for {base}, got {type(v).__name__}")
    return dict_to_msg(resolve(base), v, resolve)


def _value(field_type: str, v: Any, resolve: Callable[[str], Any]) -> Any:
    m = re.fullmatch(r"sequence<(.+?)(?:,\s*\d+)?>", field_type)
    if m:
        base = m.group(1)
        if base in ("uint8", "octet", "byte"):
            return bytes(v) if isinstance(v, (bytes, bytearray)) else bytes(int(x) for x in v)
        return [_scalar(base, x, resolve) for x in v]
    m = re.fullmatch(r"(.+)\[(\d+)\]", field_type)
    if m:
        base, n = m.group(1), int(m.group(2))
        if len(v) != n:
            raise ConvertError(f"{field_type}: expected {n} values, got {len(v)}")
        return [_scalar(base, x, resolve) for x in v]
    return _scalar(field_type, v, resolve)


def dict_to_msg(cls: Any, d: Dict[str, Any], resolve: Callable[[str], Any] = default_resolver) -> Any:
    """Instance of `cls` with every key of `d` set (unknown keys raise ConvertError; unset fields keep their defaults)."""
    msg = cls()
    types = cls.get_fields_and_field_types()
    for k, v in d.items():
        if k not in types:
            raise ConvertError(f"{cls.__name__} has no field {k!r}")
        setattr(msg, k, _value(types[k], v, resolve))
    return msg
