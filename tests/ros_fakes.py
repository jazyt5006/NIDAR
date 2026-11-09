"""Helpers for the ROS-free adapter tests: sys.path for the ros2/ package and fake (duck-typed) message objects."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROS_PKG = ROOT / "ros2" / "survivor_tagging_ros"
MSGS_PKG = ROOT / "ros2" / "survivor_tagging_msgs"
if str(ROS_PKG) not in sys.path:
    sys.path.insert(0, str(ROS_PKG))

from survivor_tagging_ros import adapters, msgdef  # noqa: E402,F401

ZERO_STAMP = NS(sec=0, nanosec=0)


def stamp(ns: int) -> NS:
    return NS(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def header(ns: int, frame: str = "map") -> NS:
    return NS(stamp=stamp(ns), frame_id=frame)


def pose_msg(ns=2_000_000_000, frame="map", pos=(1.0, 2.0, 3.0), q_wxyz=(1.0, 0.0, 0.0, 0.0), cov=None, child=None):
    w, x, y, z = q_wxyz
    m = NS(header=header(ns, frame),
           pose=NS(pose=NS(position=NS(x=pos[0], y=pos[1], z=pos[2]), orientation=NS(x=x, y=y, z=z, w=w)),
                   covariance=list(cov) if cov is not None else [0.0] * 36))
    if child is not None:
        m.child_frame_id = child
    return m


def depth_msg(arr: np.ndarray, encoding: str, step_pad: int = 0, big: bool = False, ns=1_000_000_000,
              frame="cam_color_optical"):
    h, w = arr.shape
    dt = {"16UC1": "u2", "32FC1": "f4"}[encoding]
    isz = np.dtype(dt).itemsize
    rows = arr.astype(("> " if big else "< ").replace(" ", "") + dt)
    raw = b"".join(rows[r].tobytes() + b"\xff" * step_pad for r in range(h))
    return NS(header=header(ns, frame), height=h, width=w, encoding=encoding, is_bigendian=1 if big else 0,
              step=w * isz + step_pad, data=raw)


def camera_info(w=4, h=3, fx=100.0, fy=101.0, cx=2.0, cy=1.5):
    return NS(width=w, height=h, distortion_model="plumb_bob", d=[0.0] * 5,
              k=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0])


def make_fake(registry, pkg, typ):
    """A fake message of a .msg type: nested namespaces with zero / empty values (arrays empty, fixed arrays zeros)."""
    reg = {**msgdef.BUILTIN_TYPES, **registry}

    def build(full):
        out = NS()
        for f in reg[full].fields:
            if f.type in msgdef.PRIMITIVES:
                v = "" if f.type == "string" else (False if f.type == "bool" else 0)
            else:
                v = build(msgdef.resolve(f.type, pkg))
            if f.array is not None:
                n = f.fixed_len
                v = [v if not isinstance(v, NS) else build(msgdef.resolve(f.type, pkg)) for _ in range(n)] \
                    if n else []
            setattr(out, f.name, v)
        return out

    return build(typ if "/" in typ else f"{pkg}/{typ}")
