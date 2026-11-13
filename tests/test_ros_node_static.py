"""Substep 4.1b: static checks of the rclpy node shell, launch/params files and the dict -> message converter, runnable
WITHOUT ROS. The shell itself is "not run" until the colleague returns the runbook outputs."""
import ast
import json
import re

import pytest

import ros_fakes as rf
from ros_fakes import msgdef as M
from survivor_tagging_ros import convert as C

NODE = rf.ROS_PKG / "survivor_tagging_ros" / "tagger_node.py"
PKG = "survivor_tagging_msgs"
REG = M.load_package(rf.MSGS_PKG)


# --------------------------------------------------------------------------- AC2: the shell is thin
def test_ac2_node_shell_is_at_most_250_lines_without_numerics_or_wall_clock():
    src = NODE.read_text(encoding="utf-8")
    assert len(src.splitlines()) <= 250, len(src.splitlines())
    tree = ast.parse(src)
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            imported.add(("." * n.level) + (n.module or "") if n.level else (n.module or "").split(".")[0])
    allowed = {"json", "rclpy", "geometry_msgs", "nav_msgs", "rcl_interfaces", "sensor_msgs", "std_msgs",
               "survivor_tagging_msgs", "visualization_msgs", ".convert", ".runtime"}
    assert imported <= allowed, imported - allowed
    assert not imported & {"numpy", "math", "time", "datetime", "scipy", "survivor_tagging", "survivor_tagging_ros"}
    # no arithmetic on data: the only binary operations are none at all
    assert not [n for n in ast.walk(tree) if isinstance(n, (ast.BinOp, ast.AugAssign))], "arithmetic in the shell"
    # the node clock is read exactly once, to be injected into the runtime as its arrival clock
    attrs = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    assert sum(1 for n in attrs if n.attr == "get_clock") == 1 and sum(1 for n in attrs if n.attr == "nanoseconds") == 1
    assert not [n for n in attrs if n.attr in ("stamp", "header", "sec", "nanosec")], "the shell never touches stamps"


def test_ac2_calls_into_the_runtime_are_exactly_the_documented_ones():
    tree = ast.parse(NODE.read_text(encoding="utf-8"))
    calls = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute) and n.value.attr == "rt":
            calls.add(n.attr)  # self.rt.<call> and node.rt.<call>
    assert calls == {"on_camera_info", "on_depth", "on_detections", "on_pose", "on_correction", "call_set_registration",
                     "call_commit_final_tags", "tick", "close"}, calls
    names = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "build_runtime" in names and "dict_to_msg" in names


def test_node_parameters_topics_services_and_entry_points_are_consistent():
    tree = ast.parse(NODE.read_text(encoding="utf-8"))
    params = None
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "PARAMS":
            params = ast.literal_eval(n.value)
    assert params is not None
    yaml = (rf.ROS_PKG / "config" / "tagger.params.yaml").read_text(encoding="utf-8")
    keys = set(re.findall(r"^\s{4}([A-Za-z_.]+):", yaml, flags=re.M))
    assert keys == set(params), keys ^ set(params)
    assert params["body_frame_convention"] == "FLU" and params["depth_aligned_to_color"] == "UNSET"
    assert params["pose_frame"] == "UNSET" and params["config_file"] == ""
    for k, v in params.items():
        if v in ("UNSET", "", [0.0]):
            line = next(l for l in yaml.splitlines() if re.match(rf"\s{{4}}{re.escape(k)}:", l))
            assert "UNSET" in line or v == "" and "UNSET" in yaml, k
    assert yaml.count("ASSUMPTION") >= 8 and "ros__parameters" in yaml
    src = NODE.read_text(encoding="utf-8")
    assert "BEST_EFFORT" in src and "TRANSIENT_LOCAL" in src and "/survivor_tagging/set_grid_registration" in src
    assert "/survivor_tagging/commit_final_tags" in src
    setup = (rf.ROS_PKG / "setup.py").read_text(encoding="utf-8")
    assert "launch/*.py" in setup and "config/*.yaml" in setup and "tagger_node" in setup and "smoke_test" in setup
    launch = (rf.ROS_PKG / "launch" / "tagger.launch.py").read_text(encoding="utf-8")
    ast.parse(launch)
    assert "generate_launch_description" in launch and "use_sim_time" in launch and "tagger_node" in launch
    ast.parse((rf.ROS_PKG / "survivor_tagging_ros" / "smoke_test.py").read_text(encoding="utf-8"))
    assert "<exec_depend>launch_ros</exec_depend>" in (rf.ROS_PKG / "package.xml").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- dict -> message converter
def _ros_type(f: M.Field, pkg: str) -> str:
    base = {"float32": "float", "float64": "double", "bool": "boolean", "byte": "octet"}.get(f.type, f.type)
    if "/" in base or base in M.PRIMITIVES and base not in ("float32", "float64"):
        pass
    if base not in M.PRIMITIVES and "/" not in base and base not in ("float", "double", "boolean", "octet"):
        base = f"{pkg}/{base}"
    if f.array is None:
        return base
    return f"{base}[{f.fixed_len}]" if f.fixed_len else f"sequence<{base}>"


def _fake_classes():
    """Generated-class stand-ins built from the .msg files and the standard types, with ROS-style field type strings."""
    classes, defs = {}, {**M.BUILTIN_TYPES, **REG}

    def make(full):
        if full in classes:
            return classes[full]
        pkg = full.split("/")[0]
        fields = {f.name: _ros_type(f, pkg) for f in defs[full].fields}

        class Msg:
            _fields = fields
            __name__ = full.split("/")[-1]

            @classmethod
            def get_fields_and_field_types(cls):
                return dict(cls._fields)

        Msg.__name__ = full.split("/")[-1]
        classes[full] = Msg
        return Msg

    for full in defs:
        make(full)
    return classes


FAKES = _fake_classes()


def resolve(t):
    return FAKES[t]


def _to_plain(m):
    if isinstance(m, (list, tuple)):
        return [_to_plain(x) for x in m]
    if hasattr(m, "get_fields_and_field_types"):
        return {k: _to_plain(getattr(m, k)) for k in m.get_fields_and_field_types() if hasattr(m, k)}
    return m


def test_dict_to_msg_round_trips_every_published_payload_type():
    import test_ros_adapters as T
    from types import SimpleNamespace as NS
    from ros_fakes import adapters as A
    d = A.tag_to_msg_dict(T._tag(), "map", 2, False)
    m = C.dict_to_msg(FAKES[f"{PKG}/SurvivorTag"], d, resolve)
    back = _to_plain(m)
    assert back["header"]["frame_id"] == "map" and back["cell_i"] == 3 and back["candidate_cells"][0]["probability"] == 0.93
    assert back["position_grid_xy_m"] == d["position_grid_xy_m"] and len(back["covariance_map_xy_m2"]) == 4
    assert back["warning_codes"] == d["warning_codes"] and back["first_seen_time"] == d["first_seen_time"]
    arr = A.tag_array_to_msg_dict([NS(survivor_id=1, tag_revision=0, tag=T._tag(), removed=False)] * 2, "map", 5, 1, 1,
                                  True, True)
    assert len(_to_plain(C.dict_to_msg(FAKES[f"{PKG}/SurvivorTagArray"], arr, resolve))["tags"]) == 2
    pk = A.telemetry_packet_to_msg_dict(T._record_bytes(), 9, "map")
    m = C.dict_to_msg(FAKES[f"{PKG}/TelemetryPacket"], pk, resolve)
    assert isinstance(m.data, bytes) and list(m.data) == pk["data"] and m.kind == 1
    resp = A.commit_response_dict([], "confirmed_only", None, "map", 1)
    assert _to_plain(C.dict_to_msg(FAKES[f"{PKG}/CommitFinalTags_Response"], resp, resolve))["policy_used"] == "confirmed_only"
    reg = {"accepted": True, "reason": "", "current_version": 2}
    assert C.dict_to_msg(FAKES[f"{PKG}/SetGridRegistration_Response"], reg, resolve).current_version == 2


def test_dict_to_msg_errors_and_markers():
    with pytest.raises(C.ConvertError):
        C.dict_to_msg(FAKES[f"{PKG}/CandidateCell"], {"i": 1, "bogus": 2}, resolve)
    with pytest.raises(C.ConvertError):
        C.dict_to_msg(FAKES[f"{PKG}/SurvivorTag"], {"position_grid_xy_m": [1.0]}, resolve)  # fixed array length
    with pytest.raises(C.ConvertError):
        C.dict_to_msg(FAKES[f"{PKG}/SurvivorTag"], {"header": 5}, resolve)  # nested message needs a dict
    # marker dicts only use fields of visualization_msgs/Marker
    marker_fields = {"header", "ns", "id", "type", "action", "pose", "scale", "color", "lifetime", "frame_locked",
                     "points", "colors", "text", "mesh_resource", "mesh_use_embedded_materials"}
    from types import SimpleNamespace
    from survivor_tagging_ros import runtime as R
    from survivor_tagging_ros import sim_bridge as B
    import ros_runtime_helpers as H
    placed, events, rt, bus = H.run_via_messages("standing", 0)
    for mk in bus.of("markers")[-1]["markers"]:
        assert set(mk) <= marker_fields and set(mk["pose"]) == {"position", "orientation"}
        assert set(mk["color"]) == {"r", "g", "b", "a"} and set(mk["scale"]) == {"x", "y", "z"}
        assert set(mk["lifetime"]) == {"sec", "nanosec"}
    assert R.MARKER_CUBE == 1 and R.MARKER_TEXT == 9 and R.ACTION_ADD == 0 and R.ACTION_DELETE == 2
    # service reply keys equal the response fields of the .srv file
    req = B.registration_request(SimpleNamespace(version=5, registration=placed.registration), placed.arena.n_cells)
    reply = rt.call_set_registration(B.ns(req))
    assert set(reply) == {f.name for f in REG[f"{PKG}/SetGridRegistration_Response"].fields}
    json.dumps(rt.diagnostics(10**10))  # the node publishes it as JSON text
