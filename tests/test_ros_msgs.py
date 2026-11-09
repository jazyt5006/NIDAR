"""Substep 4.1a: the .msg / .srv files without ROS: syntax validator, completeness of the library <-> message mapping,
and the fields the adapters reference. Evidence level: unit (no rosidl was run; the real build is done on the team's
Ubuntu machine)."""
import dataclasses
import math
import re
from types import SimpleNamespace as NS

import pytest

import ros_fakes as rf
from ros_fakes import adapters as A
from ros_fakes import msgdef as M
from survivor_tagging import output
from survivor_tagging.grid import GridRegistration
from survivor_tagging.types import Detection, PoseCorrection, Tag, TagStatus

PKG = "survivor_tagging_msgs"
REG = M.load_package(rf.MSGS_PKG)


def msg_fields(name):
    return {f.name for f in REG[f"{PKG}/{name}"].fields}


# --------------------------------------------------------------------------- validator on the real files
def test_package_validates_clean():
    assert M.validate_package(rf.MSGS_PKG) == []


def test_all_interfaces_are_listed_in_cmake_and_package_xml_is_consistent():
    cmake = (rf.MSGS_PKG / "CMakeLists.txt").read_text(encoding="utf-8")
    listed = set(re.findall(r'"((?:msg|srv)/[A-Za-z]+\.(?:msg|srv))"', cmake))
    on_disk = {f"msg/{p.name}" for p in (rf.MSGS_PKG / "msg").glob("*.msg")} | \
              {f"srv/{p.name}" for p in (rf.MSGS_PKG / "srv").glob("*.srv")}
    assert listed == on_disk and len(on_disk) == 9
    xml = (rf.MSGS_PKG / "package.xml").read_text(encoding="utf-8")
    used_pkgs = {t.split("/")[0] for d in REG.values() for f in d.fields if "/" in f.type for t in [f.type]}
    for pkg in used_pkgs:
        assert f"<depend>{pkg}</depend>" in xml and f"find_package({pkg} REQUIRED)" in cmake, pkg
        assert pkg in cmake.split("DEPENDENCIES")[1]
    assert "<name>survivor_tagging_msgs</name>" in xml and "ament_cmake" in xml and "rosidl_interface_packages" in xml
    ros_xml = (rf.ROS_PKG / "package.xml").read_text(encoding="utf-8")
    assert "<build_type>ament_python</build_type>" in ros_xml and "survivor_tagging_msgs" in ros_xml
    assert (rf.ROS_PKG / "resource" / "survivor_tagging_ros").exists() and (rf.ROS_PKG / "setup.cfg").exists()
    setup = (rf.ROS_PKG / "setup.py").read_text(encoding="utf-8")
    assert '"console_scripts": []' in setup  # no nodes in 4.1a


def test_required_fields_exist():
    assert {"header", "detections"} == msg_fields("DetectionArray")
    assert {"frame_index", "bbox_xyxy", "confidence", "class_id", "class_name", "tracking_id", "source_image_width",
            "source_image_height"} == msg_fields("Detection")
    bbox = next(f for f in REG[f"{PKG}/Detection"].fields if f.name == "bbox_xyxy")
    assert bbox.type == "float32" and bbox.array == "[4]"
    assert {"kind", "data", "header"} == msg_fields("TelemetryPacket")
    assert {"revision", "mode", "range_start", "range_end", "corrected_poses", "frame_transform", "header"} == \
        msg_fields("CorrectionEvent")
    for n in ("SetGridRegistration_Request", "SetGridRegistration_Response", "CommitFinalTags_Request",
              "CommitFinalTags_Response"):
        assert REG[f"{PKG}/{n}"].fields


# --------------------------------------------------------------------------- validator on bad input
@pytest.mark.parametrize("text,fragment", [
    ("float32 Bad", "lower_snake_case"), ("float32 a\nfloat32 a", "duplicate field"),
    ("uint8 low=1", "UPPER_SNAKE_CASE"), ("uint8 X=1\nuint8 X=2", "duplicate name"),
    ("float32[4 bad", "bad type"), ("string S=", "no value"), ("uint8[] ARR=1", "primitive, non-array"),
    ("Foo/Bar/Baz x", "bad type"), ("float32", "cannot parse"),
])
def test_validator_catches_syntax_errors(text, fragment):
    errs = M.parse_msg_text(text, "t.msg").errors
    assert errs and any(fragment in e for e in errs), errs


def test_validator_accepts_good_syntax_and_reports_package_problems(tmp_path):
    good = M.parse_msg_text("# c\nuint8 A_B=3\nstring S_X=hello  # note\nfloat32[4] box\nstd_msgs/Header header\n"
                            "geometry_msgs/Point[] pts\nuint8[<=8] short\nint32[3] fixed", "g.msg")
    assert good.errors == [] and good.constants["A_B"] == ("uint8", "3") and len(good.fields) == 5
    pkg = tmp_path / "bad_msgs"
    (pkg / "msg").mkdir(parents=True)
    (pkg / "srv").mkdir()
    (pkg / "msg" / "A.msg").write_text("Unknown thing\nother_pkg/Type t\nB b\n", encoding="utf-8")
    (pkg / "msg" / "B.msg").write_text("A a\n", encoding="utf-8")  # A <-> B recursion
    (pkg / "srv" / "S.srv").write_text("float32 x\n", encoding="utf-8")  # no '---'
    errs = M.validate_package(pkg)
    text = "\n".join(errs)
    assert "unknown type 'Unknown'" in text and "unknown type 'other_pkg/Type'" in text
    assert "recursive type" in text and "'---' separator" in text


# --------------------------------------------------------------------------- completeness: Tag
def test_every_library_tag_field_is_mapped_or_omitted_with_reason():
    lib = {f.name for f in dataclasses.fields(Tag)}
    assert set(A.TAG_FIELD_MAP) == lib, lib ^ set(A.TAG_FIELD_MAP)
    msg = msg_fields("SurvivorTag")
    mapped = {m for v in A.TAG_FIELD_MAP.values() for m in v}
    assert mapped <= msg, mapped - msg
    assert msg == mapped | set(A.TAG_MSG_ONLY), msg ^ (mapped | set(A.TAG_MSG_ONLY))
    assert all(isinstance(r, str) and len(r) > 20 for r in {**A.TAG_OMITTED, **A.TAG_MSG_ONLY}.values())


def test_tag_dict_has_exactly_the_message_fields_recursively():
    import test_ros_adapters as T
    d = A.tag_to_msg_dict(T._tag(), "map", 1, False)

    def check(full, v):
        fields = {f.name: f for f in {**M.BUILTIN_TYPES, **REG}[full].fields}
        assert set(v) == set(fields), (full, set(v) ^ set(fields))
        for k, f in fields.items():
            if f.type in M.PRIMITIVES:
                if f.array:
                    assert isinstance(v[k], list)
                    if f.fixed_len:
                        assert len(v[k]) == f.fixed_len, k
                continue
            full_t = M.resolve(f.type, PKG)
            for item in (v[k] if f.array else [v[k]]):
                check(full_t, item)

    check(f"{PKG}/SurvivorTag", d)
    arr = A.tag_array_to_msg_dict([NS(survivor_id=1, tag_revision=0, tag=T._tag(), removed=False)], "map", 5, 1, 1, True,
                                  False)
    check(f"{PKG}/SurvivorTagArray", arr)
    check(f"{PKG}/TelemetryPacket", A.telemetry_packet_to_msg_dict(T._record_bytes(), 9))
    resp = A.commit_response_dict([], "confirmed_only", None, "map", 1)
    assert set(resp) == msg_fields("CommitFinalTags_Response")
    assert set(A.registration_response_dict(NS(applied=True, reason=None, detail=""), 1)) == \
        msg_fields("SetGridRegistration_Response")


def test_status_constants_match_the_library_enum():
    consts = REG[f"{PKG}/SurvivorTag"].constants
    for s in TagStatus:
        assert consts[f"STATUS_{s.name}"] == ("uint8", str(int(s)))
    k = REG[f"{PKG}/TelemetryPacket"].constants
    assert k["KIND_RECORD"][1] == str(output.MSG_TAG_UPDATE) and k["KIND_SNAPSHOT"][1] == str(output.MSG_SNAPSHOT)
    assert A.KIND_RECORD == output.MSG_TAG_UPDATE and A.KIND_SNAPSHOT == output.MSG_SNAPSHOT
    c = REG[f"{PKG}/CorrectionEvent"].constants
    assert c["MODE_HISTORY"][1] == "history" and c["MODE_FRAME_CORRECTION"][1] == "frame_correction"


# --------------------------------------------------------------------------- completeness: Detection, correction
def test_detection_mapping_is_complete():
    lib = {f.name for f in dataclasses.fields(Detection)}
    assert set(A.DETECTION_FIELD_MAP) == lib
    det, arr = msg_fields("Detection"), msg_fields("DetectionArray")
    used = set()
    for lib_name, targets in A.DETECTION_FIELD_MAP.items():
        for t in targets:
            if t.startswith("header."):
                assert t.split(".")[1] in ("stamp", "frame_id") and "header" in arr
            else:
                assert t in det, t
            used.add(t)
        assert targets or lib_name in A.DETECTION_OMITTED  # unmapped library fields need a reason
    for name in det - used:  # message fields the library does not use need a reason too
        assert name in A.DETECTION_OMITTED, name


def test_correction_mapping_is_complete():
    lib = {f.name for f in dataclasses.fields(PoseCorrection)}
    assert set(A.CORRECTION_FIELD_MAP) == lib
    msg = msg_fields("CorrectionEvent")
    used = set()
    for lib_name, targets in A.CORRECTION_FIELD_MAP.items():
        for t in targets:
            assert t.split(".")[0] in msg, t
            used.add(t.split(".")[0])
        assert targets or lib_name in A.CORRECTION_OMITTED
    for name in msg - used:
        assert name in A.CORRECTION_OMITTED or name in ("range_start", "range_end") and \
            "range_start/range_end" in A.CORRECTION_OMITTED, name


# --------------------------------------------------------------------------- fields referenced by adapters exist
def test_adapters_only_reference_fields_that_exist_in_the_message_files():
    """Fake messages are built from the .msg definitions (nothing else); an adapter that reads a field the file does not
    declare raises AttributeError here."""
    cfg = A.AdapterConfig(map_frame_id="map", body_frame_id="base_link", body_convention="FLU", clock_domain="ros_time",
                          depth_aligned_to_color=True, depth_min_m=0.1, depth_max_m=10.0)
    det = rf.make_fake(REG, PKG, "Detection")
    det.bbox_xyxy = [1.0, 1.0, 20.0, 20.0]
    det.source_image_width, det.source_image_height, det.confidence, det.tracking_id = 64, 48, 0.5, -1
    arr = rf.make_fake(REG, PKG, "DetectionArray")
    arr.header.stamp.sec, arr.header.frame_id, arr.detections = 5, "cam", [det]
    assert len(A.detections_from_array(arr, cfg).detections) == 1

    ev = rf.make_fake(REG, PKG, "CorrectionEvent")
    ev.header.stamp.sec, ev.header.frame_id, ev.revision, ev.mode = 5, "map", 1, "frame_correction"
    ev.frame_transform.rotation.w = 1.0
    assert A.correction_from_msg(ev, cfg).source == "frame_correction"
    hist = rf.make_fake(REG, PKG, "CorrectionEvent")
    hist.header.stamp.sec, hist.header.frame_id, hist.revision, hist.mode = 5, "map", 1, "history"
    pose = rf.make_fake(REG, PKG, "CorrectionEvent").corrected_poses  # [] -> build one from geometry_msgs definitions
    p = rf.make_fake({}, "geometry_msgs", "geometry_msgs/PoseWithCovarianceStamped")
    p.header.stamp.sec, p.header.frame_id, p.pose.pose.orientation.w = 4, "map", 1.0
    hist.corrected_poses = pose + [p]
    assert A.correction_from_msg(hist, cfg).poses[0].stamp_ns == 4_000_000_000

    req = rf.make_fake(REG, PKG, "SetGridRegistration_Request")
    req.cell_size_m, req.numbering, req.valid = 1.0, "ij", True
    cur = GridRegistration((0.0, 0.0), 0.0)
    assert A.registration_from_request(req, cur).n_cells_x is None
    creq = rf.make_fake(REG, PKG, "CommitFinalTags_Request")
    creq.policy, creq.threshold = "expected_value", float("nan")
    assert A.commit_request_to_args(creq) == ("expected_value", None)
    pk = rf.make_fake(REG, PKG, "TelemetryPacket")
    pk.kind, pk.data = 1, b"\x01" + b"\x00" * 24
    assert A.telemetry_bytes_from_msg(pk)[0] == 1
    assert math.isnan(float("nan"))


def test_ros_package_modules_import_no_ros_and_the_library_does_not_import_them():
    for name in ("adapters.py", "msgdef.py"):
        src = (rf.ROS_PKG / "survivor_tagging_ros" / name).read_text(encoding="utf-8")
        imports = re.findall(r"^\s*(?:from|import)\s+([A-Za-z_0-9]+)", src, flags=re.M)
        assert not {"rclpy", "rosidl_runtime_py", "sensor_msgs", "geometry_msgs", "std_msgs", "nav_msgs", "tf2_ros",
                    "builtin_interfaces"} & set(imports), imports
    lib = rf.ROOT / "src" / "survivor_tagging"
    for f in lib.glob("*.py"):
        assert "survivor_tagging_ros" not in f.read_text(encoding="utf-8"), f.name
