"""ROS-dependent tests of the tagger node. SKIPPED on machines without rclpy (this development machine); the colleague
runs them on Ubuntu (docs/handoff/ros_runbook.md). Evidence: NOT RUN until the runbook outputs come back."""
import pytest

rclpy = pytest.importorskip("rclpy")

import ros_fakes as rf  # noqa: E402
from survivor_tagging_ros import tagger_node  # noqa: E402

FROZEN = str(rf.ROOT / "configs" / "frozen_synthetic_v1.json")
OVERRIDES = str(rf.ROOT / "configs" / "example_synthetic.json")  # TEST VALUES ONLY
ARGS = ["--ros-args", "-p", f"config_file:={FROZEN}", "-p", f"overrides_file:={OVERRIDES}", "-p", "pose_frame:=map",
        "-p", "body_frame:=base_link", "-p", "depth_aligned_to_color:=true", "-p", "rebuild_in_thread:=false"]


@pytest.fixture(scope="module")
def node():
    rclpy.init(args=ARGS)
    n = tagger_node.TaggerNode()
    yield n
    n.rt.close()
    try:  # the last test of this module re-initialises rclpy
        n.destroy_node()
        rclpy.shutdown()
    except Exception:  # noqa: BLE001
        pass


def test_parameters_are_declared(node):
    for name in tagger_node.PARAMS:
        assert node.has_parameter(name), name
    assert node.get_parameter("body_frame_convention").value == "FLU"
    assert node.get_parameter("pose_frame").value == "map"


def test_topics_and_services_exist(node):
    pubs = dict(node.get_publisher_names_and_types_by_node(node.get_name(), node.get_namespace()))
    subs = dict(node.get_subscriber_names_and_types_by_node(node.get_name(), node.get_namespace()))
    srvs = dict(node.get_service_names_and_types_by_node(node.get_name(), node.get_namespace()))
    p = tagger_node.PARAMS
    for key in ("tags_topic", "markers_topic", "telemetry_topic", "diagnostics_topic"):
        assert p[key] in pubs, (p[key], list(pubs))
    assert "survivor_tagging_msgs/msg/SurvivorTagArray" in pubs[p["tags_topic"]]
    for key in ("depth_topic", "camera_info_topic", "detections_topic", "pose_topic", "correction_topic"):
        assert p[key] in subs, (p[key], list(subs))
    assert "/survivor_tagging/set_grid_registration" in srvs and "/survivor_tagging/commit_final_tags" in srvs
    assert "survivor_tagging_msgs/srv/CommitFinalTags" in srvs["/survivor_tagging/commit_final_tags"]


def test_missing_required_parameter_is_an_error():
    from survivor_tagging_ros.adapters import ConfigurationError
    rclpy.shutdown() if rclpy.ok() else None
    rclpy.init(args=["--ros-args", "-p", f"config_file:={FROZEN}", "-p", f"overrides_file:={OVERRIDES}"])
    try:
        with pytest.raises(ConfigurationError):
            tagger_node.TaggerNode()  # pose_frame, body_frame, depth_aligned_to_color unset
    finally:
        rclpy.shutdown()
