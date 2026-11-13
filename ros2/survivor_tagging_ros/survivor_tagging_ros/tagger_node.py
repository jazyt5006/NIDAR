"""Thin rclpy shell around TaggerRuntime (substep 4.1b). NOT RUN: it needs ROS 2 (Humble or Jazzy), which this
development machine does not have; it is checked here only by static tests (tests/test_ros_node_static.py) and by
tests/test_ros_node.py on a machine with rclpy.

The shell declares parameters, creates subscriptions / publishers / services / one timer and FORWARDS. It contains no
numerical processing, no timestamp logic and no wall clock: the node clock (`get_clock()`, ROS time, simulated time if
use_sim_time) is only passed to the runtime as its injected arrival clock; capture time comes from message headers inside
the runtime. Calls into the runtime: build_runtime, rt.on_camera_info, rt.on_depth, rt.on_detections, rt.on_pose,
rt.on_correction, rt.call_set_registration, rt.call_commit_final_tags, rt.tick, rt.close.
"""

import json

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from survivor_tagging_msgs.msg import CorrectionEvent, DetectionArray, SurvivorTagArray, TelemetryPacket
from survivor_tagging_msgs.srv import CommitFinalTags, SetGridRegistration
from visualization_msgs.msg import MarkerArray

from .convert import dict_to_msg
from .runtime import build_runtime

# Parameter defaults. "UNSET" = no value (the runtime raises a configuration error where one is required);
# topic names and QoS are ASSUMPTIONS (docs/handoff/ros_interfaces.md).
PARAMS = {
    "config_file": "", "overrides_file": "",  # frozen JSON; flat library overrides (extr.*, clock.*, grid.*, l2.* ...)
    "depth_topic": "/camera/aligned_depth_to_color/image_raw",
    "camera_info_topic": "/camera/aligned_depth_to_color/camera_info",
    "detections_topic": "/survivor_tagging/detections", "pose_topic": "/slam/pose",
    "correction_topic": "/slam/correction", "tags_topic": "/survivor_tagging/tags",
    "markers_topic": "/survivor_tagging/markers", "telemetry_topic": "/survivor_tagging/telemetry",
    "diagnostics_topic": "/survivor_tagging/diagnostics",
    "pose_topic_type": "PoseWithCovarianceStamped",  # or "Odometry"
    "pose_frame": "UNSET", "body_frame": "UNSET", "body_frame_convention": "FLU", "clock_domain": "ros_time",
    "depth_aligned_to_color": "UNSET",  # "true" / "false"; unset is a configuration error
    "T_body_camera": [0.0],  # 16 values row-major (or 4 rows); one value = unset
    "pose.default_position_sigma_m": 0.10, "pose.default_yaw_sigma_rad": 0.05,  # ASSUMPTION, only for unknown covariance
    "depth_buffer_size": 8, "rebuild_in_thread": True, "pose_max_backward_s": 0.0, "tick_period_s": 0.1,
}
POSE_TYPES = {"PoseWithCovarianceStamped": PoseWithCovarianceStamped, "Odometry": Odometry}
PUBLISHERS = {"tags": ("tags_topic", SurvivorTagArray), "markers": ("markers_topic", MarkerArray),
              "telemetry": ("telemetry_topic", TelemetryPacket), "diagnostics": ("diagnostics_topic", String)}


class TaggerNode(Node):
    def __init__(self):
        super().__init__("tagger_node")
        desc = ParameterDescriptor(dynamic_typing=True)
        self.p = {k: self.declare_parameter(k, v, desc).value for k, v in PARAMS.items()}
        log = self.get_logger()
        self.rt = build_runtime(self.p, lambda: self.get_clock().now().nanoseconds, self._publish,
                                lambda s: log.warning(s, throttle_duration_sec=5.0))
        grp = ReentrantCallbackGroup()
        sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=2)
        reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        p = self.p
        self.create_subscription(CameraInfo, p["camera_info_topic"], self.rt.on_camera_info, sensor, callback_group=grp)
        self.create_subscription(Image, p["depth_topic"], self.rt.on_depth, sensor, callback_group=grp)
        self.create_subscription(DetectionArray, p["detections_topic"], self.rt.on_detections, reliable,
                                 callback_group=grp)
        self.create_subscription(POSE_TYPES[p["pose_topic_type"]], p["pose_topic"], self.rt.on_pose, sensor,
                                 callback_group=grp)
        self.create_subscription(CorrectionEvent, p["correction_topic"], self.rt.on_correction, latched,
                                 callback_group=grp)
        self.pubs = {key: self.create_publisher(cls, p[topic], latched if key in ("tags", "markers") else reliable)
                     for key, (topic, cls) in PUBLISHERS.items()}
        self.create_service(SetGridRegistration, "/survivor_tagging/set_grid_registration", self._set_registration,
                            callback_group=grp)
        self.create_service(CommitFinalTags, "/survivor_tagging/commit_final_tags", self._commit, callback_group=grp)
        self.create_timer(float(p["tick_period_s"]), self.rt.tick, callback_group=grp)
        log.info("tagger_node ready (NOT validated on hardware)")

    def _publish(self, key, payload):
        topic, cls = PUBLISHERS[key]
        self.pubs[key].publish(String(data=json.dumps(payload)) if key == "diagnostics" else dict_to_msg(cls, payload))

    def _set_registration(self, request, response):
        return dict_to_msg(SetGridRegistration.Response, self.rt.call_set_registration(request))

    def _commit(self, request, response):
        return dict_to_msg(CommitFinalTags.Response, self.rt.call_commit_final_tags(request))


def main(args=None):
    rclpy.init(args=args)
    node = TaggerNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.rt.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
