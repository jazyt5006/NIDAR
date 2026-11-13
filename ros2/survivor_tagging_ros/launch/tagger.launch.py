"""Launch the tagger node. NOT RUN (no ROS on the development machine). Works with Humble and Jazzy.

ros2 launch survivor_tagging_ros tagger.launch.py use_sim_time:=true params_file:=/path/to/my.params.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory("survivor_tagging_ros"), "config", "tagger.params.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params, description="node parameter YAML"),
        DeclareLaunchArgument("use_sim_time", default_value="false",
                              description="true when a simulator (Gazebo) publishes /clock; then EVERY node must use it"),
        Node(package="survivor_tagging_ros", executable="tagger_node", name="tagger_node", output="screen",
             parameters=[LaunchConfiguration("params_file"),
                         {"use_sim_time": LaunchConfiguration("use_sim_time")}]),
    ])
