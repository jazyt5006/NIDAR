cmake_minimum_required(VERSION 3.8)
project(survivor_tagging_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection.msg"
  "msg/DetectionArray.msg"
  "msg/CandidateCell.msg"
  "msg/SurvivorTag.msg"
  "msg/SurvivorTagArray.msg"
  "msg/TelemetryPacket.msg"
  "msg/CorrectionEvent.msg"
  "srv/SetGridRegistration.srv"
  "srv/CommitFinalTags.srv"
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
