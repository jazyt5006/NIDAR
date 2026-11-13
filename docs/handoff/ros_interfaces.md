# ROS 2 interfaces of the survivor tagging subsystem (handoff, substeps 4.1a and 4.1b)

Status: **interface proposal, unit tested without ROS**. The message files, the adapters, the runtime and the node were
written on a Windows machine that cannot run ROS: the runtime and adapters are unit and simulation tested there; the
message build, the rclpy node, the launch file and the smoke test's ROS mode are **not run**. The build and test steps for
the Ubuntu laptop are in `docs/handoff/ros_runbook.md`. Every topic name,
QoS choice and parameter default below is an ASSUMPTION unless it says otherwise. Written for both Humble and Jazzy
(only stable rclpy features: topics, services, parameters, timers, QoS profiles).

Packages (both under `ros2/`):
- `survivor_tagging_msgs` (ament_cmake, rosidl): the message and service files below.
- `survivor_tagging_ros` (ament_python): `survivor_tagging_ros/adapters.py` (pure Python + numpy; ROS messages are
  duck-typed, produced messages are plain dicts with the field names of the `.msg` files), `runtime.py`
  (`TaggerRuntime`: everything the node does except talking to ROS), `convert.py` (dict to message object), `tagger_node.py`
  (the rclpy shell, forwards only), `smoke_test.py`, `sim_bridge.py` (simulator to messages, test side only) and
  `msgdef.py` (a small `.msg` parser used by tests). Launch: `launch/tagger.launch.py`; parameters: `config/tagger.params.yaml`.
  The library `survivor_tagging` (repo `src/`) must be installed in the same Python (`pip install -e .` from the repo
  root); it has no ROS dependency.

## The timestamp requirement

**`DetectionArray.header.stamp` MUST be the image capture time, copied from the header of the camera image the
detection was computed on (ROS clock). It must never be the time inference finished, the time the message was
published, or a wall-clock time taken after inference.** The library looks up the drone pose at that instant; a stamp
that is late by the inference latency (for example 50 ms at 1 m/s and 90 deg/s yaw) shifts every survivor position.
The detection team's own detection records carry an ISO-8601 UTC timestamp that may be inference time: that field must
NOT be used to fill `header.stamp`. The adapters reject zero, negative and malformed stamps and (with a
`StampMonitor`) stamps that run backwards or jump; they cannot detect a stamp that is consistently late.
The depth image stamp and the pose stamps must be on the same ROS clock; the library compares them only through
`clock.camera_to_pose_offset_ns` (unset, Q S3).

## Topics, services and types (all names ASSUMPTION; every one is a parameter of the node)

| Direction | Default name | Type | Notes |
|---|---|---|---|
| in | `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image`, 16UC1 (mm, 0 = invalid) or 32FC1 (m) | depth aligned to colour (Q D2); `step` honoured; best-effort |
| in | `/camera/aligned_depth_to_color/camera_info` | `sensor_msgs/CameraInfo` | intrinsics from `k`; size must equal the image; best-effort |
| in | `/survivor_tagging/detections` | `survivor_tagging_msgs/DetectionArray` | stamp = capture time, frame_id = optical frame of that image; reliable |
| in | `/slam/pose` | `geometry_msgs/PoseWithCovarianceStamped` or `nav_msgs/Odometry` (parameter `pose_topic_type`) | body pose in the map frame; best-effort |
| in | `/slam/correction` | `survivor_tagging_msgs/CorrectionEvent` | `history` or `frame_correction` (Q S4); reliable + transient local |
| srv | `/survivor_tagging/set_grid_registration` | `survivor_tagging_msgs/SetGridRegistration` | official grid (Q O1); nothing is guessed |
| srv | `/survivor_tagging/commit_final_tags` | `survivor_tagging_msgs/CommitFinalTags` | end-of-mission policy (Q O2); waits for in-flight rebuilds |
| out | `/survivor_tagging/tags` | `survivor_tagging_msgs/SurvivorTagArray` | all fields of the library `Tag` plus tag revision; reliable + transient local |
| out | `/survivor_tagging/markers` | `visualization_msgs/MarkerArray` | per tag a flat cell square (map frame, only if the registration is valid) and a text label; colour by status (grey candidate, yellow provisional, green confirmed, blue updated, red unreliable); a removed tag deletes both; reliable + transient local |
| out | `/survivor_tagging/telemetry` | `survivor_tagging_msgs/TelemetryPacket` | library codec bytes for the radio: snapshot every 5 s (and on revision changes), updates rate-limited to one per tag per 1 s (frozen values); reliable |
| out | `/survivor_tagging/diagnostics` | `std_msgs/String` (JSON text, 1 Hz) | rejections by reason code, message counts, pose lag, queue sizes, last recompute duration and cause, pose / map / registration revisions |

Rejection keys in the diagnostics are library failure codes, optionally with a sub-reason after a colon, for example
`RGB_DEPTH_TIME_MISMATCH:no_depth_buffered`, `RGB_DEPTH_TIME_MISMATCH:skew_exceeded`, `INTRINSICS_MISSING:no_camera_info`,
`TIMESTAMP_INVALID`, `FRAME_MISMATCH`, `DEPTH_FRAME_MISMATCH`, `CORRECTION_STALE`.

Behaviour worth knowing: a depth image is matched to a detection array by the nearest stamp within `l2.max_rgb_depth_skew_s` (unset:
configuration error); the pose revision used for live poses is the number of applied correction events (or the SLAM revision number if
larger); a full recompute after a correction or a new registration runs on a worker thread and is swapped in atomically (the old tags stay
published until then); `commit_final_tags` first waits for it. Thread safety: one lock protects all pipeline state; the worker touches only a
private snapshot.

Message sizes (CDR estimate from `msgdef.cdr_size`, simulation-like content with 3 candidate cells and 2 warnings):
`SurvivorTag` 304 B, `SurvivorTagArray` with 6 tags 1810 B, `TelemetryPacket` with one 25-byte record 53 B. The radio
link (ASSUMED 20 kbit/s, Q H5) only ever carries `TelemetryPacket` payloads, never `SurvivorTagArray`.

## Frames and conventions

| Item | Convention | Status |
|---|---|---|
| Library body frame | FLU: x forward, y left, z up | fixed (docs/design/conventions.md) |
| Teammates' body frame | FRD: x forward, y right, z down (their README) | to confirm with the SLAM owner |
| Conversion | `R_flu = R_frd * diag(1, -1, -1)`, i.e. `q_flu = q_frd (x) (w=0, x=1, y=0, z=0)`; position unchanged | parameter `body_convention` (FLU / FRD), **unset = error** |
| Map frame | their world frame is ENU with origin at the maze entry; the library needs a z-up map and never assumes it is the official grid | name in parameter `map_frame_id` |
| Official grid | from `SetGridRegistration`; the SLAM origin is not assumed to be the grid origin | Q O1, unset until answered |
| Camera frame | optical: x right, y down, z forward; the detection `header.frame_id` must equal the depth image frame id | check on the robot |
| Pose covariance | ROS 6x6 row-major, order x, y, z, rotation about FIXED x, y, z of the message frame = library layout (reshape only); unaffected by FRD/FLU. All zeros or `cov[0] == -1` is treated as "unknown" (None), never as a perfect pose | tested with a non-diagonal example |
| `pose_revision` | not in ROS pose messages; the runtime increments it for each applied `CorrectionEvent` (ASSUMPTION) | needs the SLAM owner's answer (S4) |
| Unknown pose covariance | all zeros or `cov[0] == -1` means unknown. The adapter then uses `diag(sp^2 x3, sa^2 x3)` with `pose.default_position_sigma_m` 0.10 and `pose.default_yaw_sigma_rad` 0.05 (also for roll and pitch), marks the pose `covariance_assumed`, and every derived tag carries the non-blocking warning `POSE_COV_ASSUMED` | ASSUMPTION; replace once the pose source is known (Q S1) |

Open point for the SLAM owner: a FRD body orientation relative to an ENU world is unusual (FRD normally goes with NED).
The adapter converts the body axes only; if the published orientation already refers to a NED-based or otherwise rotated
world, the pose would be wrong and the first end-to-end test would show it (the survivors appear mirrored).

## QoS proposals (ASSUMPTION; to verify on the network)

| Topic | Reliability | Durability | History |
|---|---|---|---|
| depth image, camera_info | best effort | volatile | keep last 1-2 |
| detections | reliable | volatile | keep last 10 |
| pose | best effort (sensor data) or reliable if the SLAM node publishes at <= 30 Hz | volatile | keep last 50 (bracketing needs a few) |
| correction | reliable | transient local | keep last 20 |
| tags | reliable | transient local | keep last 1 (the latest array; late joiners such as the GCS get it) |
| telemetry | reliable | volatile | keep last 10 |

## Parameters

Status: **unset** = no value, the node refuses to start (`ConfigurationError` naming the parameter); **ASSUMPTION** = a
default we chose, labelled in the params file; **from freeze** = `configs/frozen_synthetic_v1.json` (applied through `config_file`).

| Node parameter (tagger_node) | Library parameter | Status |
|---|---|---|
| `config_file` | all tuning values | path of the frozen JSON; empty = library defaults (same values) |
| `overrides_file` | `extr.T_body_camera_cov`, `extr.version`, `clock.*`, `l1.depth_min_m`, `l1.depth_max_m`, `l1.depth_noise_model`, `l2.max_interp_gap_s`, `l2.max_rgb_depth_skew_s`, `l2.pose_mode`, `grid.*`, `l5.correction_source` | unset: a flat library override JSON (`configs/example_synthetic.json` shows the keys; its values are TEST values). `smoke_test --write-sim-overrides` writes a simulator one |
| `T_body_camera` | `extr.T_body_camera` | unset (16 values row-major; Q H4); overrides the file |
| `pose_frame` | map frame | unset |
| `body_frame` | pose child frame | unset |
| `body_frame_convention` | - | ASSUMPTION default FLU (the detection team's README says FRD) |
| `clock_domain` | - | ASSUMPTION `ros_time` (a label; all stamps must be on one ROS clock) |
| `depth_aligned_to_color` | `cam.detections_use_depth_aligned_frame` | unset (`"true"`/`"false"`, Q D2); unset in the node AND the overrides file is an error |
| `pose.default_position_sigma_m`, `pose.default_yaw_sigma_rad` | - | ASSUMPTION 0.10 m, 0.05 rad, only for unknown covariance; `UNSET` disables |
| `pose_topic_type`, topic names, `tick_period_s` | - | ASSUMPTION |
| `depth_buffer_size` 8, `pose_max_backward_s` 0.0, `rebuild_in_thread` true | - | ASSUMPTION |
| `use_sim_time` | - | native ROS parameter; true when a simulator provides `/clock`, then for EVERY node |
| `image_scale_tolerance` | - | ASSUMPTION 0.01 (adapter default; not a node parameter yet) |
| `l5.end_of_mission_commit_policy` | same | unset (Q O2); the `commit_final_tags` request names the policy; interim simulation choice `expected_value` 0.667 |
| all `l1.*`, `l4.*`, `l5.*` tuning values | same | from freeze (ASSUMPTION / simulation tested only) |

Detections with a different image size than the depth image are scaled to the depth image only if
`depth_aligned_to_color` is true and both scale factors agree within `image_scale_tolerance`; otherwise
DEPTH_FRAME_MISMATCH. Detections carry no tracking id (`-1`); the library treats it as absent.

## What each teammate must provide

- **Detection team:** `DetectionArray` per camera frame with `header` copied from the camera image header (capture time,
  ROS clock, colour optical frame id), `bbox_xyxy` in pixels of that image, `source_image_width/height`, `confidence`
  0..1, `tracking_id = -1` unless a tracker exists; the rate and latency on the Pi (D5); whether the colour stream is the
  D435i's (D2).
- **SLAM owner:** the pose topic name, type, frame ids, rate and clock; the body convention (FRD or FLU) and the world
  frame definition; a covariance (or a statement that none exists); whether corrected history or a map-frame jump is
  available and how a `CorrectionEvent` can be produced (S4); map origin and re-initialisation (S5); full 3D or planar
  pose (S1).
- **Hardware owner:** camera mounting extrinsics (H4), RealSense driver and depth encoding, aligned-depth topic and
  `camera_info` (H3), telemetry radio (H5), Pi OS, ROS distro (H1, H2).
- **Organisers (via the team):** the official grid definition for `SetGridRegistration` (O1) and the scoring rules for
  the commit policy (O2).
- **Controls team:** a simulation that publishes the same topics with the same stamp rule (C1-C3).

## First checks on the Ubuntu laptop (not done here)

Follow `docs/handoff/ros_runbook.md` (build, tests, launch, smoke test, files to send back). In short: `colcon build` of both packages on
the colleague's distro; `tests/test_ros_node.py` (skipped without rclpy); `smoke_test --dry-run`, then `smoke_test` against the running node;
and compare `header.stamp` of a detection message with the image header it came from (must be equal).
