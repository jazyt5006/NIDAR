# Runbook: build and test the survivor tagging ROS 2 packages (for the colleague, Ubuntu)

Status: **nothing in this runbook has been run.** The code was written and unit-tested on a Windows machine without ROS.
The runtime (`runtime.py`), the adapters and the smoke tool's dry-run mode are tested there; the message build, the rclpy
node, the launch file and the ROS mode of the smoke tool are NOT. Every step below is therefore a first run; please
send back what is listed at the end, including failures. Written for ROS 2 Humble and Jazzy (distro unknown).

## 1. Prerequisites
- Ubuntu with a ROS 2 install (Humble on 22.04 or Jazzy on 24.04). `source /opt/ros/$ROS_DISTRO/setup.bash`.
- `colcon`, `python3-venv`, `python3-pytest`, `git`. Packages used: `rclpy`, `std_msgs`, `geometry_msgs`, `sensor_msgs`, `nav_msgs`,
  `visualization_msgs`, `rcl_interfaces`, `launch`, `launch_ros`, `ament_index_python` (all normally part of ros-<distro>-ros-base /
  desktop). No internet access is needed at run time; `pip` needs it once to install numpy if the system has none.
- The repository (library in `src/`, ROS packages in `ros2/`). Python 3.10 or newer; numpy (the library's only dependency).

## 2. Python environment (UNVERIFIED recipe)
ROS 2 Python packages live in the system Python, so use a venv that can see them:
```
cd <repo>
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e .          # the survivor_tagging library from the repo root (needs pyproject.toml; if this fails, see 7.)
python -c "import numpy, survivor_tagging; print(numpy.__version__, survivor_tagging.__version__)"
```
If `pip install -e .` is not possible, use `export PYTHONPATH=<repo>/src:$PYTHONPATH` instead. Keep the venv activated
for every command below (also in the terminal that runs `ros2 launch`), otherwise the node cannot import the library.

## 3. Build both packages
```
cd <repo>/ros2
colcon build --packages-select survivor_tagging_msgs survivor_tagging_ros 2>&1 | tee build.log
source install/setup.bash
ros2 interface show survivor_tagging_msgs/msg/SurvivorTag | head -20
ros2 pkg executables survivor_tagging_ros          # expect: tagger_node and smoke_test
```
Expected: both packages `Finished`, no errors (warnings about `setup.py install` being deprecated are normal on Jazzy).
If the message package fails, the first lines of `build.log` with the rosidl error are the most useful file we can get.

## 4. ROS-dependent tests
```
cd <repo>
python -m pytest tests/test_ros_node.py tests/test_ros_node_static.py tests/test_ros_msgs.py tests/test_ros_adapters.py \
    tests/test_ros_runtime.py tests/test_ros_smoke.py -v 2>&1 | tee ros_tests.log
```
`test_ros_node.py` is skipped without rclpy and RUNS here: it constructs the node with the frozen file and the example override file
(test values) and checks the parameters, publishers, subscribers and services. Expected: all tests pass (the others are the same tests
that passed on the development machine).

## 5. Launch, then smoke test
Terminal 1 (sourced, venv active). The parameter file has UNSET entries that must be filled in; for this smoke test use the
synthetic test values:
```
cd <repo>
cp ros2/survivor_tagging_ros/config/tagger.params.yaml /tmp/smoke.params.yaml
# edit /tmp/smoke.params.yaml: config_file: "<repo>/configs/frozen_synthetic_v1.json"
#                              overrides_file: "<repo>/configs/example_synthetic.json"   (TEST VALUES, not hardware)
#                              pose_frame: map      body_frame: body      depth_aligned_to_color: "true"
ros2 launch survivor_tagging_ros tagger.launch.py params_file:=/tmp/smoke.params.yaml 2>&1 | tee node.log
```
Expected: `tagger_node ready (NOT validated on hardware)`. A `ConfigurationError ... unset parameter(s)` means a value above is missing.
NOTE: `example_synthetic.json` (6x6 grid, identity camera mount) does NOT match the scenes the smoke test publishes. For the ROS smoke
test write matching SIMULATION overrides first, and use that file as `overrides_file` (also in the params file above):
```
python -m survivor_tagging_ros.smoke_test --write-sim-overrides /tmp/sim_overrides.json
```
(The same file works for every default smoke scenario except that `registration_update` starts from a different registration; for the
first check use `--scenarios standing,sitting`.) It contains SIMULATION data and must never be used with the real drone.
Terminal 2:
```
cd <repo> && source ros2/install/setup.bash && source .venv/bin/activate
ros2 run survivor_tagging_ros smoke_test --dry-run --out smoke_report_dry.json      # no ROS traffic; must print PASS
ros2 run survivor_tagging_ros smoke_test --scenarios standing,sitting --seeds 0 --out smoke_report.json   # against the node
```
Expected output of the second command: `smoke_test PASS (ros): 2 runs, 2 with a correct cell; report smoke_report.json`. The ROS mode
replays the simulator at real-time speed (a few seconds per scenario), publishes depth, CameraInfo, detections and poses and reads
`/survivor_tagging/tags`. In another terminal you can watch: `ros2 topic echo /survivor_tagging/diagnostics` (JSON text, 1 Hz) and
`ros2 topic hz /survivor_tagging/tags`. Markers: `ros2 topic echo /survivor_tagging/markers --once` (RViz: add MarkerArray, fixed frame = map).

## 6. Files to send back
1. `build.log` (colcon output) and `ros_tests.log`.
2. `smoke_report.json` and `smoke_report_dry.json`.
3. `echo $ROS_DISTRO` and `lsb_release -a` (and `python3 --version`, `pip list | grep -i numpy`).
4. `ros2 topic list -t` and `ros2 service list -t` while the node runs.
5. The terminal output of the node's first 30 seconds (`node.log`, first 30 s).
6. Anything that looked odd, even if tests passed.

## 7. Troubleshooting
- **`ModuleNotFoundError: survivor_tagging`**: the venv is not active in that terminal, or step 2 failed; use the PYTHONPATH form.
- **`ConfigurationError`**: lists the unset parameter names. Nothing is guessed: fill the value in the params file.
- **Everything is rejected with `FRAME_MISMATCH`**: `pose_frame` must equal the `frame_id` of the pose messages (leading `/` ignored); an Odometry
  `child_frame_id` must equal `body_frame`. Check `ros2 topic echo <pose topic> --once`.
- **`RGB_DEPTH_TIME_MISMATCH:no_depth_buffered` or `:skew_exceeded` in the diagnostics**: the detection `header.stamp` is not the stamp of the
  depth image it was computed from. **The detection stamp must be the image capture time copied from the camera image header.** Compare:
  `ros2 topic echo --once --field header.stamp <detections>` against the same field of the depth image. If the detector stamps with the
  time inference finished, positions are shifted by the inference latency times the drone motion.
- **`DEPTH_FRAME_MISMATCH`**: the detection `header.frame_id` differs from the depth image frame (the aligned depth topic uses the colour
  optical frame). Fix the detector's frame id or the depth topic.
- **`TIMESTAMP_INVALID`**: a zero stamp (unset) or a stamp going backwards. Zero stamps usually mean a publisher did not set the header.
- **Clock check (important with Gazebo or bags):** compare image header stamps with the pose stamps and with `ros2 topic echo /clock`.
  `ros2 topic echo --once --field header.stamp <image topic>` and the same for the pose topic should be within a fraction of a second of
  each other and of `date +%s`-style ROS time. With a simulator, `use_sim_time` MUST be true for EVERY node (`ros2 param get <node> use_sim_time`),
  including the detector and SLAM, and `/clock` must be published; one node on wall time makes every detection "stale" or "from the future".
- **Hands-on yaw check (frames):** with the drone on the bench and SLAM running, rotate it counter-clockwise as seen from above by about 90
  degrees. In the library convention (FLU body, ENU-like map, z up) the yaw angle must INCREASE. Check with
  `ros2 topic echo <pose topic> --once` before and after (yaw = 2*atan2(z, w) for a level pose). If it decreases, the source is FRD/NED-like: set
  `body_frame_convention: FRD` only if the pose is a FRD body in a z-up world (what the detection team's README describes); if the WORLD frame is
  NED the conversion is wrong and survivors will appear mirrored: report it.
- **`POSE_COV_ASSUMED` on every tag**: the pose source publishes no covariance (all zeros), so the defaults 0.10 m / 0.05 rad (ASSUMPTIONS) are used;
  this is expected until the pose source is known.
- **Tags never CONFIRMED**: normal until the grid registration, extrinsics and body-depth offset are real (`/survivor_tagging/diagnostics` shows
  `registration_valid`); the library never confirms on an unset registration.
- **`pip install -e .` fails**: the repo has no build backend problem known, but this was never tried on Ubuntu; send the error and use PYTHONPATH.
- **uint8 array assignment errors in the node log** (`TelemetryPacket.data`): the dict-to-message converter assigns `bytes`; if rclpy on your
  distro rejects it, send the traceback (known unverified point).
