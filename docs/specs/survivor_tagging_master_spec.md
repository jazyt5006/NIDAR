# NIDAR AirMouse — Survivor Localisation and Grid Tagging
## Master Engineering Specification for Claude Code

**Project:** NIDAR AirMouse  
**Subsystem:** Survivor Localisation and Grid Tagging  
**Compute Platform:** Raspberry Pi 5 Model B, 16GB RAM  
**AI Accelerator:** Hailo-8 / Hailo-8L (exact variant to be confirmed)  
**Primary Depth Sensor:** Intel RealSense D435i  
**Mapping Sensor:** RPLIDAR C1 (owned by the separate 2D SLAM subsystem)  
**Software:** ROS 2-based architecture (distribution to be confirmed after repository and OS audit)

---

# PART 1 — YOUR ROLE AND PROJECT BOUNDARIES

You are implementing the Survivor Localisation and Grid Tagging subsystem for an autonomous indoor drone participating in NIDAR.

Your responsibility is to develop a robust algorithm that takes human detections from an independently developed detection module, estimates the spatial position of each detected survivor, transforms that position into the SLAM map coordinate frame, converts it into the official NIDAR reference grid, and maintains accurate survivor tags during flight.

The objective is to correctly identify which official grid cell contains each survivor.

## 1.1 What you MUST implement

Implement only the following:

1. Detection input adapter.
2. RGB-D frame and detection synchronisation.
3. Robust depth measurement extraction.
4. Survivor 3D position estimation.
5. Survivor ground-reference position estimation.
6. Camera-to-drone coordinate transformation.
7. Time-aligned drone pose retrieval.
8. Drone-body-to-map coordinate transformation.
9. SLAM map-to-NIDAR-grid registration.
10. Grid-cell calculation.
11. Localisation uncertainty estimation.
12. Temporal observation fusion.
13. Survivor tracking and duplicate handling.
14. SLAM correction handling.
15. Live survivor tag output and visualisation interface.
16. Unit tests, integration tests and performance benchmarks.

## 1.2 What you MUST NOT implement

Other team members are responsible for these systems.

Do NOT independently build:

- Human detection or YOLO model.
- Flight controller.
- Motor control.
- Drone stabilisation.
- Obstacle avoidance.
- Path planning.
- Autonomous navigation.
- Complete ground control station.
- Complete drone mission management system.
- Independent replacement for the team's SLAM implementation.

You must create clean interfaces to these systems.

Do not duplicate functionality already implemented in the repository.

## 1.3 Fundamental engineering principle

Build a modular, sensor-aware and testable localisation system.

Never assume that a detection automatically provides a correct spatial position.

Never assume that a depth measurement automatically belongs to the detected person.

Never assume that the SLAM map origin is the official NIDAR grid origin.

Never confirm a grid-cell assignment when the available evidence is insufficient.

---

# PART 2 — HARDWARE AND SENSOR CONFIGURATION

## 2.1 Main onboard computer

Raspberry Pi 5 Model B — 16GB RAM.

Responsibilities:

- ROS 2 communication.
- RealSense data handling.
- Depth processing.
- Survivor localisation.
- Coordinate transformations.
- Grid registration.
- Survivor tracking.
- Uncertainty calculations.
- Publishing tag outputs.

The Pi is shared with other onboard algorithms, including SLAM, obstacle avoidance, path planning and other mission software.

Therefore, the implementation must be computationally efficient.

## 2.2 AI accelerator

Hailo-8 / Hailo-8L accelerator.

The exact variant and connection must be confirmed.

The Hailo accelerator may be used by the human detection subsystem. Do not assume that it is available for arbitrary numerical operations.

This subsystem should consume detection outputs rather than run a second human detection model.

## 2.3 Primary depth camera

Intel RealSense D435i.

This is the primary RGB-D sensor for survivor localisation.

Expected capabilities include:

- RGB image.
- Stereo depth image.
- Camera intrinsics.
- Depth-to-colour alignment support.
- Camera calibration information.
- Camera timestamps.
- Integrated IMU.

The D435i should be the primary source for camera-relative survivor depth.

Do not hardcode camera intrinsics, depth scale, resolution, frame rate, extrinsics or mounting orientation.

Retrieve calibration and configuration from the actual camera and its driver.

Verify the exact firmware, librealsense version, ROS driver and supported operating system during repository/hardware audit.

## 2.4 Other hardware in the drone

The team hardware inventory also includes:

- RPLIDAR C1 — handled by the separate SLAM subsystem.
- Raspberry Pi Camera Module 3 Wide.
- VL53L5CX ×2.
- VL53L1X.
- MicoAir MTF-01.
- ESP32-S3 DevKitC-1.
- OV5647 camera.
- MLX90640 thermal camera.
- Flight controller and telemetry system.

These are not automatically part of the survivor-tagging pipeline.

Use them only if their interfaces and actual roles are confirmed.

The D435i is the primary depth source in this specification.

Do not automatically fuse measurements from other sensors. First establish their calibration, coordinate frames, timestamps, mounting positions and relevance.

---

# PART 3 — OVERALL ALGORITHM

## 3.1 Baseline

The fundamental algorithm is:

**Human Detection + Corresponding Depth → 3D Survivor Position → Map Position → Official Grid Cell → Reliable Survivor Tag**

The implementation must follow these five layers.

### Layer 1 — Robust Survivor Geometry

Reject invalid depth and background measurements; estimate a useful survivor position.

### Layer 2 — Time-Aligned Coordinate Transformation

Convert camera-relative position into the map frame using the drone pose at image capture time.

### Layer 3 — NIDAR Grid Registration

Align the SLAM map with the official reference grid and identify the survivor's cell.

### Layer 4 — Temporal Consistency and Uncertainty

Combine repeated observations and avoid committing unreliable cell assignments.

### Layer 5 — Correction Handling and Live Map Output

Maintain survivor tags through mapping corrections and publish updated results during flight.

Do not change this five-layer architecture without explaining the technical reason and obtaining approval.

---

# PART 4 — INPUT INTERFACES

The subsystem depends on three main inputs.

## 4.1 Human detection input

Human detection is developed by another teammate.

Our system must receive detection data through a defined interface.

Each detection should contain:

- Detection ID.
- Bounding box coordinates.
- Detection confidence.
- Image timestamp.
- Image frame identifier.
- Optional persistent tracking ID.
- Optional class label.

Proposed logical structure:

```cpp
struct Detection {
    int detection_id;

    float confidence;

    float x_min;
    float y_min;
    float x_max;
    float y_max;

    uint64_t timestamp_ns;

    std::string frame_id;

    int tracking_id;
};
```

This is an illustrative structure. Choose the appropriate ROS 2 message or custom interface after inspecting the repository.

Requirements:

1. Support multiple detections in one image.
2. Validate bounding-box dimensions.
3. Reject invalid coordinates.
4. Reject stale detections.
5. Preserve timestamps.
6. Support missing optional tracking IDs.
7. Do not make the detector's internal implementation a dependency.

Important: Establish whether detections come from the D435i RGB stream or a separate Pi Camera Module 3 Wide stream.

If the detector uses a different camera, implement a calibrated cross-camera association pipeline or require detections from the D435i RGB stream. Never apply D435i depth coordinates directly to another camera's image without calibration and geometric transformation.

## 4.2 RGB-D camera input

Subscribe to the appropriate RealSense streams.

Required logical inputs:

- RGB image.
- Aligned depth image.
- CameraInfo.
- Camera frame ID.
- Capture timestamp.
- Depth scale and valid depth range.

Use the RealSense ROS 2 driver and standard message types where appropriate.

Prefer aligned depth to colour when associating depth with RGB detections.

Do not assume image dimensions or stream rates. Read actual configuration.

## 4.3 Drone pose input

The subsystem needs a timestamped drone pose.

The pose provider will be the SLAM/flight telemetry integration.

Required information:

- Position.
- Orientation.
- Timestamp.
- Coordinate frame.
- Pose validity.
- Optional covariance.

Ideally, provide full 3D pose:

- x, y, z.
- Roll, pitch, yaw.

If only 2D SLAM pose is available, identify what additional altitude and attitude sources are available.

Do not silently assume zero roll, pitch or altitude.

The system must be able to query or interpolate historical poses corresponding to the camera capture timestamp.

Do not substitute the latest drone pose when the correct historical pose is unavailable.

---

# PART 5 — LAYER 1: ROBUST SURVIVOR GEOMETRY

## 5.1 Objective

Estimate a robust camera-relative 3D position for each detected survivor.

A bounding box contains both the person and potentially background objects.

A naive average of all depth pixels is not acceptable.

## 5.2 Processing pipeline

For each detection:

1. Receive bounding box.
2. Verify the box lies within the image.
3. Apply a configurable inner region or person-region mask.
4. Extract the corresponding aligned depth pixels.
5. Reject invalid measurements.
6. Reject measurements outside configured depth limits.
7. Identify plausible depth clusters.
8. Reject background contamination and isolated outliers.
9. Estimate a robust representative 3D point.
10. Estimate the survivor's ground-reference position.
11. Estimate measurement quality and uncertainty.

## 5.3 Invalid depth rejection

Reject:

- Zero depth.
- NaN.
- Infinite values.
- Invalid depth scale.
- Measurements outside the supported range.
- Stale depth frames.
- Measurements with insufficient valid pixels.

All thresholds must be configurable.

Do not invent arbitrary fixed depth limits without documenting their source and intended operating range.

## 5.4 Person-region extraction

The initial implementation may use an inset bounding-box region to reduce background contamination.

However, keep the extraction method modular so a person segmentation mask or keypoint input can be added later.

Do not assume the bounding-box centre corresponds to the person's ground position.

## 5.5 Depth clustering

Investigate robust approaches such as:

- Median depth.
- Median absolute deviation.
- Connected components in depth space.
- Depth consistency filtering.
- Spatial clustering.

Choose a lightweight method appropriate for the Raspberry Pi.

Do not select the nearest depth cluster blindly.

A wall, obstacle or other object may be closer to the camera than the person.

Where multiple clusters are plausible, return an ambiguous result rather than falsely selecting one.

## 5.6 3D reconstruction

Use camera intrinsics to reconstruct the 3D point.

For a pinhole camera model:

```text
Z = depth(u, v)

X = (u - cx) * Z / fx

Y = (v - cy) * Z / fy
```

Where:

- u, v are image coordinates.
- fx, fy are focal lengths.
- cx, cy are principal-point coordinates.
- Z is measured depth.
- X, Y, Z are camera-frame coordinates.

Use the correct RealSense intrinsics and depth units.

Confirm the coordinate convention and axis directions from the actual camera frame.

Do not hardcode example intrinsics.

## 5.7 Survivor ground-reference position

The objective is to estimate a meaningful horizontal position on the arena floor.

The body centroid is not necessarily the desired tag location.

Consider:

- Standing survivor.
- Sitting survivor.
- Prone survivor.
- Partially occluded survivor.
- Survivor near a wall.
- Survivor partly outside the image.

The algorithm should support a reference-point estimator that can later use detection keypoints or segmentation.

Do not claim reliable floor-contact localisation from the body centroid alone.

If the ground reference cannot be estimated reliably, preserve the uncertainty and return a provisional result.

## 5.8 Layer 1 output

Produce:

- Detection ID.
- Camera-frame 3D position.
- Ground-reference estimate, if available.
- Measurement timestamp.
- Number of valid depth pixels.
- Depth quality score.
- Measurement covariance or uncertainty estimate.
- Validity state.
- Failure reason if invalid.

---

# PART 6 — LAYER 2: TIME-ALIGNED COORDINATE TRANSFORMATION

## 6.1 Objective

Convert the survivor's camera-relative position into the SLAM map frame.

Coordinate transformation:

```text
Camera Frame
    ↓
Drone Body Frame
    ↓
SLAM Map Frame
```

Use calibrated transforms.

## 6.2 Mathematical transformation

Let:

- P_camera be the estimated survivor position in camera coordinates.
- T_body_camera be the calibrated camera-to-body transform.
- T_map_body(t) be the body pose in the map frame at capture time.

Then:

```text
P_map = T_map_body(t) * T_body_camera * P_camera
```

Use homogeneous transformations or the appropriate tf2 implementation.

## 6.3 Camera-to-body calibration

The camera mounting position and orientation must be represented by a configurable extrinsic transform.

Do not assume the camera faces directly downward.

Do not assume zero translation or zero rotation.

The actual mounting configuration must be confirmed.

Use a calibration procedure or documented measured extrinsics.

## 6.4 Timestamp alignment

This is mandatory.

The image capture timestamp must be associated with the correct drone pose.

Implement:

1. Timestamp preservation from camera input.
2. Pose buffering.
3. Historical pose lookup.
4. Interpolation when valid surrounding poses exist.
5. Timestamp validity checks.
6. Stale-pose rejection.
7. Explicit failure when a reliable pose cannot be obtained.

Account for timestamp clock domains and offsets.

Do not compare unrelated clock timestamps without establishing their relationship.

Do not use the time at which a detection callback executes as a replacement for image capture time.

## 6.5 Full 3D pose

Use full 3D pose where available.

If the mapping system only produces a planar pose, integrate altitude and attitude from a suitable validated source.

Document limitations if full 3D pose is unavailable.

A planar SLAM pose alone is insufficient to describe the full pose of a tilting airborne camera.

## 6.6 Layer 2 output

- Survivor position in map frame.
- Map-frame covariance.
- Capture timestamp.
- Pose timestamp or interpolation details.
- Transform validity.
- Pose quality.
- Failure reason if transformation is invalid.

---

# PART 7 — LAYER 3: NIDAR GRID REGISTRATION

## 7.1 Objective

Convert survivor coordinates in the SLAM map into official competition-grid coordinates.

The SLAM map and official grid may have different origins and orientations.

Never assume they are identical.

## 7.2 Grid registration

Define an explicit transform:

```text
T_grid_map
```

This converts positions from the map frame into the official grid frame.

Registration must account for:

- Translation.
- Rotation.
- Possible axis inversion.
- Official origin.
- Cell dimensions.
- Grid orientation.
- Cell numbering convention.

For a 2D rigid transform:

```text
P_grid = R_grid_map * P_map + t_grid_map
```

Use the actual official grid definition.

## 7.3 Grid calibration

Create a separate registration module.

It should support:

- Manually configured registration parameters.
- Validation using known reference points.
- Registration quality checks.
- Persistence of validated registration parameters.
- Clear invalid/unregistered status.

Do not assume the drone launch position is the official grid origin.

Do not assume that the entry corridor alone resolves the official coordinate system.

Use the competition's actual reference information.

If the official registration is not known, do not publish a confirmed official grid cell.

## 7.4 Grid-cell calculation

For 1 m × 1 m cells, if the grid origin is the lower-left corner and the official convention indexes cells from zero:

```text
i = floor(x_grid / 1.0)

j = floor(y_grid / 1.0)
```

This is only a provisional example.

Follow the actual NIDAR grid orientation, indexing and boundary rules.

Validate:

- Negative coordinates.
- Boundary points.
- Points exactly on cell edges.
- Points outside the grid.
- Invalid registration.
- Rotated grids.

Use a consistent boundary convention.

## 7.5 Layer 3 output

- Map-frame position.
- Official grid-frame position.
- Grid cell index.
- Registration version.
- Registration validity.
- Cell-boundary proximity.
- Out-of-grid status.

---

# PART 8 — LAYER 4: TEMPORAL CONSISTENCY AND UNCERTAINTY

## 8.1 Objective

Repeated observations should improve localisation quality and reduce false tags.

One detection must not automatically create a confirmed survivor tag.

## 8.2 Survivor tracking

Maintain tracks using:

- Persistent detector tracking ID, if available.
- Detection timestamp.
- Spatial proximity.
- Position uncertainty.
- Observation history.
- Detection confidence.

Do not depend exclusively on detector tracking IDs.

The subsystem must support duplicate association if the detector's ID changes.

## 8.3 Data maintained per survivor

Each track should store:

```text
survivor_id
detector_tracking_id
first_seen_timestamp
last_seen_timestamp
map_position
grid_position
grid_cell
position_covariance
detection_confidence
localisation_quality
observation_count
observation_history
tag_status
last_pose_revision
```

Choose appropriate C++ or ROS 2 structures after repository inspection.

## 8.4 Temporal fusion

Use repeated observations to improve the survivor position.

Potential methods:

- Weighted position averaging.
- Covariance-based fusion.
- Kalman filtering, where its assumptions are appropriate.
- Robust outlier rejection.

Do not treat adjacent frames as fully independent observations.

Avoid double-counting highly correlated measurements.

A survivor observed repeatedly from substantially different viewpoints may provide stronger localisation evidence than many nearly identical consecutive frames.

## 8.5 Uncertainty propagation

Maintain uncertainty through:

1. Depth measurement.
2. Camera geometry.
3. Camera extrinsic calibration.
4. Drone pose.
5. Map transformation.
6. Grid registration.

Where practical, propagate covariance using Jacobians or another justified uncertainty method.

Do not use an arbitrary confidence number as a substitute for position uncertainty.

## 8.6 Cell confirmation

Maintain these statuses:

```text
CANDIDATE
PROVISIONAL
CONFIRMED
UPDATED
UNRELIABLE
```

Suggested interpretation:

- CANDIDATE: detection received, localisation not yet sufficiently established.
- PROVISIONAL: position estimated but uncertainty or evidence remains insufficient.
- CONFIRMED: reliable grid-cell assignment according to validated criteria.
- UPDATED: previously recorded tag revised after new evidence or map correction.
- UNRELIABLE: current evidence cannot support a valid tag.

Thresholds must be configurable and experimentally validated.

Do not hardcode illustrative thresholds from an architecture discussion.

## 8.7 Grid-boundary handling

A survivor estimate near a cell boundary must be handled carefully.

If uncertainty spans multiple cells:

- Do not confidently assign one cell.
- Maintain candidate cell probabilities or an equivalent uncertainty representation.
- Collect additional observations where possible.
- Confirm only when the configured evidence criteria are satisfied.

Do not confuse high human-detection confidence with high localisation confidence.

They are different quantities.

## 8.8 Duplicate handling

If a previously observed survivor is detected again:

- Associate the observation with the existing track.
- Update its position.
- Update its uncertainty.
- Recalculate its grid assignment.
- Avoid creating a second survivor record.

The system must not count consecutive video frames as separate survivors.

---

# PART 9 — LAYER 5: SLAM CORRECTION HANDLING AND LIVE OUTPUT

## 9.1 Objective

SLAM may revise historical drone poses after loop closure or map optimisation.

Previously calculated survivor positions may therefore become incorrect.

The tagging subsystem must support corrections.

## 9.2 Historical observation storage

For every accepted observation, retain:

- Original detection timestamp.
- Camera-frame measurement.
- Measurement covariance.
- Camera-to-body calibration version.
- Pose association information or pose/keyframe reference.
- Map position.
- Grid position.
- Registration version.
- Survivor track association.

Store enough information to recompute affected positions.

## 9.3 Correction handling

When the SLAM system reports a map or pose correction:

1. Receive the correction event.
2. Identify affected historical poses or observations.
3. Recalculate their map-frame survivor positions.
4. Update survivor tracks.
5. Recalculate grid-frame coordinates.
6. Recalculate grid-cell assignments.
7. Update published tags.
8. Preserve a consistent revision identifier.

Do not invent a SLAM correction API.

Inspect the chosen SLAM backend and implement an adapter for its actual correction mechanism.

If the backend does not expose historical pose corrections, document the limitation and propose an integration strategy.

## 9.4 Live output

Publish survivor information through a clean interface.

Possible ROS 2 outputs:

- Custom SurvivorTag message.
- SurvivorTagArray.
- MarkerArray for visualisation.
- Diagnostics.
- Registration state.
- Tag update events.

Choose the exact interfaces after inspecting the repository.

A tag should include:

```text
survivor_id
map_position
grid_position
grid_cell
position_covariance
detection_confidence
localisation_quality
tag_status
capture_timestamp
update_timestamp
map_revision
registration_revision
```

Use ROS 2 standard messages where possible and custom messages only where necessary.

The GCS team will consume the published results. Do not build a complete GCS.

---

# PART 10 — ERROR HANDLING AND FAILURE CASES

The algorithm must explicitly handle the following cases.

| Failure | Required behaviour |
|---|---|
| No valid depth | Reject localisation measurement |
| Too few valid depth pixels | Mark measurement unreliable |
| Background dominates bounding box | Reject ambiguous depth estimate |
| Multiple plausible depth clusters | Do not blindly select one |
| Detection timestamp stale | Reject or mark stale |
| Depth and RGB timestamps mismatch | Reject or apply validated synchronisation |
| Historical pose unavailable | Do not confirm tag |
| Camera extrinsic missing | Do not publish a confirmed map position |
| Camera orientation unknown | Mark localisation unreliable |
| Grid registration unavailable | Do not publish a confirmed official cell |
| Position near cell boundary | Maintain provisional status if uncertain |
| Duplicate detection | Update existing track |
| Detector tracking ID changes | Attempt spatial track association |
| SLAM correction | Recalculate affected observations |
| Survivor leaves camera view | Retain track with last known status |
| Invalid calibration | Prevent confirmed localisation |
| Position outside official grid | Report out-of-grid status |

Every failure should have a diagnostic reason.

Avoid silently returning zero coordinates or fabricated positions.

---

# PART 11 — COMPUTATIONAL EFFICIENCY

The onboard Raspberry Pi 5 16GB will simultaneously run several other subsystems.

These include:

- 2D LiDAR SLAM.
- Obstacle avoidance.
- Path planning.
- Human detection.
- Thermal processing.
- Other mission software.
- ROS 2 communication.

Therefore, this subsystem must be lightweight.

Requirements:

1. Do not run another human detection model.
2. Process only relevant depth regions wherever practical.
3. Avoid unnecessary full-resolution image copies.
4. Reuse aligned depth frames.
5. Use asynchronous ROS 2 processing where appropriate.
6. Keep survivor history bounded and configurable.
7. Avoid expensive global recomputation unless a correction requires it.
8. Use efficient matrix and vector operations.
9. Avoid unnecessary point-cloud generation.
10. Measure CPU, RAM, latency and dropped messages.

Do not assume that the Hailo accelerator will accelerate ordinary C++ geometry, SLAM or coordinate transformations.

Do not claim the complete system meets real-time requirements until measured under simultaneous subsystem load.

---

# PART 12 — TESTING AND VALIDATION

The system must be tested in stages.

## 12.1 Layer 1 tests — Robust geometry

Test:

- Valid person depth.
- Invalid depth.
- Zero depth.
- NaN and infinite values.
- Background wall behind a person.
- Multiple depth clusters.
- Partial occlusion.
- Standing person.
- Sitting person.
- Prone person.
- Person close to the camera.
- Person near the supported depth limit.

Measure:

- 3D position error.
- Ground-reference position error.
- Valid measurement rate.
- False depth association rate.
- Processing latency.

## 12.2 Layer 2 tests — Coordinate transformation

Use known camera, body and map transforms.

Test:

- Zero rotation.
- 90-degree yaw.
- Different roll and pitch.
- Different camera mounting orientations.
- Non-zero camera translation.
- Historical pose interpolation.
- Missing historical pose.
- Timestamp mismatch.
- Pose correction.

Compare output with analytically expected positions.

## 12.3 Layer 3 tests — Grid registration

Test:

- Known origin.
- Known translation.
- Known rotation.
- Axis inversion.
- Different cell boundaries.
- Negative coordinates.
- Out-of-grid points.
- Points exactly on grid boundaries.
- Invalid registration.

Verify that the correct official grid cell is produced.

## 12.4 Layer 4 tests — Temporal consistency

Test:

- Repeated detections of one person.
- Different detector tracking IDs for the same person.
- Noisy depth measurements.
- Inconsistent observations.
- Independent viewpoints.
- Correlated consecutive frames.
- Cell-boundary uncertainty.
- Two nearby survivors.
- Temporary occlusion.

Measure:

- Correct cell rate.
- Duplicate track rate.
- False confirmation rate.
- Position error.
- Time to confirmation.

## 12.5 Layer 5 tests — Corrections and output

Test:

- SLAM map correction.
- Historical pose correction.
- Grid-registration update.
- Recalculation of survivor positions.
- Reassignment to a different cell.
- Live output consistency.
- Missing correction event.
- Restart and state-recovery behaviour, if persistence is implemented.

## 12.6 End-to-end testing

Use a controlled arena with known survivor reference positions.

Evaluate:

- Correct grid-cell assignment rate.
- Position error.
- False positive tag rate.
- False confirmed tag rate.
- Duplicate survivor rate.
- End-to-end latency.
- CPU utilisation.
- RAM usage.
- Thermal behaviour.
- Dropped frames.
- Long-duration stability.

Do not claim hardware validation based only on simulation.

Clearly label every result as:

- Unit tested.
- Simulation tested.
- Recorded-data tested.
- Hardware tested.
- Flight validated.

---

# PART 13 — IMPLEMENTATION PLAN

Implement incrementally. Do not attempt to generate the complete subsystem in one pass.

## Phase 0 — Repository and architecture audit

0.1 Inspect the existing repository structure.

0.2 Identify programming languages, ROS 2 distribution, build system and existing dependencies.

0.3 Identify existing camera, SLAM and telemetry interfaces.

0.4 Identify the actual RealSense driver and version.

0.5 Determine whether detections originate from the D435i or a separate camera.

0.6 Identify existing custom ROS 2 messages.

0.7 Identify existing coordinate frame conventions.

0.8 Prepare a proposed subsystem architecture.

0.9 Identify missing information and hardware dependencies.

0.10 Do not write implementation code until the architecture proposal has been reviewed.

## Phase 1 — Interface definitions

1.1 Define the detection input interface.

1.2 Define RGB-D input interface.

1.3 Define camera calibration input.

1.4 Define timestamp and clock-domain conventions.

1.5 Define drone pose input interface.

1.6 Define coordinate frame conventions.

1.7 Define survivor tag output interface.

1.8 Define error and diagnostics interface.

1.9 Document all assumptions.

1.10 Build interface-level tests.

## Phase 2 — Robust survivor geometry

2.1 Create the geometry-processing module.

2.2 Implement bounding-box validation.

2.3 Implement depth-frame validation.

2.4 Implement invalid depth rejection.

2.5 Implement configurable region extraction.

2.6 Implement robust depth filtering.

2.7 Implement depth-cluster selection.

2.8 Implement camera-coordinate reconstruction.

2.9 Implement ground-reference estimation interface.

2.10 Implement measurement uncertainty output.

2.11 Implement failure states.

2.12 Write unit tests.

2.13 Test with recorded RGB-D data.

## Phase 3 — Coordinate transformations

3.1 Implement camera-to-body transform handling.

3.2 Implement pose buffering.

3.3 Implement historical pose lookup.

3.4 Implement timestamp validation.

3.5 Implement pose interpolation where valid.

3.6 Implement camera-to-map transformation.

3.7 Implement covariance transformation.

3.8 Implement missing-pose handling.

3.9 Implement diagnostics.

3.10 Write coordinate transformation tests.

## Phase 4 — NIDAR grid registration

4.1 Inspect official NIDAR grid specification.

4.2 Document grid origin.

4.3 Document axis directions.

4.4 Document cell dimensions.

4.5 Document indexing convention.

4.6 Implement map-to-grid transform.

4.7 Implement registration validity checks.

4.8 Implement grid-cell calculation.

4.9 Implement boundary handling.

4.10 Implement out-of-grid handling.

4.11 Write registration and cell-calculation tests.

## Phase 5 — Temporal consistency and tracking

5.1 Define survivor track data structures.

5.2 Implement track creation.

5.3 Implement observation association.

5.4 Implement duplicate handling.

5.5 Implement repeated-observation fusion.

5.6 Implement outlier rejection.

5.7 Implement uncertainty updates.

5.8 Implement candidate/provisional/confirmed states.

5.9 Implement grid-boundary uncertainty handling.

5.10 Implement track expiry and retention rules.

5.11 Write unit and integration tests.

## Phase 6 — SLAM correction handling

6.1 Inspect the actual SLAM backend's correction interface.

6.2 Define correction-event adapter.

6.3 Store the historical data needed for recalculation.

6.4 Implement affected-observation identification.

6.5 Recalculate corrected map positions.

6.6 Update survivor tracks.

6.7 Recalculate official grid cells.

6.8 Implement revision identifiers.

6.9 Publish corrected tags.

6.10 Test loop-closure and correction scenarios.

## Phase 7 — Live output and integration

7.1 Implement SurvivorTag message or equivalent.

7.2 Implement SurvivorTagArray.

7.3 Implement visualisation output if required.

7.4 Implement diagnostics.

7.5 Connect detection input.

7.6 Connect D435i input.

7.7 Connect timestamped pose input.

7.8 Connect SLAM correction events.

7.9 Connect GCS-facing output interface.

7.10 Test the complete data flow.

## Phase 8 — Performance and validation

8.1 Benchmark geometry processing.

8.2 Benchmark coordinate transformation.

8.3 Benchmark tracking and fusion.

8.4 Measure CPU utilisation.

8.5 Measure memory usage.

8.6 Measure thermal performance.

8.7 Measure message latency.

8.8 Measure dropped messages.

8.9 Test under simultaneous onboard workload.

8.10 Conduct recorded-data end-to-end testing.

8.11 Conduct controlled hardware testing.

8.12 Document limitations and readiness.

---

# PART 14 — RULES FOR CLAUDE CODE

These instructions are mandatory.

## 14.1 Work one task at a time

Do not implement all phases at once.

Follow the numbered implementation plan.

Complete one task, test it and report its result before moving on.

## 14.2 Inspect before modifying

Before writing code:

- Inspect the existing repository.
- Read existing documentation.
- Inspect existing interfaces.
- Reuse existing code where appropriate.
- Identify conflicts before changing files.

Never overwrite working modules unnecessarily.

## 14.3 No invented hardware assumptions

Do not invent:

- Camera mounting angle.
- Camera extrinsics.
- Camera intrinsics.
- Drone altitude source.
- Pose availability.
- ROS 2 distribution.
- SLAM correction API.
- Official grid origin.
- Grid orientation.
- Grid indexing.
- Sensor timestamps.

If a value is unknown, expose it as configuration or request clarification.

Do not silently use placeholder values in a flight-ready implementation.

## 14.4 No fabricated test results

Never claim:

- A module works on the Raspberry Pi unless tested there.
- A camera works unless actually tested.
- A driver works with our exact sensor unless verified.
- A tag is accurate without ground-truth evaluation.
- A system is real-time without benchmarking.

Clearly distinguish simulated, recorded-data and hardware results.

## 14.5 Documentation

Maintain:

- README.md.
- Architecture documentation.
- Interface documentation.
- Configuration documentation.
- Test documentation.
- Known limitations.
- Build and execution instructions.

Document every assumption.

## 14.6 Code quality

Use:

- Modular architecture.
- Clear class and function responsibilities.
- Meaningful variable names.
- Error handling.
- Logging.
- Configurable parameters.
- Unit tests.
- ROS 2 best practices.
- Efficient memory management.

Avoid unnecessary abstraction and excessive dependencies.

## 14.7 Completion report

After every task, report:

1. Task number.
2. Files created or modified.
3. Functionality implemented.
4. Dependencies added.
5. Tests executed.
6. Test results.
7. Known limitations.
8. Remaining assumptions.
9. Next task recommendation.

Do not automatically continue through multiple major phases without review.

---

# PART 15 — FIRST INSTRUCTION TO CLAUDE CODE

Start with PHASE 0 ONLY.

Your first task is to audit the existing repository and prepare a detailed implementation proposal for the Survivor Localisation and Grid Tagging subsystem.

Do not write implementation code yet.

Specifically:

1. Inspect the complete repository.
2. Identify its current architecture.
3. Identify the ROS 2 distribution and build system, if present.
4. Identify existing camera and RealSense integrations.
5. Identify existing detection interfaces.
6. Identify the source of drone pose.
7. Identify existing SLAM integration and map correction capabilities.
8. Identify existing coordinate frames and conventions.
9. Identify dependencies that can be reused.
10. Identify missing information.
11. Propose the folder structure for this subsystem.
12. Propose ROS 2 node boundaries and message interfaces.
13. Propose a dependency-aware implementation order.
14. Identify computational risks for Raspberry Pi 5 16GB.
15. Prepare the first set of tasks for approval.

Return the audit report and wait for approval.

Do not build the human detector, flight controller, obstacle avoidance, path planning or complete GCS.

**Your goal is to implement a reliable, computationally efficient and testable survivor-localisation and grid-tagging subsystem—not to independently build the entire drone.**