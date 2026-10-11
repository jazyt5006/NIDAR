# NIDAR
# NIDAR AirMouse — Survivor Tagging & Localisation

A theoretical algorithm design for autonomously estimating survivor locations and marking them on the live 2D map during the NIDAR 2026–27 AirMouse mission.

## Project Overview

NIDAR AirMouse requires an autonomous drone to explore an unknown, GPS-denied indoor maze, generate a 2D map, detect survivors, mark their locations on the correct grid cells, and exit autonomously.

This module focuses specifically on **survivor localisation and tagging after detection**. The human-detection algorithm is developed separately by another team member, and mapping is handled by a separate SLAM module.

The tagging system receives survivor detections and spatial information, estimates the survivor's location in the map coordinate frame, determines the corresponding 1 m × 1 m reference-grid cell, and publishes the marker for live display.

## Objectives

- Localise detected survivors using geometric methods and available depth information.
- Transform survivor locations from camera coordinates to map coordinates.
- Register the map to the competition's reference grid.
- Assign survivor locations to the correct grid cells.
- Maintain consistent tags across repeated observations and SLAM pose corrections.
- Handle uncertain measurements without producing unjustifiably precise locations.
- Publish survivor markers and status updates during flight without manual intervention.

## Algorithm Architecture

The proposed system consists of a geometric localisation baseline followed by five competition-specific layers.

### Baseline — Depth-Assisted Geometric Localisation

Estimate a 3D survivor position using the detector's bounding box, calibrated camera information, and corresponding depth measurements.

The baseline provides the initial spatial estimate. It does not, by itself, guarantee correct grid assignment.

### Layer 1 — Robust Survivor Geometry

- Extract valid depth measurements from the detected person's image region.
- Reject invalid depth values and likely background measurements.
- Estimate a representative survivor position.
- Maintain localisation-quality information alongside the position estimate.

### Layer 2 — Time-Aligned Coordinate Transformation

- Transform the camera-relative position into the map coordinate frame.
- Use camera calibration and the camera-to-drone transformation.
- Use the drone pose corresponding to the image capture timestamp.
- Account for the relevant altitude and attitude information when required.

### Layer 3 — NIDAR Grid Registration

- Establish the transformation between the SLAM map and the official competition grid.
- Account for grid origin, orientation, and coordinate conventions.
- Convert the estimated survivor position into its corresponding 1 m × 1 m grid cell.

### Layer 4 — Temporal Consistency and Uncertainty

- Associate repeated observations with the same survivor track.
- Combine consistent observations to improve position estimates.
- Reduce tag jitter and prevent duplicate markers.
- Maintain uncertainty information and avoid confirming unreliable cell assignments.

### Layer 5 — Correction Handling and Live Map Output

- Associate survivor observations with their capture timestamps and relevant poses.
- Recalculate affected survivor positions when the SLAM trajectory is corrected.
- Update grid assignments when necessary.
- Publish survivor markers and status information for live ground-control display.

## Processing Pipeline

```text
Human Detection Module
        |
        v
Bounding Box + Timestamp
        |
        v
RGB Image + Depth Data
        |
        v
Robust Depth Extraction
        |
        v
3D Survivor Position Estimate
        |
        v
Time-Aligned Camera-to-Map Transform
        |
        v
Map-to-Reference-Grid Registration
        |
        v
Grid-Cell Assignment
        |
        v
Temporal Tracking + Uncertainty Checks
        |
        v
SLAM Correction Handling
        |
        v
Live Survivor Marker on 2D Map
```

## Inputs

The tagging module is designed to receive:

- **Detection data:** bounding box, detection confidence, and image timestamp.
- **Camera data:** RGB image, corresponding depth measurements, and camera calibration.
- **Pose data:** time-aligned drone pose and required coordinate transforms.
- **Grid registration:** the transformation between the SLAM map and the competition reference grid.
- **Optional additional evidence:** segmentation masks, tracking IDs, LiDAR range measurements, or observations from other viewpoints when available.

## Outputs

For each survivor track, the system is intended to provide:

- Estimated survivor position in map coordinates.
- Assigned competition-grid coordinate.
- Localisation uncertainty or confidence.
- Track identity and observation timestamp.
- Tag status, such as candidate, confirmed, or requiring further evidence.
- Updated marker information for the live 2D map.

Detection confidence and localisation confidence are treated as separate quantities.

## Fallback and Supporting Methods

The primary baseline is depth-assisted geometric localisation. The following methods remain possible supporting approaches:

- **Monocular ground-plane geometry:** estimates range from camera geometry when its assumptions are satisfied.
- **LiDAR range corroboration:** provides additional range evidence when a relevant return can be reliably associated with the detection.
- **Multi-view triangulation:** combines observations from different viewpoints when camera poses and correspondence are reliable.

These methods are conditional fallbacks or refinements. They do not replace the main tagging pipeline, and no fallback should force a precise tag when the available measurements are unreliable.

## Integration with the Mapping Module

Mapping and survivor tagging are separate modules.

The mapping module is responsible for generating the map and estimating the drone pose. The tagging module consumes these outputs through a defined interface.

Candidate mapping backends under evaluation include:

- Cartographer 2D
- SLAM Toolbox

The tagging architecture is intended to remain independent of the selected backend. The implementation will require an adapter that provides the necessary pose information and handles historical pose updates after loop closure.

## Intended Technology Stack

The implementation is expected to investigate:

- **ROS 2:** communication between sensors, detection, mapping, and tagging components.
- **OpenCV and camera-model utilities:** image geometry and camera calibration.
- **Depth-camera software:** depth measurements and 3D point generation.
- **ROS 2 tf2:** time-aware coordinate transformations.
- **C++ or Python:** implementation, subject to integration and performance requirements.
- **Raspberry Pi 5:** likely onboard computer; final hardware and performance remain to be validated.

These are proposed implementation technologies, not a claim that the complete system has already been implemented.

## Theoretical Validation Plan

Before physical deployment, the algorithm should be evaluated using known simulated survivor positions and controlled inputs.

Planned checks include:

1. Verify 3D position recovery from known image and depth measurements.
2. Test invalid depth, background contamination, and outlier rejection.
3. Verify camera-to-map and map-to-grid transformations.
4. Evaluate grid assignment near cell boundaries.
5. Test repeated detections, track association, and duplicate suppression.
6. Test tag updates following mapping pose corrections.
7. Measure correct-cell rate, position error, tag stability, and update latency.
8. Later evaluate the integrated implementation on the selected onboard computer and sensors.

## Current Status

**Design status:** Theoretical architecture specified.

**Implementation status:** ROS 2 nodes and executable algorithm have not yet been completed.

**Hardware status:** Final camera selection and sensor configuration remain open.

**Mapping status:** Cartographer and SLAM Toolbox are under separate evaluation.

**Validation status:** Simulation and hardware validation remain pending.

The algorithm's thresholds and performance claims will be established through later simulation and experiments rather than assumed at the design stage.

## Mission Alignment

The design targets the NIDAR AirMouse requirements for autonomous survivor localisation, correct reference-grid tagging, and live map updates during flight. The competition awards 40 marks per correctly detected and correctly tagged survivor, up to six survivors.

The priority is not merely to produce a plausible position estimate, but to place each survivor in the correct grid cell while preserving consistency between the detection, drone pose, map, and reference grid.

---

**Project:** NIDAR 2026–27 — AirMouse  
**Module:** Survivor Localisation and Tagging  
**Document status:** Theoretical Design  
**Version:** 1.0
