# Survivor Tagging: Questions for the Team and Organisers

Hi all. I'm building the survivor localisation and grid tagging library (detection + depth -> map position -> official
grid cell -> survivor tag). Scoring depends on the *correct grid cell*, so I need a few facts from each of you.
For each question there is the default I will code against until you answer. If a default is wrong, tell me early:
it changes the design, not just a number. Short answers are fine.

---

## Detection teammate

**D1. What exactly does each detection message contain?** (bbox format: pixels or normalised; xyxy or xywh;
confidence; class; track id; image timestamp; frame id; message/topic type)
- Why: I project the box into depth. Wrong format or missing timestamp = wrong position.
- Default ASSUMPTION: pixel xyxy, confidence 0-1, optional track id, and the *image capture* timestamp copied from the
  camera frame (not inference-finish time).

**D2. Which camera do detections come from: D435i colour stream or the Pi Camera Module 3 Wide?**
- Why: D435i depth can be used directly only if detections are on the D435i colour image with aligned depth. Another
  camera needs a full cross-camera calibration.
- Default ASSUMPTION: D435i colour stream, with depth aligned to colour. If it's the Pi camera, I will reject detections
  (DEPTH_FRAME_MISMATCH) until we calibrate.

**D3. Which timestamp do you put on detections, and from which clock?**
- Why: I must look up the drone pose at the moment the image was taken, not when you finished inference.
- Default ASSUMPTION: the camera frame's header stamp, passed through unchanged.

**D4. Does the model reliably detect the competition dummies (lying, sitting, partially hidden)?** Any test numbers?
- Why: if dummies are often missed or flicker, I need more tolerant tracking; if postures are mostly prone, ground
  position estimation changes.
- Default ASSUMPTION: unknown recall; I treat every single detection as a CANDIDATE only.

**D5. Detection rate (Hz) and latency on the Pi?**
- Why: sizes my pose buffer and staleness limits.
- Default ASSUMPTION: unknown; staleness limit 1 s (configurable).

## SLAM teammate

**S1. Is the pose full 3D (x, y, z, roll, pitch, yaw) or planar (x, y, yaw)? If planar, where do altitude and
roll/pitch come from (FC, rangefinder, D435i IMU)?**
- Why: a tilted camera with a planar-only pose gives wrong floor positions.
- Default ASSUMPTION: planar SLAM + altitude and attitude from another source, which must be identified. Until then,
  tags cannot become CONFIRMED.

**S2. Pose topic or tf frames, and the publish rate?** (e.g. `map` -> `base_link` via tf2, or a PoseStamped topic)
- Why: I interpolate between poses around each image time; the rate sets the maximum allowed gap.
- Default ASSUMPTION: tf2 `map` -> `base_link`; rate unknown, gap limit configurable.

**S3. Which clock are the pose timestamps in, and is it the same clock as the camera?**
- Why: comparing two unrelated clocks gives the wrong pose.
- Default ASSUMPTION: both use ROS time on the same Pi; offset 0 is *not* assumed and must be checked.

**S4. Does the backend expose loop-closure corrections, i.e. can I get corrected historical poses (or a correction
event)? Which SLAM package and version?**
- Why: after a loop closure, earlier survivor positions move. I store observations so I can recompute them, but only if
  I can get corrected poses.
- Default ASSUMPTION: no correction interface; positions are computed once and a limitation is logged.

**S6. Is the drone's height above the floor part of the pose (map z relative to the floor), and is the arena floor
flat and level? (ties to S1)**
- Why: floor removal in Layer 1 needs the floor plane in the camera frame (pose height + floor height). A sloped or
  uneven floor, or a pose without a reliable z, makes the floor plane wrong.
- Default ASSUMPTION: the floor is one flat, level plane at a configured `map.floor_z_m`; floor removal stays off
  until the height source is confirmed.

**S5. Where is the `map` origin set (take-off point?), and is the map ever re-initialised mid-flight?**
- Why: needed for registering the map to the official grid.
- Default ASSUMPTION: origin at take-off, never reset.

## Controls team (simulation)

**C1. Which simulator are you using (Gazebo version, Isaac, AirSim, custom)? Does it render RGB-D and give
ground-truth survivor positions?**
- Why: I want to test the library end-to-end with known truth before hardware.
- Default ASSUMPTION: I provide a synthetic harness; your simulator plugs in later.

**C2. How do you want to feed my library: direct Python calls per frame, recorded files, or ROS 2 topics?**
- Why: decides whether I ship a Python API first or ROS 2 nodes first.
- Default ASSUMPTION: direct Python calls (`process(detections, depth, intrinsics)`, `add_pose(pose)`), ROS 2 later.

**C3. Can the simulator give camera intrinsics, camera mount pose and drone pose with timestamps on one clock?**
- Why: same inputs the real system needs.
- Default ASSUMPTION: yes, all on simulator time.

## Hardware owner

**H1. OS image and Python version on the Pi 5 (Raspberry Pi OS Bookworm? Ubuntu 24.04?), and numpy version?**
- Why: library compatibility.
- Default ASSUMPTION: Python >= 3.10 with numpy available.

**H2. ROS 2 distribution (Humble, Jazzy, ...) and whether it runs natively or in Docker?**
- Why: decides the wrapper package and message types.
- Default ASSUMPTION: unknown; the core library has no ROS dependency, so nothing is blocked yet.

**H3. RealSense status: librealsense and realsense-ros versions, firmware, and does depth stream reliably on the Pi?
Which resolution and fps?**
- Why: depth quality and frame rate drive everything in Layer 1.
- Default ASSUMPTION: aligned depth to colour available; resolution and fps read from the driver.

**H4. Camera mounting: exact position (x, y, z in m from the body origin) and angles (forward, tilted down by how many
degrees?). Is there a CAD drawing or measurement?**
- Why: a 5 degree error in tilt at 3 m range is about 0.26 m of floor error, a quarter of a cell.
- Default ASSUMPTION: none. Extrinsics are required config; without them, no tag can be CONFIRMED.

**H5. Which telemetry radio and real data rate to the GCS?**
- Why: sizes the tag messages.
- Default ASSUMPTION: tens of kbps; I designed a 25-byte tag record plus periodic snapshots (< 1 kbps).

## Organisers (NIDAR)

**O1. Official grid definition: origin (which corner), axis directions, cell size, numbering/labels (e.g. A1 or
(row, col), 0- or 1-based), entry cell, and do walls lie on grid lines?**
- Why: scoring is per correct cell. A wrong origin or axis flip makes every tag wrong.
- Default ASSUMPTION: none. Registration is required config; without it no official cell is CONFIRMED. For tests only:
  1 m cells, origin at lower-left, 0-based (i, j).

**O2. Are wrong or extra survivor tags penalised? Is there a limit on tags?**
- Why: decides whether we submit uncertain (provisional) tags at mission end or only confirmed ones.
- Default ASSUMPTION: unknown; end-of-mission policy is a config switch, default "confirmed only".

**O3. How must the survivor map be delivered to the judges (format, at what moment, from the GCS screen)?**
- Why: our output format must match.
- Default ASSUMPTION: GCS shows tags live; no file format assumed.

**O4. What do dummies look like (size, clothing, posture), and can more than one be in the same cell?**
- Why: detection and duplicate-merging thresholds.
- Default ASSUMPTION: human-sized; two survivors can be in adjacent cells; same-cell survivors are kept separate only
  if more than 0.5 m apart (configurable).

**O5. If a survivor lies across two grid cells, which point of the person decides the cell?**
- Why: a prone person is 1.7 m long and can span two 1 m cells; scoring is per cell.
- Default ASSUMPTION: the cell containing the centre of the body.

**O6. How are two survivors in the same grid cell scored: one tag for the cell, or one tag per survivor? Are two
tags in one cell allowed?**
- Why: decides whether two nearby tracks stay two tags or are combined into one cell tag at the end of the mission.
- Default ASSUMPTION: one tag per survivor (two tracks stay two tags, even in the same cell).
