# Conventions: Frames, Time, Units

Applies to all code in `src/survivor_tagging/`. **ASSUMPTION** marks anything not yet verified from hardware,
drivers, teammates or organisers.

## Frames
| Frame | Definition | Status |
|---|---|---|
| `camera` | Optical frame of the camera that produced the depth image. x right, y down, z forward (out of the lens). | ASSUMPTION until verified from the RealSense driver's frame for the stream we actually use |
| `body` | Drone body frame. x forward, y left, z up (ROS REP 103 FLU). | ASSUMPTION until confirmed with the SLAM/FC teammates (Q S1, S2) |
| `map` | SLAM map frame, z up. Origin wherever SLAM puts it, never assumed to be the grid origin. | Needs SLAM teammate (Q S5) |
| `grid` | Official NIDAR 2D grid. Cell size 1 m (rulebook p.19). Origin, axis directions, indexing and labels unknown. | Needs organisers (Q O1) |

- Transforms are named `T_<to>_<from>`: `P_map = T_map_body(t) · T_body_camera · P_camera`, `P_grid = T_grid_map · P_map`.
- `T_body_camera` and `T_grid_map` come only from config (`extr.*`, `grid.*`), never from code constants.
- Cell index convention for tests only: `i = floor(x_grid / cell)`, `j = floor(y_grid / cell)`, 0-based,
  lower-left-closed (ASSUMPTION, `grid.boundary_rule = floor`).

## Rotations and quaternions
- Quaternions are `(w, x, y, z)`, unit norm, Hamilton product, active rotation. `quat_to_matrix(q)` is the
  child-to-parent rotation `R`, so `p_parent = R p_child + t`. Yaw/pitch/roll are rotations about +z/+y/+x,
  positive counter-clockwise when looking down the axis (right-hand rule).
- Covariance perturbations (Layer 2): pose `t' = t + dt`, `R' = Exp(dtheta) R`, with `(dt, dtheta)` in the MAP frame,
  order `(x, y, z, rx, ry, rz)` (ROS PoseWithCovariance fixed-axis convention). Extrinsic: the same in the BODY frame.

## Floor plane (Layer 1, substep 3.1b)
The floor is `z_map = map.floor_z_m`. With the capture-time `T_map_camera = T_map_body(t) T_body_camera = (R, t)`:
```
height_above_floor(p_c) = n . p_c - offset,   n = R^T e_z (third row of R),   offset = floor_z_m - t_z
```
`p_c` is a camera-frame point. Pixels with height < margin are floor. Margin = `l1.floor_margin_m` +
`l1.floor_margin_pose_sigma_k` x sqrt(pose covariance[2, 2]).
Examples (optical frame x right, y down, z forward; floor_z = 0):
- Level forward camera 1.5 m up: `n = (0, -1, 0)`, `offset = -1.5`; `p_c = (0.3, 1.5, 4.0)` (1.5 m below the optical
  axis) has height 0; `p_c = (0, 0, 2)` has height 1.5.
- Same camera pitched 30 degrees down: `n = (0, -cos 30, -sin 30) = (0, -0.866, -0.5)`; the point 2 m straight ahead
  `(0, 0, 2)` has height `1.5 - 2 sin 30 = 0.5`.
- Rolled 20 degrees: `n = (-sin 20, -cos 20, 0)`.

## Map -> grid registration (`grid.py`)
Parameters: `grid.origin_in_map_m` = map coordinates `(ox, oy)` of the official grid origin;
`grid.yaw_map_to_grid_rad` = angle `yaw` from the map +x axis to the grid +x axis, **positive counter-clockwise about
map +z**; `grid.axis_flip_x/y` mirror a grid axis (applied after the rotation); `grid.index_base` is 0 or 1.

```
d     = (x_map - ox, y_map - oy)
g_rot = R(-yaw) d          R(a) = [[cos a, -sin a], [sin a, cos a]]
g     = (s_x g_rot_x, s_y g_rot_y)      s = -1 for a flipped axis, else +1
cell  = (floor(g_x / cell_size), floor(g_y / cell_size)) + index_base
```
Inverse: `p_map = (R(yaw) diag(s) g) + (ox, oy)`. Grid z = map z. Heights are not used for the cell.

Worked example (also a unit test): origin `(2, 1)`, `yaw = +90 deg` (grid +x points along map +y), no flips,
cell 1 m, base 0.
- map `(2, 3)`: `d = (0, 2)`, `R(-90) d = (2, 0)` -> grid `(2.0, 0.0)` -> cell `(2, 0)`; with base 1: `(3, 1)`.
- map `(1, 1)`: `d = (-1, 0)` -> grid `(0, 1)`; with `axis_flip_y = true`: grid `(0, -1)` (outside the grid).
- map `(2.5, 1.5)`: `d = (0.5, 0.5)` -> grid `(0.5, -0.5)` -> cell `(0, -1)`, OUT_OF_GRID.
A point exactly on an edge belongs to the cell whose lower/left edge it is (`floor`): grid x = 3.0 -> i = 3.

## Time
- Timestamps are integer nanoseconds (`*_ns`), always paired with an explicit `clock_domain` string.
- Use the **capture** time of the image/depth frame, never the time a callback ran or inference finished.
- Look up the drone pose **at capture time** by interpolation between bracketing samples. **Never** substitute the
  latest pose. If no valid bracket exists, fail with `POSE_UNAVAILABLE_AT_CAPTURE` / `POSE_GAP_TOO_LARGE`.
- Never compare timestamps from different clock domains unless `clock.camera_to_pose_offset_ns` relates them;
  otherwise fail with `CLOCK_DOMAIN_UNKNOWN`.
- Telemetry uses `update_ms` (u32 ms since library start). This is a display clock only, not used for alignment.

## Units
SI throughout: metres, radians, seconds for durations in config (`*_s`), nanoseconds for timestamps, pixels (`*_px`)
for image coordinates. The unit is in the field name or in the docstring. Telemetry quantises to cm and 1/255.

## Unknown values
Unknown hardware or competition values are either a config parameter that ships unset (`required`, value `None`)
or a question in `teammate_questions.md`. Never a constant in code. A stage with an unset required parameter reports
`CONFIG_MISSING` and must not produce CONFIRMED tags. Tuning defaults are labelled ASSUMPTION and must be validated
experimentally.

## Corrections and revisions (Layer 5, substep 3.2)
- `pose_revision` (int) increases when SLAM re-optimises; a pose is "uncorrected" for an event if its revision did not
  reach the event's revision. A history correction names the time range it re-optimised; the frame correction covers
  [previous correction stamp, event stamp) (half-open, in the pose clock) and is exact only at its end.
- `map_revision` counts applied correction events; `registration_version` counts grid registrations. Both are sent as
  their low byte (u8) in telemetry; the survivor id and tag revision likewise.
- A correction never replaces data in place: stored observations are re-derived from their capture-time pose and the
  current registration. Two events for the same revision with different content are CORRECTION_CONFLICT.

## Score points used in evaluation tables
Points per run = 40 x (correct committed tags) - P x (other committed tags). 40 is rulebook p.19; P is ASSUMED (O2) and
always shown as a column or in the heading. "Committed" means output by the end-of-mission policy, never every
intermediate tag.
