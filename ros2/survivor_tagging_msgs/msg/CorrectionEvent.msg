# SLAM correction (library type survivor_tagging.types.PoseCorrection). One interface, two modes.
# header.stamp = time of the correction (e.g. the loop closure) in the pose clock; header.frame_id = map frame.
std_msgs/Header header
string MODE_HISTORY=history
string MODE_FRAME_CORRECTION=frame_correction
uint32 revision                     # new pose revision, monotonic, >= 1
string mode
builtin_interfaces/Time range_start # history: first / last corrected pose stamp (consistency check)
builtin_interfaces/Time range_end
geometry_msgs/PoseWithCovarianceStamped[] corrected_poses   # history: body pose in the map frame
geometry_msgs/Transform frame_transform                    # frame_correction: T_new_old (corrected <- old map)
