# All detections of ONE image.
# header.stamp MUST be the IMAGE CAPTURE time, copied from the camera image header (ROS clock), never the time the
# detector finished or the time this message was published. header.frame_id is the optical frame of that image.
std_msgs/Header header
Detection[] detections
