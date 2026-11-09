# One person detection. The pixel box refers to the image whose capture time is the header.stamp of the
# enclosing DetectionArray. No depth here: the position comes from the depth image (survivor_tagging library).
int32 frame_index                   # detector frame counter (diagnostics only)
float32[4] bbox_xyxy                # x_min, y_min, x_max, y_max in pixels of the source image
float32 confidence                  # detector confidence 0..1 (NOT localisation quality)
int32 class_id
string class_name
int32 tracking_id                   # detector track id; -1 = none (the detector does not provide one)
uint32 source_image_width           # size of the image the box refers to, pixels
uint32 source_image_height
