# Full survivor tag (library type survivor_tagging.types.Tag plus the Layer 5 publication fields).
# header.stamp = update time of the tag; header.frame_id = map frame of position_map.
std_msgs/Header header

uint8 STATUS_CANDIDATE=0
uint8 STATUS_PROVISIONAL=1
uint8 STATUS_CONFIRMED=2
uint8 STATUS_UPDATED=3
uint8 STATUS_UNRELIABLE=4

uint32 survivor_id                  # stable id (never reused)
uint32 tag_revision                 # +1 on every confirmed-cell change
uint32 map_rev                      # pose / correction revision the position is based on
uint32 reg_rev                      # grid registration version the cell is based on
uint8 status
bool removed                        # survivor merged away or dropped

bool has_position
geometry_msgs/Point position_map    # metres, map frame
float64[4] covariance_map_xy_m2     # 2x2 row-major (xx, xy, yx, yy), map frame
bool has_grid_position
float64[2] position_grid_xy_m       # official grid frame, metres
bool has_cell
int32 cell_i
int32 cell_j
string cell_label                   # official label; empty = not known (organiser question O1)
CandidateCell[] candidate_cells     # most probable cells first
bool out_of_grid

float32 sigma_horizontal_m          # NaN = unknown
float32 boundary_distance_m         # distance to the nearest cell edge; NaN = unknown
float32 max_detection_confidence    # detector confidence, NOT localisation quality
float32 localisation_quality        # probability of the reported cell
uint32 observation_count
float32 effective_independent_count # number of view bins
int32 extrinsic_version             # -1 = none
uint32 cell_changes                 # cell changes after the first confirmation

builtin_interfaces/Time first_seen_time
builtin_interfaces/Time last_seen_time
builtin_interfaces/Time capture_time        # latest contributing image capture
bool has_confirmed_time
builtin_interfaces/Time confirmed_time

string[] warning_codes              # library FailureReason values
