# Bytes of the library telemetry codec (survivor_tagging.output). The wire format stays in the library.
std_msgs/Header header
uint8 KIND_RECORD=1                 # one 25-byte tag record
uint8 KIND_SNAPSHOT=2               # 12-byte header + N records + CRC
uint8 kind
uint8[] data
