Synthetic PulseEKKO GPS generated 20260612_115120

Known LINE00 start:
lat = 47.5489431
lon = 8.538713

LINE00 direction guide:
lat = 47.5486042
lon = 8.5384317

LINE00 distance from HD FINAL POSITION:
42.240 m

Line spacing:
0.250 m

Geometry:
- local coordinates use LINE00 start as (0, 0)
- LINE00 moves toward the supplied guide direction
- perpendicular line spacing is forced toward southeast
- even lines follow LINE00 direction
- odd lines are reversed to create a zigzag/serpentine grid
- each line length is parsed from its own .HD FINAL POSITION
- per-trace coordinates are linearly interpolated along each line

Outputs:
- one GPS CSV per line
- PulseEkko_synthetic_gps_master.csv
- PulseEkko_synthetic_gps_endpoints.csv
