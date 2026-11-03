# AO grid geometry and Balmer regression evidence

The AO grid covers the **requested acquisition region**, expressed in
depth-normalized laboratory scan X and stage Y, in micrometers. A requested
3 by 3 grid partitions each requested axis into three equal intervals; the nine
AO targets are the Cartesian product of the interval centers. The number and
length of acquisition trajectories do not limit AO sampling density.

Tile assignment uses the area of intersection between each grid cell and the
tile's full lateral footprint in that same coordinate frame. A stage-scan tile
footprint contains the complete trajectory plus the centered oblique camera
projection along X, and the centered camera width along Y. Reverse trajectories
use the same physical footprint. A deterministic lowest grid index resolves
equal overlap areas. One assigned correction applies to every frame and channel
of the tile; a grid correction may remain unused.

Requested boundaries and camera-extended support are distinct. The requested
region fixes grid boundaries. The calibrated camera footprint is used for tile
overlap, not to replace or expand those boundaries. Camera-extended support may
also differ depending on whether it describes the commanded trajectory or the
finite set of retained frame centers.

## Coordinate and position semantics

Let `L` be depth-normalized laboratory scan X, `X` be commanded physical stage X,
`Z` be physical stage Z, `Zref` be the scan reference Z, `s` be the configured
sample-depth stage sign, and `theta` be the sheet angle. All spatial operands
below are in micrometers:

```
X = L + s * (Z - Zref) * cot(theta)
L = X - s * (Z - Zref) * cot(theta)
```

The existing fitted coverslip/surface determines target Z at each grid center;
that Z already includes its scheduled sample-depth offset. Transform laboratory
scan X to physical stage X once, after evaluating the surface. Do not add depth
compensation again to an already corrected stage position. Cell boundaries and
tile footprints remain in the declared laboratory frame across depths.

The legacy image metadata field `Stage.x_pos` describes a planned,
depth-normalized laboratory scan coordinate. It is generated from the scan
origin, frame index, and configured step. It is not a measured physical stage
readback. Preserve legacy fields and add explicit commanded physical stage and
laboratory coordinate fields, frame/unit/transform descriptions, and separately
labeled readbacks where actually measured. Convert units before rounding:
three decimal places in micrometers correspond to six in millimeters.

## Read-only Balmer evidence

The compact regression fixture is
[`tests/fixtures/balmer_ao_geometry.json`](../tests/fixtures/balmer_ao_geometry.json).
Its provenance records the inspected metadata files. No image arrays or fused
AO overlay were used as placement evidence.

The authoritative plan is
`E:/Balmer_mouse/stage/balmer_tissue_stage.ome.zarr/zarr.json`, under
`attributes.opm_v2.configuration.resolved_spatial_plan.regions[0]`.
It records region `ROI_b4f3`, flat coverslip Z = 2409.360 um, and requested bounds:

| Axis | Minimum (um) | Maximum (um) |
| --- | ---: | ---: |
| Laboratory scan X | 4622.8955078125 | 5962.9521484375 |
| Stage Y | -91.3197021484375 | 1011.993408203125 |

The recorded acquisition configuration specifies a 30 degree sheet angle,
0.115 um sample pixel size, and acquisition camera crop of 1900 by 256 pixels.
The configuration's top-level default camera height of 512 is not the height
used by this acquisition. Camera extents are 25.4957878874 um along scan X,
218.500 um along Y, and 14.720 um along laboratory Z.

Requested-region cell boundaries and centers, rounded only for this table:

| Axis | Cell boundaries (um) | Centers (um) |
| --- | --- | --- |
| Laboratory scan X | 4622.896, 5069.581, 5516.267, 5962.952 | 4846.238, 5292.924, 5739.609 |
| Stage Y | -91.320, 276.451, 644.222, 1011.993 | 92.566, 460.337, 828.108 |

Each `time_0_z_{0,1,2,3}_ao_grid_results` directory contains 14 planned tile
origins in `exp_stage_positions.json` and nine historical AO targets in
`ao_stage_positions.json`. The four depths are 0.000, 11.760, 23.520, and
35.280 um. The physical Z commands are 2409.360, 2421.120, 2432.880, and
2444.640 um. Every depth retains laboratory scan starts 4622.896 and
5265.176 um; physical stage X shifts by depth times `cot(30 degrees)`.

At depth zero the historical AO targets have X = 4783.470, 4944.040, and
5104.610 um. These are the interior quarter points of the **tile-origin range**
4622.896 through 5265.176 um, confirming that the old grid did not cover the
requested region. Historical Y targets likewise sample the interior quarter
points of the origin range.

Raw acquisition position-group metadata (`0`, `7`, `14`, `21`, `28`, `35`, `42`,
and `49` / `zarr.json`) records 1744 scan planes per tile, a 0.400 um planned
step, and no excess frames. Their last laboratory frame centers are 5320.096
and 5962.376 um. These are generated coordinates, not measured readbacks.
The requested plan and calibrated splitter independently reconstruct commanded
laboratory endpoints 5320.672 and 5962.952 um at the stored command precision.

With the centered camera projection, commanded scan support is approximately
4610.148 through 5975.700 um. Support swept by the recorded frame centers is
approximately 4610.148 through 5975.124 um. Y support is -200.570 through
1121.240 um. None of these support bounds equals the requested region or the
range of tile origins.

## Implementation and offline checks

The planner carries exact requested bounds and calibrated tile rectangles into
each depth's AO event. Literal position plans without region bounds explicitly
use the union of full acquired footprints. A zero-width requested axis expands
to camera support on that axis, with the original requested bounds and that
expansion recorded separately. Geometry calculations retain full precision;
stored coordinates round to three decimal places in micrometers, and ASI
millimeter commands round to six.

The existing surface fit evaluates Z at each laboratory cell center. When a
single trajectory cannot constrain a surface slope, the declared coverslip or
configured slope supplies that axis. Physical target X is then transformed
once using the evaluated Z and retained scan reference. The depth offset is
already included in Z. Tile count, trajectory splitting, frame count, channel
ordering, scan speed and AO correction-state scheduling remain as planned.

Legacy `ao_stage_positions.json`, `exp_stage_positions.json`, and `Stage.x_pos`,
`y_pos`, `z_pos` remain available. New `ao_grid_geometry.json` records bounds,
cell rectangles, tile footprints, global position indices, selected AO indices,
boundary provenance, support definitions, and the whole-tile correction scope.
Its AO position records and image `Stage` metadata add
`physical_stage_command_um`, `laboratory_scan_um`, `stage_readback_um`, and
`coordinate_transform`. Image readbacks are null because those positions are
planned. AO grid moves record actual readbacks separately. Optimization metadata
labels the legacy `stage_position` as measured and adds
`physical_stage_readback_um` and explicit units.

`tests/unit/test_position_tools.py` independently checks equal cells, long and
reverse trajectories, overlapping boundary tiles and ties, surface and depth
transforms, and the four-depth Balmer geometry. The fully mocked
`tests/integration/test_ao_grid_balmer.py` executes all 36 AO measurements and
56 tile assignments, checks command/readback separation, and verifies correction
state inheritance without creating a microscope controller. Event-planning
tests check that a single correction is applied before the complete tile's
frames and channels. Acquired datasets are read only; no fused overlay is used.
