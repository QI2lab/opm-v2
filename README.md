# opm-v2

[![License: BSD 3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)
[![Python: >=3.12](https://img.shields.io/badge/python-%3E%3D3.12-green)](https://www.python.org/)

![image](https://user-images.githubusercontent.com/26783318/124163887-eb04cb00-da54-11eb-9db8-87c5269d3996.png)
# qi2lab-OPM-v2 | Next generation control for iterative multiplexing and adaptive optics oblique plane microscopy (AO-OPM)


## Overview
This package is the 2nd generation of the Arizona State University Quantitative imaging and Inference Lab (qi2lab) oblique plane microscopy (OPM) with adaptive optics (AO). This generation is built from [Pymmcore-plus](https://github.com/pymmcore-plus/pymmcore-plus) and [Pymmcore-gui](https://github.com/pymmcore-plus/pymmcore-gui) with the integration of custom hardware controls and multi-dimensional acquisition structures. A custom acquisition handler is used to format data to be compatible with our [2nd generation processing software](https://github.com/QI2lab/opm-processing-v2).

The custom acquisition's core functions extend [Pymmcore-gui](https://github.com/pymmcore-plus/pymmcore-gui) capabilities to perform various imaging modalities including *projection imaging*, *mirror scanning*, *stage scanning* and standard *2d imaging*. 

The goal is to utilize [Pymmcore-plus](https://github.com/pymmcore-plus/pymmcore-plus) and [Pymmcore-gui](https://github.com/pymmcore-plus/pymmcore-gui) to the fullest and integrate custom features as needed.

### Features
- Sensorless AO with Imagine Optic Mirao52E and *wavekitpy*
- NIdaq control through *PyDAQmx*
- Fluidics program compatible
- Remote autofocus
- Configurator for setting custom acquisition settings
- Custom multi-dimensional acquisition event strutures
- OME-Zarr data handler using *ome-writers* with the TensorStore backend

## Installation

Clone the repository:

```bash
git clone https://github.com/QI2lab/opm-v2.git
cd opm-v2
```
Install the locked Python environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra test --locked
```

`pymmcore` is intentionally not a direct project dependency. The pinned GitHub
revision of `pymmcore-plus` installs the compatible prebuilt `pymmcore` wheel.

Install Micro-Manager and its device adapters through the pymmcore-plus CLI:

```bash
uv run mmcore install
```

Vendor SDKs are still required for physical NI-DAQ and Imagine Optic hardware.
They are not required for demo mode or automated tests.

## Usage

Launch against Micro-Manager's installed demo devices and simulated external OPM
controllers:

```bash
uv run opm-v2 --demo
```

Launch the physical instrument with an explicit Micro-Manager configuration:

```bash
uv run opm-v2 --config opm_config.json --mm-config OPM_mmgr.cfg
```

Install the official Micro-Manager test adapters and run the behavior suite:

```bash
uv run mmcore install --test-adapters --plain-output
uv run pytest
```

## AO correction initialization

AO initialization is controlled by **Starting correction** and
**Sequential correction**. To start each acquisition from system flat and build
on each completed AO result, select **system flat** and **previous AO result**.
The history resets when a new acquisition starts, including after cancellation.

Sequential correction also offers **use starting correction** (reuse the fixed
acquisition seed for every optimization) and **corresponding position in previous
Z layer** (keep a separate history for each lateral tile or AO grid point).
Previous-Z history restarts at each timepoint; previous-AO history continues
across timepoints within the acquisition. A missing previous-Z result stops AO
instead of silently substituting another correction.

**Last optimized** as the starting correction deliberately imports the saved
result from before acquisition start, including its mirror reference. Changing
these controls does not move the mirror; the selected correction is applied
when AO runs. **AO mode** still determines when optimization runs, and
**AO Metric Accept** still determines which valid fitted updates are accepted.
Existing configurations retain their previous initialization behavior until
the new settings are changed.

The zero-delta probe remeasures the currently accepted correction. If that
measurement improves its recorded score, AO retains the new score and matching
image even when a candidate is rejected. **optimal** retains the highest valid
observed score among the accepted correction, the completed mode's probes, and
the measured candidate; a winning probe is reapplied before committing it.
Failed fits, invalid metrics, and rejected mirror commands still reject the
mode. **zero** compares the candidate to the current sweep's zero probe, while
**accept all** can deliberately retain a lower-scoring valid candidate. Console
output labels trial measurements as **Candidate metric** and reports the
retained value separately as **Retained best metric (historical)**.

After a rejected update, AO restores the retained correction, waits 100 ms,
then measures it again using the same metric, frame averaging, and spot prior.
The console reports **Restored correction measured metric**, the historical
best, and the sweep's zero-probe score. This diagnostic never changes the
retained correction or acceptance threshold, even if its score is higher.
Each check is saved immediately in `ao_restore_checks.zarr/check_NNNNNN` in
the AO results directory: `image` contains the averaged image (or O3 stack);
attributes contain the fresh score, validity, historical and zero scores,
coefficients, requested actuator voltages, and controller readbacks before
and after imaging. Invalid fits are saved too. Completed checks remain
available if AO is stopped. Readbacks confirm controller state, not optical
wavefront recovery. Every rejected mode adds one metric acquisition; this
also adds illumination exposure. Camera/readback/save failures stop AO with
the retained correction already restored.

For initial alignment, select **AO mode: optimize now** and **DAQ mode: 2d**.
**Alignment: refocus every N modes** enables O2–O3 autofocus between groups
of N attempted Zernike modes (0 / Off disables it). The count continues across
iterations; no extra autofocus runs after the final mode. This option is
disabled for acquisition AO, projection mode, and the O3-stack metric.
Autofocus uses the current AO illumination and exposure with the retained
mirror correction applied. It does not open or close any shutter. STOP is
checked between autofocus frames and stage moves; an interrupted scan returns
O3 to its pre-scan position. Refocus positions are saved immediately in
`ao_alignment_focus.zarr` and included in final AO metadata. The historical
metric acceptance threshold is unchanged. Frame averaging no longer implicitly
triggers autofocus; use this explicit interval instead.

## Stage Explorer polygon regions

Rectangles and polygons can be sent to MDA using **ROIs with Grid Sub-Sequence**.
OPM replans tiles using its scan footprint and overlap settings, preserving the
polygon boundary and any per-ROI coverslip calibration. Tiles wholly outside
the polygon are skipped; boundary tiles capture their full field of view.
Projection, mirror, timelapse, and stage scans support polygon regions.
Stage mode accepts one region per acquisition and retains full-length hardware
scan segments that intersect it. The Stage Explorer Scan action also accepts
polygons for live previews.

## Simultaneous laser pairs

**Interleaved** acquisition requires identical positive exposure times for all
enabled channels. Unequal exposures are rejected before acquisition because
camera exposure edges clock the laser and mirror waveform; changing exposure
between channels can break that synchronization. Disabled channels are ignored.

In **OPM modes/settings**, choose **simultaneous 488/637**,
**simultaneous 561/637**, or **simultaneous 488/561** to acquire one combined
**488 + 637**, **561 + 637**, or **488 + 561**
channel. Set the selected pair's exposure times equal; each
laser retains its own power setting. The GUI selects only those two lasers.
Projection, mirror, stage, and timelapse imaging retain their existing geometry
and acquisition settings. Switch back to **interleaved** for individual channels.
Interleaved mirror scans capture every enabled color at each mirror position
before moving on, using equal exposure times.
AO continues to use its independently selected excitation channel.

For live imaging, the Micro-Manager **Channel** dropdown has **488 + 637**,
**561 + 637**, and **488 + 561** presets in `OPM_mmgr.cfg`.
Reload that configuration to expose all three presets.
Each enables its laser pair without changing their current power setpoints;
use the existing individual laser-power controls. Live preview uses the current
camera exposure. Acquisitions use the OPM power and exposure settings.
Legacy `simultaneous` configurations migrate to `simultaneous 488/637`.

## O3-stack adaptive optics

Select **gauss_3d_o3** in the AO metric dropdown. Enable the separate alignment
beam manually: the existing Picard alignment shutter stays closed, imaging
laser powers/TTLs are off, and both scan mirrors remain stationary. The usual
AO exposure applies; **Averaged frames** is the number of frames at each plane.

The O3 controls default to a **4 µm full range**, **0.5 µm steps** (nine planes
from −2 to +2 µm about the initial O3 position), a **30° O2–O3 angle**, and
**0.115 µm/pixel** in remote-image space. Camera Y is the tilt direction. Range
divided by step must be an even integer, giving a center plane and both endpoints.
O3 returns to the optimization's original position after every stack, including
on a camera failure. Both the scan origin and return target stay fixed throughout
each optimization. Every move checks the stage readback within 0.05 µm; a failed
return is retried once, then aborts AO if still unsuccessful. The measurement
journal records the original position and return verification status. These
checks depend on the stage adapter reporting its position accurately.

STOP is checked between AO camera frames and O3 scan planes. An in-flight
camera call, stage move, or Gaussian fit must finish first. On cancellation,
O3 returns to its original position, the last retained mirror correction is
restored, and DAQ playback stops before normal acquisition teardown. Completed
O3 measurements remain in the journal; an interrupted stack is not scored or
saved as a completed optimization.

**gauss_3d_o3** fits a 3D Gaussian directly in O2 coordinates, without interpolating
the images. With camera coordinates `u,v` and O3 displacement `w`, all in µm,
the transform is `X=u`, `Y=v*cos(30°)+w*sin(30°)`,
`Z=-v*sin(30°)+w*cos(30°)`. This is an O3-axis scan, not the sample-stage shear.
The initial stack defines a fixed crop, center prior, and ellipsoidal core
(two initial fitted sigmas per axis). The score is
`core_fraction² / (sigma_X*sigma_Y*sigma_Z*(1+relative_squared_residual))`.
Core/tail energy uses an independent background estimated from the camera-ROI
border, so the Gaussian cannot hide a secondary peak in its fitted offset.
This is a beam-quality metric; a laser through-focus stack is not automatically
a fluorescence PSF measurement.

During acquisition, each completed cropped stack is written **before fitting**
to `o3_measurements.zarr/measurement_000000/stack` (then `000001`, etc.) in the
run directory. Each measurement's attributes record its mode, delta, iteration,
mirror coefficients, positions, and fit status/rejection reason. These images
remain available when a fit fails or the optimization is interrupted; `pending`
means fitting did not complete. This journal is not overwritten by final saving.

At successful completion, `ao_results.zarr/o3_measured_stacks` also stores every acquired cropped stack,
including rejected candidates, with axes `(measurement, O3 plane, camera Y,
camera X)`. The `o3_stack` metadata records geometry, actual positions, crop,
fitted widths, core/tail fractions, residuals, and invalid-fit reasons. A stack
index links each diagnostic to its raw measurement. The crop and core reset
at the next AO position/run. Ordinary AO modes retain their selected laser.

Nine planes require roughly nine times as many exposures as a single-image
metric, plus O3 settling. The 3D mode needs hardware validation on the actual
beam; synthetic tests check geometry, secondary-peak penalties, and cleanup.

## Utility scripts

Run scripts from the repository root through the locked environment. Every
script exposes its current arguments through `--help`:

```bash
uv run python scripts/stage_positions.py --help
uv run python scripts/ao_positions.py --help
uv run python scripts/ao_grid_mapping.py --help
```

`stage_position_generation.py` and `stage_to_ao_grid_positions.py` remain as
compatible entry points for the corresponding position utilities. Acquisition
inspection and GIF generation accept a TensorStore Zarr v3 array node (for
example, `acquisition.ome.zarr/0`). The AO inspection and plotting utilities
accept the current `ao_results.zarr` layout; the explicitly deprecated parser
is retained only for existing legacy AO result stores.

Hardware-facing utilities are configurable and safe to exercise without
physical devices:

```bash
uv run python scripts/fluidics_triggering.py --simulate --rounds 1
uv run python scripts/push_pull.py --simulate-actuators 52
```

