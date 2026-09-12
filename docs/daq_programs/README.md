# OPM DAQ program trace and publication figures

This trace distinguishes the four user-facing acquisition modes from the four
low-level NI-DAQ programs. The figures are generated from the production
`OPMNIDAQ.generate_waveforms()` implementation, using simulated hardware and a
representative three-color acquisition.

## Mode map

| User-facing path | Event plan | Low-level DAQ program | Acquired axes |
| --- | --- | --- | --- |
| Standard live | Update DAQ directly from the live-mode property | `2d` | Continuous preview, one active laser |
| Projection live | Update DAQ directly from the live-mode property | `projection` | Continuous preview, one active laser |
| Timelapse | For each stage position and static image-galvo plane: `Mirror-Move`, `DAQ`, then time/channel image events | `2d` | `p, z, t, c` |
| Mirror | One `DAQ` action per prepared sequence; interleaved acquisitions are ordered plane then channel | `mirror`; `2d` when scan range is zero | `t, p, z, c` when interleaved; `t, p, c, z` otherwise |
| Projection | One DAQ sweep is retriggered for each channel exposure | `projection` | `t, p, c` |
| Stage | For every physical tile: prepare the DAQ and stage controller, then acquire while the stage moves continuously at constant speed | `stage` plus the external ASI/PLC program | `t, p, z, c` |
| Sensorless AO | AO selects its own acquisition path independently of the main imaging mode | `2d` or `projection` | Metric images, not main acquisition axes |

The GUI dispatches the four acquisition planners in
[`OPMEventBuilder.build()`](../../src/opm_v2/engine/setup_events.py#L3950).
Live Standard and Projection map to `2d` and `projection` in
[`OPMAppController.live_mode_callback()`](../../src/opm_v2/_app.py#L1923).
The canonical DAQ action validates these four low-level modes in
[`create_daq_event()`](../../src/opm_v2/engine/opm_custom_events.py#L459).

## Exact DAQ programs

Let `C` be the number of enabled lasers, `R` the image-mirror scan range,
`Δx` the requested scan step, `Z = max(1, round(R / Δx))`, `t_exp` the camera
exposure, and `f_s = 10 kHz` the internal AO sample rate.

| Program | Digital output | Analog output | Clocking and trigger behavior |
| --- | --- | --- | --- |
| `2d` | `2C` samples. Enabled lasers are high in channel order on alternating camera edges; the final sample is low when blanking is enabled. | Both galvos remain at their configured neutral voltages. | Camera EXPOSURE OUT is change-detected on both edges. It clocks the laser DO task. |
| `mirror` | `2CZ` samples, with the same interleaved channel pattern repeated at every plane. | The image galvo steps through `Z` plane centers symmetric about neutral; the projection galvo remains neutral. The last sample returns the image galvo to the first plane. | The camera edge-derived PFI2 signal clocks both DO and AO continuously. The image galvo advances after the falling edge of the last channel at each plane. |
| `projection` | `2C` samples, identical laser sequencing to `2d`. | During every exposure, ao0 ramps from `Vimg,neutral − R·kimg/2` to `Vimg,neutral + R·kimg/2`, while ao1 ramps from `+R·kproj/2` to `−R·kproj/2`. A final sample resets both ramps to their first values. `N_AO = floor(t_exp f_s) + 1`. | Every rising camera edge starts a finite internally clocked AO task. The AO task is retriggerable, so each channel gets a complete sweep. |
| `stage` | `2C` samples, repeated while the stage moves continuously. Channels are interleaved across successive exposures. | Both galvos remain at neutral. | The ASI stage/PLC starts the camera and moves the stage at constant speed. Camera EXPOSURE OUT still controls the laser digital output. The planned stage velocity is `Δx / (C t_exp)`, so the stage advances one plane spacing every `C` exposures. Exposure time and scan speed are chosen so motion during an exposure does not produce visible blur. |

The waveform formulas and arrays are implemented in
[`OPMNIDAQ.generate_waveforms()`](../../src/opm_v2/hardware/OPMNIDAQ.py#L718),
and the NI task routing is implemented in
[`OPMNIDAQ.program_daq_waveforms()`](../../src/opm_v2/hardware/OPMNIDAQ.py#L924).
At runtime, a DAQ custom action stops and clears existing tasks, selects the
program, regenerates and programs it in
[`OPMEngineV2.setup_event()`](../../src/opm_v2/engine/opm_engine.py#L1171), then
starts playback in [`OPMEngineV2.exec_event()`](../../src/opm_v2/engine/opm_engine.py#L1474).
For stage scans, ASI motion starts only after the camera sequence is armed in
[`OPMEngineV2.post_sequence_started()`](../../src/opm_v2/engine/opm_engine.py#L1288).

### Camera timing used in the waveform figure

The timing panels assume an ORCA-Fusion BT in Fast Scan mode. Mirror Sweep and
Stage scan use a 512-line region with a requested exposure of 12 ms; Projection
uses the full 2304-line sensor with a requested exposure of 150 ms. The
[Hamamatsu instruction manual](https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/static/sys/en/manual/C15440-20UP,-20UP01_IM_En.pdf)
specifies a 4.867647 µs line time and rounds the requested exposure up to the
nearest supported interval. For Mirror Sweep and Stage scan, this gives:

- actual frame period: 12.0018 ms;
- 512-line readout interval: 2.4922 ms;
- global EXPOSURE OUT high time: 9.5095 ms; and
- EXPOSURE OUT low time between frames: 2.4922 ms.

For Projection, this gives:

- actual frame period: 150.0044 ms;
- 2304-line readout interval: 11.2151 ms;
- global EXPOSURE OUT high time: 138.7894 ms; and
- EXPOSURE OUT low time between frames: 11.2151 ms.

The camera manual defines global EXPOSURE OUT as the interval when all sensor
lines expose simultaneously. For the illustrated 0.4 µm spacing across three
channels, the stage speed is 11.11 µm/s. The laser is therefore on during only
0.106 µm of stage travel per frame, rather than across the full 0.133 µm frame
pitch.

## Implementation observations

- The projection-galvo sweep is centered on `0 V`, not on the configured
  `projection_mirror_neutral_v`; the final reset likewise returns to the
  zero-centered sweep start. This follows the current production code exactly.
- For mirror scans, `R` is treated as the total sampled footprint, while the
  centers of `Z` planes span `(Z − 1)Δx`.
- With laser blanking disabled, each enabled laser line remains high for the
  complete programmed cycle; enabled lasers are therefore simultaneous rather
  than sequential.
- The plotted stage position is a schematic constant-speed trajectory. Images are
  acquired throughout the motion, with channels interleaved at exposure
  times. The timing panel repeats the three-channel sequence four times and marks
  every 0.4 µm of travel. Each 12 ms frame advances 0.133 µm, but the 9.51 ms
  global-exposure pulse limits illuminated travel to 0.106 µm. Camera
  exposure output, digital laser, and analog galvo traces follow the production
  timing program.

## Figure files

- `figures/opm_daq_waveforms.{svg,pdf,png}` — the mirror-sweep, projection, and
  stage-scan timing programs, including three-color laser sequencing and both
  galvos.
- `figures/opm_mode_map.{svg,pdf,png}` — Mirror Sweep, Projection, and Stage scan
  acquisition sequences and DAQ behavior.
- `figures/opm_daq_routing_mirror_sweep.{svg,pdf,png}` — camera timing routed to
  the digital laser control and analog mirror-position outputs.
- `figures/opm_daq_routing_projection.{svg,pdf,png}` — camera timing routed to
  the digital laser control and analog mirror-sweep outputs.
- `figures/opm_daq_routing_stage_scan.{svg,pdf,png}` — ASI/PLC camera triggering,
  digital laser control, and analog outputs that hold the mirrors steady.

SVG and PDF are editable vector outputs. PNG files are exported at 600 dpi. The
Okabe–Ito palette remains distinguishable under common color-vision deficiencies,
and text stays editable in SVG/PDF.

Regenerate every output from the repository root:

```powershell
.venv/Scripts/python.exe docs/daq_programs/generate_figures.py
```

The generator asserts the expected production array shapes and reset states
before writing any figures.

## Draft manuscript captions

**Figure 1 — OPM acquisition timing.** Camera exposure, laser illumination,
mirror position, and stage position for Mirror Sweep, Projection, and Stage scan.
The Mirror Sweep example uses three laser colors, a 4 µm range, 1 µm plane
spacing, a 512-line region, and a 12 ms exposure. Projection uses the full
2304-line sensor and a 150 ms exposure; its global EXPOSURE OUT signal is high
for 138.79 ms and low for 11.22 ms. The Stage scan example shows four repeated
three-channel groups during constant-speed motion. All three channel exposures
occur within each 0.4 µm of travel. With a 512-line region and 12 ms exposure,
the global EXPOSURE OUT signal is high for 9.51 ms and low for 2.49 ms, limiting
illuminated stage travel to 0.106 µm per frame. The stage-position trace is
illustrative because the stage controller, rather than the DAQ, produces this
motion.

**Figure 2 — Mapping acquisition modes to DAQ behavior.** Mirror Sweep,
Projection, and Stage scan use distinct acquisition sequences. In Stage scan, the
stage moves continuously at constant speed while channels are interleaved across
successive exposures.

**Figure 3 — Mirror Sweep signal routing.** The camera exposure signal enters the
digital input at PFI0. The resulting timing signal controls the digital laser
output and the analog mirror-position output. The imaging mirror steps between
planes while the projection mirror remains steady.

**Figure 4 — Projection signal routing.** The camera exposure signal enters the
digital input at PFI0. It controls the digital laser output and starts one analog
mirror sweep during each camera exposure.

**Figure 5 — Stage scan signal routing.** The stage controller starts the camera
and moves the stage continuously at constant speed. Short camera exposures sample
interleaved channels throughout the motion without visible motion blur.
The camera exposure signal enters the digital input at PFI0 and controls the
digital laser output, while the analog outputs hold both mirrors steady.
