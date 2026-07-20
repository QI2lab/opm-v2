# opm-v2

[![License](https://img.shields.io/pypi/l/opm-v2.svg?color=green)](https://github.com/qi2lab/opm-v2/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/opm-v2.svg?color=green)](https://python.org)

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

