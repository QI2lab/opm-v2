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
- Custom data handler using *zarr* + *tensorstore*  

## Installation

Clone the repository:

```bash
git clone https://github.com/QI2lab/opm-v2.git
cd opm-v2
```
Create and activate conda environment:
```
conda create -n opm-v2 python=3.12
conda activate opm-v2
```
Install opm_v2 package:
```
pip install .
```
Some hardware control modules may require vendor SDKs:
- NI-DAQ (nidaqmx)
- Imagine Optic Mirao52E (wavekit_py)

## Usage

