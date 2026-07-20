"""Run or simulate an Imagine Optic mirror push/pull calibration sequence."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def simulated_push_pull_commands(
    actuator_count: int, amplitude: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate deterministic push/pull actuator commands.

    Parameters
    ----------
    actuator_count : int
        Number of mirror actuators.
    amplitude : float
        Absolute calibration displacement.

    Returns
    -------
    list[tuple[numpy.ndarray, numpy.ndarray]]
        Push and pull vectors for every actuator.

    Raises
    ------
    ValueError
        If the actuator count or amplitude is invalid.
    """
    if actuator_count < 1 or amplitude <= 0:
        raise ValueError("Actuator count and amplitude must be positive")
    commands = []
    for index in range(actuator_count):
        push = np.zeros(actuator_count)
        push[index] = amplitude
        commands.append((push, -push))
    return commands


def run_hardware_calibration(
    wfs_config: Path,
    wfc_config: Path,
    *,
    amplitude: float,
    settle_seconds: float,
) -> int:
    """Run a push/pull calibration against connected Imagine Optic hardware.

    Parameters
    ----------
    wfs_config : Path
        HASO wavefront-sensor configuration.
    wfc_config : Path
        Wavefront-corrector configuration.
    amplitude : float
        Calibration push/pull amplitude.
    settle_seconds : float
        Delay after every mirror command.

    Returns
    -------
    int
        Number of calibrated actuators.

    Raises
    ------
    RuntimeError
        If the vendor WaveKit module is unavailable.
    """
    try:
        import wavekit_py as wkpy
    except ImportError as error:
        raise RuntimeError(
            "Imagine Optic wavekit_py is required for hardware calibration"
        ) from error
    corrector = wkpy.WavefrontCorrector(config_file_path=str(wfc_config))
    manager = wkpy.CorrDataManager(
        haso_config_file_path=str(wfs_config),
        wfc_config_file_path=str(wfc_config),
    )
    calibrated = 0
    try:
        corrector.connect(True)
        corrector.move_to_absolute_positions(np.zeros(corrector.nb_actuators))
        manager.set_calibration_prefs(amplitude)
        specifications = manager.get_specifications()
        for index in range(specifications.nb_actuators):
            preferences: Any = manager.get_actuator_prefs(index)
            if preferences.validity != wkpy.E_ACTUATOR_CONDITIONS.VALID:
                continue
            push, pull = manager.get_calibration_commands(index)
            corrector.move_to_absolute_positions(push)
            time.sleep(settle_seconds)
            corrector.move_to_absolute_positions(pull)
            time.sleep(settle_seconds)
            calibrated += 1
    finally:
        corrector.disconnect()
    return calibrated


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wfs-config", type=Path)
    parser.add_argument("--wfc-config", type=Path)
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--simulate-actuators", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a physical or simulated push/pull calibration.

    Parameters
    ----------
    argv : Sequence[str] or None
        Command-line arguments, excluding the executable name.

    Returns
    -------
    int
        Process exit status.

    Raises
    ------
    SystemExit
        If physical operation is requested without both configuration files.
    """
    args = build_parser().parse_args(argv)
    if args.simulate_actuators is not None:
        commands = simulated_push_pull_commands(args.simulate_actuators, args.amplitude)
        print(f"Generated {len(commands)} simulated actuator command pairs")
        return 0
    if args.wfs_config is None or args.wfc_config is None:
        raise SystemExit("--wfs-config and --wfc-config are required for hardware use")
    calibrated = run_hardware_calibration(
        args.wfs_config,
        args.wfc_config,
        amplitude=args.amplitude,
        settle_seconds=args.settle_seconds,
    )
    print(f"Calibrated {calibrated} actuators")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
