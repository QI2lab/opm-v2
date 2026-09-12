"""Trigger configured OB1 fluidics rounds with optional simulated hardware."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

from opm_v2.hardware.ElveFlow import OB1Controller
from opm_v2.utils.elveflow_control import run_fluidic_program

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "opm_config.json"


def create_controller(config_path: Path, *, simulate: bool) -> OB1Controller:
    """Create the singleton-backed OB1 controller from current configuration.

    Parameters
    ----------
    config_path : Path
        Current OPM JSON configuration.
    simulate : bool
        Whether to use the in-memory OB1 backend.

    Returns
    -------
    OB1Controller
        Configured process-wide controller.

    Raises
    ------
    RuntimeError
        If another OB1 singleton was initialized before this controller.
    """
    config = json.loads(config_path.read_text())
    controller = OB1Controller(
        port=str(config["OB1"]["port"]),
        to_OB1_pin=int(config["OB1"]["to_OB1_pin"]),
        from_OB1_pin=int(config["OB1"]["from_OB1_pin"]),
        simulate=simulate,
    )
    if OB1Controller.instance() is not controller:
        raise RuntimeError("OB1 singleton was initialized before this script")
    return controller


def run_rounds(
    controller: OB1Controller,
    *,
    rounds: int,
    imaging_seconds: float,
    verbose: bool,
) -> int:
    """Run fluidics triggers separated by simulated imaging intervals.

    Parameters
    ----------
    controller : OB1Controller
        Singleton-backed OB1 controller.
    rounds : int
        Number of fluidics rounds.
    imaging_seconds : float
        Delay after each completed trigger.
    verbose : bool
        Whether to print trigger status.

    Returns
    -------
    int
        Number of completed rounds.

    Raises
    ------
    ValueError
        If rounds or the imaging delay is negative.
    """
    if rounds < 0 or imaging_seconds < 0:
        raise ValueError("Rounds and imaging duration cannot be negative")
    for _round in range(rounds):
        run_fluidic_program(verbose=verbose)
        if imaging_seconds:
            time.sleep(imaging_seconds)
    return rounds


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--imaging-seconds", type=float, default=0.0)
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use the in-memory OB1 driver instead of connected hardware",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run configured OB1 fluidics rounds.

    Parameters
    ----------
    argv : Sequence[str] or None
        Command-line arguments, excluding the executable name.

    Returns
    -------
    int
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    controller = create_controller(args.config, simulate=args.simulate)
    completed = run_rounds(
        controller,
        rounds=args.rounds,
        imaging_seconds=args.imaging_seconds,
        verbose=args.verbose,
    )
    print(f"Completed {completed} fluidics rounds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
