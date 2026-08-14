#!/usr/bin/env python
"""Optimize O2-O3 coupling using images of a collimated alignment laser.

The 532 nm laser is injected through the back of the pentaband dichroic while
O3 moves along the tilted optical axis.

Shepherd 11/2022
"""

import numpy as np
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.core import StageDevice
from scipy import ndimage

from opm_v2.engine.debug_printing import info, warning
from opm_v2.hardware.PicardShutter import PicardShutter


def _snap_autofocus_image(mmc: CMMCorePlus) -> np.ndarray:
    """Acquire one autofocus image with one bounded camera recovery attempt.

    Returns
    -------
    numpy.ndarray
        Acquired camera image.

    Raises
    ------
    RuntimeError
        If the camera fails after recovery.
    """
    try:
        return mmc.snap()
    except (RuntimeError, TimeoutError) as first_error:
        warning(
            "O2/O3 AUTOFOCUS CAMERA SNAP FAILED",
            f"Camera error: {first_error}",
            "Clearing the Micro-Manager camera sequence and retrying once",
        )
        if mmc.isSequenceRunning():
            mmc.stopSequenceAcquisition()
        mmc.clearCircularBuffer()
    try:
        image = mmc.snap()
    except (RuntimeError, TimeoutError) as retry_error:
        raise RuntimeError(
            "O2/O3 autofocus camera snap failed after one recovery attempt"
        ) from retry_error
    info("O2/O3 AUTOFOCUS CAMERA RECOVERED", "Snap received after one retry")
    return image


def calculate_focus_metric(image: np.ndarray) -> float:
    """Calculate a maximum-intensity focus metric.

    Parameters
    ----------
    image : numpy.ndarray
        Camera image to score.

    Returns
    -------
    float
        Maximum filtered intensity.
    """
    # calculate focus metric
    image[image > 2**16 - 10] = 0
    image[image < 100] = 0
    kernel = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    focus_metric = np.max(ndimage.minimum_filter(image, footprint=kernel))

    # return focus metric
    return focus_metric


def find_best_O3_focus_metric(
    mmc: CMMCorePlus,
    shutter_controller: PicardShutter,
    O3_stage_name: str,
    verbose=False,
) -> float:
    """Optimize the position of O3 with respect to O2.

    Using a maximum intensity metric, this function first performs a rough search to find
    a guess at the best focus, then re-runs with a fine search to determine the best focus.

    Parameters
    ----------
    mmc : CMMCorePlus
        Shared Micro-Manager core instance.
    shutter_controller : PicardShutter
        Alignment-laser shutter controller.
    O3_stage_name : str
        Micro-Manager device name for the O3 stage.
    verbose : bool
        Whether to print autofocus progress.

    Returns
    -------
    float
        Automatically determined O3 focus position.

    """
    experiment_focus_device = str(mmc.getFocusDevice())
    o3_stage = mmc.getDeviceObject(O3_stage_name, StageDevice)
    start_um = float(np.round(o3_stage.getPosition(), 2))
    if verbose:
        print(
            f"Experiment focus device remains {experiment_focus_device}; "
            f"O3 stage {O3_stage_name} starts at {start_um} um"
        )

    def _measure(positions_um: np.ndarray, label: str) -> np.ndarray:
        if verbose:
            print(f"Starting {label} alignment.")
        metrics = np.zeros(positions_um.shape[0])
        for index, position_um in enumerate(positions_um):
            o3_stage.setPosition(float(position_um))
            o3_stage.wait()
            image = _snap_autofocus_image(mmc)
            metrics[index] = calculate_focus_metric(image)
            if verbose:
                print(
                    f"Current position: {position_um}; "
                    f"Focus metric: {metrics[index]}"
                )
        return metrics

    shutter_open = False
    completed = False
    try:
        shutter_controller.openShutter()
        shutter_open = True

        rough_positions = np.round(
            np.arange(start_um - 2.5, start_um + 2.5, 0.25),
            2,
        ).astype(np.float64)
        rough_metrics = _measure(rough_positions, "rough")
        if np.max(rough_metrics) < 150:
            print("AF failed on rough align, check shutter!")
            rough_best_um = start_um
        else:
            rough_best_um = float(rough_positions[int(np.argmax(rough_metrics))])
        if verbose:
            print(f"Rough align position: {rough_best_um} vs starting: {start_um}")

        if np.abs(rough_best_um - start_um) >= 2.0:
            if verbose:
                print("Rough focus failed to find better position.")
            best_um = start_um
        else:
            o3_stage.setPosition(rough_best_um)
            o3_stage.wait()
            fine_positions = np.round(
                np.arange(rough_best_um - 0.5, rough_best_um + 0.5, 0.1),
                2,
            ).astype(np.float64)
            fine_metrics = _measure(fine_positions, "fine")
            if np.max(fine_metrics) < 150:
                print("AF failed on fine align, check shutter!")
                fine_best_um = rough_best_um
            else:
                fine_best_um = float(
                    fine_positions[int(np.argmax(fine_metrics))]
                )
            if verbose:
                print(
                    f"Fine align position: {fine_best_um} vs rough: {rough_best_um}"
                )
            if np.abs(fine_best_um - rough_best_um) < 0.5:
                best_um = fine_best_um
            else:
                if verbose:
                    print("Fine focus failed to find better position.")
                best_um = start_um

        o3_stage.setPosition(best_um)
        o3_stage.wait()
        completed = True
        return best_um
    finally:
        if shutter_open:
            try:
                shutter_controller.closeShutter()
            except Exception as exc:
                warning("O2/O3 AUTOFOCUS CLEANUP", f"Could not close shutter: {exc}")
        if not completed:
            try:
                o3_stage.setPosition(start_um)
                o3_stage.wait()
            except Exception as exc:
                warning(
                    "O2/O3 AUTOFOCUS CLEANUP",
                    f"Could not return {O3_stage_name!r} to {start_um} um: {exc}",
                )


def manage_O3_focus(
    O3_stage_name: str,
    verbose: bool = False,
    mmc: CMMCorePlus | None = None,
) -> float:
    """Manage the focus of O3 with respect to fixed O2.

    Parameters
    ----------
    O3_stage_name : str
        Micro-Manager device name for the O3 piezo stage.
    verbose : bool
        Whether to print autofocus progress.
    mmc : CMMCorePlus or None
        Core owned by the active MDA engine. Falls back to the shared instance
        for direct utility use.

    Returns
    -------
    float
        Best O3 focus position, or the original position if focus is not found.
    """
    # get instances of core and shutter controller. Assumes they are already initialized.
    mmc = mmc or CMMCorePlus.instance()
    shutter_controller = PicardShutter.instance()

    # determine optimal O3 stage position
    updated_O3_stage_position = find_best_O3_focus_metric(
        mmc, shutter_controller, O3_stage_name, verbose
    )

    return updated_O3_stage_position
