"""
Sensorless adaptive optics and tools.

2024/12 DPS: initial work
2025/09/05 SJS: updates to synchronize with opm_custom_events and opm_config
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import sleep

import numpy as np
import zarr
from numpy.typing import NDArray
from pymmcore_plus import CMMCorePlus
from scipy.fft import dct
from scipy.ndimage import center_of_mass, laplace
from scipy.optimize import curve_fit

from opm_v2.engine.debug_printing import info, warning
from opm_v2.utils.ao_cancel import AOCancelled, check_ao_stop
from opm_v2.utils.ao_o3_stack import METRIC_NAME as O3_STACK_METRIC
from opm_v2.utils.ao_o3_stack import O3StackMetric, O3StackSettings
from opm_v2.utils.ao_policy import AOCorrectionHistory
from opm_v2.utils.metadata_serialization import json_metadata
from opm_v2.utils.position_tools import ao_grid_positions, nearest_ao_grid_indices

try:
    from opm_v2.hardware.AOMirror import AOMirror
    from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ
except Exception:
    AOMirror = None
    OPMNIDAQ = None

DEBUGGING = True

# Metric global parameters
MAXIMUM_MODE_DELTA = 0.5
DEFUALT_PSF_RADIUS_PX = 3
DEFUALT_SIGN_FIGS = 6


@dataclass
class GaussianSpotPrior:
    """Spot location fixed from the starting image of one AO optimization."""

    center: tuple[float, float] | None = None
    threshold: float = 1000.0
    center_radius_px: float = 16.0


@dataclass(frozen=True)
class AOGridResult:
    """Optimized states on the coarse AO grid, ordered by grid position."""

    modal_coefficients: NDArray
    actuator_positions: NDArray
    reference_state: str

# Modes to optimize lists
focusing_modes = [2, 7, 14, 23]
spherical_modes = [7, 14, 23]
spherical_modes_first = [
    7,
    14,
    23,
    3,
    4,
    5,
    6,
    8,
    9,
    10,
    11,
    12,
    13,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
]
all_modes = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
]
stationary_modes = [
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
]
# all_modes = [0,1,3,4,5,6,8,9,10,11,12,13,17,18,19,20,31,30,29,28,27,26,25,24,23,22]
all_modes = [2, 3, 4, 5, 6, 7, 2, 10, 11, 14, 15, 2, 16, 30, 31]
mode_names = [
    "Vert. Tilt",
    "Horz. Tilt",
    "Defocus",
    "Vert. Asm.",
    "Oblq. Asm.",
    "Vert. Coma",
    "Horz. Coma",
    "3rd Spherical",
    "Vert. Tre.",
    "Horz. Tre.",
    "Vert. 5th Asm.",
    "Oblq. 5th Asm.",
    "Vert. 5th Coma",
    "Horz. 5th Coma",
    "5th Spherical",
    "Vert. Tetra.",
    "Oblq. Tetra.",
    "Vert. 7th Tre.",
    "Horz. 7th Tre.",
    "Vert. 7th Asm.",
    "Oblq. 7th Asm.",
    "Vert. 7th Coma",
    "Horz. 7th Coma",
    "7th Spherical",
    "Vert. Penta.",
    "Horz. Penta.",
    "Vert. 9th Tetra.",
    "Oblq. 9th Tetra.",
    "Vert. 9th Tre.",
    "Horz. 9th Tre.",
    "Vert. 9th Asm.",
    "Oblq. 9th Asm.",
]


# TODO: Implement Fourier ratio metric

# -------------------------------------------------#
# Modal AO optimization
# -------------------------------------------------#


def get_metric(
    image: NDArray,
    metric_to_use: str,
    psf_radius_px: int = DEFUALT_PSF_RADIUS_PX,
    *,
    gaussian_prior: GaussianSpotPrior | None = None,
) -> float:
    """Calculate the selected metric on the provided image.

    Parameters
    ----------
    image : NDArray
        image to measure metric on
    metric_to_use : str
        descriptor for metric to use
    psf_radius_px : int
        Radius of PSF in pixels
    gaussian_prior : GaussianSpotPrior or None
        Fixed spot location and center bounds for the Gaussian metric.

    Returns
    -------
    float
        calculated metric
    """
    metrics_str_list = [
        "DCT",
        "localize_gauss_2d",
        "gauss_2d",
        "brightness",
        "fourier_ratio",
    ]
    if metric_to_use not in metrics_str_list:
        print(
            "------- SensorlessAO -------"
            f"ERROR: AO metric {metric_to_use} not supported"
        )
        return np.nan

    try:
        if "DCT" in metric_to_use:
            metric = metric_shannon_dct(image, psf_radius_px) * 100
        elif "localize_gauss_2d" in metric_to_use:
            metric = metric_localize_gauss2d(image)
        elif "gauss_2d" in metric_to_use:
            metric = (
                metric_gauss2d(image)
                if gaussian_prior is None
                else metric_gauss2d(
                    image,
                    image_center=gaussian_prior.center,
                    threshold=gaussian_prior.threshold,
                    center_radius_px=gaussian_prior.center_radius_px,
                )
            )
        elif "brightness" in metric_to_use:
            metric = metric_brightness(image, percentile=None)
        elif "fourier_ratio" in metric_to_use:
            metric = metric_laplacian_variance(image)  # TODO
    except Exception as e:
        print(
            "------- SensorlessAO -------"
            f"\nERROR: AO metric {metric_to_use} calculation failed with exception:"
            f"\n{e}"
        )
        metric = np.nan

    return metric


def _prepare_camera_for_ao_snap(mmc: CMMCorePlus) -> str:
    """Put the active camera in internal-trigger mode for software AO snaps.

    Returns
    -------
    str
        Active Micro-Manager camera label.

    Raises
    ------
    RuntimeError
        If Micro-Manager has no active camera.
    """
    camera_label = mmc.getCameraDevice()
    if not camera_label:
        raise RuntimeError("No active camera is configured for adaptive optics")
    camera = str(camera_label)
    if mmc.isSequenceRunning():
        mmc.stopSequenceAcquisition()

    recovery_properties = (
        ("Trigger", "NORMAL"),
        ("TriggerPolarity", "POSITIVE"),
        ("TRIGGER SOURCE", "INTERNAL"),
    )
    for property_name, value in recovery_properties:
        if not mmc.hasProperty(camera, property_name):
            continue
        allowed = tuple(mmc.getAllowedPropertyValues(camera, property_name))
        if allowed and value not in allowed:
            continue
        mmc.setProperty(camera, property_name, value)
        mmc.waitForDevice(camera)

    mmc.clearCircularBuffer()
    return camera


def _recover_ao_camera_snap(
    mmc: CMMCorePlus,
    opm_nidaq: OPMNIDAQ,
    camera_error: Exception,
) -> None:
    """Reset camera snap state and re-arm the unchanged AO DAQ waveform.

    Raises
    ------
    RuntimeError
        If the retained DAQ waveform cannot be reprogrammed or restarted.
    """
    scan_type = opm_nidaq.scan_type
    warning(
        "AO CAMERA SNAP FAILED",
        f"DAQ mode: {scan_type}",
        f"Camera error: {camera_error}",
        "Resetting internal trigger state and retrying once",
    )
    opm_nidaq.stop_waveform_playback()
    _prepare_camera_for_ao_snap(mmc)

    # Stopping and restarting valid tasks preserves every acquisition
    # parameter, including the two-galvo projection waveform.  If the tasks
    # were invalidated, rebuild them from the parameters already held by the
    # DAQ object rather than falling back to a different scan type.
    if not opm_nidaq.programmed():
        opm_nidaq.clear_tasks()
        opm_nidaq.generate_waveforms()
        opm_nidaq.program_daq_waveforms()
    if not opm_nidaq.programmed():
        raise RuntimeError(
            f"Could not reprogram the {scan_type!r} AO waveform after camera error"
        )

    opm_nidaq.start_waveform_playback()
    if not opm_nidaq.running():
        raise RuntimeError(
            f"Could not restart the {scan_type!r} AO waveform after camera error"
        )


def _snap_ao_image(
    mmc: CMMCorePlus,
    opm_nidaq: OPMNIDAQ | None,
) -> NDArray:
    """Snap one AO image, recovering the camera once on a device failure.

    Returns
    -------
    numpy.ndarray
        Camera image from the initial snap or its single retry.

    Raises
    ------
    RuntimeError
        If recovery fails or the camera's retry also fails.
    TimeoutError
        If no DAQ recovery controller was supplied and the initial snap times out.
    """
    try:
        return mmc.snap()
    except (RuntimeError, TimeoutError) as first_error:
        if opm_nidaq is None:
            raise
        _recover_ao_camera_snap(mmc, opm_nidaq, first_error)
    try:
        image = mmc.snap()
    except (RuntimeError, TimeoutError) as retry_error:
        raise RuntimeError(
            "AO camera snap failed after one recovery attempt"
        ) from retry_error

    info(
        "AO CAMERA RECOVERED",
        f"DAQ mode: {opm_nidaq.scan_type}",
        "Internal-trigger snap received after one retry",
    )
    return image


def acquire_metric_image(
    mmc: CMMCorePlus,
    metric_to_use: str,
    num_averaged_frames: int = 1,
    opm_nidaq: OPMNIDAQ | None = None,
    *,
    gaussian_prior: GaussianSpotPrior | None = None,
    o3_sampler: O3StackMetric | None = None,
    stop_requested=None,
) -> tuple[NDArray, float]:
    """Acquire images and calculate a consistent adaptive-optics metric.

    Parameters
    ----------
    mmc : CMMCorePlus
        Shared Micro-Manager core instance controlling the camera.
    metric_to_use : str
        Name of the image metric to compute.
    num_averaged_frames : int
        Number of camera frames to average.
    opm_nidaq : OPMNIDAQ or None
        Active AO DAQ controller. When supplied, one camera-device failure is
        recovered without changing the configured scan mode.
    gaussian_prior : GaussianSpotPrior or None
        For Gaussian AO, initialize the center from the starting (averaged)
        image once, then reuse it for every frame, probe, and candidate.
    o3_sampler : O3StackMetric or None
        Fixed through-focus scan and 3D metric state for this AO run.
    stop_requested : callable or None
        Thread-safe cancellation predicate checked between camera frames.

    Returns
    -------
    tuple[numpy.ndarray, float]
        Averaged camera image and its metric value.
    """
    def snap():
        check_ao_stop(stop_requested)
        return _snap_ao_image(mmc, opm_nidaq)

    check_ao_stop(stop_requested)
    if o3_sampler is not None:
        stack, metric = o3_sampler.acquire(
            mmc, snap, num_averaged_frames, stop_requested=stop_requested
        )
        check_ao_stop(stop_requested)
        if not np.isfinite(metric):
            warning("AO O3 STACK INVALID", o3_sampler.diagnostics[-1]["reason"])
        return stack, metric
    if num_averaged_frames > 1:
        images = [snap() for _ in range(num_averaged_frames)]
        image_stack = np.stack(images, axis=0).astype(np.float32)
        image = np.mean(image_stack, axis=0)
    else:
        image = snap()
        images = [image]

    metric_options = {}
    if metric_to_use == "gauss_2d" and gaussian_prior is not None:
        if gaussian_prior.center is None:
            gaussian_prior.center = _initial_gaussian_center(
                image, gaussian_prior.threshold
            )
        metric_options["gaussian_prior"] = gaussian_prior
    metric = float(
        np.mean([get_metric(im, metric_to_use, **metric_options) for im in images])
    )

    check_ao_stop(stop_requested)
    return image, metric


def metric_from_fit(a: float, b: float, c: float, delta: float) -> float:
    """Return the optimal metric based on the delta and quadratic fit.

    Parameters
    ----------
    a : float
        x2 term of quadratic
    b : float
        x term of quadratic
    c : float
        constant term of quadratic
    delta : float
        Optimal delta to evaluate quadratic at

    Returns
    -------
    float
        metric at the given delta and fit parameters
    """
    return a * delta**2 + b * delta + c


def round_to_sigfigs(x: float, signif_figs: int = DEFUALT_SIGN_FIGS) -> float:
    """Round value to given significant figures.

    Parameters
    ----------
    x : float
        value to round
    signif_figs : int, optional
        number of significant figures, by default DEFUALT_SIGN_FIGS

    Returns
    -------
    float
        given value rounded to significant figures
    """
    if x == 0:
        return 0
    else:
        digits = signif_figs - int(np.floor(np.log10(abs(x)))) - 1
        return np.round(x, digits)


def _restore_ao_correction(
    mirror: AOMirror, coefficients: NDArray, daq: OPMNIDAQ
) -> None:
    """Restore the accepted correction, stopping acquisition if restoration fails.

    Parameters
    ----------
    mirror : AOMirror
        Active deformable mirror.
    coefficients : NDArray
        Last accepted modal coefficients.
    daq : OPMNIDAQ
        Active AO acquisition controller.

    Raises
    ------
    RuntimeError
        If the accepted correction cannot be restored.
    """
    try:
        if not mirror.set_modal_coefficients(coefficients.copy()):
            raise RuntimeError("Mirror rejected the accepted correction")
    except Exception as exc:
        daq.stop_waveform_playback()
        raise RuntimeError(
            "Failed to restore last accepted AO mirror correction"
        ) from exc


def _measure_restored_correction(
    mirror, mmc, daq, metric_to_use, num_averaged_frames,
    metric_options, record, save_dir_path,
) -> None:
    """Measure the restored state without changing the accepted AO history.

    Write each check immediately to a separate journal, including invalid fits.
    Controller positions are readbacks, not independent optical feedback.
    """
    check_ao_stop(metric_options.get("stop_requested"))
    time.sleep(0.1)  # Same additional settling delay as candidate measurements.
    mirror._update_current_state()
    record = {
        **record,
        "timestamp_unix_s": time.time(),
        "num_averaged_frames": num_averaged_frames,
        "metric_name": metric_to_use,
        "requested_voltage": mirror.last_requested_voltage.copy(),
        "reported_voltage_before": mirror.current_voltage.copy(),
    }
    sampler = metric_options.get("o3_sampler")
    if sampler is not None:
        sampler.context = {**record, "kind": "restoration_verification"}
    image, metric = acquire_metric_image(
        mmc, metric_to_use, num_averaged_frames, opm_nidaq=daq, **metric_options
    )
    mirror._update_current_state()
    record.update(
        measured_metric=float(metric),
        metric_valid=bool(np.isfinite(metric)),
        reported_voltage_after=mirror.current_voltage.copy(),
    )
    record["max_voltage_readback_error"] = float(max(
        np.max(np.abs(record[key] - record["requested_voltage"]))
        for key in ("reported_voltage_before", "reported_voltage_after")
    ))
    if save_dir_path is not None:
        save_dir_path.mkdir(parents=True, exist_ok=True)
        root = zarr.open_group(str(save_dir_path / "ao_restore_checks.zarr"), mode="a")
        group = root.create_group(f"check_{record['step_index']:06d}")
        group.create_array("image", data=np.asarray(image))
        group.attrs.update(json_metadata(record))
    print(
        f"    Restored correction measured metric: {metric:.6f}; "
        f"historical best: {record['historical_best_metric']:.6f}; "
        f"zero probe: {record['zero_probe_metric']:.6f}; "
        f"max actuator readback error: {record['max_voltage_readback_error']:.6g}"
    )


def run_ao_optimization(
    exposure_ms: float,
    channel_states: tuple[bool],
    metric_to_use: str = "DCT",
    daq_mode: str = "projection",
    image_mirror_range_um: float = 100,
    num_iterations: int = 3,
    num_mode_samples: int = 3,
    starting_coef_delta: float = 0.25,
    coef_delta_scale: float = 0.9,
    metric_precision: int = DEFUALT_SIGN_FIGS,
    modes_to_optimize: str = "spherical first",
    starting_mirror_state: str = "last optimized",
    starting_mirror_coefficients: NDArray | None = None,
    starting_reference_state: str = "system_flat",
    starting_coefficients_source: str | None = None,
    mode_acceptance: str = "zero",
    num_averaged_frames: int = 50,
    pos_idx: int = None,
    save_dir_path: Path | None = None,
    save_prefix: str | None = None,
    verbose: bool = True,
    initialization_metadata: dict | None = None,
    o3_stack_settings: dict | None = None,
    alignment_refocus_every_n_modes: int = 0,
    stop_requested=None,
) -> bool:
    """Run sensorless adaptive optics.

    Parameters
    ----------
    exposure_ms : float
        Camera exposure in milliseconds.
    channel_states : tuple[bool]
        Enabled excitation channels passed to the DAQ.
    metric_to_use : str
        Camera-image metric used to score mirror states.
    daq_mode : str
        DAQ scan mode.
    image_mirror_range_um : float
        Projection-mode image-mirror scan range in micrometers.
    num_iterations : int
        Number of optimization passes.
    num_mode_samples : int
        Number of perturbations sampled per mirror mode.
    starting_coef_delta : float
        Initial modal perturbation magnitude.
    coef_delta_scale : float
        Per-iteration reduction applied to modal perturbations.
    metric_precision : int
        Significant figures retained in metric values.
    modes_to_optimize : str
        Named set of mirror modes to optimize.
    starting_mirror_state : str
        Initial mirror state, such as ``system flat`` or ``last optimized``.
    starting_mirror_coefficients : numpy.ndarray or None
        Explicit modal coefficients used instead of a named mirror state. This
        supports point-matched initialization from the preceding AO Z level.
    starting_reference_state : str
        Mirror reference associated with explicit starting coefficients.
    starting_coefficients_source : str or None
        Description recorded for an explicit coefficient initializer.
    initialization_metadata : dict or None
        Acquisition policies and logical time, layer, and position identifiers.
    o3_stack_settings : dict or None
        O3 scan range, step, angle, remote pixel scale, and stage device.
    alignment_refocus_every_n_modes : int
        Initial 2D alignment only: refocus O3 between each N attempted modes.
        Zero disables. Uses current illumination without shutter control.
    mode_acceptance : str
        Rule used to accept a fitted modal update.
    num_averaged_frames : int
        Number of camera frames averaged for each metric.
    pos_idx : int or None
        Optional acquisition-position index included in saved results.
    save_dir_path : Path or None
        Directory in which to save optimization results.
    save_prefix : str or None
        Prefix for the saved mirror state.
    verbose : bool
        Whether to print optimization progress.

    Returns
    -------
    bool
        Whether optimization completed successfully; False on cooperative STOP.

    Raises
    ------
    ConnectionError
        If DAQ waveform playback stops during image acquisition.
    ImportError
        If required hardware controllers are unavailable.
    RuntimeError
        If a starting or candidate metric image cannot be acquired.
    ValueError
        If a mirror state or mode selection is invalid.
    """
    if (isinstance(alignment_refocus_every_n_modes, bool)
            or not isinstance(alignment_refocus_every_n_modes, (int, np.integer))
            or alignment_refocus_every_n_modes < 0):
        raise ValueError("Alignment refocus interval must be a nonnegative integer")
    if alignment_refocus_every_n_modes and (daq_mode != "2d" or metric_to_use == O3_STACK_METRIC):
        raise ValueError("Periodic alignment autofocus requires 2D, non-stack AO")
    if verbose:
        print("\n+++++++++++      RUNNING SENSORLESS AO      +++++++++++\n")

    # ---------------------------------------------#
    # Create hardware controller instances
    # ---------------------------------------------#

    opmNIDAQ_local = OPMNIDAQ.instance()
    AOMirror_local = AOMirror.instance()
    AOMirror_local.output_path = save_dir_path

    if AOMirror is None or OPMNIDAQ is None:
        raise ImportError("Required hardware modules not available.")

    mmc = CMMCorePlus.instance()
    o3_sampler = None
    if metric_to_use == O3_STACK_METRIC:
        from opm_v2.hardware.PicardShutter import PicardShutter

        settings = O3StackSettings(**(o3_stack_settings or {}))
        settings.offsets()  # Validate before moving any device.
        PicardShutter.instance().closeShutter()
        o3_sampler = O3StackMetric(settings, output_path=save_dir_path)
        daq_mode = "2d"
        channel_states = [False] * len(channel_states)

    # This routine executes the feedback loop behind the AO custom event.  Use
    # the active camera and the Core exposure API; only adapter-specific
    # trigger properties require direct property access.
    _prepare_camera_for_ao_snap(mmc)
    mmc.setExposure(float(exposure_ms))

    # TODO: Setup Camera for linescanning

    # get the current stage position
    stage_position = {
        "z": mmc.getZPosition(),
        "y": mmc.getYPosition(),
        "x": mmc.getXPosition(),
    }

    # ---------------------------------------------#
    # setup the daq for the selected imaging mode
    # ---------------------------------------------#

    if "projection" not in daq_mode:
        image_mirror_range_um = None
    opmNIDAQ_local.stop_waveform_playback()
    opmNIDAQ_local.clear_tasks()
    opmNIDAQ_local.set_acquisition_params(
        scan_type=daq_mode,
        channel_states=channel_states,
        image_mirror_range_um=image_mirror_range_um,
        laser_blanking=True,
        exposure_ms=exposure_ms,
    )
    opmNIDAQ_local.generate_waveforms()
    opmNIDAQ_local.program_daq_waveforms()
    opmNIDAQ_local.start_waveform_playback()

    # ---------------------------------------------#
    # Configure modes and acceptance settings
    # ---------------------------------------------#

    if isinstance(modes_to_optimize, str):
        if modes_to_optimize == "focusing only":
            modes_to_optimize = focusing_modes
        elif modes_to_optimize == "spherical only":
            modes_to_optimize = spherical_modes
        elif modes_to_optimize == "spherical first":
            modes_to_optimize = spherical_modes_first
        elif modes_to_optimize == "all modes":
            modes_to_optimize = all_modes
        elif modes_to_optimize == "stationary only":
            modes_to_optimize = stationary_modes
        elif modes_to_optimize == "high order first":
            modes_to_optimize = stationary_modes[::-1]
        else:
            raise ValueError(f"Invalid modes_to_optimize:{modes_to_optimize}")
    else:
        raise ValueError("modes_to_optimize must be a valid string")

    # ---------------------------------------------#
    # Set starting mirror state
    # ---------------------------------------------#

    starting_state_applied = True
    if starting_mirror_coefficients is not None:
        inherited_coefficients = np.asarray(
            starting_mirror_coefficients,
            dtype=np.float32,
        )
        if inherited_coefficients.shape != AOMirror_local.current_coeffs.shape:
            raise ValueError(
                "Inherited AO coefficients do not match the configured mirror modes"
            )
        AOMirror_local.set_reference_state(starting_reference_state)
        if not AOMirror_local.set_modal_coefficients(inherited_coefficients.copy()):
            opmNIDAQ_local.stop_waveform_playback()
            raise RuntimeError("Failed to apply inherited AO mirror coefficients")
        coefficient_source = starting_coefficients_source or "explicit coefficients"
        starting_state_label = f"{coefficient_source} ({starting_reference_state})"
    elif "system" in starting_mirror_state:
        starting_state_applied = AOMirror_local.apply_system_flat_voltage()
        starting_state_label = starting_mirror_state
    elif "optimized" in starting_mirror_state:
        starting_state_applied = AOMirror_local.apply_optimized_voltage()
        starting_state_label = starting_mirror_state
    elif "factory" in starting_mirror_state:
        starting_state_applied = AOMirror_local.apply_factory_flat_voltage()
        starting_state_label = starting_mirror_state
    elif "zeros" in starting_mirror_state:
        starting_state_applied = AOMirror_local.apply_zeros_voltage()
        starting_state_label = starting_mirror_state
    else:
        raise ValueError(f"Invalid starting mirror state: {starting_mirror_state}")

    if not starting_state_applied:
        opmNIDAQ_local.stop_waveform_playback()
        raise RuntimeError("Failed to apply starting AO mirror correction")

    if initialization_metadata:
        info("AO INITIALIZATION", initialization_metadata)
    if verbose:
        print(f"\n------- INFO -------\nStarting mirror state: {starting_state_label}")

    # ---------------------------------------------#
    # Setup tracking for images / metrics / coefficients
    # ---------------------------------------------#

    """Definition of variables

    -- Saved arrays
    all_images: ALL images passed including starting and optimal.
    all_metrics: ALL measured metrics, include starting and optimal measurements
    optimal_metrics: ALL optimal metrics calculated when when applying optimal_delta.
                     This list is appended after each mode, If no update is applied the
                     previous optimal metric is appended. NOT including starting_metric.
    optimal_images: ALL images obtained and kept after apply the optimal_delta.
                    If no update is applied, the last optimal image is appended.
                    NOT including starting_image.
    optimal_coeffs: Mirror modal coefficients at the end of each iteration.
                    Does not include starting_coeffs.
    starting_coeffs: Mirror coefficients at the start of optimization.
    update_status: A list of bools indicating whether modes are updated.

    -- Arrays for tracking modal coefficients and applying pertubations to the Mirror
    current_coeffs: Mirror modal coefficients initiated at the start of each iteration.
                    Array is updated when new modal coefficients are accepted.
    iteration_optimal_metrics: Optimal metrics initiated with starting or prior optimal
                               metric. List is initiated at the start of each iteration.
    active_coeffs: Modal coefficients to apply to the mirror before acquiring an image
                   and metric. This array is only used for perturbing the mirror.
    """
    from opm_v2.utils.autofocus_remote_unit import manage_O3_focus

    # Saved lists, covert to arrays at the end
    all_images = []
    all_metrics = []
    optimal_metrics = []
    optimal_images = []
    optimal_coeffs = []
    starting_coeffs = AOMirror_local.current_coeffs.copy()
    update_status = []

    # ---------------------------------------------#
    # Setup metadata
    # ---------------------------------------------#
    mode_deltas = [
        starting_coef_delta * (coef_delta_scale**k) for k in range(num_iterations)
    ]
    metadata = {
        "initialization": initialization_metadata or {},
        "starting_mirror_state": starting_state_label,
        "starting_coeffs": starting_coeffs.tolist(),
        "stage_position": stage_position,
        "opm_mode": daq_mode,
        "metric_name": metric_to_use,
        "num_averaged_frames": num_averaged_frames,
        "restoration_verification_store": "ao_restore_checks.zarr",
        "alignment_refocus_every_n_modes": alignment_refocus_every_n_modes,
        "alignment_refocus": [],
        "num_iterations": num_iterations,
        "num_mode_samples": num_mode_samples,
        "mode_deltas": list(mode_deltas),
        "modes_to_optimize": list(modes_to_optimize),
        "modes_to_optimize_names": [mode_names[i] for i in modes_to_optimize],
        "mode_acceptance": mode_acceptance,
        "channel_states": channel_states,
        "channel_exposure_ms": exposure_ms,
        "image_mirror_range_um": image_mirror_range_um,
        "pos_idx": 0 if pos_idx is None else pos_idx,
    }
    if o3_sampler is not None:
        metadata["o3_stack"] = {
            **asdict(o3_sampler.settings),
            "illumination": "manually enabled external alignment beam",
            "axes": ["o3", "camera_y", "camera_x"],
            "geometry": "rotation about camera X into O2; no resampling",
            "measurements": o3_sampler.diagnostics,
        }
    # ---------------------------------------------#
    # Start AO optimization
    # ---------------------------------------------#

    if verbose:
        print("\n++++++++    STARTING SENSORLESS AO LOOP    ++++++++\n")

    # A new prior for each AO position/run, held fixed across all its probes.
    metric_options = (
        {"gaussian_prior": GaussianSpotPrior()} if metric_to_use == "gauss_2d" else {}
    )
    if o3_sampler is not None:
        metric_options = {"o3_sampler": o3_sampler}
        o3_sampler.context = {
            "kind": "starting", "modal_coefficients": starting_coeffs.tolist()
        }

    metric_options["stop_requested"] = stop_requested
    # Aqcuire starting image and metric
    try:
        starting_image, starting_metric = acquire_metric_image(
            mmc,
            metric_to_use,
            num_averaged_frames,
            opm_nidaq=opmNIDAQ_local,
            **metric_options,
        )
        if not np.isfinite(starting_metric):
            raise ValueError("Starting AO metric must be finite")
        starting_metric = round_to_sigfigs(starting_metric, metric_precision)

        # append to tracking lists
        all_images.append(starting_image)
        all_metrics.append(starting_metric)
    except AOCancelled:
        opmNIDAQ_local.stop_waveform_playback()
        return False
    except Exception as e:
        opmNIDAQ_local.stop_waveform_playback()
        raise RuntimeError(f"Failed to get starting metric: {e}") from e

    # Start AO iterations
    for k in range(num_iterations):
        if k == 0:
            iteration_optimal_metrics = [starting_metric]
            current_coeffs = starting_coeffs.copy()
        else:
            iteration_optimal_metrics = [optimal_metrics[-1]]

        if verbose:
            print(
                "\n\n",
                "-------------------------------------------------------------------",
                f"Optimization iteration: {k + 1} / {num_iterations}",
                f"Coeff pertubation amplitude: {mode_deltas[k]:.6f}",
                f"Starting mirror coefficients: \n{current_coeffs}",
                "-------------------------------------------------------------------",
                sep="\n",
            )

        # Iterate over modes to optimize
        for mode in modes_to_optimize:
            if stop_requested is not None and stop_requested():
                _restore_ao_correction(AOMirror_local, current_coeffs, opmNIDAQ_local)
                opmNIDAQ_local.stop_waveform_playback()
                return False
            completed_modes = len(optimal_metrics)
            if (alignment_refocus_every_n_modes and completed_modes
                    and completed_modes % alignment_refocus_every_n_modes == 0):
                _restore_ao_correction(AOMirror_local, current_coeffs, opmNIDAQ_local)
                stage_name = (o3_stack_settings or {}).get("stage_name", "MCL NanoDrive Z Stage")
                try:
                    stage = mmc.getDeviceObject(stage_name)
                    stage.wait()
                    start_um = float(stage.getPosition())
                    best_um = manage_O3_focus(
                        stage_name, verbose=verbose, mmc=mmc,
                        control_shutter=False, stop_requested=stop_requested,
                    )
                    record = {
                        "after_modes": completed_modes, "start_um": start_um,
                        "best_um": float(best_um), "timestamp_unix_s": time.time(),
                        "modal_coefficients": current_coeffs.tolist(),
                    }
                    metadata["alignment_refocus"].append(record)
                    if save_dir_path is not None:
                        save_dir_path.mkdir(parents=True, exist_ok=True)
                        journal = zarr.open_group(str(save_dir_path / "ao_alignment_focus.zarr"), mode="a")
                        journal.create_group(f"after_{completed_modes:06d}").attrs.update(json_metadata(record))
                    print(f"AO alignment refocus after {completed_modes} modes: {start_um:.3f} -> {best_um:.3f} um")
                except AOCancelled:
                    opmNIDAQ_local.stop_waveform_playback()
                    return False
                except Exception:
                    opmNIDAQ_local.stop_waveform_playback()
                    raise
            if verbose:
                print(f"\nOptimizing Zernike mode: {mode + 1}:{mode_names[mode]} . . .")

            # Add deltas to the current mirror coeffs.
            modal_coeff_deltas = np.linspace(
                -mode_deltas[k], mode_deltas[k], num_mode_samples
            )
            # Even sample counts must also include a true reference measurement.
            modal_coeff_deltas[num_mode_samples // 2] = 0.0
            perturbed_modal_coeffs = np.tile(
                current_coeffs.copy(), (num_mode_samples, 1)
            )
            perturbed_modal_coeffs[:, mode] += modal_coeff_deltas

            mode_success = True
            metrics = []
            probe_images = []
            for ii, modal_coeff in enumerate(perturbed_modal_coeffs):
                # Keep failed/skipped samples in the saved arrays, but never
                # fit a partial sweep or acquire an image for a rejected command.
                try:
                    success = mode_success and AOMirror_local.set_modal_coefficients(
                        modal_coeff
                    )
                except Exception:
                    _restore_ao_correction(
                        AOMirror_local, current_coeffs, opmNIDAQ_local
                    )
                    raise

                if not (success):
                    mode_success = False
                    metric = np.nan
                    image = np.zeros_like(starting_image)
                else:
                    if not opmNIDAQ_local.running():
                        _restore_ao_correction(
                            AOMirror_local, current_coeffs, opmNIDAQ_local
                        )
                        raise ConnectionError("DAQ is not running, check for errors")
                    if o3_sampler is not None:
                        o3_sampler.context = {
                            "kind": "probe", "iteration": k + 1,
                            "mode_index": int(mode), "mode_name": mode_names[mode],
                            "delta": float(modal_coeff_deltas[ii]),
                            "modal_coefficients": modal_coeff.tolist(),
                        }
                    try:
                        image, metric = acquire_metric_image(
                            mmc,
                            metric_to_use,
                            num_averaged_frames,
                            opm_nidaq=opmNIDAQ_local,
                            **metric_options,
                        )
                    except AOCancelled:
                        _restore_ao_correction(AOMirror_local, current_coeffs, opmNIDAQ_local)
                        opmNIDAQ_local.stop_waveform_playback()
                        return False
                    except Exception as e:
                        _restore_ao_correction(
                            AOMirror_local, current_coeffs, opmNIDAQ_local
                        )
                        raise RuntimeError("Exception in acquiring image") from e
                    try:
                        metric = round_to_sigfigs(metric, metric_precision)
                    except Exception:
                        metric = np.nan
                        image = np.zeros_like(starting_image)
                        mode_success = False

                    # Catch bad metric calculations
                    if not np.isfinite(metric):
                        mode_success = False
                        if verbose:
                            print(
                                "\n------- WARNING -------",
                                "\n Metrics calculation failed",
                            )

                # Update all metrics / images
                metrics.append(metric)
                probe_images.append(image)
                all_metrics.append(metric)
                all_images.append(image)
                if verbose:
                    print(
                        f"    Delta={modal_coeff_deltas[ii]:.6f}, Metric={metric:.6f}"
                    )

            # ---------------------------------------------#
            # Fit metrics to determine optimal delta
            # ---------------------------------------------#
            fit_valid = mode_success
            if mode_success:
                try:
                    # Are metrics monotonic, if so use the maximum
                    is_increasing = all(
                        x < y
                        for x, y in zip(np.asarray(metrics), np.asarray(metrics)[1:])
                    )
                    is_decreasing = all(
                        x > y
                        for x, y in zip(np.asarray(metrics), np.asarray(metrics)[1:])
                    )
                    if is_increasing or is_decreasing:
                        optimal_delta = modal_coeff_deltas[np.argmax(metrics)]
                        if verbose:
                            print(
                                "\n------- WARNING -------",
                                "\n Metrics increasing/decreasing using maximum",
                            )
                    else:
                        # Fit and use the peak of quadratic fit
                        a, b, c = quadratic_fit(modal_coeff_deltas, metrics)
                        if not np.all(np.isfinite([a, b, c])):
                            raise ValueError("AO fit coefficients must be finite")
                        # Reject if metrics have positive curvature
                        if a >= 0:
                            optimal_delta = modal_coeff_deltas[int(np.argmax(metrics))]
                            # optimal_delta = 0
                            # optimal_image = np.zeros_like(starting_image)
                            # optimal_metric = np.nan
                            if verbose:
                                print(
                                    "\n------- WARNING -------"
                                    "\n Metrics have positive curvature!"
                                )
                        else:
                            optimal_delta = -b / (2 * a)
                        if not np.isfinite(optimal_delta):
                            raise ValueError("AO fitted delta must be finite")
                        # Clip optimal delta at the edge of sampled range
                        optimal_delta = np.clip(
                            optimal_delta,
                            modal_coeff_deltas.min(),
                            modal_coeff_deltas.max(),
                        )
                except Exception as e:
                    fit_valid = False
                    optimal_delta = 0
                    optimal_image = np.zeros_like(starting_image)
                    if verbose:
                        print(
                            "\n------- WARNING -------"
                            f"\n Exception in fit occurred! \n {e}"
                        )
            else:
                optimal_delta = 0
                optimal_metric = np.nan
                optimal_image = np.zeros_like(starting_image)
                if verbose:
                    print("\n------- WARNING -------\n Error occured in metrics!")

            # ---------------------------------------------#
            # Validate the optimal delta
            # ---------------------------------------------#
            current_opt_metric = iteration_optimal_metrics[-1]
            current_opt_image = optimal_images[-1] if optimal_images else starting_image
            zero_metric_index = num_mode_samples // 2
            # Zero delta is the same correction already accepted. Retain a
            # higher observed score and its image even when the candidate loses.
            if np.isfinite(metrics[zero_metric_index]) and metrics[zero_metric_index] > current_opt_metric:
                current_opt_metric = metrics[zero_metric_index]
                current_opt_image = probe_images[zero_metric_index]
            update = False
            candidate_valid = False

            if np.isfinite(optimal_delta) and optimal_delta != 0:
                # remeasure metric to verify new position
                active_coeffs = current_coeffs.copy()
                active_coeffs[mode] += optimal_delta
                try:
                    success = AOMirror_local.set_modal_coefficients(active_coeffs)
                except Exception:
                    _restore_ao_correction(
                        AOMirror_local, current_coeffs, opmNIDAQ_local
                    )
                    raise
                if success:
                    time.sleep(0.1)
                    # Only score a correction that was actually applied.
                    if not opmNIDAQ_local.running():
                        _restore_ao_correction(
                            AOMirror_local, current_coeffs, opmNIDAQ_local
                        )
                        raise ConnectionError("DAQ is not running, check for errors")
                    if o3_sampler is not None:
                        o3_sampler.context = {
                            "kind": "candidate", "iteration": k + 1,
                            "mode_index": int(mode), "mode_name": mode_names[mode],
                            "delta": float(optimal_delta),
                            "modal_coefficients": active_coeffs.tolist(),
                        }
                    try:
                        optimal_image, optimal_metric = acquire_metric_image(
                            mmc,
                            metric_to_use,
                            num_averaged_frames,
                            opm_nidaq=opmNIDAQ_local,
                            **metric_options,
                        )
                    except AOCancelled:
                        _restore_ao_correction(AOMirror_local, current_coeffs, opmNIDAQ_local)
                        opmNIDAQ_local.stop_waveform_playback()
                        return False
                    except Exception as e:
                        _restore_ao_correction(
                            AOMirror_local, current_coeffs, opmNIDAQ_local
                        )
                        raise RuntimeError(
                            "Exception in acquiring optimal image"
                        ) from e

                    if np.isfinite(optimal_metric):
                        optimal_metric = round_to_sigfigs(
                            optimal_metric, metric_precision
                        )

                    if verbose:
                        print(
                            f"    Candidate delta: {optimal_delta:.6f}",
                            f"    Candidate metric: {optimal_metric:.5f}\n",
                            sep="\n",
                        )

                    # Acceptance policies only compare valid, applied states.
                    if np.isfinite(optimal_metric):
                        candidate_valid = True
                        if "all" in mode_acceptance:
                            update = True
                        elif "optimal" in mode_acceptance:
                            update = optimal_metric >= current_opt_metric
                        elif "zero" in mode_acceptance:
                            update = optimal_metric >= metrics[zero_metric_index]

            # A quadratic vertex is a proposal, not necessarily the best
            # observed correction. Under "optimal", keep a better valid probe
            # rather than discarding it in favor of the fitted candidate.
            # Failed fits/commands/measurements still invalidate the mode.
            if (
                "optimal" in mode_acceptance and fit_valid
                and (candidate_valid or optimal_delta == 0)
            ):
                best_probe = int(np.argmax(metrics))
                candidate_score = optimal_metric if candidate_valid else -np.inf
                if metrics[best_probe] > max(current_opt_metric, candidate_score):
                    selected_delta = float(modal_coeff_deltas[best_probe])
                    selected_coeffs = current_coeffs.copy()
                    selected_coeffs[mode] += selected_delta
                    try:
                        success = AOMirror_local.set_modal_coefficients(selected_coeffs)
                    except Exception:
                        _restore_ao_correction(AOMirror_local, current_coeffs, opmNIDAQ_local)
                        raise
                    update = bool(success)
                    if update:
                        optimal_delta = selected_delta
                        optimal_metric = metrics[best_probe]
                        optimal_image = probe_images[best_probe]
                        if verbose:
                            print(
                                f"    Keeping measured delta: {optimal_delta:.6f}, "
                                f"metric: {optimal_metric:.6f}"
                            )

            # Update lists dependant on update status
            if update:
                current_coeffs[mode] += optimal_delta
            else:
                _restore_ao_correction(
                    AOMirror_local, current_coeffs, opmNIDAQ_local
                )
                try:
                    _measure_restored_correction(
                        AOMirror_local, mmc, opmNIDAQ_local, metric_to_use,
                        num_averaged_frames, metric_options,
                        {
                            "step_index": len(optimal_metrics),
                            "iteration": k + 1,
                            "mode_index": int(mode),
                            "mode_name": mode_names[mode],
                            "modal_coefficients": current_coeffs.copy(),
                            "historical_best_metric": float(current_opt_metric),
                            "zero_probe_metric": float(metrics[zero_metric_index]),
                        },
                        save_dir_path,
                    )
                except AOCancelled:
                    opmNIDAQ_local.stop_waveform_playback()
                    return False
                except Exception as exc:
                    opmNIDAQ_local.stop_waveform_playback()
                    raise RuntimeError("Failed to verify restored AO correction") from exc
                optimal_image = current_opt_image
                optimal_metric = current_opt_metric

            optimal_images.append(optimal_image)
            optimal_metrics.append(optimal_metric)
            iteration_optimal_metrics.append(optimal_metric)
            all_images.append(optimal_image)
            all_metrics.append(optimal_metric)
            optimal_coeffs.append(current_coeffs.copy())

            # Update the update status list
            update_status.append(update)

            if verbose:
                print(
                    f"\n+++++     Was mirror updated: {update}     +++++",
                    f"Retained best metric (historical): {iteration_optimal_metrics[-1]:.6f}",
                    sep="\n",
                )

            """Loop back to top and do the next mode until all modes are done"""

        """Loop back to top and do the next iteration"""

    # ---------------------------------------------#
    # Optimization complete!
    # ---------------------------------------------#

    # Update the mirror state and positions arrays
    _restore_ao_correction(AOMirror_local, optimal_coeffs[-1], opmNIDAQ_local)
    AOMirror_local.update_optimized_array()
    if pos_idx is not None:
        try:
            AOMirror_local.update_positions_array(pos_idx)
        except Exception as e:
            print(f"----- Failed to update the AO MIRROR position array ----- \n{e}")

    if verbose:
        print(
            "\n+++++     SENSORLESS AO COMPLETE     +++++",
            f"\nStarting Zernike mode amplitude: \n{starting_coeffs}",
            f"\nFinal optimized Zernike mode amplitude: \n{current_coeffs}",
        )

    # Stop the DAQ programming
    opmNIDAQ_local.stop_waveform_playback()

    if save_dir_path:
        if not (save_dir_path.exists()):
            save_dir_path.mkdir(exist_ok=True)

        # Save mirror states
        save_prefix = "optimized"
        AOMirror_local.save_current_state(save_prefix)
        AOMirror_local.save_positions_array(save_prefix)

        results_step = "Saving AO results"
        try:
            # Save and plot optimization results
            all_images = np.asarray(all_images)
            all_metrics = np.asarray(all_metrics)
            optimal_images = np.asarray(optimal_images)
            optimal_metrics = np.asarray(optimal_metrics)
            optimal_coeffs = np.asarray(optimal_coeffs)
            starting_coeffs = np.asarray(starting_coeffs)
            starting_metric = np.array([starting_metric])
            update_status = np.asarray(update_status)
            save_optimization_results(
                all_images=all_images,
                all_metrics=all_metrics,
                optimal_images=optimal_images,
                optimal_metrics=optimal_metrics,
                optimal_coeffs=optimal_coeffs,
                starting_coeffs=starting_coeffs,
                starting_metric=starting_metric,
                starting_image=starting_image,
                update_status=update_status,
                metadata=metadata,
                save_dir_path=save_dir_path,
                **({"o3_stacks": np.asarray(o3_sampler.measured_stacks)}
                   if o3_sampler is not None else {}),
            )
            results_step = "Loading saved AO results"
            ao_results = load_optimization_results(
                save_dir_path / Path("ao_results.zarr")
            )
            results_step = "Plotting AO Zernike coefficients (results already saved)"
            plot_zernike_coeffs(
                ao_results=ao_results,
                zernike_mode_names=mode_names,
                save_dir_path=save_dir_path,
                show_fig=False,
                x_range=starting_coef_delta,
            )
            results_step = "Plotting AO metric progress (results already saved)"
            plot_metric_progress(
                ao_results=ao_results,
                zernike_mode_names=mode_names,
                save_dir_path=save_dir_path,
                show_fig=False,
            )
        except Exception as e:
            print(
                "\n------- WARNING -------",
                f"{results_step} failed!",
                f"{e}",
                sep="\n",
            )
            traceback.print_exc()

    return True


# -------------------------------------------------#
# Plotting functions
# -------------------------------------------------#


def plot_zernike_coeffs(
    ao_results: dict,
    zernike_mode_names: NDArray,
    save_dir_path: Path = None,
    show_fig: bool | None = False,
    x_range=0.1,
) -> None:
    """Plot the Zernike coefficient values per iteration.

    Parameters
    ----------
    ao_results : dict
        Optimization results and metadata.
    zernike_mode_names : NDArray
    save_dir_path : Path, optional
        Path to save figure, by default None
    show_fig : Optional[bool], optional
        whether to display figure, by default False
    x_range : float, optional
        the coefficient magnitude to display on axis, by default 0.1
    """
    import matplotlib
    import matplotlib.pyplot as plt

    if not show_fig:
        matplotlib.use("Agg")
    from matplotlib.ticker import FormatStrFormatter

    plt.rcParams.update({
        "font.size": 14,  # base font size
        "axes.titlesize": 18,  # title size
        "axes.labelsize": 15,  # x/y label size
        "xtick.labelsize": 15,  # x-tick label size
        "ytick.labelsize": 15,  # y-tick label size
        "legend.fontsize": 14,  # legend size
    })
    metadata = ao_results["metadata"]
    num_iterations = int(metadata["num_iterations"])
    optimal_coeffs = np.array(ao_results["optimal_coeffs"])
    starting_coeffs = np.array(ao_results["starting_coeffs"])
    # Current stores contain one correction per mode update. Older stores
    # contain only the final correction from each iteration.
    if optimal_coeffs.shape[0] != num_iterations:
        num_modes = len(metadata["modes_to_optimize"])
        optimal_coeffs = optimal_coeffs.reshape(
            num_iterations, num_modes, -1
        )[:, -1, :]

    # Create the plot
    fig, ax = plt.subplots(figsize=(7.5, 8))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # Define colors and markers for each iteration
    colors = ["b", "g", "r", "c", "m"]
    markers = ["x", "o", "^", "s", "*"]

    # Plot the starting Coeffs
    if starting_coeffs is not None:
        for i in range(len(zernike_mode_names)):
            if i == 0:
                label_str = "Starting Coeffs"
            else:
                label_str = None
            ax.scatter(
                starting_coeffs[i],
                i,
                s=140,
                marker="*",
                color="k",
                label=label_str,
                # alpha=0.5
            )

    # Plot the coeffs at the end of each iteration
    for i in range(len(zernike_mode_names)):
        for j in range(num_iterations):
            if i == 0:
                label_str = f"Iter.: {j + 1}"
            else:
                label_str = None
            marker_style = markers[j % len(markers)]
            ax.scatter(
                optimal_coeffs[j, i],
                i,
                color=colors[j % len(colors)],
                s=125,
                marker=marker_style,
                alpha=0.5,
                label=label_str,
            )
        ax.axhline(y=i, linestyle="--", linewidth=1, color="k")

    # Plot a vertical line at 0 for reference
    ax.axvline(0, color="k", linestyle="-", linewidth=1)

    # Customize the plot
    ax.set(
        yticks=np.round(np.arange(len(zernike_mode_names)), 1),
        yticklabels=zernike_mode_names,
        xlabel=r"$\vert Z_n \vert$",
        title="Zernike Coefficients per Iteration",
        xlim=(-x_range, x_range),
    )

    # Add a legend for time points
    ax.legend(loc="upper right")

    # Remove grid lines
    ax.grid(False)

    plt.tight_layout()

    plt.tight_layout()
    if show_fig:
        plt.show()
    if save_dir_path:
        fig.savefig(save_dir_path / Path("ao_zernike_coeffs.png"))
    if not show_fig:
        plt.close(fig)


def plot_metric_progress_depr(
    all_metrics: NDArray,
    num_iterations: int,
    modes_to_optimize: list[int],
    zernike_mode_names: list[str],
    save_dir_path: Path = None,
    show_fig: bool | None = False,
) -> None:
    """Plot the metric magnitude throughout optimization.

    Parameters
    ----------
    all_metrics : NDArray
    num_iterations : float
    modes_to_optimize : List[int]
    zernike_mode_names : List[str]
    save_dir_path : Path, optional
        Path to save figure, by default None
    show_fig : Optional[bool], optional
        whether to display figure, by default False
    """
    import matplotlib.pyplot as plt

    if not show_fig:
        import matplotlib

        matplotlib.use("Agg")

    plt.rcParams.update({
        "font.size": 14,  # base font size
        "axes.titlesize": 18,  # title size
        "axes.labelsize": 15,  # x/y label size
        "xtick.labelsize": 15,  # x-tick label size
        "ytick.labelsize": 15,  # y-tick label size
        "legend.fontsize": 14,  # legend size
    })
    num_modes = len(modes_to_optimize)
    samples_per_mode = np.ceil(
        (len(all_metrics) - 1) // num_iterations / len(modes_to_optimize)
    ).astype(int)
    modes_to_use_names = [zernike_mode_names[i] for i in modes_to_optimize]

    # Ignore starting metric
    metrics = np.array(all_metrics[1:])

    # Reshape into: (iterations, modes, samples)
    metrics_by_iteration = metrics.reshape(
        num_iterations, num_modes, int(samples_per_mode)
    )
    zero_metrics_by_iteration = metrics_by_iteration[:, :, int(samples_per_mode // 2)]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors and markers for each iteration
    colors = ["b", "g", "r", "c", "m"]
    markers = ["x", "o", "^", "s", "*"]

    # Loop over iterations and plot each series
    for ii in range(num_iterations):
        ax.plot(
            np.linspace(0, num_modes - 1, num_modes),
            zero_metrics_by_iteration[ii],
            color=colors[ii % len(colors)],
            label=f"iteration {ii}",
            marker=markers[ii % len(markers)],
            linestyle="--",
            linewidth=1,
        )

    ax.set(
        title="Metric Progress per Iteration",
        ylabel="Metric",
        xticks=np.arange(len(modes_to_use_names)),
    )
    ax.set_xticklabels(modes_to_use_names, rotation=60, ha="right", fontsize=16)
    ax.legend()
    plt.tight_layout()

    if show_fig:
        plt.show()
    if save_dir_path:
        fig.savefig(save_dir_path / Path("ao_metrics.png"))


def plot_metric_progress(
    ao_results: dict,
    zernike_mode_names: list[str],
    display_optimal: bool = True,
    display_zero: bool = True,
    save_dir_path: Path = None,
    show_fig: bool | None = False,
) -> None:
    """Plot the metric magnitude throughout optimization.

    Parameters
    ----------
    ao_results : dict
        Optimization results and metadata.
    zernike_mode_names : List[str]
        Display names for the available mirror modes.
    display_optimal : bool
        Whether to display accepted metric values.
    display_zero : bool
        Whether to display zero-perturbation metric values.
    save_dir_path : Path, optional
        Path to save figure, by default None
    show_fig : Optional[bool], optional
        whether to display figure, by default False
    """
    import matplotlib.pyplot as plt

    if not show_fig:
        import matplotlib

        matplotlib.use("Agg")

    plt.rcParams.update({
        "font.size": 14,  # base font size
        "axes.titlesize": 18,  # title size
        "axes.labelsize": 15,  # x/y label size
        "xtick.labelsize": 15,  # x-tick label size
        "ytick.labelsize": 15,  # y-tick label size
        "legend.fontsize": 14,  # legend size
    })
    metadata = ao_results["metadata"]
    num_mode_samples = int(metadata["num_mode_samples"])
    num_iterations = int(metadata["num_iterations"])
    metric_name = str(metadata["metric_name"])
    modes_to_optimize = metadata["modes_to_optimize"]
    num_modes = len(modes_to_optimize)
    try:
        mode_names = metadata["modes_to_optimize_names"]
    except Exception:
        mode_names = [zernike_mode_names[i] for i in modes_to_optimize]

    # Load metrics, ignore the starting metric
    metrics = np.array(ao_results["all_metrics"][1:])
    optimal_metrics = np.array(ao_results["optimal_metrics"])

    # Reshape into: (iterations, modes, samples+1) +1 for optimal image
    optimal_metrics_by_iteration = optimal_metrics.reshape(num_iterations, num_modes)
    metrics_by_iteration = metrics.reshape(
        num_iterations, num_modes, num_mode_samples + 1
    )
    zero_metrics_by_iteration = metrics_by_iteration[:, :, int(num_mode_samples // 2)]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors and markers for each iteration
    colors = ["b", "g", "r", "c", "m"]
    markers = ["x", "o", "^", "s", "*"]

    # Loop over iterations and plot each series
    for ii in range(num_iterations):
        if display_zero:
            ax.plot(
                np.linspace(0, num_modes - 1, num_modes),
                zero_metrics_by_iteration[ii],
                color=colors[ii % len(colors)],
                label=f"\u0394=0, iteration {ii}",
                marker=markers[ii % len(markers)],
                linestyle="--",
                linewidth=1,
            )
        if display_optimal:
            ax.plot(
                np.linspace(0, num_modes - 1, num_modes),
                optimal_metrics_by_iteration[ii],
                color=colors[ii % len(colors)],
                label=f"optimal iteration {ii}",
                marker=markers[ii % len(markers)],
                linestyle="-",
                linewidth=1,
            )

    ax.set(
        title="Metric Progress per Iteration",
        ylabel=metric_name,
        xticks=np.arange(len(mode_names)),
    )
    ax.set_xticklabels(mode_names, rotation=60, ha="right", fontsize=16)
    ax.legend()
    plt.tight_layout()

    if show_fig:
        plt.show()
    if save_dir_path:
        fig.savefig(save_dir_path / Path("ao_metrics.png"))
    if not show_fig:
        plt.close(fig)


def plot_phase(phase: dict, save_dir_path: Path = None, show_fig: bool = False) -> None:
    """Plot the 2d Phase for a given set of modal coeffs.

    Parameters
    ----------
    phase : Dict
        Wavefront phase data and its extent.
    save_dir_path : Path, optional
        Path to save figure, by default None
    show_fig : bool, optional
        whether to display figure, by default False
    """
    import matplotlib
    import matplotlib.pyplot as plt

    if not show_fig:
        matplotlib.use("Agg")
    # --- Set rcParams (this affects all plots until you change/reset it) ---
    plt.rcParams.update({
        "font.size": 14,  # base font size
        "axes.titlesize": 18,  # title size
        "axes.labelsize": 16,  # x/y label size
        "xtick.labelsize": 14,  # x-tick label size
        "ytick.labelsize": 14,  # y-tick label size
        "legend.fontsize": 14,  # legend size
    })
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    vrange = np.max([np.abs(phase["min"]), np.abs(phase["max"])])
    im = ax.imshow(phase["phase"], cmap="seismic", vmin=-vrange, vmax=vrange)
    plt.colorbar(im)
    ax.set_title("Wavefront Phase")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    if show_fig:
        plt.show()
    if save_dir_path:
        fig.savefig(save_dir_path / Path("phase.png"))


def plot_2d_localization_fit_summary(
    fit_results, img, coords_2d, save_dir_path: Path = None, showfig: bool = False
):
    """Generate a figure showing the localization an fit results.

    Parameters
    ----------
    fit_results : _type_
        _description_
    img : _type_
        _description_
    coords_2d : _type_
        _description_
    save_dir_path : Path, optional
        _description_, by default None
    showfig : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    # imports
    import matplotlib
    import matplotlib.pyplot as plt
    from localize_psf.fit_psf import sxy2na
    from localize_psf.localize import plot_bead_locations

    matplotlib.use("Agg")

    to_keep = fit_results["to_keep"]
    sxy = fit_results["fit_params"][to_keep, 4]
    amp = fit_results["fit_params"][to_keep, 0]
    bg = fit_results["fit_params"][to_keep, 6]
    centers = fit_results["fit_params"][to_keep][:, (3, 2, 1)]
    cx = centers[:, 2]
    cy = centers[:, 1]

    width_ratios = [1, 0.7, 0.1, 0.1, 0.1]
    height_ratios = [1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5]
    figh_sum = plt.figure(figsize=(10, 8))
    grid_sum = figh_sum.add_gridspec(
        nrows=len(height_ratios),
        ncols=len(width_ratios),
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        hspace=0.2,
        wspace=0.3,
    )

    ax_proj_sxy = figh_sum.add_subplot(grid_sum[0, :2])
    ax_cmap_i_sxy = figh_sum.add_subplot(grid_sum[0, 2])
    ax_cmap_sxy = figh_sum.add_subplot(grid_sum[0, 4])
    figh_sum = plot_bead_locations(
        img,
        centers,
        weights=[fit_results["fit_params"][to_keep, 4]],
        color_lists=["autumn"],
        color_limits=[[0.05, 0.5]],
        cbar_labels=[r"$\sigma_{xy}$"],
        title="Max intensity projection with Sxy",
        coords=coords_2d,
        gamma=0.5,
        axes=[ax_proj_sxy, ax_cmap_i_sxy, ax_cmap_sxy],
    )
    sxy_na = sxy2na(0.473, np.median(sxy))
    ax_proj_sxy.set_title(
        f"Sxy: mean={np.mean(sxy):.3f},"
        f"median={np.median(sxy):.3f},"
        f"NA (median):{sxy_na:.2f}"
    )

    # Create axes for plotting x, y specific results
    ax_sxy_cx = figh_sum.add_subplot(grid_sum[3, 0])
    ax_sxy_cy = figh_sum.add_subplot(grid_sum[3, 1:], sharey=ax_sxy_cx)
    ax_amp_cx = figh_sum.add_subplot(grid_sum[4, 0], sharex=ax_sxy_cx)
    ax_amp_cy = figh_sum.add_subplot(
        grid_sum[4, 1:], sharey=ax_amp_cx, sharex=ax_sxy_cy
    )
    ax_bg_cx = figh_sum.add_subplot(grid_sum[5, 0], sharex=ax_sxy_cx)
    ax_bg_cy = figh_sum.add_subplot(grid_sum[5, 1:], sharey=ax_bg_cx, sharex=ax_sxy_cy)
    ax_sxy_cx.set_ylabel(r"$\sigma_{xy}$ ($\mu m$)")
    ax_amp_cx.set_ylabel("amplitude")
    ax_bg_cx.set_ylabel("background")
    ax_bg_cx.set_xlabel(r"$C_x$ $\mu m$")
    ax_bg_cy.set_xlabel(r"$C_y$ $\mu m$")
    for ax in [ax_sxy_cy, ax_amp_cy, ax_bg_cy]:
        ax.tick_params(labelleft=False)
    for ax in [ax_sxy_cx, ax_sxy_cy, ax_amp_cx, ax_amp_cy]:
        ax.tick_params(labelbottom=False)

    # Set limits for visualizing sz
    if max(amp) > 65000:
        amp_max = 15000
    else:
        amp_max = np.max(amp) * 1.1
    ax_sxy_cx.set_ylim(0, 1.0)
    ax_sxy_cy.set_ylim(0, 1.0)
    ax_amp_cx.set_ylim(0, amp_max)
    ax_amp_cy.set_ylim(0, amp_max)
    ax_bg_cx.set_ylim(0, amp_max)
    ax_bg_cy.set_ylim(0, amp_max)
    ax_sxy_cx.set_xlim(0, img.shape[1] * 0.115)
    ax_sxy_cy.set_xlim(0, img.shape[0] * 0.115)
    ax_amp_cx.set_xlim(0, img.shape[1] * 0.115)
    ax_amp_cy.set_xlim(0, img.shape[0] * 0.115)
    ax_bg_cx.set_xlim(0, img.shape[1] * 0.115)
    ax_bg_cy.set_xlim(0, img.shape[0] * 0.115)
    # Plot directional results
    ax_sxy_cx.plot(cx, sxy, c="b", marker=".", markersize=3, linestyle="none")
    ax_sxy_cy.plot(cy, sxy, c="b", marker=".", markersize=3, linestyle="none")
    ax_amp_cx.plot(cx, amp, c="b", marker=".", markersize=3, linestyle="none")
    ax_amp_cy.plot(cy, amp, c="b", marker=".", markersize=3, linestyle="none")
    ax_bg_cx.plot(cx, bg, c="b", marker=".", markersize=3, linestyle="none")
    ax_bg_cy.plot(cy, bg, c="b", marker=".", markersize=3, linestyle="none")

    if showfig:
        figh_sum.show()
        plt.show()
    else:
        plt.close(figh_sum)
    if save_dir_path:
        figh_sum.savefig(save_dir_path / Path("ao_localization_results.png"), dpi=150)

    figh_sum = None
    del figh_sum
    return None


# -------------------------------------------------#
# Functions for preparing data
# -------------------------------------------------#


def get_image_center(image: NDArray, threshold: float) -> tuple[int, int]:
    """
    Calculate the center of an image using a thresh-holded binary mask.

    Parameters
    ----------
    image : NDArray
        2D image array.
    threshold : float
        Intensity threshold for binarization.

    Returns
    -------
    center : Tuple[int, int]
        Estimated center coordinates (x, y).

    Raises
    ------
    ValueError
        If the threshold mask has no finite center.
    """
    try:
        binary_image = image > threshold
        center_y, center_x = center_of_mass(binary_image)
        if not np.isfinite(center_x) or not np.isfinite(center_y):
            raise ValueError("Could not determine image center from threshold mask")
        center = (int(center_x), int(center_y))
    except Exception:
        center = (image.shape[1] // 2, image.shape[0] // 2)
    return center


def get_cropped_image(
    image: NDArray, crop_size: int, center: tuple[int, int]
) -> NDArray:
    """
    Extract a square region from an image centered at a given point.

    Parameters
    ----------
    image : NDArray
        Input 2D or 3D image.
    crop_size : int
        Half-width of the cropping region.
    center : Tuple[int, int]
        Center coordinates (x, y) of the crop.

    Returns
    -------
    cropped_image : NDArray
        Cropped region from the input image.
    """
    if center is None:
        center = get_image_center(image, threshold=1000)

    # TODO: Validate Cropping, is it x,y or y,x
    if len(image.shape) == 3:
        x_min, x_max = (
            max(center[0] - crop_size, 0),
            min(center[0] + crop_size, image.shape[2]),
        )
        y_min, y_max = (
            max(center[1] - crop_size, 0),
            min(center[1] + crop_size, image.shape[1]),
        )
        cropped_image = image[:, y_min:y_max, x_min:x_max]
    else:
        x_min, x_max = (
            max(center[0] - crop_size, 0),
            min(center[0] + crop_size, image.shape[1]),
        )
        y_min, y_max = (
            max(center[1] - crop_size, 0),
            min(center[1] + crop_size, image.shape[0]),
        )
        cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image


# -------------------------------------------------#
# Functions for fitting and calculations
# -------------------------------------------------#


def metric_r_power_integral(
    img: np.ndarray, integration_radius: int = 40, power: int = 2
) -> float:
    """TODO.

    Parameters
    ----------
    img : np.ndarray
        _description_
    integration_radius : int, optional
        _description_, by default 40
    power : int, optional
        _description_, by default 2

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    h, w = img.shape
    if min(h, w) < 2 * integration_radius:
        raise ValueError("Radius too large for image size")

    bg = np.percentile(img, 99)
    roi_binary = np.zeros(img.shape)
    roi_binary[img > bg] = 1

    cy, cx = center_of_mass(roi_binary)
    # print(f"Center of mass {cy},{cx}")

    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2)
    mask = (dist <= integration_radius).astype(int)

    weighted = img * mask * (dist**power)
    return float(weighted.sum() / mask.sum())


def gauss2d(
    coords_xy: NDArray,
    amplitude: float,
    center_x: float,
    center_y: float,
    sigma_x: float,
    sigma_y: float,
    offset: float,
) -> NDArray:
    """
    Generate a 2D Gaussian function for curve fitting.

    Parameters
    ----------
    coords_xy : NDArray
        Meshgrid coordinates (x, y).
    amplitude : float
        Peak intensity of the Gaussian.
    center_x : float
        X-coordinate of the Gaussian center.
    center_y : float
        Y-coordinate of the Gaussian center.
    sigma_x : float
        Standard deviation along the x-axis.
    sigma_y : float
        Standard deviation along the y-axis.
    offset : float
        Background offset intensity.

    Returns
    -------
    raveled_gauss2d : NDArray
        Flattened 2D Gaussian function values.
    """
    x, y = coords_xy
    raveled_gauss2d = (
        offset
        + amplitude
        * np.exp(
            -(
                ((x - center_x) ** 2 / (2 * sigma_x**2))
                + ((y - center_y) ** 2 / (2 * sigma_y**2))
            )
        )
    ).ravel()

    return raveled_gauss2d


def compute_otf_support_px(img: NDArray, psf_radius_px: float) -> int:
    """
    Compute the optical transfer function (OTF) cutoff frequency.

    Parameters
    ----------
    img : NDArray
        2D image.
    psf_radius_px : float
        Estimated point spread function (PSF) radius in pixels.

    Returns
    -------
    cutoff : int
        OTF cutoff frequency in pixels.
    """
    w = min(img.shape)
    psf_radius_px = max(1, np.ceil(psf_radius_px))  # clip all PSF radii below 1 px to 1
    cutoff = np.ceil(w / (2 * psf_radius_px)).astype(int)

    return cutoff


def normL2(x: NDArray) -> float:
    """
    Compute the L2 norm of an n-dimensional array.

    Parameters
    ----------
    x : NDArray
        Input array.

    Returns
    -------
    l2norm : float
        L2 norm of the array.
    """
    l2norm = np.sqrt(np.sum(x.flatten() ** 2))

    return l2norm


def shannon(spectrum_2d: NDArray, otf_support_px: int = 100) -> float:
    """
    Compute Shannon entropy within an image spectrum's OTF radius.

    Parameters
    ----------
    spectrum_2d : NDArray
        2D spectrum of an image (e.g., from DCT or FFT).
    otf_support_px : int, optional
        OTF support radius in pixels (default is 100).

    Returns
    -------
    entropy : float
        Shannon entropy of the spectrum.
    """
    h, w = spectrum_2d.shape
    y, x = np.ogrid[:h, :w]

    # Circular mask centered at (0,0) for DCT
    support = (x**2 + y**2) < otf_support_px**2
    # Exclude DC component at (0,0)
    # support[0, 0] = False
    spectrum_values = np.abs(spectrum_2d[support])

    # Calculate total energy
    total_energy = np.sum(spectrum_values)
    if total_energy <= 0:
        return 0.0

    # Define probabilities
    p = spectrum_values / total_energy
    p = p[p > 0]

    # Calculate entropy and metric
    entropy = -np.sum(p * np.log2(p))
    metric = np.log10(entropy) if entropy > 0 else 0.0
    return metric


def dct_2d(image: NDArray) -> NDArray:
    """
    Compute the two-dimensional discrete cosine transform of an image.

    Parameters
    ----------
    image : NDArray
        2D image array.

    Returns
    -------
    dct_2d : NDArray
        Transformed image using DCT.
    """
    dct_2d = dct(
        dct(image.astype(np.float32), axis=0, norm="ortho"), axis=1, norm="ortho"
    )

    return dct_2d


def quadratic(x: float | NDArray, a: float, b: float, c: float) -> float | NDArray:
    """
    Quadratic function evaluation at x.

    Parameters
    ----------
    x : float
        Point to evaluate.
    a : float
        x^2 coefficient.
    b : float
        x coefficient.
    c : float
        Offset.

    Returns
    -------
    value : float
        a * x^2 + b * x + c
    """
    return a * x**2 + b * x + c


def quadratic_fit(x: NDArray, y: NDArray) -> Sequence[float]:
    """
    Quadratic function for curve fitting.

    Parameters
    ----------
    x : NDArray
        1D x-axis data.
    y : NDArray
        1D y-axis data.

    Returns
    -------
    coeffs : Sequence[float]
        Fitting parameters.
    """
    A = np.vstack([x**2, x, np.ones_like(x)]).T
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]

    return coeffs


def normalize_roi(roi, bg_percentile=25.0, debug_mode=False):
    """Normalize an image region to the interval from zero to one.

    Scale values between the configured low percentile and the maximum.

    Parameters
    ----------
    roi : numpy.ndarray
        Two-dimensional image region.
    bg_percentile : float
        Percentile used as the background level.
    debug_mode : bool
        Whether to print normalization values.

    Returns
    -------
    numpy.ndarray
        Normalized image region.
    """
    assert len(roi.shape) == 2, "Error: ROI captured by camera must be 2D."
    bg = np.percentile(roi, bg_percentile)
    peak = roi.max()
    roi_normalized = (roi - bg) / (peak - bg)
    roi_normalized = np.clip(roi_normalized, 0, 1)
    if debug_mode:
        print(
            "normalize_roi() values (low_percentile, background, peak):"
            + str(bg_percentile)
            + ",  "
            + f"{bg:2.3f}"
            + ",  "
            + f"{peak:2.3f}"
        )
    return roi_normalized


# -------------------------------------------------#
# Localization methods to generate ROIs for fitting
# -------------------------------------------------#


def localize_2d_img(
    img,
    dxy,
    localize_psf_filters=None,
    save_dir_path: Path = None,
    label: str = "",
    showfig: bool = False,
    verbose: bool = True,
):
    """TODO.

    Parameters
    ----------
    img : _type_
        _description_
    dxy : _type_
        _description_
    localize_psf_filters : dict or None, optional
        Localization thresholds and parameter bounds. Uses standard OPM
        localization bounds when omitted.
    save_dir_path : Path, optional
        _description_, by default None
    label : str, optional
        _description_, by default ""
    showfig : bool, optional
        _description_, by default False
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    from localize_psf.fit_psf import gaussian3d_psf_model
    from localize_psf.localize import (
        get_coords,
        get_param_filter,
        localize_beads_generic,
    )

    if localize_psf_filters is None:
        localize_psf_filters = {
            "threshold": 300,
            "amp_bounds": (200, 50000),
            "sxy_bounds": (0.100, 1.0),
        }

    # Define fitting model and coordinates
    model = gaussian3d_psf_model()
    coords_3d = get_coords((1,) + img.shape, (1, dxy, dxy))
    coords_2d = get_coords(img.shape, (dxy, dxy))

    # Set fit bounds and parameter filters
    threshold = localize_psf_filters["threshold"]
    amp_bounds = localize_psf_filters["amp_bounds"]
    sxy_bounds = localize_psf_filters["sxy_bounds"]
    fit_dist_max_err = (0, dxy * 2)
    fit_roi_size = (1, int(dxy * 9), int(dxy * 9))
    # fit_roi_size = (1, 9, 9)
    min_spot_sep = (0, 1)
    dist_boundary_min = (0, 1.0)

    param_filter = get_param_filter(
        coords_3d,
        fit_dist_max_err=fit_dist_max_err,
        min_spot_sep=min_spot_sep,
        amp_bounds=amp_bounds,
        dist_boundary_min=dist_boundary_min,
        sigma_bounds=((0, sxy_bounds[0]), (1, sxy_bounds[1])),
    )
    # Run localization function
    _, r, _ = localize_beads_generic(
        img,
        (1, dxy, dxy),
        threshold=threshold,
        roi_size=fit_roi_size,
        filter_sigma_small=None,
        filter_sigma_large=None,
        min_spot_sep=min_spot_sep,
        model=model,
        filter=param_filter,
        max_nfit_iterations=100,
        use_gpu_fit=False,
        use_gpu_filter=False,
        return_filtered_images=False,
        fit_filtered_images=False,
        verbose=True,
    )
    save_dir_path = Path(r"F:\optimize_now")
    if save_dir_path:
        plot_2d_localization_fit_summary(r, img, coords_2d, save_dir_path, showfig)

    return r


# -------------------------------------------------#
# Functions to calculate image metrics
# -------------------------------------------------#


def metric_brightness(
    image: NDArray,
    crop_size: int = None,
    threshold: float | None = 100,
    percentile: float | None = None,
    image_center: int = None,
    return_image: bool | None = False,
) -> float:
    """Compute weighted metric for 2D Gaussian.

    Parameters
    ----------
    image : NDArray
        2D image.
    crop_size : int, optional
        Width of the optional square crop in pixels.
    threshold : float, optional
        Initial threshold used to find the spot.
    percentile : float or None
        Optional upper intensity percentile to average.
    image_center : int, optional
        Center of the image to crop (default is None).
    return_image : Optional[bool], optional
        Whether to return the cropped image (default is False).

    Returns
    -------
    weighted_metric : float
        Weighted metric value.
    """
    if crop_size:
        if image_center is None:
            center = get_image_center(image, threshold)
        else:
            center = image_center
        image = get_cropped_image(image, crop_size, center)

    if len(image.shape) == 3:
        image = np.max(image, axis=0)

    if percentile:
        image_perc = np.percentile(image, percentile)
        max_pixels = image[image >= image_perc]
        image_max = np.mean(max_pixels)
    else:
        image_max = np.max(image)

    if return_image:
        return float(image_max), image
    else:
        return float(image_max)


def metric_shannon_dct(
    image: NDArray,
    psf_radius_px: float = 3,
    crop_size: int = None,
    threshold: float | None = 300,
    image_center: int = None,
    return_image: bool | None = False,
) -> float:
    """Compute the Shannon entropy metric using DCT.

    NOTE: The shannon DCT calculation requires a square image!

    Parameters
    ----------
    image : NDArray
        2D image.
    psf_radius_px : float, optional
        Estimated point spread function (PSF) radius in pixels (default: 3).
    crop_size : int, optional
        Crop size for image (default: 501).
    threshold : Optional[float], optional
        Intensity threshold to find the center (default: 100).
    image_center : int, optional
        Custom image center (default: None).
    return_image : Optional[bool], optional
        Whether to return the image along with the metric (default: False).

    Returns
    -------
    entropy_metric : float
        Shannon entropy metric.
    """
    # TODO: Test changes!
    # Crop image if necessary
    if crop_size:
        if image_center is None:
            center = get_image_center(image, threshold)
        else:
            center = image_center
        image = get_cropped_image(image, crop_size, center)

    # if not crop_size:
    #     crop_size = min(image.shape)-1

    # Compute the cutoff frequency based on OTF radius
    cutoff = compute_otf_support_px(image, psf_radius_px)

    # Compute DCT
    dct_result = dct_2d(image)

    # Compute Shannon entropy within the cutoff radius
    shannon_dct = shannon(dct_result, cutoff)

    if return_image:
        return shannon_dct, image
    else:
        return shannon_dct


def metric_laplacian_variance(
    image: NDArray,
    crop_size: int | None = None,
    threshold: float | None = 1000,
    image_center: tuple[int, int] | None = None,
    bg_percentile: float = 10.0,
    normalize: bool = True,
    return_image: bool | None = False,
) -> float:
    """Compute a focus metric from the variance of the image Laplacian.

    Larger values indicate sharper images. The optional background subtraction
    and normalization make the metric less dependent on camera offset and laser
    power while preserving high-frequency structure.

    Parameters
    ----------
    image : NDArray
        Two- or three-dimensional camera image.
    crop_size : int or None
        Width of the optional square crop in pixels.
    threshold : float
        Intensity threshold used to locate the image center.
    image_center : tuple[int, int] or None
        Optional center coordinates for cropping.
    bg_percentile : float
        Percentile subtracted as camera background.
    normalize : bool
        Whether to normalize by the upper intensity percentile.
    return_image : bool
        Whether to return the processed image with the metric.

    Returns
    -------
    float or tuple[float, NDArray]
        Laplacian-variance metric, optionally paired with the processed image.
    """
    if crop_size:
        center = image_center or get_image_center(image, threshold)
        image = get_cropped_image(image, crop_size, center)

    if len(image.shape) == 3:
        image = np.max(image, axis=0)

    image = image.astype(np.float32, copy=False)
    image = image - np.percentile(image, bg_percentile)
    image[image < 0] = 0

    if normalize:
        scale = np.percentile(image, 99.9)
        if not np.isfinite(scale) or scale <= 0:
            metric = np.nan
            return (metric, image) if return_image else metric
        image = image / scale

    laplacian = laplace(image)
    metric = float(np.var(laplacian))
    if not np.isfinite(metric) or metric <= 0:
        metric = np.nan

    if return_image:
        return metric, image
    return metric


def _initial_gaussian_center(image: NDArray, threshold: float) -> tuple[float, float]:
    """Locate the initial spot using an intensity threshold, without a fallback.

    Returns
    -------
    tuple[float, float]
        Intensity-weighted center (x, y) of the above-threshold pixels.

    Raises
    ------
    ValueError
        If the image is invalid or contains no signal above the threshold.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or not image.size or not np.all(np.isfinite(image)):
        raise ValueError("Gaussian AO requires a finite, nonempty 2D image")
    mask = image > threshold
    if not np.any(mask):
        raise ValueError(
            f"Gaussian AO starting image has no spot above threshold {threshold:g}"
        )
    ys, xs = np.nonzero(mask)
    weights = image[mask] - threshold
    return float(np.average(xs, weights=weights)), float(
        np.average(ys, weights=weights)
    )


def metric_gauss2d(
    image: NDArray,
    crop_size: int = 128,
    threshold: float = 1000,
    image_center: tuple[float, float] | None = None,
    return_image: bool = False,
    *,
    center_radius_px: float = 16.0,
) -> float | tuple[float, NDArray]:
    """Fit a single Gaussian spot constrained by its initial AO location.

    Parameters
    ----------
    image : NDArray
        2D image.
    crop_size : int, optional
        Half-width of the fixed crop in pixels; zero uses the full image.
    threshold : float, optional
        Threshold for locating the spot when no initial center is supplied.
        AO determines this center once from its starting image.
    image_center : tuple[float, float] or None
        Initial spot center (x, y), held fixed throughout an AO optimization.
    return_image : bool, optional
        Also return the normalized crop, including when a fit fails.
    center_radius_px : float, optional
        Maximum fitted-center displacement on each axis from the initial
        center. Both Gaussian FWHM widths must exceed one pixel. Fits pinned to
        the center or sigma bounds are invalid rather than rewarded.

    Returns
    -------
    float or tuple[float, NDArray]
        Weighted amplitude/width/symmetry score (NaN for invalid fits),
        optionally with the normalized crop.
    """  # noqa: DOC501 - validation exceptions are caught and logged below.
    image = np.asarray(image, dtype=float)
    weighted_metric = np.nan
    try:
        if image.ndim != 2 or not image.size or not np.all(np.isfinite(image)):
            raise ValueError("Expected a finite, nonempty 2D image")
        center = (
            _initial_gaussian_center(image, threshold)
            if image_center is None
            else image_center
        )
        if (
            len(center) != 2
            or not np.all(np.isfinite(center))
            or not 0 <= center[0] < image.shape[1]
            or not 0 <= center[1] < image.shape[0]
            or not np.isfinite(center_radius_px)
            or center_radius_px <= 0
        ):
            raise ValueError("Invalid initial spot center or center bounding box")
        origin_x = origin_y = 0
        if crop_size:
            crop_center = (int(round(center[0])), int(round(center[1])))
            origin_x = max(crop_center[0] - crop_size, 0)
            origin_y = max(crop_center[1] - crop_size, 0)
            image = get_cropped_image(image, crop_size, crop_center)
        prior_x, prior_y = center[0] - origin_x, center[1] - origin_y
        image_max = np.max(image)
        if image_max <= 0 or np.ptp(image) == 0:
            raise ValueError("No spot contrast in the initial spot crop")
        image = image / image_max
        y, x = np.indices(image.shape, dtype=float)
        background = float(np.median(image))
        # Clip bounds in crop coordinates, including crops at camera edges.
        lower_x = max(0.0, prior_x - center_radius_px)
        upper_x = min(image.shape[1] - 1.0, prior_x + center_radius_px)
        lower_y = max(0.0, prior_y - center_radius_px)
        upper_y = min(image.shape[0] - 1.0, prior_y + center_radius_px)
        # The requested minimum is a full width, while gauss2d fits sigma.
        fwhm_per_sigma = 2.0 * np.sqrt(2.0 * np.log(2.0))
        sigma_floor = (1.0 + 1e-6) / fwhm_per_sigma
        lower = [0, lower_x, lower_y, sigma_floor, sigma_floor, 0]
        upper = [1.5, upper_x, upper_y, 100, 100, 10]
        initial_guess = (
            image.max() - background,
            prior_x,
            prior_y,
            3,
            3,
            max(background, 0.0),
        )
        popt, _ = curve_fit(
            gauss2d,
            (x, y),
            image.ravel(),
            p0=initial_guess,
            bounds=(lower, upper),
            maxfev=1000,
        )
        if not np.all(np.isfinite(popt)):
            raise ValueError("Nonfinite Gaussian fit parameters")
        # A boundary solution can represent a hot pixel or a spot outside
        # the allowed region; neither should drive the mirror correction.
        bound_hits = []
        for index, name in enumerate(("center_x", "center_y", "sigma_x", "sigma_y"), 1):
            for side, bounds in (("lower", lower), ("upper", upper)):
                if np.isclose(popt[index], bounds[index], atol=1e-4, rtol=0):
                    bound_hits.append(
                        f"{name}={popt[index]:.6g} px ({side} bound {bounds[index]:.6g})"
                    )
                    if index in (3, 4):
                        bound_hits[-1] += (
                            f"; FWHM={popt[index] * fwhm_per_sigma:.6g} px"
                            f" ({side} bound {bounds[index] * fwhm_per_sigma:.6g} px)"
                        )
        if bound_hits:
            raise ValueError("Gaussian fit reached a bound: " + "; ".join(bound_hits))
        amplitude, _, _, sigma_x, sigma_y, _ = popt
        sigma_area = sigma_x * sigma_y
        ellipticity = abs(sigma_x - sigma_y) / (sigma_x + sigma_y)
        score = amplitude / sigma_area**2 * np.exp(-4.0 * ellipticity**2)
        if not np.isfinite(score) or score <= 0:
            raise ValueError("Invalid Gaussian metric")
        weighted_metric = float(score)
    except Exception as exc:
        warning("AO GAUSSIAN FIT INVALID", str(exc))

    return (weighted_metric, image) if return_image else weighted_metric


def metric_localize_gauss2d(image: NDArray) -> float:
    """TODO.

    Parameters
    ----------
    image : NDArray
        _description_

    Returns
    -------
    float
        _description_
    """
    try:
        fit_results = localize_2d_img(
            image,
            0.115,
            {
                "threshold": 1000,
                "amp_bounds": (1000, 65000),
                "sxy_bounds": (0.100, 2.0),
            },
            save_dir_path=None,
            label="",
            showfig=False,
            verbose=False,
        )

        to_keep = fit_results["to_keep"]
        sxy = fit_results["fit_params"][to_keep, 4]
        metric = 1.0 / np.median(sxy)
    except Exception as e:
        print(f"2d localization and fit exceptions: {e}")
        metric = np.nan

    return metric


# -------------------------------------------------#
# Helper function for generating grid
# -------------------------------------------------#


def run_ao_grid_mapping(
    ao_dict: dict,
    stage_positions: list,
    position_indices: list[int] | None = None,
    previous_grid_coefficients: NDArray | None = None,
    previous_grid_reference_state: str | None = None,
    num_tile_positions: int = 1,
    num_scan_positions: int = 1,
    save_dir_path: Path = None,
    verbose: bool = True,
    correction_history: AOCorrectionHistory | None = None,
    time_idx: int = 0,
    z_idx: int = 0,
    stop_requested=None,
) -> AOGridResult | None:
    """Run adaptive optics on a grid and interpolate across stage positions.

    Parameters
    ----------
    stage_positions : list
        Experimental stage positions. Optimized for stage scan acquisitions
    position_indices : list[int] or None
        Global AO position-array indices represented by ``stage_positions``.
        ``None`` maps the positions contiguously from index zero.
    previous_grid_coefficients : numpy.ndarray or None
        Coarse AO-grid coefficients from the preceding logical Z level, in the
        same grid-point order. ``None`` starts every AO point from the GUI state.
    previous_grid_reference_state : str or None
        Mirror reference associated with ``previous_grid_coefficients``.
    ao_dict : dict
        A dictionary containing AO optimization parameters
    num_tile_positions : int
        Number of lateral tile positions represented in the grid.
    num_scan_positions : int
        Number of scan-axis positions represented in the grid.
    save_dir_path : Path, optional
        Path to save AO optimization data. Default is None.
    verbose : bool, optional
        If True, prints additional debugging information. Default is False.

    correction_history : AOCorrectionHistory or None
        Acquisition-scoped initializer and completed-result history.
    time_idx : int
        Logical acquisition timepoint.
    z_idx : int
        Logical acquisition depth layer.

    Returns
    -------
    AOGridResult
        Exact optimized states on this coarse AO grid.

    Raises
    ------
    RuntimeError
        If a grid-point optimization fails to complete.
    KeyError
        If the AO configuration omits a required optimization setting.
    ValueError
        If global position indices do not match the supplied stage positions.
    """
    required_ao_keys = {
        "channel_states",
        "daq_mode",
        "exposure_ms",
        "image_mirror_range_um",
        "iterations",
        "metric",
        "metric_acceptance",
        "metric_precision",
        "mirror_state",
        "modal_alpha",
        "modal_delta",
        "modes_to_optimize",
        "num_averaged_frames",
        "num_mode_samples",
    }
    missing_ao_keys = required_ao_keys.difference(ao_dict)
    if missing_ao_keys:
        missing = ", ".join(sorted(missing_ao_keys))
        raise KeyError(f"AO grid ao_dict is missing required keys: {missing}")

    grid_counts = {
        "num_scan_positions": num_scan_positions,
        "num_tile_positions": num_tile_positions,
    }
    normalized_counts: dict[str, int] = {}
    for name, value in grid_counts.items():
        numeric_value = float(value)
        if (
            not np.isfinite(numeric_value)
            or not numeric_value.is_integer()
            or numeric_value < 1
        ):
            raise ValueError(f"{name} must be a positive integer; received {value!r}")
        normalized_counts[name] = int(numeric_value)
    num_scan_positions = normalized_counts["num_scan_positions"]
    num_tile_positions = normalized_counts["num_tile_positions"]

    if verbose:
        print(
            "\n++++++++++++++++ RUNNING SENSORLESS GRID AO ++++++++++++++++\n",
            f"\nAO save directory: {save_dir_path}",
            f"\nNumber of experiment stage positions: {len(stage_positions)}\n",
        )
    AOMirror_local = AOMirror.instance()
    mmc = CMMCorePlus.instance()
    if position_indices is None:
        position_indices = list(range(len(stage_positions)))
    if len(position_indices) != len(stage_positions):
        raise ValueError(
            "AO grid position indices must match the number of stage positions"
        )

    ao_stage_positions = ao_grid_positions(
        stage_positions,
        num_scan_positions=num_scan_positions,
        num_tile_positions=num_tile_positions,
    )
    num_scan_positions = len({position["x"] for position in ao_stage_positions})
    num_tile_positions = len({position["y"] for position in ao_stage_positions})

    # Save AO optimization results here
    num_ao_pos = len(ao_stage_positions)
    ao_grid_wfc_coeffs = np.zeros((
        num_ao_pos,
        AOMirror_local.positions_modal_array.shape[1],
    ))
    ao_grid_wfc_positions = np.zeros((
        num_ao_pos,
        AOMirror_local.positions_voltage_array.shape[1],
    ))
    if previous_grid_coefficients is not None:
        previous_grid_coefficients = np.asarray(previous_grid_coefficients)
        if previous_grid_coefficients.shape != ao_grid_wfc_coeffs.shape:
            raise ValueError(
                "Previous-Z AO grid coefficients must match the current coarse grid"
            )
        if previous_grid_reference_state is None:
            raise ValueError(
                "Previous-Z AO grid coefficients require a mirror reference state"
            )

    mirror_state = str(ao_dict["mirror_state"])
    if "factory" in mirror_state:
        grid_reference_state = "factory_flat"
        gui_start_coefficients = np.zeros_like(AOMirror_local.current_coeffs)
    elif "zeros" in mirror_state:
        grid_reference_state = "zeros_voltage"
        gui_start_coefficients = np.zeros_like(AOMirror_local.current_coeffs)
    elif "optimized" in mirror_state:
        grid_reference_state = getattr(AOMirror_local, "optimized_reference_state", "system_flat")
        gui_start_coefficients = AOMirror_local.optimized_modal_coeffs.copy()
    elif "system" in mirror_state:
        grid_reference_state = "system_flat"
        gui_start_coefficients = np.zeros_like(AOMirror_local.current_coeffs)
    else:
        raise ValueError(f"Invalid starting mirror state: {mirror_state}")
    if previous_grid_reference_state is not None:
        grid_reference_state = previous_grid_reference_state

    if correction_history is not None:
        grid_reference_state = correction_history.reference_state

    # Run AO optimization for each stage position
    if verbose:
        print(
            "---------------- AO GRID GENERATION ---------------",
            f"\nNumber of positions for AO GRID: {num_ao_pos}",
            f"\nNum. AO Tile positions (Y): {num_tile_positions}",
            f"\nNum. AO Scan positions (X): {num_scan_positions}\n",
        )

    if save_dir_path is None:
        save_dir_path = Path(r"F:\optimize_now\grid_ao_optimizeNOW")
        save_dir_path.mkdir(exist_ok=True)

    print(
        "---------------- AO Save path ---------------"
        f"\nSaving AO grid mapping data to: {save_dir_path} \n"
    )
    # save stage positions
    import json

    ao_positions_path = save_dir_path / Path("ao_stage_positions.json")
    exp_positions_path = save_dir_path / Path("exp_stage_positions.json")
    with open(ao_positions_path, "w") as file:
        json.dump(ao_stage_positions, file, indent=4)
    with open(exp_positions_path, "w") as file:
        json.dump(stage_positions, file, indent=4)

    # Visit AO stage positions and measure best WFC positions
    if verbose:
        print("\n\nStarting AO grid optimization . . .")
    for ao_pos_idx in range(num_ao_pos):
        if stop_requested is not None and stop_requested():
            return None
        if verbose:
            print(
                f"+++ \nMoving stage to position {ao_pos_idx + 1}/{num_ao_pos}: \n{ao_stage_positions[ao_pos_idx]}+++"
            )

        target_x = ao_stage_positions[ao_pos_idx]["x"]
        target_y = ao_stage_positions[ao_pos_idx]["y"]
        target_z = ao_stage_positions[ao_pos_idx]["z"]

        mmc.setPosition(np.round(target_z, 2))
        mmc.waitForDevice(mmc.getFocusDevice())
        mmc.setXYPosition(np.round(float(target_x), 2), np.round(float(target_y), 2))
        current_x, current_y = mmc.getXYPosition()

        while not (np.isclose(current_x, target_x, 0.0, 1.0)) or not (
            np.isclose(current_y, target_y, 0.0, 1.0)
        ):
            current_x, current_y = mmc.getXYPosition()
            sleep(0.5)

        current_save_dir = save_dir_path / Path(f"grid_pos_{int(ao_pos_idx)}")
        current_save_dir.mkdir(exist_ok=True)

        context = {
            "time_idx": time_idx,
            "z_idx": z_idx,
            "position": ("grid", num_scan_positions, num_tile_positions, ao_pos_idx),
        }
        initializer = (
            correction_history.initializer(**context)
            if correction_history is not None
            else {
                "starting_mirror_coefficients": (
                    gui_start_coefficients
                    if previous_grid_coefficients is None
                    else previous_grid_coefficients[ao_pos_idx]
                ),
                "starting_reference_state": grid_reference_state,
                "starting_coefficients_source": (
                    "GUI state" if previous_grid_coefficients is None else "previous Z"
                ),
            }
        )
        completed = run_ao_optimization(
            stop_requested=stop_requested,
            exposure_ms=ao_dict["exposure_ms"],
            channel_states=ao_dict["channel_states"],
            metric_to_use=ao_dict["metric"],
            daq_mode=ao_dict["daq_mode"],
            image_mirror_range_um=ao_dict["image_mirror_range_um"],
            num_iterations=ao_dict["iterations"],
            num_mode_samples=ao_dict["num_mode_samples"],
            starting_coef_delta=ao_dict["modal_delta"],
            coef_delta_scale=ao_dict["modal_alpha"],
            metric_precision=ao_dict["metric_precision"],
            modes_to_optimize=ao_dict["modes_to_optimize"],
            starting_mirror_state=mirror_state,
            **initializer,
            mode_acceptance=ao_dict["metric_acceptance"],
            num_averaged_frames=ao_dict["num_averaged_frames"],
            o3_stack_settings=ao_dict.get("o3_stack_settings"),
            pos_idx=None,
            save_dir_path=current_save_dir,
            verbose=verbose,
        )
        if completed is not True:
            if stop_requested is not None and stop_requested():
                return None
            raise RuntimeError("AO grid optimization did not complete")
        if correction_history is not None:
            correction_history.complete(AOMirror_local.current_coeffs, **context)
        ao_grid_wfc_coeffs[ao_pos_idx] = AOMirror_local.current_coeffs.copy()
        ao_grid_wfc_positions[ao_pos_idx] = AOMirror_local.current_voltage.copy()

    if verbose:
        print("\n++++++++++++++++ AO GRID OPTIMIZATION COMPLETE ++++++++++++++++\n")

    # Map ao_grid_wfc_coeffs to experiment stage positions.
    position_wfc_coeffs = AOMirror_local.positions_modal_array.copy()
    position_wfc_positions = AOMirror_local.positions_voltage_array.copy()
    ao_grid_indices = nearest_ao_grid_indices(
        stage_positions,
        ao_stage_positions,
    )
    for local_pos_idx, ao_grid_idx in enumerate(ao_grid_indices):
        # Assign AO data
        pos_idx = int(position_indices[local_pos_idx])
        position_wfc_positions[pos_idx] = ao_grid_wfc_positions[ao_grid_idx]
        position_wfc_coeffs[pos_idx] = ao_grid_wfc_coeffs[ao_grid_idx]

        if DEBUGGING:
            print(
                f"\n\n ++++ AO grid position: {ao_stage_positions[ao_grid_idx]} ++++",
                f"\n ++++ Exp. stage position: "
                f"{stage_positions[local_pos_idx]} ++++",
            )
    AOMirror_local.positions_modal_array = position_wfc_coeffs
    AOMirror_local.positions_voltage_array = position_wfc_positions

    if verbose:
        print("\nAO grid mapping complete. Mirror positions array updated.")
    return AOGridResult(
        modal_coefficients=ao_grid_wfc_coeffs.copy(),
        actuator_positions=ao_grid_wfc_positions.copy(),
        reference_state=grid_reference_state,
    )


# -------------------------------------------------#
# Helper functions for saving optmization results
# -------------------------------------------------#


def save_optimization_results(
    all_images: NDArray,
    all_metrics: NDArray,
    optimal_images: NDArray,
    optimal_metrics: NDArray,
    optimal_coeffs: NDArray,
    starting_coeffs: NDArray,
    starting_metric: float,
    starting_image: NDArray,
    update_status: NDArray,
    metadata: dict,
    save_dir_path: Path,
    o3_stacks: NDArray | None = None,
) -> None:
    """Save the results from running AO-optimize.

    NOTE: optimal_images has been changed to optimal images
    NOTE: Likewise for optimal_metrics

    Parameters
    ----------
    all_images : NDArray
        all images acquired during optimization
    all_metrics : NDArray
        all metrics measured during optimization
    optimal_images : NDArray
        optimal images per interation, including the starting image
    optimal_metrics : NDArray
        optmial metrics per iteration, including the starting metric
    optimal_coeffs : NDArray
        optimal coefficients per iteration
    starting_coeffs : NDArray
        Mirror coefficients at the start of optimization.
    starting_metric : float
        Image metric at the start of optimization.
    starting_image : NDArray
        Camera image at the start of optimization.
    update_status : NDArray
        Acceptance status for each attempted modal update.
    metadata : Dict
        run_optimization parameters
    save_dir_path : Path
        zarr destination path
    o3_stacks : NDArray or None
        Every measured O3 stack, including rejected candidate measurements.
        Indices correspond to the O3 measurement diagnostics in metadata.
    """
    results_path = save_dir_path / Path("ao_results.zarr")
    # Validate before mode="w" can replace an existing result store.
    attributes = json_metadata({**metadata, "zernike_mode_names": list(mode_names)})
    root = zarr.open_group(
        str(results_path),
        mode="w",
        attributes=attributes,
    )

    # Create datasets in the Zarr store
    root.create_array("all_images", data=all_images)
    root.create_array("all_metrics", data=all_metrics)
    root.create_array("optimal_images", data=optimal_images)
    root.create_array("optimal_metrics", data=optimal_metrics)
    root.create_array("optimal_coeffs", data=optimal_coeffs)
    root.create_array("starting_coeffs", data=starting_coeffs)
    root.create_array("starting_metric", data=np.asarray(starting_metric))
    root.create_array("starting_image", data=starting_image)
    root.create_array("update_status", data=update_status)
    if o3_stacks is not None:
        root.create_array("o3_measured_stacks", data=o3_stacks)


def load_optimization_results(results_path: Path):
    """Load optimization results from a Zarr store.

    Parameters
    ----------
    results_path : Path
        Path to the Zarr directory containing the data.

    Returns
    -------
    dict
        Stored image arrays, metrics, coefficients, status, and metadata.
    """
    results = zarr.open_group(str(results_path), mode="r")

    all_images = results["all_images"][:]
    all_metrics = results["all_metrics"][:]
    optimal_images = results["optimal_images"][:]
    optimal_metrics = results["optimal_metrics"][:]
    optimal_coeffs = results["optimal_coeffs"][:]
    starting_coeffs = results["starting_coeffs"][:]
    starting_metric = np.asarray(results["starting_metric"][...]).squeeze().item()
    starting_image = results["starting_image"][:]
    update_status = results["update_status"][:]
    metadata = dict(results.attrs)
    if "zernike_mode_names" in metadata:
        zernike_mode_names = list(metadata["zernike_mode_names"])
    else:
        zernike_mode_names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in results["zernike_mode_names"][:]
        ]

    ao_results = {
        "all_images": all_images,
        "all_metrics": all_metrics,
        "optimal_metrics": optimal_metrics,
        "optimal_images": optimal_images,
        "optimal_coeffs": optimal_coeffs,
        "starting_coeffs": starting_coeffs,
        "starting_metric": float(starting_metric),
        "starting_image": starting_image,
        "update_status": update_status,
        "mode_names": zernike_mode_names,
        "metadata": metadata,
    }
    ao_results.update(results.attrs)

    return ao_results


# -------------------------------------------------#
# Run to "keeps mirror flat"
# -------------------------------------------------#

if __name__ == "__main__":
    """Keeps the mirror in it"s flat position
    """
    wfc_config_file_path = Path(
        r"C:\Users\qi2lab\Documents\github\opm_ao\Configuration Files\WaveFrontCorrector_mirao52-e_0329.dat"
    )
    wfc_correction_file_path = Path(
        r"C:\Users\qi2lab\Documents\github\opm_ao\OUT_FILES\correction_data_backup_starter.aoc"
    )
    haso_config_file_path = Path(
        r"C:\Users\qi2lab\Documents\github\opm_ao\Configuration Files\WFS_HASO4_VIS_7635.dat"
    )
    wfc_flat_file_path = Path(
        r"C:\Users\qi2lab\Documents\github\opm_ao\OUT_FILES\flat_actuator_positions.wcs"
    )
    wfc_calibrated_flat_path = Path(
        r"C:\Users\qi2lab\Documents\github\opm_ao\OUT_FILES\20250215_tilted_brightness_laser_actuator_positions.wcs"
    )

    # ao_mirror puts the mirror in the flat_position state to start.
    ao_mirror = AOMirror(
        wfc_config_file_path=wfc_config_file_path,
        haso_config_file_path=haso_config_file_path,
        interaction_matrix_file_path=wfc_correction_file_path,
        system_flat_file_path=wfc_calibrated_flat_path,
    )

    input("Press enter to exit . . . ")
    ao_mirror = None
