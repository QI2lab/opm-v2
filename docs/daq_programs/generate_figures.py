"""Generate publication-ready diagrams of the OPM DAQ programs.

The waveform panel is generated from :class:`OPMNIDAQ` in simulated mode, so
the digital and analog arrays are the arrays used by the hardware controller.
The scan-axis position trajectory is schematic because it is produced by the
external ASI controller rather than the NI-DAQ.

Run from any directory with::

    .venv/Scripts/python.exe docs/daq_programs/generate_figures.py
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

REPOSITORY = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
sys.path.insert(0, str(REPOSITORY / "src"))
OPMNIDAQ = importlib.import_module("opm_v2.hardware.OPMNIDAQ").OPMNIDAQ


# Okabe-Ito palette: distinguishable for common forms of color-vision deficiency.
COLORS = {
    "ink": "#222222",
    "muted": "#6B7280",
    "grid": "#D8DEE7",
    "camera": "#222222",
    "405": "#0072B2",
    "488": "#009E73",
    "561": "#E69F00",
    "image": "#CC79A7",
    "projection": "#D55E00",
    "stage": "#56B4E9",
    "light_blue": "#E8F3F8",
    "light_orange": "#FFF1DF",
    "light_green": "#E7F4EE",
    "light_purple": "#F4EAF2",
}

STAGE_CHANNELS = 3
STAGE_PLANE_SPACING_UM = 0.4
STAGE_REPEAT_GROUPS = 4
MIRROR_STAGE_READOUT_LINES = 512
PROJECTION_READOUT_LINES = 2304
FUSION_BT_FAST_LINE_TIME_US = 4.867647
FUSION_BT_EXPOSURE_OFFSET_US = 3.029411
MIRROR_STAGE_REQUESTED_EXPOSURE_MS = 12.0
PROJECTION_REQUESTED_EXPOSURE_MS = 150.0


def _fusion_bt_fast_timing(
    requested_exposure_ms: float,
    readout_lines: int,
) -> tuple[float, float, float]:
    """Calculate Fusion BT Fast Scan timing.

    Returns
    -------
    tuple
        Frame period, readout interval, and global-exposure high time in
        milliseconds.
    """
    exposure_steps = math.ceil(
        (requested_exposure_ms * 1000 - FUSION_BT_EXPOSURE_OFFSET_US)
        / FUSION_BT_FAST_LINE_TIME_US
    )
    frame_period_ms = (
        exposure_steps * FUSION_BT_FAST_LINE_TIME_US + FUSION_BT_EXPOSURE_OFFSET_US
    ) / 1000
    readout_ms = readout_lines * FUSION_BT_FAST_LINE_TIME_US / 1000
    return frame_period_ms, readout_ms, frame_period_ms - readout_ms


(
    MIRROR_STAGE_FRAME_PERIOD_MS,
    MIRROR_STAGE_READOUT_MS,
    MIRROR_STAGE_EXPOSURE_OUT_HIGH_MS,
) = _fusion_bt_fast_timing(
    MIRROR_STAGE_REQUESTED_EXPOSURE_MS,
    MIRROR_STAGE_READOUT_LINES,
)
(
    PROJECTION_FRAME_PERIOD_MS,
    PROJECTION_READOUT_MS,
    PROJECTION_EXPOSURE_OUT_HIGH_MS,
) = _fusion_bt_fast_timing(
    PROJECTION_REQUESTED_EXPOSURE_MS,
    PROJECTION_READOUT_LINES,
)
STAGE_FRAME_PITCH_UM = STAGE_PLANE_SPACING_UM / STAGE_CHANNELS
STAGE_SPEED_UM_S = STAGE_FRAME_PITCH_UM / MIRROR_STAGE_FRAME_PERIOD_MS * 1000
STAGE_ILLUMINATED_TRAVEL_UM = (
    STAGE_SPEED_UM_S * MIRROR_STAGE_EXPOSURE_OUT_HIGH_MS / 1000
)


def configure_matplotlib() -> None:
    """Apply journal-friendly typography and vector-output settings."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def build_waveforms() -> dict[str, dict[str, np.ndarray | float | int]]:
    """Generate and validate representative arrays with production DAQ code.

    Returns
    -------
    dict
        Analog and digital arrays plus acquisition parameters, keyed by mode.
    """
    OPMNIDAQ.reset_instance()
    daq = OPMNIDAQ(
        scan_type="2d",
        exposure_ms=MIRROR_STAGE_REQUESTED_EXPOSURE_MS,
        laser_blanking=True,
        image_mirror_calibration=0.0433,
        projection_mirror_calibration=0.00566,
        image_mirror_neutral_v=0.0,
        projection_mirror_neutral_v=0.0,
        image_mirror_step_um=1.0,
        simulate=True,
    )
    states = [True, True, True, False, False]
    result: dict[str, dict[str, np.ndarray | float | int]] = {}
    for mode in ("2d", "mirror", "projection", "stage"):
        exposure_ms = (
            PROJECTION_REQUESTED_EXPOSURE_MS
            if mode == "projection"
            else MIRROR_STAGE_REQUESTED_EXPOSURE_MS
        )
        daq.set_acquisition_params(
            scan_type=mode,
            channel_states=states,
            image_mirror_range_um=4.0,
            image_mirror_step_um=1.0,
            exposure_ms=exposure_ms,
            laser_blanking=True,
        )
        daq.generate_waveforms()
        result[mode] = {
            "digital": daq.digital_waveform,
            "analog": daq.analog_waveform,
            "n_scan_steps": daq.n_scan_steps,
            "exposure_ms": daq.exposure_ms,
            "sample_rate_hz": daq._daq_sample_rate_hz,
        }

    channels = 3
    planes = 4
    assert result["2d"]["digital"].shape == (2 * channels, 8)
    assert result["stage"]["digital"].shape == (2 * channels, 8)
    assert result["mirror"]["digital"].shape == (2 * channels * planes, 8)
    assert result["mirror"]["analog"].shape == (2 * channels * planes, 2)
    assert result["projection"]["analog"].shape == (1501, 2)
    assert np.allclose(result["2d"]["analog"], 0.0)
    assert np.allclose(result["stage"]["analog"], 0.0)
    assert np.allclose(result["mirror"]["analog"][:, 1], 0.0)
    assert np.allclose(
        result["projection"]["analog"][-1],
        result["projection"]["analog"][0],
    )
    return result


def _lane(values: np.ndarray, lane_index: int, amplitude: float = 0.66) -> np.ndarray:
    """Scale a trace into one timing-diagram lane.

    Returns
    -------
    numpy.ndarray
        Values normalized and offset into the requested lane.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.allclose(values, values[0]):
        scaled = np.zeros_like(values)
    else:
        scaled = (values - values.min()) / (values.max() - values.min())
    return lane_index + amplitude * scaled


def _plot_discrete_mode(
    ax: plt.Axes,
    _mode: str,
    payload: dict[str, np.ndarray | float | int],
) -> None:
    """Plot a camera-edge-clocked Mirror Sweep program."""
    digital = np.asarray(payload["digital"])
    analog = np.asarray(payload["analog"])
    samples = digital.shape[0]
    frame_count = samples // 2
    frame_starts = np.arange(frame_count) * MIRROR_STAGE_FRAME_PERIOD_MS
    exposure_ends = frame_starts + MIRROR_STAGE_EXPOSURE_OUT_HIGH_MS
    x = np.ravel(np.column_stack((frame_starts, exposure_ends)))
    x = np.r_[x, frame_count * MIRROR_STAGE_FRAME_PERIOD_MS]
    camera = np.r_[np.tile([1.0, 0.0], frame_count), 0.0]

    traces: list[tuple[np.ndarray, str, str, str]] = [
        (camera, "Camera exposure out", COLORS["camera"], "-"),
        (np.r_[digital[:, 0], digital[-1, 0]], "Laser 405", COLORS["405"], "-"),
        (np.r_[digital[:, 1], digital[-1, 1]], "Laser 488", COLORS["488"], "-"),
        (np.r_[digital[:, 2], digital[-1, 2]], "Laser 561", COLORS["561"], "-"),
    ]

    if analog.shape[0] == samples:
        image_galvo = np.r_[analog[:, 0], analog[-1, 0]]
        projection_galvo = np.r_[analog[:, 1], analog[-1, 1]]
    else:
        image_galvo = np.full(samples + 1, analog[0, 0])
        projection_galvo = np.full(samples + 1, analog[0, 1])
    stage = np.zeros(samples + 1)
    stage_label = "Stage scan axis position"
    traces.extend([
        (image_galvo, "Image galvo", COLORS["image"], "-"),
        (projection_galvo, "Projection galvo", COLORS["projection"], "-"),
        (stage, stage_label, COLORS["stage"], "--"),
    ])

    lane_positions = np.arange(len(traces))[::-1]
    for lane_position, (values, label, color, linestyle) in zip(lane_positions, traces):
        ax.step(
            x,
            _lane(values, int(lane_position)),
            where="post",
            color=color,
            linestyle=linestyle,
            solid_capstyle="butt",
        )

    group_positions = (
        np.arange(frame_count // STAGE_CHANNELS + 1)
        * STAGE_CHANNELS
        * MIRROR_STAGE_FRAME_PERIOD_MS
    )
    ax.set_xlim(0, frame_count * MIRROR_STAGE_FRAME_PERIOD_MS)
    ax.set_ylim(-0.15, len(traces) - 0.1)
    ax.set_yticks(lane_positions + 0.28)
    ax.set_yticklabels([item[1] for item in traces])
    ax.set_xticks(group_positions)
    ax.set_xticklabels([f"{position:.0f}" for position in group_positions])
    ax.set_xticks(frame_starts, minor=True)
    ax.set_xlabel("Time from first exposure (ms)")
    ax.grid(axis="x", which="major", color=COLORS["grid"], linewidth=0.65)
    ax.grid(axis="x", which="minor", color=COLORS["grid"], linewidth=0.35, alpha=0.6)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", which="minor", length=2)
    ax.spines[["top", "right", "left"]].set_visible(False)


def _plot_stage_mode(
    ax: plt.Axes,
    payload: dict[str, np.ndarray | float | int],
) -> None:
    """Plot repeated interleaved channels during constant-speed stage motion."""
    base_digital = np.asarray(payload["digital"])
    analog = np.asarray(payload["analog"])
    frame_count = STAGE_REPEAT_GROUPS * STAGE_CHANNELS
    scan_distance_um = STAGE_REPEAT_GROUPS * STAGE_PLANE_SPACING_UM
    exposure_out_duty_cycle = (
        MIRROR_STAGE_EXPOSURE_OUT_HIGH_MS / MIRROR_STAGE_FRAME_PERIOD_MS
    )

    x_points: list[float] = []
    camera_points: list[float] = []
    laser_points: list[list[float]] = [[], [], []]
    for frame_index in range(frame_count):
        frame_start = frame_index * STAGE_FRAME_PITCH_UM
        exposure_out_end = frame_start + (
            STAGE_FRAME_PITCH_UM * exposure_out_duty_cycle
        )
        x_points.extend([frame_start, exposure_out_end])
        camera_points.extend([1.0, 0.0])
        active_channel = frame_index % STAGE_CHANNELS
        for channel_index in range(STAGE_CHANNELS):
            laser_points[channel_index].extend([
                float(channel_index == active_channel),
                0.0,
            ])

    x = np.asarray([*x_points, scan_distance_um])
    camera = np.asarray([*camera_points, 0.0])
    lasers = [np.asarray([*points, 0.0]) for points in laser_points]
    stage_position = x.copy()

    traces: list[tuple[np.ndarray, str, str, str]] = [
        (camera, "Camera exposure out", COLORS["camera"], "-"),
        (lasers[0], "Laser 405", COLORS["405"], "-"),
        (lasers[1], "Laser 488", COLORS["488"], "-"),
        (lasers[2], "Laser 561", COLORS["561"], "-"),
        (
            np.full(x.size, analog[0, 0]),
            "Image galvo",
            COLORS["image"],
            "-",
        ),
        (
            np.full(x.size, analog[0, 1]),
            "Projection galvo",
            COLORS["projection"],
            "-",
        ),
        (
            stage_position,
            "Stage position (constant speed)",
            COLORS["stage"],
            "--",
        ),
    ]

    lane_positions = np.arange(len(traces))[::-1]
    for lane_position, (values, label, color, linestyle) in zip(lane_positions, traces):
        if label == "Stage position (constant speed)":
            ax.plot(
                x,
                _lane(values, int(lane_position)),
                color=color,
                linestyle=linestyle,
                solid_capstyle="butt",
            )
        else:
            ax.step(
                x,
                _lane(values, int(lane_position)),
                where="post",
                color=color,
                linestyle=linestyle,
                solid_capstyle="butt",
            )

    group_positions = np.arange(STAGE_REPEAT_GROUPS + 1) * STAGE_PLANE_SPACING_UM
    exposure_positions = (
        np.arange(STAGE_REPEAT_GROUPS * STAGE_CHANNELS + 1)
        * STAGE_PLANE_SPACING_UM
        / STAGE_CHANNELS
    )
    ax.set_xlim(0.0, scan_distance_um)
    ax.set_ylim(-0.15, len(traces) - 0.1)
    ax.set_yticks(lane_positions + 0.28)
    ax.set_yticklabels([item[1] for item in traces])
    ax.set_xticks(group_positions)
    ax.set_xticklabels([f"{position:.1f}" for position in group_positions])
    ax.set_xticks(exposure_positions, minor=True)
    ax.set_xlabel("Stage scan axis position (µm)")
    ax.grid(axis="x", which="major", color=COLORS["grid"], linewidth=0.65)
    ax.grid(axis="x", which="minor", color=COLORS["grid"], linewidth=0.35, alpha=0.6)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", which="minor", length=2)
    ax.spines[["top", "right", "left"]].set_visible(False)

    assert np.array_equal(base_digital[::2, :STAGE_CHANNELS], np.eye(STAGE_CHANNELS))


def _plot_projection_mode(
    ax: plt.Axes,
    payload: dict[str, np.ndarray | float | int],
) -> None:
    """Plot projection AO ramps expanded inside each camera exposure."""
    digital = np.asarray(payload["digital"])
    analog = np.asarray(payload["analog"])
    channels = 3
    points_per_frame = analog.shape[0]
    x_parts: list[np.ndarray] = []
    camera_parts: list[np.ndarray] = []
    laser_parts: list[list[np.ndarray]] = [[], [], []]
    image_parts: list[np.ndarray] = []
    projection_parts: list[np.ndarray] = []

    for channel in range(channels):
        frame_start = channel * PROJECTION_FRAME_PERIOD_MS
        frame_end = (channel + 1) * PROJECTION_FRAME_PERIOD_MS
        frame_x = np.linspace(frame_start, frame_end, points_per_frame, endpoint=True)
        exposure_out_high = (
            frame_x - frame_start < PROJECTION_EXPOSURE_OUT_HIGH_MS
        ).astype(float)
        x_parts.append(frame_x)
        camera_parts.append(exposure_out_high)
        for laser in range(channels):
            laser_parts[laser].append(exposure_out_high * float(laser == channel))
        image_parts.append(analog[:, 0])
        projection_parts.append(analog[:, 1])

    x = np.concatenate(x_parts)
    traces: list[tuple[np.ndarray, str, str, str]] = [
        (
            np.concatenate(camera_parts),
            "Camera exposure out",
            COLORS["camera"],
            "-",
        ),
        (np.concatenate(laser_parts[0]), "Laser 405", COLORS["405"], "-"),
        (np.concatenate(laser_parts[1]), "Laser 488", COLORS["488"], "-"),
        (np.concatenate(laser_parts[2]), "Laser 561", COLORS["561"], "-"),
        (np.concatenate(image_parts), "Image galvo", COLORS["image"], "-"),
        (
            np.concatenate(projection_parts),
            "Projection galvo",
            COLORS["projection"],
            "-",
        ),
        (np.zeros_like(x), "Stage scan axis position", COLORS["stage"], "--"),
    ]
    lane_positions = np.arange(len(traces))[::-1]
    for lane_position, (values, _label, color, linestyle) in zip(
        lane_positions, traces
    ):
        ax.plot(
            x,
            _lane(values, int(lane_position)),
            color=color,
            linestyle=linestyle,
            solid_capstyle="butt",
        )
    total_time_ms = channels * PROJECTION_FRAME_PERIOD_MS
    for edge in np.arange(channels + 1) * PROJECTION_FRAME_PERIOD_MS:
        ax.axvline(edge, color=COLORS["grid"], linewidth=0.55, zorder=0)

    ax.set_xlim(0, total_time_ms)
    ax.set_ylim(-0.15, len(traces) - 0.1)
    ax.set_yticks(lane_positions + 0.28)
    ax.set_yticklabels([item[1] for item in traces])
    ax.set_xticks((np.arange(channels) + 0.5) * PROJECTION_FRAME_PERIOD_MS)
    ax.set_xticklabels(["405 frame", "488 frame", "561 frame"])
    ax.set_xlabel("One mirror sweep per 150 ms camera frame")
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)

    # Confirm the plotted laser order is the production digital program.
    assert np.array_equal(digital[::2, :3], np.eye(channels, dtype=np.uint8))


def make_waveform_figure(
    waveforms: dict[str, dict[str, np.ndarray | float | int]],
) -> plt.Figure:
    """Create the three-panel scanning-program timing figure.

    Returns
    -------
    matplotlib.figure.Figure
        Three vertically stacked timing diagrams.
    """
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.0), constrained_layout=True)
    panel_specs = [
        (
            "mirror",
            "a",
            "Mirror Sweep",
            (
                f"12 ms, 512 lines: {MIRROR_STAGE_EXPOSURE_OUT_HIGH_MS:.2f} ms "
                f"high + {MIRROR_STAGE_READOUT_MS:.2f} ms gap"
            ),
        ),
        (
            "projection",
            "b",
            "Projection",
            (
                f"150 ms, 2304 lines: {PROJECTION_EXPOSURE_OUT_HIGH_MS:.2f} ms "
                f"high + {PROJECTION_READOUT_MS:.2f} ms gap"
            ),
        ),
        (
            "stage",
            "c",
            "Stage scan",
            (
                f"EXPOSURE OUT: {MIRROR_STAGE_EXPOSURE_OUT_HIGH_MS:.2f} ms high + "
                f"{MIRROR_STAGE_READOUT_MS:.2f} ms gap; illuminated travel "
                f"{STAGE_ILLUMINATED_TRAVEL_UM:.2f} µm"
            ),
        ),
    ]
    for ax, (mode, letter, title, equation) in zip(axes, panel_specs):
        if mode == "projection":
            _plot_projection_mode(ax, waveforms[mode])
        elif mode == "stage":
            _plot_stage_mode(ax, waveforms[mode])
        else:
            _plot_discrete_mode(ax, mode, waveforms[mode])
        ax.set_title(f"{letter}   {title}", loc="left", fontweight="bold", pad=5)
        ax.text(
            0.995,
            1.025,
            equation,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.2,
            color=COLORS["muted"],
        )

    fig.suptitle(
        "OPM acquisition timing programs",
        fontsize=13,
        fontweight="bold",
    )
    return fig


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = "#4B5563",
    fontsize: float = 7.5,
    weight: str = "normal",
) -> None:
    """Draw a rounded flow-chart box in axes coordinates."""
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.016",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.8,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    text: str | None = None,
    color: str = "#4B5563",
    connectionstyle: str = "arc3",
) -> None:
    """Draw a directed connector in axes coordinates."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 0.85,
            "color": color,
            "shrinkA": 1,
            "shrinkB": 1,
            "connectionstyle": connectionstyle,
        },
    )
    if text:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.045,
            text,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=COLORS["muted"],
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 1.0,
            },
            zorder=5,
        )


def make_mode_map_figure() -> plt.Figure:
    """Create a mode-to-program and event-lifecycle trace.

    Returns
    -------
    matplotlib.figure.Figure
        Acquisition-mode map and common DAQ lifecycle.
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.05))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.04,
        0.95,
        "Acquisition mode → DAQ behavior",
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    headers = ("Mode", "Acquisition sequence", "DAQ behavior")
    for x, header in zip((0.04, 0.355, 0.73), headers):
        ax.text(x, 0.82, header, fontsize=7, color=COLORS["muted"], fontweight="bold")

    rows = [
        (
            0.60,
            "Mirror Sweep",
            "Acquire planes and colors\nwhile the mirror steps",
            "Step imaging mirror",
            COLORS["light_purple"],
        ),
        (
            0.37,
            "Projection",
            "Acquire one projected image\nfor each color",
            "Sweep both mirrors",
            COLORS["light_orange"],
        ),
        (
            0.14,
            "Stage scan",
            "Stage moves continuously\nat constant speed",
            "Interleaved channels",
            COLORS["light_green"],
        ),
    ]
    for y, user_mode, event_plan, daq_mode, facecolor in rows:
        _box(ax, 0.04, y, 0.22, 0.13, user_mode, facecolor=facecolor, weight="bold")
        _box(ax, 0.345, y, 0.31, 0.13, event_plan, facecolor="#FFFFFF", fontsize=7)
        _box(ax, 0.735, y, 0.22, 0.13, daq_mode, facecolor=facecolor, weight="bold")
        _arrow(ax, (0.26, y + 0.065), (0.345, y + 0.065))
        _arrow(ax, (0.655, y + 0.065), (0.735, y + 0.065))
    return fig


def _routing_canvas(title: str) -> tuple[plt.Figure, plt.Axes]:
    """Create a consistently styled routing-diagram canvas.

    Returns
    -------
    tuple
        Matplotlib figure and axes used for one acquisition mode.
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=8)
    return fig, ax


def make_mirror_routing_figure() -> plt.Figure:
    """Create the Mirror Sweep camera-edge routing diagram.

    Returns
    -------
    matplotlib.figure.Figure
        Mirror Sweep DI, DO, and AO clock topology.
    """
    fig, ax = _routing_canvas("Mirror Sweep routing")
    _box(
        ax,
        0.04,
        0.43,
        0.21,
        0.20,
        "Camera\nEXPOSURE OUT\nto PFI0",
        facecolor=COLORS["light_blue"],
        weight="bold",
    )
    _box(
        ax,
        0.36,
        0.43,
        0.22,
        0.20,
        "Digital in\ncamera timing\nport PFI0",
        facecolor="#F7F8FA",
        fontsize=6.6,
    )
    _box(
        ax,
        0.68,
        0.65,
        0.28,
        0.17,
        "Digital out\nlaser control\ntiming from PFI2",
        facecolor=COLORS["light_green"],
        fontsize=6.6,
    )
    _box(
        ax,
        0.68,
        0.27,
        0.28,
        0.20,
        "Analog out\nmirror position\ntiming from PFI2\nimaging mirror steps",
        facecolor=COLORS["light_purple"],
        fontsize=6.4,
    )
    _arrow(ax, (0.25, 0.53), (0.36, 0.53))
    _arrow(ax, (0.58, 0.56), (0.68, 0.735))
    _arrow(ax, (0.58, 0.49), (0.68, 0.37))
    return fig


def make_projection_routing_figure() -> plt.Figure:
    """Create the Projection dual-clock routing diagram.

    Returns
    -------
    matplotlib.figure.Figure
        Projection DO edge clock and retriggerable AO topology.
    """
    fig, ax = _routing_canvas("Projection routing")
    _box(
        ax,
        0.04,
        0.43,
        0.21,
        0.20,
        "Camera\nEXPOSURE OUT\nto PFI0",
        facecolor=COLORS["light_blue"],
        weight="bold",
    )
    _box(
        ax,
        0.36,
        0.65,
        0.22,
        0.18,
        "Digital in\ncamera timing\nport PFI0",
        facecolor="#F7F8FA",
        fontsize=6.6,
    )
    _box(
        ax,
        0.70,
        0.65,
        0.25,
        0.18,
        "Digital out\nlaser control\ntiming from PFI2",
        facecolor=COLORS["light_green"],
        fontsize=6.6,
    )
    _box(
        ax,
        0.60,
        0.22,
        0.36,
        0.22,
        "Analog out\nmirror sweep\nstarted by camera exposure",
        facecolor=COLORS["light_orange"],
        fontsize=6.4,
    )
    _arrow(ax, (0.25, 0.57), (0.36, 0.70))
    _arrow(ax, (0.58, 0.74), (0.70, 0.74))
    _arrow(
        ax,
        (0.25, 0.49),
        (0.60, 0.33),
        connectionstyle="arc3,rad=0.12",
    )
    return fig


def make_stage_routing_figure() -> plt.Figure:
    """Create the Stage scan external-trigger routing diagram.

    Returns
    -------
    matplotlib.figure.Figure
        ASI/PLC, camera, and NI-DAQ stage-scan topology.
    """
    fig, ax = _routing_canvas("Stage scan routing")
    _box(
        ax,
        0.04,
        0.63,
        0.21,
        0.18,
        "Stage controller\nconstant-speed motion\ncamera start signal",
        facecolor=COLORS["light_green"],
        fontsize=6.8,
    )
    _box(
        ax,
        0.36,
        0.63,
        0.22,
        0.18,
        "Camera\nshort exposures\nduring stage motion",
        facecolor=COLORS["light_blue"],
        fontsize=6.8,
    )
    _box(
        ax,
        0.70,
        0.63,
        0.25,
        0.18,
        "Camera EXPOSURE OUT\nto PFI0",
        facecolor=COLORS["light_blue"],
        fontsize=6.8,
    )
    _box(
        ax,
        0.36,
        0.24,
        0.22,
        0.18,
        "Digital in\ncamera timing\nport PFI0",
        facecolor="#F7F8FA",
        fontsize=6.6,
    )
    _box(
        ax,
        0.70,
        0.24,
        0.25,
        0.18,
        "Digital out\nlaser control\ntiming from PFI2",
        facecolor=COLORS["light_green"],
        fontsize=6.6,
    )
    _box(
        ax,
        0.04,
        0.24,
        0.21,
        0.18,
        "Analog out\nmirrors held steady",
        facecolor="#F7F8FA",
    )
    _arrow(ax, (0.25, 0.72), (0.36, 0.72))
    _arrow(ax, (0.58, 0.72), (0.70, 0.72))
    _arrow(
        ax,
        (0.825, 0.63),
        (0.47, 0.42),
        connectionstyle="arc3,rad=-0.10",
    )
    _arrow(ax, (0.58, 0.33), (0.70, 0.33))
    return fig


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save editable vector and high-resolution raster variants."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    common_metadata = {"Creator": "opm_v2/docs/daq_programs/generate_figures.py"}
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", metadata=common_metadata)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", metadata=common_metadata)
    fig.savefig(
        OUTPUT_DIR / f"{stem}.png",
        dpi=600,
        metadata={"Software": common_metadata["Creator"]},
    )
    plt.close(fig)


def main() -> None:
    """Regenerate every DAQ program figure."""
    configure_matplotlib()
    waveforms = build_waveforms()
    save_figure(make_waveform_figure(waveforms), "opm_daq_waveforms")
    save_figure(make_mode_map_figure(), "opm_mode_map")
    save_figure(make_mirror_routing_figure(), "opm_daq_routing_mirror_sweep")
    save_figure(make_projection_routing_figure(), "opm_daq_routing_projection")
    save_figure(make_stage_routing_figure(), "opm_daq_routing_stage_scan")
    print(f"Wrote SVG, PDF, and 600 dpi PNG figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
