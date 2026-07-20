#!/usr/bin/python
"""Provide low-level ASI Tiger stage and PLC hardware commands."""

# ----------------------------------------------------------------------------------------
# Import
# ----------------------------------------------------------------------------------------
from pymmcore_plus import RemoteMMCore
import time


def check_if_busy(mmcore_stage):
    """Wait until the ASI Tiger controller is no longer busy.

    Parameters
    ----------
    mmcore_stage : RemoteMMCore
        Active Micro-Manager core controlling the ASI hardware.
    """
    # turn on 'transmit repeated commands' for Tiger
    mmcore_stage.setProperty("TigerCommHub", "OnlySendSerialCommandOnChange", "No")

    # check to make sure Tiger is not busy
    ready = "B"
    while ready != "N":
        command = "STATUS"
        mmcore_stage.setProperty("TigerCommHub", "SerialCommand", command)
        ready = mmcore_stage.getProperty("TigerCommHub", "SerialResponse")
        time.sleep(0.010)

    # turn off 'transmit repeated commands' for Tiger
    mmcore_stage.setProperty("TigerCommHub", "OnlySendSerialCommandOnChange", "Yes")


def set_joystick_mode(mmcore_stage, x_stage_name, z_stage_name, joystick_mode):
    """Enable or disable ASI Tiger joystick input.

    Parameters
    ----------
    mmcore_stage : RemoteMMCore
        Active Micro-Manager core controlling the ASI hardware.
    x_stage_name : str
        XY-stage device name.
    z_stage_name : str
        Z-stage device name.
    joystick_mode : bool
        Requested joystick-enabled state.
    """
    if joystick_mode:
        mmcore_stage.setProperty(x_stage_name, "JoystickEnabled", "Yes")
        mmcore_stage.setProperty(z_stage_name, "JoystickInput", "22 - right wheel")
    else:
        mmcore_stage.setProperty(x_stage_name, "JoystickEnabled", "No")
        mmcore_stage.setProperty(z_stage_name, "JoystickInput", "0 - none")


def set_axis_speed(mmcore_stage, axis, axis_speed):
    """Change ASI Tiger X- or Y-axis movement speed.

    Parameters
    ----------
    mmcore_stage : RemoteMMCore
        Active Micro-Manager core controlling the ASI hardware.
    axis : str
        Axis name, ``"X"`` or ``"Y"``.
    axis_speed : float
        Movement speed in millimeters per second.
    """
    if axis == "X":
        command = "SPEED X=" + str(axis_speed)
        mmcore_stage.setProperty("TigerCommHub", "SerialCommand", command)
    elif axis == "Y":
        command = "SPEED Y=" + str(axis_speed)
        mmcore_stage.setProperty("TigerCommHub", "SerialCommand", command)


def set_xy_position(mmcore_stage, stage_x, stage_y):
    """Set the ASI Tiger XY-stage position.

    Parameters
    ----------
    mmcore_stage : RemoteMMCore
        Active Micro-Manager core controlling the ASI hardware.
    stage_x : float
        X coordinate in micrometers.
    stage_y : float
        Y coordinate in micrometers.
    """
    mmcore_stage.setXYPosition(stage_x, stage_y)
    mmcore_stage.waitForDevice(mmcore_stage.getXYStageDevice())


def set_z_position(mmcore_stage, stage_z):
    """Set the ASI Tiger Z-stage position.

    Parameters
    ----------
    mmcore_stage : RemoteMMCore
        Active Micro-Manager core controlling the ASI hardware.
    stage_z : float
        Z coordinate in micrometers.
    """
    mmcore_stage.setZPosition(stage_z)
    mmcore_stage.waitForDevice(mmcore_stage.getFocusDevice())


def set_1d_stage_scan(mmcore_stage):
    """Configure a constant-speed X-axis stage scan.

    Parameters
    ----------
    mmcore_stage : RemoteMMCore
        Active Micro-Manager core controlling the ASI hardware.
    """
    command = "1SCAN X? Y=0 Z=9 F=0"
    mmcore_stage.setProperty("TigerCommHub", "SerialCommand", command)


def set_1d_stage_scan_area(mmcore_stage, scan_axis_start_mm, scan_axis_end_mm):
    """Configure X-axis stage-scan limits.

    Parameters
    ----------
    mmcore_stage : RemoteMMCore
        Active Micro-Manager core controlling the ASI hardware.
    scan_axis_start_mm : float
        Scan start coordinate in millimeters.
    scan_axis_end_mm : float
        Scan end coordinate in millimeters.
    """
    scan_axis_start_mm = scan_axis_start_mm
    scan_axis_end_mm = scan_axis_end_mm
    command = (
        "1SCANR X=" + str(scan_axis_start_mm) + " Y=" + str(scan_axis_end_mm) + " R=10"
    )
    mmcore_stage.setProperty("TigerCommHub", "SerialCommand", command)


def setup_start_trigger_output(mmcore_stage):
    """Configure the ASI PLC stage-sync output.

    Parameters
    ----------
    mmcore_stage : RemoteMMCore
        Active Micro-Manager core controlling the ASI hardware.
    """
    plcName = "PLogic:E:36"
    propPosition = "PointerPosition"
    propCellConfig = "EditCellConfig"
    addrOutputBNC1 = 33  # BNC1 on the PLC front panel
    addrStageSync = 46  # TTL5 on Tiger backplane = stage sync signal
    # connect stage sync signal to BNC output
    mmcore_stage.setProperty(plcName, propPosition, addrOutputBNC1)
    mmcore_stage.setProperty(plcName, propCellConfig, addrStageSync)


def start_1d_stage_scan(mmcore_stage):
    """Start the configured constant-speed stage scan.

    Call this after the camera acquisition sequence is ready.

    Parameters
    ----------
    mmcore_stage : RemoteMMCore
        Active Micro-Manager core controlling the ASI hardware.
    """
    command = "1SCAN"
    mmcore_stage.setProperty("TigerCommHub", "SerialCommand", command)


def get_xyz_position(mmcore_stage):
    """Read the ASI Tiger XYZ-stage position.

    Parameters
    ----------
    mmcore_stage : RemoteMMCore
        Active Micro-Manager core controlling the ASI hardware.

    Returns
    -------
    tuple[float, float, float]
        X, Y, and Z positions in micrometers.
    """
    xy_pos = mmcore_stage.getXYPosition()
    stage_x_um = xy_pos[0]
    stage_y_um = xy_pos[1]
    stage_z_um = mmcore_stage.getPosition()

    return stage_x_um, stage_y_um, stage_z_um
