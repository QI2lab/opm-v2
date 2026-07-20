"""Run the wavefront-sensor closed-loop setup.

Required modules :
-os
-matplotlib
-PySide2
-sys
-time.
"""

import ctypes
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

WAVEKIT_PYTHON_ROOT = Path(r"C:\Users\qi2lab\Documents\github\wavekit_python")
WAVEKIT_PACKAGE_DIR = WAVEKIT_PYTHON_ROOT / "wavekit_py"
WAVEKIT_DLL_DIR = (
    WAVEKIT_PYTHON_ROOT / "dlls" / ("x64" if sys.maxsize > 2**32 else "Win32")
)
WAVEKIT_DLL_PATH = WAVEKIT_DLL_DIR / (
    "imop_wavekit_4_2_c_vc100_x64.dll"
    if sys.maxsize > 2**32
    else "imop_wavekit_4_2_c_vc100_Win32.dll"
)

HASO_CONFIG_FILE_PATH = Path(
    r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\wfc_configuration_files\WFS_HASO4_VIS_7635.dat"
)
WFC_CONFIG_FILE_PATH = Path(
    r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\wfc_configuration_files\WaveFrontCorrector_mirao52-e_0329.dat"
)
INTERACTION_MATRIX_FILE_PATH = Path(
    r"E:\Alignment\20260708\wfc_files\20260708_straight_interaction_matrix.aoc"
)
CLOSED_LOOP_UI_PATH = Path(
    r"C:\Users\qi2lab\Documents\github\wavekit_python\Examples\DEMO\Closed loop\closed_loop.ui"
)

# Keep these global so Windows keeps the DLL directories active.
DLL_DIRECTORY_HANDLES = []


def setup_wavekit_paths() -> None:
    """Validate and register the local WaveKit Python and DLL paths.

    Raises
    ------
    FileNotFoundError
        If a required WaveKit directory or DLL is missing.
    """
    if not WAVEKIT_PYTHON_ROOT.exists():
        raise FileNotFoundError(
            f"WaveKit Python root does not exist: {WAVEKIT_PYTHON_ROOT}"
        )
    if not WAVEKIT_PACKAGE_DIR.exists():
        raise FileNotFoundError(
            f"WaveKit package directory does not exist: {WAVEKIT_PACKAGE_DIR}"
        )
    if not WAVEKIT_DLL_DIR.exists():
        raise FileNotFoundError(
            f"WaveKit DLL directory does not exist: {WAVEKIT_DLL_DIR}"
        )
    if not WAVEKIT_DLL_PATH.exists():
        raise FileNotFoundError(f"WaveKit DLL does not exist: {WAVEKIT_DLL_PATH}")

    if str(WAVEKIT_PYTHON_ROOT) not in sys.path:
        sys.path.append(str(WAVEKIT_PYTHON_ROOT))
    if str(WAVEKIT_PACKAGE_DIR) not in sys.path:
        sys.path.append(str(WAVEKIT_PACKAGE_DIR))
    DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(WAVEKIT_DLL_DIR)))
    os.environ["PATH"] = str(WAVEKIT_DLL_DIR) + os.pathsep + os.environ["PATH"]


setup_wavekit_paths()

# from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtCore import QIODevice, Qt

exp_ms = 8000
include_curvature = False
include_tilt = False
output_prefix = "20260708_straight"
if include_curvature:
    output_prefix += "_curvature"
else:
    output_prefix += "_no_curvature"
if include_tilt:
    output_prefix += "_tilt"
else:
    output_prefix += "_no_tilt"

output_file_path = Path(r"E:\Alignment\20260708") / Path(
    output_prefix + "_closed_loop_output.wcs"
)

import io_thirdparty_load_library as wavekit_library
import wavekit_py as wkpy


def load_wavekit_dll():
    """Load the WaveKit DLL while its containing directory is active.

    Returns
    -------
    ctypes.CDLL
        Loaded WaveKit dynamic library.

    Raises
    ------
    Exception
        If Windows cannot load the configured DLL.
    """
    current_dir = Path.cwd()
    try:
        os.chdir(WAVEKIT_DLL_DIR)
        return ctypes.cdll.LoadLibrary(str(WAVEKIT_DLL_PATH))
    except OSError as exc:
        raise Exception(
            "IO_Error",
            f"---CAN NOT GET DLLS--- {WAVEKIT_DLL_PATH}: {exc}",
        )
    finally:
        os.chdir(current_dir)


wavekit_library.load_dll = load_wavekit_dll


def print_wavekit_diagnostics() -> None:
    """Print the active WaveKit module, DLL, and working-directory paths."""
    print(f"wavekit_py loaded from: {getattr(wkpy, '__file__', 'unknown')}")
    print(f"WaveKit DLL directory: {WAVEKIT_DLL_DIR}")
    print(f"WaveKit DLL directory exists: {WAVEKIT_DLL_DIR.exists()}")
    print(f"WaveKit DLL path: {WAVEKIT_DLL_PATH}")
    print(f"WaveKit DLL exists: {WAVEKIT_DLL_PATH.exists()}")
    print(f"Current working directory: {Path.cwd()}")


class Interface(QtWidgets.QMainWindow):
    """Display and control the WaveKit closed-loop correction workflow."""

    def __init__(self, parent=None):
        """Initialize the closed-loop user interface and its hardware state.

        Parameters
        ----------
        parent : QWidget or None
            Optional Qt parent widget.
        """
        # Init mother class
        super(Interface, self).__init__(parent)

        # Load interface
        # loader = QtWidgets.QUiLoader()
        file = QtCore.QFile(str(CLOSED_LOOP_UI_PATH))
        file.open(QIODevice.OpenModeFlag.ReadOnly)
        self.window = uic.loadUi(file, parent)
        file.close()
        self.window.show()
        self.window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        # self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # Init variable
        self.backup_path = str(INTERACTION_MATRIX_FILE_PATH)
        self.haso_path = str(HASO_CONFIG_FILE_PATH)
        self.wfc_path = str(WFC_CONFIG_FILE_PATH)
        self.image = None
        self.init_ui()
        self.load_default_paths()
        self.init_connections()
        QtCore.QTimer.singleShot(0, self.auto_connect_hardware)

    def init_ui(self) -> None:
        """Initialize plots, canvases, status indicators, and controls."""
        fg_color = "white"

        # Init variable
        self.image_cb = None
        self.image_canvas = None
        self.command_canvas = None
        self.command_plot = None

        # Create figure containing image
        self.image_figure, (self.image_ax, self.image_cb_ax) = plt.subplots(
            1, 2, gridspec_kw={"width_ratios": [20, 1]}
        )
        self.image_figure.suptitle(r"$\Delta$ Wavefront", fontsize=16, color=fg_color)
        self.image_figure.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
        self.image_ax.axes.tick_params(color=fg_color, labelcolor=fg_color)
        self.image_cb_ax.yaxis.tick_right()
        self.image_cb_ax.xaxis.set_visible(False)
        self.image_cb_ax.axes.tick_params(color=fg_color, labelcolor=fg_color)

        # Create canvas containing image figure
        self.image_canvas = FigureCanvas(self.image_figure)
        # Add image canvas to ImageDisplay widget
        self.window.ImageDisplay.layout().addWidget(self.image_canvas)
        self.image_ax.set_visible(False)
        self.image_cb_ax.set_visible(False)

        # Create figure containing plot
        self.command_figure = plt.figure("Command")
        self.command_ax = self.command_figure.add_subplot(1, 1, 1)
        self.command_ax.axes.tick_params(color=fg_color, labelcolor=fg_color)
        self.command_figure.suptitle(
            r"$\Delta$ Positions of mirror actuators", fontsize=16, color=fg_color
        )
        self.command_figure.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
        self.command_ax.set_visible(False)

        self.command_max = 1
        self.command_min = -1

        # Create canvas containing plot figure
        self.command_canvas = FigureCanvas(self.command_figure)
        # Add plot canvas to PlotDisplay widget
        self.window.PlotDisplay.layout().addWidget(self.command_canvas)

        self.window.AcquisitionStatus.setEnabled(False)
        self.window.HasoStatus.setEnabled(False)
        self.window.WFCStatus.setEnabled(False)

        self.window.StartButton.setEnabled(False)

        self.window.WFSconnectButton.setEnabled(False)
        self.window.WFSdisconnectButton.setEnabled(False)
        self.window.WFCconnectButton.setEnabled(False)
        self.window.WFCdisconnectButton.setEnabled(False)

    def load_default_paths(self) -> None:
        """Populate path controls and report missing default files."""
        self.window.HasoFilePath.setText(self.haso_path)
        self.window.WFCFilePath.setText(self.wfc_path)
        self.window.BackupFilePath.setText(self.backup_path)

        self.window.WFSconnectButton.setEnabled(Path(self.haso_path).exists())
        self.window.WFCconnectButton.setEnabled(Path(self.wfc_path).exists())

        for label, path in (
            ("HASO config", self.haso_path),
            ("WFC config", self.wfc_path),
            ("interaction matrix", self.backup_path),
        ):
            if not Path(path).exists():
                print(f"WARNING: default {label} path does not exist: {path}")

    def auto_connect_hardware(self) -> None:
        """Connect available WFS and WFC hardware after validating paths."""
        if not Path(self.haso_path).exists():
            self.error(f"HASO config not found: {self.haso_path}")
            return
        if not Path(self.wfc_path).exists():
            self.error(f"WFC config not found: {self.wfc_path}")
            return
        if not Path(self.backup_path).exists():
            self.error(f"Interaction matrix not found: {self.backup_path}")
            return

        print("Auto-loading WFS, WFC, and interaction matrix paths.")
        if not self.window.HasoStatus.isEnabled():
            self.connect_haso()
        if not self.window.WFCStatus.isEnabled():
            self.connect_wfc()
        self.check_start_ready()

    def init_connections(self) -> None:
        """Connect Qt control signals to their callbacks."""
        self.window.getHasoFileButton.clicked.connect(self.get_haso_path)
        self.window.getWFCFileButton.clicked.connect(self.get_wfc_path)
        self.window.getBackupFileButton.clicked.connect(self.get_backup_path)

        # haso connection
        self.window.WFSconnectButton.clicked.connect(self.connect_haso)
        self.window.WFSdisconnectButton.clicked.connect(self.disconnect_haso)

        # corrector connection
        self.window.WFCconnectButton.clicked.connect(self.connect_wfc)
        self.window.WFCdisconnectButton.clicked.connect(self.disconnect_wfc)

        self.window.StartButton.clicked.connect(self.start_loop)

    def get_backup_path(self) -> None:
        """Select the interaction-matrix backup file."""
        research = QtWidgets.QFileDialog()
        research.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
        file = research.getOpenFileName(self, "Choose file", __file__)
        self.backup_path = str(file[0])
        self.window.BackupFilePath.setText(self.backup_path)
        self.check_start_ready()

    def get_haso_path(self) -> None:
        """Select the HASO wavefront-sensor configuration file."""
        research = QtWidgets.QFileDialog()
        research.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
        file = research.getOpenFileName(self, "Choose file", __file__)
        self.haso_path = str(file[0])
        self.window.HasoFilePath.setText(self.haso_path)
        self.window.WFSconnectButton.setEnabled(True)

    def get_wfc_path(self) -> None:
        """Select the wavefront-corrector configuration file."""
        research = QtWidgets.QFileDialog()
        research.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
        file = research.getOpenFileName(self, "Choose file", __file__)
        self.wfc_path = str(file[0])
        self.window.WFCFilePath.setText(self.wfc_path)
        self.window.WFCconnectButton.setEnabled(True)

    def error(self, message) -> None:
        """Briefly display an error in the window status bar.

        Parameters
        ----------
        message : str
            Error text to display.
        """
        self.statusBar().showMessage(message)
        time.sleep(1.5)
        self.statusBar().clearMessage()

    def connect_haso(self) -> None:
        """Connect and start the configured HASO wavefront-sensor camera."""
        try:
            print_wavekit_diagnostics()
            self.camera = wkpy.Camera(config_file_path=self.haso_path)
            self.camera.connect()
            self.camera.start(
                wkpy.E_CAMERA_ACQUISITION_MODE.NEW,
                wkpy.E_CAMERA_SYNCHRONIZATION_MODE.SYNCHRONOUS,
            )
            self.camera.set_parameter_value("exposure_duration_us", exp_ms)
            self.image = self.camera.get_raw_image()
            self.on_haso_connection(True)
        except Exception as e:
            print(str(e))
            self.error(str(e))

    def disconnect_haso(self) -> None:
        """Disconnect the HASO wavefront-sensor camera."""
        try:
            self.camera.disconnect()
            self.on_haso_connection(False)
        except Exception as e:
            self.error(str(e))

    def on_haso_connection(self, connected) -> None:
        """Update controls after the HASO connection state changes.

        Parameters
        ----------
        connected : bool
            Whether the HASO camera is connected.
        """
        self.window.WFSconnectButton.setEnabled(not connected)
        self.window.WFSdisconnectButton.setEnabled(connected)
        self.window.HasoStatus.setEnabled(connected)
        self.window.getHasoFileButton.setEnabled(not connected)
        self.check_start_ready()

    def connect_wfc(self) -> None:
        """Connect the configured wavefront corrector."""
        try:
            print_wavekit_diagnostics()
            self.wfc = wkpy.WavefrontCorrector(config_file_path=self.wfc_path)
            self.wfc.connect(True)
            self.on_wfc_connection(True)
        except Exception as e:
            print(str(e))
            self.error(str(e))

    def disconnect_wfc(self) -> None:
        """Disconnect the configured wavefront corrector."""
        try:
            self.wfc.disconnect()
            self.on_wfc_connection(False)
        except Exception as e:
            self.error(str(e))

    def on_wfc_connection(self, connected) -> None:
        """Update controls after the WFC connection state changes.

        Parameters
        ----------
        connected : bool
            Whether the wavefront corrector is connected.
        """
        self.window.WFCconnectButton.setEnabled(not connected)
        self.window.WFCdisconnectButton.setEnabled(connected)
        self.window.WFCStatus.setEnabled(connected)
        self.window.getWFCFileButton.setEnabled(not connected)
        self.check_start_ready()

    def check_start_ready(self) -> None:
        """Enable loop execution only when hardware and data are ready."""
        self.window.StartButton.setEnabled(
            self.window.HasoStatus.isEnabled()
            and self.window.WFCStatus.isEnabled()
            and self.backup_path is not None
        )

    def start_loop(self) -> None:
        """Mark acquisition active and execute the closed-loop correction."""
        self.on_loop_running(True)
        self.closed_loop()

    def on_loop_running(self, running) -> None:
        """Update controls for the closed-loop running state.

        Parameters
        ----------
        running : bool
            Whether closed-loop correction is active.
        """
        self.window.StartButton.setEnabled(not running)
        self.window.WFSdisconnectButton.setEnabled(not running)
        self.window.WFCdisconnectButton.setEnabled(not running)
        self.window.AcquisitionStatus.setEnabled(running)

    def closed_loop(self) -> None:
        """Run correction iterations and save the final WFC positions."""
        try:
            nb_iter = self.window.nbLoop.value()
            corr_data_manager = wkpy.CorrDataManager(
                haso_config_file_path=self.haso_path,
                interaction_matrix_file_path=self.backup_path,
            )

            """Get command dynamic range
            """
            prefs = corr_data_manager.get_actuator_prefs(0)
            self.command_min = prefs.min_value * 0.01
            self.command_max = prefs.max_value * 0.01

            """Compute reference slopes target tilt and focus
            """
            hasoengine = wkpy.HasoEngine(config_file_path=self.haso_path)
            """Get computed slopes
            """
            slopes = hasoengine.compute_slopes(self.image, False)[1]

            """Create ref_hasoslopeserence slopes
            """
            ref_hasoslopes = wkpy.HasoSlopes(hasoslopes=slopes)

            processor_list = wkpy.SlopesPostProcessorList()

            if include_tilt:
                if include_curvature:
                    processor_list.insert_filter(0, True, True, True, True, True, True)
                else:
                    processor_list.insert_filter(0, True, True, False, True, True, True)
            else:
                if include_curvature:
                    processor_list.insert_filter(
                        0, False, False, True, True, True, True
                    )
                else:
                    processor_list.insert_filter(
                        0, False, False, False, True, True, True
                    )
            processor_list.insert_filter(0, False, False, False, False, False, False)

            hasodata = wkpy.HasoData(hasoslopes=ref_hasoslopes)
            hasodata.apply_slopes_post_processor_list(processor_list)
            ref_hasoslopes = hasodata.get_hasoslopes()[0]  # Get only computed slopes

            processor_list.delete_processor(0)
            processor_list.insert_substractor(0, ref_hasoslopes, "")

            hasodata.set_hasoslopes(slopes)
            hasodata.apply_slopes_post_processor_list(processor_list)
            delta_hasoslopes = hasodata.get_hasoslopes()[0]

            """Set wavefrontcorrector pref_hasoslopeserences
            """
            self.wfc.set_temporization(100)

            """Compute command matrix
            """
            corr_data_manager.set_command_matrix_prefs(32, False)
            corr_data_manager.compute_command_matrix()

            """Loop with RMS printing
            """
            compute_phase_set = wkpy.ComputePhaseSet(
                type_phase=wkpy.E_COMPUTEPHASESET.ZONAL
            )
            loop_smoothing = wkpy.LoopSmoothing(level="MEDIUM")
            gain = 0.2

            self.image_ax.set_visible(True)
            self.image_cb_ax.set_visible(True)
            self.command_ax.set_visible(True)

            for x in range(nb_iter):
                self.image = self.camera.get_raw_image()

                slopes = hasoengine.compute_slopes(self.image, False)[1]
                hasodata.set_hasoslopes(slopes)
                hasodata.apply_slopes_post_processor_list(processor_list)
                delta_hasoslopes = hasodata.get_hasoslopes()[0]
                phase = wkpy.Compute.phase_zonal(compute_phase_set, hasodata)
                delta_commands, applied_gain = (
                    corr_data_manager.compute_closed_loop_iteration(
                        delta_hasoslopes, False, loop_smoothing, gain
                    )
                )
                self.wfc.move_to_relative_positions(delta_commands)

                rms, pv, max_, min_ = phase.get_statistics()

                self.display(phase.get_data()[0].copy(), delta_commands, rms, pv)

                """
                1sec wait between iterations
                You can improve calculations performance by removing this waiting instruction
                (but the display refresh of this demo is not optimized for the full performance)
                If you do so, you can improve the display refresh by either using threads or by flushing events
                """
                time.sleep(0.1)

            self.on_loop_running(False)

            self.wfc.save_current_positions_to_file(str(output_file_path))

        except Exception as e:
            print(str(e))
            self.error(str(e))
            self.on_loop_running(False)

    def display(self, data, command, rms, pv) -> None:
        """Display the current wavefront and actuator corrections.

        Parameters
        ----------
        data : numpy.ndarray
            Current wavefront phase image.
        command : Sequence[float]
            Relative actuator commands applied in this iteration.
        rms : float
            Root-mean-square wavefront error.
        pv : float
            Peak-to-valley wavefront error.
        """
        image_plot = self.image_ax.imshow(data, interpolation="none")
        self.image_ax.invert_yaxis()
        self.image_cb = self.image_figure.colorbar(image_plot, cax=self.image_cb_ax)
        self.image_canvas.draw()

        self.command_ax.clear()
        self.command_ax.bar(range(len(command)), command)
        ymin = self.command_min
        ymax = self.command_max
        for x in command:
            if x > ymax:
                ymax = x
            if x < ymin:
                ymin = x

        self.command_ax.set_ylim([ymin, ymax])
        self.command_canvas.draw()

        self.window.RMSValue.setText(str(rms))
        self.window.PVValue.setText(str(pv))

        self.repaint()

    def main(self) -> None:
        """Display the main window."""
        # self.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    interface = Interface()
    interface.main()
    app.exec()
