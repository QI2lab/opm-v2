from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
import traceback
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, cast

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication
from superqt.utils import WorkerBase

from pymmcore_gui import MicroManagerGUI,WidgetAction, __version__

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    ExcTuple = tuple[type[BaseException], BaseException, TracebackType | None]

APP_NAME = "Micro-Manager GUI"
APP_VERSION = __version__
ORG_NAME = "pymmcore-plus"
ORG_DOMAIN = "pymmcore-plus"
APP_ID = f"{ORG_DOMAIN}.{ORG_NAME}.{APP_NAME}.{APP_VERSION}"
RESOURCES = Path(__file__).parent / "resources"
ICON = RESOURCES / ("icon.ico" if sys.platform.startswith("win") else "logo.png")
IS_FROZEN = getattr(sys, "frozen", False)


class MMQApplication(QApplication):
    exceptionRaised = pyqtSignal(BaseException)

    def __init__(self, argv: list[str]) -> None:
        if sys.platform == "darwin" and not argv[0].endswith("mmgui"):
            # Make sure the app name in the Application menu is `mmgui`
            # which is taken from the basename of sys.argv[0]; we use
            # a copy so the original value is still available at sys.argv
            argv[0] = "napari"

        super().__init__(argv)
        self.setApplicationName("Micro-Manager GUI")
        self.setWindowIcon(QIcon(str(ICON)))

        self.setApplicationName(APP_NAME)
        self.setApplicationVersion(APP_VERSION)
        self.setOrganizationName(ORG_NAME)
        self.setOrganizationDomain(ORG_DOMAIN)
        if os.name == "nt" and not IS_FROZEN:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)  # type: ignore

        self.aboutToQuit.connect(WorkerBase.await_workers)


def parse_args(args: Sequence[str] = ()) -> argparse.Namespace:
    if not args:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Enter string")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Config file to load",
        nargs="?",
    )
    return parser.parse_args(args)


def main() -> None:
    """Run the Micro-Manager GUI."""
    args = parse_args()

    app = MMQApplication(sys.argv)
    _install_excepthook()

    win = MicroManagerGUI()
    mmc = win.mmcore
    mmc.loadSystemConfiguration(Path(r"C:/Users/dpshe/Documents/GitHub/OPMv2/OPM_demo.cfg"))

    win.setWindowIcon(QIcon(str(ICON)))

    win.showMaximized()
    win.show()

    splsh = "_PYI_SPLASH_IPC" in os.environ and importlib.util.find_spec("pyi_splash")
    if splsh:  # pragma: no cover
        import pyi_splash  # pyright: ignore [reportMissingModuleSource]

        pyi_splash.update_text("UI Loaded ...")
        pyi_splash.close()

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # ----------------Begin custom qi2lab code for running OPM control----------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    widget = win.get_widget(WidgetAction.MDA_WIDGET)
    
    def custom_execute_mda(output: Path | str | object | None) -> None:
        """Custom execute_mda method that modifies the sequence before running it.
        
        Parameters
        ----------
        output : Path | str | object | None
            The output path for the MDA sequence.
        """

        sequence = widget.value()

        # check OPM mode
        if "Projection" in mmc.getProperty("OPM-mode", "Label"):
            print("projection mode")
        elif "Stage" in mmc.getProperty("OPM-mode", "Label"):
            print("stage mode")
        elif "Standard" in mmc.getProperty("OPM-mode", "Label"):
            print("standard mode")

        # check AO mode
        if "System-correction" in mmc.getProperty("AO-mode", "Label"):
            print("system correction AO")
        elif "First-run" in mmc.getProperty("AO-mode", "Label"):
            print("first run AO")
        elif "Always-run" in mmc.getProperty("AO-mode", "Label"):
            print("always run AO")
        elif "Optimize" in mmc.getProperty("AO-mode", "Label"):
            print("optimize AO now")

        # check fluidics mode
        if "None" in mmc.getProperty("Fluidics-mode", "Label"):
            print("No fluidics")
        elif "Thin-16bit" in mmc.getProperty("Fluidics-mode", "Label"):
            print("Thin 16bit fluidics")
        elif "Thin-22bit" in mmc.getProperty("Fluidics-mode", "Label"):
            print("Thin 22bit fluidics")
        elif "Thick-16bit" in mmc.getProperty("Fluidics-mode", "Label"):
            print("Thick 16bit fluidics")
        elif "Thick-2bit" in mmc.getProperty("Fluidics-mode", "Label"):
            print("Thick 22bit fluidics")

        # check O2-O3 autofocus
        if "Initial-only" in mmc.getProperty("O2O3focus-mode", "Label"):
            print("Run initial O2-O3 autofocus")
        elif "After-every" in mmc.getProperty("O2O3focus-mode", "Label"):
            print("Run O2-O3 autofocus after every sequenced acquistion")
        elif "After-30min" in mmc.getProperty("O2O3focus-mode", "Label"):
            print("Run O2-O3 autofocus after 30 minutes")
        elif "None" in mmc.getProperty("O2O3focus-mode", "Label"):
            print("No O2-O3 autofocus")

        print("modifying sequence")
        # go nuts here, modify the sequence however you want
        # the only rule is that it has to still be an `Iterable[MDAEvent]` when you pass it in
        widget._mmc.run_mda(sequence, output=output)

    # modify the method on the instance
    widget.execute_mda = custom_execute_mda

    def setup_preview_mode_callback():
        """Callback to intercept preview mode and setup the NIDAQ accordingly."""

        # Get galvo range and active channel
        image_galvo_range = np.round(float(mmc.getProperty("ImageGalvoMirror", "Position")),2)
        active_channel = mmc.getProperty("LED", "Label")
        
        # Check OPM mode and set up NIDAQ accordingly
        if "Projection" in mmc.getProperty("OPM-mode", "Label"):
            # ... call our NIDAQ code to setup multiple analog waveforms and digital ouput
            print("projection mode")
        else:
            # .... call our NIDAQ code to setup digital ouput
            print("standard mode")

    mmc.events.continuousSequenceAcquisitionStarting.connect(setup_preview_mode_callback)

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # -----------------End custom qi2lab code for running OPM control-----------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    app.exec()


# ------------------- Custom excepthook -------------------


def _install_excepthook() -> None:
    """Install a custom excepthook that does not raise sys.exit().

    This is necessary to prevent the application from closing when an exception
    is raised.
    """
    if hasattr(sys, "_original_excepthook_"):
        return
    sys._original_excepthook_ = sys.excepthook  # type: ignore
    sys.excepthook = ndv_excepthook


def rich_print_exception(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
) -> None:
    import psygnal
    from rich.console import Console
    from rich.traceback import Traceback

    tb = Traceback.from_exception(
        exc_type,
        exc_value,
        exc_traceback,
        suppress=[psygnal],
        max_frames=100 if IS_FROZEN else 10,
        show_locals=True,
    )
    Console(stderr=True).print(tb)


def _print_exception(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
) -> None:
    try:
        rich_print_exception(exc_type, exc_value, exc_traceback)
    except ImportError:
        traceback.print_exception(exc_type, value=exc_value, tb=exc_traceback)


# This log list is used by the ExceptionLog widget
# Be aware that it's currently possible for that widget to clear this list.
# If an immutable record of exceptions is needed, additional logic will be required.
EXCEPTION_LOG: list[ExcTuple] = []


def ndv_excepthook(
    exc_type: type[BaseException], exc_value: BaseException, tb: TracebackType | None
) -> None:
    EXCEPTION_LOG.append((exc_type, exc_value, tb))
    _print_exception(exc_type, exc_value, tb)
    if sig := getattr(QApplication.instance(), "exceptionRaised", None):
        sig.emit(exc_value)
    if not tb:
        return

    # if we're running in a vscode debugger, let the debugger handle the exception
    if (
        (debugpy := sys.modules.get("debugpy"))
        and debugpy.is_client_connected()
        and ("pydevd" in sys.modules)
    ):  # pragma: no cover
        with suppress(Exception):
            import threading

            import pydevd  # pyright: ignore [reportMissingImports]

            if (py_db := pydevd.get_global_debugger()) is None:
                return

            py_db = cast("pydevd.PyDB", py_db)
            thread = threading.current_thread()
            additional_info = py_db.set_additional_thread_info(thread)
            additional_info.is_tracing += 1

            try:
                arg = (exc_type, exc_value, tb)
                py_db.stop_on_unhandled_exception(py_db, thread, additional_info, arg)
            finally:
                additional_info.is_tracing -= 1
    # otherwise, if MMGUI_DEBUG_EXCEPTIONS is set, drop into pdb
    elif os.getenv("MMGUI_DEBUG_EXCEPTIONS"):
        import pdb

        pdb.post_mortem(tb)

    # after handling the exception, exit if MMGUI_EXIT_ON_EXCEPTION is set
    if os.getenv("MMGUI_EXIT_ON_EXCEPTION"):
        print("\nMMGUI_EXIT_ON_EXCEPTION is set, exiting.")
        sys.exit(1)