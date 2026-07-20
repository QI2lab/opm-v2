"""Picard USB shutter control with an in-memory simulation mode."""

from __future__ import annotations

import gc
import time
from typing import Any

try:
    import clr

    clr.AddReference("PiUsbNet")
    import PiUsbNet  # type: ignore[import-not-found]
except Exception:
    PiUsbNet = None


_instance_shutter = None


class PicardShutter:
    """Control a Picard shutter while retaining the process singleton.

    Parameters
    ----------
    shutter_id : int or None
        Physical shutter identifier.
    verbose : bool
        Whether to print driver status.
    simulate : bool
        Whether to use the in-memory backend.
    """

    @classmethod
    def instance(cls) -> PicardShutter:
        """Return the initialized process-wide shutter instance.

        Returns
        -------
        PicardShutter
            Existing shutter instance.

        Raises
        ------
        RuntimeError
            If the shutter has not been initialized.
        """
        global _instance_shutter
        if _instance_shutter is None:
            raise RuntimeError("PicardShutter must be initialized before instance()")
        return _instance_shutter

    @classmethod
    def reset_instance(cls) -> None:
        """Release the process singleton reference for controlled teardown."""
        global _instance_shutter
        _instance_shutter = None

    def __init__(
        self,
        shutter_id: int | None = None,
        verbose: bool = False,
        simulate: bool = False,
    ) -> None:
        """Initialize a physical or simulated shutter.

        Parameters
        ----------
        shutter_id : int or None
            Physical shutter identifier.
        verbose : bool
            Whether to print driver status.
        simulate : bool
            Whether to use the in-memory backend.

        Raises
        ------
        RuntimeError
            If the physical driver cannot be loaded.
        ValueError
            If physical mode is selected without a shutter identifier.
        """
        global _instance_shutter
        if _instance_shutter is not None:
            raise RuntimeError("PicardShutter is already initialized; use instance()")
        _instance_shutter = self

        self.shutter_id = shutter_id
        self.verbose = verbose
        self.simulate = simulate
        self.shutter: Any | None = None
        self.state = "Closed"
        self.is_connected = False

        if self.simulate:
            self.is_connected = True
            return
        if PiUsbNet is None:
            raise RuntimeError(
                "PiUsbNet could not be loaded. Install the Picard driver or "
                "construct PicardShutter with simulate=True."
            )
        if shutter_id is None:
            raise ValueError("shutter_id is required for a physical Picard shutter")

        try:
            self.shutter = PiUsbNet.Shutter()
            self.shutter.StateChanged += self._shutter_state_changed
            self.shutter.Open(self.shutter_id)
            self.is_connected = bool(self.shutter.IsConnected)
            if not self.is_connected and self.verbose:
                print("Shutter not found")
        except PiUsbNet.UsbDeviceException as exc:
            if self.verbose:
                print(f"PiUsbNet exception: {exc}")

    def _shutter_state_changed(self, _sender: Any, args: Any) -> None:
        """Store and optionally print a hardware state-change event.

        Parameters
        ----------
        _sender : Any
            Unused event sender supplied by PiUsbNet.
        args : Any
            PiUsbNet event arguments containing the shutter state.
        """
        self.state = str(args.State)
        if self.verbose:
            print(f"Shutter state: {args.State}")

    def printShutterState(self) -> None:
        """Print the current shutter state."""
        print(self.state if self.simulate else self.shutter.State)

    def openShutter(self) -> None:
        """Open the shutter."""
        self._set_state("Open")

    def closeShutter(self) -> None:
        """Close the shutter."""
        self._set_state("Closed")

    def _set_state(self, state: str) -> None:
        """Apply a simulated or physical shutter state.

        Parameters
        ----------
        state : str
            PiUsbNet state name to apply.

        Raises
        ------
        RuntimeError
            If a physical shutter is not connected.
        """
        if self.simulate:
            self.state = state
            return
        if self.shutter is None:
            raise RuntimeError("Picard shutter is not connected")

        new_state = getattr(PiUsbNet.ShutterState, state)
        try:
            self.shutter.State = new_state
            start_time = time.time()
            while self.shutter.State != new_state and time.time() - start_time < 10.0:
                time.sleep(1.0)
            self.state = state
            if self.shutter.State != new_state and self.verbose:
                print("Shutter change state timeout")
        except PiUsbNet.UsbDeviceException as exc:
            if self.verbose:
                print(f"PiUsbNet exception: {exc}")

    def shutDown(self) -> None:
        """Close and release the shutter connection."""
        self.closeShutter()
        self.is_connected = False
        self.shutter = None
        if not self.simulate:
            time.sleep(0.1)
        gc.collect()
