#!/usr/bin/python
"""Control the Arduino interface to an Elveflow OB1 controller."""

from time import perf_counter

import pyfirmata2

_instance_ob1 = None


class OB1Controller:
    """Control OB1 handshake signals through Arduino pins.

    Parameters
    ----------
    port : str
        Arduino serial port.
    to_OB1_pin : int
        Arduino output pin connected to the OB1.
    from_OB1_pin : int
        Arduino input pin receiving the OB1 acknowledgement.
    simulate : bool
        Whether to use the in-memory backend.
    """

    @classmethod
    def instance(cls) -> "OB1Controller":
        """Return the process-wide OB1 controller instance.

        Returns
        -------
        OB1Controller
            Existing initialized instance.

        Raises
        ------
        RuntimeError
            If no OB1 controller has been initialized by the application.
        """
        global _instance_ob1
        if _instance_ob1 is None:
            raise RuntimeError("OB1Controller must be initialized before instance()")
        return _instance_ob1

    @classmethod
    def reset_instance(cls) -> None:
        """Release the process singleton reference for controlled teardown."""
        global _instance_ob1
        _instance_ob1 = None

    def __init__(
        self,
        port: str = "COM9",
        to_OB1_pin: int = 7,
        from_OB1_pin: int = 8,
        simulate: bool = False,
    ):
        """Initialize the Arduino interface to the OB1 controller.

        Parameters
        ----------
        port : str
            Arduino serial port.
        to_OB1_pin : int
            Arduino output pin connected to the OB1.
        from_OB1_pin : int
            Arduino input pin receiving the OB1 acknowledgement.
        simulate : bool
            Whether to use the in-memory backend.

        Raises
        ------
        RuntimeError
            If another OB1 controller already owns the process singleton.
        """
        global _instance_ob1
        if _instance_ob1 is not None:
            raise RuntimeError("OB1Controller is already initialized; use instance()")
        _instance_ob1 = self

        self.port = port
        self.simulate = simulate
        self.to_OB1_pin_location = f"d:{to_OB1_pin}:o"
        self.from_OB1_pin_location = f"d:{from_OB1_pin}:i"
        self._from_OB1_pin_high = False
        self.board = None
        self.from_OB1_pin = None
        self.to_OB1_pin = None
        self.trigger_count = 0
        self.last_pulse_duration = None
        self.polling_rate_ms = 1000

        # SJS: Should we not init board in the _init_? This way we can init and close in the fluidics loop?
        # self.init_board()

    def init_board(self):
        """Initialize the Arduino connection and configure OB1 pins."""
        if self.simulate:
            self.board = True
            self._from_OB1_pin_high = False
            return

        self.board = pyfirmata2.Arduino(self.port)

        # start polling
        self.set_polling_rate()

        # Configure DO pin to ElveFlow controller
        self.to_OB1_pin = self.board.get_pin(self.to_OB1_pin_location)
        self.to_OB1_pin.write(False)

        # Configure DI pin received from ElveFlow controller
        self.from_OB1_pin = self.board.get_pin(self.from_OB1_pin_location)
        self.from_OB1_pin.register_callback(self._input_callback)
        self.from_OB1_pin.enable_reporting()

    def close_board(self):
        """Close the Arduino connection and reset pin state."""
        if self.simulate:
            self.board = None
            self.from_OB1_pin = None
            self.to_OB1_pin = None
            self._from_OB1_pin_high = False
            return
        self.to_OB1_pin.write(False)
        self.board.exit()
        self.board = None
        self.from_OB1_pin = None
        self.to_OB1_pin = None

    def set_polling_rate(self, polling_rate_ms: int = 1000):
        """Set the Arduino sampling interval.

        The interval should be short enough to observe every OB1 output pulse.

        Parameters
        ----------
        polling_rate_ms : int
            Sampling interval in milliseconds.
        """
        self.polling_rate_ms = polling_rate_ms
        if not self.simulate:
            self.board.setSamplingInterval(polling_rate_ms)

    def _input_callback(self, data: float = None, verbose: bool = False):
        """Record an OB1 input-state transition.

        Parameters
        ----------
        data : float or None
            Digital input value supplied by pyfirmata2.
        verbose : bool
            Whether to print received triggers.
        """
        if data == 1:
            self._from_OB1_pin_high = True
            if verbose:
                print("received trigger")
        else:
            self._from_OB1_pin_high = False

    def wait_for_OB1(self):
        """Wait until the OB1 acknowledgement input goes high.

        Raises
        ------
        RuntimeError
            If called in simulation before a trigger has been issued.
        """
        while not self._from_OB1_pin_high:
            if self.simulate:
                raise RuntimeError("Simulated OB1 has not been triggered")
            self.board.iterate()

        # reset the input to false
        self._from_OB1_pin_high = False

    def trigger_OB1(self, pulse_duration: float = 0.500):
        """Send a high pulse to the OB1 controller.

        Parameters
        ----------
        pulse_duration : float
            Pulse duration in seconds.
        """
        self.trigger_count += 1
        self.last_pulse_duration = float(pulse_duration)
        if self.simulate:
            self._from_OB1_pin_high = True
            return

        timer_start = perf_counter()
        self.to_OB1_pin.write(True)
        while perf_counter() - timer_start < pulse_duration:
            continue
        self.to_OB1_pin.write(False)
