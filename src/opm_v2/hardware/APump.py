#!/usr/bin/python
"""Control a Gibson peristaltic pump with an in-memory simulation backend."""
# ----------------------------------------------------------------------------------------
# The basic I/O class for a Gibson peristaltic pump
# ----------------------------------------------------------------------------------------
# Modified by qi2lab (Douglas Shepherd)
# 2025/02
# douglas.shepherd@asu.edu
#
# Original code:
# George Emanuel with modifications by Jeff Moffitt
# 11/16/15
# jeffmoffitt@gmail.com
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# Import
# ----------------------------------------------------------------------------------------
import serial

acknowledge = "\x06"
start = "\x0a"
stop = "\x0d"

_instance_pump = None


# ----------------------------------------------------------------------------------------
# GlisonMP3 Class Definition
# ----------------------------------------------------------------------------------------
class APump:
    """Control a peristaltic pump through its serial command protocol.

    Parameters
    ----------
    parameters : dict or None
        Pump port, identifier, direction, verbosity, and simulation settings.
    """

    @classmethod
    def instance(cls) -> "APump":
        """Return the process-wide pump instance.

        Returns
        -------
        APump
            Existing instance, or a new simulated instance when uninitialized.
        """
        global _instance_pump
        if _instance_pump is None:
            _instance_pump = cls()
        return _instance_pump

    def __init__(self, parameters: dict | None = None) -> None:
        """Initialize pump state and its selected backend.

        Parameters
        ----------
        parameters : dict or None
            Pump port, identifier, direction, verbosity, and simulation settings.
        """
        parameters = parameters or {}

        # Set the first instance of this class as the global singleton
        global _instance_pump
        if _instance_pump is None:
            _instance_pump = self

        # Define attributes
        self.com_port = parameters.get("pump_com_port", "COM3")
        self.pump_ID = parameters.get("pump_ID", 30)
        self.verbose = parameters.get("verbose", True)
        self.simulate = parameters.get("simulate_pump", True)
        self.serial_verbose = parameters.get("serial_verbose", False)
        self.flip_flow_direction = parameters.get("flip_flow_direction", False)

        # Define initial pump status
        self.flow_status = "Stopped"
        self.speed = 0.0
        self.direction = "Forward"
        self.remote_enabled = False
        self.command_log: list[str] = []

        # The simulated backend deliberately never opens a COM port.
        self.serial = None
        if not self.simulate:
            self.serial = serial.Serial(
                port=self.com_port,
                baudrate=19200,
                parity=serial.PARITY_EVEN,
                bytesize=serial.EIGHTBITS,
                stopbits=serial.STOPBITS_TWO,
                timeout=0.1,
            )

        self.disconnect()
        self.enableRemoteControl(1)
        self.startFlow(self.speed, self.direction)
        self.identification = self.getIdentification()

    def getIdentification(self):
        """Read the pump identification string.

        Returns
        -------
        str
            Pump identification response.
        """
        return self.sendImmediate(self.pump_ID, "%")

    def enableRemoteControl(self, remote):
        """Enable or disable remote pump control.

        Parameters
        ----------
        remote : bool
            Requested remote-control state.
        """
        if remote:
            self.sendBuffered(self.pump_ID, "SR")
        else:
            self.sendBuffered(self.pump_ID, "SK")
        self.remote_enabled = bool(remote)

    def readDisplay(self):
        """Read the pump display.

        Returns
        -------
        str
            Pump display response.
        """
        return self.sendImmediate(self.pump_ID, "R")

    def getStatus(self):
        """Decode the current pump status.

        Returns
        -------
        tuple
            Flow status, speed, direction, control mode, auto-start state,
            and error state.
        """
        message = self.readDisplay()

        if self.flip_flow_direction:
            direction = {" ": "Not Running", "-": "Forward", "+": "Reverse"}.get(
                message[0], "Unknown"
            )
        else:
            direction = {" ": "Not Running", "+": "Forward", "-": "Reverse"}.get(
                message[0], "Unknown"
            )

        status = "Stopped" if direction == "Not Running" else "Flowing"

        control = {"K": "Keypad", "R": "Remote"}.get(message[-1], "Unknown")

        auto_start = "Disabled"

        speed = float(message[1 : len(message) - 1])

        return (status, speed, direction, control, auto_start, "No Error")

    def close(self):
        """Disable remote control and close the serial connection."""
        self.enableRemoteControl(0)
        if self.serial is not None:
            self.serial.close()

    def setFlowDirection(self, forward):
        """Set the pump flow direction.

        Parameters
        ----------
        forward : bool
            ``True`` for forward flow and ``False`` for reverse flow.
        """
        if self.flip_flow_direction:
            if forward:
                self.sendBuffered(self.pump_ID, "K<")
            else:
                self.sendBuffered(self.pump_ID, "K>")
        else:
            if forward:
                self.sendBuffered(self.pump_ID, "K>")
            else:
                self.sendBuffered(self.pump_ID, "K<")
        self.direction = "Forward" if forward else "Reverse"

    def setSpeed(self, rotation_speed):
        """Set pump rotation speed.

        Parameters
        ----------
        rotation_speed : float
            Rotation speed between 0 and 48.
        """
        if rotation_speed >= 0 and rotation_speed <= 48:
            rotation_int = int(rotation_speed * 100)
            self.sendBuffered(self.pump_ID, "R" + ("%04d" % rotation_int))
            self.speed = float(rotation_speed)
            self.flow_status = "Flowing" if self.speed > 0 else "Stopped"

    def startFlow(self, speed, direction="Forward"):
        """Start flow at a requested speed and direction.

        Parameters
        ----------
        speed : float
            Rotation speed between 0 and 48.
        direction : str
            ``"Forward"`` or ``"Reverse"``.
        """
        self.setSpeed(speed)
        self.setFlowDirection(direction == "Forward")

    def stopFlow(self):
        """Stop pump flow.

        Returns
        -------
        bool
            ``True`` after the stop command is issued.
        """
        self.setSpeed(0.0)
        return True

    def sendImmediate(self, unitNumber, command):
        """Send an immediate pump command.

        Parameters
        ----------
        unitNumber : int
            Pump unit identifier.
        command : str
            Immediate command text.

        Returns
        -------
        str
            Pump response.
        """
        if self.simulate:
            self.command_log.append(f"IMMEDIATE:{unitNumber}:{command}")
            if command == "%":
                return "SIMULATED APUMP"
            if command == "R":
                sign = (
                    " "
                    if self.speed == 0
                    else ("+" if self.direction == "Forward" else "-")
                )
                control = "R" if self.remote_enabled else "K"
                return f"{sign}{self.speed:.2f}{control}"
            return ""
        self.selectUnit(unitNumber)
        self.sendString(command[0])
        newCharacter = self.getResponse()
        response = ""
        while not (ord(newCharacter) & 0x80):
            response += newCharacter.decode()
            self.sendString(acknowledge)
            newCharacter = self.getResponse()

        response += chr(ord(newCharacter) & ~0x80)
        self.disconnect()

        return response

    def sendBuffered(self, unitNumber, command):
        """Send a buffered pump command.

        Parameters
        ----------
        unitNumber : int
            Pump unit identifier.
        command : str
            Buffered command text.
        """
        if self.simulate:
            self.command_log.append(f"BUFFERED:{unitNumber}:{command}")
            return
        self.selectUnit(unitNumber)
        self.sendAndAcknowledge(start + command + stop)
        self.disconnect()

    def disconnect(self):
        """End the current pump command session."""
        if self.simulate:
            self.command_log.append("DISCONNECT")
            return
        self.sendAndAcknowledge("\xff")

    def selectUnit(self, unitNumber):
        """Select a pump unit.

        Parameters
        ----------
        unitNumber : int
            Pump unit identifier.

        Returns
        -------
        bool
            Whether the pump acknowledged the selection.
        """
        devSelect = chr(0x80 | unitNumber)
        self.sendString(devSelect)

        return self.getResponse() == devSelect

    def sendAndAcknowledge(self, string):
        """Send command characters and read each acknowledgement.

        Parameters
        ----------
        string : str
            Command text.
        """
        for i in range(0, len(string)):
            self.sendString(string[i])
            self.getResponse()

    def sendString(self, string):
        """Write text to the serial connection.

        Parameters
        ----------
        string : str
            Text to write.
        """
        self.serial.write(string.encode())

    def getResponse(self):
        """Read a response byte from the serial connection.

        Returns
        -------
        bytes
            Response byte.
        """
        return self.serial.read()
        # return self.serial.read().decode()
