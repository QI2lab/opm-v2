#!/usr/bin/python

import pyfirmata2
from time import perf_counter

_instance_ob1 = None

class OB1Controller():
    """_summary_
    """
    @classmethod
    def instance(cls) -> 'OB1Controller':
        """Return the global singleton instance of `OB1Controller`.

        """
        global _instance_ob1
        if _instance_ob1 is None:
            _instance_ob1 = cls()
        return _instance_ob1
        

    def __init__( self,
                  port: str = 'COM9',
                  to_OB1_pin: int = 7,
                  from_OB1_pin: int = 8):
        """Arduino controller for the Elveflow OB1 controller.
        Send and receive signals to the input/output ports

        Parameters
        ----------
        port : str, optional
            _description_, by default 'COM7'
        to_OB1_pin : int, optional
            _description_, by default 8
        from_OB1_pin : int, optional
            _description_, by default 6
        """
        
        global _instance_ob1
        if _instance_ob1 is None:
            _instance_ob1 = self
        
        self.port = port
        self.to_OB1_pin_location= f'd:{to_OB1_pin}:o'
        self.from_OB1_pin_location = f'd:{from_OB1_pin}:i'
        self._from_OB1_pin_high = False

        # SJS: Should we not init board in the _init_? This way we can init and close in the fluidics loop?
        # self.init_board()
        
    def init_board(self):
        """
        Initialize Arduino connection 
        - open connection
        - configure pins
        """
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
        self.to_OB1_pin.write(False)
        self.board.exit()
        self.board = None
        self.from_OB1_pin = None
        self.to_OB1_pin = None
                
    def set_polling_rate(self,polling_rate_ms: int = 1000):
        """
        Set the sampling interval for the Arduino
        Note: should be set such that there is no possibility the output
              pulse from the ElveFlow controller is missed.
        """
        self.board.setSamplingInterval(polling_rate_ms)

    def _input_callback(self,
                        data: float = None,
                        verbose: bool = False):
        """
        Function to recognize when the input goes high.
        """
        if data == 1:
            self._from_OB1_pin_high = True
            if verbose:
                print('received trigger')
        else:
            self._from_OB1_pin_high = False

    def wait_for_OB1(self):
        """
        Function to pause program until a high pulse is received
        """
        while not self._from_OB1_pin_high:
            self.board.iterate()
        
        # reset the input to false
        self._from_OB1_pin_high = False
        

    def trigger_OB1(self, pulse_duration: float = 0.500):
        """
        Send high pulse to_OB1_pin.
        """
        timer_start = perf_counter()
        self.to_OB1_pin.write(True)
        while (perf_counter() - timer_start < pulse_duration):
            continue
        self.to_OB1_pin.write(False)

    
    