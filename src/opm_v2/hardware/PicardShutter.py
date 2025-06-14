#!/usr/bin/python
# ----------------------------------------------------------------------------------------
# The basic I/O class for Picard USB Shutter
# ----------------------------------------------------------------------------------------
# Doug Shepherd
# 11/2022
# douglas.shepherd@asu.edu
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# Import
# ----------------------------------------------------------------------------------------
import time
import clr
clr.AddReference('PiUsbNet')
import PiUsbNet
import gc

_instance_shutter = None

# ----------------------------------------------------------------------------------------
# PicardShutter Class Definition
# ----------------------------------------------------------------------------------------
class PicardShutter():

    @classmethod
    def instance(cls) -> 'PicardShutter':
        """Return the global singleton instance of `PicardShutter`.

        """
        global _instance_shutter
        if _instance_shutter is None:
            _instance_shutter = cls()
        return _instance_shutter
    
    def __init__(self,shutter_id,verbose=False):

        # Set the first instance of this class as the global singleton
        global _instance_shutter
        if _instance_shutter is None:
            _instance_shutter = self
            
        

        # Define attributes
        self.shutter_id = shutter_id
        self.verbose = verbose  
        
        try:
            self.shutter: PiUsbNet.Shutter = PiUsbNet.Shutter()
            self.shutter.StateChanged += self._shutter_state_changed
            self.shutter.Open(self.shutter_id)
            if not self.shutter.IsConnected:
                if self.verbose: print('Shutter not found')
        except PiUsbNet.UsbDeviceException as exc:
            if self.verbose: print(f'PiUsbNet exception: {exc}')

    # Event handler function. Called by PiUsbNet.dll when the position changes.
    # This function runs in a worker thread.
    def _shutter_state_changed(self, sender: PiUsbNet.Shutter, args: PiUsbNet.ShutterStateChangedEventArgs):
        if (self.verbose): print(f'Shutter state: {args.State}')

    def printShutterState(self):
        try:
            print(self.shutter.State)
        except PiUsbNet.UsbDeviceException as exc:
            if self.verbose: print(f'PiUsbNet exception: {exc}')

    # open shutter
    def openShutter(self):
        try:
            new_state = PiUsbNet.ShutterState.Open
            self.shutter.State = new_state

            # Wait until new state is signaled. Timeout after 10 sec
            start_time = time.time()
            while self.shutter.State != new_state and (time.time()-start_time < 10.0):
                time.sleep(1.0)

            if self.shutter.State != new_state:
                if self.verbose: print('Shutter change state timeout')
            else:
                if self.verbose: print(f'Shutter at new state: {self.shutter.State}')
        except PiUsbNet.UsbDeviceException as exc:
            if self.verbose: print(f'PiUsbNet exception: {exc}')
    
    # close shutter
    def closeShutter(self):
        try:
            new_state = PiUsbNet.ShutterState.Closed
            self.shutter.State = new_state

            # Wait until new state is signaled. Timeout after 10 sec
            start_time = time.time()
            while self.shutter.State != new_state and (time.time()-start_time < 10.0):
                time.sleep(1.0)

            if self.shutter.State != new_state:
                if self.verbose: print('Shutter change state timeout')
            else:
                if self.verbose: print(f'Shutter at new state: {self.shutter.State}')
        except PiUsbNet.UsbDeviceException as exc:
            if self.verbose: print(f'PiUsbNet exception: {exc}')

    # make sure shutter is closed at shutdown
    def shutDown(self):
        self.closeShutter()
        self.shutter = None
        time.sleep(.1)
        gc.collect()
