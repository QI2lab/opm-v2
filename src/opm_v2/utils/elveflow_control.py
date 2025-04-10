#!/usr/bin/python

from opm_v2.hardware.ElveFlow import OB1Controller
import time

def run_fluidic_program(verbose: bool = False):
    """Send a trigger pulse to OB1 Controller and wait for a return pulse.
    
    Requires the fluidics sequence to be running with a IF step before sequences to be run, 
    and TRIG step after, or the acquisition will be stuck waiting here.

    Parameters
    ----------
    verbose : bool, optional
        Show print statements, by default False

    Returns
    -------
    bool
        Acknowledgment the sequence was completed
    """
    opmOB1_local = OB1Controller.instance()
    opmOB1_local.init_board()
        
    # wait for user to verify ESI sequence is running
    # input("press enter after starting ESI sequence")
    
    opmOB1_local.trigger_OB1()
    
    if verbose:
        print(" Triggering OB1")
        
    time.sleep(1)
    
    if verbose:
        print(" Waiting for OB1 trigger")
        
    opmOB1_local.wait_for_OB1()
    
    if verbose:
        print(" OB1 trigger recieved, ESI sequence complete")
        
    opmOB1_local.close_board()
    
    return True