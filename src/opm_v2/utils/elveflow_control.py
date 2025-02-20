#!/usr/bin/python

from opm_v2.hardware.ElveFlow import OB1Controller
import time

def run_fluidic_program(verbose: bool = False):

    """
    
    :param r_idx: int
        fluidics round to execute
    :param df_program: dataframe
        dataframe containing entire fluidics program
    :param mvp_controller: HamiltonMVP
        handle to initialized chain of Hamilton MVP valves
    :param pump_controller: APump
        handle to initialized pump

    :return True: boolean
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