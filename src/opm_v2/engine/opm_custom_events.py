"""
Methods to create OPM acquisition custom events

2025/09/05 SJS: initialization
"""
from useq import MDAEvent, CustomAction
from typing import Optional, List, Dict
from pathlib import Path   

def create_timelapse_event(interval: int, time_steps: int, timepoint: int) -> MDAEvent:
    """Create an event that pauses the acquisition

    Parameters
    ----------
    interval : int
        time interval for pause in seconds
    time_steps : int
        number of time points in acquisition
    timepoint : int
        the current time point

    Returns
    -------
    MDAEvent
        Custom event which pauses the running acquisition
    """
    timelapse_event = MDAEvent(
        action=CustomAction(
            name='Timelapse',
            data = {
                'plan' : {                    
                    'interval' : interval,
                    'timepoint' : timepoint,
                    'time_steps': time_steps
                }
            }
        )
    )
    return timelapse_event

def create_fluidics_event(total_rounds: int, current_round:int) -> MDAEvent:
    """Create an event that runs the fluidics program

    Parameters
    ----------
    total_rounds : int
        number of fluidics rounds 
    current_round : int
        current fluidics round to run

    Returns
    -------
    MDAEvent
        Custom event that runs the current fluidics round
    """
    FP_event = MDAEvent(
        # exposure = exposure_ms,
        action=CustomAction(
            name='Fluidics',
            data = {
                'Fluidics' : {
                    'total_rounds' : int(total_rounds),
                    'current_round' : int(current_round)
                }
                }
        )
    )
    return FP_event

def create_ao_optimize_event(
    config: Dict,
    output_dir_path: Optional[Path] = None
) -> MDAEvent:
    """_summary_

    Parameters
    ----------
    config : Dict
        OPM configuration from disk
    output_dir_path : Optional[Path], optional
        Path to save optimization results, by default None

    Returns
    -------
    MDAEvent
        Custom event that runs the sensorless A.O.
    """
    # Break up config dict to simplify use
    acq_config = config['acq_config']
    ao_config = config['acq_config']['AO']
    num_channels = len(config['OPM']['channel_ids'])
    
    # set the camera FOV based on the daq mode
    daq_mode = str(ao_config['daq_mode'])
    camera_crop_x = int(acq_config['camera_roi']['crop_x'])
    camera_center_y = int(acq_config['camera_roi']['center_y'])
    camera_center_x = int(acq_config['camera_roi']['center_x'])    
    if '2d' in daq_mode:
        camera_crop_y = int(acq_config['camera_roi']['crop_y'])
    elif 'projection' in daq_mode:
        camera_crop_y = int(
            acq_config['AO']['image_mirror_range_um']/config['OPM']['pixel_size_um']
            )
    
    # Create channel selection lists
    channel_states = [False] * num_channels
    channel_powers = [0.] * num_channels
    active_channel_id = ao_config['active_channel_id']
    for chan_idx, chan_str in enumerate(config['OPM']['channel_ids']):
        if active_channel_id==chan_str:
            channel_states[chan_idx] = True
            channel_powers[chan_idx] = ao_config['active_channel_power']
            
    # check to make sure there exist a laser power > 0
    if sum(channel_powers)==0:
        print('All AO laser powers are set to 0!')
        return None
        
    # Event for running AO optimization
    ao_optimize_event = MDAEvent(
        action=CustomAction(
            name='AO-optimize',
            data = {
                'AO' : {
                    'channel_states': channel_states,
                    'channel_powers' : channel_powers,
                    'mirror_state': str(ao_config['mirror_state']),
                    'daq_mode': str(ao_config['daq_mode']),
                    'exposure_ms': float(ao_config['exposure_ms']),
                    'modal_delta': float(ao_config['mode_delta']),
                    'metric_precision':int(ao_config['metric_precision']),
                    'modal_alpha':float(ao_config['mode_alpha']),                        
                    'iterations': int(ao_config['num_iterations']),
                    'metric': str(ao_config['metric']),
                    'image_mirror_range_um' : ao_config['image_mirror_range_um'],
                    'lightsheet_mode': str(ao_config['lightsheet_mode']),
                    'readout_ms': float(ao_config['readout_ms']),
                    'apply_existing': bool(False),
                    'pos_idx': int(0),
                    'output_path':output_dir_path
                },
                'Camera' : {
                    'exposure_ms': int(ao_config['exposure_ms']),
                    'camera_crop' : [
                        int(camera_center_x - camera_crop_x//2),
                        int(camera_center_y - camera_crop_y//2),
                        int(camera_crop_x),
                        int(camera_crop_y)
                    ]
                }
            }
        )
    )
    
    return ao_optimize_event

def create_ao_grid_event(
    config: Dict,
    output_dir_path: Optional[Path] = None
) -> MDAEvent:
    """Custom MDA event to run AO grid function

    Parameters
    ----------
    config : Dict
        OPM configuration from disk
    output_dir_path : Optional[Path], optional
        Path to save optimization results, by default None

    Returns
    -------
    MDAEvent
        custom event to run ao grid
    """
    # Break up config dict to simplify use
    acq_config = config['acq_config']
    ao_config = config['acq_config']['AO']
    num_channels = len(config['OPM']['channel_ids'])
    
    # set the camera FOV based on the daq mode
    daq_mode = str(ao_config['daq_mode'])
    camera_crop_x = int(acq_config['camera_roi']['crop_x'])
    camera_center_y = int(acq_config['camera_roi']['center_y'])
    camera_center_x = int(acq_config['camera_roi']['center_x'])    
    if '2d' in daq_mode:
        camera_crop_y = int(acq_config['camera_roi']['crop_y'])
    elif 'projection' in daq_mode:
        camera_crop_y = int(
            acq_config['AO']['image_mirror_range_um']/config['OPM']['pixel_size_um']
            )
    
    # Create channel selection lists
    channel_states = [False] * num_channels
    channel_powers = [0.] * num_channels
    active_channel_id = ao_config['active_channel_id']
    for chan_idx, chan_str in enumerate(config['OPM']['channel_ids']):
        if active_channel_id==chan_str:
            channel_states[chan_idx] = True
            channel_powers[chan_idx] = ao_config['active_channel_power']
            
    # check to make sure there exist a laser power > 0
    if sum(channel_powers)==0:
        print('All AO laser powers are set to 0!')
        return None
    
    ao_grid_event = MDAEvent(
        action=CustomAction(
            name='AO-grid',
            data = {
                'AO' : 
                    {
                    'stage_positions': None,
                    'num_scan_positions':ao_config['num_scan_positions'],
                    'num_tile_positions':ao_config['num_tile_positions'],
                    'output_path': output_dir_path,
                    'apply_ao_map': bool(False),
                    'pos_idx':int(0),
                    'ao_dict': {
                        'mirror_state': str(ao_config['mirror_state']),
                        'daq_mode': None,
                        'channel_states': channel_states,
                        'channel_powers' : channel_powers,
                        'exposure_ms': float(ao_config['exposure_ms']),
                        'modal_delta': float(ao_config['mode_delta']),
                        'metric_precision':int(ao_config['metric_precision']),
                        'modal_alpha':float(ao_config['mode_alpha']),                        
                        'iterations': int(ao_config['num_iterations']),
                        'metric': str(ao_config['metric']),
                        'image_mirror_range_um' : ao_config['image_mirror_range_um'],
                        'lightsheet_mode': str(ao_config['lightsheet_mode']),
                        'readout_ms': float(ao_config['readout_ms']),
                    }
                },
                'Camera' : {
                    'exposure_ms': int(ao_config['exposure_ms']),
                    'camera_crop' : [
                        int(camera_center_x - camera_crop_x//2),
                        int(camera_center_y - camera_crop_y//2),
                        int(camera_crop_x),
                        int(camera_crop_y)
                    ]
                }
            }
        )
    )
    return ao_grid_event

def create_ao_mirror_update_event(
    mirror_coeffs: Optional[List] = None,
    mirror_positions: Optional[List] = None
) -> MDAEvent:
    """Create an event to set the AO mirror state

    Parameters
    ----------
    mirror_coeffs : Optional[List], optional
        Mirror modal coefficients, by default None
    mirror_positions : Optional[List], optional
        Mirror voltage values, by default None

    Returns
    -------
    MDAEvent
        custom event to set mirror state
    """
    if mirror_coeffs is None and mirror_positions is None:
        return None 
    else:
        ao_mirror_update = MDAEvent(
            action=CustomAction(
                name='AO-mirrorUpdate',
                data = {
                    'AOmirror' : {
                        'coefficients' : mirror_coeffs.tolist() if mirror_coeffs is not None else None,
                        'voltages' : mirror_positions.tolist() if mirror_positions is not None else None,
                    }
                }
            )
        )
        return ao_mirror_update
    
def create_o2o3_autofocus_event(
    exposure_ms: int,
    camera_center: List[int],
    camera_crop: List[int]
) -> MDAEvent:
    """Create a custom MDA event to run the o2-o3 autofocus

    Parameters
    ----------
    exposure_ms : int
        camera exposure in ms
    camera_center : List[int]
        camera center, [x, y]
    camera_crop : List[int]
        camera crop, [x, y]

    Returns
    -------
    MDAEvent
        Custom event that runs the o2-o3 autofocus routine
    """
    af_event = MDAEvent(
        action=CustomAction(
            name='O2O3-autofocus',
            data = {
                'Camera' : {                    
                    'exposure_ms' : exposure_ms,
                    'camera_crop' : [
                        int(camera_center[0] - camera_crop[0]//2),
                        int(camera_center[1] - camera_crop[1]//2),
                        int(camera_crop[0]),
                        int(camera_crop[1]),
                    ]
                }
            }
        )
    )
    return af_event

def create_daq_move_event(image_mirror_v: float) -> MDAEvent:
    """Create a custom event to modify the image mirror nuetral position

    Parameters
    ----------
    image_mirror_v : float
        mirror voltage to apply as nuetral position

    Returns
    -------
    MDAEvent
        Custom event that modifies the image mirror neutral position 
    """
    daq_move_event = MDAEvent(
        action=CustomAction(
            name='Mirror-Move',
            data = {
                'DAQ' : {
                    'image_mirror_v': image_mirror_v
                }
            }
        )
    )
    return daq_move_event
    
def create_daq_event(
    mode: str = '2d',
    channel_states: List[bool] = [False, False, False, False, False],
    channel_powers: List[bool] = [0, 0, 0, 0, 0],
    channel_exposures_ms: List[int] = [0, 0, 0, 0, 0],
    camera_center: List[int] = [0, 0],
    camera_crop: List[int] = [0, 0],
    interleaved: bool = False,
    laser_blanking: bool = True,
    image_mirror_range_um: Optional[float] = 0,
    image_mirror_step_um: Optional[float] = 0.4
) -> MDAEvent:
    """Creates a daq event that updates the daq state to run in a given mode

    Parameters
    ----------
    mode : str, optional
        daq mode to use, [2d, projection, mirror, stage], by default 2d
    channel_states : List[bool], optional
        channel states for all sources, by default [False, False, False, False, False]
    channel_powers : List[bool], optional
        laser powers for each channel, by default [0, 0, 0, 0, 0]
    channel_exposures_ms : List[int], optional
        camera exposures for each channel, by default [0, 0, 0, 0, 0]
    camera_center : List[int], optional
        camera center [x,y], by default [0, 0]
    camera_crop : List[int], optional
        camera crop [x,y], by default [0, 0]
    interleaved : bool, optional
        by default False
    laser_blanking : bool, optional
        by default True
    image_mirror_range_um : Optional[float], optional
        by default 0
    image_mirror_step_um : Optional[float], optional
        by default 0.4

    Returns
    -------
    MDAEvent
        Custom event that programs the daq
    """
    # create DAQ hardware setup event
    DAQ_event = MDAEvent(
        action=CustomAction(
            name='DAQ',
            data = {
                'DAQ' : {
                    'mode' : mode,
                    'channel_states' : channel_states,
                    'channel_powers' : channel_powers,
                    'interleaved' : interleaved,
                    'blanking' : laser_blanking, 
                    'image_mirror_range_um': image_mirror_range_um,
                    'image_mirror_step_um': image_mirror_step_um
                },
                'Camera' : {
                    'exposure_channels' : channel_exposures_ms,
                    'camera_crop' : [
                        int(camera_center[0] - camera_crop[0]//2),
                        int(camera_center[1] - camera_crop[1]//2),
                        int(camera_crop[0]),
                        int(camera_crop[1]),
                    ]
                }
            }
        )
    )
    return DAQ_event

def create_stage_event(stage_position:Dict) -> MDAEvent:
    """Create an event that moves the stage to given position

    Parameters
    ----------
    stage_position : Dict
        Dict containing the 'x', 'y' and 'z' stage position

    Returns
    -------
    MDAEvent
        Custom event that moves the xyz stage
    """
    stage_event = MDAEvent(
        action=CustomAction(
            name= 'Stage-Move',
            data = {
                'Stage' : {
                    'x_pos' : stage_position['x'],
                    'y_pos' : stage_position['y'],
                    'z_pos' : stage_position['z'],
                }
            }
        )
    )
    return stage_event

def create_asi_scan_setup_event(start_mm: float, end_mm: float, speed_mm_s:float) -> MDAEvent:
    """Create a custom event that sets the ASI controller up for stage scan
    NOTE: positions are in mm and rounded to 2
    
    Parameters
    ----------
    start_mm : float
        scan start position in mm
    end_mm : float
        scan end position in mm
    speed_mm_s : _type_
        stage scan speed in mm/s

    Returns
    -------
    MDAEvent
        Custom event that sets the ASI controller up for a stage scan along X-axis
    """
    asi_setup_event = MDAEvent(
        action=CustomAction(
            name='ASI-setupscan',
            data = {
                'ASI' : {
                    'mode' : 'scan',
                    'scan_axis_start_mm' : start_mm,
                    'scan_axis_end_mm' : end_mm,
                    'scan_axis_speed_mm_s' : speed_mm_s
                }
            }
        )
    )

    return asi_setup_event