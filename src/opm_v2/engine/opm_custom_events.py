
from useq import MDAEvent, CustomAction
from typing import Optional, List, Dict
from pathlib import Path   

# Event for running AO grid generation
grid_event = MDAEvent(
    exposure = None,
    action=CustomAction(
        name='AO-grid',
        data = {
            'AO' : 
                {
                'stage_positions': None,
                'num_scan_positions':None,
                'num_tile_positions':None,
                'ao_dict': {
                    'daq_mode': None,
                    'channel_states': None,
                    'channel_powers' : None,
                    'exposure_ms': None,
                    'modal_delta': None,
                    'modal_alpha':None,                        
                    'iterations': None,
                    'metric': None,
                    'image_mirror_range_um' : None,
                },
                'apply_ao_map': bool(False),
                'pos_idx':int(0),
                'output_path':None
            },
            'Camera' : 
                {
                'exposure_ms': None,
                'camera_crop' : [
                    None,
                    None,
                    None,
                    None
                ]
            }
        }
    )
)


def create_timelapse_event(interval: int, time_steps: int, timepoint: int):
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

def create_fluidics_event(total_rounds: int, current_round:int):
    """_summary_

    Parameters
    ----------
    total_rounds : int
        _description_
    current_round : int
        _description_
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

def create_ao_optimize_event(config: Dict,
                             output_dir_path: Optional[Path] = None
):
    
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
                    'readout_time': float(ao_config['readout_time']),
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

def create_ao_grid_event():
    pass

def create_ao_mirror_update_event(
    mirror_coeffs: Optional[List],
    mirror_positions: Optional[List]
):
    if not mirror_coeffs and not mirror_positions:
        return None 
    else:
        ao_mirror_update = MDAEvent(
            action=CustomAction(
                name='AO-mirrorUpdate',
                data = {
                    'AOmirror' : {
                        'coefficients' : mirror_coeffs,
                        'voltages' : mirror_positions,
                    }
                }
            )
        )
        return ao_mirror_update
    
def create_o2o3_autofocus_event(
    exposure_ms: int,
    camera_center: List[int],
    camera_crop: List[int]
):
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

def create_daq_move_event(image_mirror_v: float):
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
    mode: str = None,
    channel_states: List[bool] = [False, False, False, False, False],
    channel_powers: List[bool] = [0, 0, 0, 0, 0],
    channel_exposures_ms: List[int] = [0, 0, 0, 0, 0],
    camera_center: List[int] = [0, 0],
    camera_crop: List[int] = [0, 0],
    interleaved: bool = False,
    laser_blanking: bool = True,
    image_mirror_range_um: Optional[float] = 0,
):
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

def create_stage_event(stage_position:Dict):
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

def create_asi_scan_setup_event(start_mm, end_mm, speed_mm_s):
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