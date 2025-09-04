
import numpy as np

from useq import MDAEvent, CustomAction, MDASequence
from types import MappingProxyType as mappingproxy
from pymmcore_plus import CMMCorePlus
from pathlib import Path
import json
from datetime import datetime
from tqdm import trange
from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ
from opm_v2.handlers.opm_mirror_handler import OPMMirrorHandler
from opm_v2.engine.opm_custom_actions import (
    O2O3_af_event,
    AO_optimize_event,
    AO_grid_event,
    FP_event,
    DAQ_event,
    ASI_setup_event,
    Timelapse_event
)
DEBUGGING = True
MAX_IMAGE_MIRROR_RANGE_UM = 250
MIN_PROJECTION_EXP_MS = 50

def stage_positions_from_grid(
    mda_grid_plan,
    mda_z_plan,
    opm_mode: str,
    camera_crop_x: int,
    camera_crop_y: int,
    scan_range_um: float,
    scan_axis_overlap: float,
    tile_axis_overlap: float,
    z_axis_overlap: float = None,
    coverslip_max_dz: float = None,
    coverslip_slope_x: float = 0,
    coverslip_slope_y: float = 0
    ):
    """_summary_

    Parameters
    ----------
    mda_grid_plan : _type_
        _description_
    mda_z_plan : _type_
        _description_
    opm_mode : str
        _description_
    camera_crop_x : int
        _description_
    camera_crop_y : int
        _description_
    scan_range_um : float
        _description_
    tile_axis_overlap : float
        _description_
    z_axis_overlap : float, optional
        _description_, by default None
    coverslip_max_dz : float, optional
        _description_, by default None
    coverslip_slope_x : float, optional
        _description_, by default 0
    coverslip_slope_y : float, optional
        _description_, by default 0

    Returns
    -------
    _type_
        _description_
    """
    stage_positions = []
    _mmc = CMMCorePlus.instance()
    
    if mda_z_plan is not None:
        max_z_pos = float(mda_z_plan['top'])
        min_z_pos = float(mda_z_plan['bottom'])
    else:
        min_z_pos = _mmc.getZPosition()
        max_z_pos = _mmc.getZPosition()
        
    # Force projection mode to a single z-position
    if 'projection' in opm_mode:
        n_z_positions = 1
        z_step_um = 0
    elif ('mirror' in opm_mode) or ('stage' in opm_mode):
        if min_z_pos==max_z_pos:
            n_z_positions = 1
            z_step_um = 0
        else:
            range_z_um = np.round(np.abs(max_z_pos - min_z_pos),2)
            z_step_max = (
                camera_crop_y
                * _mmc.getPixelSizeUm()
                * (1-z_axis_overlap)
                * np.sin((np.pi/180.)*float(30)) 
            )
            n_z_positions = int(np.ceil(range_z_um / z_step_max))
            z_step_um = np.round(range_z_um / n_z_positions,2)
            
    # grab grid plan extents
    min_y_pos = mda_grid_plan['bottom']
    max_y_pos = mda_grid_plan['top']
    min_x_pos = mda_grid_plan['left']
    max_x_pos = mda_grid_plan['right']
    cs_min_pos = min_z_pos

    # Correct directions for stage moves
    if min_z_pos > max_z_pos:
        z_step_um *= -1
    if min_x_pos > max_x_pos:
        min_x_pos, max_x_pos = max_x_pos, min_x_pos
        
    # Set grid axes ranges
    range_x_um = np.round(np.abs(max_x_pos - min_x_pos),2)
    range_y_um = np.round(np.abs(max_y_pos - min_y_pos),2)
    
    if range_y_um==0:
        y_step_um = 0
        n_y_positions = 1
    else:
        y_step_max = (
            camera_crop_x
            * _mmc.getPixelSizeUm()
            * (1-tile_axis_overlap)
        )
        n_y_positions = int(np.ceil(range_y_um / y_step_max))
        y_step_um = np.round(range_y_um / n_y_positions,2)
        
    if scan_range_um==0:
        print("Scan range == 0!")
        return []
    else:
        x_step_max = (scan_range_um * (1-scan_axis_overlap))
        n_x_positions = int(np.ceil(range_x_um / x_step_max))
        x_step_um = np.round(range_x_um / n_x_positions,2)

    if 'projection' in opm_mode or 'mirror' in opm_mode:
        if x_step_max > MAX_IMAGE_MIRROR_RANGE_UM:
            x_step_max =  MAX_IMAGE_MIRROR_RANGE_UM
        
    # account for coverslip slopes
    if coverslip_slope_x != 0:
        cs_x_max_pos = cs_min_pos + range_x_um * coverslip_slope_x
        cs_x_range_um = np.round(np.abs(cs_x_max_pos - cs_min_pos),2)
        dz_per_x_tile = np.round(cs_x_range_um/n_x_positions,2)
    else:
        dz_per_x_tile = 0

    if coverslip_slope_y != 0:
        cs_y_max_pos = cs_min_pos + range_y_um * coverslip_slope_y
        cs_y_range_um = np.round(np.abs(cs_y_max_pos - cs_min_pos),2)
        dz_per_y_tile = np.round(cs_y_range_um/n_y_positions,2)
    else:
        dz_per_y_tile = 0

    # populate positions list
    for kk in range(n_z_positions):
        for jj in range(n_y_positions):
            # For mirror and projection modes, generate a snake-pattern grid
            if 'mirror' in opm_mode or 'projection' in opm_mode:
                # move stage left to right
                if jj % 2 == 0:
                    x_iterator = range(n_x_positions)
                else:
                    x_iterator = reversed(range(n_x_positions))
            else:
                x_iterator = range(n_x_positions)
                
            for ii in x_iterator:
                    stage_positions.append(
                        {
                            'x': min_x_pos + ii * x_step_um,
                            'y': min_y_pos + jj * y_step_um,
                            'z': min_z_pos + kk * z_step_um + ii * dz_per_x_tile + jj * dz_per_y_tile
                        }  
                    )
    if DEBUGGING: 
        print(
            '\n\nXYZ Stage position settings:',
            f'\n  x start: {min_x_pos}',
            f'\n  x end: {max_x_pos}',
            f'\n  y start: {min_y_pos}',
            f'\n  y end: {max_y_pos}',
            f'\n  z position min:{min_z_pos}',
            f'\n  z position max:{max_z_pos}',
            f'\n  scan range (um): {scan_range_um}',
            f'\n  Coverslip slope (x/y): {coverslip_slope_x}/{coverslip_slope_y}',
            f'\n  Number x tiles: {n_x_positions}',
            f'\n  Number y tiles: {n_y_positions}',
            f'\n  Number z tiles: {n_z_positions}',
            f'\n  x tile length um: {x_step_um}',
            f'\n  y tile length um: {y_step_um}'
        )
    
    return stage_positions

def setup_timelapse(
    mmc: CMMCorePlus,
    config: dict,
    sequence: MDASequence,
    output: Path,
) -> list[MDAEvent]:
    
    OPMdaq_setup = OPMNIDAQ.instance()
    opm_events = [] 
    
    #--------------------------------------------------------------------#
    # Compile acquisition settings from configuration
    #--------------------------------------------------------------------#
    # Get the acquisition modes
    opm_mode = config['acq_config']['opm_mode']
    ao_mode = config['acq_config']['AO']['ao_mode']
    o2o3_mode = config['acq_config']['O2O3-autofocus']['o2o3_mode']
    
    # Get pixel size and deskew Y-scale factor
    pixel_size_um = np.round(float(mmc.getPixelSizeUm()),3) # unit: um
    
    # Get the camera crop values
    camera_crop_y = int(config['acq_config']['camera_roi']['crop_y'])
    camera_crop_x = int(config['acq_config']['camera_roi']['crop_x'])
    camera_center_y = int(config['acq_config']['camera_roi']['center_y'])
    camera_center_x = int(config['acq_config']['camera_roi']['center_x'])
    
    #----------------------------------------------------------------#
    # Get channel settings
    # laser_blanking = config['acq_config'][opm_mode+'_scan']['laser_blanking']
    channel_states = config['acq_config']['timelapse']['channel_states']
    channel_powers = config['acq_config']['timelapse']['channel_powers']
    channel_exposures_ms = config['acq_config']['timelapse']['channel_exposures_ms']
    channel_names = config['OPM']['channel_ids']
    
    n_active_channels = sum(channel_states)
    active_channel_names = [_name for _, _name in zip(channel_states, channel_names) if _]
    
    # Interleave only available if all channels have the same exposure.
    active_channel_exps = []
    for ii, ch_state in enumerate(channel_states):
        if ch_state:
            active_channel_exps.append(channel_exposures_ms[ii])
        else:
            # set not used channel powers to 0
            channel_powers[ii] = 0
        
    if len(set(active_channel_exps))==1:
        interleaved_acq = True
    else:
        interleaved_acq = False
            
    if sum(channel_powers)==0:
        print('All lasers set to 0!')
        return None, None
        
    #----------------------------------------------------------------#
    # try to get camera conversion factor information
    try:
        offset = mmc.getProperty(
            config['Camera']['camera_id'],
            'CONVERSION FACTOR OFFSET'
        )
        e_to_ADU = mmc.getProperty(
            config['Camera']['camera_id'],
            'CONVERSION FACTOR COEFF'
        )
    except Exception:
        offset = 0.
        e_to_ADU = 1.

    #----------------------------------------------------------------#
    # get the scan mirror positions
    scan_range_um = config['acq_config']['mirror_scan']['scan_range_um']
    scan_step_um = config['acq_config']['mirror_scan']['scan_step_size_um']
    laser_blanking = config['acq_config']['mirror_scan']['laser_blanking']
    
    # get the number of scan steps as expected by the daq
    if scan_range_um == 0.0:
        # setup daq for 2d scan
        scan_mode = '2d'
        OPMdaq_setup.set_acquisition_params(
            scan_type=scan_mode,
            channel_states=channel_states,
        )
        n_scan_steps = 1
        mirror_voltages = np.array([config['NIDAQ']['image_mirror_neutral_v']])
    else:
        scan_mode = 'mirror'
        OPMdaq_setup.set_acquisition_params(
            scan_type=scan_mode,
            channel_states=channel_states,
            image_mirror_range_um=scan_range_um,
            image_mirror_step_um=scan_step_um
        )
        OPMdaq_setup.generate_waveforms()
        mirror_voltages = np.unique(OPMdaq_setup._ao_waveform[:,0])
        n_scan_steps = mirror_voltages.shape[0]
    #--------------------------------------------------------------------#
    # Compile mda acquisition settings from active tabs
    #--------------------------------------------------------------------#
    
    # Split apart sequence dictionary
    sequence_dict = json.loads(sequence.model_dump_json())
    mda_positions_plan = sequence_dict['stage_positions']
    mda_time_plan = sequence_dict['time_plan']
    
    if (mda_positions_plan is None) or (mda_time_plan is None):
        print('Must select MDA positions AND time plan for mirror based timelapse')
        return None, None
    
    #----------------------------------------------------------------#
    # Create custom action data
    #----------------------------------------------------------------#
           
    #----------------------------------------------------------------#
    # Create DAQ event
    daq_action_data = {
        'DAQ' : {
            'mode' : '2d',
            'channel_states' : channel_states,
            'channel_powers' : channel_powers,
            'interleaved' : interleaved_acq,
            'active_channels': channel_states,
            'blanking' : laser_blanking, 
        },
        'Camera' : {
            'exposure_channels' : channel_exposures_ms,
            'camera_crop' : [
                int(camera_center_x - camera_crop_x//2),
                int(camera_center_y - camera_crop_y//2),
                int(camera_crop_x),
                int(camera_crop_y),
            ]
        }
    }

    # Create DAQ event to run before acquiring each 'image'
    daq_event = MDAEvent(**DAQ_event.model_dump())
    daq_event.action.data.update(daq_action_data)

    # create mirror move event to modify the neutral position in a 2d scan
    daq_move_event = MDAEvent(
        action=CustomAction(
            name='Mirror-Move',
            data = {
                'DAQ' : {
                    'image_mirror_v': float(0)
                }
            }
        )
    )
    
    #----------------------------------------------------------------#
    # Create the o2o3 AF event data
    if 'none' not in o2o3_mode:
        o2o3_action_data = {
            'Camera' : {                    
                'exposure_ms' : config['O2O3-autofocus']['exposure_ms'],
                'camera_crop' : [
                    int(camera_center_x - camera_crop_x//2),
                    int(camera_center_y - config['acq_config']['O2O3-autofocus']['roi_crop_y']//2),
                    int(camera_crop_x),
                    int(config['acq_config']['O2O3-autofocus']['roi_crop_y'])
                    ]
                }
            }
        
        o2o3_event = MDAEvent(**O2O3_af_event.model_dump())
        o2o3_event.action.data.update(o2o3_action_data)
        
    #----------------------------------------------------------------#
    # Create the AO event data    
    if 'none' not in ao_mode:
        AOmirror_setup = AOMirror.instance()
        if 'grid' in ao_mode:
            print(
                'AO Grid selected, but one stage position is used, running AO optimize once'
                )
            
        # Create a new directory in output.root for saving AO results
        ao_output_dir = output.parent / Path(f'{output.stem}_ao_results')
        ao_output_dir.mkdir(exist_ok=True)
        
        AO_daq_mode = str(config['acq_config']['AO']['daq_mode'])
        if '2d' in AO_daq_mode:
            AO_camera_crop_y = int(config['acq_config']['camera_roi']['crop_y'])
        elif 'projection' in AO_daq_mode:
            AO_camera_crop_y = int(
                config['acq_config']['AO']['image_mirror_range_um']/mmc.getPixelSizeUm()
                )
        AO_channel_states = [False] * len(channel_names) 
        AO_channel_powers = [0.] * len(channel_names)
        AO_active_channel_id = config['acq_config']['AO']['active_channel_id']
        AO_save_path = ao_output_dir
        
        # Set the active channel in the daq channel list
        for chan_idx, chan_str in enumerate(config['OPM']['channel_ids']):
            if AO_active_channel_id==chan_str:
                AO_channel_states[chan_idx] = True
                AO_channel_powers[chan_idx] = config['acq_config']['AO']['active_channel_power']
                
        # check to make sure there exist a laser power > 0
        if sum(AO_channel_powers)==0:
            print('All AO laser powers are set to 0!')
            return None, None
        
        # Define AO optimization action data   
        ao_action_data = {
            'AO' : {
                'channel_states': AO_channel_states,
                'channel_powers' : AO_channel_powers,
                'daq_mode': str(config['acq_config']['AO']['daq_mode']),
                'exposure_ms': float(config['acq_config']['AO']['exposure_ms']),
                'modal_delta': float(config['acq_config']['AO']['mode_delta']),
                'modal_alpha':float(config['acq_config']['AO']['mode_alpha']),                        
                'iterations': int(config['acq_config']['AO']['num_iterations']),
                'metric': str(config['acq_config']['AO']['metric']),
                'image_mirror_range_um' : config['acq_config']['AO']['image_mirror_range_um'],
                'blanking': bool(True),
                'apply_existing': bool(False),
                'pos_idx': int(0),
                'output_path':AO_save_path
            },
            'Camera' : {
                'exposure_ms': config['acq_config']['AO']['exposure_ms'],
                'camera_crop' : [
                    int(camera_center_x - camera_crop_x//2),
                    int(camera_center_y - AO_camera_crop_y//2),
                    int(camera_crop_x),
                    int(AO_camera_crop_y)
                ]
            }
        }
        ao_event = MDAEvent(**AO_optimize_event.model_dump())
        ao_event.action.data.update(ao_action_data)
        AOmirror_setup.output_path = AO_save_path
            
    #----------------------------------------------------------------#
    # Compile mda positions from active tabs and config
    #----------------------------------------------------------------#

    # Get time points
    n_time_steps = mda_time_plan['loops']
    # time_interval = mda_time_plan['interval']

    # ----------------------------------------------------------------#
    # Get xyz stage position
    stage_positions = []
    for stage_pos in mda_positions_plan:
        stage_positions.append(
            {
                'x': float(stage_pos['x']),
                'y': float(stage_pos['y']),
                'z': float(stage_pos['z'])
            }
        )
    n_stage_positions = len(stage_positions)
    if n_stage_positions > 1:
        print(
            f"Multiple stage positions selected, using the first:\n"
        )
    stage_position = stage_positions[0]
    #----------------------------------------------------------------#
    # Create MDA event structure
    #----------------------------------------------------------------#

    opm_events: list[MDAEvent] = []
        
    # move stage to position    
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
    opm_events.append(stage_event)
    
    # Check if run AF at start only
    if 'start' in o2o3_mode:
        opm_events.append(o2o3_event)
        
    # check if run AO at start only
    if 'start' in ao_mode:
        opm_events.append(ao_event)
    
    #----------------------------------------------------------------#
    # setup 1p x nC x nZ x nT mirror-based AO-OPM acquisition event structure
    
    if DEBUGGING: 
            print(
                'Acquisition shape values:'
                f'\n  timepoints: {n_time_steps}',
                f'\n  Stage positions: {n_stage_positions}',
                f'\n  Active channels: {n_active_channels}',
                f'\n  Num scan positions: {n_scan_steps}'
            )
    
    for scan_idx in trange(n_scan_steps, desc='Mirror scan positions:', leave=True):
        # Move the image mirror to position
        daq_event_move = MDAEvent(**daq_move_event.model_dump())
        daq_event_move.action.data['DAQ']['image_mirror_v'] = mirror_voltages[scan_idx]
        opm_events.append(daq_event_move)
        
        # check for optimization per position
        if 'position' in o2o3_mode:
            opm_events.append(o2o3_event)
                
        if 'position' in ao_mode:
            # Run AO optimization before acquiring current position
            current_ao_event = MDAEvent(**ao_event.model_dump())
            current_ao_event.action.data['AO']['output_path'] = ao_output_dir / Path(f'pos_{scan_idx}_ao_optimize')
            current_ao_event.action.data['AO']['apply_existing'] = False
            current_ao_event.action.data['AO']['pos_idx'] = int(scan_idx)
            opm_events.append(current_ao_event)
                            
        # Move the image mirror to position, and set to 2d scan
        opm_events.append(daq_event)
        
        # acquire sequenced timelapse images
        for time_idx in trange(n_time_steps, desc= 'Timepoints:', leave=False):  
            current_chan_idx = 0
            for chan_idx, chan_bool in enumerate(channel_states):
                if chan_bool:
                    # Create image event for current t / p / c 
                    image_event = MDAEvent(
                        index=mappingproxy(
                            {
                                't': time_idx, 
                                'p': 0, 
                                'c': current_chan_idx,
                                'z': scan_idx
                            }
                        ),
                        metadata = {
                            'DAQ' : {
                                'mode' : '2d',
                                'image_mirror_position' : float(mirror_voltages[scan_idx]), # TODO
                                'image_mirror_step_um': float(scan_step_um),
                                'active_channels' : channel_states,
                                'exposure_channels_ms': channel_exposures_ms,
                                'laser_powers' : channel_powers,
                                'interleaved' : True,
                                'blanking' : True,
                                'current_channel' : channel_names[chan_idx]
                            },
                            'Camera' : {
                                'exposure_ms' : float(channel_exposures_ms[chan_idx]),
                                'camera_center_x' : int(camera_center_x),
                                'camera_center_y' : int(camera_center_y),
                                'camera_crop_x' : int(camera_crop_x),
                                'camera_crop_y' : int(camera_crop_y),
                                'offset' : float(offset),
                                'e_to_ADU': float(e_to_ADU)
                            },
                            'OPM' : {
                                'angle_deg' : float(config['OPM']['angle_deg']),
                                'camera_Zstage_orientation' : str(config['OPM']['camera_Zstage_orientation']),
                                'camera_XYstage_orientation' : str(config['OPM']['camera_XYstage_orientation']),
                                'camera_mirror_orientation' : str(config['OPM']['camera_mirror_orientation'])
                            },
                            'Stage' : {
                                'x_pos' : float(stage_position['x']),
                                'y_pos' : float(stage_position['y']),
                                'z_pos' : float(stage_position['z']),
                            }
                        }
                    )
                    opm_events.append(image_event)
                    current_chan_idx += 1 

    # Check if path ends if .zarr. If so, use our OutputHandler
    if len(Path(output).suffixes) == 1 and Path(output).suffix ==  '.zarr':
        indice_sizes = {
            't' : int(np.maximum(1,n_time_steps)),
            'p' : int(1),
            'c' : int(np.maximum(1,n_active_channels)),
            'z' : int(np.maximum(1,n_scan_steps))
        }
        handler = OPMMirrorHandler(
            path=Path(output),
            indice_sizes=indice_sizes,
            delete_existing=True
        )
        print(f'\nUsing Qi2lab handler,\nindices: {indice_sizes}\n')
    else:
        print('Using default handler')
        handler = Path(output)
            
    return opm_events, handler

def setup_optimizenow(
        mmc: CMMCorePlus,
        config: dict,
) -> list[MDAEvent]:
    """_summary_

    Parameters
    ----------
    mmc : CMMCorePlus
        _description_
    config : dict
        _description_

    Returns
    -------
    list[MDAEvent]
        _description_
    """
    ao_mode = config['acq_config']['AO']['ao_mode']
    o2o3_mode = config['acq_config']['O2O3-autofocus']['o2o3_mode']
    
    opm_events: list[MDAEvent] = []
    
    if 'now' in o2o3_mode:
        o2o3_event = MDAEvent(**O2O3_af_event.model_dump())
        o2o3_event.action.data['Camera']['exposure_ms'] = float(config['O2O3-autofocus']['exposure_ms'])
        o2o3_event.action.data['Camera']['camera_crop'] = [
            config['Camera']['roi_center_x'] - int(config['Camera']['roi_crop_x']//2),
            config['Camera']['roi_center_y'] - int(config['O2O3-autofocus']['roi_crop_y']//2),
            config['Camera']['roi_crop_x'],
            config['O2O3-autofocus']['roi_crop_y']
        ]
        opm_events.append(o2o3_event)
        
    if 'now' in ao_mode:
        now = datetime.now()
        timestamp = f'{now.year:4d}{now.month:2d}{now.day:2d}_{now.hour:2d}{now.minute:2d}{now.second:2d}'
        
        # setup AO using values in the config widget, NOT the MDA widget
        AO_channel_states = [False] * len(config['OPM']['channel_ids']) 
        AO_channel_powers = [0.] * len(config['OPM']['channel_ids'])
        AO_active_channel_id = config['acq_config']['AO']['active_channel_id']
        AO_daq_mode = str(config['acq_config']['AO']['daq_mode'])
        if '2d' in AO_daq_mode:
            AO_camera_crop_y = int(config['acq_config']['camera_roi']['crop_y'])
        elif 'projection' in AO_daq_mode:
            AO_camera_crop_y = int(config['acq_config']['AO']['image_mirror_range_um']/mmc.getPixelSizeUm())
            
        AO_save_path = Path(str(config['acq_config']['AO']['save_dir_path'])) / Path(f'{timestamp}_ao_optimizeNOW')
        
        # Set the active channel in the daq channel list
        for chan_idx, chan_str in enumerate(config['OPM']['channel_ids']):
            if AO_active_channel_id==chan_str:
                AO_channel_states[chan_idx] = True
                AO_channel_powers[chan_idx] = config['acq_config']['AO']['active_channel_power']
                
        # check to make sure there exist a laser power > 0
        if sum(AO_channel_powers)==0:
            print('All AO lasers set to 0!')
            return
        
        # Define AO optimization action data   
        ao_action_data = {
            'AO' : {
                'starting_state': str(config['acq_config']['AO']['mirror_state']),
                'daq_mode':str(config['acq_config']['AO']['daq_mode']),
                'channel_states': AO_channel_states,
                'channel_powers' : AO_channel_powers,
                'exposure_ms': float(config['acq_config']['AO']['exposure_ms']),
                'modal_delta': float(config['acq_config']['AO']['mode_delta']),
                'modal_alpha':float(config['acq_config']['AO']['mode_alpha']),                        
                'iterations': int(config['acq_config']['AO']['num_iterations']),
                'metric': str(config['acq_config']['AO']['metric']),
                'image_mirror_range_um' : config['acq_config']['AO']['image_mirror_range_um'],
                'blanking': bool(True),
                'apply_existing': bool(False),
                'pos_idx': None,
                'output_path':AO_save_path
            },
            'Camera' : {
                'exposure_ms': config['acq_config']['AO']['exposure_ms'],
                'camera_crop' : [
                    int(config['acq_config']['camera_roi']['center_x'] - int(config['acq_config']['camera_roi']['crop_x']//2)),
                    int(config['acq_config']['camera_roi']['center_y'] - int(AO_camera_crop_y//2)),
                    int(config['acq_config']['camera_roi']['crop_x']),
                    int(AO_camera_crop_y)
                ]
            }
            
        }
        ao_optimize_event = MDAEvent(**AO_optimize_event.model_dump())
        ao_optimize_event.action.data.update(ao_action_data)
        opm_events.append(ao_optimize_event)
    
    return opm_events, None

def setup_projection(
        mmc: CMMCorePlus,
        config: dict,
        sequence: MDASequence,
        output: Path,
) -> list[MDAEvent]:
    """Parse GUI settings and setup event structure for Projection scan + AO + AF."""    
    AOmirror_setup = AOMirror.instance()

    #--------------------------------------------------------------------#
    # Compile acquisition settings from configuration
    #--------------------------------------------------------------------#
    # Get the acquisition modes
    opm_mode = config['acq_config']['opm_mode']
    ao_mode = config['acq_config']['AO']['ao_mode']
    o2o3_mode = config['acq_config']['O2O3-autofocus']['o2o3_mode']
    fluidics_mode = config['acq_config']['fluidics']
    
    # Get pixel size
    pixel_size_um = np.round(float(mmc.getPixelSizeUm()),3) # unit: um
    
    # Get the scan range, coverslip slope and overlaps
    coverslip_slope_x = config['acq_config']['projection_scan']['coverslip_slope_x']
    coverslip_slope_y = config['acq_config']['projection_scan']['coverslip_slope_y']
    scan_range_um = float(config['acq_config']['projection_scan']['scan_range_um'])
    tile_axis_overlap = float(config['acq_config']['projection_scan']['tile_axis_overlap'])
    
    # Get the camera crop values
    camera_crop_y = int(scan_range_um / pixel_size_um)
    camera_crop_x = int(config['acq_config']['camera_roi']['crop_x'])
    camera_center_y = int(config['acq_config']['camera_roi']['center_y'])
    camera_center_x = int(config['acq_config']['camera_roi']['center_x'])
    
    # Get channel settings
    laser_blanking = config['acq_config']['projection_scan']['laser_blanking']
    channel_states = config['acq_config']['projection_scan']['channel_states']
    channel_powers = config['acq_config']['projection_scan']['channel_powers']
    channel_exposures_ms = config['acq_config']['projection_scan']['channel_exposures_ms']
    channel_names = config['OPM']['channel_ids']
    
    # Compile active channel settings, has length of n_active_channels
    n_active_channels = sum(channel_states)
    active_channel_exps = []
    for ii, ch_state in enumerate(channel_states):
        if ch_state:
            active_channel_exps.append(np.round(channel_exposures_ms[ii],2))
        else:
            # set not used channel powers to 0
            channel_powers[ii] = 0
            
    if len(set(active_channel_exps))==1:
        interleaved_acq = True
    else:
        interleaved_acq = False
        
    if sum(channel_powers)==0:
        print('All lasers set to 0!')
        return None, None

    #----------------------------------------------------------------#
    # try to get camera conversion factor information
    try:
        offset = mmc.getProperty(
            config['Camera']['camera_id'],
            'CONVERSION FACTOR OFFSET'
        )
        e_to_ADU = mmc.getProperty(
            config['Camera']['camera_id'],
            'CONVERSION FACTOR COEFF'
        )
    except Exception:
        offset = 0.
        e_to_ADU = 1.

    #--------------------------------------------------------------------#
    # Compile mda acquisition settings from active tabs
    #--------------------------------------------------------------------#
    
    # Split apart sequence dictionary
    sequence_dict = json.loads(sequence.model_dump_json())
    mda_grid_plan = sequence_dict['grid_plan']
    mda_time_plan = sequence_dict['time_plan']
    mda_positions_plan = sequence_dict['stage_positions']
    mda_z_plan = sequence_dict['z_plan']
    
    if (mda_grid_plan is None) and (mda_positions_plan is None):
        print('Must select MDA grid or positions plan for projection scanning')
        return None, None
    
    #----------------------------------------------------------------#
    # Create custom action data
    #----------------------------------------------------------------#
            
    #----------------------------------------------------------------#
    # Create DAQ event
    daq_action_data = {
        'DAQ' : {
            'mode' : 'projection',
            'channel_states' : channel_states,
            'channel_powers' : channel_powers,
            'image_mirror_range_um' : scan_range_um,
            'interleaved' : interleaved_acq,
            'blanking' : laser_blanking, 
            'active_channels': None
        },
        'Camera' : {
            'exposure_channels' : channel_exposures_ms,
            'camera_crop' : [
                int(camera_center_x - camera_crop_x//2),
                int(camera_center_y - camera_crop_y//2),
                int(camera_crop_x),
                int(camera_crop_y),
            ]
        }
    }
    daq_update_event = MDAEvent(**DAQ_event.model_dump())
    daq_update_event.action.data.update(daq_action_data)
            
    #----------------------------------------------------------------#
    # Create the AO event data    
    if 'none' not in ao_mode:
        # Check the daq mode and set the camera properties
        AO_daq_mode = str(config['acq_config']['AO']['daq_mode'])
        if '2d' in AO_daq_mode:
            AO_camera_crop_y = int(config['acq_config']['camera_roi']['crop_y'])
        elif 'projection' in AO_daq_mode:
            AO_camera_crop_y = int(config['acq_config']['AO']['image_mirror_range_um']/mmc.getPixelSizeUm())
            
        # Set the active channel in the daq channel list
        AO_channel_states = [False] * len(channel_names) 
        AO_channel_powers = [0.] * len(channel_names)
        AO_active_channel_id = config['acq_config']['AO']['active_channel_id']
        for chan_idx, chan_str in enumerate(config['OPM']['channel_ids']):
            if AO_active_channel_id==chan_str:
                AO_channel_states[chan_idx] = True
                AO_channel_powers[chan_idx] = config['acq_config']['AO']['active_channel_power']
                
        # check to make sure there exist a laser power > 0
        if sum(AO_channel_powers)==0:
            print('All A.O. laser powers are set to 0!')
            return None, None
        
        # Define AO Grid generation action data
        if 'grid' in ao_mode:
            # Create a new directory in output.root for saving AO results
            ao_output_dir = output.parent / Path(f'{output.stem}_ao_grid_results')
            ao_output_dir.mkdir(exist_ok=True)
            AO_save_path = ao_output_dir
            ao_action_data = {
                'AO' : {
                    'starting_state': str(config['acq_config']['AO']['mirror_state']),
                    'stage_positions': None,
                    'num_scan_positions':int(config['acq_config']['AO']['num_scan_positions']),
                    'num_tile_positions':int(config['acq_config']['AO']['num_tile_positions']),
                    'apply_ao_map': bool(False),
                    'pos_idx': int(0),
                    'output_path':AO_save_path,
                    'ao_dict': {
                        'daq_mode': str(config['acq_config']['AO']['daq_mode']),
                        'channel_states': AO_channel_states,
                        'channel_powers' : AO_channel_powers,
                        'exposure_ms': float(config['acq_config']['AO']['exposure_ms']),
                        'modal_delta': float(config['acq_config']['AO']['mode_delta']),
                        'modal_alpha':float(config['acq_config']['AO']['mode_alpha']),                        
                        'iterations': int(config['acq_config']['AO']['num_iterations']),
                        'metric': str(config['acq_config']['AO']['metric']),
                        'image_mirror_range_um' : config['acq_config']['AO']['image_mirror_range_um'],
                    }
                },
                'Camera' : {
                    'exposure_ms': config['acq_config']['AO']['exposure_ms'],
                    'camera_crop' : [
                        int(camera_center_x - camera_crop_x//2),
                        int(camera_center_y - AO_camera_crop_y//2),
                        int(camera_crop_x),
                        int(AO_camera_crop_y)
                    ]
                }
            }
            ao_grid_event = MDAEvent(**AO_grid_event.model_dump())
            ao_grid_event.action.data.update(ao_action_data)
            AOmirror_setup.output_path = AO_save_path
            
        # Define AO optimization action data
        else:
            # Create a new directory in output.root for saving AO results
            ao_output_dir = output.parent / Path(f'{output.stem}_ao_results')
            ao_output_dir.mkdir(exist_ok=True)
            AO_save_path = ao_output_dir
            ao_action_data = {
                'AO' : {
                    'starting_state': str(config['acq_config']['AO']['mirror_state']),
                    'channel_states': AO_channel_states,
                    'channel_powers' : AO_channel_powers,
                    # 'starting_mirror_state':str(config['acq_config']['AO']['starting_mirror_state']),
                    'daq_mode': str(config['acq_config']['AO']['daq_mode']),
                    'exposure_ms': float(config['acq_config']['AO']['exposure_ms']),
                    'modal_delta': float(config['acq_config']['AO']['mode_delta']),
                    'modal_alpha':float(config['acq_config']['AO']['mode_alpha']),                        
                    'iterations': int(config['acq_config']['AO']['num_iterations']),
                    'metric': str(config['acq_config']['AO']['metric']),
                    'image_mirror_range_um' : config['acq_config']['AO']['image_mirror_range_um'],
                    'blanking': bool(True),
                    'apply_existing': bool(False),
                    'pos_idx': int(0),
                    'output_path':AO_save_path
                },
                'Camera' : {
                    'exposure_ms': config['acq_config']['AO']['exposure_ms'],
                    'camera_crop' : [
                        int(camera_center_x - camera_crop_x//2),
                        int(camera_center_y - AO_camera_crop_y//2),
                        int(camera_crop_x),
                        int(AO_camera_crop_y)
                    ]
                }
            }
            ao_optimization_event = MDAEvent(**AO_optimize_event.model_dump())
            ao_optimization_event.action.data.update(ao_action_data)
            AOmirror_setup.output_path = AO_save_path
        
    #----------------------------------------------------------------#
    # Create the o2o3 AF event data
    if 'none' not in o2o3_mode:
        o2o3_action_data = {
            'Camera' : {                    
                'exposure_ms' : config['O2O3-autofocus']['exposure_ms'],
                'camera_crop' : [
                    int(camera_center_x - camera_crop_x//2),
                    int(camera_center_y - config['acq_config']['O2O3-autofocus']['roi_crop_y']//2),
                    int(camera_crop_x),
                    int(config['acq_config']['O2O3-autofocus']['roi_crop_y'])
                    ]
                }
            }
        
        o2o3_event = MDAEvent(**O2O3_af_event.model_dump())
        o2o3_event.action.data.update(o2o3_action_data)
        
    #----------------------------------------------------------------#
    # Create the fluidics event data
    if 'none' not in fluidics_mode:
        fluidics_rounds = int(fluidics_mode)
        fp_action_data = {
            'Fluidics': {
                'total_rounds': fluidics_rounds,
                'current_round': 0
            }
        }
        
        fp_event = MDAEvent(**FP_event.model_dump())
        fp_event.action.data.update(fp_action_data)
        
    #----------------------------------------------------------------#
    # Compile mda positions from active tabs and config
    #----------------------------------------------------------------#

    # Define the time indexing
    if 'none' not in fluidics_mode:
        n_time_steps = fluidics_rounds
        time_interval = 0
    elif mda_time_plan is not None:
        n_time_steps = mda_time_plan['loops']
        time_interval = mda_time_plan['interval']
        if time_interval>0:
            timelapse_data = {
                'plan': {
                   'interval':time_interval,
                    'time_steps':n_time_steps,
                    'timepoint':0 
                }
            }
            timelapse_event = MDAEvent(**Timelapse_event.model_dump())
            timelapse_event.action.data.update(timelapse_data)
        
        if DEBUGGING:
            print(
                '\nTimelapse parameters:',
                f'\n  interval: {time_interval}',
                f'\n  timepoints: {n_time_steps}'
            )
    else:
        n_time_steps = 1
        time_interval = 0
        
    #----------------------------------------------------------------#
    # Generate xyz stage positions
    if mda_grid_plan is not None:
        stage_positions = stage_positions_from_grid(
            opm_mode=opm_mode,
            mda_grid_plan=mda_grid_plan,
            mda_z_plan=mda_z_plan,
            camera_crop_x=camera_crop_x,
            camera_crop_y=camera_crop_y,
            scan_range_um=scan_range_um,
            tile_axis_overlap=tile_axis_overlap,
            scan_axis_overlap=tile_axis_overlap,
            coverslip_slope_x=coverslip_slope_x,
            coverslip_slope_y=coverslip_slope_y
        )
    elif mda_positions_plan is not None:
        stage_positions = []
        for stage_pos in mda_positions_plan:
            stage_positions.append(
                {
                    'x': float(stage_pos['x']),
                    'y': float(stage_pos['y']),
                    'z': float(stage_pos['z'])
                }
            )
    n_stage_positions = len(stage_positions)
    
    #----------------------------------------------------------------#
    # Create MDA event structure
    #----------------------------------------------------------------#
    need_to_setup_stage = True
    need_to_setup_daq = True
    
    opm_events: list[MDAEvent] = []

    # update mirror positions array shape
    if 'none' not in ao_mode:
        AOmirror_setup.n_positions = n_stage_positions
        
    #----------------------------------------------------------------#
    # setup nD mirror-based AO-OPM acquisition event structure
    
    if DEBUGGING: 
            print(
                'Acquisition shape values:'
                f'\n  timepoints: {n_time_steps}',
                f'\n  Stage positions: {n_stage_positions}',
                f'\n  Active channels: {n_active_channels}',
                f'\n  AO frequency: {ao_mode}',
                f'\n  o2o3 focus frequency: {o2o3_mode}'
            )
            
    for time_idx in trange(n_time_steps, desc= 'Timepoints:', leave=True):
        # Run fluidics starting at the second timepoint if present
        if 'none' not in fluidics_mode and time_idx!=0:
            current_FP_event = MDAEvent(**fp_event.model_dump())
            current_FP_event.action.data['Fluidics']['round'] = int(time_idx)
            opm_events.append(current_FP_event)
        # Create pause events starting at the second timepoint for timelapse acq.
        elif (mda_time_plan is not None) and (time_idx>0):
            current_timepoint_event = MDAEvent(**timelapse_event.model_dump())
            current_timepoint_event.action.data['plan']['timepoint'] = time_idx
            opm_events.append(current_timepoint_event)

        for pos_idx in range(n_stage_positions):
            # Setup stage event to move to position
            # NOTE: Optimization events occur after stage move
            if need_to_setup_stage:      
                stage_event = MDAEvent(
                    action=CustomAction(
                        name= 'Stage-Move',
                        data = {
                            'Stage' : {
                                'x_pos' : stage_positions[pos_idx]['x'],
                                'y_pos' : stage_positions[pos_idx]['y'],
                                'z_pos' : stage_positions[pos_idx]['z'],
                            }
                        }
                    )
                )
                opm_events.append(stage_event)
                
                if n_stage_positions > 1:
                    need_to_setup_stage = True
                else:
                    need_to_setup_stage = False
                    
            #----------------------------------------------------------------#
            # Create 'start' and 'time-point' optimization events
            if pos_idx==0:
                # Create o2o3 optimization events for running at start only
                if ('start' in o2o3_mode) and (time_idx==0):
                    opm_events.append(o2o3_event)
                
                # Create o2o3 optimization events for running every time-point
                elif 'timepoint' in o2o3_mode:
                    opm_events.append(o2o3_event)

                # Create AO optimization events for running at the 'start'
                if 'start' in ao_mode:
                    if time_idx == 0:
                        # Run AO at t=0
                        if 'grid' in ao_mode:
                            current_ao_event = MDAEvent(**ao_grid_event.model_dump())
                            current_ao_event.action.data['AO']['stage_positions'] = stage_positions
                        else:
                            current_ao_event = MDAEvent(**ao_optimization_event.model_dump())
                        opm_events.append(current_ao_event)
                
                # Create AO optimization events for running each 'timepoint'
                elif 'timepoint' in ao_mode:
                    if 'grid' in ao_mode:
                        current_ao_event = MDAEvent(**ao_grid_event.model_dump())
                        current_ao_event.action.data['AO']['output_path'] = ao_output_dir / Path(f'time_{time_idx}_ao_grid')
                        current_ao_event.action.data['AO']['stage_positions'] = stage_positions
                    else:
                        # NOTE: for single AO optimization, force pos_idx=0
                        current_ao_event = MDAEvent(**ao_optimization_event.model_dump())
                        current_ao_event.action.data['AO']['output_path'] = ao_output_dir / Path(f'time_{time_idx}_ao_optimize')
                        current_ao_event.action.data['AO']['pos_idx'] = 0
                        current_ao_event.action.data['AO']['apply_existing'] = False
                    opm_events.append(current_ao_event)
            
            # Create mirror update events for 'start' and 'time-point' AO events
            # NOTE: Update mirror every time-point and stage-position
            # NOTE: for single position optimization, only refer to pos_idx==0, 
            #       Currently not filling the entire position array!
            if ('start' in ao_mode) or ('timepoint' in ao_mode):
                if 'grid' in ao_mode:
                    current_ao_update_event = MDAEvent(**ao_grid_event.model_dump())
                    current_ao_update_event.action.data['AO']['apply_ao_map'] = True
                    current_ao_update_event.action.data['AO']['pos_idx'] = pos_idx
                else:
                    current_ao_update_event = MDAEvent(**ao_optimization_event.model_dump())
                    current_ao_update_event.action.data['AO']['apply_existing'] = True
                    current_ao_update_event.action.data['AO']['pos_idx'] = 0
                opm_events.append(current_ao_update_event)
                
            #----------------------------------------------------------------#
            # Create 'xyz' optimization events 
            if 'xyz' in o2o3_mode:
                opm_events.append(o2o3_event)
                
            if 'xyz' in ao_mode:
                # NOTE: AO-grid does not have an xyz option
                # Run the AO at the first time-point for each position
                if time_idx==0:
                    current_ao_event = MDAEvent(**ao_optimization_event.model_dump())
                    current_ao_event.action.data['AO']['output_path'] = ao_output_dir / Path(f'pos_{pos_idx}_ao_results')
                    current_ao_event.action.data['AO']['pos_idx'] = int(pos_idx)
                    current_ao_event.action.data['AO']['apply_existing'] = False
                    opm_events.append(current_ao_event)
        
                # Update the mirror state for at each position for all time-points
                current_ao_update_event = MDAEvent(**ao_optimization_event.model_dump())
                current_ao_update_event.action.data['AO']['pos_idx'] = int(pos_idx)
                current_ao_update_event.action.data['AO']['apply_existing'] = True
                opm_events.append(current_ao_update_event)
             
            #----------------------------------------------------------------#
            # Handle acquiring images
            if interleaved_acq:
                # Update daq state to sequence all channels
                current_daq_event = MDAEvent(**daq_update_event.model_dump())
                current_daq_event.action.data['DAQ']['active_channels'] = channel_states
                current_daq_event.action.data['DAQ']['channel_powers'] = channel_powers
                current_daq_event.action.data['Camera']['exposure_channels'] = channel_exposures_ms
                opm_events.append(current_daq_event)
                
            # Create image events
            current_chan_idx = 0
            for chan_idx, chan_bool in enumerate(channel_states):               
                if chan_bool:
                    if not(interleaved_acq):
                        # Update daq for each channel separately
                        temp_channels = [False] * len(channel_states)
                        temp_exposures = [0] * len(channel_exposures_ms)
                        temp_powers = [0] * len(channel_powers)
        
                        temp_channels[chan_idx] = True
                        temp_exposures[chan_idx] = channel_exposures_ms[chan_idx]
                        temp_powers[chan_idx] = channel_powers[chan_idx]
                        
                        # create daq event for a single channel                    
                        current_daq_event = MDAEvent(**daq_update_event.model_dump())
                        current_daq_event.action.data['DAQ']['active_channels'] = temp_channels
                        current_daq_event.action.data['DAQ']['channel_powers'] = temp_powers
                        current_daq_event.action.data['Camera']['exposure_channels'] = temp_exposures
                        opm_events.append(current_daq_event)
                    
                    # Create image event for current t / p / c 
                    image_event = MDAEvent(
                        index=mappingproxy(
                            {
                                't': time_idx, 
                                'p': pos_idx, 
                                'c': current_chan_idx
                            }
                        ),
                        metadata = {
                            'DAQ' : {
                                'mode' : 'projection',
                                'image_mirror_range_um' : float(scan_range_um),
                                'active_channels' : channel_states,
                                'exposure_channels_ms': channel_exposures_ms,
                                'laser_powers' : channel_powers,
                                'interleaved' : interleaved_acq,
                                'blanking' : laser_blanking,
                                'current_channel' : channel_names[chan_idx]
                            },
                            'Camera' : {
                                'exposure_ms' : float(channel_exposures_ms[chan_idx]),
                                'camera_center_x' : int(camera_center_x),
                                'camera_center_y' : int(camera_center_y),
                                'camera_crop_x' : int(camera_crop_x),
                                'camera_crop_y' : int(camera_crop_y),
                                'offset' : float(offset),
                                'e_to_ADU': float(e_to_ADU)
                            },
                            'OPM' : {
                                'angle_deg' : float(config['OPM']['angle_deg']),
                                'camera_Zstage_orientation' : str(config['OPM']['camera_Zstage_orientation']),
                                'camera_XYstage_orientation' : str(config['OPM']['camera_XYstage_orientation']),
                                'camera_mirror_orientation' : str(config['OPM']['camera_mirror_orientation'])
                            },
                            'Stage' : {
                                'x_pos' : float(stage_positions[pos_idx]['x']),
                                'y_pos' : float(stage_positions[pos_idx]['y']),
                                'z_pos' : float(stage_positions[pos_idx]['z']),
                            }
                        }
                    )
                    opm_events.append(image_event)
                    current_chan_idx += 1

    # Check if path ends if .zarr. If so, use our OutputHandler
    if len(Path(output).suffixes) == 1 and Path(output).suffix ==  '.zarr':
        indice_sizes = {
            't' : int(np.maximum(1,n_time_steps)),
            'p' : int(np.maximum(1,n_stage_positions)),
            'c' : int(np.maximum(1,n_active_channels))
        }
        handler = OPMMirrorHandler(
            path=Path(output),
            indice_sizes=indice_sizes,
            delete_existing=True
        )
        print(f'\nUsing Qi2lab handler,\nindices: {indice_sizes}\n')
    else:
        print('Using default handler')
        handler = Path(output)
    
    return opm_events, handler

def setup_mirrorscan(
        mmc: CMMCorePlus,
        config: dict,
        sequence: MDASequence,
        output: Path,
) -> list[MDAEvent]:
    """Parse GUI settings and setup event structure for Mirror scan + AO + AF."""    
    AOmirror_setup = AOMirror.instance()
    OPMdaq_setup = OPMNIDAQ.instance()
    
    #--------------------------------------------------------------------#
    # Compile acquisition settings from configuration
    #--------------------------------------------------------------------#
    # Get the acquisition modes
    opm_mode = config['acq_config']['opm_mode']
    ao_mode = config['acq_config']['AO']['ao_mode']
    o2o3_mode = config['acq_config']['O2O3-autofocus']['o2o3_mode']
    fluidics_mode = config['acq_config']['fluidics']
    
    # Get pixel size and deskew Y-scale factor
    pixel_size_um = np.round(float(mmc.getPixelSizeUm()),3) # unit: um
    
    # Get the scan range, coverslip slope and overlaps
    coverslip_slope_x = config['acq_config']['mirror_scan']['coverslip_slope_x']
    coverslip_slope_y = config['acq_config']['mirror_scan']['coverslip_slope_y']
    scan_range_um = float(config['acq_config']['mirror_scan']['scan_range_um'])
    scan_step_um = float(config['acq_config']['mirror_scan']['scan_step_size_um'])
    tile_axis_overlap = float(config['acq_config']['mirror_scan']['tile_axis_overlap'])
    z_axis_overlap = float(config['acq_config']['mirror_scan']['z_axis_overlap'])
    
    # Flag for setting up a static mirror acquisition
    # NOTE: use the daq controller to calculate n_scan_steps
    if scan_range_um == 0.0:
        scan_mode = '2d'
        print("Setting up a 2d scan mode:")
        OPMdaq_setup.set_acquisition_params(
            scan_type='2d'
        )
        n_scan_steps = 1
    else:
        scan_mode = 'mirror'
        OPMdaq_setup.set_acquisition_params(
            scan_type="mirror",
            image_mirror_range_um=scan_range_um,
            image_mirror_step_um=scan_step_um
        )
        n_scan_steps = OPMdaq_setup.n_scan_steps

    # Get the camera crop values
    camera_crop_y = int(config['acq_config']['camera_roi']['crop_y'])
    camera_crop_x = int(config['acq_config']['camera_roi']['crop_x'])
    camera_center_y = int(config['acq_config']['camera_roi']['center_y'])
    camera_center_x = int(config['acq_config']['camera_roi']['center_x'])
    
    # Get channel settings
    laser_blanking = config['acq_config']['mirror_scan']['laser_blanking']
    channel_states = config['acq_config']['mirror_scan']['channel_states']
    channel_powers = config['acq_config']['mirror_scan']['channel_powers']
    channel_exposures_ms = config['acq_config']['mirror_scan']['channel_exposures_ms']
    channel_names = config['OPM']['channel_ids']
    
    # Compile active channel settings, has length of n_active_channels
    n_active_channels = sum(channel_states)
    active_channel_names = [_name for _, _name in zip(channel_states, channel_names) if _]
    active_channel_exps = []
    for ii, ch_state in enumerate(channel_states):
        if ch_state:
            active_channel_exps.append(np.round(channel_exposures_ms[ii],2))
        else:
            # set not used channel powers to 0
            channel_powers[ii] = 0
            
    if len(set(active_channel_exps))==1:
        interleaved_acq = True
    else:
        interleaved_acq = False         
           
    if sum(channel_powers)==0:
        print('All lasers set to 0!')
        return None, None
                
    #----------------------------------------------------------------#
    # try to get camera conversion factor information
    try:
        offset = mmc.getProperty(
            config['Camera']['camera_id'],
            'CONVERSION FACTOR OFFSET'
        )
        e_to_ADU = mmc.getProperty(
            config['Camera']['camera_id'],
            'CONVERSION FACTOR COEFF'
        )
    except Exception:
        offset = 0.
        e_to_ADU = 1.
        
    #--------------------------------------------------------------------#
    # Compile mda acquisition settings from active tabs
    #--------------------------------------------------------------------#
    
    # Split apart sequence dictionary
    sequence_dict = json.loads(sequence.model_dump_json())
    mda_grid_plan = sequence_dict['grid_plan']
    mda_time_plan = sequence_dict['time_plan']
    mda_positions_plan = sequence_dict['stage_positions']
    mda_z_plan = sequence_dict['z_plan']
    
    if (mda_grid_plan is None) and (mda_positions_plan is None):
        print('Must select MDA grid or positions plan for mirror scanning')
        return None, None
    
    #----------------------------------------------------------------#
    # Create custom action data
    #----------------------------------------------------------------#
           
    #----------------------------------------------------------------#
    # Create DAQ event
    daq_action_data = {
        'DAQ' : {
            'mode' : scan_mode,
            'channel_states' : channel_states,
            'channel_powers' : channel_powers,
            'image_mirror_step_um': scan_step_um,
            'image_mirror_range_um' : scan_range_um,
            'interleaved' : interleaved_acq,
            'blanking' : laser_blanking, 
            'active_channels': None
        },
        'Camera' : {
            'exposure_channels' : channel_exposures_ms,
            'camera_crop' : [
                int(camera_center_x - camera_crop_x//2),
                int(camera_center_y - camera_crop_y//2),
                int(camera_crop_x),
                int(camera_crop_y),
            ]
        }
    }

    # Create DAQ event to run before acquiring each 'image'
    daq_event = MDAEvent(**DAQ_event.model_dump())
    daq_event.action.data.update(daq_action_data)
            
    #----------------------------------------------------------------#
    # Create the AO event data    
    if 'none' not in ao_mode:
        # Create a new directory in output.root for saving AO results
        ao_output_dir = output.parent / Path(f'{output.stem}_ao_results')
        ao_output_dir.mkdir(exist_ok=True)
        
        AO_daq_mode = str(config['acq_config']['AO']['daq_mode'])
        if '2d' in AO_daq_mode:
            AO_camera_crop_y = int(config['acq_config']['camera_roi']['crop_y'])
        elif 'projection' in AO_daq_mode:
            AO_camera_crop_y = int(config['acq_config']['AO']['image_mirror_range_um']/mmc.getPixelSizeUm())
        AO_channel_states = [False] * len(channel_names) 
        AO_channel_powers = [0.] * len(channel_names)
        AO_active_channel_id = config['acq_config']['AO']['active_channel_id']
        
        # Set the active channel in the daq channel list
        for chan_idx, chan_str in enumerate(config['OPM']['channel_ids']):
            if AO_active_channel_id==chan_str:
                AO_channel_states[chan_idx] = True
                AO_channel_powers[chan_idx] = config['acq_config']['AO']['active_channel_power']
                
        # check to make sure there exist a laser power > 0
        if sum(AO_channel_powers)==0:
            print('All AO laser powers are set to 0!')
            return None, None
        
        if 'grid' in ao_mode:
            # Create a new directory in output.root for saving AO results
            ao_output_dir = output.parent / Path(f'{output.stem}_ao_grid_results')
            ao_output_dir.mkdir(exist_ok=True)
            AO_save_path = ao_output_dir
            # Define AO Grid generation action data
            ao_action_data = {
                'AO' : {
                    'starting_state': str(config['acq_config']['AO']['mirror_state']),
                    'stage_positions': None,
                    'num_scan_positions':int(config['acq_config']['AO']['num_scan_positions']),
                    'num_tile_positions':int(config['acq_config']['AO']['num_tile_positions']),
                    'ao_dict': {
                        'daq_mode': str(config['acq_config']['AO']['daq_mode']),
                        'channel_states': AO_channel_states,
                        'channel_powers' : AO_channel_powers,
                        'exposure_ms': float(config['acq_config']['AO']['exposure_ms']),
                        'modal_delta': float(np.round(config['acq_config']['AO']['mode_delta'], 2)),
                        'modal_alpha':float(np.round(config['acq_config']['AO']['mode_alpha'], 2)),                        
                        'iterations': int(config['acq_config']['AO']['num_iterations']),
                        'metric': str(config['acq_config']['AO']['metric']),
                        'image_mirror_range_um' : config['acq_config']['AO']['image_mirror_range_um'],
                    },
                    'apply_ao_map': bool(False),
                    'pos_idx': int(0),
                    'output_path':AO_save_path
                },
                'Camera' : {
                    'exposure_ms': config['acq_config']['AO']['exposure_ms'],
                    'camera_crop' : [
                        int(camera_center_x - camera_crop_x//2),
                        int(camera_center_y - AO_camera_crop_y//2),
                        int(camera_crop_x),
                        int(AO_camera_crop_y)
                    ]
                }
            }
            ao_grid_event = MDAEvent(**AO_grid_event.model_dump())
            ao_grid_event.action.data.update(ao_action_data)
            AOmirror_setup.output_path = AO_save_path
        else:
            # Create a new directory in output.root for saving AO results
            ao_output_dir = output.parent / Path(f'{output.stem}_ao_results')
            ao_output_dir.mkdir(exist_ok=True)
            AO_save_path = ao_output_dir
            
            # Define AO optimization action data   
            ao_action_data = {
                'AO' : {
                    'starting_state': str(config['acq_config']['AO']['mirror_state']),
                    'channel_states': AO_channel_states,
                    'channel_powers' : AO_channel_powers,
                    'daq_mode': str(config['acq_config']['AO']['daq_mode']),
                    'exposure_ms': float(config['acq_config']['AO']['exposure_ms']),
                    'modal_delta': float(config['acq_config']['AO']['mode_delta']),
                    'modal_alpha':float(config['acq_config']['AO']['mode_alpha']),                        
                    'iterations': int(config['acq_config']['AO']['num_iterations']),
                    'metric': str(config['acq_config']['AO']['metric']),
                    'image_mirror_range_um' : config['acq_config']['AO']['image_mirror_range_um'],
                    'blanking': bool(True),
                    'apply_existing': bool(False),
                    'pos_idx': int(0),
                    'output_path':AO_save_path
                },
                'Camera' : {
                    'exposure_ms': config['acq_config']['AO']['exposure_ms'],
                    'camera_crop' : [
                        int(camera_center_x - camera_crop_x//2),
                        int(camera_center_y - AO_camera_crop_y//2),
                        int(camera_crop_x),
                        int(AO_camera_crop_y)
                    ]
                }
            }
            ao_optimization_event = MDAEvent(**AO_optimize_event.model_dump())
            ao_optimization_event.action.data.update(ao_action_data)
            AOmirror_setup.output_path = AO_save_path
            
    #----------------------------------------------------------------#
    # Create the o2o3 AF event data
    if 'none' not in o2o3_mode:
        o2o3_action_data = {
            'Camera' : {                    
                'exposure_ms' : config['O2O3-autofocus']['exposure_ms'],
                'camera_crop' : [
                    int(camera_center_x - camera_crop_x//2),
                    int(camera_center_y - config['acq_config']['O2O3-autofocus']['roi_crop_y']//2),
                    int(camera_crop_x),
                    int(config['acq_config']['O2O3-autofocus']['roi_crop_y'])
                    ]
                }
            }
        
        o2o3_event = MDAEvent(**O2O3_af_event.model_dump())
        o2o3_event.action.data.update(o2o3_action_data)
        
    #----------------------------------------------------------------#
    # Create the fluidics event data
    if 'none' not in fluidics_mode:
        fluidics_rounds = int(fluidics_mode)
        fp_action_data = {
            'Fluidics': {
                'total_rounds': fluidics_rounds,
                'current_round': 0
            }
        }
        
        fp_event = MDAEvent(**FP_event.model_dump())
        fp_event.action.data.update(fp_action_data)
        
    #----------------------------------------------------------------#
    # Compile mda positions from active tabs and config
    #----------------------------------------------------------------#

    # Get time points
    if 'none' not in fluidics_mode:
        n_time_steps = fluidics_rounds
        time_interval = 0

    elif mda_time_plan is not None:
        n_time_steps = mda_time_plan['loops']
        time_interval = mda_time_plan['interval']

        if time_interval>0:
            timelapse_data = {
                'plan': {
                   'interval':time_interval,
                    'time_steps':n_time_steps,
                    'timepoint':0 
                }
            }
            timelapse_event = MDAEvent(**Timelapse_event.model_dump())
            timelapse_event.action.data.update(timelapse_data)
        
        if DEBUGGING:
            print(
                '\nTimelapse parameters:',
                f'\n  interval: {time_interval}',
                f'\n  timepoints: {n_time_steps}'
            )
    else:
        n_time_steps = 1
        time_interval = 0
        
    #----------------------------------------------------------------#
    # Generate xyz stage positions

    if mda_grid_plan is not None:
        stage_positions = stage_positions_from_grid(
            opm_mode=opm_mode,
            mda_grid_plan=mda_grid_plan,
            mda_z_plan=mda_z_plan,
            camera_crop_x=camera_crop_x,
            camera_crop_y=camera_crop_y,
            scan_range_um=scan_range_um,
            tile_axis_overlap=tile_axis_overlap,
            scan_axis_overlap=tile_axis_overlap,
            z_axis_overlap=z_axis_overlap,
            coverslip_slope_x=coverslip_slope_x,
            coverslip_slope_y=coverslip_slope_y            
        )
    elif mda_positions_plan is not None:
        stage_positions = []
        for stage_pos in mda_positions_plan:
            stage_positions.append(
                {
                    'x': float(stage_pos['x']),
                    'y': float(stage_pos['y']),
                    'z': float(stage_pos['z'])
                }
            )
    n_stage_positions = len(stage_positions)

    #----------------------------------------------------------------#
    # Create MDA event structure
    #----------------------------------------------------------------#
    need_to_setup_stage = True
    need_to_setup_DAQ = True

    opm_events: list[MDAEvent] = []

    if 'none' not in ao_mode:
        AOmirror_setup.n_positions = n_stage_positions
        
    # Check if run AF at start only
    if 'start' in o2o3_mode:
        opm_events.append(o2o3_event)
        
    # check if run AO at start only
    if 'start' in ao_mode:
        opm_events.append(ao_optimization_event)
    
    # check if generating AO grid        
    if 'grid' in ao_mode:
        current_ao_grid_event = MDAEvent(**ao_grid_event.model_dump())
        current_ao_grid_event.action.data['AO']['stage_positions'] = stage_positions
        opm_events.append(current_ao_grid_event)
        
    #----------------------------------------------------------------#
    # setup nD mirror-based AO-OPM acquisition event structure
    
    if DEBUGGING: 
            print(
                'Acquisition shape values:'
                f'\n  timepoints: {n_time_steps}',
                f'\n  Stage positions: {n_stage_positions}',
                f'\n  Active channels: {n_active_channels}',
                f'\n  Num scan positions: {n_scan_steps}',
                '\nMirror scan settings:\n'
                f'\n  num scan steps: {n_scan_steps}\n',
                f'\n  scan range (um): {scan_range_um}',
                f'\n  scan step (um): {scan_step_um}',
                f'\n  DAQ scan mode: {scan_mode}'
            )
            
    for time_idx in trange(n_time_steps, desc= 'Timepoints:', leave=True):
        # Run fluidics starting at the second timepoint if present
        if 'none' not in fluidics_mode and time_idx!=0:
            current_FP_event = MDAEvent(**fp_event.model_dump())
            current_FP_event.action.data['Fluidics']['round'] = int(time_idx)
            opm_events.append(current_FP_event)
        # Create pause events starting at the second timepoint for timelapse acq.
        elif (mda_time_plan is not None) and (time_idx>0) and (time_interval>0):
            current_timepoint_event = MDAEvent(**timelapse_event.model_dump())
            current_timepoint_event.action.data['plan']['timepoint'] = time_idx
            opm_events.append(current_timepoint_event)

        for pos_idx in range(n_stage_positions):
            # Setup stage event to move to position
            # NOTE: Optimization events occur after stage move
            if need_to_setup_stage:      
                stage_event = MDAEvent(
                    action=CustomAction(
                        name= 'Stage-Move',
                        data = {
                            'Stage' : {
                                'x_pos' : stage_positions[pos_idx]['x'],
                                'y_pos' : stage_positions[pos_idx]['y'],
                                'z_pos' : stage_positions[pos_idx]['z'],
                            }
                        }
                    )
                )
                opm_events.append(stage_event)
                
                if n_stage_positions > 1:
                    need_to_setup_stage = True
                else:
                    need_to_setup_stage = False
                    
            #----------------------------------------------------------------#
            # Create 'start' and 'time-point' optimization events
            if pos_idx==0:
                # Create o2o3 optimization events for running at start only
                if ('start' in o2o3_mode) and (time_idx==0):
                    opm_events.append(o2o3_event)
                
                # Create o2o3 optimization events for running every time-point
                elif 'timepoint' in o2o3_mode:
                    opm_events.append(o2o3_event)

                # Create AO optimization events for running at the 'start'
                if 'start' in ao_mode:
                    if time_idx == 0:
                        # Run AO at t=0
                        if 'grid' in ao_mode:
                            current_ao_event = MDAEvent(**ao_grid_event.model_dump())
                            current_ao_event.action.data['AO']['stage_positions'] = stage_positions
                        else:
                            current_ao_event = MDAEvent(**ao_optimization_event.model_dump())
                        opm_events.append(current_ao_event)
                
                # Create AO optimization events for running each 'timepoint'
                elif 'timepoint' in ao_mode:
                    if 'grid' in ao_mode:
                        current_ao_event = MDAEvent(**ao_grid_event.model_dump())
                        current_ao_event.action.data['AO']['output_path'] = ao_output_dir / Path(f'time_{time_idx}_ao_grid')
                        current_ao_event.action.data['AO']['stage_positions'] = stage_positions
                    else:
                        # NOTE: for single AO optimization, force pos_idx=0
                        current_ao_event = MDAEvent(**ao_optimization_event.model_dump())
                        current_ao_event.action.data['AO']['output_path'] = ao_output_dir / Path(f'time_{time_idx}_ao_optimize')
                        current_ao_event.action.data['AO']['pos_idx'] = 0
                        current_ao_event.action.data['AO']['apply_existing'] = False
                    opm_events.append(current_ao_event)
            
            # Create mirror update events for 'start' and 'time-point' AO events
            # NOTE: Update mirror every time-point and stage-position
            # NOTE: for single position optimization, only refer to pos_idx==0, 
            #       Currently not filling the entire position array!
            if ('start' in ao_mode) or ('timepoint' in ao_mode):
                if 'grid' in ao_mode:
                    current_ao_update_event = MDAEvent(**ao_grid_event.model_dump())
                    current_ao_update_event.action.data['AO']['apply_ao_map'] = True
                    current_ao_update_event.action.data['AO']['pos_idx'] = pos_idx
                else:
                    current_ao_update_event = MDAEvent(**ao_optimization_event.model_dump())
                    current_ao_update_event.action.data['AO']['apply_existing'] = True
                    current_ao_update_event.action.data['AO']['pos_idx'] = 0
                opm_events.append(current_ao_update_event)
            
            #----------------------------------------------------------------#
            # Create 'xyz' optimization events 
            if 'xyz' in o2o3_mode:
                opm_events.append(o2o3_event)
                
            if 'xyz' in ao_mode:
                # NOTE: AO-grid does not have an xyz option
                # Run the AO at the first time-point for each position
                if time_idx==0:
                    current_ao_event = MDAEvent(**ao_optimization_event.model_dump())
                    current_ao_event.action.data['AO']['output_path'] = ao_output_dir / Path(f'pos_{pos_idx}_ao_optimize')
                    current_ao_event.action.data['AO']['pos_idx'] = int(pos_idx)
                    current_ao_event.action.data['AO']['apply_existing'] = False
                    opm_events.append(current_ao_event)
        
                # Update the mirror state for at each position for all time-points
                current_ao_update_event = MDAEvent(**ao_optimization_event.model_dump())
                current_ao_update_event.action.data['AO']['pos_idx'] = int(pos_idx)
                current_ao_update_event.action.data['AO']['apply_existing'] = True
                opm_events.append(current_ao_update_event)
             
            #----------------------------------------------------------------#
            # Handle acquiring images
            # Mirror scan acquisition are forced interleaved acquisitions,
            # The daq programs each channel per mirror voltage position and advances 
            # each camera trigger output.
                     
            # Create image events     
            for scan_idx in range(n_scan_steps):
                current_chan_idx = 0
                for chan_idx, chan_bool in enumerate(channel_states):               
                    if chan_bool:
                                                    
                        # Create image event for current t / p / c / scan idx
                        image_event = MDAEvent(
                            index=mappingproxy(
                                {
                                    't': time_idx, 
                                    'p': pos_idx, 
                                    'c': current_chan_idx,
                                    'z': scan_idx
                                }
                            ),
                            metadata = {
                                'DAQ' : {
                                    'mode' : scan_mode,
                                    'image_mirror_range_um' : float(scan_range_um),
                                    'image_mirror_step_um': float(scan_step_um),
                                    'active_channels' : channel_states,
                                    'exposure_channels_ms': channel_exposures_ms,
                                    'laser_powers' : channel_powers,
                                    'interleaved' : interleaved_acq,
                                    'blanking' : laser_blanking,
                                    'current_channel' : channel_names[chan_idx]
                                },
                                'Camera' : {
                                    'exposure_ms' : float(channel_exposures_ms[chan_idx]),
                                    'camera_center_x' : int(camera_center_x),
                                    'camera_center_y' : int(camera_center_y),
                                    'camera_crop_x' : int(camera_crop_x),
                                    'camera_crop_y' : int(camera_crop_y),
                                    'offset' : float(offset),
                                    'e_to_ADU': float(e_to_ADU)
                                },
                                'OPM' : {
                                    'angle_deg' : float(config['OPM']['angle_deg']),
                                    'camera_Zstage_orientation' : str(config['OPM']['camera_Zstage_orientation']),
                                    'camera_XYstage_orientation' : str(config['OPM']['camera_XYstage_orientation']),
                                    'camera_mirror_orientation' : str(config['OPM']['camera_mirror_orientation'])
                                },
                                'Stage' : {
                                    'x_pos' : float(stage_positions[pos_idx]['x']),
                                    'y_pos' : float(stage_positions[pos_idx]['y']),
                                    'z_pos' : float(stage_positions[pos_idx]['z']),
                                }
                            }
                        )
                        opm_events.append(image_event)
                        current_chan_idx += 1

    # Check if path ends if .zarr. If so, use our OutputHandler
    if len(Path(output).suffixes) == 1 and Path(output).suffix ==  '.zarr':
        indice_sizes = {
            't' : int(np.maximum(1,n_time_steps)),
            'p' : int(np.maximum(1,n_stage_positions)),
            'c' : int(np.maximum(1,n_active_channels)),
            'z' : int(np.maximum(1,n_scan_steps))
        }
        handler = OPMMirrorHandler(
            path=Path(output),
            indice_sizes=indice_sizes,
            delete_existing=True
        )
        print(f'\nUsing Qi2lab handler,\nindices: {indice_sizes}\n')
    else:
        print('Using default handler')
        handler = Path(output)
            
    return opm_events, handler

def setup_stagescan(
        mmc: CMMCorePlus,
        config: dict,
        sequence: MDASequence,
        output: Path,
) -> list[MDAEvent]:
    """Parse GUI settings and setup event structure for stage scan + AO + AF."""
    
    AOmirror_setup = AOMirror.instance()
    
    #--------------------------------------------------------------------#
    # Compile acquisition settings from configuration
    #--------------------------------------------------------------------#
    # Get the acquisition modes
    opm_mode = config['acq_config']['opm_mode']
    ao_mode = config['acq_config']['AO']['ao_mode']
    o2o3_mode = config['acq_config']['O2O3-autofocus']['o2o3_mode']
    fluidics_mode = config['acq_config']['fluidics']
    
    # Get the camera crop values
    camera_crop_y = int(config['acq_config']['camera_roi']['crop_y'])
    camera_crop_x = int(config['acq_config']['camera_roi']['crop_x'])
    camera_center_y = int(config['acq_config']['camera_roi']['center_y'])
    camera_center_x = int(config['acq_config']['camera_roi']['center_x'])
    
    # Get pixel size and deskew Y-scale factor
    pixel_size_um = np.round(float(mmc.getPixelSizeUm()),3) # unit: um
    opm_angle_scale = np.sin((np.pi/180.)*float(config['OPM']['angle_deg']))
    
    # Get the stage scan range, coverslip slope, and maximum CS dz change
    coverslip_slope = config['acq_config']['stage_scan']['coverslip_slope_x']
    scan_axis_max_range = config['acq_config']['stage_scan']['stage_scan_range_um']
    coverslip_max_dz = float(config['acq_config']['stage_scan']['coverslip_max_dz'])
    
    # Get the tile overlap settings
    tile_axis_overlap = float(config['acq_config']['stage_scan']['tile_axis_overlap'])
    scan_axis_step_um = float(config['acq_config']['stage_scan']['scan_step_size_um'])  # unit: um 
    scan_tile_overlap_um = camera_crop_y * opm_angle_scale * pixel_size_um + float(config['acq_config']['stage_scan']['scan_axis_overlap_um'])
    scan_tile_overlap_mm = scan_tile_overlap_um/1000.
    
    # Get the excess start / end
    excess_starting_images = int(config['acq_config']['stage_scan']['excess_start_frames'])
    excess_ending_images = int(config['acq_config']['stage_scan']['excess_end_frames'])
    
    #----------------------------------------------------------------#
    # Get channel settings
    laser_blanking = config['acq_config'][opm_mode+'_scan']['laser_blanking']
    channel_states = config['acq_config'][opm_mode+'_scan']['channel_states']
    channel_powers = config['acq_config'][opm_mode+'_scan']['channel_powers']
    channel_exposures_ms = config['acq_config'][opm_mode+'_scan']['channel_exposures_ms']
    channel_names = config['OPM']['channel_ids']
    
    n_active_channels = sum(channel_states)
    active_channel_names = [_name for _, _name in zip(channel_states, channel_names) if _]
    active_channel_exps = []
    for ii, ch_state in enumerate(channel_states):
        if ch_state:
            active_channel_exps.append(np.round(channel_exposures_ms[ii],2))
        else:
            # set not used channel powers to 0
            channel_powers[ii] = 0
        
    # Interleave only available if all channels have the same exposure.
    if len(set(active_channel_exps))==1:
        interleaved_acq = True
    else:
        interleaved_acq = False
            
    if sum(channel_powers)==0:
        print('All lasers set to 0!')
        return None, None
        
    # Get the exposure
    exposure_ms = np.round(active_channel_exps[0],2)
    exposure_s = np.round(exposure_ms / 1000.,2)
    
    #----------------------------------------------------------------#
    # try to get camera conversion factor information
    try:
        offset = mmc.getProperty(
            config['Camera']['camera_id'],
            'CONVERSION FACTOR OFFSET'
        )
        e_to_ADU = mmc.getProperty(
            config['Camera']['camera_id'],
            'CONVERSION FACTOR COEFF'
        )
    except Exception:
        offset = 0.
        e_to_ADU = 1.

    #--------------------------------------------------------------------#
    # Compile mda acquisition settings from active tabs
    #--------------------------------------------------------------------#
    
    # Split apart sequence dictionary
    sequence_dict = json.loads(sequence.model_dump_json())
    mda_grid_plan = sequence_dict['grid_plan']
    mda_positions_plan = sequence_dict['stage_positions']
    mda_time_plan = sequence_dict['time_plan']
    mda_z_plan = sequence_dict['z_plan']
    
    if (mda_grid_plan is None) and (mda_positions_plan is None):
        print('Must select MDA grid or positions plan for stage scanning')
        return None, None

    #----------------------------------------------------------------#
    # Create custom action data
    #----------------------------------------------------------------#
            
    #----------------------------------------------------------------#
    # Create DAQ event
    # NOTE: stage scan is interleaved
    daq_action_data = {
        'DAQ' : {
            'mode' : 'stage',
            'channel_states' : channel_states,
            'channel_powers' : channel_powers,
            'interleaved' : interleaved_acq,
            'blanking' : laser_blanking, 
            'active_channels': channel_states
        },
        'Camera' : {
            'exposure_channels' : channel_exposures_ms,
            'camera_crop' : [
                int(camera_center_x - camera_crop_x//2),
                int(camera_center_y - camera_crop_y//2),
                int(camera_crop_x),
                int(camera_crop_y),
            ]
        }
    }

    # Create DAQ event to run before acquiring each 'image'
    daq_event = MDAEvent(**DAQ_event.model_dump())
    daq_event.action.data.update(daq_action_data)
            
    #----------------------------------------------------------------#
    # Create the AO event data    
    if 'none' not in ao_mode:
        # Create a new directory in output.root for saving AO results
        ao_output_dir = output.parent / Path(f'{output.stem}_ao_results')
        ao_output_dir.mkdir(exist_ok=True)
        AO_save_path = ao_output_dir

        # Check the daq mode and set the camera properties
        AO_daq_mode = str(config['acq_config']['AO']['daq_mode'])
        if '2d' in AO_daq_mode:
            AO_camera_crop_y = int(config['acq_config']['camera_roi']['crop_y'])
        elif 'projection' in AO_daq_mode:
            AO_camera_crop_y = int(config['acq_config']['AO']['image_mirror_range_um']/mmc.getPixelSizeUm())
            
        # Set the active channel in the daq channel list
        AO_channel_states = [False] * len(channel_names) 
        AO_channel_powers = [0.] * len(channel_names)
        AO_active_channel_id = config['acq_config']['AO']['active_channel_id']
        for chan_idx, chan_str in enumerate(config['OPM']['channel_ids']):
            if AO_active_channel_id==chan_str:
                AO_channel_states[chan_idx] = True
                AO_channel_powers[chan_idx] = config['acq_config']['AO']['active_channel_power']
                
        # check to make sure there exist a laser power > 0
        if sum(AO_channel_powers)==0:
            print('All A.O. laser powers are set to 0!')
            return None, None
        
        # Define AO Grid generation action data
        if 'grid' in ao_mode:
            ao_action_data = {
                'AO' : {
                    'starting_state': str(config['acq_config']['AO']['mirror_state']),
                    'stage_positions': None,
                    'num_scan_positions':int(config['acq_config']['AO']['num_scan_positions']),
                    'num_tile_positions':int(config['acq_config']['AO']['num_tile_positions']),
                    'apply_ao_map': bool(False),
                    'pos_idx': int(0),
                    'output_path':AO_save_path,
                    'ao_dict': {
                        'daq_mode': str(config['acq_config']['AO']['daq_mode']),
                        'channel_states': AO_channel_states,
                        'channel_powers' : AO_channel_powers,
                        'exposure_ms': float(config['acq_config']['AO']['exposure_ms']),
                        'modal_delta': float(config['acq_config']['AO']['mode_delta']),
                        'modal_alpha':float(config['acq_config']['AO']['mode_alpha']),                        
                        'iterations': int(config['acq_config']['AO']['num_iterations']),
                        'metric': str(config['acq_config']['AO']['metric']),
                        'image_mirror_range_um' : config['acq_config']['AO']['image_mirror_range_um'],
                    }
                },
                'Camera' : {
                    'exposure_ms': config['acq_config']['AO']['exposure_ms'],
                    'camera_crop' : [
                        int(camera_center_x - camera_crop_x//2),
                        int(camera_center_y - AO_camera_crop_y//2),
                        int(camera_crop_x),
                        int(AO_camera_crop_y)
                    ]
                }
            }
            ao_grid_event = MDAEvent(**AO_grid_event.model_dump())
            ao_grid_event.action.data.update(ao_action_data)
            AOmirror_setup.output_path = AO_save_path
            
        # Define AO optimization action data
        else:
            ao_action_data = {
                'AO' : {
                    'starting_state': str(config['acq_config']['AO']['mirror_state']),
                    'channel_states': AO_channel_states,
                    'channel_powers' : AO_channel_powers,
                    'daq_mode': str(config['acq_config']['AO']['daq_mode']),
                    'exposure_ms': float(config['acq_config']['AO']['exposure_ms']),
                    'modal_delta': float(config['acq_config']['AO']['mode_delta']),
                    'modal_alpha':float(config['acq_config']['AO']['mode_alpha']),                        
                    'iterations': int(config['acq_config']['AO']['num_iterations']),
                    'metric': str(config['acq_config']['AO']['metric']),
                    'image_mirror_range_um' : config['acq_config']['AO']['image_mirror_range_um'],
                    'blanking': bool(True),
                    'apply_existing': bool(False),
                    'pos_idx': int(0),
                    'output_path':AO_save_path
                },
                'Camera' : {
                    'exposure_ms': config['acq_config']['AO']['exposure_ms'],
                    'camera_crop' : [
                        int(camera_center_x - camera_crop_x//2),
                        int(camera_center_y - AO_camera_crop_y//2),
                        int(camera_crop_x),
                        int(AO_camera_crop_y)
                    ]
                }
            }
            ao_optimization_event = MDAEvent(**AO_optimize_event.model_dump())
            ao_optimization_event.action.data.update(ao_action_data)
            AOmirror_setup.output_path = AO_save_path
            
    #----------------------------------------------------------------#
    # Create the o2o3 AF event data
    if 'none' not in o2o3_mode:
        o2o3_action_data = {
            'Camera' : {                    
                'exposure_ms' : config['O2O3-autofocus']['exposure_ms'],
                'camera_crop' : [
                    int(camera_center_x - camera_crop_x//2),
                    int(camera_center_y - config['acq_config']['O2O3-autofocus']['roi_crop_y']//2),
                    int(camera_crop_x),
                    int(config['acq_config']['O2O3-autofocus']['roi_crop_y'])
                    ]
                }
            }
        
        o2o3_event = MDAEvent(**O2O3_af_event.model_dump())
        o2o3_event.action.data.update(o2o3_action_data)
            
    #----------------------------------------------------------------#
    # Create the fluidics event data
    if 'none' not in fluidics_mode:
        fluidics_rounds = int(fluidics_mode)
        fp_action_data = {
            'Fluidics': {
                'total_rounds': fluidics_rounds,
                'current_round': int(0)
            }
        }
        
        fp_event = MDAEvent(**FP_event.model_dump())
        fp_event.action.data.update(fp_action_data)
        
    #----------------------------------------------------------------#
    # Compile mda positions from active tabs, and config
    #----------------------------------------------------------------#

    # Define the time indexing
    if 'none' not in fluidics_mode:
        n_time_steps = fluidics_rounds
        time_interval = 0
    elif mda_time_plan is not None:
        n_time_steps = mda_time_plan['loops']
        time_interval = mda_time_plan['interval']
        if time_interval>0:
            timelapse_data = {
                'plan': {
                   'interval':time_interval,
                    'time_steps':n_time_steps,
                    'timepoint':0 
                }
            }
            timelapse_event = MDAEvent(**Timelapse_event.model_dump())
            timelapse_event.action.data.update(timelapse_data)
        
        if DEBUGGING:
            print(
                '\nTimelapse parameters:',
                f'\n  interval: {time_interval}',
                f'\n  timepoints: {n_time_steps}'
            )
    else:
        n_time_steps = 1
        time_interval = 0

    #----------------------------------------------------------------#
    # Generate xyz stage positions
    stage_positions = []
    
    # grab grid plan extents
    min_y_pos = mda_grid_plan['bottom']
    max_y_pos = mda_grid_plan['top']
    min_x_pos = mda_grid_plan['left']
    max_x_pos = mda_grid_plan['right']
    
    if mda_z_plan is not None:
        max_z_pos = float(mda_z_plan['top'])
        min_z_pos = float(mda_z_plan['bottom'])
    else:
        min_z_pos = mmc.getZPosition()
        max_z_pos = mmc.getZPosition()
        
    # Set grid axes ranges
    range_x_um = np.round(np.abs(max_x_pos - min_x_pos),2)
    range_y_um = np.round(np.abs(max_y_pos - min_y_pos),2)
    range_z_um = np.round(np.abs(max_z_pos - min_z_pos),2)
    
    # Define coverslip bounds, to offset Z positions
    cs_min_pos = min_z_pos
    cs_max_pos = cs_min_pos + range_x_um * coverslip_slope
    cs_range_um = np.round(np.abs(cs_max_pos - cs_min_pos),2)

    #--------------------------------------------------------------------#
    # Calculate tile steps / range
    z_axis_step_max = (
        camera_crop_y
        * pixel_size_um
        * opm_angle_scale 
        * (1-tile_axis_overlap)
    )
    tile_axis_step_max = (
        camera_crop_x
        * pixel_size_um
        * (1-tile_axis_overlap)
    )
    
    if coverslip_slope != 0:
        # Set the max scan range using coverslip slope
        scan_axis_max_range = np.abs(coverslip_max_dz / coverslip_slope)
        
        # Use the config value as a global maximum
        if scan_axis_max_range > float(config['acq_config']['stage_scan']['stage_scan_range_um']):
            scan_axis_max_range = float(config['acq_config']['stage_scan']['stage_scan_range_um'])
    else:
        scan_axis_max_range = float(config['acq_config']['stage_scan']['stage_scan_range_um'])

    # Correct directions for stage moves
    if min_z_pos > max_z_pos:
        z_axis_step_max *= -1
    if min_x_pos > max_x_pos:
        min_x_pos, max_x_pos = max_x_pos, min_x_pos        

    if DEBUGGING: 
        print(
            '\n\nXYZ Stage position settings:',
            f'\n  Scan start: {min_x_pos}',
            f'\n  Scan end: {max_x_pos}',
            f'\n  Tile start: {min_y_pos}',
            f'\n  Tile end: {max_y_pos}',
            f'\n  Z position min:{min_z_pos}',
            f'\n  Z position max:{max_z_pos}',
            f'\n  Coverslip slope: {coverslip_slope}',
            f'\n  Coverslip low: {cs_min_pos}',
            f'\n  Coverslip high: {cs_max_pos}',
            f'\n  Max scan range (CS used?:{coverslip_slope!=0}): {scan_axis_max_range}\n'
        )
        
    #--------------------------------------------------------------------#
    # calculate scan axis tile locations, units: mm and s
    
    # Break scan range up using max scan range
    if scan_axis_max_range >= range_x_um:
        n_scan_positions = 1
        scan_tile_length_um = range_x_um
    else:
        # Round up so that the scan length is never longer than the max scan range
        n_scan_positions = int(
            np.ceil(range_x_um / (scan_axis_max_range))
        )
        scan_tile_length_um = np.round(
            (range_x_um/n_scan_positions) + (n_scan_positions-1)*(scan_tile_overlap_um/(n_scan_positions)),
            2
        )
    scan_axis_step_mm = scan_axis_step_um / 1000. # unit: mm
    scan_axis_start_mm = min_x_pos / 1000. # unit: mm
    scan_axis_end_mm = max_x_pos / 1000. # unit: mm
    scan_tile_length_mm = np.round(scan_tile_length_um / 1000.,2) # unit: mm

    # Initialize scan position start/end arrays with the scan start / end values
    scan_axis_start_pos_mm = np.full(n_scan_positions, scan_axis_start_mm)
    scan_axis_end_pos_mm = np.full(n_scan_positions, scan_axis_end_mm)
    for ii in range(n_scan_positions):
        scan_axis_start_pos_mm[ii] = scan_axis_start_mm + ii * (scan_tile_length_mm - scan_tile_overlap_mm)
        scan_axis_end_pos_mm[ii] = scan_axis_start_pos_mm[ii] + scan_tile_length_mm
        
    scan_axis_start_pos_mm = np.round(scan_axis_start_pos_mm,2)
    scan_axis_end_pos_mm = np.round(scan_axis_end_pos_mm,2)
    scan_tile_length_w_overlap_mm = np.round(np.abs(scan_axis_end_pos_mm[0]-scan_axis_start_pos_mm[0]),2)
    scan_axis_positions = np.rint(scan_tile_length_w_overlap_mm / scan_axis_step_mm).astype(int)
    scan_axis_speed = np.round(scan_axis_step_mm / exposure_s / n_active_channels,5) 
    scan_tile_sizes = [np.round(np.abs(scan_axis_end_pos_mm[ii]-scan_axis_start_pos_mm[ii]),2) for ii in range(len(scan_axis_end_pos_mm))]
    
    # Check for scan speed actual settings
    mmc.setProperty(mmc.getXYStageDevice(), 'MotorSpeedX-S(mm/s)', scan_axis_speed)
    mmc.waitForDevice(mmc.getXYStageDevice())
    actual_speed_x = float(mmc.getProperty(mmc.getXYStageDevice(), 'MotorSpeedX-S(mm/s)'))
    actual_exposure = np.round(scan_axis_step_mm / actual_speed_x / n_active_channels,5) 
    channel_exposures_ms = [actual_exposure*1000]*len(channel_exposures_ms)
    
    # update acq settings with the actual exposure and stage scan speed
    daq_event.action.data['Camera']['exposure_channels'] = channel_exposures_ms
    exposure_ms = actual_exposure*1000
    exposure_s = actual_exposure
    scan_axis_speed = actual_speed_x
    
    if DEBUGGING: 
        print(
            '\nScan-axis calculated parameters:',
            f'\n  Number scan tiles: {n_scan_positions}',
            f'\n  tile length um: {scan_tile_length_um}',
            f'\n  tile overlap um: {scan_tile_overlap_um}',
            f'\n  tile length mm: {scan_tile_length_mm}',
            f'\n  tile length with overlap (mm): {scan_tile_length_w_overlap_mm}',
            f'\n  Is the scan tile w/ overlap the same as the scan tile length?: {scan_tile_length_mm==scan_tile_length_w_overlap_mm}',
            f'\n  step size (mm): {scan_axis_step_mm}',
            f'\n  exposure: {exposure_s}',
            f'\n  number of active channels: {n_active_channels}',
            f'\n  Scan axis speed (mm/s): {scan_axis_speed}\n'
            f'\n  Stage scan positions, units: mm',
            f'\n  Scan axis start positions: {scan_axis_start_pos_mm}.',
            f'\n  Scan axis end positions: {scan_axis_end_pos_mm}.',
            f'\n  Number of scan positions: {scan_axis_positions}',
            f'\n  Are all scan tiles the same size: {np.allclose(scan_tile_sizes, scan_tile_sizes[0])}'
        )

    #--------------------------------------------------------------------#
    # Generate tile axis positions
    n_tile_positions = int(np.ceil(range_y_um / tile_axis_step_max)) + 1
    tile_axis_positions = np.round(np.linspace(min_y_pos, max_y_pos, n_tile_positions),2)
    if n_tile_positions==1:
        tile_axis_step = 0
    else:
        tile_axis_step = tile_axis_positions[1]-tile_axis_positions[0]
    
    if DEBUGGING:
        print('Tile axis positions units: um')
        print(f'Tile axis positions: {tile_axis_positions}')
        print(f'Num tile axis positions: {n_tile_positions}')
        print(f'Tile axis step: {tile_axis_step}')

    #--------------------------------------------------------------------#
    # Generate z axis positions, ignoring coverslip slope
    n_z_positions = int(np.ceil(np.abs(range_z_um / z_axis_step_max))) + 1
    z_positions = np.round(np.linspace(min_z_pos, max_z_pos, n_z_positions), 2)
    if n_z_positions==1:
        z_axis_step_um = 0.
    else:
        z_axis_step_um = np.round(z_positions[1] - z_positions[0],2)

    # Calculate the stage z change along the scan axis
    dz_per_scan_tile = (cs_range_um / n_scan_positions) * np.sign(coverslip_slope)

    if DEBUGGING:
        print(
            'Z axis positions, units: um',
            f'\n  Z axis positions: {z_positions}',
            f'\n  Z axis range: {range_z_um} um',
            f'\n  Z axis step: {z_axis_step_um} um',
            f'\n  Num z axis positions: {n_z_positions}',
            f'\n  Z offset per x-scan-tile: {dz_per_scan_tile} um',
            f'\n  Z axis step max: {z_axis_step_max}'
        )   

    #--------------------------------------------------------------------#
    # Generate stage positions 
    n_stage_positions = n_scan_positions * n_tile_positions * n_z_positions
    stage_positions = []
    for z_idx in range(n_z_positions):
        for scan_idx in range(n_scan_positions):
            for tile_idx in range(n_tile_positions):
                stage_positions.append(
                    {
                        'x': float(np.round(scan_axis_start_pos_mm[scan_idx]*1000, 2)),
                        'y': float(np.round(tile_axis_positions[tile_idx], 2)),
                        'z': float(np.round(z_positions[z_idx] + dz_per_scan_tile*scan_idx, 2))
                    }
                )
                
    #----------------------------------------------------------------#
    # Create MDA event structure
    #----------------------------------------------------------------#

    opm_events: list[MDAEvent] = []

    if 'none' not in ao_mode:
        AOmirror_setup.n_positions = n_stage_positions
        
    # Flags to help ensure sequence-able events are kept together 
    need_to_setup_stage = True
            
    #----------------------------------------------------------------#
    # setup nD mirror-based AO-OPM acquisition event structure
    
    if DEBUGGING: 
            print(
                'Acquisition shape values:'
                f'\n  timepoints: {n_time_steps}',
                f'\n  Stage positions: {n_stage_positions}',
                f'\n  Scan positions: {scan_axis_positions+int(excess_starting_images)+int(excess_ending_images)}',
                f'\n  Active channels: {n_active_channels}',
                f'\n  Excess frame values (start/end): {excess_starting_images} / {excess_ending_images}',
                f'\n  AO frequency: {ao_mode}',
                f'\n  o2o3 focus frequency: {o2o3_mode}'
            )
            
    for time_idx in trange(n_time_steps, desc= 'Timepoints:', leave=True):
        # Run fluidics starting at the second timepoint if present
        if 'none' not in fluidics_mode and time_idx!=0:
            current_FP_event = MDAEvent(**fp_event.model_dump())
            current_FP_event.action.data['Fluidics']['round'] = int(time_idx)
            opm_events.append(current_FP_event)
        # Create pause events starting at the second timepoint for timelapse acq.
        elif (mda_time_plan is not None) and (time_idx>0):
            current_timepoint_event = MDAEvent(**timelapse_event.model_dump())
            current_timepoint_event.action.data['plan']['timepoint'] = time_idx
            opm_events.append(current_timepoint_event)
            
        if 'time' in o2o3_mode:
            opm_events.append(o2o3_event)
        
        pos_idx = 0        
        for z_idx in trange(n_z_positions, desc= 'Z-axis-tiles:', leave=False):
            for scan_idx in trange(n_scan_positions, desc= 'Scan-axis-tiles:', leave=False):
                for tile_idx in trange(n_tile_positions, desc= 'Tile-axis-tiles:', leave=False):
                    if need_to_setup_stage:      
                        stage_event = MDAEvent(
                            action=CustomAction(
                                name= 'Stage-Move',
                                data = {
                                    'Stage' : {
                                        'x_pos' : stage_positions[pos_idx]['x'],
                                        'y_pos' : stage_positions[pos_idx]['y'],
                                        'z_pos' : stage_positions[pos_idx]['z'],
                                    }
                                }
                            )
                        )
                        opm_events.append(stage_event)
                        
                        if n_stage_positions > 1:
                            need_to_setup_stage = True
                        else:
                            need_to_setup_stage = False
                
                    #----------------------------------------------------------------#
                    # Create 'start' and 'time-point' optimization events
                    if pos_idx==0:
                        # Create o2o3 optimization events for running at start only
                        if ('start' in o2o3_mode) and (time_idx==0):
                            opm_events.append(o2o3_event)
                        
                        # Create o2o3 optimization events for running every time-point
                        elif 'timepoint' in o2o3_mode:
                            opm_events.append(o2o3_event)

                        # Create AO optimization events for running at the 'start'
                        if 'start' in ao_mode:
                            if time_idx == 0:
                                # Run AO at t=0
                                if 'grid' in ao_mode:
                                    current_ao_event = MDAEvent(**ao_grid_event.model_dump())
                                    current_ao_event.action.data['AO']['stage_positions'] = stage_positions
                                else:
                                    current_ao_event = MDAEvent(**ao_optimization_event.model_dump())
                                opm_events.append(current_ao_event)
                                if 'grid' in ao_mode:
                                    opm_events.append(stage_event)
                        
                        # Create AO optimization events for running each 'timepoint'
                        elif 'timepoint' in ao_mode:
                            if 'grid' in ao_mode:
                                current_ao_event = MDAEvent(**ao_grid_event.model_dump())
                                current_ao_event.action.data['AO']['output_path'] = ao_output_dir / Path(f'time_{time_idx}_ao_grid')
                                current_ao_event.action.data['AO']['stage_positions'] = stage_positions
                            else:
                                # NOTE: for single AO optimization, force pos_idx=0
                                current_ao_event = MDAEvent(**ao_optimization_event.model_dump())
                                current_ao_event.action.data['AO']['output_path'] = ao_output_dir / Path(f'time_{time_idx}_ao_optimize')
                                current_ao_event.action.data['AO']['pos_idx'] = 0
                                current_ao_event.action.data['AO']['apply_existing'] = False
                            opm_events.append(current_ao_event)
                    
                    # Create mirror update events for 'start' and 'time-point' AO events
                    # NOTE: Update mirror every time-point and stage-position
                    # NOTE: for single position optimization, only refer to pos_idx==0, 
                    #       Currently not filling the entire position array!
                    if ('start' in ao_mode) or ('timepoint' in ao_mode):
                        if 'grid' in ao_mode:
                            current_ao_update_event = MDAEvent(**ao_grid_event.model_dump())
                            current_ao_update_event.action.data['AO']['apply_ao_map'] = True
                            current_ao_update_event.action.data['AO']['pos_idx'] = pos_idx
                        else:
                            current_ao_update_event = MDAEvent(**ao_optimization_event.model_dump())
                            current_ao_update_event.action.data['AO']['apply_existing'] = True
                            current_ao_update_event.action.data['AO']['pos_idx'] = 0
                        opm_events.append(current_ao_update_event)
                    
                    #----------------------------------------------------------------#
                    # Create 'xyz' optimization events 
                    if 'xyz' in o2o3_mode:
                        opm_events.append(o2o3_event)
                        
                    if 'xyz' in ao_mode:
                        # NOTE: AO-grid does not have an xyz option
                        # Run the AO at the first time-point for each position
                        if time_idx==0:
                            current_ao_event = MDAEvent(**ao_optimization_event.model_dump())
                            current_ao_event.action.data['AO']['output_path'] = ao_output_dir / Path(f'pos_{pos_idx}_ao_optimize')
                            current_ao_event.action.data['AO']['pos_idx'] = int(pos_idx)
                            current_ao_event.action.data['AO']['apply_existing'] = False
                            opm_events.append(current_ao_event)
                
                        # Update the mirror state for at each position for all time-points
                        current_ao_update_event = MDAEvent(**ao_optimization_event.model_dump())
                        current_ao_update_event.action.data['AO']['pos_idx'] = int(pos_idx)
                        current_ao_update_event.action.data['AO']['apply_existing'] = True
                        opm_events.append(current_ao_update_event)
                        
                    # Finally, handle acquiring images. 
                    # By defualt, update the daq before each scan
                    opm_events.append(daq_event)
                    
                    # Setup ASI controller for stage scanning and Camera for external START trigger
                    current_ASI_setup_event = MDAEvent(**ASI_setup_event.model_dump())
                    current_ASI_setup_event.action.data['ASI']['scan_axis_start_mm'] = float(scan_axis_start_pos_mm[scan_idx])
                    current_ASI_setup_event.action.data['ASI']['scan_axis_end_mm'] = float(scan_axis_end_pos_mm[scan_idx])
                    current_ASI_setup_event.action.data['ASI']['scan_axis_speed_mm_s'] = float(scan_axis_speed)
                    opm_events.append(current_ASI_setup_event)
                                            
                    # Create image events
                    for scan_axis_idx in range(scan_axis_positions+int(excess_starting_images) + int(excess_ending_images)):
                        for chan_idx in range(n_active_channels):
                            if (scan_axis_idx < excess_starting_images) or (scan_axis_idx > (scan_axis_positions + excess_starting_images)):
                                is_excess_image = True
                            else:
                                is_excess_image = False
                            image_event = MDAEvent(
                                index=mappingproxy(
                                    {
                                        't': time_idx, 
                                        'p': pos_idx, 
                                        'c': chan_idx, 
                                        'z': scan_axis_idx
                                    }
                                ),
                                metadata = {
                                    'DAQ' : {
                                        'mode' : 'stage',
                                        'scan_axis_step_um' : float(scan_axis_step_um),
                                        'active_channels' : channel_states,
                                        'exposure_channels_ms': channel_exposures_ms,
                                        'interleaved' : True,
                                        'laser_powers' : channel_powers,
                                        'blanking' : laser_blanking,
                                        'current_channel' : active_channel_names[chan_idx]
                                    },
                                    'Camera' : {
                                        'exposure_ms' : float(channel_exposures_ms[chan_idx]),
                                        'camera_center_x' : camera_center_x - int(camera_crop_x//2),
                                        'camera_center_y' : camera_center_y - int(camera_crop_y//2),
                                        'camera_crop_x' : int(camera_crop_x),
                                        'camera_crop_y' : int(camera_crop_y),
                                        'offset' : float(offset),
                                        'e_to_ADU': float(e_to_ADU)
                                    },
                                    'OPM' : {
                                        'angle_deg' : float(config['OPM']['angle_deg']),
                                        'camera_Zstage_orientation' : str(config['OPM']['camera_Zstage_orientation']),
                                        'camera_XYstage_orientation' : str(config['OPM']['camera_XYstage_orientation']),
                                        'camera_mirror_orientation' : str(config['OPM']['camera_mirror_orientation']),
                                        'excess_scan_positions' : int(excess_starting_images),
                                        'excess_scan_end_positions' : int(excess_ending_images), 
                                        'excess_scan_start_positions' : int(excess_starting_images)
                                    },
                                    'Stage' : {
                                        'x_pos' : stage_positions[pos_idx]['x'] + (scan_axis_idx * scan_axis_step_um),
                                        'y_pos' : stage_positions[pos_idx]['y'],
                                        'z_pos' : stage_positions[pos_idx]['z'],
                                        'excess_image': is_excess_image
                                    }
                                }
                            )
                            opm_events.append(image_event)
                            # TODO: Update processing code to account for start and end excess frames for stage scan.
                    pos_idx = pos_idx + 1
                        
    # Check if path ends if .zarr. If so, use our OutputHandler
    if len(Path(output).suffixes) == 1 and Path(output).suffix ==  '.zarr':
        indice_sizes = {
            't' : int(np.maximum(1,n_time_steps)),
            'p' : int(np.maximum(1,n_stage_positions)),
            'c' : int(np.maximum(1,n_active_channels)),
            'z' : int(np.maximum(1,scan_axis_positions+int(excess_starting_images)+int(excess_ending_images)))
        }
        handler = OPMMirrorHandler(
            path=Path(output),
            indice_sizes=indice_sizes,
            delete_existing=True
        )
        print(f'\nUsing Qi2lab handler,\nindices: {indice_sizes}\n')
    else:
        print('Using default handler')
        handler = Path(output)
            
    return opm_events, handler