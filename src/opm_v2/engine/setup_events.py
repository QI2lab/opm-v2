"""
Methods for creating OPM acquisition event structures

- Optimize now (o2o3 autofocus and adaptive optics)
- Timelapse (Fast image time series, 2d or 3d with multiple image-mirror positions)
- Projection scan (2d 'sum' projection)
- Mirror scan (3d image-mirror scanning)
- Stage scan (3d stage scanning)

2025/09/05 SJS: Update to use opm_custom_events
"""
import json
from datetime import datetime
from pathlib import Path
from types import MappingProxyType as mappingproxy
from typing import Dict, List, Optional

import numpy as np
from pymmcore_plus import CMMCorePlus
from tqdm import trange
from useq import MDAEvent, MDASequence

from opm_v2.engine.opm_custom_events import (
    create_ao_grid_event,
    create_ao_mirror_update_event,
    create_ao_optimize_event,
    create_asi_scan_setup_event,
    create_daq_event,
    create_daq_move_event,
    create_fluidics_event,
    create_o2o3_autofocus_event,
    create_stage_event,
    create_timelapse_event,
)
from opm_v2.handlers.opm_mirror_handler import OPMMirrorHandler
from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ

DEBUGGING = True
MAX_IMAGE_MIRROR_RANGE_UM = 250

#---------------------------------------------------------#
# Helper methods for consistency
#---------------------------------------------------------#

def stage_positions_from_grid(
    mda_grid_plan,
    mda_z_plan,
    opm_mode: str,
    camera_crop_x: int,
    camera_crop_y: int,
    scan_range_um: float,
    scan_axis_overlap: Optional[float] = 0.2,
    tile_axis_overlap: Optional[float] = 0.2,
    z_axis_overlap: Optional[float] = 0.2,
    coverslip_max_dz: Optional[float] = None,
    coverslip_slope_x: Optional[float] = 0,
    coverslip_slope_y: Optional[float] = 0
    ) -> List[Dict]:
    """Generates stage positions list

    TODO: Break up scans based on coverslip slope in each direction.
    
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
    scan_axis_overlap : Optional[float], optional
        _description_, by default 0.2
    tile_axis_overlap : Optional[float], optional
        _description_, by default 0.2
    z_axis_overlap : Optional[float], optional
        _description_, by default 0.2
    coverslip_max_dz : Optional[float], optional
        _description_, by default None
    coverslip_slope_x : Optional[float], optional
        _description_, by default 0
    coverslip_slope_y : Optional[float], optional
        _description_, by default 0

    Returns
    -------
    List[Dict]
        List of stage positions stored as dictionaries
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
                * (1.0-z_axis_overlap)
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
                        'x': min_x_pos + ii*x_step_um,
                        'y': min_y_pos + jj*y_step_um,
                        'z': min_z_pos + kk*z_step_um+ii*dz_per_x_tile+jj*dz_per_y_tile
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


def populate_opm_metadata(
    daq_mode: str,
    mirror_step: float,
    channel_states: list,
    channel_exposures_ms: list,
    laser_powers: list,
    interleaved: bool,
    blanking: bool,
    current_channel: str,
    exposure_ms: float,
    camera_center_x: int,
    camera_center_y: int,
    camera_crop_x: int,
    camera_crop_y: int,
    offset: float,
    e_to_ADU: float,
    angle_deg: float,
    camera_Zstage_orientation: str,
    camera_XYstage_orientation: str,
    camera_mirror_orientation: str,
    stage_position: dict,
    mirror_voltage: float = None,
    image_mirror_range_um: float = None,
):
    # Assign the DAQ-image mode specific variables
    image_mirror_position = float(mirror_voltage) if mirror_voltage else None
    image_mirror_range_um = float(image_mirror_range_um) if image_mirror_range_um else None
    image_mirror_step_um = float(mirror_step) if mirror_step else None
    
    metadata = {
        'DAQ' : {
            'mode' : str(daq_mode),
            'image_mirror_position' : image_mirror_position,
            'image_mirror_range_um' : image_mirror_range_um,
            'image_mirror_step_um' : image_mirror_step_um,
            'channel_states' : channel_states,
            'exposure_channels_ms': channel_exposures_ms,
            'laser_powers' : laser_powers,
            'interleaved' : interleaved,
            'blanking' : blanking,
            'current_channel' : current_channel
        },
        'Camera' : {
            'exposure_ms' : float(exposure_ms),
            'camera_center_x' : int(camera_center_x),
            'camera_center_y' : int(camera_center_y),
            'camera_crop_x' : int(camera_crop_x),
            'camera_crop_y' : int(camera_crop_y),
            'offset' : float(offset),
            'e_to_ADU': float(e_to_ADU)
        },
        'OPM' : {
            'angle_deg' : float(angle_deg),
            'camera_Zstage_orientation' : str(camera_Zstage_orientation),
            'camera_XYstage_orientation' : str(camera_XYstage_orientation),
            'camera_mirror_orientation' : str(camera_mirror_orientation)
        },
        'Stage' : {
            'x_pos' : float(stage_position['x']),
            'y_pos' : float(stage_position['y']),
            'z_pos' : float(stage_position['z']),
        }
    }
    return metadata
#---------------------------------------------------------#
# Methods for generating OPM custom acquisitions
#---------------------------------------------------------#

def setup_optimizenow(
        mmc: CMMCorePlus,
        config: dict,
) -> list[MDAEvent]:
    """Runs either A.O. optimization or O2O3 auto focus

    Parameters
    ----------
    mmc : CMMCorePlus
        MMCor instance
    config : dict
        OPM config from disk

    Returns
    -------
    list[MDAEvent]
        OPM events
    """
    ao_mode = config['acq_config']['AO']['ao_mode']
    o2o3_mode = config['acq_config']['O2O3-autofocus']['o2o3_mode']
    
    # Sequentially run auto focus then AO optmization
    opm_events: list[MDAEvent] = []
    
    if 'now' in o2o3_mode:
        roi_center_x = config['Camera']['roi_center_x']
        roi_center_y = config['Camera']['roi_center_y']
        camera_crop_x = config['Camera']['roi_crop_x']
        camera_crop_y = config['O2O3-autofocus']['roi_crop_y']
        o2o3_event = create_o2o3_autofocus_event(
            exposure_ms=config['O2O3-autofocus']['exposure_ms'],
            camera_center=[roi_center_x, roi_center_y],
            camera_crop=[camera_crop_x, camera_crop_y]
        )
        opm_events.append(o2o3_event)
        
    if 'now' in ao_mode:
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        ao_output_path = (
            Path(str(config['acq_config']['AO']['save_dir_path'])) 
            / Path(f'{time}_ao_optimizeNOW')
        )
        ao_optimize_event = create_ao_optimize_event(config, ao_output_path)
        opm_events.append(ao_optimize_event)
    
    return opm_events, None

def setup_timelapse(
    mmc: CMMCorePlus,
    config: dict,
    sequence: MDASequence,
    output: Path,
) -> list[MDAEvent]:
    """Timelapse acquisition with optional multiple scan mirror positions.
    
    This imaging mode holds the mirror static and acquires the N timepoints,
    then moves to a new scan or stage position. Intended for acquiring
    fast static 2D images in sequence.
    
    For a static 2D timelapse, set image mirror range to 0.
    Produces an event structure where for each scan mirror position, 
    N timepoints are acquired for each active channel.
    
    NOTE: Not tested with multiple channels
    
    t / c / z / p
    
    
    Parameters
    ----------
    mmc : CMMCorePlus
        MMC core in use
    config : dict
        OPM config loaded from disk
    sequence : MDASequence
        MDA sequence to run
    output : Path
        Output path from MDA widget

    Returns
    -------
    list[MDAEvent]
        _description_
    Handler
        OPM zarr file saving handler
    """
    if DEBUGGING:
        print('\n++++ Setting up a timelapse acquisition ++++')
    
    OPMdaq_setup = OPMNIDAQ.instance()
    opm_events = [] 
    
    #--------------------------------------------------------------------#
    # Compile acquisition settings from configuration
    #--------------------------------------------------------------------#
    
    # Get the acquisition modes
    ao_mode = config['acq_config']['AO']['ao_mode']
    o2o3_mode = config['acq_config']['O2O3-autofocus']['o2o3_mode']
    
    # Get the camera crop values
    camera_crop_y = int(config['acq_config']['camera_roi']['crop_y'])
    camera_crop_x = int(config['acq_config']['camera_roi']['crop_x'])
    camera_center_y = int(config['acq_config']['camera_roi']['center_y'])
    camera_center_x = int(config['acq_config']['camera_roi']['center_x'])
    
    #----------------------------------------------------------------#
    # Get channel settings
    channel_states = config['acq_config']['timelapse']['channel_states']
    channel_powers = config['acq_config']['timelapse']['channel_powers']
    channel_exposures_ms = config['acq_config']['timelapse']['channel_exposures_ms']
    channel_names = config['OPM']['channel_ids']
    
    n_active_channels = sum(channel_states)
    
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
        raise Exception('All lasers set to 0!')
        
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
    except Exception as e:
        if DEBUGGING:
            print(f"--- Failed to get offset or e_to_adu properties: ---\n {e}")
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
        raise Exception('Must select MDA Positions AND Time plan')
    
    #----------------------------------------------------------------#
    # Create custom events
    #----------------------------------------------------------------#
           
    #----------------------------------------------------------------#
    # Create DAQ event
    daq_event = create_daq_event(
        '2d',
        channel_states=channel_states,
        channel_powers=channel_powers,
        channel_exposures_ms=channel_exposures_ms,
        camera_center=[camera_center_x, camera_center_y],
        camera_crop=[camera_crop_x, camera_crop_y],
        interleaved=interleaved_acq,
        laser_blanking=laser_blanking
    )
    
    #----------------------------------------------------------------#
    # Create the o2o3 AF event data
    if 'none' not in o2o3_mode:
        af_camera_crop_y = config['acq_config']['O2O3-autofocus']['roi_crop_y']
        o2o3_event = create_o2o3_autofocus_event(
            exposure_ms=config['O2O3-autofocus']['exposure_ms'],
            camera_center=[camera_center_x, camera_center_y],
            camera_crop=[camera_crop_x, af_camera_crop_y]
        )
                
    #----------------------------------------------------------------#
    # Create the AO custom events
    AOmirror_setup = AOMirror.instance()
    
    # Create an OA optimization event
    if 'none' not in ao_mode:
        if 'grid' in ao_mode:
            ao_mode = 'per xyz position'
            print(
                'AO Grid selected, running optimization at each XYZ position'
                )            
        ao_root_dir = output.parent / Path(f'{output.stem}_ao_results')
        ao_root_dir.mkdir(exist_ok=True)
        
    #----------------------------------------------------------------#
    # Compile tine points and positions from active MDA tabs and config
    #----------------------------------------------------------------#

    # ----------------------------------------------------------------#
    # Get time points
    n_time_steps = mda_time_plan['loops']

    # estimate the timeloop duration
    estimated_loop_duration_s = (
        (sum(active_channel_exps)/1000.)
        * n_time_steps
        + 1.0 # daq start/stop
    )

    # If the timelapse loop is longer tan 6 hours,
    # include a AO mirror update in the middle of the loop
    if estimated_loop_duration_s > 6*60*60:
        update_ao_mirror_mid_loop = True
    else:
        update_ao_mirror_mid_loop = False

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
    
    #----------------------------------------------------------------#
    # Create MDA event structure
    #----------------------------------------------------------------#

    opm_events: list[MDAEvent] = []
        
    # move stage to starting position
    opm_events.append(create_stage_event(stage_positions[0]))
    
    # ----------------------------------------------------------------#
    # Check for optimization at start
    
    if 'start' in o2o3_mode:
        opm_events.append(o2o3_event)
    if 'start' in ao_mode:
        current_ao_dir = ao_root_dir / Path('start_ao_optimize')
        current_ao_event = create_ao_optimize_event(config, current_ao_dir)
        opm_events.append(current_ao_event)
    
    #----------------------------------------------------------------#
    # setup np x 1C x nT x nZ mirror-based AO-OPM acquisition event structure
    
    if DEBUGGING: 
            print(
                'Acquisition shape values:'
                f'\n  timepoints: {n_time_steps}'
                f'\n  Num scan positions: {n_scan_steps}',
                f'\n  Stage positions: {n_stage_positions}'
                f'\n  Active channels: {n_active_channels}'
                f'\n  Estimated loop duration (s): {estimated_loop_duration_s:.2f}',
                f'\n  Update AO mid loop: {update_ao_mirror_mid_loop}'
            )
            
    for pos_idx in trange(n_stage_positions,desc='Stage positions:',leave=True):
        if pos_idx > 0:
            opm_events.append(create_stage_event(stage_positions[pos_idx]))

        # Loop over mirror scan positions
        for scan_idx in trange(n_scan_steps,desc='Mirror scan positions:',leave=False): 
                
            #------------------------------------------------------------#
            # Move the image mirror to position
            daq_move_event = create_daq_move_event(mirror_voltages[scan_idx])
            opm_events.append(daq_move_event)
            
            #------------------------------------------------------------#
            # check for optimization per position (Stage or Mirror),
            # or update the AO mirror state 
            # NOTE: Currently, the AO is running every Stage and Mirror Position
            #       Because the time for a single Mirror scan can be hours long.
            if 'xyz' in o2o3_mode:
                opm_events.append(o2o3_event)
            if 'xyz' in ao_mode:
                current_ao_dir = ao_root_dir / Path(
                    f'stage_pos_{pos_idx}_scan_pos_{scan_idx}_ao_optimize'
                )
                current_ao_event = create_ao_optimize_event(config, current_ao_dir)
                current_ao_event.action.data['AO']['pos_idx'] = int(pos_idx)
                current_ao_event.action.data['AO']['scan_idx'] = int(scan_idx)
                opm_events.append(current_ao_event)
            else:
                current_coeffs = AOmirror_setup.current_coeffs.copy()
                ao_mirror_update = create_ao_mirror_update_event(
                    mirror_coeffs=current_coeffs
                )
                opm_events.append(ao_mirror_update)
                
            #------------------------------------------------------------#
            # acquire sequenced timelapse images
            # Update daq to perform a 2d scan
            opm_events.append(daq_event) 
            for time_idx in trange(n_time_steps, desc= 'Timepoints:', leave=False):  
                # check for AO update mid loop
                if update_ao_mirror_mid_loop and time_idx==int(n_time_steps//2):
                    current_coeffs = AOmirror_setup.current_coeffs.copy()
                    ao_mirror_update = create_ao_mirror_update_event(
                        mirror_coeffs=current_coeffs
                    )
                    opm_events.append(ao_mirror_update)
                
                # create the timelapse image events
                current_chan_idx = 0
                for chan_idx, chan_bool in enumerate(channel_states):
                    if chan_bool:
                        # Create image event for current t / p / c / z
                        image_event = MDAEvent(
                            index=mappingproxy(
                                {
                                    't': time_idx, 
                                    'p': pos_idx, 
                                    'c': current_chan_idx,
                                    'z': scan_idx
                                }
                            ),
                            metadata = populate_opm_metadata(
                                daq_mode="2d",
                                mirror_voltage=mirror_voltages[scan_idx],
                                mirror_step = scan_step_um,
                                channel_states=channel_states,
                                channel_exposures_ms=channel_exposures_ms,
                                laser_powers=channel_powers,
                                interleaved=interleaved_acq,
                                blanking=laser_blanking,
                                current_channel=channel_names[chan_idx],
                                exposure_ms=channel_exposures_ms[chan_idx],
                                camera_center_x=camera_center_x,
                                camera_center_y=camera_center_y,
                                camera_crop_x=camera_crop_x,
                                camera_crop_y=camera_crop_y,
                                offset=offset,
                                e_to_ADU=e_to_ADU,
                                angle_deg=config['OPM']['angle_deg'],
                                camera_Zstage_orientation=config['OPM']['camera_Zstage_orientation'],
                                camera_XYstage_orientation=config['OPM']['camera_XYstage_orientation'],
                                camera_mirror_orientation=config['OPM']['camera_mirror_orientation'],
                                stage_position=stage_positions[pos_idx]
                            )
                        )                        
                        opm_events.append(image_event)
                        current_chan_idx += 1 

    # ----------------------------------------------------------------#
    # Setup OPM custom handler
    # NOTE: output path needs to only have a single '.', or multiple suffixes are found!
    if len(Path(output).suffixes) == 1 and Path(output).suffix ==  '.zarr':
        indice_sizes = {
            't' : int(np.maximum(1,n_time_steps)),
            'p' : int(n_stage_positions),
            'c' : int(np.maximum(1,n_active_channels)),
            'z' : int(np.maximum(1,n_scan_steps))
        }
        handler = OPMMirrorHandler(
            path=Path(output),
            indice_sizes=indice_sizes,
            delete_existing=True
        )
        print(
            'Using Qi2lab handler'
            f'\nindices: {indice_sizes}'
        )
        return opm_events, handler
    else:
        raise Exception('Defualt handler selected, modify save path!')
            
def setup_projection(
        mmc: CMMCorePlus,
        config: dict,
        sequence: MDASequence,
        output: Path,
) -> list[MDAEvent]:
    """OPM projection scan acquisition.
    
    Creates an event structure:    
    t / p / c 
    
    Parameters
    ----------
    mmc : CMMCorePlus
        MMC core in use
    config : dict
        OPM config loaded from disk
    sequence : MDASequence
        MDA sequence to run
    output : Path
        Output path from MDA widget

    Returns
    -------
    list[MDAEvent]
        _description_
    Handler
        OPM zarr file saving handler
    """
    
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
    tile_axis_overlap = float(
        config['acq_config']['projection_scan']['tile_axis_overlap']
    )
    
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
        raise Exception('All lasers set to 0!')

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
        raise Exception('Must select MDA grid or positions plan')
    
    #----------------------------------------------------------------#
    # Create custom action data
    #----------------------------------------------------------------#
            
    #----------------------------------------------------------------#
    # Create DAQ event

    daq_event = create_daq_event(
        mode='projection',
        channel_states=channel_states,
        channel_powers=channel_powers,
        channel_exposures_ms=channel_exposures_ms,
        camera_center=[camera_center_x, camera_center_y],
        camera_crop=[camera_crop_x, camera_crop_y],
        interleaved=interleaved_acq,
        laser_blanking=laser_blanking,
        image_mirror_range_um=scan_range_um
    )
    
    #----------------------------------------------------------------#
    # Create the AO event data    
    if 'none' not in ao_mode:
        # Create AO grid event
        if 'grid' in ao_mode:
            ao_output_dir = output.parent / Path(f'{output.stem}_ao_grid_results')
            ao_output_dir.mkdir(exist_ok=True)
            ao_grid_event = create_ao_grid_event(config, None)
        else:
            ao_output_dir = output.parent / Path(f'{output.stem}_ao_results')
            ao_output_dir.mkdir(exist_ok=True)
            ao_optimize_event = create_ao_optimize_event(config, None)
        
    #----------------------------------------------------------------#
    # Create the o2o3 AF event data
    if 'none' not in o2o3_mode:
        af_camera_crop_y = config['acq_config']['O2O3-autofocus']['roi_crop_y']
        o2o3_event = create_o2o3_autofocus_event(
            exposure_ms=config['O2O3-autofocus']['exposure_ms'],
            camera_center=[camera_center_x, camera_center_y],
            camera_crop=[camera_crop_x, af_camera_crop_y]
        )
        
    #----------------------------------------------------------------#
    # Create the fluidics event data
    if 'none' not in fluidics_mode:
        fluidics_rounds = int(fluidics_mode)
        
    #----------------------------------------------------------------#
    # Compile mda positions from active tabs and config
    #----------------------------------------------------------------#

    #----------------------------------------------------------------#
    # Define the time indexing
    if 'none' not in fluidics_mode:
        n_time_steps = fluidics_rounds
        time_interval = 0
    elif mda_time_plan is not None:
        n_time_steps = mda_time_plan['loops']
        time_interval = mda_time_plan['interval']
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
    
    # update AO grid event with stage positions
    if 'grid' in ao_mode:
        ao_grid_event.action.data['AO']['stage_positions'] = stage_positions
    if 'none' not in ao_mode:
        AOmirror_setup.n_positions = n_stage_positions
        
    #----------------------------------------------------------------#
    # Create MDA event structure
    #----------------------------------------------------------------#
    need_to_setup_stage = True    
    opm_events: list[MDAEvent] = []
    
    #----------------------------------------------------------------#
    # Setup Nt/ Np / Nc projection acquisition
    
    if DEBUGGING: 
        print(
            'Acquisition settings:'
            f'\n  timepoints / interval: {n_time_steps} / {time_interval}'
            f'\n  Stage positions: {n_stage_positions}'
            f'\n  Active channels: {n_active_channels}'
            f'\n  AO frequency: {ao_mode}'
            f'\n  o2o3 focus frequency: {o2o3_mode}'
        )
            
    for time_idx in trange(n_time_steps, desc= 'Timepoints:', leave=True):
        #--------------------------------------------------------------------#
        # Run fluidics starting at the second timepoint if present
        if 'none' not in fluidics_mode and time_idx!=0:
            current_fluidics_event = create_fluidics_event(fluidics_rounds, time_idx)
            opm_events.append(current_fluidics_event)
            
        #--------------------------------------------------------------------#
        # Create pause events starting at the second timepoint for timelapse acq.
        elif (mda_time_plan is not None) and (time_idx>0) and (int(time_interval)>0):
            current_timepoint_event = create_timelapse_event(
                time_interval,
                n_time_steps, 
                time_idx
            )
            opm_events.append(current_timepoint_event)

        #--------------------------------------------------------------------#
        # Create events to run before acquisition
        if time_idx == 0:
            # move stage to starting position
            initial_stage_event = create_stage_event(stage_positions[0])
            opm_events.append(initial_stage_event)
            
            if n_stage_positions==1:
                need_to_setup_stage = False
            else:
                need_to_setup_stage = True
            
            # Create 'start' optimization events
            if 'start' in o2o3_mode:
                opm_events.append(o2o3_event)
            if 'start' in ao_mode:
                if 'grid' in ao_mode:
                    ao_grid_path = ao_output_dir / Path('start_ao_grid')
                    curr_ao_grid_event = MDAEvent(**ao_grid_event.model_dump())
                    curr_ao_grid_event.action.data['AO']['output_path'] = ao_grid_path
                    opm_events.append(ao_grid_event)
                else:
                    ao_opt_path = ao_output_dir / Path('start_ao_optimize')
                    curr_ao_opt_event = MDAEvent(**ao_optimize_event.model_dump())
                    curr_ao_opt_event.action.data['AO']['output_path'] = ao_opt_path
                    opm_events.append(curr_ao_opt_event)
        
        #--------------------------------------------------------------------#
        # Create events to run each timepoint
        if 'timepoint' in o2o3_mode:
            opm_events.append(o2o3_event)          
            
        # NOTE: Projection mode does not currently sequence over time points, 
        #       To do so we need to disable the AO mirror updates, and
        #       only select 1 stage position.
        # TODO add option to set N timepoints
        if 'timepoint' in ao_mode:
            # if time_idx % 5 == 0:
            if 'grid' in ao_mode:
                current_ao_dir = ao_output_dir / Path(f'time_{time_idx}_ao_grid')
                curr_ao_grid_event = MDAEvent(**ao_grid_event.model_dump())
                curr_ao_grid_event.action.data['AO']['output_path'] = current_ao_dir
                opm_events.append(curr_ao_grid_event)
            else:
                current_ao_dir = ao_output_dir / Path(f'time_{time_idx}_ao_optimize')
                curr_ao_opt_event = MDAEvent(**ao_optimize_event.model_dump())
                curr_ao_opt_event.action.data['AO']['output_path'] = current_ao_dir
                opm_events.append(curr_ao_opt_event)
        if 'none' in ao_mode and (time_interval>0):
            ao_mirror_update_event = create_ao_mirror_update_event(
                mirror_coeffs=AOmirror_setup.current_coeffs.copy()
            )
            opm_events.append(ao_mirror_update_event)
        
        #--------------------------------------------------------------------#    
        # iterate over stage positions
        for pos_idx in range(n_stage_positions):
            # Move stage to position
            if need_to_setup_stage and pos_idx>0:      
                stage_event = create_stage_event(stage_positions[pos_idx])
                opm_events.append(stage_event)
                
            #----------------------------------------------------------------#
            # Create mirror state update events for 'start' and 'time-point' ao modes
            if ('start' in ao_mode) or ('timepoint' in ao_mode):
                if 'grid' in ao_mode:
                    current_ao_event = MDAEvent(**ao_grid_event.model_dump())
                    current_ao_event.action.data['AO']['apply_ao_map'] = True
                    current_ao_event.action.data['AO']['pos_idx'] = pos_idx
                else:
                    current_ao_event = MDAEvent(**ao_optimize_event.model_dump())
                    current_ao_event.action.data['AO']['apply_existing'] = True
                    current_ao_event.action.data['AO']['pos_idx'] = 0
                opm_events.append(current_ao_event)
                
            #----------------------------------------------------------------#
            # Create 'xyz' optimization events 
            if 'xyz' in o2o3_mode:
                opm_events.append(o2o3_event)
                
            if 'xyz' in ao_mode:
                # Run the ao optmization at the first time-point for each position
                if time_idx==0:
                    current_ao_dir = ao_output_dir / Path(f'pos_{pos_idx}_ao_results')
                    current_ao_event = MDAEvent(**ao_optimize_event.model_dump())
                    current_ao_event.action.data['AO']['output_path'] = current_ao_dir
                    current_ao_event.action.data['AO']['pos_idx'] = int(pos_idx)
                    current_ao_event.action.data['AO']['time_idx'] = int(time_idx)
                    current_ao_event.action.data['AO']['apply_existing'] = False
                    opm_events.append(current_ao_event)
                    
                # Update the mirror state for at each position for all time-points
                current_ao_event = MDAEvent(**ao_optimize_event.model_dump())
                current_ao_event.action.data['AO']['pos_idx'] = int(pos_idx)
                current_ao_event.action.data['AO']['apply_existing'] = True
                opm_events.append(current_ao_event)
             
            #----------------------------------------------------------------#
            # Handle acquiring images
            #----------------------------------------------------------------#
            
            if interleaved_acq:
                # Update daq state to sequence all channels
                opm_events.append(MDAEvent(**daq_event.model_dump()))
                
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
                        current_daq_event = MDAEvent(**daq_event.model_dump())
                        current_daq_event.action.data['DAQ']['channel_states'] = temp_channels
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
                        metadata = populate_opm_metadata(
                            daq_mode="projection",
                            image_mirror_range_um=scan_range_um,
                            channel_states=channel_states,
                            channel_exposures_ms=channel_exposures_ms,
                            laser_powers=channel_powers,
                            interleaved=interleaved_acq,
                            blanking=laser_blanking,
                            current_channel=channel_names[chan_idx],
                            exposure_ms=channel_exposures_ms[chan_idx],
                            camera_center_x=camera_center_x,
                            camera_center_y=camera_center_y,
                            camera_crop_x=camera_crop_x,
                            camera_crop_y=camera_crop_y,
                            offset=offset,
                            e_to_ADU=e_to_ADU,
                            angle_deg=config['OPM']['angle_deg'],
                            camera_Zstage_orientation=config['OPM']['camera_Zstage_orientation'],
                            camera_XYstage_orientation=config['OPM']['camera_XYstage_orientation'],
                            camera_mirror_orientation=config['OPM']['camera_mirror_orientation'],
                            stage_position=stage_positions[pos_idx]
                        )
                    )
                    opm_events.append(image_event)
                    current_chan_idx += 1

    # Check if path ends if .zarr. If so, use our OutputHandler
    # NOTE: output path needs to only have a single '.', or multiple suffixes are found!
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
        return opm_events, handler
    else:
        raise Exception('Defualt handler selected, modify save path!')

def setup_mirrorscan(
        mmc: CMMCorePlus,
        config: dict,
        sequence: MDASequence,
        output: Path,
) -> list[MDAEvent]:
    """Creates an OPM mirror scan acquisition
    
    For a static mirror acquisition set image range to 0.
    When mirror scan range == 0, produces an image sequence similar to Timelapse
    
    t / p / c / z
    
    Parameters
    ----------
    mmc : CMMCorePlus
        MMC core in use
    config : dict
        OPM config loaded from disk
    sequence : MDASequence
        MDA sequence to run
    output : Path
        Output path from MDA widget

    Returns
    -------
    list[MDAEvent]
        _description_
    Handler
        OPM zarr file saving handler
    """
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
    
    # Get the scan range, coverslip slope and overlaps
    coverslip_slope_x = config['acq_config']['mirror_scan']['coverslip_slope_x']
    coverslip_slope_y = config['acq_config']['mirror_scan']['coverslip_slope_y']
    scan_range_um = float(config['acq_config']['mirror_scan']['scan_range_um'])
    scan_step_um = float(config['acq_config']['mirror_scan']['scan_step_size_um'])
    tile_axis_overlap = float(config['acq_config']['mirror_scan']['tile_axis_overlap'])
    z_axis_overlap = float(config['acq_config']['mirror_scan']['z_axis_overlap'])
    
    # Flag for setting up a static mirror acquisition
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
        raise Exception('All lasers set to 0!')
                
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
        raise Exception('Must select MDA grid or positions plan for mirror scanning')
    
    #----------------------------------------------------------------#
    # Create custom action data
    #----------------------------------------------------------------#
           
    #----------------------------------------------------------------#
    # Create DAQ event
    daq_event = create_daq_event(
        mode=scan_mode,
        channel_states=channel_states,
        channel_powers=channel_powers,
        channel_exposures_ms=channel_exposures_ms,
        image_mirror_range_um=scan_range_um,
        image_mirror_step_um=scan_step_um,
        camera_center=[camera_center_x, camera_center_y],
        camera_crop = [camera_crop_x, camera_crop_y],
        interleaved=interleaved_acq,
        laser_blanking=laser_blanking
    )
            
    #----------------------------------------------------------------#
    # Create the AO event data    
    if 'none' not in ao_mode:
        if 'grid' in ao_mode:
            ao_output_dir = output.parent / Path(f'{output.stem}_ao_grid_results')
            ao_output_dir.mkdir(exist_ok=True)
            ao_grid_event = create_ao_grid_event(config, None)
        else:
            ao_output_dir = output.parent / Path(f'{output.stem}_ao_results')
            ao_output_dir.mkdir(exist_ok=True)
            ao_optimize_event = create_ao_optimize_event(config, None)
    #----------------------------------------------------------------#
    # Create the o2o3 AF event data
    if 'none' not in o2o3_mode:
        af_camera_crop_y = config['acq_config']['O2O3-autofocus']['roi_crop_y']
        o2o3_event = create_o2o3_autofocus_event(
            exposure_ms=config['O2O3-autofocus']['exposure_ms'],
            camera_center=[camera_center_x, camera_center_y],
            camera_crop=[camera_crop_x, af_camera_crop_y]
        )
                
    #----------------------------------------------------------------#
    # Create the fluidics event data
    if 'none' not in fluidics_mode:
        fluidics_rounds = int(fluidics_mode)
        
    #----------------------------------------------------------------#
    # Compile mda positions from active tabs and config
    #----------------------------------------------------------------#
    
    #----------------------------------------------------------------#
    # Define the time indexing
    if 'none' not in fluidics_mode:
        n_time_steps = fluidics_rounds
        time_interval = 0
    elif mda_time_plan is not None:
        n_time_steps = mda_time_plan['loops']
        time_interval = mda_time_plan['interval']
    else:
        n_time_steps = 1
        time_interval = 1
    
    if time_interval == 0:
        need_to_setup_daq = False
        if "timepoint" in ao_mode:
            ao_mode = "start"
            print(
                "AO mode is set to timepoint, but 0 interval selected, running at start"
            )
        if "timepoint" in o2o3_mode:
            o2o3_mode = "start"
            print(
                "AF mode is set to timepoint, but 0 interval selected, running at start"
            )
            
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

    # update AO grid event with stage positions
    if 'grid' in ao_mode:
        ao_grid_event.action.data['AO']['stage_positions'] = stage_positions
    if 'none' not in ao_mode:
        AOmirror_setup.n_positions = n_stage_positions
        
    #----------------------------------------------------------------#
    # Create MDA event structure
    #----------------------------------------------------------------#
    need_to_setup_stage = True
    opm_events: list[MDAEvent] = []
    
    #----------------------------------------------------------------#
    # Setup Nt / Np / Nc / Nz mirror scan acquisition
    if DEBUGGING: 
        print(
            'Acquisition settings:'
            f'\n  timepoints / interval: {n_time_steps} / {time_interval}'
            f'\n  Stage positions: {n_stage_positions}'
            f'\n  Active channels: {n_active_channels}'
            f'\n  AO frequency: {ao_mode}'
            f'\n  o2o3 focus frequency: {o2o3_mode}'
            '\nMirror scan settings:'
            f'\n  num scan steps: {n_scan_steps}'
            f'\n  scan range (um): {scan_range_um}'
            f'\n  scan step (um): {scan_step_um}'
            f'\n  DAQ scan mode: {scan_mode}'
        )
        
    for time_idx in trange(n_time_steps, desc= 'Timepoints:', leave=True):
        #--------------------------------------------------------------------#
        # Run fluidics starting at the second timepoint if present
        if 'none' not in fluidics_mode and time_idx!=0:
            current_fluidics_event = create_fluidics_event(fluidics_rounds, time_idx)
            opm_events.append(current_fluidics_event)
            
        #--------------------------------------------------------------------#
        # Create pause events starting at the second timepoint for timelapse acq.
        elif (mda_time_plan is not None) and (time_idx>0) and (int(time_interval)>0):
            current_timepoint_event = create_timelapse_event(
                time_interval, 
                n_time_steps, 
                time_idx
            )
            opm_events.append(current_timepoint_event)

        #--------------------------------------------------------------------#
        # Create events to run before acquisition
        if time_idx == 0:
            # move stage to starting position
            initial_stage_event = create_stage_event(stage_positions[0])
            opm_events.append(initial_stage_event)
            
            if n_stage_positions==1:
                need_to_setup_stage = False
            else:
                need_to_setup_stage = True
            
            # Create 'start' optimization events
            
            if 'start' in o2o3_mode:
                opm_events.append(o2o3_event)
            if 'start' in ao_mode:
                if 'grid' in ao_mode:
                    ao_grid_path = ao_output_dir / Path('start_ao_grid')
                    curr_ao_grid_event = MDAEvent(**ao_grid_event.model_dump())
                    curr_ao_grid_event.action.data['AO']['output_path'] = ao_grid_path
                    opm_events.append(ao_grid_event)
                else:
                    ao_opt_path = ao_output_dir / Path('start_ao_optimize')
                    curr_ao_opt_event = MDAEvent(**ao_optimize_event.model_dump())
                    curr_ao_opt_event.action.data['AO']['output_path'] = ao_opt_path
                    opm_events.append(curr_ao_opt_event)

        #--------------------------------------------------------------------#
        # Create events to run each timepoint
        if 'timepoint' in o2o3_mode:
            opm_events.append(o2o3_event)          
            
        if 'timepoint' in ao_mode:
            if 'grid' in ao_mode:
                current_ao_dir = ao_output_dir / Path(f'time_{time_idx}_ao_grid')
                curr_ao_grid_event = MDAEvent(**ao_grid_event.model_dump())
                curr_ao_grid_event.action.data['AO']['output_path'] = current_ao_dir
                opm_events.append(curr_ao_grid_event)
            else:
                current_ao_dir = ao_output_dir / Path(f'time_{time_idx}_ao_optimize')
                curr_ao_opt_event = MDAEvent(**ao_optimize_event.model_dump())
                curr_ao_opt_event.action.data['AO']['output_path'] = current_ao_dir
                opm_events.append(curr_ao_opt_event)
                
        if 'none' in ao_mode and (time_interval>0):
            ao_mirror_update_event = create_ao_mirror_update_event(
                mirror_coeffs=AOmirror_setup.current_coeffs.copy()
            )
            opm_events.append(ao_mirror_update_event)
                   
        #--------------------------------------------------------------------#    
        # iterate over stage positions
        for pos_idx in range(n_stage_positions):
             # Move stage to position
            if need_to_setup_stage and pos_idx!=0:      
                stage_event = create_stage_event(stage_positions[pos_idx])
                opm_events.append(stage_event)
                
            #----------------------------------------------------------------#
            # Create mirror state update events for 'start' and 'time-point' ao modes
            if ('start' in ao_mode) or ('timepoint' in ao_mode) and (time_interval>0):
                if 'grid' in ao_mode:
                    current_ao_event = MDAEvent(**ao_grid_event.model_dump())
                    current_ao_event.action.data['AO']['apply_ao_map'] = True
                    current_ao_event.action.data['AO']['pos_idx'] = pos_idx
                else:
                    current_ao_event = MDAEvent(**ao_optimize_event.model_dump())
                    current_ao_event.action.data['AO']['apply_existing'] = True
                    current_ao_event.action.data['AO']['pos_idx'] = 0
                opm_events.append(current_ao_event)

            #----------------------------------------------------------------#
            # Create 'xyz' optimization events 
            if 'xyz' in o2o3_mode:
                opm_events.append(o2o3_event)
                
            if 'xyz' in ao_mode:
                # Run the ao optimization at the first time-point for each position
                if time_idx==0:
                    current_ao_dir = ao_output_dir / Path(f'pos_{pos_idx}_ao_results')
                    current_ao_event = MDAEvent(**ao_optimize_event.model_dump())
                    current_ao_event.action.data['AO']['output_path'] = current_ao_dir
                    current_ao_event.action.data['AO']['pos_idx'] = int(pos_idx)
                    current_ao_event.action.data['AO']['apply_existing'] = False
                    opm_events.append(current_ao_event)
                    
                # Update the mirror state for at each position for all time-points
                current_ao_event = MDAEvent(**ao_optimize_event.model_dump())
                current_ao_event.action.data['AO']['pos_idx'] = int(pos_idx)
                current_ao_event.action.data['AO']['apply_existing'] = True
                opm_events.append(current_ao_event)
             
            #----------------------------------------------------------------#
            # Handle acquiring images
            #----------------------------------------------------------------#
            
            if interleaved_acq:
                if time_idx==0:
                    # Update daq state to sequence all channels
                    opm_events.append(MDAEvent(**daq_event.model_dump()))
                if need_to_setup_daq:
                    opm_events.append(MDAEvent(**daq_event.model_dump()))

                # Create image event for current t / p / c / scan idx
                for scan_idx in range(n_scan_steps):
                    current_chan_idx = 0 
                    for chan_idx, chan_bool in enumerate(channel_states):
                        if chan_bool: 
                            image_event = MDAEvent(
                                index=mappingproxy(
                                    {
                                        't': time_idx, 
                                        'p': pos_idx, 
                                        'c': current_chan_idx,
                                        'z': scan_idx
                                    }
                                ),
                                metadata = populate_opm_metadata(
                                    daq_mode=scan_mode,
                                    image_mirror_range_um=scan_range_um,
                                    mirror_step=scan_step_um,
                                    channel_states=channel_states,
                                    channel_exposures_ms=channel_exposures_ms,
                                    laser_powers=channel_powers,
                                    interleaved=interleaved_acq,
                                    blanking=laser_blanking,
                                    current_channel=channel_names[chan_idx],
                                    exposure_ms=channel_exposures_ms[chan_idx],
                                    camera_center_x=camera_center_x,
                                    camera_center_y=camera_center_y,
                                    camera_crop_x=camera_crop_x,
                                    camera_crop_y=camera_crop_y,
                                    offset=offset,
                                    e_to_ADU=e_to_ADU,
                                    angle_deg=config['OPM']['angle_deg'],
                                    camera_Zstage_orientation=config['OPM']['camera_Zstage_orientation'],
                                    camera_XYstage_orientation=config['OPM']['camera_XYstage_orientation'],
                                    camera_mirror_orientation=config['OPM']['camera_mirror_orientation'],
                                    stage_position=stage_positions[pos_idx]
                                )
                            )
                            opm_events.append(image_event)
                            current_chan_idx += 1
                            
            elif not interleaved_acq:
                # iterate of active channels then acquire scan position
                current_chan_idx = 0
                for chan_idx, chan_bool in enumerate(channel_states):
                    if chan_bool:
                        # Need a custom daq event for each channel
                        temp_channels = [False] * len(channel_states)
                        temp_exposures = [0] * len(channel_exposures_ms)
                        temp_powers = [0] * len(channel_powers)
                        temp_channels[chan_idx] = True
                        temp_exposures[chan_idx] = channel_exposures_ms[chan_idx]
                        temp_powers[chan_idx] = channel_powers[chan_idx]
                        
                        # create daq event for a single channel                    
                        current_daq_event = MDAEvent(**daq_event.model_dump())
                        current_daq_event.action.data['DAQ']['channel_states'] = temp_channels
                        current_daq_event.action.data['DAQ']['channel_powers'] = temp_powers
                        current_daq_event.action.data['Camera']['exposure_channels'] = temp_exposures
                        opm_events.append(current_daq_event)

                        # now acquire mirror scan with a single channel
                        # Create image event for current t / p / c / scan idx
                        for scan_idx in range(n_scan_steps):
                            image_event = MDAEvent(
                                index=mappingproxy(
                                    {
                                        't': time_idx, 
                                        'p': pos_idx, 
                                        'c': current_chan_idx,
                                        'z': scan_idx
                                    }
                                ),
                                metadata = populate_opm_metadata(
                                    daq_mode=scan_mode,
                                    image_mirror_range_um=scan_range_um,
                                    mirror_step=scan_step_um,
                                    channel_states=temp_channels,
                                    channel_exposures_ms=temp_exposures,
                                    laser_powers=temp_powers,
                                    interleaved=interleaved_acq,
                                    blanking=laser_blanking,
                                    current_channel=channel_names[chan_idx],
                                    exposure_ms=channel_exposures_ms[chan_idx],
                                    camera_center_x=camera_center_x,
                                    camera_center_y=camera_center_y,
                                    camera_crop_x=camera_crop_x,
                                    camera_crop_y=camera_crop_y,
                                    offset=offset,
                                    e_to_ADU=e_to_ADU,
                                    angle_deg=config['OPM']['angle_deg'],
                                    camera_Zstage_orientation=config['OPM']['camera_Zstage_orientation'],
                                    camera_XYstage_orientation=config['OPM']['camera_XYstage_orientation'],
                                    camera_mirror_orientation=config['OPM']['camera_mirror_orientation'],
                                    stage_position=stage_positions[pos_idx]
                                )
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
        return opm_events, handler
    else:
        raise Exception('Defualt handler selected, modify save path!')

def setup_stagescan(
        mmc: CMMCorePlus,
        config: dict,
        sequence: MDASequence,
        output: Path,
) -> list[MDAEvent]:
    """Setup an OPM stage scan acquisition
    
    TODO: add logic to allow for non-interleaved acquisitions
    
    t / p / c / z
    
    Parameters
    ----------
    mmc : CMMCorePlus
        MMC core in use
    config : dict
        OPM config loaded from disk
    sequence : MDASequence
        MDA sequence to run
    output : Path
        Output path from MDA widget

    Returns
    -------
    list[MDAEvent]
        _description_
    Handler
        OPM zarr file saving handler
    """
    
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
    coverslip_slope = float(config['acq_config']['stage_scan']['coverslip_slope_x'])
    scan_axis_max_range = float(
        config['acq_config']['stage_scan']['stage_scan_range_um']
    )
    coverslip_max_dz = float(config['acq_config']['stage_scan']['coverslip_max_dz'])
    
    # Get the tile overlap settings
    tile_axis_overlap = float(config['acq_config']['stage_scan']['tile_axis_overlap'])
    scan_axis_step_um = float(config['acq_config']['stage_scan']['scan_step_size_um'])
    scan_tile_overlap_um = (
        camera_crop_y * opm_angle_scale * pixel_size_um 
        + float(config['acq_config']['stage_scan']['scan_axis_overlap_um'])
    )
    scan_tile_overlap_mm = scan_tile_overlap_um/1000.
    
    # Get the excess start / end
    excess_start_images = int(
        config['acq_config']['stage_scan']['excess_start_frames']
    )
    excess_end_images = int(
        config['acq_config']['stage_scan']['excess_end_frames']
    )
    
    #----------------------------------------------------------------#
    # Get channel settings
    laser_blanking = config['acq_config'][opm_mode + '_scan']['laser_blanking']
    channel_states = config['acq_config'][opm_mode + '_scan']['channel_states']
    channel_powers = config['acq_config'][opm_mode + '_scan']['channel_powers']
    channel_exposures_ms = config['acq_config'][opm_mode+'_scan']['channel_exposures_ms']
    channel_names = config['OPM']['channel_ids']
    
    n_active_channels = sum(channel_states)
    active_channel_names = [_n for _, _n in zip(channel_states,channel_names) if _]
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
        # TODO
            
    if sum(channel_powers)==0:
        raise Exception('All lasers set to 0!')
        
    # Get the exposure, assumes equal exposures
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
        raise Exception('Must select MDA grid or positions plan for stage scanning')
    
    #----------------------------------------------------------------#
    # Create custom action data
    #----------------------------------------------------------------#
            
    #----------------------------------------------------------------#
    # Create DAQ event
    daq_event = create_daq_event(
        mode='stage',
        channel_states=channel_states,
        channel_powers=channel_powers,
        channel_exposures_ms=channel_exposures_ms,
        camera_center=[camera_center_x, camera_center_y],
        camera_crop = [camera_crop_x, camera_crop_y],
        interleaved=True,
        laser_blanking=laser_blanking
    )
    
    #----------------------------------------------------------------#
    # Create the AO event data    
    if 'none' not in ao_mode:
        if 'grid' in ao_mode:
            ao_output_dir = output.parent / Path(f'{output.stem}_ao_grid_results')
            ao_output_dir.mkdir(exist_ok=True)
            ao_grid_event = create_ao_grid_event(config, None)
        else:
            ao_output_dir = output.parent / Path(f'{output.stem}_ao_results')
            ao_output_dir.mkdir(exist_ok=True)
            ao_optimize_event = create_ao_optimize_event(config, None)
                
    #----------------------------------------------------------------#
    # Create the o2o3 AF event data
    if 'none' not in o2o3_mode:
        af_camera_crop_y = config['acq_config']['O2O3-autofocus']['roi_crop_y']
        o2o3_event = create_o2o3_autofocus_event(
            exposure_ms=config['O2O3-autofocus']['exposure_ms'],
            camera_center=[camera_center_x, camera_center_y],
            camera_crop=[camera_crop_x, af_camera_crop_y]
        )
                
    #----------------------------------------------------------------#
    # Create the fluidics event data
    if 'none' not in fluidics_mode:
        fluidics_rounds = int(fluidics_mode)
        
    #----------------------------------------------------------------#
    # Compile mda positions from active tabs, and config
    #----------------------------------------------------------------#

    #----------------------------------------------------------------#
    # Define the time indexing
    if 'none' not in fluidics_mode:
        n_time_steps = fluidics_rounds
        time_interval = 0
    elif mda_time_plan is not None:
        n_time_steps = mda_time_plan['loops']
        time_interval = mda_time_plan['interval']
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
    
    # Check if the coverslip slope determines the max scan range
    if coverslip_slope != 0:
        coverslip_max_scan_range = np.abs(coverslip_max_dz / coverslip_slope)
        if coverslip_max_scan_range < scan_axis_max_range:
            scan_axis_max_range = coverslip_max_scan_range

    # Correct directions for stage moves
    if min_z_pos > max_z_pos:
        z_axis_step_max *= -1
    if min_x_pos > max_x_pos:
        min_x_pos, max_x_pos = max_x_pos, min_x_pos        

    if DEBUGGING: 
        print(
            '\n\nXYZ Stage scan position settings:',
            f'\n  Scan start: {min_x_pos}',
            f'\n  Scan end: {max_x_pos}',
            f'\n  Tile start: {min_y_pos}',
            f'\n  Tile end: {max_y_pos}',
            f'\n  Z position min:{min_z_pos}',
            f'\n  Z position max:{max_z_pos}',
            f'\n  Coverslip slope: {coverslip_slope}',
            f'\n  Coverslip low: {cs_min_pos}',
            f'\n  Coverslip high: {cs_max_pos}',
            f'\n  Max scan range (CS used?:{coverslip_slope!=0}): {scan_axis_max_range}'
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
            (range_x_um / n_scan_positions) 
            + (n_scan_positions - 1) * (scan_tile_overlap_um / n_scan_positions),
            2
        )
    scan_axis_step_mm = scan_axis_step_um / 1000.
    scan_axis_start_mm = min_x_pos / 1000.
    scan_axis_end_mm = max_x_pos / 1000.
    scan_tile_length_mm = np.round(scan_tile_length_um / 1000.,2)

    # Initialize scan position start/end arrays with the scan start / end values
    scan_axis_start_pos_mm = np.full(n_scan_positions, scan_axis_start_mm)
    scan_axis_end_pos_mm = np.full(n_scan_positions, scan_axis_end_mm)
    for ii in range(n_scan_positions):
        scan_axis_start_pos_mm[ii] = (scan_axis_start_mm + 
                                      ii*(scan_tile_length_mm - scan_tile_overlap_mm)
        )
        scan_axis_end_pos_mm[ii] = scan_axis_start_pos_mm[ii] + scan_tile_length_mm
        
    scan_axis_start_pos_mm = np.round(scan_axis_start_pos_mm,2)
    scan_axis_end_pos_mm = np.round(scan_axis_end_pos_mm,2)
    scan_tile_length_w_overlap_mm = np.round(
        np.abs(scan_axis_end_pos_mm[0]-scan_axis_start_pos_mm[0])
        ,2
    )
    scan_axis_positions = np.rint(
        scan_tile_length_w_overlap_mm / scan_axis_step_mm
    ).astype(int)
    scan_axis_speed = np.round(scan_axis_step_mm / exposure_s / n_active_channels,5) 
    scan_tile_sizes = [
        np.round(
            np.abs(scan_axis_end_pos_mm[ii]-scan_axis_start_pos_mm[ii]),
            2
        ) for ii in range(len(scan_axis_end_pos_mm))
    ]
    n_scan_axis_indices = (
        scan_axis_positions + int(excess_start_images)+int(excess_end_images)
    )   
    # Check for scan speed actual settings
    mmc.setProperty(mmc.getXYStageDevice(), 'MotorSpeedX-S(mm/s)', scan_axis_speed)
    mmc.waitForDevice(mmc.getXYStageDevice())
    actual_speed_x = float(
        mmc.getProperty(mmc.getXYStageDevice(), 'MotorSpeedX-S(mm/s)')
    )
    actual_exposure = np.round(
        scan_axis_step_mm / actual_speed_x / n_active_channels,
        5
    ) 
    channel_exposures_ms = [actual_exposure*1000]*len(channel_exposures_ms)
    
    # update acq settings with the actual exposure and stage scan speed
    daq_event.action.data['Camera']['exposure_channels'] = channel_exposures_ms
    exposure_ms = actual_exposure*1000
    exposure_s = actual_exposure
    scan_axis_speed = actual_speed_x
    
    if DEBUGGING: 
        test_scan_length = scan_tile_length_mm==scan_tile_length_w_overlap_mm
        test_tile_sizes = np.allclose(scan_tile_sizes, scan_tile_sizes[0])
        print(
            '\nScan-axis calculated parameters:',
            f'\n  Number scan tiles: {n_scan_positions}'
            f'\n  tile length um: {scan_tile_length_um}'
            f'\n  tile overlap um: {scan_tile_overlap_um}'
            f'\n  tile length mm: {scan_tile_length_mm}'
            f'\n  tile length with overlap (mm): {scan_tile_length_w_overlap_mm}'
            f'\n  Does scan tile w/ overlap equal scan tile length?: {test_scan_length}'
            f'\n  step size (mm): {scan_axis_step_mm}'
            f'\n  exposure: {exposure_s}'
            f'\n  number of active channels: {n_active_channels}'
            f'\n  Scan axis speed (mm/s): {scan_axis_speed}\n'
            f'\n  Stage scan positions, units: mm'
            f'\n  Scan axis start positions: {scan_axis_start_pos_mm}.'
            f'\n  Scan axis end positions: {scan_axis_end_pos_mm}.'
            f'\n  Number of scan positions: {scan_axis_positions}'
            f'\n  Are all scan tiles the same size: {test_tile_sizes}'
        )

    #--------------------------------------------------------------------#
    # Generate tile axis positions
    n_tile_positions = int(np.ceil(range_y_um / tile_axis_step_max)) + 1
    tile_axis_positions = np.round(np.linspace(min_y_pos, max_y_pos,n_tile_positions),2)
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
            f'\n  Z axis positions: {z_positions}'
            f'\n  Z axis range: {range_z_um} um'
            f'\n  Z axis step: {z_axis_step_um} um'
            f'\n  Num z axis positions: {n_z_positions}'
            f'\n  Z offset per x-scan-tile: {dz_per_scan_tile} um'
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
                
    # update AO grid event with stage positions
    if 'grid' in ao_mode:
        ao_grid_event.action.data['AO']['stage_positions'] = stage_positions
    if 'none' not in ao_mode:
        AOmirror_setup.n_positions = n_stage_positions
    
    #----------------------------------------------------------------#
    # Create MDA event structure
    #----------------------------------------------------------------#
    need_to_setup_stage = True
    opm_events: list[MDAEvent] = []
            
    #----------------------------------------------------------------#
    # Setup Nt / Np / Nc / Nz mirror scan acquisition
    if DEBUGGING: 
        print(
            'Acquisition shape values:'
            f'\n  timepoints / interval: {n_time_steps} / {time_interval}'
            f'\n  Stage positions: {n_stage_positions}'
            f'\n  Scan positions: {n_scan_axis_indices}'
            f'\n  Active channels: {n_active_channels}'
            f'\n  Excess frames (S/E):{excess_start_images}/{excess_end_images}'
            f'\n  AO frequency: {ao_mode}'
            f'\n  o2o3 focus frequency: {o2o3_mode}'
        )
            
    for time_idx in trange(n_time_steps, desc= 'Timepoints:', leave=True):
        #--------------------------------------------------------------------#
        # Run fluidics starting at the second timepoint if present
        if 'none' not in fluidics_mode and time_idx!=0:
            current_fluidics_event = create_fluidics_event(fluidics_rounds, time_idx)
            opm_events.append(current_fluidics_event)
            
        #--------------------------------------------------------------------#
        # Create pause events starting at the second timepoint for timelapse acq.
        elif (mda_time_plan is not None) and (time_idx>0) and (time_interval>0):
            current_timepoint_event = create_timelapse_event(
                time_interval,
                n_time_steps, 
                time_idx
            )
            opm_events.append(current_timepoint_event)

        #--------------------------------------------------------------------#
        # Create events to run before acquisition
        if time_idx == 0:
            # move stage to starting position
            initial_stage_event = create_stage_event(stage_positions[0])
            opm_events.append(initial_stage_event)
            
            # Create 'start' optimization events
            if 'start' in o2o3_mode:
                opm_events.append(o2o3_event)
            if 'start' in ao_mode:
                if 'grid' in ao_mode:
                    ao_grid_path = ao_output_dir / Path('start_ao_grid')
                    curr_ao_grid_event = MDAEvent(**ao_grid_event.model_dump())
                    curr_ao_grid_event.action.data['AO']['output_path'] = ao_grid_path
                    opm_events.append(ao_grid_event)
                else:
                    ao_opt_path = ao_output_dir / Path('start_ao_optimize')
                    curr_ao_opt_event = MDAEvent(**ao_optimize_event.model_dump())
                    curr_ao_opt_event.action.data['AO']['output_path'] = ao_opt_path
                    opm_events.append(curr_ao_opt_event)
        
        #--------------------------------------------------------------------#
        # Create events to run each timepoint
        if 'timepoint' in o2o3_mode:
            opm_events.append(o2o3_event)          
            
        if 'timepoint' in ao_mode:
            if 'grid' in ao_mode:
                current_ao_dir = ao_output_dir / Path(f'time_{time_idx}_ao_grid')
                curr_ao_grid_event = MDAEvent(**ao_grid_event.model_dump())
                curr_ao_grid_event.action.data['AO']['output_path'] = current_ao_dir
                curr_ao_grid_event.action.data['AO']['time_idx'] = time_idx
                opm_events.append(curr_ao_grid_event)
            else:
                current_ao_dir = ao_output_dir / Path(f'time_{time_idx}_ao_optimize')
                curr_ao_opt_event = MDAEvent(**ao_optimize_event.model_dump())
                curr_ao_opt_event.action.data['AO']['output_path'] = current_ao_dir
                curr_ao_opt_event.action.data['AO']['time_idx'] = time_idx
                opm_events.append(curr_ao_opt_event)
                
        if 'none' in ao_mode and (time_interval>0):
            ao_mirror_update_event = create_ao_mirror_update_event(
                mirror_coeffs=AOmirror_setup.current_coeffs.copy()
            )
            opm_events.append(ao_mirror_update_event)     
                   
        #--------------------------------------------------------------------#    
        # iterate over stage positions
        pos_idx = 0        
        for z_idx in trange(n_z_positions, desc= 'Z-axis-tiles:', leave=False):
            for scan_idx in trange(n_scan_positions, desc= 'Scan-axis-tiles:', leave=False):
                for tile_idx in trange(n_tile_positions, desc= 'Tile-axis-tiles:', leave=False):
                    #----------------------------------------------------------------#
                    # Move stage to position
                    current_stage_event = create_stage_event(stage_positions[pos_idx])
                    opm_events.append(current_stage_event)
                        
                    #----------------------------------------------------------------#
                    # Create mirror state update events for 'start' and 'time-point' AO
                    # NOTE: Update mirror every time-point and stage-position
                    # NOTE: for single position optimization, only refer to pos_idx==0, 
                    #       Currently not filling the entire position array!
                    if ('start' in ao_mode) or ('timepoint' in ao_mode):
                        if 'grid' in ao_mode:
                            current_ao_event=MDAEvent(**ao_grid_event.model_dump())
                            current_ao_event.action.data['AO']['apply_ao_map'] = True
                            current_ao_event.action.data['AO']['pos_idx'] = pos_idx
                        else:
                            current_ao_event=MDAEvent(**ao_optimize_event.model_dump())
                            current_ao_event.action.data['AO']['apply_existing'] = True
                            current_ao_event.action.data['AO']['pos_idx'] = 0
                        opm_events.append(current_ao_event)
                        
                    #----------------------------------------------------------------#
                    # Create 'xyz' optimization events 
                    if 'xyz' in o2o3_mode:
                        opm_events.append(o2o3_event)
                        
                    if 'xyz' in ao_mode:
                        # AO optimization at the first timepoint at all positions
                        if time_idx==0:
                            current_ao_dir = ao_output_dir / Path(
                                f'pos_{pos_idx}_ao_results'
                            )
                            current_ao_event=MDAEvent(**ao_optimize_event.model_dump())
                            current_ao_event.action.data['AO']['output_path'] = current_ao_dir
                            current_ao_event.action.data['AO']['pos_idx'] = int(pos_idx)
                            current_ao_event.action.data['AO']['apply_existing'] = False
                            opm_events.append(current_ao_event)
                            
                        # Update the mirror state for at each position and timepoint
                        current_ao_event=MDAEvent(**ao_optimize_event.model_dump())
                        current_ao_event.action.data['AO']['pos_idx'] = int(pos_idx)
                        current_ao_event.action.data['AO']['apply_existing'] = True
                        opm_events.append(current_ao_event)
                        
                    #----------------------------------------------------------------#
                    # Handle acquiring images
                    #----------------------------------------------------------------#
                    # TODO: edit logic to include non-interleaved acq
                    opm_events.append(daq_event)
                    
                    # Set ASI controller for stage scanning and Camera for external Trig
                    current_asi_setup_event = create_asi_scan_setup_event(
                        start_mm=float(scan_axis_start_pos_mm[scan_idx]),
                        end_mm=float(scan_axis_end_pos_mm[scan_idx]),
                        speed_mm_s=float(scan_axis_speed)
                    )
                    opm_events.append(current_asi_setup_event)
                                            
                    # Create image events
                    for scan_axis_idx in range(n_scan_axis_indices):
                        for chan_idx in range(n_active_channels):
                            end_excess_idx = scan_axis_positions + excess_start_images
                            if scan_axis_idx < excess_start_images:
                                is_excess_image = True
                            elif scan_axis_idx > end_excess_idx:
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
                                        'channel_states' : channel_states,
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
                                        'excess_scan_positions' : int(excess_start_images),
                                        'excess_scan_end_positions' : int(excess_end_images), 
                                        'excess_scan_start_positions' : int(excess_start_images)
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
                    pos_idx = pos_idx + 1
                        
    # Check if path ends if .zarr. If so, use our OutputHandler
    if len(Path(output).suffixes) == 1 and Path(output).suffix ==  '.zarr':
        indice_sizes = {
            't' : int(np.maximum(1,n_time_steps)),
            'p' : int(np.maximum(1,n_stage_positions)),
            'c' : int(np.maximum(1,n_active_channels)),
            'z' : int(np.maximum(1,n_scan_axis_indices))
        }
        handler = OPMMirrorHandler(
            path=Path(output),
            indice_sizes=indice_sizes,
            delete_existing=True
        )
        print(f'\nUsing Qi2lab handler,\nindices: {indice_sizes}\n')
        return opm_events, handler
    else:
        raise Exception('Defualt handler selected, modify save path!')
