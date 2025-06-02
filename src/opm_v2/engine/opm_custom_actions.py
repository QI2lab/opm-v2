
from useq import MDAEvent, CustomAction

O2O3_af_event = MDAEvent(
    action=CustomAction(
        name='O2O3-autofocus',
        data = {
            'Camera' : {                    
                'exposure_ms' : None,
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

Timelapse_event = MDAEvent(
    action=CustomAction(
        name='Timelapse',
        data = {
            'plan' : {                    
                'interval' : None,
                'timepoint' : None,
                'time_steps': None
            }
        }
    )
)

# Event for running AO optimization
AO_optimize_event = MDAEvent(
    exposure = None,
    action=CustomAction(
        name='AO-optimize',
        data = {
            'AO' : {
                'opm_mode': None,
                'channel_states': None,
                'channel_powers' : None,
                'exposure_ms': None,  
                'modal_delta': None,
                'modal_alpha': None, 
                'mode': None,
                'iterations': None,
                'metric': None,
                'image_mirror_range_um' : None,
                'blanking': bool(True),
                'apply_existing': bool(False),
                'pos_idx':int(0),
                'output_path':None
            },
            'Camera' : {
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

# Event for running AO grid generation
AO_grid_event = MDAEvent(
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

# Create Fluidics program event
FP_event = MDAEvent(
    # exposure = AO_exposure_ms,
    action=CustomAction(
        name='Fluidics',
        data = {
            'Fluidics' : {
                'mode' : None,
                'round' : None
            }
            }
    )
)

# create DAQ hardware setup event
DAQ_event = MDAEvent(
    action=CustomAction(
        name='DAQ',
        data = {
            'DAQ' : {
                'mode' : None,
                'image_mirror_step_um' : None,
                'image_mirror_range_um' : None,
                'channel_states' : None,
                'channel_powers' : None,
                'interleaved' : None,
                'active_channels': None,
                'blanking' : None, 
            },
            'Camera' : {
                'exposure_channels' : None,
                'camera_crop' : [
                    None,
                    None,
                    None,
                    None,
                ]
            }
        }
    )
)

# create ASI hardware setup event for stage scanning
ASI_setup_event = MDAEvent(
    action=CustomAction(
        name='ASI-setupscan',
        data = {
            'ASI' : {
                'mode' : 'scan',
                'scan_axis_start_mm' : None,
                'scan_axis_end_mm' : None,
                'scan_axis_speed_mm_s' : None
            }
        }
    )
)
