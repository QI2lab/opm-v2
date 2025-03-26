
import numpy as np
from useq import MDAEvent, CustomAction, MDASequence
from types import MappingProxyType as mappingproxy
from pymmcore_plus import CMMCorePlus
from pathlib import Path
import json
from datetime import datetime
from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ
from opm_v2.handlers.opm_mirror_handler import OPMMirrorHandler
from opm_v2.engine.opm_custom_actions import (O2O3_af_event,
                                              AO_optimize_event,
                                              FP_event,
                                              DAQ_event,
                                              ASI_setup_event)
DEBUG = True
use_mda_channels = False
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
    ao_mode = config["acq_config"]["AO"]["ao_mode"]
    o2o3_mode = config["acq_config"]["O2O3-autofocus"]["o2o3_mode"]
    
    opm_events: list[MDAEvent] = []
    
    if "now" in o2o3_mode:
        o2o3_event = MDAEvent(**O2O3_af_event.model_dump())
        o2o3_event.action.data["Camera"]["exposure_ms"] = float(config["O2O3-autofocus"]["exposure_ms"])
        o2o3_event.action.data["Camera"]["camera_crop"] = [
            config["Camera"]["roi_center_x"] - int(config["Camera"]["roi_crop_x"]//2),
            config["Camera"]["roi_center_y"] - int(config["O2O3-autofocus"]["roi_crop_y"]//2),
            config["Camera"]["roi_crop_x"],
            config["O2O3-autofocus"]["roi_crop_y"]
        ]
        opm_events.append(o2o3_event)
        
    if "now" in ao_mode:
        now = datetime.now()
        timestamp = f"{now.year:4d}{now.month:2d}{now.day:2d}_{now.hour:2d}{now.minute:2d}{now.second:2d}"
        
        # setup AO using values in the config widget, NOT the MDA widget
        AO_channel_states = [False] * len(config["OPM"]["channel_ids"]) 
        AO_channel_powers = [0.] * len(config["OPM"]["channel_ids"])
        AO_active_channel_id = config["acq_config"]["AO"]["active_channel_id"]
        AO_camera_crop_y = int(config["acq_config"]["AO"]["image_mirror_range_um"]/mmc.getPixelSizeUm())
        AO_save_path = Path(str(config["acq_config"]["AO"]["save_dir_path"])) / Path(f"{timestamp}_ao_optimizeNOW")
        
        # Set the active channel in the daq channel list
        for chan_idx, chan_str in enumerate(config["OPM"]["channel_ids"]):
            if AO_active_channel_id==chan_str:
                AO_channel_states[chan_idx] = True
                AO_channel_powers[chan_idx] = config["acq_config"]["AO"]["active_channel_power"]
                
        # check to make sure there exist a laser power > 0
        if sum(AO_channel_powers)==0:
            print("All AO lasers set to 0!")
            return
        
        # Define AO optimization action data   
        ao_action_data = {
            "AO" : {
                "channel_states": AO_channel_states,
                "channel_powers" : AO_channel_powers,
                "exposure_ms": float(config["acq_config"]["AO"]["exposure_ms"]),
                "modal_delta": float(config["acq_config"]["AO"]["mode_delta"]),
                "modal_alpha":float(config["acq_config"]["AO"]["mode_alpha"]),                        
                "iterations": int(config["acq_config"]["AO"]["num_iterations"]),
                "metric": str(config["acq_config"]["AO"]["metric"]),
                "image_mirror_range_um" : config["acq_config"]["AO"]["image_mirror_range_um"],
                "blanking": bool(True),
                "apply_existing": bool(False),
                "pos_idx": None,
                "output_path":AO_save_path
            },
            "Camera" : {
                "exposure_ms": config["acq_config"]["AO"]["exposure_ms"],
                "camera_crop" : [
                    int(config["acq_config"]["camera_roi"]["center_x"] - int(config["acq_config"]["camera_roi"]["crop_x"]//2)),
                    int(config["acq_config"]["camera_roi"]["center_y"] - int(AO_camera_crop_y//2)),
                    int(config["acq_config"]["camera_roi"]["crop_x"]),
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
    AOmirror_setup = AOMirror.instance()
    OPMdaq_setup = OPMNIDAQ.instance()
    
    ao_mode = config["acq_config"]["AO"]["ao_mode"]
    o2o3_mode = config["acq_config"]["O2O3-autofocus"]["o2o3_mode"]
    fluidics_mode = config["acq_config"]["fluidics"]
    image_mirror_range_um = config["acq_config"]["mirror_scan"]["image_mirror_range_um"]

    # get pixel size
    pixel_size_um = np.round(float(mmc.getPixelSizeUm()),3) # unit: um
    
    # get the camera crop value
    camera_crop_y = int(image_mirror_range_um / pixel_size_um)
    camera_crop_x = int(config["acq_config"]["camera_roi"]["crop_x"])
    camera_center_y = int(config["acq_config"]["camera_roi"]["center_y"])
    camera_center_x = int(config["acq_config"]["camera_roi"]["center_x"])
    
    
    # get the tile overlap, used for xy and z positions.
    tile_overlap_scale = (1-2*config["OPM"]["tile_overlap_perc"]/100)
    opm_angle_scale = np.sin((np.pi/180.)*float(config["OPM"]["angle_deg"]))
    
    # try to get camera conversion factor information
    try:
        offset = mmc.getProperty(
            config["Camera"]["camera_id"],
            "CONVERSION FACTOR OFFSET"
        )
        e_to_ADU = mmc.getProperty(
            config["Camera"]["camera_id"],
            "CONVERSION FACTOR COEFF"
        )
    except Exception:
        offset = 0.
        e_to_ADU = 1.
        
    #--------------------------------------------------------------------#
    # Compile mda positions from active tabs and config.
    #--------------------------------------------------------------------#

    # Split apart sequence dictionary
    sequence_dict = json.loads(sequence.model_dump_json())
    mda_stage_positions = sequence_dict["stage_positions"]
    mda_grid_plan = sequence_dict["grid_plan"]
    mda_time_plan = sequence_dict["time_plan"]
    mda_z_plan = sequence_dict["z_plan"]
    
    if (mda_grid_plan is None) and (mda_stage_positions is None):
        print("Must select a position or grid in MDA")
        return None, None
    
    if not(use_mda_channels):
        laser_blanking = config["acq_config"]["mirror_scan"]["laser_blanking"]
        channel_states = config["acq_config"]["mirror_scan"]["channel_states"]
        channel_powers = config["acq_config"]["mirror_scan"]["channel_powers"]
        channel_exposures_ms = config["acq_config"]["mirror_scan"]["channel_exposures_ms"]
        channel_names = config["OPM"]["channel_ids"]
        
    else:
        mda_channels = sequence_dict["channels"]        
        
        if not(mda_channels):
            print("Must select channels to use in MDA widget")
            return None, None
        elif "Channel" not in mda_channels["group"]:
            print("Must select channels to use in MDA widget")
            return None, None
        
        channel_names = config["OPM"]["channel_ids"]
        channel_states = [False] * len(channel_names) 
        channel_exposures_ms = [0.] * len(channel_names) 
        channel_powers = [0.] * len(channel_names) 
        
        if mmc.getProperty("LaserBlanking", "Label")=="On":
            laser_blanking = True
        else:
            laser_blanking = False
            
        # Iterate through MDA checked channels and update active lasers list
        for mda_ch in mda_channels:
            ch_id = mda_ch["config"]
            ch_idx = config["OPM"]["channel_ids"].index(ch_id)
            
            # update active channel and powers
            channel_states[ch_idx] = True
            channel_exposures_ms[ch_idx] = mda_ch["exposure"]
            channel_powers[ch_idx] = float(
                mmc.getProperty(
                    "Coherent-Scientific Remote",
                    config["Lasers"]["laser_names"][ch_idx] + " - PowerSetpoint (%)"
                    )
                )
            
    #----------------------------------------------------------------#
    # Compile the active channel info and validate powers
    
    if sum(channel_powers)==0:
        print("All lasers set to 0!")
        return None, None
    
    n_active_channels = sum(channel_states)
    active_channel_names = [_name for _, _name in zip(channel_states, channel_names) if _]
    
    # Interleave only available if all channels have the same exposure.
    active_channel_exps = [_exp for _, _exp in zip(channel_states, channel_exposures_ms) if _]
    if len(set(active_channel_exps))==1:
        interleaved_acq = True
    else:
        interleaved_acq = False
    
    #----------------------------------------------------------------#
    # Set the daq event data for the selected opm_mode
    #----------------------------------------------------------------#
            
    daq_action_data = {
        "DAQ" : {
            "mode" : "mirror",
            "image_mirror_range_um" : float(image_mirror_range_um),
            "channel_states" : channel_states,
            "channel_powers" : channel_powers,
            "interleaved" : interleaved_acq,
            "blanking" : laser_blanking, 
        },
        "Camera" : {
            "exposure_channels" : channel_exposures_ms,
            "camera_crop" : [
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
    
    n_scan_steps = 1
    
    #----------------------------------------------------------------#
    # Create the AO event data
    #----------------------------------------------------------------#
    
    if "none" not in ao_mode:
        # Create a new directory in output.root for saving AO results
        ao_output_dir = output / Path("ao_results")
        ao_output_dir.mkdir(exist_ok=True)
        
        AO_channel_states = [False] * len(channel_names) 
        AO_channel_powers = [0.] * len(channel_names)
        AO_image_mirror_range = config["acq_config"]["AO"]["image_mirror_range_um"]
        AO_active_channel_id = config["acq_config"]["AO"]["active_channel_id"]
        AO_camera_crop_y = int(AO_image_mirror_range/pixel_size_um)
        AO_save_path = ao_output_dir
        
        # Set the active channel in the daq channel list
        for chan_idx, chan_str in enumerate(config["OPM"]["channel_ids"]):
            if AO_active_channel_id==chan_str:
                AO_channel_states[chan_idx] = True
                AO_channel_powers[chan_idx] = config["acq_config"]["AO"]["active_channel_power"]
                
        # check to make sure there exist a laser power > 0
        if sum(AO_channel_powers)==0:
            print("All AO laser powers are set to 0!")
            return None, None
        
        # Define AO optimization action data   
        ao_action_data = {
            "AO" : {
                "channel_states": AO_channel_states,
                "channel_powers" : AO_channel_powers,
                "modal_delta": float(config["acq_config"]["AO"]["mode_delta"]),
                "modal_alpha":float(config["acq_config"]["AO"]["mode_alpha"]),                        
                "iterations": int(config["acq_config"]["AO"]["num_iterations"]),
                "metric": str(config["acq_config"]["AO"]["metric"]),
                "image_mirror_range_um" : AO_image_mirror_range,
                "blanking": bool(True),
                "apply_existing": bool(False),
                "pos_idx": int(0),
                "output_path":AO_save_path
            },
            "Camera" : {
                "exposure_ms": config["acq_config"]["AO"]["exposure_ms"],
                "camera_crop" : [
                    int(camera_center_x - int(camera_crop_x//2)),
                    int(camera_center_y - int(AO_camera_crop_y//2)),
                    int(camera_crop_x),
                    int(AO_camera_crop_y)
                ]
            }
            
        }
        ao_optimization_event = MDAEvent(**AO_optimize_event.model_dump())
        ao_optimization_event.action.data.update(ao_action_data)
    
    #----------------------------------------------------------------#
    # Create the o2o3 AF event data
    #----------------------------------------------------------------#
    
    if "none" not in o2o3_mode:
        o2o3_action_data = {
            "Camera" : {                    
                "exposure_ms" : config["O2O3-autofocus"]["exposure_ms"],
                "camera_crop" : [
                    int(camera_center_x - camera_crop_x//2),
                    int(camera_center_y - config["acq_config"]["O2O3-autofocus"]["roi_crop_y"]//2),
                    int(camera_crop_x),
                    int(config["acq_config"]["O2O3-autofocus"]["roi_crop_y"])
                    ]
                }
            }
        
        o2o3_event = MDAEvent(**O2O3_af_event.model_dump())
        o2o3_event.action.data.update(o2o3_action_data)

    #----------------------------------------------------------------#
    # Create the fluidics event data
    #----------------------------------------------------------------#
    
    if "none" in fluidics_mode:
        fp_action_data = {
            "Fluidics": {
                "total_rounds": int(fluidics_mode),
                "current_round": int(0)
            }
        }
        
        fp_event = MDAEvent(**FP_event.model_dump())
        fp_event.action.data.update(fp_action_data)
    
    #----------------------------------------------------------------#
    # Compile mda positions from active tabs and config
    #----------------------------------------------------------------#

    # Get time points
    if "none" not in fluidics_mode:
        n_time_steps = int(fluidics_mode)
        time_interval = 0

    elif mda_time_plan is not None:
        n_time_steps = mda_time_plan["loops"]
        time_interval = mda_time_plan["interval"]
    
    else:
        n_time_steps = 1
        time_interval = 0
    
    # Get the z positions   
    if mda_z_plan is not None:
        max_z_pos = float(mda_z_plan["top"])
        min_z_pos = float(mda_z_plan["bottom"])
        step_z = (
            tile_overlap_scale
            * camera_crop_y 
            * opm_angle_scale 
            * pixel_size_um
        )
        if min_z_pos > max_z_pos:
            step_z = -1 * step_z
        num_z_pos = int(np.ceil(np.abs((max_z_pos - min_z_pos) / step_z)))

    else:
        min_z_pos = mmc.getZPosition()
        step_z = 0
        num_z_pos = 1
    
    # Generate xy stage positions
    stage_positions = []
    
    if mda_grid_plan is not None:
        min_y_pos = mda_grid_plan["bottom"]
        max_y_pos = mda_grid_plan["top"]
        min_x_pos = mda_grid_plan["left"]
        max_x_pos = mda_grid_plan["right"]
        num_x_pos = int(
            np.ceil(
                np.abs(max_x_pos - min_x_pos) / (
                    tile_overlap_scale
                    * camera_crop_x
                    * pixel_size_um
                )
            )
        )
        
        num_y_pos = int(
            np.ceil(
                np.abs(max_y_pos - min_y_pos) / (tile_overlap_scale*image_mirror_range_um)
            )
        )
        
        step_x = (max_x_pos - min_x_pos) / num_x_pos
        step_y = (max_y_pos - min_y_pos) / num_y_pos
        
        # Generate stage positions in a snake like pattern
        for x_pos in range(num_x_pos):
            if x_pos % 2 == 0:                          
                y_range = range(num_y_pos)
            else:  
                y_range = range(num_y_pos - 1, -1, -1)

            for y_pos in y_range:
                for z_pos in range(num_z_pos):
                    stage_positions.append(
                        {
                            "x": float(np.round(min_x_pos + x_pos * step_x, 2)),
                            "y": float(np.round(min_y_pos + y_pos * step_y, 2)),
                            "z": float(np.round(min_z_pos + z_pos * step_z, 2))
                        }
                    )
                            
    elif mda_stage_positions is not None:
        for stage_pos in mda_stage_positions:
            stage_positions.append(
                {
                    'x': float(stage_pos['x']),
                    'y': float(stage_pos['y']),
                    'z': float(stage_pos['z'])
                }
            )
            
    # update the wfc mirror positions array shape
    n_stage_pos = len(stage_positions)
    AOmirror_setup.n_positions = n_stage_pos
    
    #----------------------------------------------------------------#
    # Populate a {t, p, c, z} event sequence
    #----------------------------------------------------------------#
    
    opm_events: list[MDAEvent] = []
    
    # Flags to help ensure sequence-able events are kept together 
    need_to_setup_DAQ = True
    need_to_setup_stage = True
    
    if "start" in o2o3_mode:
        opm_events.append(o2o3_event)
        
    if "start" in ao_mode:
        opm_events.append(ao_optimization_event)
        
    #----------------------------------------------------------------#
    # setup nD mirror-based AO-OPM acquisition event structure
    
    for time_idx in range(n_time_steps):

        if not("none" in fluidics_mode) and not(time_idx==0):
            current_FP_event = MDAEvent(**fp_event.model_dump())
            current_FP_event.action.data["Fluidics"]["round"] = int(time_idx)
            opm_events.append(current_FP_event)
        
        if "time" in o2o3_mode:
            opm_events.append(o2o3_event)
            
        for pos_idx in range(n_stage_pos):
            if need_to_setup_stage:
                stage_event = MDAEvent(
                    action=CustomAction(
                        name="Stage-Move",
                        data = {
                            "Stage" : {
                                "x_pos" : stage_positions[pos_idx]["x"],
                                "y_pos" : stage_positions[pos_idx]["y"],
                                "z_pos" : stage_positions[pos_idx]["z"],
                            }
                        }
                    )
                )
                opm_events.append(stage_event)
                
                if n_stage_pos > 1:
                    need_to_setup_stage = True
                else:
                    need_to_setup_stage = False
            
            if "xyz" in o2o3_mode:
                opm_events.append(o2o3_event)
                
            # Run AO optimization before acquiring current position
            if ("xyz" in ao_mode) and (time_idx == 0):
                need_to_setup_DAQ = True
                current_AO_event = MDAEvent(**ao_optimization_event.model_dump())
                current_AO_event.action.data["AO"]["output_path"] = AO_save_path / Path(f"pos_{pos_idx}_ao_optimize") 
                current_AO_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                current_AO_event.action.data["AO"]["apply_existing"] = False
                opm_events.append(current_AO_event)
                
            # Apply mirror correction for this position if time_idx > 0
            elif ("xyz" in ao_mode) and (time_idx > 0):
                need_to_setup_DAQ = True
                current_AO_event = MDAEvent(**ao_optimization_event.model_dump())
                current_AO_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                current_AO_event.action.data["AO"]["apply_existing"] = True
                opm_events.append(current_AO_event)
                
            # Finally, handle acquiring images. 
            # These events are passed through to the normal MDAEngine and *should* be sequenced. 
            if interleaved_acq:
                if need_to_setup_DAQ:
                    need_to_setup_DAQ = True
                    opm_events.append(daq_event)
                    
                for scan_idx in range(n_scan_steps):
                    for chan_idx in range(n_active_channels):
                        image_event = MDAEvent(
                            index=mappingproxy(
                                {
                                    "t": time_idx, 
                                    "p": pos_idx, 
                                    "c": chan_idx, 
                                    "z": scan_idx
                                }
                            ),
                            metadata = {
                                "DAQ" : {
                                    "mode" : "projection",
                                    "image_mirror_range_um" : float(image_mirror_range_um),
                                    "active_channels" : channel_states,
                                    "exposure_channels_ms": channel_exposures_ms,
                                    "interleaved" : interleaved_acq,
                                    "laser_powers" : channel_powers,
                                    "blanking" : laser_blanking,
                                    "current_channel" : active_channel_names[chan_idx]
                                },
                                "Camera" : {
                                    "exposure_ms" : float(channel_exposures_ms[chan_idx]),
                                    "camera_center_x" : int(camera_center_x),
                                    "camera_center_y" : int(camera_center_y),
                                    "camera_crop_x" : int(camera_crop_x),
                                    "camera_crop_y" : int(camera_crop_y),
                                    "offset" : float(offset),
                                    "e_to_ADU": float(e_to_ADU)
                                },
                                "OPM" : {
                                    "angle_deg" : float(config["OPM"]["angle_deg"]),
                                    "camera_Zstage_orientation" : str(config["OPM"]["camera_Zstage_orientation"]),
                                    "camera_XYstage_orientation" : str(config["OPM"]["camera_XYstage_orientation"]),
                                    "camera_mirror_orientation" : str(config["OPM"]["camera_mirror_orientation"])
                                },
                                "Stage" : {
                                    "x_pos" : float(stage_positions[pos_idx]["x"]),
                                    "y_pos" : float(stage_positions[pos_idx]["y"]),
                                    "z_pos" : float(stage_positions[pos_idx]["z"]),
                                }
                            }
                        )
                        opm_events.append(image_event)
            else:
                # Mirror scan each channel separately
                for chan_idx, chan_bool in enumerate(channel_states):
                    temp_channels = [False] * len(channel_states)
                    temp_exposures = [0] * len(channel_exposures_ms)
                    if chan_bool:
                        if need_to_setup_DAQ:
                            need_to_setup_DAQ = True
                            temp_channels[chan_idx] = True
                            temp_exposures[chan_idx] = channel_exposures_ms[chan_idx]
                            
                            current_DAQ_event = MDAEvent(**daq_event.model_dump())
                            current_DAQ_event.action.data["DAQ"]["active_channels"] = temp_channels
                            current_DAQ_event.action.data["Camera"]["exposure_channels"] = temp_exposures 
                            opm_events.append(current_DAQ_event)
                            
                        for scan_idx in range(n_scan_steps):
                            image_event = MDAEvent(
                                index=mappingproxy(
                                    {
                                        "t": time_idx, 
                                        "p": pos_idx, 
                                        "c": chan_idx, 
                                        "z": scan_idx
                                    }
                                ),
                                metadata = {
                                    "DAQ" : {
                                        "mode" : "projection",
                                        "active_channels" : channel_states,
                                        "exposure_channels_ms": channel_exposures_ms,
                                        "interleaved" : interleaved_acq,
                                        "laser_powers" : channel_powers,
                                        "blanking" : laser_blanking,
                                        "current_channel" : active_channel_names[chan_idx]
                                    },
                                    "Camera" : {
                                        "exposure_ms" : float(channel_exposures_ms[chan_idx]),
                                        "camera_center_x" : int(camera_center_x),
                                        "camera_center_y" : int(camera_center_y),
                                        "camera_crop_x" : int(camera_crop_x),
                                        "camera_crop_y" : int(camera_crop_y),
                                        "offset" : float(offset),
                                        "e_to_ADU": float(e_to_ADU)
                                    },
                                    "OPM" : {
                                        "angle_deg" : float(config["OPM"]["angle_deg"]),
                                        "camera_Zstage_orientation" : str(config["OPM"]["camera_Zstage_orientation"]),
                                        "camera_XYstage_orientation" : str(config["OPM"]["camera_XYstage_orientation"]),
                                        "camera_mirror_orientation" : str(config["OPM"]["camera_mirror_orientation"])
                                    },
                                    "Stage" : {
                                        "x_pos" : float(stage_positions[pos_idx]["x"]),
                                        "y_pos" : float(stage_positions[pos_idx]["y"]),
                                        "z_pos" : float(stage_positions[pos_idx]["z"]),
                                    }
                                }
                            )
                            opm_events.append(image_event)

    # Check if path ends if .zarr. If so, use Qi2lab OutputHandler
    if len(Path(output).suffixes) == 1 and Path(output).suffix == ".zarr":
        indice_sizes = {
            't' : int(np.maximum(1,n_time_steps)),
            'p' : int(np.maximum(1,n_stage_pos)),
            'c' : int(np.maximum(1,n_active_channels)),
            'z' : int(1)
        }
        handler = OPMMirrorHandler(
            path=Path(output),
            indice_sizes=indice_sizes,
            delete_existing=True
            )
    else:
        handler = Path(output)
        
    return opm_events, handler
    
def setup_mirrorscan(
        mmc: CMMCorePlus,
        config: dict,
        sequence: MDASequence,
        output: Path,
) -> list[MDAEvent]:
    
    AOmirror_setup = AOMirror.instance()
    OPMdaq_setup = OPMNIDAQ.instance()
    
    ao_mode = config["acq_config"]["AO"]["ao_mode"]
    o2o3_mode = config["acq_config"]["O2O3-autofocus"]["o2o3_mode"]
    fluidics_mode = config["acq_config"]["fluidics"]
    image_mirror_step_um = config["acq_config"]["mirror_scan"]["image_mirror_step_size_um"]
    image_mirror_range_um = config["acq_config"]["mirror_scan"]["image_mirror_range_um"]

    # get the camera crop value
    camera_crop_y = int(config["acq_config"]["camera_roi"]["crop_y"])
    camera_crop_x = int(config["acq_config"]["camera_roi"]["crop_x"])
    camera_center_y = int(config["acq_config"]["camera_roi"]["center_y"])
    camera_center_x = int(config["acq_config"]["camera_roi"]["center_x"])
    
    # get pixel size
    pixel_size_um = np.round(float(mmc.getPixelSizeUm()),3) # unit: um
    
    # get the tile overlap, used for xy and z positions.
    tile_overlap_scale = (1-2*config["OPM"]["tile_overlap_perc"]/100)
    opm_angle_scale = np.sin((np.pi/180.)*float(config["OPM"]["angle_deg"]))
    
    # try to get camera conversion factor information
    try:
        offset = mmc.getProperty(
            config["Camera"]["camera_id"],
            "CONVERSION FACTOR OFFSET"
        )
        e_to_ADU = mmc.getProperty(
            config["Camera"]["camera_id"],
            "CONVERSION FACTOR COEFF"
        )
    except Exception:
        offset = 0.
        e_to_ADU = 1.
        
    #--------------------------------------------------------------------#
    # Compile mda positions from active tabs and config.
    #--------------------------------------------------------------------#

    # Split apart sequence dictionary
    sequence_dict = json.loads(sequence.model_dump_json())
    mda_stage_positions = sequence_dict["stage_positions"]
    mda_grid_plan = sequence_dict["grid_plan"]
    mda_time_plan = sequence_dict["time_plan"]
    mda_z_plan = sequence_dict["z_plan"]
    
    if (mda_grid_plan is None) and (mda_stage_positions is None):
        print("Must select a position or grid in MDA")
        return None, None
    
    if not(use_mda_channels):
        laser_blanking = config["acq_config"]["mirror_scan"]["laser_blanking"]
        channel_states = config["acq_config"]["mirror_scan"]["channel_states"]
        channel_powers = config["acq_config"]["mirror_scan"]["channel_powers"]
        channel_exposures_ms = config["acq_config"]["mirror_scan"]["channel_exposures_ms"]
        channel_names = config["OPM"]["channel_ids"]
        
    else:
        mda_channels = sequence_dict["channels"]        
        
        if not(mda_channels):
            print("Must select channels to use in MDA widget")
            return None, None
        elif "Channel" not in mda_channels["group"]:
            print("Must select channels to use in MDA widget")
            return None, None
        
        channel_names = config["OPM"]["channel_ids"]
        channel_states = [False] * len(channel_names) 
        channel_exposures_ms = [0.] * len(channel_names) 
        channel_powers = [0.] * len(channel_names) 
        
        if mmc.getProperty("LaserBlanking", "Label")=="On":
            laser_blanking = True
        else:
            laser_blanking = False
            
        # Iterate through MDA checked channels and update active lasers list
        for mda_ch in mda_channels:
            ch_id = mda_ch["config"]
            ch_idx = config["OPM"]["channel_ids"].index(ch_id)
            
            # update active channel and powers
            channel_states[ch_idx] = True
            channel_exposures_ms[ch_idx] = mda_ch["exposure"]
            channel_powers[ch_idx] = float(
                mmc.getProperty(
                    "Coherent-Scientific Remote",
                    config["Lasers"]["laser_names"][ch_idx] + " - PowerSetpoint (%)"
                    )
                )
            
    #----------------------------------------------------------------#
    # Compile the active channel info and validate powers
    
    if sum(channel_powers)==0:
        print("All lasers set to 0!")
        return None, None
    
    n_active_channels = sum(channel_states)
    active_channel_names = [_name for _, _name in zip(channel_states, channel_names) if _]
    
    # Interleave only available if all channels have the same exposure.
    active_channel_exps = [_exp for _, _exp in zip(channel_states, channel_exposures_ms) if _]
    if len(set(active_channel_exps))==1:
        interleaved_acq = True
    else:
        interleaved_acq = False
    
    #----------------------------------------------------------------#
    # Set the daq event data for the selected opm_mode
    #----------------------------------------------------------------#
            
    daq_action_data = {
        "DAQ" : {
            "mode" : "mirror",
            "image_mirror_step_um" : float(image_mirror_step_um),
            "image_mirror_range_um" : float(image_mirror_range_um),
            "channel_states" : channel_states,
            "channel_powers" : channel_powers,
            "interleaved" : interleaved_acq,
            "blanking" : laser_blanking, 
        },
        "Camera" : {
            "exposure_channels" : channel_exposures_ms,
            "camera_crop" : [
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
    
    # use OPMNIDAQ class calculation for number of scan steps to ensure consistency
    OPMdaq_setup.set_acquisition_params(
        scan_type="mirror",
        channel_states = [False,False,False,False,False],
        image_mirror_step_size_um=image_mirror_step_um,
        image_mirror_range_um=image_mirror_range_um,
        laser_blanking=True,
        exposure_ms=100.
    )
    n_scan_steps = OPMdaq_setup.n_scan_steps
    
    #----------------------------------------------------------------#
    # Create the AO event data
    #----------------------------------------------------------------#
    
    if "none" not in ao_mode:
        # Create a new directory in output.root for saving AO results
        ao_output_dir = output / Path("ao_results")
        ao_output_dir.mkdir(exist_ok=True)
        
        AO_channel_states = [False] * len(channel_names) 
        AO_channel_powers = [0.] * len(channel_names)
        AO_image_mirror_range = config["acq_config"]["AO"]["image_mirror_range_um"]
        AO_active_channel_id = config["acq_config"]["AO"]["active_channel_id"]
        AO_camera_crop_y = int(AO_image_mirror_range/pixel_size_um)
        AO_save_path = ao_output_dir
        
        # Set the active channel in the daq channel list
        for chan_idx, chan_str in enumerate(config["OPM"]["channel_ids"]):
            if AO_active_channel_id==chan_str:
                AO_channel_states[chan_idx] = True
                AO_channel_powers[chan_idx] = config["acq_config"]["AO"]["active_channel_power"]
                
        # check to make sure there exist a laser power > 0
        if sum(AO_channel_powers)==0:
            print("All AO laser powers are set to 0!")
            return None, None
        
        # Define AO optimization action data   
        ao_action_data = {
            "AO" : {
                "channel_states": AO_channel_states,
                "channel_powers" : AO_channel_powers,
                "modal_delta": float(config["acq_config"]["AO"]["mode_delta"]),
                "modal_alpha":float(config["acq_config"]["AO"]["mode_alpha"]),                        
                "iterations": int(config["acq_config"]["AO"]["num_iterations"]),
                "metric": str(config["acq_config"]["AO"]["metric"]),
                "image_mirror_range_um" : AO_image_mirror_range,
                "blanking": bool(True),
                "apply_existing": bool(False),
                "pos_idx": int(0),
                "output_path":AO_save_path
            },
            "Camera" : {
                "exposure_ms": config["acq_config"]["AO"]["exposure_ms"],
                "camera_crop" : [
                    int(camera_center_x - camera_crop_x//2),
                    int(camera_center_y - AO_camera_crop_y//2),
                    int(camera_crop_x),
                    int(AO_camera_crop_y)
                ]
            }
            
        }
        ao_optimization_event = MDAEvent(**AO_optimize_event.model_dump())
        ao_optimization_event.action.data.update(ao_action_data)
    
    #----------------------------------------------------------------#
    # Create the o2o3 AF event data
    #----------------------------------------------------------------#
    
    if "none" not in o2o3_mode:
        o2o3_action_data = {
            "Camera" : {                    
                "exposure_ms" : config["O2O3-autofocus"]["exposure_ms"],
                "camera_crop" : [
                    int(camera_center_x - camera_crop_x//2),
                    int(camera_center_y - config["acq_config"]["O2O3-autofocus"]["roi_crop_y"]//2),
                    int(camera_crop_x),
                    int(config["acq_config"]["O2O3-autofocus"]["roi_crop_y"])
                    ]
                }
            }
        
        o2o3_event = MDAEvent(**O2O3_af_event.model_dump())
        o2o3_event.action.data.update(o2o3_action_data)

    #----------------------------------------------------------------#
    # Create the fluidics event data
    #----------------------------------------------------------------#
    
    if not("none" in fluidics_mode):
        fp_action_data = {
            "Fluidics": {
                "total_rounds": int(fluidics_mode),
                "current_round": int(0)
            }
        }
        
        fp_event = MDAEvent(**FP_event.model_dump())
        fp_event.action.data.update(fp_action_data)
    
    #----------------------------------------------------------------#
    # Compile mda positions from active tabs and config
    #----------------------------------------------------------------#

    # Get time points
    if not("none" in fluidics_mode):
        n_time_steps = int(fluidics_mode)
        time_interval = 0

    elif mda_time_plan is not None:
        n_time_steps = mda_time_plan["loops"]
        time_interval = mda_time_plan["interval"]
    
    else:
        n_time_steps = 1
        time_interval = 0
    
    # Get the z positions   
    if mda_z_plan is not None:
        max_z_pos = float(mda_z_plan["top"])
        min_z_pos = float(mda_z_plan["bottom"])
        step_z = (
            tile_overlap_scale
            * camera_crop_y 
            * opm_angle_scale 
            * pixel_size_um
        )
        if min_z_pos > max_z_pos:
            step_z = -1 * step_z
        num_z_pos = int(np.ceil(np.abs((max_z_pos - min_z_pos) / step_z)))

    else:
        min_z_pos = mmc.getZPosition()
        step_z = 0
        num_z_pos = 1
    
    # Generate xy stage positions
    stage_positions = []
    
    if mda_grid_plan is not None:
        min_y_pos = mda_grid_plan["bottom"]
        max_y_pos = mda_grid_plan["top"]
        min_x_pos = mda_grid_plan["left"]
        max_x_pos = mda_grid_plan["right"]
        num_x_pos = int(
            np.ceil(
                np.abs(max_x_pos - min_x_pos) / (
                    tile_overlap_scale
                    * camera_crop_x
                    * pixel_size_um
                )
            )
        )
        
        num_y_pos = int(
            np.ceil(
                np.abs(max_y_pos - min_y_pos) / (tile_overlap_scale*image_mirror_range_um)
            )
        )
        
        step_x = (max_x_pos - min_x_pos) / num_x_pos
        step_y = (max_y_pos - min_y_pos) / num_y_pos
        
        # Generate stage positions in a snake like pattern
        for x_pos in range(num_x_pos):
            if x_pos % 2 == 0:                          
                y_range = range(num_y_pos)
            else:  
                y_range = range(num_y_pos - 1, -1, -1)

            for y_pos in y_range:
                for z_pos in range(num_z_pos):
                    stage_positions.append(
                        {
                            "x": float(np.round(min_x_pos + x_pos * step_x, 2)),
                            "y": float(np.round(min_y_pos + y_pos * step_y, 2)),
                            "z": float(np.round(min_z_pos + z_pos * step_z, 2))
                        }
                    )
                            
    elif mda_stage_positions is not None:
        for stage_pos in mda_stage_positions:
            stage_positions.append(
                {
                    'x': float(stage_pos['x']),
                    'y': float(stage_pos['y']),
                    'z': float(stage_pos['z'])
                }
            )
            
    # update the wfc mirror positions array shape
    n_stage_pos = len(stage_positions)
    AOmirror_setup.n_positions = n_stage_pos
    
    #----------------------------------------------------------------#
    # Populate a {t, p, c, z} event sequence
    #----------------------------------------------------------------#
    
    opm_events: list[MDAEvent] = []
    
    # Flags to help ensure sequence-able events are kept together 
    need_to_setup_DAQ = True
    need_to_setup_stage = True
    
    if "start" in o2o3_mode:
        opm_events.append(o2o3_event)
        
    if "start" in ao_mode:
        opm_events.append(ao_optimization_event)
        
    #----------------------------------------------------------------#
    # setup nD mirror-based AO-OPM acquisition event structure
    
    for time_idx in range(n_time_steps):

        if not("none" in fluidics_mode) and not(time_idx==0):
            current_FP_event = MDAEvent(**fp_event.model_dump())
            current_FP_event.action.data["Fluidics"]["round"] = int(time_idx)
            opm_events.append(current_FP_event)
        
        if "time" in o2o3_mode:
            opm_events.append(o2o3_event)
            
        for pos_idx in range(n_stage_pos):
            if need_to_setup_stage:
                stage_event = MDAEvent(
                    action=CustomAction(
                        name="Stage-Move",
                        data = {
                            "Stage" : {
                                "x_pos" : stage_positions[pos_idx]["x"],
                                "y_pos" : stage_positions[pos_idx]["y"],
                                "z_pos" : stage_positions[pos_idx]["z"],
                            }
                        }
                    )
                )
                opm_events.append(stage_event)
                
                if n_stage_pos > 1:
                    need_to_setup_stage = True
                else:
                    need_to_setup_stage = False
            
            if "xyz" in o2o3_mode:
                opm_events.append(o2o3_event)
                
            # Run AO optimization before acquiring current position
            if ("xyz" in ao_mode) and (time_idx == 0):
                need_to_setup_DAQ = True
                current_AO_event = MDAEvent(**ao_optimization_event.model_dump())
                current_AO_event.action.data["AO"]["output_path"] = AO_save_path / Path(f"pos_{pos_idx}_ao_optimize") 
                current_AO_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                current_AO_event.action.data["AO"]["apply_existing"] = False
                opm_events.append(current_AO_event)
                
            # Apply mirror correction for this position if time_idx > 0
            elif ("xyz" in ao_mode) and (time_idx > 0):
                need_to_setup_DAQ = True
                current_AO_event = MDAEvent(**ao_optimization_event.model_dump())
                current_AO_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                current_AO_event.action.data["AO"]["apply_existing"] = True
                opm_events.append(current_AO_event)
                
            # Finally, handle acquiring images. 
            # These events are passed through to the normal MDAEngine and *should* be sequenced. 
            if interleaved_acq:
                if need_to_setup_DAQ:
                    need_to_setup_DAQ = True
                    opm_events.append(daq_event)
                    
                for scan_idx in range(n_scan_steps):
                    for chan_idx in range(n_active_channels):
                        image_event = MDAEvent(
                            index=mappingproxy(
                                {
                                    "t": time_idx, 
                                    "p": pos_idx, 
                                    "c": chan_idx, 
                                    "z": scan_idx
                                }
                            ),
                            metadata = {
                                "DAQ" : {
                                    "mode" : "mirror",
                                    "image_mirror_step_um" : float(image_mirror_step_um),
                                    "image_mirror_range_um" : float(image_mirror_range_um),
                                    "active_channels" : channel_states,
                                    "exposure_channels_ms": channel_exposures_ms,
                                    "interleaved" : interleaved_acq,
                                    "laser_powers" : channel_powers,
                                    "blanking" : laser_blanking,
                                    "current_channel" : active_channel_names[chan_idx]
                                },
                                "Camera" : {
                                    "exposure_ms" : float(channel_exposures_ms[chan_idx]),
                                    "camera_center_x" : int(camera_center_x),
                                    "camera_center_y" : int(camera_center_y),
                                    "camera_crop_x" : int(camera_crop_x),
                                    "camera_crop_y" : int(camera_crop_y),
                                    "offset" : float(offset),
                                    "e_to_ADU": float(e_to_ADU)
                                },
                                "OPM" : {
                                    "angle_deg" : float(config["OPM"]["angle_deg"]),
                                    "camera_Zstage_orientation" : str(config["OPM"]["camera_Zstage_orientation"]),
                                    "camera_XYstage_orientation" : str(config["OPM"]["camera_XYstage_orientation"]),
                                    "camera_mirror_orientation" : str(config["OPM"]["camera_mirror_orientation"])
                                },
                                "Stage" : {
                                    "x_pos" : float(stage_positions[pos_idx]["x"]),
                                    "y_pos" : float(stage_positions[pos_idx]["y"]),
                                    "z_pos" : float(stage_positions[pos_idx]["z"]),
                                }
                            }
                        )
                        opm_events.append(image_event)
            else:
                # Mirror scan each channel separately
                for chan_idx, chan_bool in enumerate(channel_states):
                    temp_channels = [False] * len(channel_states)
                    temp_exposures = [0] * len(channel_exposures_ms)
                    if chan_bool:
                        if need_to_setup_DAQ:
                            need_to_setup_DAQ = True
                            temp_channels[chan_idx] = True
                            temp_exposures[chan_idx] = channel_exposures_ms[chan_idx]
                            
                            current_DAQ_event = MDAEvent(**daq_event.model_dump())
                            current_DAQ_event.action.data["DAQ"]["active_channels"] = temp_channels
                            current_DAQ_event.action.data["Camera"]["exposure_channels"] = temp_exposures 
                            opm_events.append(current_DAQ_event)
                            
                        for scan_idx in range(n_scan_steps):
                            image_event = MDAEvent(
                                index=mappingproxy(
                                    {
                                        "t": time_idx, 
                                        "p": pos_idx, 
                                        "c": chan_idx, 
                                        "z": scan_idx
                                    }
                                ),
                                metadata = {
                                    "DAQ" : {
                                        "mode" : "mirror",
                                        "image_mirror_step_um" : float(image_mirror_step_um),
                                        "image_mirror_range_um" : float(image_mirror_range_um),
                                        "active_channels" : channel_states,
                                        "exposure_channels_ms": channel_exposures_ms,
                                        "interleaved" : interleaved_acq,
                                        "laser_powers" : channel_powers,
                                        "blanking" : laser_blanking,
                                        "current_channel" : active_channel_names[chan_idx]
                                    },
                                    "Camera" : {
                                        "exposure_ms" : float(channel_exposures_ms[chan_idx]),
                                        "camera_center_x" : int(camera_center_x),
                                        "camera_center_y" : int(camera_center_y),
                                        "camera_crop_x" : int(camera_crop_x),
                                        "camera_crop_y" : int(camera_crop_y),
                                        "offset" : float(offset),
                                        "e_to_ADU": float(e_to_ADU)
                                    },
                                    "OPM" : {
                                        "angle_deg" : float(config["OPM"]["angle_deg"]),
                                        "camera_Zstage_orientation" : str(config["OPM"]["camera_Zstage_orientation"]),
                                        "camera_XYstage_orientation" : str(config["OPM"]["camera_XYstage_orientation"]),
                                        "camera_mirror_orientation" : str(config["OPM"]["camera_mirror_orientation"])
                                    },
                                    "Stage" : {
                                        "x_pos" : float(stage_positions[pos_idx]["x"]),
                                        "y_pos" : float(stage_positions[pos_idx]["y"]),
                                        "z_pos" : float(stage_positions[pos_idx]["z"]),
                                    }
                                }
                            )
                            opm_events.append(image_event)

    # Check if path ends if .zarr. If so, use Qi2lab OutputHandler
    if len(Path(output).suffixes) == 1 and Path(output).suffix == ".zarr":
        indice_sizes = {
            't' : int(np.maximum(1,n_time_steps)),
            'p' : int(np.maximum(1,n_stage_pos)),
            'c' : int(np.maximum(1,n_active_channels)),
            'z' : int(np.maximum(1,n_scan_steps))
        }
        handler = OPMMirrorHandler(
            path=Path(output),
            indice_sizes=indice_sizes,
            delete_existing=True
            )
    else:
        print("Using default handler")
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
    
    opm_mode = config["acq_config"]["opm_mode"]
    ao_mode = config["acq_config"]["AO"]["ao_mode"]
    o2o3_mode = config["acq_config"]["O2O3-autofocus"]["o2o3_mode"]
    fluidics_mode = config["acq_config"]["fluidics"]

    # get the camera crop value
    camera_crop_y = int(config["acq_config"]["camera_roi"]["crop_y"])
    camera_crop_x = int(config["acq_config"]["camera_roi"]["crop_x"])
    camera_center_y = int(config["acq_config"]["camera_roi"]["center_y"])
    camera_center_x = int(config["acq_config"]["camera_roi"]["center_x"])
    
    # set pixel size
    pixel_size_um = np.round(float(mmc.getPixelSizeUm()),3) # unit: um
    
    # get the tile overlap, used for xy and z positions.
    tile_overlap_scale = (1-2*config["OPM"]["tile_overlap_perc"]/100)
    opm_angle_scale = np.sin((np.pi/180.)*float(config["OPM"]["angle_deg"]))
    
    # try to get camera conversion factor information
    try:
        offset = mmc.getProperty(
            config["Camera"]["camera_id"],
            "CONVERSION FACTOR OFFSET"
        )
        e_to_ADU = mmc.getProperty(
            config["Camera"]["camera_id"],
            "CONVERSION FACTOR COEFF"
        )
    except Exception:
        offset = 0.
        e_to_ADU = 1.
        
    #--------------------------------------------------------------------#
    # Compile mda positions from active tabs and config
    #--------------------------------------------------------------------#

    # Split apart sequence dictionary
    sequence_dict = json.loads(sequence.model_dump_json())
    mda_stage_positions = sequence_dict["stage_positions"]
    mda_grid_plan = sequence_dict["grid_plan"]
    mda_channels = sequence_dict["channels"]        
    mda_time_plan = sequence_dict["time_plan"]
    mda_z_plan = sequence_dict["z_plan"]
    
    if not(use_mda_channels):
        laser_blanking = config["acq_config"][opm_mode+"_scan"]["laser_blanking"]
        channel_states = config["acq_config"][opm_mode+"_scan"]["channel_states"]
        channel_powers = config["acq_config"][opm_mode+"_scan"]["channel_powers"]
        channel_exposures_ms = config["acq_config"][opm_mode+"_scan"]["channel_exposures_ms"]
        channel_names = config["OPM"]["channel_ids"]
        
    else:
        
        if not(mda_channels):
            print("Must select channels to use in MDA widget")
            return None, None
        elif "Channel" not in mda_channels[0]["group"]:
            print("Must select channels to use in MDA widget")
            return None, None
        
        channel_names = config["OPM"]["channel_ids"]
        channel_states = [False] * len(channel_names) 
        channel_exposures_ms = [0.] * len(channel_names) 
        channel_powers = [0.] * len(channel_names) 
        
        if mmc.getProperty("LaserBlanking", "Label")=="On":
            laser_blanking = True
        else:
            laser_blanking = False
            
        # Iterate through MDA checked channels and update active lasers list
        for mda_ch in mda_channels:
            # Get the matching channel idx active lasers list
            ch_id = mda_ch["config"]
            ch_idx = config["OPM"]["channel_ids"].index(ch_id)
            
            # update active channel and powers
            channel_states[ch_idx] = True
            channel_exposures_ms[ch_idx] = mda_ch["exposure"]
            channel_powers[ch_idx] = float(
                mmc.getProperty(
                    "Coherent-Scientific Remote",
                    config["Lasers"]["laser_names"][ch_idx] + " - PowerSetpoint (%)"
                )
            )
            
    n_active_channels = sum(channel_states)
    active_channel_names = [_name for _, _name in zip(channel_states, channel_names) if _]
    
    # Interleave only available if all channels have the same exposure.
    active_channel_exps = [_exp for _, _exp in zip(channel_states, channel_exposures_ms) if _]
    if len(set(active_channel_exps))==1:
        interleaved_acq = True
    else:
        interleaved_acq = False
            
    if sum(channel_powers)==0:
        print("All lasers set to 0!")
        return None, None
    
    #----------------------------------------------------------------#
    # Set the daq event data for the selected opm_mode
    #----------------------------------------------------------------#
            
    daq_action_data = {
        "DAQ" : {
            "mode" : "stage",
            "channel_states" : channel_states,
            "channel_powers" : channel_powers,
            "interleaved" : interleaved_acq,
            "blanking" : laser_blanking, 
        },
        "Camera" : {
            "exposure_channels" : channel_exposures_ms,
            "camera_crop" : [
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
    #----------------------------------------------------------------#
    
    if "none" not in ao_mode:
        # Create a new directory in output.root for saving AO results
        ao_output_dir = output.parent / Path(f"{output.stem}_ao_results")
        ao_output_dir.mkdir(exist_ok=True)
        
        AO_channel_states = [False] * len(channel_names) 
        AO_channel_powers = [0.] * len(channel_names)
        AO_active_channel_id = config["acq_config"]["AO"]["active_channel_id"]
        AO_camera_crop_y = int(config["acq_config"]["AO"]["image_mirror_range_um"]/pixel_size_um)
        AO_save_path = ao_output_dir
        
        # Set the active channel in the daq channel list
        for chan_idx, chan_str in enumerate(config["OPM"]["channel_ids"]):
            if AO_active_channel_id==chan_str:
                AO_channel_states[chan_idx] = True
                AO_channel_powers[chan_idx] = config["acq_config"]["AO"]["active_channel_power"]
                
        # check to make sure there exist a laser power > 0
        if sum(AO_channel_powers)==0:
            print("All AO laser powers are set to 0!")
            return None, None
        
        # Define AO optimization action data   
        ao_action_data = {
            "AO" : {
                "channel_states": AO_channel_states,
                "channel_powers" : AO_channel_powers,
                "exposure_ms": float(config["acq_config"]["AO"]["exposure_ms"]),
                "modal_delta": float(config["acq_config"]["AO"]["mode_delta"]),
                "modal_alpha":float(config["acq_config"]["AO"]["mode_alpha"]),                        
                "iterations": int(config["acq_config"]["AO"]["num_iterations"]),
                "metric": str(config["acq_config"]["AO"]["metric"]),
                "image_mirror_range_um" : config["acq_config"]["AO"]["image_mirror_range_um"],
                "blanking": bool(True),
                "apply_existing": bool(False),
                "pos_idx": int(0),
                "output_path":AO_save_path
            },
            "Camera" : {
                "exposure_ms": config["acq_config"]["AO"]["exposure_ms"],
                "camera_crop" : [
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
    #----------------------------------------------------------------#
    
    if "none" not in o2o3_mode:
        o2o3_action_data = {
            "Camera" : {                    
                "exposure_ms" : config["O2O3-autofocus"]["exposure_ms"],
                "camera_crop" : [
                    int(camera_center_x - camera_crop_x//2),
                    int(camera_center_y - config["acq_config"]["O2O3-autofocus"]["roi_crop_y"]//2),
                    int(camera_crop_x),
                    int(config["acq_config"]["O2O3-autofocus"]["roi_crop_y"])
                    ]
                }
            }
        
        o2o3_event = MDAEvent(**O2O3_af_event.model_dump())
        o2o3_event.action.data.update(o2o3_action_data)

    #----------------------------------------------------------------#
    # Create the fluidics event data
    #----------------------------------------------------------------#
    if "none" not in fluidics_mode:
        fluidics_rounds = 11 # DPS added to get right number of timepoints for event structure
        fp_action_data = {
            "Fluidics": {
                # hardcoding 11 rounds
                "total_rounds": 11, # int(fluidics_mode),
                "current_round": int(0)
            }
        }
        
        fp_event = MDAEvent(**FP_event.model_dump())
        fp_event.action.data.update(fp_action_data)
    
    #----------------------------------------------------------------#
    # Compile mda positions from active tabs, and config
    #----------------------------------------------------------------#

    # Generate time points
    if "none" not in fluidics_mode:
        n_time_steps = int(fluidics_rounds)
        time_interval = 0
        
    elif mda_time_plan is not None:
        n_time_steps = mda_time_plan["loops"]
        time_interval = mda_time_plan["interval"]
    
    else:
        n_time_steps = 1
        time_interval = 0
    
    # Get the z positions
    if mda_z_plan is not None:
        max_z_pos = float(mda_z_plan["top"])
        min_z_pos = float(mda_z_plan["bottom"])
        # step_z = (
        #     tile_overlap_scale
        #     * camera_crop_y 
        #     * opm_angle_scale 
        #     * pixel_size_um
        # )
        # if min_z_pos > max_z_pos:
        #     step_z = -1 * step_z
        # num_z_pos = int(np.ceil(np.abs((max_z_pos - min_z_pos) / step_z)))
        # print(f"Z step: {step_z}, number of zs: {num_z_pos}")
        
        step_z = (    
            tile_overlap_scale
            * camera_crop_y 
            * opm_angle_scale
            * pixel_size_um
        )
        if min_z_pos > max_z_pos:
            step_z = -1 * step_z
        num_z_pos = int(np.ceil(np.abs((max_z_pos - min_z_pos) / step_z)))
        delta_z_um = np.round(np.abs(max_z_pos - min_z_pos),2)
        

    # If no other z positions are set, resort to the current z position 
    else:
        min_z_pos = mmc.getZPosition()
        max_z_pos = mmc.getZPosition()
        step_z = 0
        num_z_pos = 1
        delta_z_um = 0
        
   
    if DEBUG:
        print
    # Generate xy stage positions
    stage_positions = []
                        
    if mda_grid_plan is not None:
        min_y_pos = mda_grid_plan["bottom"]
        max_y_pos = mda_grid_plan["top"]
        min_x_pos = mda_grid_plan["left"]
        max_x_pos = mda_grid_plan["right"]
        range_x_um = np.round(np.abs(max_x_pos - min_x_pos),2)

        exposure_ms = np.round(active_channel_exps[0],0) # unit: ms

        # calculate slope using height positions, assuming user set with coverslip at top of camera ROI
        if delta_z_um == 0:
            coverslip_slope_um = 0
        else:
            coverslip_slope_um = np.round((delta_z_um / range_x_um),6)
        
        # TODO:
        # coverslip_slope_um = 0
        
        if DEBUG: 
            print(f'Coverslip low: {min_z_pos}')
            print(f'Coverslip high: {max_z_pos}')
            print(f'Scan start: {min_x_pos}')
            print(f'Scan end: {max_x_pos}')
            print(f'Coverslip slope: {coverslip_slope_um}')

        # maximum allowed height change
        max_z_change_um = float(config["Stage"]["max_z_change_um"])

        # calculate allowed scan length and number of scan tiles for allowed coverslip height change
        if coverslip_slope_um == 0:
            scan_tile_length_um = range_x_um
            num_scan_tiles = 1
        else:
            scan_tile_length_um = np.round((max_z_change_um / coverslip_slope_um),2)
            num_scan_tiles = np.rint(range_x_um / scan_tile_length_um).astype(int)
            
        if max_x_pos < min_x_pos:
                scan_tile_length_um = -1 * scan_tile_length_um

        # calculate scan axis tile locations
        scan_tile_overlap = .2 # unit: percentage
        scan_axis_step_um = float(config["Stage"]["stage_step_size_um"])  # unit: um 
        scan_axis_step_mm = scan_axis_step_um / 1000. #unit: mm
        scan_axis_start_mm = min_x_pos / 1000. #unit: mm
        scan_axis_end_mm = max_x_pos / 1000. #unit: mm
        scan_tile_length_mm = scan_tile_length_um / 1000. # unit: mm
        
        if num_scan_tiles > 1:
            scan_axis_start_pos_mm = np.round(np.arange(scan_axis_start_mm,scan_axis_end_mm+(1-scan_tile_overlap)*scan_tile_length_mm,(1-scan_tile_overlap)*scan_tile_length_mm),2) #unit: mm
            scan_axis_end_pos_mm = np.round(scan_axis_start_pos_mm + scan_tile_length_mm * (1+scan_tile_overlap),2)
            scan_axis_start_pos_mm = scan_axis_start_pos_mm[0:-2]
            scan_axis_end_pos_mm = scan_axis_end_pos_mm[0:-2]
            scan_axis_end_pos_mm[-1] = np.round(scan_axis_end_mm,2) # enforce boundary from grid
            scan_tile_length_w_overlap_mm = np.round(np.abs(scan_axis_end_pos_mm[0]-scan_axis_start_pos_mm[0]),2)
            scan_axis_positions = np.rint(scan_tile_length_w_overlap_mm / scan_axis_step_mm).astype(int)

        else:
            scan_axis_start_pos_mm = [scan_axis_start_mm]
            scan_axis_end_pos_mm = [scan_axis_end_mm]
            scan_tile_length_w_overlap_mm = np.round(np.abs(scan_axis_end_pos_mm[0]-scan_axis_start_pos_mm[0]),2)
            scan_axis_positions = np.rint(scan_tile_length_w_overlap_mm / scan_axis_step_mm).astype(int)
            num_scan_tiles = 1
        
        actual_exposure_s = exposure_ms / 1000. #unit: s
        scan_axis_speed = np.round(scan_axis_step_mm / actual_exposure_s / n_active_channels,5) #unit: mm/s
        if DEBUG: 
            print(f'Number scan tiles: {num_scan_tiles}')
            print(f'Scan axis start positions: {scan_axis_start_pos_mm}.')
            print(f'Scan axis end positions: {scan_axis_end_pos_mm}.')
            print(f'Scan axis positions: {scan_axis_positions}')
            print(f'Scan tile size: {scan_tile_length_w_overlap_mm}')
            print(f'Scan axis speed (mm/s): {scan_axis_speed}')

        # calculate starting height axis locations
        # z_positions = np.round(np.linspace(min_z_pos,max_z_pos,len(scan_axis_start_pos_mm)),2)
        # if delta_z_um we are tracking the coverslip and unique combinations of Z and XY tile
        if delta_z_um > 0:
            z_positions = np.round(np.linspace(min_z_pos,max_z_pos,len(scan_axis_start_pos_mm)),2)
        # else we are doing multiple z positions per XY tile
        elif num_z_pos > 1:
            z_positions = np.round(np.arange(min_z_pos, max_z_pos+step_z, step_z) , 2)
        # otherwise only one Z
        else:
            z_positions = [min_z_pos]
        if len(z_positions) > 1:
            step_z_um = float(z_positions[1]-z_positions[0])
        else:
            step_z_um = 0

        if DEBUG: 
            print(f'Height axis start positions: {z_positions}.')
            print(f"min/max y pos: {min_y_pos} / {max_y_pos}")
            
        # calculate tile axis locations
        tile_axis_overlap = 0.15 #unit: percentage
        tile_axis_ROI_um = np.round(float(camera_crop_x)*pixel_size_um,2)  #unit: um
        step_tile_axis_um = np.round((tile_axis_ROI_um) * (1-2*tile_axis_overlap),2) #unit: umnp-

        if min_y_pos==max_y_pos:
            tile_axis_positions = np.round(np.array([min_y_pos]), 2)
        else:
            tile_axis_ROI_um = np.round(float(camera_crop_x)*pixel_size_um,2)  #unit: um
            # tile_axis_positions = []
            # num_tile_positions = int(np.ceil((max_y_pos-min_y_pos)/step_tile_axis_um))
            # for tile_idx in range(num_tile_positions):
            #     tile_axis_positions.append(min_y_pos + step_tile_axis_um * tile_idx)
            # tile_axis_positions = np.asarray(tile_axis_positions)
            tile_axis_positions = np.round(np.arange(min_y_pos,max_y_pos+step_tile_axis_um,step_tile_axis_um),2)
        
        if DEBUG: 
            print(f'Tile axis step: {step_tile_axis_um}')
            print(f'Tile axis start positions: {tile_axis_positions}.')
            
        # update the wfc mirror positions array shape
        # if delta_z_um we are tracking the coverslip and have a different number of stage positions
        if delta_z_um > 0:
            n_stage_pos = num_scan_tiles*len(tile_axis_positions)
        else:
            n_stage_pos = num_scan_tiles*len(tile_axis_positions)*len(z_positions)
        num_tile_positions = len(tile_axis_positions)
        num_z_pos = len(z_positions)
        
        stage_positions = []
        # if delta_z_um we are tracking the coverslip and need a different set of stage positions
        if delta_z_um > 0:
            for scan_idx in range(num_scan_tiles):
                for tile_idx in range(num_tile_positions):
                        stage_positions.append(
                            {
                                "x": float(np.round(scan_axis_start_pos_mm[scan_idx]*1000, 2)),
                                "y": float(np.round(tile_axis_positions[tile_idx], 2)),
                                "z": float(np.round(z_positions[scan_idx], 2))
                            }
                        )
        else:
            for scan_idx in range(num_scan_tiles):
                for tile_idx in range(num_tile_positions):
                    for z_idx in range(num_z_pos):
                        stage_positions.append(
                            {
                                "x": float(np.round(min_x_pos + scan_idx * scan_tile_length_um, 2)),
                                "y": float(np.round(min_y_pos + tile_idx * step_tile_axis_um, 2)),
                                "z": float(np.round(min_z_pos + z_idx * step_z, 2))
                            }
                        )
    elif mda_stage_positions is not None:
        for stage_pos in mda_stage_positions:
            stage_positions.append(
                {
                    'x': float(stage_pos['x']),
                    'y': float(stage_pos['y']),
                    'z': float(stage_pos['z'])
                }
            )
    
    #----------------------------------------------------------------#
    # Create MDA event structure
    #----------------------------------------------------------------#

    opm_events: list[MDAEvent] = []

    if "none" not in ao_mode:
        AOmirror_setup.n_positions = n_stage_pos
        # AOmirror_setup.output_path = output
        
    # Flags to help ensure sequence-able events are kept together 
    need_to_setup_DAQ = True
    need_to_setup_stage = True
    
    # Check if run AF at start only
    if "start" in o2o3_mode:
        opm_events.append(o2o3_event)
        
    # check if run AO at start only
    if "start" in ao_mode:
        opm_events.append(ao_optimization_event)
        
    #----------------------------------------------------------------#
    # setup nD mirror-based AO-OPM acquisition event structure
    
    if DEBUG: 
            print(f'timepoints: {n_time_steps}')
            print(f'Stage positions: {n_stage_pos}.')
            print(f'Scan positions: {scan_axis_positions+int(config["Stage"]["excess_positions"])}.')
            print(f'Active channels: {n_active_channels}')
            
    for time_idx in range(n_time_steps):
        
        if "none" not in fluidics_mode and time_idx!=0:
            current_FP_event = MDAEvent(**fp_event.model_dump())
            current_FP_event.action.data["Fluidics"]["round"] = int(time_idx)
            opm_events.append(current_FP_event)
            
        if "time" in o2o3_mode:
            opm_events.append(o2o3_event)
        
        # if delta_z_um we are tracking the coverslip and need a different event structure
        if delta_z_um > 0:
            pos_idx = 0
            for scan_idx in range(num_scan_tiles):
                for tile_idx in range(num_tile_positions):
                    if need_to_setup_stage:      
                        stage_event = MDAEvent(
                            action=CustomAction(
                                name="Stage-Move",
                                data = {
                                    "Stage" : {
                                        "x_pos" : stage_positions[scan_idx]["x"],
                                        "y_pos" : stage_positions[tile_idx]["y"],
                                        "z_pos" : stage_positions[scan_idx]["z"],
                                    }
                                }
                            )
                        )
                        opm_events.append(stage_event)
                        
                        if n_stage_pos > 1:
                            need_to_setup_stage = True
                        else:
                            need_to_setup_stage = False
                
                    if "xyz" in o2o3_mode:
                        opm_events.append(o2o3_event)
                        
                    # Run AO optimization before acquiring current position
                    if ("xyz" in ao_mode) and (time_idx == 0):
                        need_to_setup_DAQ = True
                        current_AO_event = MDAEvent(**ao_optimization_event.model_dump())
                        current_AO_event.action.data["AO"]["output_path"] = ao_output_dir / Path(f"pos_{pos_idx}_ao_optimize")
                        current_AO_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                        current_AO_event.action.data["AO"]["apply_existing"] = False
                        opm_events.append(current_AO_event)
                        
                    # Apply mirror correction for this position if time_idx > 0
                    elif ("xyz" in ao_mode) and (time_idx > 0):
                        need_to_setup_DAQ = True
                        current_AO_event = MDAEvent(**ao_optimization_event.model_dump())
                        current_AO_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                        current_AO_event.action.data["AO"]["apply_existing"] = True
                        opm_events.append(current_AO_event)
                    
                    # Finally, handle acquiring images. 
                    # These events are passed through to the normal MDAEngine and *should* be sequenced. 
                    # if interleaved_acq:
                    if need_to_setup_DAQ:
                        need_to_setup_DAQ = True
                        opm_events.append(daq_event)
                    
                    # Setup ASI controller for stage scanning and Camera for external START trigger
                    current_ASI_setup_event = MDAEvent(**ASI_setup_event.model_dump())
                    current_ASI_setup_event.action.data["ASI"]["scan_axis_start_mm"] = float(scan_axis_start_pos_mm[scan_idx])
                    current_ASI_setup_event.action.data["ASI"]["scan_axis_end_mm"] = float(scan_axis_end_pos_mm[scan_idx])
                    current_ASI_setup_event.action.data["ASI"]["scan_axis_speed_mm_s"] = float(scan_axis_speed)
                    opm_events.append(current_ASI_setup_event)
                    
                    # create camera events
                    for scan_axis_idx in range(scan_axis_positions+int(config["Stage"]["excess_positions"])):
                        for chan_idx in range(n_active_channels):
                            if scan_axis_idx < int(config["Stage"]["excess_positions"]):
                                is_excess_image = True
                            else:
                                is_excess_image = False
                            image_event = MDAEvent(
                                index=mappingproxy(
                                    {
                                        "t": time_idx, 
                                        "p": pos_idx, 
                                        "c": chan_idx, 
                                        "z": scan_axis_idx
                                    }
                                ),
                                metadata = {
                                    "DAQ" : {
                                        "mode" : "stage",
                                        "scan_axis_step_um" : float(scan_axis_step_um),
                                        "active_channels" : channel_states,
                                        "exposure_channels_ms": channel_exposures_ms,
                                        "interleaved" : True,
                                        "laser_powers" : channel_powers,
                                        "blanking" : laser_blanking,
                                        "current_channel" : active_channel_names[chan_idx]
                                    },
                                    "Camera" : {
                                        "exposure_ms" : float(channel_exposures_ms[chan_idx]),
                                        "camera_center_x" : camera_center_x - int(camera_crop_x//2),
                                        "camera_center_y" : camera_center_y - int(camera_crop_y//2),
                                        "camera_crop_x" : int(camera_crop_x),
                                        "camera_crop_y" : int(camera_crop_y),
                                        "offset" : float(offset),
                                        "e_to_ADU": float(e_to_ADU)
                                    },
                                    "OPM" : {
                                        "angle_deg" : float(config["OPM"]["angle_deg"]),
                                        "camera_Zstage_orientation" : str(config["OPM"]["camera_Zstage_orientation"]),
                                        "camera_XYstage_orientation" : str(config["OPM"]["camera_XYstage_orientation"]),
                                        "camera_mirror_orientation" : str(config["OPM"]["camera_mirror_orientation"]),
                                        "excess_scan_positions" : int(config["Stage"]["excess_positions"])
                                    },
                                    "Stage" : {
                                        "x_pos" : stage_positions[scan_idx]["x"] + (scan_axis_idx * scan_axis_step_um),
                                        "y_pos" : stage_positions[tile_idx]["y"],
                                        "z_pos" : stage_positions[scan_idx]["z"],
                                        "excess_image": is_excess_image
                                    }
                                }
                            )
                            opm_events.append(image_event)
                    pos_idx = pos_idx + 1
        else:
            pos_idx = 0
            for scan_idx in range(num_scan_tiles):
                for tile_idx in range(num_tile_positions):
                    for z_idx in range(num_z_pos):
                        if need_to_setup_stage:      
                            stage_event = MDAEvent(
                                action=CustomAction(
                                    name="Stage-Move",
                                    data = {
                                        "Stage" : {
                                            "x_pos" : stage_positions[scan_idx]["x"],
                                            "y_pos" : stage_positions[tile_idx]["y"],
                                            "z_pos" : stage_positions[z_idx]["z"],
                                        }
                                    }
                                )
                            )
                            opm_events.append(stage_event)
                            
                            if n_stage_pos > 1:
                                need_to_setup_stage = True
                            else:
                                need_to_setup_stage = False
                    
                        if "xyz" in o2o3_mode:
                            opm_events.append(o2o3_event)
                            
                        # Run AO optimization before acquiring current position
                        if ("xyz" in ao_mode) and (time_idx == 0):
                            need_to_setup_DAQ = True
                            current_AO_event = MDAEvent(**ao_optimization_event.model_dump())
                            current_AO_event.action.data["AO"]["output_path"] = ao_output_dir / Path(f"pos_{pos_idx}_ao_optimize")
                            current_AO_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                            current_AO_event.action.data["AO"]["apply_existing"] = False
                            opm_events.append(current_AO_event)
                            
                        # Apply mirror correction for this position if time_idx > 0
                        elif ("xyz" in ao_mode) and (time_idx > 0):
                            need_to_setup_DAQ = True
                            current_AO_event = MDAEvent(**ao_optimization_event.model_dump())
                            current_AO_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                            current_AO_event.action.data["AO"]["apply_existing"] = True
                            opm_events.append(current_AO_event)
                        
                        # Finally, handle acquiring images. 
                        # These events are passed through to the normal MDAEngine and *should* be sequenced. 
                        # if interleaved_acq:
                        if need_to_setup_DAQ:
                            need_to_setup_DAQ = True
                            opm_events.append(daq_event)
                        
                        # Setup ASI controller for stage scanning and Camera for external START trigger
                        current_ASI_setup_event = MDAEvent(**ASI_setup_event.model_dump())
                        current_ASI_setup_event.action.data["ASI"]["scan_axis_start_mm"] = float(scan_axis_start_pos_mm[scan_idx])
                        current_ASI_setup_event.action.data["ASI"]["scan_axis_end_mm"] = float(scan_axis_end_pos_mm[scan_idx])
                        current_ASI_setup_event.action.data["ASI"]["scan_axis_speed_mm_s"] = float(scan_axis_speed)
                        opm_events.append(current_ASI_setup_event)
                        
                        # create camera events
                        for scan_axis_idx in range(scan_axis_positions+int(config["Stage"]["excess_positions"])):
                            for chan_idx in range(n_active_channels):
                                if scan_axis_idx < int(config["Stage"]["excess_positions"]):
                                    is_excess_image = True
                                else:
                                    is_excess_image = False
                                image_event = MDAEvent(
                                    index=mappingproxy(
                                        {
                                            "t": time_idx, 
                                            "p": pos_idx, 
                                            "c": chan_idx, 
                                            "z": scan_axis_idx
                                        }
                                    ),
                                    metadata = {
                                        "DAQ" : {
                                            "mode" : "stage",
                                            "scan_axis_step_um" : float(scan_axis_step_um),
                                            "active_channels" : channel_states,
                                            "exposure_channels_ms": channel_exposures_ms,
                                            "interleaved" : True,
                                            "laser_powers" : channel_powers,
                                            "blanking" : laser_blanking,
                                            "current_channel" : active_channel_names[chan_idx]
                                        },
                                        "Camera" : {
                                            "exposure_ms" : float(channel_exposures_ms[chan_idx]),
                                            "camera_center_x" : camera_center_x - int(camera_crop_x//2),
                                            "camera_center_y" : camera_center_y - int(camera_crop_y//2),
                                            "camera_crop_x" : int(camera_crop_x),
                                            "camera_crop_y" : int(camera_crop_y),
                                            "offset" : float(offset),
                                            "e_to_ADU": float(e_to_ADU)
                                        },
                                        "OPM" : {
                                            "angle_deg" : float(config["OPM"]["angle_deg"]),
                                            "camera_Zstage_orientation" : str(config["OPM"]["camera_Zstage_orientation"]),
                                            "camera_XYstage_orientation" : str(config["OPM"]["camera_XYstage_orientation"]),
                                            "camera_mirror_orientation" : str(config["OPM"]["camera_mirror_orientation"]),
                                            "excess_scan_positions" : int(config["Stage"]["excess_positions"])
                                        },
                                        "Stage" : {
                                            "x_pos" : stage_positions[scan_idx]["x"] + (scan_axis_idx * scan_axis_step_um),
                                            "y_pos" : stage_positions[tile_idx]["y"],
                                            "z_pos" : stage_positions[z_idx]["z"],
                                            "excess_image": is_excess_image
                                        }
                                    }
                                )
                                opm_events.append(image_event)
                        pos_idx = pos_idx + 1
    # Check if path ends if .zarr. If so, use our OutputHandler
    if len(Path(output).suffixes) == 1 and Path(output).suffix == ".zarr":

        indice_sizes = {
            't' : int(np.maximum(1,n_time_steps)),
            'p' : int(np.maximum(1,n_stage_pos)),
            'c' : int(np.maximum(1,n_active_channels)),
            'z' : int(np.maximum(1,scan_axis_positions+int(config["Stage"]["excess_positions"])))
        }

        handler = OPMMirrorHandler(
            path=Path(output),
            indice_sizes=indice_sizes,
            delete_existing=True
            )
            
        print(f"Using Qi2lab handler,\nindices: {indice_sizes}")
            
    else:
        print("Using default handler")
        handler = Path(output)
            
    return opm_events, handler