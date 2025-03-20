
import numpy as np
from useq import MDAEvent, CustomAction, MDASequence
from types import MappingProxyType as mappingproxy
from pymmcore_plus import CMMCorePlus
from pathlib import Path
import json
from opm_v2.hardware.AOMirror import AOMirror
from datetime import datetime
from opm_v2.handlers.opm_mirror_handler import OPMMirrorHandler

DEBUG = True

def setup_stagescan(
        mmc: CMMCorePlus,
        config: dict,
        sequence: MDASequence,
        output: Path,
        FP_mode: str = "None",
        FP_num_rounds: int = 1,
) -> list[MDAEvent]:
    """Parse GUI settings and setup event structure for stage scan + AO."""

    AOmirror_stage = AOMirror.instance()

    # get AO mode / interval
    if "System-correction" in mmc.getProperty("AO-mode", "Label"):
        AO_mode = "System-correction"
    elif "Before-each-XYZ" in mmc.getProperty("AO-mode", "Label"):
        AO_mode = "Before-each-xyz"
    elif "Before-every-acq" in mmc.getProperty("AO-mode", "Label"):
        AO_mode = "Before-every-acq"

    # get O2-O3 focus mode / interval
    if "Initial-only" in mmc.getProperty("O2O3focus-mode", "Label"):
        O2O3_mode = "Initial-only"
    elif "Before-each-XYZ" in mmc.getProperty("O2O3focus-mode", "Label"):
        O2O3_mode = "Before-each-xyz"
    elif "Before-each-t" in mmc.getProperty("O2O3focus-mode", "Label"):
        O2O3_mode = "Before-each-t"
    elif "After-30min" in mmc.getProperty("O2O3focus-mode", "Label"):
        O2O3_mode = "After-30min"

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

    # get the camera crop value
    camera_crop_y = int(mmc.getProperty("ImageCameraCrop","Label"))
    camera_crop_x = int(config["Camera"]["camera_crop_x"])
        
    #--------------------------------------------------------------------#
    #--------------------------------------------------------------------#
    # Compile mda positions from active tabs, extract the relevant portions for qi2lab-OPM
    # -  if grid plan is none, try on the stage positions sequence.
    # -  if running with fluidics, set the number of timepoints as the number of rounds.
    # -  if no time plan is selected, assume single timepoint.
    #--------------------------------------------------------------------#
    #--------------------------------------------------------------------#

    # Split apart sequence dictionary
    sequence_dict = json.loads(sequence.model_dump_json())
    mda_stage_positions = sequence_dict["stage_positions"]
    mda_grid_plan = sequence_dict["grid_plan"]
    mda_channels = sequence_dict["channels"]        
    mda_time_plan = sequence_dict["time_plan"]
    mda_z_plan = sequence_dict["z_plan"]
                
    #--------------------------------------------------------------------#
    # Generate time points
    #--------------------------------------------------------------------#
        
    # If running a fluidics experiment, the number of time points are the number of rounds.
    if FP_mode != "None": 
        n_time_steps = FP_num_rounds
        time_interval = 0
        
    # If the time plan is selected, generate a timelapse 
    elif mda_time_plan is not None:
        n_time_steps = mda_time_plan["loops"]
        time_interval = mda_time_plan["interval"]
    
    # If no time plan is given, assume a single time point
    else:
        n_time_steps = 1
        time_interval = 0  # noqa: F841

    #--------------------------------------------------------------------#
    # Generate channel selection, not used in optimize now
    #--------------------------------------------------------------------#
    
    # Create daq channel lists
    channel_names = config["OPM"]["channel_ids"]
    active_channels = [False] * len(channel_names) 
    exposure_channels = [0.] * len(channel_names) 
    laser_powers = [0.] * len(channel_names) 
    
    # Validate mda channels is set to channels and not running optimize now.
    if not(mda_channels):
        print("Must select channels to use in MDA widget")
        return
    elif "Channel" not in mda_channels[0]["group"]:
        print("Must select channels to use in MDA widget")
        return

    if mda_channels is not None:
        # Iterate through MDA checked channels and update active lasers list
        for mda_ch in mda_channels:
            # Get the matching channel idx active lasers list
            ch_id = mda_ch["config"]
            ch_idx = config["OPM"]["channel_ids"].index(ch_id)
            
            # update active channel and powers
            active_channels[ch_idx] = True
            exposure_channels[ch_idx] = mda_ch["exposure"]
            laser_powers[ch_idx] = float(
                mmc.getProperty(
                    "Coherent-Scientific Remote",
                    config["Lasers"]["laser_names"][ch_idx] + " - PowerSetpoint (%)"
                    )
                )

        # if mda is selected, check to make sure there exist a laser power > 0
        if sum(laser_powers)==0:
            print("All lasers set to 0!")
            return

    n_active_channels = sum(active_channels)
    
    # Create lists containing only the active channel names and exposures
    active_channel_names = [_name for _, _name in zip(active_channels, channel_names) if _]
    active_channel_exps = [_exp for _, _exp in zip(active_channels, exposure_channels) if _]  # noqa: F841

    # Force interleaved acquisition for stage scanning
    interleaved_acq = True

    # get laser blanking values
    if "On" in mmc.getProperty("LaserBlanking","Label"):
        laser_blanking = True
    elif "Off" in mmc.getProperty("LaserBlanking","Label"):
        laser_blanking = False
    else:
        laser_blanking = True
        

    #--------------------------------------------------------------------#
    # Generate z stage positions
    #--------------------------------------------------------------------#

    # Check if multiple z plans are setup in MDA
    if mda_z_plan is not None:
        min_z_pos = float(mda_z_plan["top"])
        max_z_pos = float(mda_z_plan["bottom"])
        step_z = 0.6 * float(mmc.getProperty("ImageCameraCrop","Label") ) * np.sin((np.pi/180.)*float(config["OPM"]["angle_deg"])) * mmc.getPixelSizeUm()
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
            
    #--------------------------------------------------------------------#
    # Generate xy stage positions
    #--------------------------------------------------------------------#
    
    stage_positions = []
    
    # If only the stage positions tab is selected, use the MDA positions tab
    if (mda_stage_positions is not None) and (mda_grid_plan is None):
        for stage_pos in mda_stage_positions:
            stage_positions.append({
                        'x': float(stage_pos['x']),
                        'y': float(stage_pos['y']),
                        'z': float(stage_pos['z'])
                    })
                    
    # If the grid tab is selected, generate positions from the grid plan
    elif mda_grid_plan is not None:
        
        #----------------------------------------------------------------# 
        # Generate stage position list using the grid's generated positions
        #----------------------------------------------------------------#
        
        # iterate through sequence to grab stage positions from grid plan
        stage_positions_array = []
        for event in sequence:
            json_event = json.loads(event.model_dump_json())
            stage_positions_array.append([
                float(json_event['z_pos']),
                float(json_event['y_pos']),
                float(json_event['x_pos'])]
            )
        print(mda_grid_plan)
        
        stage_positions_array = np.asarray(stage_positions_array, dtype=np.float32)
        # Define stage positions
        
        min_y_pos = mda_grid_plan["bottom"]
        max_y_pos = mda_grid_plan["top"]
        min_x_pos = mda_grid_plan["left"]
        max_x_pos = mda_grid_plan["right"]

        # min_y_pos = np.min(stage_positions_array[:,1])
        # max_y_pos = np.max(stage_positions_array[:,1])
        # min_x_pos = np.min(stage_positions_array[:,2])
        # max_x_pos = np.max(stage_positions_array[:,2])

        range_x_um = np.round(np.abs(max_x_pos - min_x_pos),2)

        # set pixel size
        pixel_size_um = np.round(float(mmc.getPixelSizeUm()),2) # unit: um
        
        # get exposure time from main window
        exposure_ms = np.round(active_channel_exps[0],0) # unit: ms

        # calculate slope using height positions, assuming user set with coverslip at top of camera ROI
        if delta_z_um == 0:
            coverslip_slope_um = 0
        else:
            coverslip_slope_um = np.round((delta_z_um / range_x_um),6)
        
        if DEBUG: 
            print(f'Coverslip low: {min_z_pos}')
            print(f'Coverslip high: {max_z_pos}')
            print(f'Scan start: {min_x_pos}')
            print(f'Scan end: {max_x_pos}')
            print(f'Coverslip slope: {coverslip_slope_um}')

        # maximum allowed height change
        # for now, hardcode to 10% of coverslip height
        max_z_change_um = float(config["Stage"]["max_z_change_um"])

        # calculate allowed scan length and number of scan tiles for allowed coverslip height change
        if coverslip_slope_um == 0:
            scan_tile_length_um = range_x_um
            num_scan_tiles = 1
        else:
            scan_tile_length_um = np.round((max_z_change_um / coverslip_slope_um),2)
            num_scan_tiles = np.rint(range_x_um / scan_tile_length_um)

        # calculate scan axis tile locations
        scan_tile_overlap = .2 # unit: percentage
        scan_axis_step_um = float(config["Stage"]["stage_step_size_um"])  # unit: um 
        scan_axis_step_mm = scan_axis_step_um / 1000. #unit: mm
        scan_axis_start_mm = min_x_pos / 1000. #unit: mm
        scan_axis_end_mm = max_x_pos / 1000. #unit: mm
        scan_tile_length_mm = scan_tile_length_um / 1000. # unit: mm
        
        if num_scan_tiles > 1: # SJS changed from > 0, arange was returning empty
            scan_axis_start_pos_mm = np.round(np.arange(scan_axis_start_mm,scan_axis_end_mm+(1-scan_tile_overlap)*scan_tile_length_mm,(1-scan_tile_overlap)*scan_tile_length_mm),2) #unit: mm
            scan_axis_end_pos_mm = np.round(scan_axis_start_pos_mm + scan_tile_length_mm * (1+scan_tile_overlap),2)
            scan_axis_start_pos_mm = scan_axis_start_pos_mm[0:-2]
            scan_axis_end_pos_mm = scan_axis_end_pos_mm[0:-2]
            scan_tile_length_w_overlap_mm = np.abs(scan_axis_end_pos_mm[0]-scan_axis_start_pos_mm[0])
            scan_axis_positions = np.rint(scan_tile_length_w_overlap_mm / scan_axis_step_mm).astype(int)

        else:
            # Only 1 tile position to acquire
            scan_axis_start_pos_mm = [scan_axis_start_mm]
            scan_axis_end_pos_mm = [scan_axis_end_mm]
            scan_tile_length_w_overlap_mm = np.abs(scan_axis_end_pos_mm[0]-scan_axis_start_pos_mm[0])
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
        z_positions = np.round(np.linspace(min_z_pos,max_z_pos,len(scan_axis_start_pos_mm)),2)
        if len(z_positions) > 1:
            step_z_um = float(z_positions[1]-z_positions[0])
        else:
            step_z_um = 0
        if DEBUG: 
            print(f'Height axis start positions: {z_positions}.')
            print(f"min/max y pos: {min_y_pos} / {max_y_pos}")
            
        # calculate tile axis locations
        tile_axis_overlap=0.15 #unit: percentage
        tile_axis_ROI_um = np.round(float(camera_crop_x)*pixel_size_um,2)  #unit: um
        step_tile_axis_um = np.round((tile_axis_ROI_um) * (1-tile_axis_overlap),2) #unit: um
        tile_axis_positions = np.round(np.arange(min_y_pos,max_y_pos+step_tile_axis_um,step_tile_axis_um),2)
        if DEBUG: 
            print(f'Tile axis step: {step_tile_axis_um}')
            print(f'Tile axis start positions: {tile_axis_positions}.')
        
    # update the wfc mirror positions array shape
    n_stage_pos = num_scan_tiles*len(tile_axis_positions)*len(z_positions)
    num_tile_positions = len(tile_axis_positions)
    num_z_pos = len(z_positions)
    
    stage_positions_mm = []
    stage_positions = []
    for scan_idx in range(num_scan_tiles):
        for tile_idx in range(num_tile_positions):
            for z_idx in range(num_z_pos):
                stage_positions_mm.append({
                    "x": float(np.round((min_x_pos + scan_idx * scan_tile_length_um) / 1000., 2)),
                    "y": float(np.round((min_y_pos + tile_idx * step_tile_axis_um) / 1000., 2)),
                    "z": float(np.round((min_z_pos + z_idx * step_z_um) / 1000., 2))
                })
                stage_positions.append({
                    "x": float(np.round(min_x_pos + scan_idx * scan_tile_length_um, 2)),
                    "y": float(np.round(min_y_pos + tile_idx * step_tile_axis_um, 2)),
                    "z": float(np.round(min_z_pos + z_idx * step_z_um, 2))
                })
    print(f"validate number of stage positions: {n_stage_pos} vs from loop: {len(stage_positions_mm)}")
    AOmirror_stage.n_positions = n_stage_pos
    #SJS Stage positions are 0
    #--------------------------------------------------------------------#
    #--------------------------------------------------------------------#
    # Create CustomAction events
    # - O2O3 autofocus
    # - AO mirror optimization
    # - Fluidics program
    # - DAQ hardware setup
    # - Image events (to be sequenced)
    #--------------------------------------------------------------------#
    #--------------------------------------------------------------------#
        
    #--------------------------------------------------------------------#
    # Create CustomAction autofocus O2O3 events
    #--------------------------------------------------------------------#
    
    O2O3_event = MDAEvent(
        action=CustomAction(
            name="O2O3-autofocus",
            data = {
                "Camera" : {                    
                    "exposure_ms" : float(config["O2O3-autofocus"]["exposure_ms"]),
                    "camera_crop" : [
                        config["Camera"]["camera_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
                        config["Camera"]["camera_center_y"] - int(config["O2O3-autofocus"]["camera_crop_y"]//2),
                        config["Camera"]["camera_crop_x"],
                        config["O2O3-autofocus"]["camera_crop_y"]
                    ]
                }
            }
        )
    )
    
    #--------------------------------------------------------------------#
    # Create CustomAction events for running AO optimization
    #--------------------------------------------------------------------#
    # if not("system" in AO_mode):
    
    def calculate_projection_crop(image_mirror_range_um):
        # Calculate the the number of Y pixels for the scan range
        roi_height_um = image_mirror_range_um # * np.cos(np.deg2rad(30))
        roi_height_px = int(roi_height_um / mmc.getPixelSizeUm())
        return roi_height_px

    now = datetime.now()
    timestamp = f"{now.year:4d}{now.month:2d}{now.day:2d}_{now.hour:2d}{now.minute:2d}{now.second:2d}"
   
    AO_exposure_ms = round(float(config["AO-projection"]["exposure_ms"]),0)
    AO_active_channels = list(map(bool,config["AO-projection"]["channel_states"]))
    AO_laser_powers = list(float(_) for _ in config["AO-projection"]["channel_powers"])
    AO_camera_crop_y = int(calculate_projection_crop(config["AO-projection"]["image_mirror_range_um"]))
    AO_image_mirror_range_um = float(config["AO-projection"]["image_mirror_range_um"])
    AO_save_path = Path(output).parent / Path(f"{timestamp}_ao_optimize")
    
    # These AO parameters are determined by the config file.
    AO_iterations = int(config["AO-projection"]["iterations"])
    AO_metric = str(config["AO-projection"]["mode"])
    
    # Create AO event
    AO_event = MDAEvent(
        exposure = AO_exposure_ms,
        action=CustomAction(
            name="AO-projection",
            data = {
                "AO" : {
                    "opm_mode": str("projection"),
                    "active_channels": AO_active_channels,
                    "laser_powers" : AO_laser_powers,
                    "mode": AO_metric,
                    "iterations": AO_iterations,
                    "image_mirror_range_um" : AO_image_mirror_range_um,
                    "blanking": bool(True),
                    "apply_existing": bool(False),
                    "pos_idx":int(0),
                    "output_path":AO_save_path
                },
                "Camera" : {
                    "exposure_ms": AO_exposure_ms,
                    "camera_crop" : [
                        config["Camera"]["camera_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
                        config["Camera"]["camera_center_y"] - int(AO_camera_crop_y//2),
                        config["Camera"]["camera_crop_x"],
                        AO_camera_crop_y
                    ]
                }
            }
        )
    )
    

    #--------------------------------------------------------------------#
    # Create CustomAction Fluidics program
    #--------------------------------------------------------------------#

    # Create Fluidics program event
    FP_event = MDAEvent(
        # exposure = AO_exposure_ms,
        action=CustomAction(
            name="Fluidics",
            data = {
                "Fluidics" : {
                    "mode" : FP_mode,
                    "round" : int(0)
                }
                }
        )
    )
    
    #--------------------------------------------------------------------#
    # Create CustomAction DAQ event for Stage imaging modes
    #--------------------------------------------------------------------#
       
    # create DAQ hardware setup event
    DAQ_event = MDAEvent(
        action=CustomAction(
            name="DAQ-stage",
            data = {
                "DAQ" : {
                    "mode" : "stage",
                    "channel_states" : active_channels,
                    "channel_powers" : laser_powers,
                    "interleaved" : interleaved_acq,
                    "blanking" : laser_blanking, 
                },
                "Camera" : {
                    "exposure_channels" : exposure_channels,
                    "camera_crop" : [
                        config["Camera"]["camera_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
                        config["Camera"]["camera_center_y"] - int(camera_crop_y//2),
                        config["Camera"]["camera_crop_x"],
                        int(camera_crop_y),
                    ]
                }
            }
        )
    )

    #--------------------------------------------------------------------#
    # Create CustomAction events for setup constant speed scanning
    #--------------------------------------------------------------------#

    # create ASI hardware setup event
    ASI_setup_event = MDAEvent(
        action=CustomAction(
            name="ASI-setupscan",
            data = {
                "ASI" : {
                    "mode" : "scan",
                    "scan_axis_start_mm" : float(scan_axis_start_mm),
                    "scan_axis_end_mm" : float(scan_axis_end_mm),
                    "scan_axis_speed_mm_s" : float(scan_axis_speed)
                }
            }
        )
    )

    #--------------------------------------------------------------------#
    #--------------------------------------------------------------------#
    # Create event structure
    #--------------------------------------------------------------------#
    #--------------------------------------------------------------------#
    
    opm_events: list[MDAEvent] = []
       
    #--------------------------------------------------------------------#
    # Else populate a {t, p, c, z} event sequence
    # - Run fliudics, if present
    # - Run O2O3 AF before time points or positions
    # - Run AO before each acq or positions
    # - Program the daq
    # - Acquire images
    #--------------------------------------------------------------------#
    
    # Flags to help ensure sequence-able events are kept together 
    need_to_setup_DAQ = True
    need_to_setup_stage = True
    
    # Check if running AF initially to run the autofocus
    if O2O3_mode=="Initial-only":
        opm_events.append(O2O3_event)
    
    if DEBUG: 
            print(f'timepoints: {n_time_steps}')
            print(f'Stage positions: {n_stage_pos}.')
            print(f'Scan positions: {scan_axis_positions+int(config["Stage"]["excess_positions"])}.')
            print(f'Active channels: {n_active_channels}')
            
    # setup nD mirror-based AO-OPM acquisition event structure
    for time_idx in range(n_time_steps):
        # TODO Clarify how the acquisition should run. 
        # Right now, the first round is run manually in ESI, and then the imaging is setup afterwards. 
        # This offsets the number of rounds and if running fluidics, we acquire the first round images then run fluidics at the second time point.
        # Check if fluidics active
        if not(FP_mode=="None") and not(time_idx==0):
            current_FP_event = MDAEvent(**FP_event.model_dump())
            current_FP_event.action.data["Fluidics"]["round"] = int(time_idx)
            opm_events.append(current_FP_event)
        
        # Check if AF before each time point
        if O2O3_mode == "Before-each-t":
            opm_events.append(O2O3_event)
            
        # Check for multi-position acq.
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
                            
                    # Check if autofocus before each XYZ position and not initial-only mode
                    if O2O3_mode == "Before-each-xyz":
                        opm_events.append(O2O3_event)
                        
                    # Check if run AO opt. before each XYZ on first time we see this position
                    if (AO_mode == "Before-each-xyz") and (time_idx == 0):
                        need_to_setup_DAQ = True
                        current_AO_event = MDAEvent(**AO_event.model_dump())
                        current_AO_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                        current_AO_event.action.data["AO"]["apply_existing"] = False
                        opm_events.append(current_AO_event)
                        
                    # Apply correction for this position if time_idx > 0
                    elif (AO_mode == "Before-each-xyz") and (time_idx > 0):
                        need_to_setup_DAQ = True
                        current_AO_event = MDAEvent(**AO_event.model_dump())
                        current_AO_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                        current_AO_event.action.data["AO"]["apply_existing"] = True
                        opm_events.append(current_AO_event)
                        
                    # Otherwise, run AO opt. before every acquisition. COSTLY in time and photons!
                    elif AO_mode == "Before-every-acq":
                        need_to_setup_DAQ = True
                        current_AO_event = MDAEvent(**AO_event.model_dump())
                        current_AO_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                        current_AO_event.action.data["AO"]["apply_existing"] = False
                        opm_events.append(current_AO_event)
                        
                    # Finally, handle acquiring images. 
                    # These events are passed through to the normal MDAEngine and *should* be sequenced. 

                    # The DAQ waveform can be repeated for all time and spatial positions
                    if need_to_setup_DAQ:
                        need_to_setup_DAQ = True
                        opm_events.append(MDAEvent(**DAQ_event.model_dump()))

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
                                index=mappingproxy({
                                    "t": time_idx, 
                                    "p": pos_idx, 
                                    "c": chan_idx, 
                                    "z": scan_axis_idx
                                }),
                                metadata = {
                                    "DAQ" : {
                                        "mode" : "stage",
                                        "scan_axis_step_um" : float(scan_axis_step_um),
                                        "active_channels" : active_channels,
                                        "exposure_channels_ms": exposure_channels,
                                        "interleaved" : True,
                                        "laser_powers" : laser_powers,
                                        "blanking" : laser_blanking,
                                        "current_channel" : active_channel_names[chan_idx]
                                    },
                                    "Camera" : {
                                        "exposure_ms" : float(exposure_channels[chan_idx]),
                                        "camera_center_x" : config["Camera"]["camera_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
                                        "camera_center_y" : config["Camera"]["camera_center_y"] - int(camera_crop_y//2),
                                        "camera_crop_x" : config["Camera"]["camera_crop_x"],
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

            # Setup modified tensorstore handler
            handler = OPMMirrorHandler(
                path=Path(output),
                indice_sizes=indice_sizes,
                delete_existing=True
                )
                
            print(f"Using Qi2lab handler,\n  indices: {indice_sizes}")
                
        else:
            # If not, use built-in handler based on suffix
            print("Using defualt handler")
            handler = Path(output)
                    
    return opm_events, handler

