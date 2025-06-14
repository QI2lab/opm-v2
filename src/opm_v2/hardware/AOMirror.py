import numpy as np
from pathlib import Path
from numpy.typing import NDArray
from typing import List
import wavekit_py as wkpy
import time
import json

DEBUGGING = True
MIRROR_SETTLE_MS = 100

_instance_mirror = None

class AOMirror:
    """Class to control Imagine Optic Mirao52E.
    
    This class implements a subset of the `wavekit_py` SDK. Mainly,
    it allows for direct setting of mirror voltage and modal coefficients.
    There are safety factors built in to stop over-voltage for individual
    mirrors and the total voltage on the mirror.
    
    Parameters
    ----------
    wfc_config_file_path : Path
        _description_
    haso_config_file_path : Path
        _description_
    interaction_matrix_file_path : Path
        _description_
    system_flat_file_path : Path, default=  None
        _description_
    coeff_file_path : Path, default = None
        _description_
    n_modes : int, default = 32
        _description_
    n_positions: int, default = 1
        _description_
    modes_to_ignore : list[int], default = []
        _description_
    """

    @classmethod
    def instance(cls) -> 'AOMirror':
        """Return the global singleton instance of `AOMirror`.

        """
        global _instance_mirror
        if _instance_mirror is None:
            _instance_mirror = cls()
        return _instance_mirror

    def __init__(
        self,
        wfc_config_file_path: Path,
        haso_config_file_path: Path,
        interaction_matrix_file_path: Path,
        system_flat_file_path: Path = None,
        mirror_flat_file_path: Path = None,
        coeff_file_path: Path = None,
        n_modes: int = 32,
        n_positions: int = 1,
        modes_to_ignore: List[int] = [],
        output_path: Path = None,
        control_mode: str = 'modal'
        ):
        
        # Set the first instance of this class as the global singleton
        global _instance_mirror
        if _instance_mirror is None:
            _instance_mirror = self

        self._haso_config_file_path = haso_config_file_path
        self._wfc_config_file_path = wfc_config_file_path
        self._interaction_matrix_file_path = interaction_matrix_file_path
        self._coeff_file_path = coeff_file_path
        self._system_flat_file_path = system_flat_file_path
        self._mirror_flat_file_path = mirror_flat_file_path
        self._n_modes = n_modes
        self._modes_to_ignore = modes_to_ignore
        self._n_positions = n_positions
        self._output_path = output_path
        self._control_mode = control_mode
        
        #---------------------------------------------#
        # Start wfc 
        #---------------------------------------------#
        
        # Wavefront corrector and set objects
        self.wfc = wkpy.WavefrontCorrector(
            config_file_path = str(self._wfc_config_file_path)
        )
        self.wfc_set = wkpy.WavefrontCorrectorSet(wavefrontcorrector = self.wfc)
        self.wfc.connect(True)
        
        # create corrdata manager object and compute command matrix
        self.corr_data_manager = wkpy.CorrDataManager(
            haso_config_file_path = str(haso_config_file_path),
            interaction_matrix_file_path = str(interaction_matrix_file_path)
        )
        self.corr_data_manager.set_command_matrix_prefs(self._n_modes,True)
        self.corr_data_manager.compute_command_matrix()

        self.wfc.set_temporization(MIRROR_SETTLE_MS)

        # create the configuration object
        self.haso_config, self.haso_specs, _ = wkpy.HasoConfig.get_config(config_file_path=str(self._haso_config_file_path))
        
        # construct pupil dimensions from haso specs
        pupil_dimensions = wkpy.dimensions(self.haso_specs.nb_subapertures,self.haso_specs.ulens_step)

        # create HasoSlopes object
        self.haso_slopes = wkpy.HasoSlopes(
            dimensions=pupil_dimensions,
            serial_number = self.haso_config.serial_number
        )
        # initiate an empty pupil
        self.pupil = wkpy.Pupil(dimensions=pupil_dimensions, value=False)
        pupil_buffer = self.corr_data_manager.get_greatest_common_pupil()
        self.pupil.set_data(pupil_buffer)

        center, radius = wkpy.ComputePupil.fit_zernike_pupil(
            self.pupil,
            wkpy.E_PUPIL_DETECTION.AUTOMATIC,
            wkpy.E_PUPIL_COVERING.CIRCUMSCRIBED,
            False
        )
        # create modal coeff object
        self.modal_coeff = wkpy.ModalCoef(modal_type=wkpy.E_MODAL.ZERNIKE)
        self.modal_coeff.set_zernike_prefs(
            zernike_normalisation=wkpy.E_ZERNIKE_NORM.RMS, 
            nb_zernike_coefs_total=self._n_modes, 
            coefs_to_filter=self._modes_to_ignore,
            projection_pupil=wkpy.ZernikePupil_t(
                center,
                radius
            )
        )
        # update modal data with zero coeffs.
        self.modal_coeff.set_data(
            coef_array = np.zeros(n_modes),
            index_array = np.arange(1, self._n_modes+1, 1),
            pupil = self.pupil
        ) 
        
        #---------------------------------------------#
        # Set up wfc positions and mirror position tracking
        #---------------------------------------------#
        
        if self._system_flat_file_path is not None:
            self.system_flat_voltage = np.asarray(
                self.wfc.get_positions_from_file(str(self._system_flat_file_path))
            )
        else:
            self.system_flat_voltage = np.zeros(self.wfc.nb_actuators)
            
        if self._mirror_flat_file_path is not None:
            self.mirror_flat_voltage = np.asarray(
                self.wfc.get_positions_from_file(str(self._mirror_flat_file_path))
            )
        else:
            self.mirror_flat_voltage = np.zeros(self.wfc.nb_actuators)

        # For tracking the current mirror state
        self._current_coeffs = np.zeros(n_modes,dtype=np.float32)
        self._current_voltage = np.asarray(self.system_flat_voltage)
        self._current_zernikes = {}
        # Initialize last opt arrays with zeros
        self.last_opt_modal_array = self._current_coeffs
        self.last_opt_volt_array = self._current_voltage
        # Store a 1d array of ao mirror positions that correspond to stage positions array.
        self.positions_voltage_array = np.zeros((n_positions,self.wfc.nb_actuators))
        self.positions_modal_array = np.zeros((n_positions,31)) # shape matches mode names length
        
        # Set mirror in system flat by defualt
        self.apply_system_flat_voltage()
        # Set mirror in system flat by defualt
        self.apply_system_flat_voltage()

        #---------------------------------------------#
        # Define mode names matching the mirror modes
        #---------------------------------------------#
        
        self.mode_names = [
            "Vert. Tilt", # 0
            "Horz. Tilt", # 1
            "Defocus", # 2
            "Vert. Asm.", # 3
            "Oblq. Asm.", # 4
            "Vert. Coma", # 5
            "Horz. Coma", # 6
            "3rd Spherical",  # 7
            "Vert. Tre.", # 8
            "Horz. Tre.", # 9
            "Vert. 5th Asm.", # 10
            "Oblq. 5th Asm.", # 11
            "Vert. 5th Coma", # 12
            "Horz. 5th Coma", # 13
            "5th Spherical", # 14
            "Vert. Tetra.", # 15
            "Oblq. Tetra.", # 16
            "Vert. 7th Tre.", # 17
            "Horz. 7th Tre.", # 18
            "Vert. 7th Asm.", # 19
            "Oblq. 7th Asm.", # 20
            "Vert. 7th Coma", # 21
            "Horz. 7th Coma", # 22
            "7th Spherical", # 23
            "Vert. Penta.", # 24
            "Horz. Penta.", # 25
            "Vert. 9th Tetra.", # 26
            "Oblq. 9th Tetra.", # 27
            "Vert. 9th Tre.", # 28
            "Horz. 9th Tre.", # 29
            "Vert. 9th Asm.", # 30
            "Oblq. 9th Asm.", # 31
        ]
    
    @property
    def control_mode(self) -> str:
        return self._control_mode

    @control_mode.setter
    def control_mode(self, mode: str):
        if mode not in ('modal', 'position'):
            raise ValueError("AOmirror control_mode must be 'modal' or 'position'")
        self._control_mode = mode
        
    @property
    def output_path(self) -> str|Path:
        """Output path.
        
        Returns
        -------
        output_path: str
            output path
        """
        
        return getattr(self,"_output_path",None)

    @output_path.setter
    def output_path(self, value: str|Path):
        """Set the output path.
        
        Parameters
        ----------
        value: str|Path
            output_path 
        """
        
        if not hasattr(self, "_output_path") or self._output_path is None:
            self._output_path = value
        else:
            self._output_path = value

    @property
    def n_positions(self) -> int:
        """Number of experimental "positions".
        
        Returns
        -------
        n_positions: int
            number of wavefronts to store, tied to experimental "positions".
        """
        
        return getattr(self,"_n_positions",None)

    @n_positions.setter
    def n_positions(self, value: int):
        """Set the number of experimental "positions".
        
        Parameters
        ----------
        value: int
            number of wavefronts to store, tied to experimental "positions".
        """
        
        if not hasattr(self, "_n_positions") or self._n_positions is None:
            self._n_positions = value
        else:
            self._n_positions = value
        self.positions_voltage_array = np.zeros((self._n_positions,self.wfc.nb_actuators))
        self.positions_modal_array = np.zeros((self._n_positions,len(self.mode_names)))

    @property
    def current_voltage(self) -> np.ndarray:
        """Get current mirror positions."""
        return self._current_voltage
        return self._current_voltage

    @current_voltage.setter
    def current_voltage(self, value: np.ndarray):
        """Set and update current mirror positions."""
        self._current_voltage = value
        self._deltas = self._current_voltage - self.system_flat_voltage

    @property
    def current_coeffs(self) -> np.ndarray:
        """Get current modal coefficients."""
        return self._current_coeffs

    @current_coeffs.setter
    def current_coeffs(self, value: np.ndarray):
        """Set current modal coefficients."""
        self._current_coeffs = value

    @property
    def current_zernikes(self) -> dict:
        """Get current Zernike modes and amplitude"""
        return self._current_zernikes
    
    @current_zernikes.setter
    def current_zernikes(self):
        """Set the current zernike dict"""
        # Save dictionary of current coefficients where the key is the mode name
        mode_dict = {}
        for c in range(len(self.mode_names)):
            mode_dict[self.mode_names[c - 1]] = f"{self.current_coeffs[c-1]:.4f}"
        self._current_zernikes = mode_dict
        
    @property
    def deltas(self) -> np.ndarray:
        """Get the difference between current and flat mirror positions."""
        return self._deltas

    def __del__(self):
        """Disconnect from mirror on close"""
        self.wfc.disconnect()

    def _validate_voltage(self, volts: NDArray) -> bool:
        """Ensure mirror positions are within safe voltage limits."""
        if volts.shape[0] != self.wfc.nb_actuators:
            print(f"Volts array must have shape = {self.wfc.nb_actuators}")
            return False
        if np.sum(np.where(np.abs(volts) >= 0.99,1,0)) > 1:
            print('Individual actuator voltage too high.')
            return False
        if np.sum(np.abs(volts)) >= 25:
            print('Total voltage too high.')
            return False
        else:
            return True
        
    def _update_current_state(self, key: str=None):
        """Update stored mirror positions from wavefront corrector."""
        # Only track modal coeffs if committed
        if self.control_mode == 'modal':
            self.current_coeffs = np.asarray(self.modal_coeff.get_coefs_values()[0])
        # Always track positions
        self.current_voltage = np.array(self.wfc.get_current_positions())   

    def update_positions_array(self, idx):
        """Update stage position arrays with the current voltage or modal coeffs.

        Parameters
        ----------
        idx : int
            stage positions array index
        """
        if self.control_mode == 'modal':
            self.positions_modal_array[idx, :] = self.current_coeffs.copy() 
        self.positions_voltage_array[idx, :] = self.current_voltage.copy()
    
    def update_optimized_array(self):
        """Update the last optimized voltage and modal arrays
        """
        if self.control_mode == 'modal':
            self.last_opt_modal_array = self.current_coeffs.copy() 
        self.last_opt_volt_array = self.current_voltage.copy()
    
    def apply_system_flat_voltage(self):
        """Set mirror to positions to system flat."""
        if self.control_mode == 'modal':
            self.set_modal_coefficients(np.zeros(self._n_modes))       
        else:
            self.set_mirror_voltage(self.system_flat_voltage)
        self._update_current_state()
        
    def apply_last_optimized(self):
        """Set mirror to positions to last optimized
        
        Note: Return mirror positions to 'last optimized' synchronizes the mirror 
        positions with the modal model coefficients.
        """
        if self.control_mode == 'modal':
            self.set_modal_coefficients(self.last_opt_modal_array)
        else:
            self.set_mirror_voltage(self.last_opt_volt_array)
        self._update_current_state()
        
    def apply_positions_array(self, idx: int = 0):
        """Set mirror positions from stored array.
        
        Used in nD acquisitions where each "position" has a unique correction.

        Parameters
        ----------
        idx: int, default = 0
            position index to use
        """
        try:
            if self.control_mode == 'modal':
                self.set_modal_coefficients(self.positions_modal_array[idx, :])
            else:
                self.set_mirror_voltage(self.positions_voltage_array[idx,:])
            self._update_current_state()
        except Exception as e:
            print(f'\nSetting mirror positions from POS array failed!/n  e:{e}\n')
            
            
    def set_mirror_voltage(self, positions: NDArray):
        """Set mirror positions.

        Parameters
        ----------
        positions : NDArray
            Flatten array of actuators 
        """
        
        if self.control_mode != "positions":
            print('Control mode set to modal, cannot use actuator positions!')
            return False
        
        if self._validate_voltage(positions):
            self.wfc.move_to_absolute_positions(positions)
            time.sleep(MIRROR_SETTLE_MS*10**-3)
            self._update_current_state()
            return True
        else:
            return False    
        
    def set_modal_coefficients(self,amps: NDArray):
        """Set modal coefficients.
        Amps are relative to zeros from the selected mirror flat file.
        
        Parameters
        ----------
        amps : NDArray
            Flatten array of Zernike modes
        """
        if self.control_mode != "modal":
            print('Control mode set to positions, cannot use modal coefficients!')
            return False
        
        assert amps.shape[0]==self._n_modes, "amps array must have the same shape as the number of Zernike modes."
        
        # update modal data
        self.modal_coeff.set_data(
            coef_array = amps,
            index_array = np.arange(1, self._n_modes+1, 1),
            pupil = self.pupil
        ) 
        
        # create a new haso_slope from the new modal coefficients
        haso_slopes = wkpy.HasoSlopes(
            modalcoef = self.modal_coeff, 
            config_file_path=str(self._haso_config_file_path)
        )
        # calculate the voltage delta to achieve the desired modalcoef
        deltas = self.corr_data_manager.compute_delta_command_from_delta_slopes(delta_slopes=haso_slopes)
        new_positions = np.asarray(self.system_flat_voltage) + np.asarray(deltas)
        new_positions = np.asarray(self.system_flat_voltage) + np.asarray(deltas)
            
        if self._validate_voltage(new_positions):
            # Move mirror actuators
            self.wfc.move_to_absolute_positions(new_positions)
            # blocking pause for mirror settling
            time.sleep(MIRROR_SETTLE_MS*10**-3)
            # Update local mirror position and coeff amplitude arrays
            self._update_current_state()
            return True
        else:
            return False  

    def save_current_state(self, prefix: str = 'current'):
        """Save current mirror positions to disk.

        Parameters
        ----------
        prefix : str
        prefix : str
            _description_
        """
        self._update_current_state()
        
        try:
            if self._output_path is None:
                raise Exception('missing output path')
            
            else:
                # save wfc compatible file
                actuator_save_path = self._output_path / Path(f"{prefix}_wfc_voltage.wcs") 
                self.wfc.save_current_voltage_to_file(pmc_file_path=str(actuator_save_path))
                
                # save current state and last optimized positions to disk
                metadata_path = self._output_path / Path(f'{prefix}_aoMirror_state.json')
                metadata = {
                    'mode_names':self.mode_names,
                    'last_optimized':{
                        'mode_amplitudes': self.last_opt_modal_array,
                        'volt_amplitudes': self.last_opt_volt_array
                    },
                    'current_state':{
                        'mode_amplitudes': self.current_coeffs,
                        'volt_amplitudes': self.current_voltage
                    }
                }
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f)
                
                # Save dictionary of current coefficients where the key is the mode name
                modal_save_path = self._output_path / Path(f"{prefix}_aoMirror_zernike.json")
                mode_dict = {}
                for c in range(len(self.mode_names)):
                    mode_dict[self.mode_names[c - 1]] = f"{self.current_coeffs[c-1]:.4f}"

                with open(modal_save_path, "w") as f:
                    json.dump(mode_dict, f)
            
        except Exception as e:
            print(f'--- AOmirror: Failed to save current state! ----\n  {e} \n')
            
    def save_positions_array(self, prefix : str = "stage_position"):
        """Save wfc positions array to disk
        
        Parameters
        ----------
        prefix : str
            Filename prefix, by default "stage_position"
        """
        if self._output_path:
            positions_file_path = self._output_path / Path(f"{prefix}_voltage.json")
            positions_list = self.positions_voltage_array.tolist()
            with open(positions_file_path, "w") as f:
                json.dump(positions_list, f)
                
            positions_file_path = self._output_path / Path(f"{prefix}_mode_amplitude.json")
            wfc_coeffs_list = self.positions_modal_array.tolist()
            with open(positions_file_path, "w") as f:
                json.dump(wfc_coeffs_list, f)
        else:
            pass
            
    def load_positions_array(self, prefix : str = "stage_position"):
        """Load positions from json saved using 'save_positions_array'

        Parameters
        ----------
        prefix : str, optional
            file name prefix, by default "stage_position"
        """
        file_path = self._output_path / Path(f"{prefix}_voltage.json")
        with open(file_path, "r") as f:
            voltage_list = json.load(f)
        self.positions_voltage_array = np.asarray(voltage_list)

        file_path = self._output_path / Path(f"{prefix}_mode_amplitude.json")
        with open(file_path, "r") as f:
            mode_amplitudes = json.load(f)
        self.positions_modal_array = np.asarray(mode_amplitudes)

def DM_voltage_to_map(v):
    """Reshape mirror to a map.

    Reshape the 52-long vector v into 2D matrix representing the actual DM aperture.
    Corners of the matrix are set to None for plotting.

    Author: Nikita Vladimirov

    Parameters
    ----------
    v: float array of length 52
    
    Returns
    -------
    output: 8x8 ndarray of doubles.
    """

    M = np.zeros((8,8))
    M[:,:] = None
    M[2:6,0] = v[:4]
    M[1:7,1] = v[4:10]
    M[:,2] = v[10:18]
    M[:,3] = v[18:26]
    M[:,4] = v[26:34]
    M[:,5] = v[34:42]
    M[1:7,6] = v[42:48]
    M[2:6,7] = v[48:52]

    return M
    
def plotDM(
    cmd: NDArray, 
    title:str = "", 
    cmap: str = "jet", 
    vmin: float =-0.25,
    vmax: float =0.25,
    save_dir_path: Path = None,
    show_fig: bool = False
):
    """Plot the current mirror state.

    Parameters
    ----------
    cmd : NDArray
        (8,8) array of DM voltages
    title : str, default = ""
        Title of plot window
    cmap : str, default = "jet"
        Colormap to use.
    vmin : float, default = -0.25
        Colormap minimum value
    vmax : float, default = +0.25
        Colormap maximum value
    save_dir_path : Path, default = None
        Path to save DM plot
    show_fig : bool, default = False
        Show result of DM map
    """

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1)
    valmax = np.nanmax(cmd)
    valmin = np.nanmin(cmd)
    im = ax.imshow(
        DM_voltage_to_map(cmd),
        vmin=vmin, 
        vmax=vmax,
        interpolation='nearest', 
        cmap = cmap
    )
    ax.text(
        0,
        -1, 
        title + '\n min=' + "{:1.2f}".format(valmin) + ', max=' + "{:1.2f}".format(valmax) + ' V',
        fontsize=12
    )

    plt.colorbar(im)
    
    if save_dir_path:
        fig.savefig(save_dir_path / Path("mirror_positions.png"))
    if show_fig:
        plt.show()