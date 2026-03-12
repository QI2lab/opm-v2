import json
import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import wavekit_py as wkpy
from numpy.typing import ArrayLike, NDArray

DEBUGGING = False

# TODO: Test effect of tilt filtering
# TODO: Test the effect of wait_after_move, adding a pause on top of temperization
# NOTE: Update modal_coef before setting the mirror voltage!
# NOTE: Only call update_current_state if the mirror changes!
# NOTE: Changing mirror state to factory_flat or zeros does not modify modal_coef!
# NOTE: Changing mirror state to factory_flat and zeros is not exposed!
# TODO: How find the modal_coef that transforms from system -> zeros or flat voltages.

_instance_mirror = None


class AOMirror:
    """Class to control Imagine Optic Mirao52E.

    This class implements a subset of the `wavekit_py` SDK. Mainly,
    it allows for direct setting of mirror voltage states and modal coefficients.
    There are safety factors built in to stop over-voltage for individual
    mirrors and the total voltage on the mirror.

    The only mirror voltage states are 'system_flat' which are loaded from a .wfc file,
    'factory_flat' which is the preloaded flat setting, and 'zero_voltage' which applies
    zero volts to each actuator. Modifying the mirror states is only allowed by applying
    Zernike modal coefficients, defined by an array of mode amplitudes.

    Parameters
    ----------
    wfc_config_file_path : Path
        wavefront corrector config file *.
    haso_config_file_path : Path
        _description_
    interaction_matrix_file_path : Path
        _description_
    system_flat_file_path : Path, default=  None
        _description_
    n_positions: int, default = 1
        _description_
    modes_to_ignore : list[int], default = None
        _description_
    n_modes : int, default = 32
        _description_
    tilt_filtering: bool, default = False
        _description
    mirror_settle_ms: float = 50.0
        _description
    wait_after_move: bool = False
        Whether to apply a sleep period after move
    output_path: Path, default = None
        _description
    """

    @classmethod
    def instance(cls) -> "AOMirror":
        """Return the global singleton instance of `AOMirror`."""
        global _instance_mirror
        if _instance_mirror is None:
            _instance_mirror = cls()
        return _instance_mirror

    def __init__(
        self,
        wfc_config_file_path: Path,
        haso_config_file_path: Path,
        interaction_matrix_file_path: Path,
        system_flat_file_path: Path | None = None,
        n_positions: int = 1,
        modes_to_ignore: List[int] | None = None,
        n_modes: int = 32,
        tilt_filtering: bool = False,
        mirror_settle_ms: float = 50.0,
        wait_after_move: bool = False,
        output_path: Path | None = None,
    ):
        # Set the first instance of this class as the global singleton
        global _instance_mirror
        if _instance_mirror is None:
            _instance_mirror = self

        self._haso_config_file_path = haso_config_file_path
        self._wfc_config_file_path = wfc_config_file_path
        self._interaction_matrix_file_path = interaction_matrix_file_path
        self._system_flat_file_path = system_flat_file_path
        self._output_path = output_path
        self._ignored_modes = (
            list(modes_to_ignore) if modes_to_ignore is not None else []
        )
        self._n_positions = n_positions
        self._n_modes = int(n_modes)
        self._tilt_filtering = bool(tilt_filtering)
        self._mode_indices = np.arange(1, self._n_modes + 1, dtype=np.uint32)
        self._mirror_settle_ms = int(mirror_settle_ms)
        self._wait_after_move = bool(wait_after_move)

        # ---------------------------------------------#
        # Start wfc and modal / corr_data_manager / modal_coef objects
        # ---------------------------------------------#

        self.wfc = wkpy.WavefrontCorrector(
            config_file_path=str(self._wfc_config_file_path)
        )
        self.wfc_set = wkpy.WavefrontCorrectorSet(wavefrontcorrector=self.wfc)
        self.wfc.connect(True)
        self.wfc.set_temporization(self._mirror_settle_ms)
        self.n_actuators = self.wfc.nb_actuators

        # Create corrdata manager object and compute command matrix
        self.corr_data_manager = wkpy.CorrDataManager(
            haso_config_file_path=str(haso_config_file_path),
            interaction_matrix_file_path=str(interaction_matrix_file_path),
        )
        self.corr_data_manager.set_command_matrix_prefs(
            self._n_modes, self._tilt_filtering
        )
        self.corr_data_manager.compute_command_matrix()

        # Create the Haso configuration object
        self.haso_config, self.haso_specs, _ = wkpy.HasoConfig.get_config(
            config_file_path=str(self._haso_config_file_path)
        )

        # Initiate pupil
        pupil_dimensions = wkpy.dimensions(
            self.haso_specs.nb_subapertures, self.haso_specs.ulens_step
        )
        self.pupil = wkpy.Pupil(dimensions=pupil_dimensions, value=False)
        pupil_buffer = self.corr_data_manager.get_greatest_common_pupil()
        self.pupil.set_data(pupil_buffer)
        center, radius = wkpy.ComputePupil.fit_zernike_pupil(
            self.pupil,
            wkpy.E_PUPIL_DETECTION.AUTOMATIC,
            wkpy.E_PUPIL_COVERING.CIRCUMSCRIBED,
            False,
        )

        # Create modal coeff object with Zernike
        self.modal_coeff = wkpy.ModalCoef(modal_type=wkpy.E_MODAL.ZERNIKE)
        self.modal_coeff.set_zernike_prefs(
            zernike_normalisation=wkpy.E_ZERNIKE_NORM.RMS,
            nb_zernike_coefs_total=self._n_modes,
            coefs_to_filter=self._ignored_modes,
            projection_pupil=wkpy.ZernikePupil_t(center, radius),
        )
        self.modal_coeff.set_data(
            coef_array = np.zeros(n_modes),
            index_array = self._mode_indices,
            pupil = self.pupil
        )

        # ---------------------------------------------#
        # Set up wfc positions and mirror position tracking
        # - Reference state and voltage for applying modal coeffs from
        # - system flat, loaded from file
        # - factory flat, loaded from wfc set
        # - zeros, set by applying 0V to each actuator
        # - Track the optimized mirror coeffs and voltage, apply voltages!
        # - Track positions arrays for coeffs and voltage, apply voltages!
        # ---------------------------------------------#
        
        # init the reference state variables
        self._reference_state = None
        self._ref_voltage = np.zeros(self.n_actuators)

        # Get the system flat voltage from file
        if self._system_flat_file_path is not None:
            self.system_flat_voltage = np.array(
                self.wfc.get_positions_from_file(str(self._system_flat_file_path)),
                dtype=np.float32,
                copy=True,
            )
        else:
            self.system_flat_voltage = np.zeros(self.n_actuators)

        # Get the factory_flat position
        self.factory_flat_voltage = np.array(
            self.wfc_set.get_flat_mirror_positions(), dtype=np.float32, copy=True
        )
        # Get the zero-voltage position
        self.zeros_voltage = np.zeros(self.n_actuators)

        # Setup arrays for tracking applied mirror voltages and Zernike amplitudes
        self._current_coeffs = np.zeros(self._n_modes, dtype=np.float32)
        self._current_voltage = self.system_flat_voltage.copy()

        # Setup the arrays to hold optimized values. 
        self._optimized_modal_coefs = self._current_coeffs.copy()
        self._optimized_voltage = self._current_voltage.copy()

        # Setup the 2d array of AOmirror positions that match to exp. stage positions
        self.positions_voltage_array = np.zeros((self._n_positions, self.n_actuators))
        self.positions_modal_array = np.zeros((self._n_positions, self._n_modes))

        # Start mirror at system flat voltage
        self.apply_system_flat_voltage()
        self.update_optimized_array()

        print(f"system_flat_voltage: {self.system_flat_voltage}")
        # Define mode names matching the mirror modes
        self.mode_names = [
            "Vert. Tilt",  # 0
            "Horz. Tilt",  # 1
            "Defocus",  # 2
            "Vert. Asm.",  # 3
            "Oblq. Asm.",  # 4
            "Vert. Coma",  # 5
            "Horz. Coma",  # 6
            "3rd Spherical",  # 7
            "Vert. Tre.",  # 8
            "Horz. Tre.",  # 9
            "Vert. 5th Asm.",  # 10
            "Oblq. 5th Asm.",  # 11
            "Vert. 5th Coma",  # 12
            "Horz. 5th Coma",  # 13
            "5th Spherical",  # 14
            "Vert. Tetra.",  # 15
            "Oblq. Tetra.",  # 16
            "Vert. 7th Tre.",  # 17
            "Horz. 7th Tre.",  # 18
            "Vert. 7th Asm.",  # 19
            "Oblq. 7th Asm.",  # 20
            "Vert. 7th Coma",  # 21
            "Horz. 7th Coma",  # 22
            "7th Spherical",  # 23
            "Vert. Penta.",  # 24
            "Horz. Penta.",  # 25
            "Vert. 9th Tetra.",  # 26
            "Oblq. 9th Tetra.",  # 27
            "Vert. 9th Tre.",  # 28
            "Horz. 9th Tre.",  # 29
            "Vert. 9th Asm.",  # 30
            "Oblq. 9th Asm.",  # 31
        ]
    
    @property
    def output_path(self) -> str | Path:
        """Output path.

        Returns
        -------
        output_path: str
            output path
        """
        return self._output_path

    @output_path.setter
    def output_path(self, value: str | Path):
        """Set the output path.

        Parameters
        ----------
        value: str|Path
            output_path
        """
        self._output_path = Path(value)

    @property
    def n_positions(self) -> int:
        """Number of experimental "positions".

        Returns
        -------
        n_positions: int
            number of wavefronts to store, tied to experimental "positions".
        """
        return self._n_positions

    @n_positions.setter
    def n_positions(self, value: int):
        """Set the number of experimental "positions".

        Parameters
        ----------
        value: int
            number of wavefronts to store, tied to experimental "positions".
        """
        if value == 0:
            value = 1
        self._n_positions = value
        self.positions_voltage_array = np.vstack(
            [self.system_flat_voltage] * self._n_positions
        )
        self.positions_modal_array = np.zeros((self._n_positions, len(self.mode_names)))

    @property
    def current_voltage(self) -> NDArray:
        """Get current mirror positions."""
        return self._current_voltage

    @current_voltage.setter
    def current_voltage(self, value: NDArray):
        """Set and update current mirror positions."""
        if value.shape != (self.n_actuators,):
            raise ValueError("Current voltage array shape mismatch!")

        self._current_voltage = value
        self._deltas = self._current_voltage - self.system_flat_voltage

    @property
    def current_coeffs(self) -> NDArray:
        """Get current modal coefficients."""
        return self._current_coeffs

    @current_coeffs.setter
    def current_coeffs(self, value: NDArray):
        """Set current modal coefficients."""
        if value.shape != (self._n_modes,):
            raise ValueError("Current modal_coef array shape mismatch!")

        self._current_coeffs = value

    @property
    def optimized_voltage(self) -> NDArray:
        """Get optimized voltage."""
        return self._optimized_voltage

    @optimized_voltage.setter
    def optimized_voltage(self, value: NDArray):
        """Set optimized voltage."""
        if value.shape != (self.n_actuators,):
            raise ValueError("Current voltage array shape mismatch!")

        self._optimized_voltage = value

    @property
    def optimized_modal_coeffs(self) -> NDArray:
        """Get current modal coefficients."""
        return self._optimized_modal_coefs

    @optimized_modal_coeffs.setter
    def optimized_modal_coeffs(self, value: NDArray):
        """Set current modal coefficients."""
        if value.shape != (self._n_modes,):
            raise ValueError("Current modal_coef array shape mismatch!")

        self._optimized_modal_coefs = value
        
    @property
    def current_zernikes(self) -> dict:
        """Get current Zernike modes and amplitude"""
        return {
            name: f"{float(self._current_coeffs[i]):.4f}"
            for i, name in enumerate(self.mode_names)
        }

    @property
    def deltas(self) -> NDArray:
        """Get the difference between current and flat mirror positions."""
        return self._deltas


    
    def __del__(self):
        """Disconnect from mirror on close"""
        self.wfc.disconnect()

    def _validate_voltage(self, volts: NDArray) -> bool:
        """Ensure mirror positions are within safe voltage limits."""
        if volts.shape != (self.n_actuators,):
            print(
                "------- AOmirror VOLTAGE ERROR -------\n"
                f"Voltage validation: Volts array must have shape = {self.n_actuators}"
            )
            return False
        elif np.sum(np.where(np.abs(volts) >= 0.99, 1, 0)) > 1:
            print(
                "------- AOmirror -------\n"
                "Voltage validation: Individual actuator voltage too high."
            )
            return False
        else:
            return True
    
    def _verify_current_voltage(self, volts: NDArray, tolerance:int=0.001) -> bool:
        """Validate the current mirror state matches the requested volts."""
        success = np.allclose(
                self.current_voltage, volts, rtol=tolerance, atol=0
            )
        if success:
            return success
        else:
            print(
                "------- AOmirror: VOLTAGE ERROR -------",
                f"Requested mirror voltage: {volts}",
                f"Current voltage: {self.current_voltage}",
                f"Current ref state: {self._reference_state}",
                sep="\n"
            )
            return False
        
    # -------------------------------------------------#
    # Method for updating and saving mirror reference states,
    # used for applying modal coefficients
    # -------------------------------------------------#

    def set_reference_state(self, state: str):
        """Set the reference state for applying modal coefficients. """
        if state == "system_flat":
            self._reference_state = "system_flat"
            self._ref_voltage = self.system_flat_voltage.copy()
        elif state == "factory_flat":
            self._reference_state = "factory_flat"
            self._ref_voltage = self.factory_flat_voltage.copy()
        elif state == "zeros_voltage":
            self._reference_state = "zeros_voltage"
            self._ref_voltage = self.zeros_voltage.copy()
        else:
            raise ValueError(
                "Invalid reference state, must be 'system_flat', 'factory_flat', or 'zeros_voltage'!"
            )

    # -------------------------------------------------#
    # Methods for updating saved arrays
    # -------------------------------------------------#

    def _update_current_state(self):
        """Update mirror state tracking arrays

        Voltage values from WFC
        Modal coef amplitudes from the model_coef
        """
        self.current_coeffs = np.array(
            self.modal_coeff.get_coefs_values()[0], dtype=np.float32, copy=True
        )
        self.current_voltage = np.array(
            self.wfc.get_current_positions(), dtype=np.float32, copy=True
        )
        if DEBUGGING:
            print(
                "------- AOmirror -------\n"
                "Updated current state arrays:\n"
                f"current_coefs: {self.current_coeffs}\n"
                f"current_voltages: {self.current_voltage}"
            )

    def update_positions_array(self, idx):
        """Update stage position arrays with the current voltage or modal coeffs.

        Parameters
        ----------
        idx : int
            stage positions array index
        """
        self._update_current_state()
        self.positions_modal_array[idx, :] = self.current_coeffs.copy()
        self.positions_voltage_array[idx, :] = self.current_voltage.copy()

        if DEBUGGING:
            print(
                "------- AOmirror -------\n"
                "Updated positions arrays:\n"
                f"modal_coefs: {self.positions_modal_array}\n"
                f"voltages: {self.positions_voltage_array}"
            )

    def update_optimized_array(self):
        """Update the last optimized voltage and modal arrays"""
        self._update_current_state()
        self.optimized_modal_coefs = self.current_coeffs.copy()
        self.optimized_voltage = self.current_voltage.copy()

        if DEBUGGING:
            print(
                "------- AOmirror -------\n"
                "Updated optimized arrays:\n"
                f"modal_coefs: {self.optimized_modal_coefs}\n"
                f"voltages: {self.optimized_voltage}"
            )

    # -------------------------------------------------#
    # Methods for applying saved mirror states
    # -------------------------------------------------#
    
    def apply_factory_flat_voltage(self):
        """Set mirror to positions to system flat."""
        try:
            _old_voltage = self.current_voltage.copy()
            self.set_reference_state("factory_flat")
            self.set_modal_coefficients(np.zeros(self._n_modes))
            self._update_current_state()
            success = self._verify_current_voltage(self.factory_flat_voltage)
        except Exception as e:
            print(f"Exception occured in setting FACTORY FLAT state:\n {e}")
            self.set_mirror_voltage(_old_voltage)
            success = False
        
        return success

    def apply_zeros_voltage(self):
        """Set mirror to positions to system flat."""
        try:
            _old_voltage = self.current_voltage.copy()
            self.set_reference_state("zeros_voltage")
            self.set_modal_coefficients(np.zeros(self._n_modes))
            self._update_current_state()
            success = self._verify_current_voltage(self.zeros_voltage)
        except Exception as e:
            print(f"Exception occured in setting ZEROS state:\n {e}")
            self.set_mirror_voltage(_old_voltage)
            success = False
            
        return success

    def apply_system_flat_voltage(self):
        """Set mirror to positions to system flat."""
        try:
            _old_voltage = self.current_voltage.copy()
            self.set_reference_state("system_flat")
            self.set_modal_coefficients(np.zeros(self._n_modes))
            self._update_current_state()
            success = self._verify_current_voltage(self.system_flat_voltage)
        except Exception as e:
            print(f"Exception occured in setting SYSTEM FLAT state:\n {e}")
            self.set_mirror_voltage(_old_voltage)
            success = False
            
        return success

    def apply_optimized_voltage(self, reference_state:str="system_flat"):
        """Set mirror to positions to last optimized"""
        try:
            _old_voltage = self.current_voltage.copy()
            self.set_reference_state(reference_state)
            self.set_modal_coefficients(self.optimized_modal_coefs)
            self._update_current_state()
            success = self._verify_current_voltage(self.optimized_voltage)
        except Exception as e:
            print(f"Exception occured in setting SYSTEM FLAT state:\n {e}")
            self.set_mirror_voltage(_old_voltage)
            success = False
        return success

    def apply_positions_array(self, idx: int = 0):
        """Set mirror positions from stored array.

        Used in nD acquisitions where each "position" has a unique correction.

        Parameters
        ----------
        idx: int, default = 0
            position index to use
        """
        try:
            _old_voltage = self.current_voltage.copy()
            self.set_modal_coefficients(self.positions_modal_array[idx, :])
            self._update_current_state()
            success = self._verify_current_voltage(self.positions_voltage_array[idx])
        except Exception as e:
            print(f"Exception occured in applying POSITION ARRAY idx={idx}:\n {e}")
            self.set_mirror_voltage(_old_voltage)  
        
        return success

    def apply_wfc_voltage_file(self, wfc_path: Path):
        """ Set mirror to the positions given by wfc file
        WARNING: THIS METHOD BREAKS THE MODAL COEFF REFERENCE STATE
                 ONLY USE FOR LOADING SAVED MIRROR STATES
        """
        if ".wcs" not in wfc_path.name:
            print("WFC file not provided")
            return False
        else:
            # Update the modal coeff model and get wfc voltage state
            self.modal_coeff.set_coefs_values(
                np.zeros(self._n_modes), index_array=self._mode_indices
            )
            wfc_voltage = np.array(
                self.wfc.get_positions_from_file(str(wfc_path)),
                dtype=np.float32,
                copy=True,
            )
            success = self.set_mirror_voltage(wfc_voltage)
            self._update_current_state()
            
            return success
    
    # -------------------------------------------------#
    # Methods for changing mirror positions
    # -------------------------------------------------#

    def set_mirror_voltage(self, positions: NDArray):
        """Set mirror actuator voltages.

        Parameters
        ----------
        positions : NDArray
            1d array of actuators
        Returns
        -------
        success
            bool: whether mirror voltages where applied
        """
        if self._validate_voltage(positions):
            try:
                # Apply mirror voltages
                self.wfc.move_to_absolute_positions(positions)
                if self._wait_after_move:
                    time.sleep(self._mirror_settle_ms * 10**-3)
                self._update_current_state()
                success = self._verify_current_voltage(positions)
            except Exception as e:
                success = False
                print(
                    "------- AOmirror ERROR -------",
                    f"\nException in setting mirror voltage:\n{e}"
                )
        else:
            success = False
            
        return success

    def set_modal_coefficients(self, amps: ArrayLike):
        """Set mirror voltages from Zernike modal coefficients.
        Zernike modes are relative to system flat voltages!

        Parameters
        ----------
        amps : NDArray
            Flatten array of Zernike mode amplitudes
        """
        # Validate amplitude array
        amps = np.asarray(amps, dtype=np.float32)
        if amps.shape != (self._n_modes,):
            raise ValueError("amps must be shape (n_modes,)")
            
        # update modal_coef model
        self.modal_coeff.set_coefs_values(
            coef_array=amps, index_array=self._mode_indices
        )
        # Create a new haso_slope from the new modal_coefs
        haso_slopes = wkpy.HasoSlopes(
            modalcoef=self.modal_coeff,
            config_file_path=str(self._haso_config_file_path),
        )
        # Calculate the voltage delta to achieve the desired modalcoef
        deltas = self.corr_data_manager.compute_delta_command_from_delta_slopes(
            delta_slopes=haso_slopes
        )
        # New voltage relative reference voltage
        new_voltage = self._ref_voltage + np.asarray(deltas)
        success = self.set_mirror_voltage(new_voltage)

        if not success:
            # Revert to the modal_coef to the last saved state and update
            self.modal_coeff.set_coefs_values(
                coef_array=self.current_coeffs, index_array=self._mode_indices
            )
        # Update the current state
        self._update_current_state()
        return success

    # -------------------------------------------------#
    # Methods for saving/visualizing mirror positions
    # -------------------------------------------------#

    def get_current_phase(self):
        """Get the phase from current modal_coeffs
        # TODO: Validate method
        """
        phase_object = wkpy.Phase(modalcoeff=self.modal_coeff, filter=[])
        phase_stats = phase_object.get_statistics()
        phase = {
            "rms": phase_stats[0],
            "pv": phase_stats[1],
            "max": phase_stats[2],
            "min": phase_stats[3],
            "phase": phase_object.get_data()[0],
        }
        del phase_object

        return phase

    def save_current_state(self, prefix: str = "current"):
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
                raise ValueError("Cannot save current state, missing output path")
            else:
                # save wfc compatible file
                actuator_save_path = self._output_path / Path(
                    f"{prefix}_wfc_voltage.wcs"
                )
                self.wfc.save_current_positions_to_file(str(actuator_save_path))

                # save current state and last optimized positions to disk
                metadata_path = self._output_path / Path(
                    f"{prefix}_aoMirror_state.json"
                )
                metadata = {
                    "mode_names": self.mode_names,
                    "optimized_state": {
                        "mode_amplitudes": self.optimized_modal_coefs.tolist(),
                        "volt_amplitudes": self.optimized_voltage.tolist(),
                    },
                    "current_state": {
                        "mode_amplitudes": self.current_coeffs.tolist(),
                        "volt_amplitudes": self.current_voltage.tolist(),
                    },
                }
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f)

                # Save dictionary of current coefficients where the key is the mode name
                modal_save_path = self._output_path / Path(
                    f"{prefix}_aoMirror_zernike.json"
                )
                mode_dict = self.current_zernikes

                with open(modal_save_path, "w") as f:
                    json.dump(mode_dict, f)

        except Exception as e:
            print(f"------- AOmirror -------\nFailed to save current state!\n{e}")

    def save_positions_array(self, prefix: str = "stage_position"):
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

            positions_file_path = self._output_path / Path(
                f"{prefix}_mode_amplitude.json"
            )
            wfc_coeffs_list = self.positions_modal_array.tolist()
            with open(positions_file_path, "w") as f:
                json.dump(wfc_coeffs_list, f)
        else:
            pass

    def load_positions_array(self, prefix: str = "stage_position"):
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

    M = np.zeros((8, 8))
    M[:, :] = None
    M[2:6, 0] = v[:4]
    M[1:7, 1] = v[4:10]
    M[:, 2] = v[10:18]
    M[:, 3] = v[18:26]
    M[:, 4] = v[26:34]
    M[:, 5] = v[34:42]
    M[1:7, 6] = v[42:48]
    M[2:6, 7] = v[48:52]

    return M


def plotDM(
    cmd: NDArray,
    title: str = "",
    cmap: str = "jet",
    vmin: float = -0.25,
    vmax: float = 0.25,
    save_dir_path: Path = None,
    show_fig: bool = False,
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

    fig, ax = plt.subplots(1, 1)
    valmax = np.nanmax(cmd)
    valmin = np.nanmin(cmd)
    im = ax.imshow(
        DM_voltage_to_map(cmd), vmin=vmin, vmax=vmax, interpolation="nearest", cmap=cmap
    )
    ax.text(
        0,
        -1,
        title
        + "\n min="
        + "{:1.2f}".format(valmin)
        + ", max="
        + "{:1.2f}".format(valmax)
        + " V",
        fontsize=12,
    )

    plt.colorbar(im)

    if save_dir_path:
        fig.savefig(save_dir_path / Path("mirror_positions.png"))
    if show_fig:
        plt.show()
