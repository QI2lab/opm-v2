"""
QtWidget for setting the OPM configuration.

2025/03/07 Sheppard: Initial setup
"""
import sys
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (QWidget, QApplication, QDoubleSpinBox, 
                             QHBoxLayout, QVBoxLayout, QGroupBox, 
                             QLabel, QComboBox, QSlider, QCheckBox, QSpinBox)
from pathlib import Path
import json


class OPMSettings(QWidget):
    
    settings_changed = pyqtSignal()
    
    def __init__(self,
                 config_path: Path):

        super().__init__()
        
        self.config_path = config_path
        with open(self.config_path, "r") as config_file:
            config = json.load(config_file)
        
        self.config = config
        self.widgets = {}       
        self.create_ui()
        self.update_config()


    def create_ui(self):
        #--------------------------------------------------------------------#
        #--------------------------------------------------------------------#
        # Create a group for setting the AO configuration
        #--------------------------------------------------------------------#
        #--------------------------------------------------------------------#
        
        self.group_ao_settings = QGroupBox("AO optimization settings")
        self.layout_ao_settings = QVBoxLayout()
        
        #--------------------------------------------------------------------#
        # Create individual widgets for each setting in config
        # - connect widgets to update function that writes to the config file.
        #--------------------------------------------------------------------#
        
        #--------------------------------------------------------------------#
        self.cmbx_ao_metric =  QComboBox()
        self.cmbx_ao_metric.addItems(self.config["OPM"]["ao_metrics"])
        self.cmbx_ao_metric.setFixedWidth(80)
        self.cmbx_ao_metric.currentIndexChanged.connect(self.update_config)
        
        self.layout_ao_metric = QHBoxLayout()
        self.layout_ao_metric.addWidget(QLabel("Metric key:"))
        self.layout_ao_metric.addWidget(self.cmbx_ao_metric)
        
        self.cmbx_ao_active_channel =  QComboBox()
        self.cmbx_ao_active_channel.addItems(self.config["OPM"]["channel_ids"])
        self.cmbx_ao_active_channel.setFixedWidth(80)
        self.cmbx_ao_active_channel.currentIndexChanged.connect(self.update_config)
        
        self.layout_ao_active_channel = QHBoxLayout()
        self.layout_ao_active_channel.addWidget(QLabel("Active Channel"))
        self.layout_ao_active_channel.addWidget(self.cmbx_ao_active_channel)
        
        #--------------------------------------------------------------------#
        self.sldr_active_channel_power = QSlider(Qt.Orientation.Horizontal)
        self.sldr_active_channel_power.setMinimum(0)   
        self.sldr_active_channel_power.setMaximum(100)
        self.sldr_active_channel_power.setValue(int(self.config["acq_config"]["AO"]["active_channel_power"]))
        self.sldr_active_channel_power.setTickInterval(1)
        self.sldr_active_channel_power.valueChanged.connect(self.update_ao_active_power_spbx)
            
        self.spbx_active_channel_power = QDoubleSpinBox()
        self.spbx_active_channel_power.setRange(0, 100)
        self.spbx_active_channel_power.setDecimals(1)
        self.spbx_active_channel_power.setValue(self.config["acq_config"]["AO"]["active_channel_power"])
        self.spbx_active_channel_power.setFixedWidth(80)
        self.spbx_active_channel_power.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_active_channel_power.valueChanged.connect(self.update_ao_active_power_slider)
        self.spbx_active_channel_power.valueChanged.connect(self.update_config)
        
        self.layout_active_channel_power = QHBoxLayout()
        self.layout_active_channel_power.addWidget(QLabel("Laser power:"))
        self.layout_active_channel_power.addWidget(self.sldr_active_channel_power)
        self.layout_active_channel_power.addWidget(self.spbx_active_channel_power)
        self.layout_active_channel_power.addWidget(QLabel("%"))
                
        #--------------------------------------------------------------------#
        self.sldr_ao_exposure = QSlider(Qt.Orientation.Horizontal)
        self.sldr_ao_exposure.setMinimum(1)   
        self.sldr_ao_exposure.setMaximum(1000)
        self.sldr_ao_exposure.setValue(int(self.config["acq_config"]["AO"]["exposure_ms"]))
        self.sldr_ao_exposure.setTickInterval(1)
        self.sldr_ao_exposure.valueChanged.connect(self.update_ao_exposure_spbx)
        
        self.spbx_ao_exposure = QDoubleSpinBox()
        self.spbx_ao_exposure.setRange(50, 1000)
        self.spbx_ao_exposure.setDecimals(0)
        self.spbx_ao_exposure.setValue(self.config["acq_config"]["AO"]["exposure_ms"])
        self.spbx_ao_exposure.setFixedWidth(80)
        self.spbx_ao_exposure.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_ao_exposure.valueChanged.connect(self.update_config)
        self.sldr_ao_exposure.valueChanged.connect(self.update_ao_exposure_slider)
        
        self.layout_ao_exposure = QHBoxLayout()
        self.layout_ao_exposure.addWidget(QLabel("Exposure:"))
        self.layout_ao_exposure.addWidget(self.sldr_ao_exposure)
        self.layout_ao_exposure.addWidget(self.spbx_ao_exposure)
        self.layout_ao_exposure.addWidget(QLabel("ms"))

        #--------------------------------------------------------------------#
        self.sldr_ao_mirror_range = QSlider(Qt.Orientation.Horizontal)
        self.sldr_ao_mirror_range.setMinimum(1)   
        self.sldr_ao_mirror_range.setMaximum(1000)
        self.sldr_ao_mirror_range.setValue(int(self.config["acq_config"]["AO"]["image_mirror_range_um"]))
        self.sldr_ao_mirror_range.setTickInterval(1)
        self.sldr_ao_mirror_range.valueChanged.connect(self.update_ao_mirror_range_spbx)
        
        self.spbx_ao_mirror_range =  QDoubleSpinBox()  
        self.spbx_ao_mirror_range.setDecimals(1)
        self.spbx_ao_mirror_range.setRange(0, 250)
        self.spbx_ao_mirror_range.setSingleStep(1)
        self.spbx_ao_mirror_range.setFixedWidth(80)
        self.spbx_ao_mirror_range.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_ao_mirror_range.setValue(self.config["acq_config"]["AO"]["image_mirror_range_um"])
        self.spbx_ao_mirror_range.valueChanged.connect(self.update_ao_mirror_range_slider)
        self.spbx_ao_mirror_range.valueChanged.connect(self.update_config)
        
        self.layout_ao_mirror_range = QHBoxLayout()
        self.layout_ao_mirror_range.addWidget(QLabel("Projection/Image mirror range:"))
        self.layout_ao_mirror_range.addStretch()
        self.layout_ao_mirror_range.addWidget(self.spbx_ao_mirror_range)
        self.layout_ao_mirror_range.addWidget(QLabel("\u00B5m"))
        
        #--------------------------------------------------------------------#
        self.cmbx_ao_mode =  QComboBox()
        self.cmbx_ao_mode.addItems(self.config["OPM"]["ao_modes"])
        self.cmbx_ao_mode.setFixedWidth(125)
        self.cmbx_ao_mode.currentIndexChanged.connect(self.update_config)
        
        self.layout_ao_mode = QHBoxLayout()
        self.layout_ao_mode.addWidget(QLabel("AO mode:"))
        self.layout_ao_mode.addWidget(self.cmbx_ao_mode)
                
        #--------------------------------------------------------------------#
        self.cmbx_ao_metric =  QComboBox()
        self.cmbx_ao_metric.addItems(self.config["OPM"]["ao_metrics"])
        self.cmbx_ao_metric.setFixedWidth(80)
        self.cmbx_ao_metric.currentIndexChanged.connect(self.update_config)
        
        self.layout_ao_metric = QHBoxLayout()
        self.layout_ao_metric.addWidget(QLabel("Metric:"))
        self.layout_ao_metric.addWidget(self.cmbx_ao_metric)
        
        #--------------------------------------------------------------------#
        self.spbx_num_iterations =  QSpinBox()
        self.spbx_num_iterations.setRange(1, 10)
        self.spbx_num_iterations.setValue(self.config["acq_config"]["AO"]["num_iterations"])
        self.spbx_num_iterations.setFixedWidth(80)
        self.spbx_num_iterations.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_num_iterations.valueChanged.connect(self.update_config)
        
        self.layout_num_iterations = QHBoxLayout()
        self.layout_num_iterations.addWidget(QLabel("Number of iterations:"))
        self.layout_num_iterations.addWidget(self.spbx_num_iterations)
        
        #--------------------------------------------------------------------#
        self.spbx_mode_delta =  QDoubleSpinBox()
        self.spbx_mode_delta.setDecimals(2)
        self.spbx_mode_delta.setRange(0, 1.0)
        self.spbx_mode_delta.setSingleStep(0.010)
        self.spbx_mode_delta.setFixedWidth(80)
        self.spbx_mode_delta.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_mode_delta.setValue(self.config["acq_config"]["AO"]["mode_delta"])
        self.spbx_mode_delta.valueChanged.connect(self.update_config)
        
        self.layout_mode_delta = QHBoxLayout()
        self.layout_mode_delta.addWidget(QLabel("Initial mode delta:"))
        self.layout_mode_delta.addWidget(self.spbx_mode_delta)
        
        #--------------------------------------------------------------------#
        self.spbx_mode_alpha =  QDoubleSpinBox()
        self.spbx_mode_alpha.setDecimals(2)
        self.spbx_mode_alpha.setRange(0, 1.0)
        self.spbx_mode_alpha.setSingleStep(0.1)
        self.spbx_mode_alpha.setFixedWidth(80)
        self.spbx_mode_alpha.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_mode_alpha.setValue(self.config["acq_config"]["AO"]["mode_alpha"])
        self.spbx_mode_alpha.valueChanged.connect(self.update_config)
        
        self.layout_mode_alpha = QHBoxLayout()
        self.layout_mode_alpha.addWidget(QLabel("Mode range alpha:"))
        self.layout_mode_alpha.addWidget(self.spbx_mode_alpha)
        
        #--------------------------------------------------------------------#
        # setup sub layouts in ao settings
        self.layout_ao_settings.addLayout(self.layout_ao_mode)
        self.layout_ao_settings.addLayout(self.layout_ao_metric)
        self.layout_ao_settings.addLayout(self.layout_ao_active_channel)
        self.layout_ao_settings.addLayout(self.layout_active_channel_power)
        self.layout_ao_settings.addLayout(self.layout_ao_exposure)     
        self.layout_ao_settings.addLayout(self.layout_ao_mirror_range)
        self.layout_ao_settings.addLayout(self.layout_num_iterations)
        self.layout_ao_settings.addLayout(self.layout_mode_delta)
        self.layout_ao_settings.addLayout(self.layout_mode_alpha)
        self.group_ao_settings.setLayout(self.layout_ao_settings)

        self.widgets.update(
            {"AO":{
                    "metric":self.cmbx_ao_metric,
                    "mode_delta": self.spbx_mode_delta,
                    "mode_alpha": self.spbx_mode_alpha,
                    "num_iterations": self.spbx_num_iterations,
                    "image_mirror_range_um": self.spbx_ao_mirror_range,
                    "active_channel_id": self.cmbx_ao_active_channel,
                    "active_channel_power": self.spbx_active_channel_power,
                    "exposure_ms": self.spbx_ao_exposure,
                    "ao_mode": self.cmbx_ao_mode
                    }  
                }
        )
        
        #--------------------------------------------------------------------#
        #--------------------------------------------------------------------#
        # Create group for setting the imaging settings
        # - stage scan settings
        # - image mirror scan settings
        # - projection settings
        #--------------------------------------------------------------------#
        #--------------------------------------------------------------------#

        self.group_imaging_settings = QGroupBox("OPM imaging settings")
        self.layout_imaging_settings = QVBoxLayout()
        
        #--------------------------------------------------------------------#
        # Configure channel properties
        self.cmbx_opm_mode =  QComboBox()
        self.cmbx_opm_mode.addItems(self.config["OPM"]["imaging_modes"])
        self.cmbx_opm_mode.setFixedWidth(100)
        self.cmbx_opm_mode.currentIndexChanged.connect(self.update_config)
        self.layout_opm_mode = QHBoxLayout()
        self.layout_opm_mode.addWidget(QLabel("OPM mode:"))
        self.layout_opm_mode.addWidget(self.cmbx_opm_mode)        
        
        #--------------------------------------------------------------------#
        # Configure channel properties
        self.cmbx_o2o3_mode =  QComboBox()
        self.cmbx_o2o3_mode.addItems(self.config["OPM"]["autofocus_frequencies"])
        self.cmbx_o2o3_mode.setFixedWidth(125)
        self.cmbx_o2o3_mode.currentIndexChanged.connect(self.update_config)
        self.layout_o2o3_mode = QHBoxLayout()
        self.layout_o2o3_mode.addWidget(QLabel("O2O3 Autofocus:"))
        self.layout_o2o3_mode.addWidget(self.cmbx_o2o3_mode)
                
        #--------------------------------------------------------------------#
        # Configure channel properties
        self.cmbx_fluidics_mode =  QComboBox()
        self.cmbx_fluidics_mode.addItems(["none", "1", "2", "8", "16", "22"])
        self.cmbx_fluidics_mode.setFixedWidth(125)
        self.cmbx_fluidics_mode.currentIndexChanged.connect(self.update_config)
        self.layout_fluidics_mode = QHBoxLayout()
        self.layout_fluidics_mode.addWidget(QLabel("Fluidics rounds:"))
        self.layout_fluidics_mode.addWidget(self.cmbx_fluidics_mode)
        
        #--------------------------------------------------------------------#
        # Configure channel properties
        self.cmbx_laser_blanking =  QComboBox()
        self.cmbx_laser_blanking.addItems(["on", "off"])
        self.cmbx_laser_blanking.setFixedWidth(125)
        self.cmbx_laser_blanking.currentIndexChanged.connect(self.update_config)
        self.layout_laser_blanking = QHBoxLayout()
        self.layout_laser_blanking.addWidget(QLabel("Laser blanking:"))
        self.layout_laser_blanking.addWidget(self.cmbx_laser_blanking)
        
        #--------------------------------------------------------------------#
        self.sldr_405_power = QSlider(Qt.Orientation.Horizontal)
        self.sldr_405_power.setMinimum(0)   
        self.sldr_405_power.setMaximum(100)
        self.sldr_405_power.setValue(0)
        self.sldr_405_power.setTickInterval(1)
        self.sldr_405_power.valueChanged.connect(self.update_405_spbx)
            
        self.spbx_405_power = QDoubleSpinBox()
        self.spbx_405_power.setRange(0, 100)
        self.spbx_405_power.setDecimals(1)
        self.spbx_405_power.setValue(0)
        self.spbx_405_power.setFixedWidth(80)
        self.spbx_405_power.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_405_power.valueChanged.connect(self.update_405_slider)
        self.spbx_405_power.valueChanged.connect(self.update_405_state)
        
        self.spbx_405_exp = QDoubleSpinBox()
        self.spbx_405_exp.setRange(1, 1000)
        self.spbx_405_exp.setDecimals(0)
        self.spbx_405_exp.setValue(50)
        self.spbx_405_exp.setFixedWidth(80)
        self.spbx_405_exp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_405_exp.valueChanged.connect(self.update_405_state)
        
        self.chx_405_state = QCheckBox()
        self.chx_405_state.setChecked(False)
        self.chx_405_state.checkStateChanged.connect(self.update_405_state)
        
        self.layout_405 = QHBoxLayout()
        self.layout_405.addWidget(QLabel("405nm:"))
        self.layout_405.addWidget(self.sldr_405_power)
        self.layout_405.addWidget(self.spbx_405_power)
        self.layout_405.addWidget(QLabel("%"))
        self.layout_405.addWidget(self.spbx_405_exp)
        self.layout_405.addWidget(QLabel("ms"))
        self.layout_405.addWidget(self.chx_405_state)
        
        #--------------------------------------------------------------------#
        self.sldr_488_power = QSlider(Qt.Orientation.Horizontal)
        self.sldr_488_power.setMinimum(0)   
        self.sldr_488_power.setMaximum(100)
        self.sldr_488_power.setValue(0)
        self.sldr_488_power.setTickInterval(1)
        self.sldr_488_power.valueChanged.connect(self.update_488_spbx)
            
        self.spbx_488_power = QDoubleSpinBox()
        self.spbx_488_power.setRange(0, 100)
        self.spbx_488_power.setDecimals(1)
        self.spbx_488_power.setValue(0)
        self.spbx_488_power.setFixedWidth(80)
        self.spbx_488_power.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_488_power.valueChanged.connect(self.update_488_slider)
        self.spbx_488_power.valueChanged.connect(self.update_488_state)
        
        self.spbx_488_exp = QDoubleSpinBox()
        self.spbx_488_exp.setRange(1, 1000)
        self.spbx_488_exp.setDecimals(0)
        self.spbx_488_exp.setValue(50)
        self.spbx_488_exp.setFixedWidth(80)
        self.spbx_488_exp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_488_exp.valueChanged.connect(self.update_488_state)
        
        self.chx_488_state = QCheckBox()
        self.chx_488_state.setChecked(False)
        self.chx_488_state.checkStateChanged.connect(self.update_488_state)
        
        self.layout_488 = QHBoxLayout()
        self.layout_488.addWidget(QLabel("488nm:"))
        self.layout_488.addWidget(self.sldr_488_power)
        self.layout_488.addWidget(self.spbx_488_power)
        self.layout_488.addWidget(QLabel("%"))
        self.layout_488.addWidget(self.spbx_488_exp)
        self.layout_488.addWidget(QLabel("ms"))
        self.layout_488.addWidget(self.chx_488_state)
        
        #--------------------------------------------------------------------#
        self.sldr_561_power = QSlider(Qt.Orientation.Horizontal)
        self.sldr_561_power.setMinimum(0)   
        self.sldr_561_power.setMaximum(100)
        self.sldr_561_power.setValue(0)
        self.sldr_561_power.setTickInterval(1)
        self.sldr_561_power.valueChanged.connect(self.update_561_spbx)
            
        self.spbx_561_power = QDoubleSpinBox()
        self.spbx_561_power.setRange(0, 100)
        self.spbx_561_power.setDecimals(1)
        self.spbx_561_power.setValue(0)
        self.spbx_561_power.setFixedWidth(80)
        self.spbx_561_power.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_561_power.valueChanged.connect(self.update_ao_active_power_slider)
        self.spbx_561_power.valueChanged.connect(self.update_561_state)
        
        self.spbx_561_exp = QDoubleSpinBox()
        self.spbx_561_exp.setRange(1, 1000)
        self.spbx_561_exp.setDecimals(0)
        self.spbx_561_exp.setValue(50)
        self.spbx_561_exp.setFixedWidth(80)
        self.spbx_561_exp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_561_exp.valueChanged.connect(self.update_561_state)
        
        self.chx_561_state = QCheckBox()
        self.chx_561_state.setChecked(False)
        self.chx_561_state.checkStateChanged.connect(self.update_561_state)
        
        self.layout_561 = QHBoxLayout()
        self.layout_561.addWidget(QLabel("561nm:"))
        self.layout_561.addWidget(self.sldr_561_power)
        self.layout_561.addWidget(self.spbx_561_power)
        self.layout_561.addWidget(QLabel("%"))
        self.layout_561.addWidget(self.spbx_561_exp)
        self.layout_561.addWidget(QLabel("ms"))
        self.layout_561.addWidget(self.chx_561_state)
        
        #--------------------------------------------------------------------#
        self.sldr_638_power = QSlider(Qt.Orientation.Horizontal)
        self.sldr_638_power.setMinimum(0)   
        self.sldr_638_power.setMaximum(100)
        self.sldr_638_power.setValue(0)
        self.sldr_638_power.setTickInterval(1)
        self.sldr_638_power.valueChanged.connect(self.update_638_spbx)
            
        self.spbx_638_power = QDoubleSpinBox()
        self.spbx_638_power.setRange(0, 100)
        self.spbx_638_power.setDecimals(1)
        self.spbx_638_power.setValue(0)
        self.spbx_638_power.setFixedWidth(80)
        self.spbx_638_power.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_638_power.valueChanged.connect(self.update_638_slider)
        self.spbx_638_power.valueChanged.connect(self.update_638_state)
        
        self.spbx_638_exp = QDoubleSpinBox()
        self.spbx_638_exp.setRange(1, 1000)
        self.spbx_638_exp.setDecimals(0)
        self.spbx_638_exp.setValue(50)
        self.spbx_638_exp.setFixedWidth(80)
        self.spbx_638_exp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_638_exp.valueChanged.connect(self.update_638_state)
        
        self.chx_638_state = QCheckBox()
        self.chx_638_state.setChecked(False)
        self.chx_638_state.checkStateChanged.connect(self.update_638_state)
        
        self.layout_638 = QHBoxLayout()
        self.layout_638.addWidget(QLabel("638nm:"))
        self.layout_638.addWidget(self.sldr_638_power)
        self.layout_638.addWidget(self.spbx_638_power)
        self.layout_638.addWidget(QLabel("%"))
        self.layout_638.addWidget(self.spbx_638_exp)
        self.layout_638.addWidget(QLabel("ms"))
        self.layout_638.addWidget(self.chx_638_state)
        
        #--------------------------------------------------------------------#
        self.sldr_705_power = QSlider(Qt.Orientation.Horizontal)
        self.sldr_705_power.setMinimum(0)   
        self.sldr_705_power.setMaximum(100)
        self.sldr_705_power.setValue(0)
        self.sldr_705_power.setTickInterval(1)
        self.sldr_705_power.valueChanged.connect(self.update_705_spbx)
            
        self.spbx_705_power = QDoubleSpinBox()
        self.spbx_705_power.setRange(0, 100)
        self.spbx_705_power.setDecimals(1)
        self.spbx_705_power.setValue(0)
        self.spbx_705_power.setFixedWidth(80)
        self.spbx_705_power.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_705_power.valueChanged.connect(self.update_705_slider)
        self.spbx_705_power.valueChanged.connect(self.update_705_state)
        
        self.spbx_705_exp = QDoubleSpinBox()
        self.spbx_705_exp.setRange(1, 1000)
        self.spbx_705_exp.setDecimals(0)
        self.spbx_705_exp.setValue(50)
        self.spbx_705_exp.setFixedWidth(80)
        self.spbx_705_exp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_705_exp.valueChanged.connect(self.update_705_state)
        
        self.chx_705_state = QCheckBox()
        self.chx_705_state.setChecked(False)
        self.chx_705_state.checkStateChanged.connect(self.update_705_state)
        
        self.layout_705 = QHBoxLayout()
        self.layout_705.addWidget(QLabel("705nm:"))
        self.layout_705.addWidget(self.sldr_705_power)
        self.layout_705.addWidget(self.spbx_705_power)
        self.layout_705.addWidget(QLabel("%"))
        self.layout_705.addWidget(self.spbx_705_exp)
        self.layout_705.addWidget(QLabel("ms"))
        self.layout_705.addWidget(self.chx_705_state)
        
        #--------------------------------------------------------------------#
        self.sldr_mirror_image_range = QSlider(Qt.Orientation.Horizontal)
        self.sldr_mirror_image_range.setMinimum(1)   
        self.sldr_mirror_image_range.setMaximum(250)
        self.sldr_mirror_image_range.setValue(int(self.config["acq_config"]["mirror_scan"]["image_mirror_range_um"]))
        self.sldr_mirror_image_range.setTickInterval(1)
        self.sldr_mirror_image_range.valueChanged.connect(self.update_mirror_image_range_spbx)
        
        self.spbx_mirror_image_range =  QDoubleSpinBox()  
        self.spbx_mirror_image_range.setDecimals(1)
        self.spbx_mirror_image_range.setRange(0, 250)
        self.spbx_mirror_image_range.setSingleStep(1)
        self.spbx_mirror_image_range.setFixedWidth(80)
        self.spbx_mirror_image_range.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_mirror_image_range.setValue(self.config["acq_config"]["mirror_scan"]["image_mirror_range_um"])
        self.spbx_mirror_image_range.valueChanged.connect(self.update_mirror_image_range_slider)
        self.spbx_mirror_image_range.valueChanged.connect(self.update_config)
        
        self.layout_mirror_image_range = QHBoxLayout()
        self.layout_mirror_image_range.addWidget(QLabel("Mirror scan image range:"))
        self.layout_mirror_image_range.addWidget(self.sldr_mirror_image_range)
        self.layout_mirror_image_range.addWidget(self.spbx_mirror_image_range)
        self.layout_mirror_image_range.addWidget(QLabel("\u00B5m"))

        #--------------------------------------------------------------------#
        self.sldr_mirror_image_step = QSlider(Qt.Orientation.Horizontal)
        self.sldr_mirror_image_step.setMinimum(0)   
        self.sldr_mirror_image_step.setMaximum(200)
        self.sldr_mirror_image_step.setValue(int(self.config["acq_config"]["mirror_scan"]["image_mirror_step_size_um"]*100))
        self.sldr_mirror_image_step.setTickInterval(1)
        self.sldr_mirror_image_step.valueChanged.connect(self.update_mirror_image_step_spbx)
        
        self.spbx_mirror_image_step =  QDoubleSpinBox()  
        self.spbx_mirror_image_step.setDecimals(2)
        self.spbx_mirror_image_step.setRange(0.05, 2)
        self.spbx_mirror_image_step.setSingleStep(0.1)
        self.spbx_mirror_image_step.setFixedWidth(80)
        self.spbx_mirror_image_step.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_mirror_image_step.setValue(self.config["acq_config"]["mirror_scan"]["image_mirror_step_size_um"])
        self.spbx_mirror_image_range.valueChanged.connect(self.update_mirror_image_step_slider)
        self.spbx_mirror_image_step.valueChanged.connect(self.update_config)
        
        self.layout_mirror_image_step = QHBoxLayout()
        self.layout_mirror_image_step.addWidget(QLabel("Mirror scan step size:"))
        self.layout_mirror_image_step.addWidget(self.sldr_mirror_image_step)
        self.layout_mirror_image_step.addWidget(self.spbx_mirror_image_step)
        self.layout_mirror_image_step.addWidget(QLabel("\u00B5m"))
        
        #--------------------------------------------------------------------#
        self.sldr_stage_image_range = QSlider(Qt.Orientation.Horizontal)
        self.sldr_stage_image_range.setMinimum(0)   
        self.sldr_stage_image_range.setMaximum(250)
        self.sldr_stage_image_range.setValue(int(self.config["acq_config"]["stage_scan"]["stage_scan_range_um"]))
        # self.sldr_stage_image_range.setTickInterval(0.1)
        self.sldr_stage_image_range.valueChanged.connect(self.update_stage_image_range_spbx)
        
        self.spbx_stage_image_range =  QDoubleSpinBox()  
        self.spbx_stage_image_range.setDecimals(1)
        self.spbx_stage_image_range.setRange(0, 1000)
        self.spbx_stage_image_range.setSingleStep(1)
        self.spbx_stage_image_range.setFixedWidth(80)
        self.spbx_stage_image_range.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_stage_image_range.setValue(self.config["acq_config"]["stage_scan"]["stage_scan_range_um"])
        self.spbx_mirror_image_range.valueChanged.connect(self.update_stage_image_range_slider)
        self.spbx_stage_image_range.valueChanged.connect(self.update_config)
        
        self.layout_stage_image_range = QHBoxLayout()
        self.layout_stage_image_range.addWidget(QLabel("Stage scan range:"))
        self.layout_stage_image_range.addWidget(self.sldr_stage_image_range)
        self.layout_stage_image_range.addWidget(self.spbx_stage_image_range)
        self.layout_stage_image_range.addWidget(QLabel("\u00B5m"))

        #--------------------------------------------------------------------#
        self.spbx_stage_slope =  QDoubleSpinBox()  
        self.spbx_stage_slope.setDecimals(3)
        self.spbx_stage_slope.setRange(0, 0.10)
        self.spbx_stage_slope.setSingleStep(0.001)
        self.spbx_stage_slope.setFixedWidth(80)
        self.spbx_stage_slope.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_stage_slope.setValue(self.config["acq_config"]["stage_scan"]["coverslip_slope"])
        self.spbx_mirror_image_range.valueChanged.connect(self.update_stage_image_range_slider)
        self.spbx_stage_slope.valueChanged.connect(self.update_config)
        
        self.layout_stage_slope = QHBoxLayout()
        self.layout_stage_slope.addWidget(QLabel("Coverslip slope (rise/run):"))
        self.layout_stage_slope.addStretch()
        self.layout_stage_slope.addWidget(self.spbx_stage_slope)
        # self.layout_stage_slope.addWidget(QLabel("\u00B5m"))
        
        #--------------------------------------------------------------------#
        self.sldr_proj_image_range = QSlider(Qt.Orientation.Horizontal)
        self.sldr_proj_image_range.setMinimum(0)   
        self.sldr_proj_image_range.setMaximum(250)
        self.sldr_proj_image_range.setValue(int(self.config["acq_config"]["projection_scan"]["image_mirror_range_um"]))
        # self.sldr_proj_image_range.setTickInterval(0.1)
        self.sldr_proj_image_range.valueChanged.connect(self.update_proj_image_range_spbx)
        
        self.spbx_proj_image_range =  QDoubleSpinBox()  
        self.spbx_proj_image_range.setDecimals(1)
        self.spbx_proj_image_range.setRange(0, 250)
        self.spbx_proj_image_range.setSingleStep(1)
        self.spbx_proj_image_range.setFixedWidth(80)
        self.spbx_proj_image_range.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_proj_image_range.setValue(self.config["acq_config"]["projection_scan"]["image_mirror_range_um"])
        self.spbx_mirror_image_range.valueChanged.connect(self.update_proj_image_range_slider)
        self.spbx_proj_image_range.valueChanged.connect(self.update_config)
        
        self.layout_proj_image_range = QHBoxLayout()
        self.layout_proj_image_range.addWidget(QLabel("Projection image range:"))
        self.layout_proj_image_range.addWidget(self.sldr_proj_image_range)
        self.layout_proj_image_range.addWidget(self.spbx_proj_image_range)
        self.layout_proj_image_range.addWidget(QLabel("\u00B5m"))
        
        #--------------------------------------------------------------------#
        self.spbx_roi_crop_x =  QDoubleSpinBox()  
        self.spbx_roi_crop_x.setDecimals(0)
        self.spbx_roi_crop_x.setRange(0, 2000)
        self.spbx_roi_crop_x.setSingleStep(1)
        self.spbx_roi_crop_x.setFixedWidth(80)
        self.spbx_roi_crop_x.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_roi_crop_x.setValue(self.config["acq_config"]["camera_roi"]["crop_x"])
        self.spbx_roi_crop_x.valueChanged.connect(self.update_config)
        
        self.spbx_roi_crop_y =  QDoubleSpinBox()  
        self.spbx_roi_crop_y.setDecimals(0)
        self.spbx_roi_crop_y.setRange(0, 2000)
        self.spbx_roi_crop_y.setSingleStep(1)
        self.spbx_roi_crop_y.setFixedWidth(80)
        self.spbx_roi_crop_y.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_roi_crop_y.setValue(self.config["acq_config"]["camera_roi"]["crop_y"])
        self.spbx_roi_crop_y.valueChanged.connect(self.update_config)
 
        
        self.spbx_roi_center_x =  QDoubleSpinBox()  
        self.spbx_roi_center_x.setDecimals(0)
        self.spbx_roi_center_x.setRange(0, 2000)
        self.spbx_roi_center_x.setSingleStep(1)
        self.spbx_roi_center_x.setFixedWidth(80)
        self.spbx_roi_center_x.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_roi_center_x.setValue(self.config["acq_config"]["camera_roi"]["center_x"])
        self.spbx_roi_center_x.valueChanged.connect(self.update_config)
        
        self.spbx_roi_center_y =  QDoubleSpinBox()  
        self.spbx_roi_center_y.setDecimals(0)
        self.spbx_roi_center_y.setRange(0, 2000)
        self.spbx_roi_center_y.setSingleStep(1)
        self.spbx_roi_center_y.setFixedWidth(80)
        self.spbx_roi_center_y.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spbx_roi_center_y.setValue(self.config["acq_config"]["camera_roi"]["center_y"])
        self.spbx_roi_center_y.valueChanged.connect(self.update_config)

        self.layout_roi_x = QHBoxLayout()
        self.layout_roi_x.addWidget(QLabel("ROI center x:"))
        self.layout_roi_x.addWidget(self.spbx_roi_center_x)
        self.layout_roi_x.addWidget(QLabel("ROI crop x:"))
        self.layout_roi_x.addWidget(self.spbx_roi_crop_x)
        self.layout_roi_y = QHBoxLayout()
        self.layout_roi_y.addWidget(QLabel("ROI center y:"))
        self.layout_roi_y.addWidget(self.spbx_roi_center_y)
        self.layout_roi_y.addWidget(QLabel("ROI crop y:"))
        self.layout_roi_y.addWidget(self.spbx_roi_crop_y)
        self.layout_camera_roi = QVBoxLayout()
        self.layout_camera_roi.addLayout(self.layout_roi_x)
        self.layout_camera_roi.addLayout(self.layout_roi_y)
        self.group_camera_roi = QGroupBox("Camera ROI")
        self.group_camera_roi.setLayout(self.layout_camera_roi)
        
        #--------------------------------------------------------------------#
        # setup sub layouts in opm settings
        self.layout_imaging_settings.addLayout(self.layout_opm_mode)
        self.layout_imaging_settings.addLayout(self.layout_o2o3_mode)
        self.layout_imaging_settings.addLayout(self.layout_fluidics_mode)
        self.layout_imaging_settings.addLayout(self.layout_laser_blanking)
        self.layout_imaging_settings.addLayout(self.layout_405)
        self.layout_imaging_settings.addLayout(self.layout_488)
        self.layout_imaging_settings.addLayout(self.layout_561)
        self.layout_imaging_settings.addLayout(self.layout_638)
        self.layout_imaging_settings.addLayout(self.layout_705)
        self.layout_imaging_settings.addLayout(self.layout_mirror_image_range)
        self.layout_imaging_settings.addLayout(self.layout_mirror_image_step)
        self.layout_imaging_settings.addLayout(self.layout_stage_image_range)
        self.layout_imaging_settings.addLayout(self.layout_stage_slope)
        self.layout_imaging_settings.addLayout(self.layout_proj_image_range)
        self.layout_imaging_settings.addWidget(self.group_camera_roi)
        self.group_imaging_settings.setLayout(self.layout_imaging_settings)
        
        self.widgets.update({
            "O2O3-autofocus": {
              "o2o3_mode": self.cmbx_o2o3_mode  
            },
            "mirror_scan": {
                "image_mirror_range_um": self.spbx_mirror_image_range,
                "image_mirror_step_size_um": self.spbx_mirror_image_step
            },
            "projection_scan":{
                "image_mirror_range_um": self.spbx_proj_image_range
            },
            "stage_scan": {
                "stage_scan_range_um": self.spbx_stage_image_range,
                "coverslip_slope": self.spbx_stage_slope
            },
            "camera_roi": {
                "center_x": self.spbx_roi_center_x,
                "center_y": self.spbx_roi_center_y,
                "crop_x": self.spbx_roi_crop_x,
                "crop_y": self.spbx_roi_crop_y,
            }
        })
        
        #--------------------------------------------------------------------#
        #--------------------------------------------------------------------#
        # update main layout
        #--------------------------------------------------------------------#
        #--------------------------------------------------------------------#
        
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.group_ao_settings)
        self.main_layout.addWidget(self.group_imaging_settings)
        self.setLayout(self.main_layout)
        self.layout()

    #--------------------------------------------------------------------#
    # Methods to update sliders and spinboxes channel states
    #--------------------------------------------------------------------#
    def update_405_spbx(self):
        # Update the spinbox value when the slider value changes
        self.spbx_405_power.setValue(self.sldr_405_power.value())

    def update_405_slider(self):
        # Update the slider value when the spinbox value changes
        self.sldr_405_power.setValue(int(self.spbx_405_power.value()))
    
    def update_488_spbx(self):
        # Update the spinbox value when the slider value changes
        self.spbx_488_power.setValue(self.sldr_488_power.value())

    def update_488_slider(self):
        # Update the slider value when the spinbox value changes
        self.sldr_488_power.setValue(int(self.spbx_488_power.value()))
        
    def update_561_spbx(self):
        # Update the spinbox value when the slider value changes
        self.spbx_561_power.setValue(self.sldr_561_power.value())

    def update_561_slider(self):
        # Update the slider value when the spinbox value changes
        self.sldr_561_power.setValue(int(self.spbx_561_power.value()))
    
    def update_638_spbx(self):
        # Update the spinbox value when the slider value changes
        self.spbx_638_power.setValue(self.sldr_638_power.value())

    def update_638_slider(self):
        # Update the slider value when the spinbox value changes
        self.sldr_638_power.setValue(int(self.spbx_638_power.value()))
        
    def update_705_spbx(self):
        # Update the spinbox value when the slider value changes
        self.spbx_705_power.setValue(self.sldr_705_power.value())

    def update_705_slider(self):
        # Update the slider value when the spinbox value changes
        self.sldr_705_power.setValue(int(self.spbx_705_power.value()))

    def update_ao_active_power_spbx(self):
        # Update the spinbox value when the slider value changes
        self.spbx_active_channel_power.setValue(self.sldr_active_channel_power.value())

    def update_ao_active_power_slider(self):
        # Update the slider value when the spinbox value changes
        self.sldr_active_channel_power.setValue(int(self.spbx_active_channel_power.value()))
        
    def update_ao_exposure_spbx(self):
        # Update the spinbox value when the slider value changes
        self.spbx_ao_exposure.setValue(self.sldr_ao_exposure.value())

    def update_ao_exposure_slider(self):
        # Update the slider value when the spinbox value changes
        self.sldr_ao_exposure.setValue(int(self.spbx_ao_exposure.value()))
    
    def update_ao_mirror_range_spbx(self):
        # Update the spinbox value when the slider value changes
        self.spbx_ao_mirror_range.setValue(self.sldr_ao_mirror_range.value())

    def update_ao_mirror_range_slider(self):
        # Update the slider value when the spinbox value changes
        self.sldr_ao_mirror_range.setValue(int(self.spbx_ao_mirror_range.value()))
            
    def update_mirror_image_range_spbx(self):
        # Update the spinbox value when the slider value changes
        self.spbx_mirror_image_range.setValue(self.sldr_mirror_image_range.value())

    def update_mirror_image_range_slider(self):
        # Update the slider value when the spinbox value changes
        self.sldr_mirror_image_range.setValue(int(self.spbx_mirror_image_range.value()))
        
    def update_mirror_image_step_spbx(self):
        # Update the spinbox value when the slider value changes
        self.spbx_mirror_image_step.setValue(self.sldr_mirror_image_step.value()/100)

    def update_mirror_image_step_slider(self):
        # Update the slider value when the spinbox value changes
        self.sldr_mirror_image_step.setValue(int(self.spbx_mirror_image_step.value() * 100))  
                  
    def update_stage_image_range_spbx(self):
        # Update the spinbox value when the slider value changes
        self.spbx_stage_image_range.setValue(self.sldr_stage_image_range.value())

    def update_stage_image_range_slider(self):
        # Update the slider value when the spinbox value changes
        self.sldr_stage_image_range.setValue(int(self.spbx_stage_image_range.value()))
            
    def update_proj_image_range_spbx(self):
        # Update the spinbox value when the slider value changes
        self.spbx_proj_image_range.setValue(self.sldr_proj_image_range.value())

    def update_proj_image_range_slider(self):
        # Update the slider value when the spinbox value changes
        self.sldr_proj_image_range.setValue(int(self.spbx_proj_image_range.value()))
    
    #--------------------------------------------------------------------#
    # Methods to update acquisition channel states
    #--------------------------------------------------------------------#

    def update_405_state(self):
        checked = self.chx_405_state.isChecked()
        power = self.spbx_405_power.value()
        exposure_ms = self.spbx_405_exp.value()
        for _mode in self.config["OPM"]["imaging_modes"]:
            self.config["acq_config"][_mode+"_scan"]["channel_states"][0] = checked
            self.config["acq_config"][_mode+"_scan"]["channel_powers"][0] = power
            self.config["acq_config"][_mode+"_scan"]["channel_exposures_ms"][0] = exposure_ms
            
        self.update_config()
               
    def update_488_state(self):
        checked = self.chx_488_state.isChecked()
        power = self.spbx_488_power.value()
        exposure_ms = self.spbx_488_exp.value()
        for _mode in self.config["OPM"]["imaging_modes"]:
            self.config["acq_config"][_mode+"_scan"]["channel_states"][1] = checked
            self.config["acq_config"][_mode+"_scan"]["channel_powers"][1] = power
            self.config["acq_config"][_mode+"_scan"]["channel_exposures_ms"][1] = exposure_ms
            
        self.update_config()
    
    def update_561_state(self):
        checked = self.chx_561_state.isChecked()
        power = self.spbx_561_power.value()
        exposure_ms = self.spbx_561_exp.value()
        for _mode in self.config["OPM"]["imaging_modes"]:
            self.config["acq_config"][_mode+"_scan"]["channel_states"][2] = checked
            self.config["acq_config"][_mode+"_scan"]["channel_powers"][2] = power
            self.config["acq_config"][_mode+"_scan"]["channel_exposures_ms"][2] = exposure_ms
            
        self.update_config()
        
    def update_638_state(self):
        checked = self.chx_638_state.isChecked()
        power = self.spbx_638_power.value()
        exposure_ms = self.spbx_638_exp.value()
        for _mode in self.config["OPM"]["imaging_modes"]:
            self.config["acq_config"][_mode+"_scan"]["channel_states"][3] = checked
            self.config["acq_config"][_mode+"_scan"]["channel_powers"][3] = power
            self.config["acq_config"][_mode+"_scan"]["channel_exposures_ms"][3] = exposure_ms

        self.update_config()
    
    def update_705_state(self):
        checked = self.chx_705_state.isChecked()
        power = self.spbx_705_power.value()
        exposure_ms = self.spbx_705_exp.value()
        for _mode in self.config["OPM"]["imaging_modes"]:
            self.config["acq_config"][_mode+"_scan"]["channel_states"][4] = checked
            self.config["acq_config"][_mode+"_scan"]["channel_powers"][4] = power
            self.config["acq_config"][_mode+"_scan"]["channel_exposures_ms"][4] = exposure_ms
       
        self.update_config()
        
    #--------------------------------------------------------------------#
    # Methods to update configuration file and emit a signal when settings are updated.
    #--------------------------------------------------------------------#
    
    def update_config(self):
        """
        Update configuration file and local dict.
        """
        with open(self.config_path, "r") as config_file:
            config = json.load(config_file)
            
        for key_id in self.widgets.keys():
            for key in self.widgets[key_id]:
                widget = self.widgets[key_id][key]
                    
                if isinstance(widget, QSpinBox): 
                    config["acq_config"][key_id][key] = widget.value()
                elif isinstance(widget, QDoubleSpinBox): 
                    config["acq_config"][key_id][key] = widget.value()
                elif isinstance(widget, QComboBox): 
                        config["acq_config"][key_id][key] = widget.currentText()
        
        config["acq_config"]["opm_mode"] = self.cmbx_opm_mode.currentText()
        config["acq_config"]["fluidics"] = self.cmbx_fluidics_mode.currentText()
        if self.cmbx_fluidics_mode.currentText()=="on":
            laser_blanking = True
        else:
            laser_blanking = False
        for _mode in config["OPM"]["imaging_modes"]:
            config["acq_config"][_mode+"_scan"]["laser_blanking"] = laser_blanking

        self.config = config
        
        with open(self.config_path, "w") as file:
                json.dump(self.config, file, indent=4)
        
        self.settings_changed.emit()
        

if __name__ ==  "__main__":
    app = QApplication(sys.argv)
    # Put your path here.
    config_path = Path("/home/steven/Documents/qi2lab/github/opm_v2/opm_config_20250312.json")
    window = OPMSettings(config_path)
    window.show()
    
    def signal_recieved():
        print("signal triggered")
    window.settings_changed.connect(signal_recieved)
    
    sys.exit(app.exec())
